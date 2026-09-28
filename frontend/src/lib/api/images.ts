import {
  type ImageRecord,
  type ImageSearchResponse,
  type ImageUploadResponse,
  imageDetailResponseSchema,
  imageSearchResponseSchema,
  imageUploadResponseSchema,
  type PatchImageStatusBody,
  patchImageStatusBodySchema,
} from "../../types/schemas";
import {
  API_BASE_URL,
  ApiError,
  apiRequest,
  apiRequestVoid,
  extractMessageField,
  jsonBody,
} from "./client";

export const ALLOWED_IMAGE_MIME_TYPES = ["image/jpeg", "image/png", "image/webp"] as const;
export const MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024; // 5 MiB

/**
 * Imágenes que todavía necesitan anotarse: incluye "pending" y también
 * "in_progress" (una imagen con borrador guardado no debe desaparecer de
 * la cola hasta que se marque como completada).
 */
export function searchPendingImages(page = 1, pageSize = 50): Promise<ImageSearchResponse> {
  const params = new URLSearchParams({
    status: "pending,in_progress",
    page: String(page),
    pageSize: String(pageSize),
  });
  return apiRequest(`/images/search?${params.toString()}`, imageSearchResponseSchema);
}

/**
 * Sube una imagen con feedback de progreso. Se usa XMLHttpRequest en vez de
 * fetch porque fetch no expone progreso de subida de forma nativa.
 */
export function uploadImage(
  file: File,
  onProgress?: (percent: number) => void
): Promise<ImageUploadResponse> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const formData = new FormData();
    formData.append("image", file);

    xhr.open("POST", `${API_BASE_URL}/images`);

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && onProgress) {
        onProgress(Math.round((event.loaded / event.total) * 100));
      }
    };

    xhr.onload = () => {
      if (xhr.status === 201) {
        let json: unknown;
        try {
          json = JSON.parse(xhr.responseText);
        } catch {
          reject(new Error("Respuesta inválida del servidor al subir la imagen."));
          return;
        }
        const parsed = imageUploadResponseSchema.safeParse(json);
        if (!parsed.success) {
          reject(new Error("Respuesta inválida del servidor al subir la imagen."));
          return;
        }
        resolve(parsed.data);
        return;
      }

      let message =
        xhr.status === 413
          ? "Excede el límite de 5 MiB."
          : xhr.status === 400
            ? "Archivo inválido o tipo no soportado."
            : `No se pudo subir el archivo (${xhr.status}).`;
      try {
        const body: unknown = JSON.parse(xhr.responseText);
        message = extractMessageField(body) ?? message;
      } catch {
        // se conserva el mensaje por defecto
      }
      reject(new ApiError(xhr.status, message));
    };

    xhr.onerror = () => reject(new ApiError(0, "Error de red al subir el archivo."));

    xhr.send(formData);
  });
}

/**
 * URL para usar como src de <img>. El bucket de MinIO es privado, por lo que
 * SIEMPRE se sirve a través de este endpoint del backend, nunca directo.
 */
export function getImageFileUrl(imageId: number): string {
  return `${API_BASE_URL}/images/${imageId}/file`;
}

/**
 * Resuelve una ruta relativa del backend (por ejemplo el `thumbnailUrl` que
 * viene en las respuestas, `/images/:id/file`) contra la base del API.
 *
 * El backend devuelve rutas canónicas sin el prefijo del proxy; el frontend
 * es quien sabe que en desarrollo el backend vive detrás de `/api`. Si la
 * URL ya es absoluta (http...) se devuelve tal cual.
 */
export function resolveBackendUrl(path: string): string {
  if (/^https?:\/\//.test(path)) return path;
  const normalized = path.startsWith("/") ? path : `/${path}`;
  return `${API_BASE_URL}${normalized}`;
}

/**
 * Carga las dimensiones reales de una imagen pidiéndole al navegador que la
 * descargue (usa naturalWidth/naturalHeight). Esto NO depende de ningún
 * endpoint adicional: usa exactamente `GET /images/:id/file`, que ya está
 * documentado como asumido en el prompt original. Es la fuente de verdad
 * para el tamaño de imagen que usa el canvas.
 */
export function loadImageDimensions(url: string): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight });
    img.onerror = () => reject(new Error("No se pudo cargar la imagen."));
    img.src = url;
  });
}

// ASUNCIÓN OPCIONAL, no requerida para que la pantalla funcione: si el
// backend real expone GET /images/:id, se usa para enriquecer el badge de
// estado y el nombre de archivo. Si no existe (404 o cualquier error), se
// ignora en silencio y se usan los valores por defecto/pasados por
// navegación — ver `useImageAnnotations`. Ajustar o eliminar esta función si
// el backend real no la va a implementar.
export function getImage(imageId: number): Promise<ImageRecord> {
  return apiRequest(`/images/${imageId}`, imageDetailResponseSchema);
}

export function patchImageStatus(
  imageId: number,
  status: PatchImageStatusBody["status"]
): Promise<void> {
  const body = patchImageStatusBodySchema.parse({ status });
  return apiRequestVoid(`/images/${imageId}/status`, {
    method: "PATCH",
    ...jsonBody(body),
  });
}

export function deleteImage(imageId: number): Promise<void> {
  return apiRequestVoid(`/images/${imageId}`, { method: "DELETE" });
}

/** Validación de cliente: no reemplaza la del backend, es feedback inmediato. */
export function validateImageFile(file: File): string | null {
  if (!ALLOWED_IMAGE_MIME_TYPES.includes(file.type as (typeof ALLOWED_IMAGE_MIME_TYPES)[number])) {
    return "Tipo de archivo no soportado. Usa JPEG, PNG o WebP.";
  }
  if (file.size > MAX_IMAGE_SIZE_BYTES) {
    return "Excede el límite de 5 MiB.";
  }
  return null;
}
