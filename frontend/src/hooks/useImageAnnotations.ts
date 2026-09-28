import { useCallback, useEffect, useRef, useState } from "react";
import {
  createAnnotation,
  deleteAnnotation,
  getAnnotations,
  updateAnnotation,
} from "../lib/api/annotations";
import {
  getImage,
  getImageFileUrl,
  loadImageDimensions,
  patchImageStatus,
} from "../lib/api/images";
import type { Annotation, BBox, ImageStatus } from "../types/schemas";
import { useUndoStack } from "./useUndoStack";

interface ImageMeta {
  width: number;
  height: number;
  status: ImageStatus;
  filename: string;
}

type ShowToast = (message: string, variant?: "error" | "success" | "info") => void;

interface UseImageAnnotationsOptions {
  /**
   * Valores conocidos de antemano (p. ej. pasados por estado de navegación
   * desde Upload/Search) para no depender de GET /images/:id. Si no se dan,
   * se usan valores por defecto seguros: 'pending' y `Imagen #<id>`.
   */
  initialFilename?: string;
  initialStatus?: ImageStatus;
}

export function useImageAnnotations(
  imageId: number,
  showToast: ShowToast,
  options: UseImageAnnotationsOptions = {}
) {
  const { initialFilename, initialStatus } = options;
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [imageMeta, setImageMeta] = useState<ImageMeta | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const { push: pushUndo, undo, clear: clearUndo, canUndo } = useUndoStack();

  // Evita aplicar la respuesta de una carga vieja si `imageId` cambió rápido
  // (p. ej. al navegar rápido entre imágenes de una cola).
  const requestIdRef = useRef(0);

  const load = useCallback(() => {
    const requestId = ++requestIdRef.current;
    setIsLoading(true);
    setLoadError(null);
    // La pila de deshacer es por imagen: si venimos de otra imagen de la
    // cola, sus acciones no deben poder deshacerse aquí (los ids de
    // anotación pertenecen a otra imagen).
    clearUndo();

    Promise.all([loadImageDimensions(getImageFileUrl(imageId)), getAnnotations(imageId)])
      .then(([dimensions, anns]) => {
        if (requestIdRef.current !== requestId) return;
        setImageMeta({
          width: dimensions.width,
          height: dimensions.height,
          status: initialStatus ?? "pending",
          filename: initialFilename ?? `Imagen #${imageId}`,
        });
        setAnnotations(anns);
        setIsLoading(false);

        // Mejora best-effort, no bloqueante: si el backend expone
        // GET /images/:id, se usa para corregir status/filename reales
        // (por ejemplo si se entró por URL directa sin estado de
        // navegación). Si falla, no pasa nada — ya se cargó lo esencial.
        getImage(imageId)
          .then((image) => {
            if (requestIdRef.current !== requestId) return;
            setImageMeta((prev) =>
              prev
                ? {
                    ...prev,
                    status: image.status,
                    filename: image.filename,
                    width: image.width,
                    height: image.height,
                  }
                : prev
            );
          })
          .catch(() => {
            // Endpoint no disponible o falló: se ignora, la pantalla ya
            // funciona con los valores por defecto/pasados por navegación.
          });
      })
      .catch((err: unknown) => {
        if (requestIdRef.current !== requestId) return;
        setLoadError(err instanceof Error ? err.message : "No se pudo cargar la imagen.");
        setIsLoading(false);
      });
  }, [imageId, initialFilename, initialStatus, clearUndo]);

  useEffect(() => {
    load();
    setSelectedId(null);
  }, [load]);

  const markInProgressIfNeeded = useCallback(() => {
    setImageMeta((prev) =>
      prev && prev.status === "pending" ? { ...prev, status: "in_progress" } : prev
    );
  }, []);

  const createBox = useCallback(
    async (categoryId: number, bbox: BBox) => {
      try {
        const created = await createAnnotation(imageId, { categoryId, ...bbox });
        setAnnotations((prev) => [...prev, created]);
        markInProgressIfNeeded();
        pushUndo({
          label: "crear caja",
          undo: async () => {
            await deleteAnnotation(created.id);
            setAnnotations((prev) => prev.filter((a) => a.id !== created.id));
          },
        });
        return created;
      } catch (err) {
        showToast(err instanceof Error ? err.message : "No se pudo guardar la caja.", "error");
        return null;
      }
    },
    [imageId, markInProgressIfNeeded, pushUndo, showToast]
  );

  /** Usado para mover/redimensionar. `before` es el bbox previo, para poder deshacer. */
  const commitBoxChange = useCallback(
    async (annotationId: number, before: BBox, after: BBox) => {
      // Optimista: refleja el cambio de inmediato en la UI.
      setAnnotations((prev) => prev.map((a) => (a.id === annotationId ? { ...a, ...after } : a)));
      try {
        const updated = await updateAnnotation(annotationId, after);
        setAnnotations((prev) => prev.map((a) => (a.id === annotationId ? updated : a)));
        pushUndo({
          label: "mover/redimensionar caja",
          undo: async () => {
            setAnnotations((prev) =>
              prev.map((a) => (a.id === annotationId ? { ...a, ...before } : a))
            );
            const reverted = await updateAnnotation(annotationId, before);
            setAnnotations((prev) => prev.map((a) => (a.id === annotationId ? reverted : a)));
          },
        });
      } catch (err) {
        // revierte el optimismo si el backend rechaza el cambio
        setAnnotations((prev) =>
          prev.map((a) => (a.id === annotationId ? { ...a, ...before } : a))
        );
        showToast(err instanceof Error ? err.message : "No se pudo guardar el cambio.", "error");
      }
    },
    [pushUndo, showToast]
  );

  const removeBox = useCallback(
    async (annotationId: number) => {
      const existing = annotations.find((a) => a.id === annotationId);
      if (!existing) return;
      setAnnotations((prev) => prev.filter((a) => a.id !== annotationId));
      if (selectedId === annotationId) setSelectedId(null);
      try {
        await deleteAnnotation(annotationId);
        pushUndo({
          label: "borrar caja",
          undo: async () => {
            const recreated = await createAnnotation(imageId, {
              categoryId: existing.categoryId,
              bboxX: existing.bboxX,
              bboxY: existing.bboxY,
              bboxWidth: existing.bboxWidth,
              bboxHeight: existing.bboxHeight,
              iscrowd: existing.iscrowd,
            });
            setAnnotations((prev) => [...prev, recreated]);
          },
        });
      } catch (err) {
        // si falla el borrado en backend, restaura la caja localmente
        setAnnotations((prev) => [...prev, existing]);
        showToast(err instanceof Error ? err.message : "No se pudo borrar la caja.", "error");
      }
    },
    [annotations, imageId, pushUndo, selectedId, showToast]
  );

  /**
   * Persiste el status "in_progress" en el backend. `markInProgressIfNeeded`
   * solo actualiza el estado local (optimista) al crear la primera caja —
   * esto es lo que realmente lo guarda, para usarse p. ej. al salir de la
   * pantalla con un borrador ("Guardar borrador").
   */
  const saveDraft = useCallback(async () => {
    try {
      await patchImageStatus(imageId, "in_progress");
      setImageMeta((prev) =>
        prev && prev.status !== "completed" ? { ...prev, status: "in_progress" } : prev
      );
      return true;
    } catch (err) {
      showToast(err instanceof Error ? err.message : "No se pudo guardar el borrador.", "error");
      return false;
    }
  }, [imageId, showToast]);

  const finalize = useCallback(async () => {
    try {
      await patchImageStatus(imageId, "completed");
      setImageMeta((prev) => (prev ? { ...prev, status: "completed" } : prev));
      showToast("Imagen marcada como completada.", "success");
      return true;
    } catch (err) {
      showToast(err instanceof Error ? err.message : "No se pudo finalizar la imagen.", "error");
      return false;
    }
  }, [imageId, showToast]);

  const handleUndo = useCallback(async () => {
    const label = await undo().catch((err: unknown) => {
      showToast(err instanceof Error ? err.message : "No se pudo deshacer.", "error");
      return null;
    });
    if (label) showToast(`Deshecho: ${label}`, "info");
  }, [showToast, undo]);

  return {
    annotations,
    imageMeta,
    isLoading,
    loadError,
    reload: load,
    selectedId,
    setSelectedId,
    createBox,
    commitBoxChange,
    removeBox,
    finalize,
    saveDraft,
    undo: handleUndo,
    canUndo,
  };
}
