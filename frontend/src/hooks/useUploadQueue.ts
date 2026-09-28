import { useCallback, useEffect, useState } from "react";
import {
  deleteImage,
  searchPendingImages,
  uploadImage,
  validateImageFile,
} from "../lib/api/images";
import type { ImageRecord, ImageStatus } from "../types/schemas";

export type UploadStatus = "invalid" | "uploading" | "success" | "error";

export interface UploadItem {
  clientId: string;
  file: File;
  status: UploadStatus;
  progress: number;
  errorMessage?: string;
  imageId?: number;
  previewUrl: string;
}

/** Entrada unificada para la lista de "pendientes" (subidas ahora + ya existentes). */
export interface PendingEntry {
  id: number;
  filename: string;
  previewUrl: string | null;
  status: ImageStatus;
}

function makeClientId(): string {
  return `upload-${crypto.randomUUID()}`;
}

export function useUploadQueue(
  showToast: (message: string, variant?: "error" | "success" | "info") => void
) {
  const [uploadItems, setUploadItems] = useState<UploadItem[]>([]);
  const [serverPending, setServerPending] = useState<ImageRecord[]>([]);
  const [isLoadingPending, setIsLoadingPending] = useState(true);
  const [pendingError, setPendingError] = useState<string | null>(null);
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());

  const loadServerPending = useCallback(() => {
    setIsLoadingPending(true);
    setPendingError(null);
    searchPendingImages()
      .then((res) => setServerPending(res.data))
      .catch((err: unknown) => {
        setPendingError(
          err instanceof Error ? err.message : "No se pudieron cargar las imágenes pendientes."
        );
      })
      .finally(() => setIsLoadingPending(false));
  }, []);

  useEffect(() => {
    loadServerPending();
  }, [loadServerPending]);

  const updateItem = useCallback((clientId: string, patch: Partial<UploadItem>) => {
    setUploadItems((prev) =>
      prev.map((item) => (item.clientId === clientId ? { ...item, ...patch } : item))
    );
  }, []);

  const uploadOne = useCallback(
    (item: UploadItem) => {
      updateItem(item.clientId, { status: "uploading", progress: 0 });
      uploadImage(item.file, (percent) => updateItem(item.clientId, { progress: percent }))
        .then((res) => {
          updateItem(item.clientId, { status: "success", progress: 100, imageId: res.id });
          setSelectedIds((prev) => {
            const next = new Set(prev);
            next.add(res.id);
            return next;
          });
        })
        .catch((err: unknown) => {
          const message = err instanceof Error ? err.message : "No se pudo subir el archivo.";
          updateItem(item.clientId, { status: "error", errorMessage: message });
          showToast(`${item.file.name}: ${message}`, "error");
        });
    },
    [showToast, updateItem]
  );

  const addFiles = useCallback(
    (files: FileList | File[]) => {
      const list = Array.from(files);
      const newItems: UploadItem[] = list.map((file) => {
        const invalidReason = validateImageFile(file);
        return {
          clientId: makeClientId(),
          file,
          status: invalidReason ? "invalid" : "uploading",
          progress: 0,
          errorMessage: invalidReason ?? undefined,
          previewUrl: URL.createObjectURL(file),
        };
      });

      setUploadItems((prev) => [...prev, ...newItems]);

      for (const item of newItems) {
        if (item.status === "invalid") {
          showToast(`${item.file.name}: ${item.errorMessage ?? "archivo inválido"}`, "error");
          continue;
        }
        uploadOne(item);
      }
    },
    [showToast, uploadOne]
  );

  const retryUpload = useCallback(
    (clientId: string) => {
      const item = uploadItems.find((i) => i.clientId === clientId);
      if (item) uploadOne({ ...item, status: "uploading", progress: 0 });
    },
    [uploadItems, uploadOne]
  );

  const toggleSelect = useCallback((id: number) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  // Lista combinada: pendientes ya existentes en el servidor + subidas recién
  // completadas que aún no aparecerían en un refetch (evita duplicados por id).
  const seenIds = new Set(serverPending.map((img) => img.id));
  const freshEntries: PendingEntry[] = [];
  for (const item of uploadItems) {
    if (item.status === "success" && item.imageId !== undefined && !seenIds.has(item.imageId)) {
      freshEntries.push({
        id: item.imageId,
        filename: item.file.name,
        previewUrl: item.previewUrl,
        status: "pending",
      });
    }
  }

  const pendingEntries: PendingEntry[] = [
    ...freshEntries,
    ...serverPending.map((img) => ({
      id: img.id,
      filename: img.filename,
      previewUrl: null,
      status: img.status,
    })),
  ];

  const deleteEntry = useCallback(
    async (id: number) => {
      try {
        await deleteImage(id);
        setServerPending((prev) => prev.filter((img) => img.id !== id));
        setUploadItems((prev) => prev.filter((item) => item.imageId !== id));
        setSelectedIds((prev) => {
          if (!prev.has(id)) return prev;
          const next = new Set(prev);
          next.delete(id);
          return next;
        });
      } catch (err) {
        showToast(err instanceof Error ? err.message : "No se pudo eliminar la imagen.", "error");
      }
    },
    [showToast]
  );

  return {
    uploadItems,
    pendingEntries,
    isLoadingPending,
    pendingError,
    selectedIds,
    addFiles,
    retryUpload,
    toggleSelect,
    deleteEntry,
    reloadPending: loadServerPending,
  };
}
