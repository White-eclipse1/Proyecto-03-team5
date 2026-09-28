import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import type { ImageSearchItem } from "@/api/schemas";
import { Modal } from "@/components/ui/Modal";
import { getAnnotations } from "@/lib/api/annotations";
import { getImageFileUrl } from "@/lib/api/images";
import type { AnnotateNavigationState } from "../../types/navigation";
import type { Annotation } from "../../types/schemas";
import { StatusBadge } from "./StatusBadge";

interface ImagePreviewModalProps {
  /** Resultados actualmente visibles en la grilla; la navegación ← → se mueve entre ellos. */
  images: ImageSearchItem[];
  initialIndex: number;
  onClose: () => void;
}

/**
 * Modal de solo lectura para ver una foto y sus objetos anotados, con
 * navegación a la imagen anterior/siguiente de los resultados actuales. No
 * permite editar cajas — para eso existe el botón "Editar anotaciones", que
 * lleva a la pantalla de anotación de pantalla completa.
 */
export function ImagePreviewModal({
  images,
  initialIndex,
  onClose,
}: ImagePreviewModalProps): JSX.Element | null {
  const navigate = useNavigate();
  const [index, setIndex] = useState(initialIndex);
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const image = images[index];

  useEffect(() => {
    if (!image) return;
    let cancelled = false;
    setIsLoading(true);
    setError(null);

    getAnnotations(image.id)
      .then((anns) => {
        if (cancelled) return;
        setAnnotations(anns);
        setIsLoading(false);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "No se pudieron cargar los objetos.");
        setIsLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [image]);

  const goTo = useCallback(
    (nextIndex: number) => {
      if (nextIndex < 0 || nextIndex >= images.length) return;
      setIndex(nextIndex);
    },
    [images.length]
  );

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "ArrowLeft") goTo(index - 1);
      if (event.key === "ArrowRight") goTo(index + 1);
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [goTo, index]);

  if (!image) return null;

  const handleEdit = (): void => {
    const state: AnnotateNavigationState = { from: "search" };
    navigate(`/annotate/${image.id}`, { state });
  };

  return (
    <Modal title={image.filename} onClose={onClose} maxWidthClassName="max-w-3xl">
      <div className="flex flex-col gap-4">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <StatusBadge status={image.status} />
            <span className="text-xs text-ink-muted">{image.annotationsCount} anot.</span>
          </div>
          <button
            type="button"
            onClick={handleEdit}
            className="rounded-lg border border-border px-3 py-1.5 text-xs font-medium text-ink-muted transition-colors hover:bg-sidebar hover:text-ink"
          >
            Editar anotaciones
          </button>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            aria-label="Imagen anterior"
            disabled={index === 0}
            onClick={() => goTo(index - 1)}
            className="shrink-0 rounded-lg border border-border p-2 text-ink-muted transition-colors hover:bg-sidebar disabled:cursor-not-allowed disabled:opacity-40"
          >
            ←
          </button>

          <div
            className="relative w-full overflow-hidden rounded-xl bg-canvas"
            style={{ aspectRatio: `${image.width} / ${image.height}` }}
          >
            <img
              src={getImageFileUrl(image.id)}
              alt={image.filename}
              className="h-full w-full object-contain"
            />

            {!isLoading &&
              !error &&
              annotations.map((annotation) => (
                <div
                  key={annotation.id}
                  className="absolute"
                  style={{
                    left: `${(annotation.bboxX / image.width) * 100}%`,
                    top: `${(annotation.bboxY / image.height) * 100}%`,
                    width: `${(annotation.bboxWidth / image.width) * 100}%`,
                    height: `${(annotation.bboxHeight / image.height) * 100}%`,
                    border: `2px solid ${annotation.category.color}`,
                    backgroundColor: `${annotation.category.color}22`,
                  }}
                >
                  <span
                    className="pointer-events-none absolute -top-5 left-0 truncate rounded px-1 text-[10px] font-medium text-white"
                    style={{ backgroundColor: annotation.category.color, maxWidth: "100%" }}
                  >
                    {annotation.category.name}
                  </span>
                </div>
              ))}
          </div>

          <button
            type="button"
            aria-label="Imagen siguiente"
            disabled={index === images.length - 1}
            onClick={() => goTo(index + 1)}
            className="shrink-0 rounded-lg border border-border p-2 text-ink-muted transition-colors hover:bg-sidebar disabled:cursor-not-allowed disabled:opacity-40"
          >
            →
          </button>
        </div>

        {error && <p className="text-xs text-red-700">{error}</p>}

        <p className="text-center text-xs text-ink-faint">
          {index + 1} de {images.length}
        </p>
      </div>
    </Modal>
  );
}
