import type { Annotation } from "../../types/schemas";

const THUMB_SIZE = 48;
const MAX_THUMB_SCALE = 6;

interface AnnotationsSidebarProps {
  annotations: Annotation[];
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
  selectedId: number | null;
  onSelect: (id: number) => void;
  onDelete: (id: number) => void;
}

/** Recorte visual (miniatura) de la zona de la imagen que cubre una caja. */
function BoxThumbnail({
  annotation,
  imageUrl,
  imageWidth,
  imageHeight,
}: {
  annotation: Annotation;
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
}) {
  const scale = Math.min(
    MAX_THUMB_SCALE,
    THUMB_SIZE / Math.max(annotation.bboxWidth, annotation.bboxHeight, 1)
  );
  const centerX = annotation.bboxX + annotation.bboxWidth / 2;
  const centerY = annotation.bboxY + annotation.bboxHeight / 2;

  return (
    <div
      className="relative shrink-0 overflow-hidden rounded-lg border-2"
      style={{ width: THUMB_SIZE, height: THUMB_SIZE, borderColor: annotation.category.color }}
    >
      <img
        src={imageUrl}
        alt=""
        draggable={false}
        className="absolute max-w-none"
        style={{
          width: imageWidth * scale,
          height: imageHeight * scale,
          left: THUMB_SIZE / 2 - centerX * scale,
          top: THUMB_SIZE / 2 - centerY * scale,
        }}
      />
    </div>
  );
}

function DeleteIcon() {
  return (
    <svg viewBox="0 0 20 20" className="h-4 w-4" fill="currentColor" aria-hidden>
      <path
        fillRule="evenodd"
        d="M8.75 1a.75.75 0 0 0-.75.75V2h-3a.75.75 0 0 0 0 1.5h.3l.7 12.2A2 2 0 0 0 7.99 17.5h4.02a2 2 0 0 0 1.99-1.8l.7-12.2h.3a.75.75 0 0 0 0-1.5h-3v-.25a.75.75 0 0 0-.75-.75h-2.5Z"
        clipRule="evenodd"
      />
    </svg>
  );
}

export function AnnotationsSidebar({
  annotations,
  imageUrl,
  imageWidth,
  imageHeight,
  selectedId,
  onSelect,
  onDelete,
}: AnnotationsSidebarProps) {
  return (
    <aside className="flex max-h-64 w-full shrink-0 flex-col border-t border-border bg-surface lg:h-auto lg:max-h-none lg:w-80 lg:border-l lg:border-t-0">
      <div className="flex items-center justify-between border-b border-border px-4 py-3">
        <p className="text-sm font-semibold text-ink">Objetos</p>
        <span className="rounded-full bg-sidebar px-2 py-0.5 text-xs font-medium text-ink-muted">
          {annotations.length}
        </span>
      </div>

      {annotations.length === 0 ? (
        <div className="flex flex-1 flex-col items-center justify-center gap-2 px-6 py-6 text-center lg:py-10">
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-sidebar text-ink-faint">
            <svg
              viewBox="0 0 20 20"
              className="h-5 w-5"
              fill="none"
              stroke="currentColor"
              strokeWidth={1.5}
              aria-hidden
            >
              <rect x="3.5" y="3.5" width="13" height="13" rx="2.5" />
            </svg>
          </div>
          <p className="text-xs leading-relaxed text-ink-faint">
            Dibuja un recuadro sobre la imagen para agregar el primer objeto.
          </p>
        </div>
      ) : (
        <ul className="flex-1 space-y-1 overflow-y-auto p-2">
          {annotations.map((annotation) => {
            const isSelected = selectedId === annotation.id;
            return (
              <li key={annotation.id}>
                {/* biome-ignore lint/a11y/useSemanticElements: contiene un <button> de borrar; anidar button-en-button es HTML inválido. */}
                <div
                  role="button"
                  tabIndex={0}
                  onClick={() => onSelect(annotation.id)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") onSelect(annotation.id);
                  }}
                  className={`group flex w-full items-center gap-3 rounded-xl px-2 py-2 text-left transition-colors ${
                    isSelected
                      ? "bg-accent-lilac-soft ring-1 ring-inset ring-accent-lilac/30"
                      : "hover:bg-sidebar"
                  }`}
                >
                  <BoxThumbnail
                    annotation={annotation}
                    imageUrl={imageUrl}
                    imageWidth={imageWidth}
                    imageHeight={imageHeight}
                  />
                  <span
                    className={`min-w-0 flex-1 truncate text-sm ${
                      isSelected ? "font-medium text-accent-lilac" : "text-ink"
                    }`}
                  >
                    {annotation.category.name}
                  </span>
                  <button
                    type="button"
                    aria-label={`Borrar caja de ${annotation.category.name}`}
                    onClick={(e) => {
                      e.stopPropagation();
                      onDelete(annotation.id);
                    }}
                    className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-ink-faint transition-colors hover:bg-red-50 hover:text-red-600"
                  >
                    <DeleteIcon />
                  </button>
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </aside>
  );
}
