import type { PointerEvent as ReactPointerEvent } from "react";
import type { Annotation } from "../../types/schemas";

export type Corner = "nw" | "ne" | "sw" | "se";

export const CORNERS: { corner: Corner; className: string }[] = [
  { corner: "nw", className: "-left-1.5 -top-1.5 cursor-nwse-resize" },
  { corner: "ne", className: "-right-1.5 -top-1.5 cursor-nesw-resize" },
  { corner: "sw", className: "-left-1.5 -bottom-1.5 cursor-nesw-resize" },
  { corner: "se", className: "-right-1.5 -bottom-1.5 cursor-nwse-resize" },
];

interface BoundingBoxProps {
  annotation: Annotation;
  displayScale: number;
  isSelected: boolean;
  onSelect: () => void;
  onStartMove: (event: ReactPointerEvent<HTMLDivElement>) => void;
  onStartResize: (corner: Corner, event: ReactPointerEvent<HTMLDivElement>) => void;
  onDelete: () => void;
}

export function BoundingBox({
  annotation,
  displayScale,
  isSelected,
  onSelect,
  onStartMove,
  onStartResize,
  onDelete,
}: BoundingBoxProps) {
  const color = annotation.category.color;

  return (
    // biome-ignore lint/a11y/useSemanticElements: contiene el <button> de borrar; anidar button-en-button es HTML inválido.
    <div
      role="button"
      tabIndex={0}
      aria-label={`Caja de ${annotation.category.name}`}
      onPointerDown={(e) => {
        e.stopPropagation();
        e.preventDefault();
        // Sin esto, Safari a veces no sigue enviando pointermove/pointerup
        // a los listeners de window una vez que el puntero se mueve fuera
        // del elemento que originó el pointerdown (a diferencia de Chrome/
        // Firefox, que sí los entregan sin capture explícito): la caja
        // quedaba "atascada" y no se podía mover ni redimensionar.
        e.currentTarget.setPointerCapture(e.pointerId);
        onSelect();
        onStartMove(e);
      }}
      className="absolute"
      style={{
        left: annotation.bboxX * displayScale,
        top: annotation.bboxY * displayScale,
        width: annotation.bboxWidth * displayScale,
        height: annotation.bboxHeight * displayScale,
        border: `2px solid ${color}`,
        backgroundColor: isSelected ? `${color}22` : `${color}0d`,
        boxShadow: isSelected ? `0 0 0 2px ${color}55` : undefined,
      }}
    >
      <span
        className="pointer-events-none absolute -top-5 left-0 truncate rounded px-1 text-[10px] font-medium text-white"
        style={{ backgroundColor: color, maxWidth: "100%" }}
      >
        {annotation.category.name}
      </span>

      {isSelected && (
        <>
          {CORNERS.map(({ corner, className }) => (
            <div
              key={corner}
              onPointerDown={(e) => {
                e.stopPropagation();
                e.preventDefault();
                e.currentTarget.setPointerCapture(e.pointerId);
                onStartResize(corner, e);
              }}
              className={`absolute h-3 w-3 rounded-sm border border-white bg-ink ${className}`}
            />
          ))}
          <button
            type="button"
            aria-label="Borrar caja"
            onPointerDown={(e) => e.stopPropagation()}
            onClick={(e) => {
              e.stopPropagation();
              onDelete();
            }}
            className="absolute -right-3 -top-3 flex h-6 w-6 items-center justify-center rounded-full border border-border bg-surface text-ink-muted shadow-sm transition-colors hover:bg-red-50 hover:text-red-600"
          >
            <svg viewBox="0 0 20 20" className="h-3.5 w-3.5" fill="currentColor" aria-hidden>
              <path
                fillRule="evenodd"
                d="M8.75 1a.75.75 0 0 0-.75.75V2h-3a.75.75 0 0 0 0 1.5h.3l.7 12.2A2 2 0 0 0 7.99 17.5h4.02a2 2 0 0 0 1.99-1.8l.7-12.2h.3a.75.75 0 0 0 0-1.5h-3v-.25a.75.75 0 0 0-.75-.75h-2.5Z"
                clipRule="evenodd"
              />
            </svg>
          </button>
        </>
      )}
    </div>
  );
}
