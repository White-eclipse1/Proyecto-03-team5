import type { ImageSearchItem } from "@/api/schemas";
import { resolveBackendUrl } from "@/lib/api/images";
import { CategoryBadge } from "./CategoryBadge";
import { StatusBadge } from "./StatusBadge";

interface ImageCardProps {
  image: ImageSearchItem;
  isSelected: boolean;
  onClick: () => void;
  onToggleSelect: () => void;
}

export function ImageCard({
  image,
  isSelected,
  onClick,
  onToggleSelect,
}: ImageCardProps): JSX.Element {
  return (
    <div
      className={`group relative flex flex-col overflow-hidden rounded-2xl border bg-surface shadow-card transition-shadow hover:shadow-popover ${
        isSelected ? "border-accent-lilac ring-2 ring-accent-lilac/40" : "border-border"
      }`}
    >
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          onToggleSelect();
        }}
        aria-pressed={isSelected}
        aria-label={isSelected ? "Quitar de la selección" : "Seleccionar"}
        className={`absolute left-2 top-2 z-10 flex h-5 w-5 items-center justify-center rounded-md border shadow-sm transition-colors ${
          isSelected
            ? "border-accent-lilac bg-accent-lilac text-white"
            : "border-border-strong bg-surface/90 text-transparent hover:border-accent-lilac"
        }`}
      >
        <svg viewBox="0 0 20 20" className="h-3.5 w-3.5" fill="currentColor" aria-hidden>
          <path
            fillRule="evenodd"
            d="M16.704 5.29a1 1 0 0 1 0 1.415l-7.5 7.5a1 1 0 0 1-1.415 0l-3.5-3.5a1 1 0 1 1 1.415-1.414L8.5 12.086l6.79-6.79a1 1 0 0 1 1.414-.005Z"
            clipRule="evenodd"
          />
        </svg>
      </button>

      <button type="button" onClick={onClick} className="flex flex-col text-left">
        <div className="aspect-square w-full overflow-hidden bg-sidebar">
          {/* thumbnailUrl viene del backend como ruta canónica; el frontend la
              resuelve contra la base del API (proxy /api en desarrollo). */}
          <img
            src={resolveBackendUrl(image.thumbnailUrl)}
            alt={image.filename}
            loading="lazy"
            className="h-full w-full object-cover transition-transform duration-200 group-hover:scale-[1.03]"
          />
        </div>
      </button>

      <div className="flex flex-col gap-2 px-3 py-2.5">
        <StatusBadge status={image.status} />
        <div className="flex flex-wrap gap-1">
          {image.categories.length === 0 ? (
            <span className="text-[10px] text-ink-faint">Sin categorías</span>
          ) : (
            image.categories.map((category) => (
              <CategoryBadge key={category.id} name={category.name} color={category.color} />
            ))
          )}
        </div>
      </div>
    </div>
  );
}
