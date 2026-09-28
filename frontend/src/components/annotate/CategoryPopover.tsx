import { type CSSProperties, useState } from "react";
import type { Category } from "../../types/schemas";

interface CategoryPopoverProps {
  categories: Category[];
  /** Posición en píxeles de pantalla (ya calculada por el caller) */
  style: CSSProperties;
  onConfirm: (categoryId: number) => void;
  onCancel: () => void;
}

export function CategoryPopover({ categories, style, onConfirm, onCancel }: CategoryPopoverProps) {
  const [selectedId, setSelectedId] = useState<number | null>(null);

  return (
    <div
      style={style}
      className="absolute z-30 w-56 origin-top-left animate-popover-in rounded-xl border border-border bg-surface p-3 shadow-popover"
    >
      <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-ink-faint">
        Elige una categoría
      </p>
      {categories.length === 0 ? (
        <p className="text-xs text-ink-faint">No hay categorías disponibles.</p>
      ) : (
        <ul className="mb-3 max-h-48 overflow-y-auto">
          {categories.map((category) => (
            <li key={category.id}>
              <button
                type="button"
                onClick={() => setSelectedId(category.id)}
                className={`flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left text-sm transition-colors ${
                  selectedId === category.id
                    ? "bg-accent-lilac-soft font-medium text-accent-lilac ring-1 ring-inset ring-accent-lilac/30"
                    : "text-ink hover:bg-sidebar"
                }`}
              >
                <span
                  aria-hidden
                  className="h-3 w-3 flex-shrink-0 rounded-full border border-black/10"
                  style={{ backgroundColor: category.color }}
                />
                <span className="truncate">{category.name}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
      <div className="flex justify-end gap-2 border-t border-border pt-2.5">
        <button
          type="button"
          onClick={onCancel}
          className="rounded-lg px-2.5 py-1.5 text-xs font-medium text-ink-muted transition-colors hover:bg-sidebar"
        >
          Cancelar
        </button>
        <button
          type="button"
          disabled={selectedId === null}
          onClick={() => selectedId !== null && onConfirm(selectedId)}
          className="rounded-lg bg-accent-lilac px-3 py-1.5 text-xs font-medium text-white shadow-sm transition-colors hover:bg-accent-lilac/90 disabled:cursor-not-allowed disabled:bg-border disabled:text-ink-faint disabled:shadow-none"
        >
          Confirmar
        </button>
      </div>
    </div>
  );
}
