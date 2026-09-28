import type { Category, ImageStatus } from "@/api/schemas";
import type { SearchFilters } from "@/lib/searchFilters";

const STATUS_OPTIONS: { value: ImageStatus; label: string }[] = [
  { value: "pending", label: "Pendiente" },
  { value: "in_progress", label: "En progreso" },
  { value: "completed", label: "Completada" },
];

interface FiltersPanelProps {
  filters: SearchFilters;
  categories: Category[];
  categoriesLoading: boolean;
  categoriesError: string | null;
  onRetryCategories: () => void;
  onChange: (next: Partial<SearchFilters>) => void;
}

export function FiltersPanel({
  filters,
  categories,
  categoriesLoading,
  categoriesError,
  onRetryCategories,
  onChange,
}: FiltersPanelProps): JSX.Element {
  function toggleCategory(id: number): void {
    const exists = filters.categoryIds.includes(id);
    const next = exists
      ? filters.categoryIds.filter((c) => c !== id)
      : [...filters.categoryIds, id];
    onChange({ categoryIds: next, page: 1 });
  }

  function toggleStatus(status: ImageStatus): void {
    const exists = filters.statuses.includes(status);
    const next = exists
      ? filters.statuses.filter((s) => s !== status)
      : [...filters.statuses, status];
    onChange({ statuses: next, page: 1 });
  }

  return (
    <div className="flex flex-col gap-6">
      <section>
        <h3 className="mb-2 text-xs font-medium text-ink-muted">Categorías</h3>

        {categoriesLoading && (
          <div className="flex flex-col gap-2">
            {[0, 1, 2].map((i) => (
              <div key={i} className="h-5 w-full animate-pulse rounded bg-border" />
            ))}
          </div>
        )}

        {!categoriesLoading && categoriesError && (
          <div className="flex flex-col items-start gap-1.5">
            <p className="text-xs text-status-pending">No se pudieron cargar.</p>
            <button
              type="button"
              onClick={onRetryCategories}
              className="text-xs font-medium text-accent-lilac hover:underline"
            >
              Reintentar
            </button>
          </div>
        )}

        {!categoriesLoading && !categoriesError && categories.length === 0 && (
          <p className="text-xs text-ink-faint">No hay categorías registradas.</p>
        )}

        {!categoriesLoading && !categoriesError && categories.length > 0 && (
          <ul className="flex flex-col gap-2">
            {categories.map((category) => (
              <li key={category.id}>
                <label className="flex cursor-pointer items-center gap-2 text-sm text-ink">
                  <input
                    type="checkbox"
                    checked={filters.categoryIds.includes(category.id)}
                    onChange={() => toggleCategory(category.id)}
                    className="h-3.5 w-3.5 rounded border-border-strong text-accent-lilac focus:ring-accent-lilac"
                  />
                  <span
                    className="h-2.5 w-2.5 shrink-0 rounded-full"
                    style={{ backgroundColor: category.color }}
                    aria-hidden="true"
                  />
                  <span className="truncate">{category.name}</span>
                </label>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section>
        <h3 className="mb-2 text-xs font-medium text-ink-muted">Status</h3>
        <ul className="flex flex-col gap-2">
          {STATUS_OPTIONS.map((option) => (
            <li key={option.value}>
              <label className="flex cursor-pointer items-center gap-2 text-sm text-ink">
                <input
                  type="checkbox"
                  checked={filters.statuses.includes(option.value)}
                  onChange={() => toggleStatus(option.value)}
                  className="h-3.5 w-3.5 rounded border-border-strong text-accent-lilac focus:ring-accent-lilac"
                />
                <span>{option.label}</span>
              </label>
            </li>
          ))}
        </ul>
      </section>

      <section>
        <h3 className="mb-2 text-xs font-medium text-ink-muted">Rango de fechas</h3>
        <div className="flex flex-col gap-2">
          <label className="flex flex-col gap-1 text-xs text-ink-muted">
            Desde
            <input
              type="date"
              value={filters.dateFrom}
              onChange={(e) => onChange({ dateFrom: e.target.value, page: 1 })}
              className="rounded-lg border border-border-strong bg-surface px-2 py-1.5 text-sm text-ink"
            />
          </label>
          <label className="flex flex-col gap-1 text-xs text-ink-muted">
            Hasta
            <input
              type="date"
              value={filters.dateTo}
              onChange={(e) => onChange({ dateTo: e.target.value, page: 1 })}
              className="rounded-lg border border-border-strong bg-surface px-2 py-1.5 text-sm text-ink"
            />
          </label>
        </div>
      </section>
    </div>
  );
}
