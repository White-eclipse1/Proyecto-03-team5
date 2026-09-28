import type { Category, ImageStatus } from "@/api/schemas";
import type { SearchFilters } from "@/lib/searchFilters";

const STATUS_LABELS: Record<ImageStatus, string> = {
  pending: "Pendiente",
  in_progress: "En progreso",
  completed: "Completada",
};

interface Chip {
  key: string;
  label: string;
  onRemove: () => void;
}

interface FilterChipsProps {
  filters: SearchFilters;
  categories: Category[];
  onChange: (next: Partial<SearchFilters>) => void;
}

export function FilterChips({
  filters,
  categories,
  onChange,
}: FilterChipsProps): JSX.Element | null {
  const categoryById = new Map(categories.map((c) => [c.id, c]));

  const chips: Chip[] = [];

  if (filters.q.trim() !== "") {
    chips.push({
      key: "q",
      label: `"${filters.q.trim()}"`,
      onRemove: () => onChange({ q: "", page: 1 }),
    });
  }

  for (const id of filters.categoryIds) {
    const category = categoryById.get(id);
    chips.push({
      key: `cat-${id}`,
      label: category?.name ?? `Categoría #${id}`,
      onRemove: () =>
        onChange({ categoryIds: filters.categoryIds.filter((c) => c !== id), page: 1 }),
    });
  }

  for (const status of filters.statuses) {
    chips.push({
      key: `status-${status}`,
      label: STATUS_LABELS[status],
      onRemove: () => onChange({ statuses: filters.statuses.filter((s) => s !== status), page: 1 }),
    });
  }

  if (filters.dateFrom !== "" || filters.dateTo !== "") {
    chips.push({
      key: "date",
      label: `${filters.dateFrom || "…"} → ${filters.dateTo || "…"}`,
      onRemove: () => onChange({ dateFrom: "", dateTo: "", page: 1 }),
    });
  }

  if (chips.length === 0) return null;

  return (
    <div className="flex flex-wrap items-center gap-2">
      {chips.map((chip) => (
        <span
          key={chip.key}
          className="inline-flex items-center gap-1.5 rounded-full bg-accent-lilac-soft px-3 py-1 text-xs font-medium text-accent-lilac"
        >
          {chip.label}
          <button
            type="button"
            onClick={chip.onRemove}
            aria-label={`Quitar filtro ${chip.label}`}
            className="rounded-full text-accent-lilac/70 hover:text-accent-lilac"
          >
            ×
          </button>
        </span>
      ))}
      {chips.length > 1 && (
        <button
          type="button"
          onClick={() =>
            onChange({ q: "", categoryIds: [], statuses: [], dateFrom: "", dateTo: "", page: 1 })
          }
          className="text-xs font-medium text-ink-faint underline-offset-2 hover:text-ink-muted hover:underline"
        >
          Limpiar todo
        </button>
      )}
    </div>
  );
}
