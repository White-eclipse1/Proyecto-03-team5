import type { Category } from "@/api/schemas";
import type { SearchFilters } from "@/lib/searchFilters";
import { FiltersPanel } from "./FiltersPanel";

interface FiltersSidebarProps {
  filters: SearchFilters;
  categories: Category[];
  categoriesLoading: boolean;
  categoriesError: string | null;
  onRetryCategories: () => void;
  onChangeFilters: (next: Partial<SearchFilters>) => void;
}

/**
 * Panel de filtros de Búsqueda. Es específico de esta página, pero vive
 * dentro del sidebar global (GlobalNav/AppLayout, vía `sidebarExtra`) en vez
 * de tener su propio `<aside>` al lado.
 */
export function FiltersSidebar({
  filters,
  categories,
  categoriesLoading,
  categoriesError,
  onRetryCategories,
  onChangeFilters,
}: FiltersSidebarProps): JSX.Element {
  return (
    <div>
      <h2 className="mb-4 text-sm font-semibold text-ink">Filtros</h2>
      <FiltersPanel
        filters={filters}
        categories={categories}
        categoriesLoading={categoriesLoading}
        categoriesError={categoriesError}
        onRetryCategories={onRetryCategories}
        onChange={onChangeFilters}
      />
    </div>
  );
}
