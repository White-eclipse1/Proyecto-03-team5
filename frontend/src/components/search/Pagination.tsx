import type { Pagination as PaginationData } from "@/api/schemas";

interface PaginationProps {
  pagination: PaginationData;
  onPageChange: (page: number) => void;
}

export function Pagination({ pagination, onPageChange }: PaginationProps): JSX.Element {
  const { page, totalPages } = pagination;
  const isFirst = page <= 1;
  const isLast = page >= totalPages;

  const buttonClass =
    "rounded-lg border border-border-strong px-3 py-1.5 text-sm text-ink transition-colors hover:bg-sidebar disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-transparent";

  return (
    <nav className="flex items-center justify-center gap-2" aria-label="Paginación de resultados">
      <button
        type="button"
        className={buttonClass}
        disabled={isFirst}
        onClick={() => onPageChange(1)}
      >
        « Primera
      </button>
      <button
        type="button"
        className={buttonClass}
        disabled={isFirst}
        onClick={() => onPageChange(page - 1)}
      >
        ‹ Anterior
      </button>

      <span className="px-3 text-sm text-ink-muted">
        Página <span className="font-medium text-ink">{page}</span> de{" "}
        <span className="font-medium text-ink">{Math.max(totalPages, 1)}</span>
      </span>

      <button
        type="button"
        className={buttonClass}
        disabled={isLast}
        onClick={() => onPageChange(page + 1)}
      >
        Siguiente ›
      </button>
      <button
        type="button"
        className={buttonClass}
        disabled={isLast}
        onClick={() => onPageChange(totalPages)}
      >
        Última »
      </button>
    </nav>
  );
}
