interface EmptyStateProps {
  hasFilters: boolean;
}

export function EmptyState({ hasFilters }: EmptyStateProps): JSX.Element {
  return (
    <div className="flex flex-col items-center justify-center gap-2 rounded-2xl border border-dashed border-border-strong bg-surface px-6 py-16 text-center">
      <p className="text-sm font-medium text-ink">
        {hasFilters ? "No se encontraron imágenes con esos filtros" : "Todavía no hay imágenes"}
      </p>
      <p className="max-w-sm text-sm text-ink-muted">
        {hasFilters
          ? "Prueba quitando algún filtro o ajustando el rango de fechas."
          : "Cuando se suban imágenes al repositorio, aparecerán aquí."}
      </p>
    </div>
  );
}
