import { Button } from "@/components/ui/Button";

interface SelectionToolbarProps {
  count: number;
  onClearSelection: () => void;
  onDelete: () => void;
  onExport: () => void;
}

/**
 * Barra de acciones masivas para Búsqueda: aparece debajo de la barra de
 * texto cuando hay al menos una card seleccionada. "Exportar" es solo visual
 * por ahora (a propósito, sin conectar todavía a ningún endpoint).
 */
export function SelectionToolbar({
  count,
  onClearSelection,
  onDelete,
  onExport,
}: SelectionToolbarProps): JSX.Element {
  return (
    <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-accent-lilac/30 bg-accent-lilac-soft px-4 py-2.5">
      <div className="flex items-center gap-3">
        <span className="text-sm font-medium text-accent-lilac">
          {count} {count === 1 ? "seleccionada" : "seleccionadas"}
        </span>
        <button
          type="button"
          onClick={onClearSelection}
          className="text-xs font-medium text-ink-muted underline-offset-2 hover:text-ink hover:underline"
        >
          Deseleccionar todo
        </button>
      </div>

      <div className="flex items-center gap-2">
        <Button variant="secondary" size="sm" onClick={onExport}>
          Exportar
        </Button>
        <Button variant="destructive" size="sm" onClick={onDelete}>
          Eliminar
        </Button>
      </div>
    </div>
  );
}
