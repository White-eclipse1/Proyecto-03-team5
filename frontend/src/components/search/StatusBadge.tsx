import type { ImageStatus } from "@/api/schemas";

const STATUS_CONFIG: Record<ImageStatus, { label: string; dot: string; bg: string; text: string }> =
  {
    pending: {
      label: "Pendiente",
      dot: "bg-status-pending",
      bg: "bg-status-pending-soft",
      text: "text-status-pending",
    },
    in_progress: {
      label: "En progreso",
      dot: "bg-status-progress",
      bg: "bg-status-progress-soft",
      text: "text-status-progress",
    },
    completed: {
      label: "Completada",
      dot: "bg-status-done",
      bg: "bg-status-done-soft",
      text: "text-status-done",
    },
  };

interface StatusBadgeProps {
  status: ImageStatus;
}

export function StatusBadge({ status }: StatusBadgeProps): JSX.Element {
  const config = STATUS_CONFIG[status];

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs font-medium ${config.bg} ${config.text}`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${config.dot}`} aria-hidden="true" />
      {config.label}
    </span>
  );
}
