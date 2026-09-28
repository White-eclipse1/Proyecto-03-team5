import { resolveBackendUrl } from "../../lib/api/images";
import type { RecentUpload } from "../../types/dashboard";

type RecentUploadsProps = {
  uploads: RecentUpload[];
};

const statusConfig: Record<RecentUpload["status"], { label: string; dotClassName: string }> = {
  completed: { label: "Completada", dotClassName: "bg-status-done" },
  in_progress: { label: "En progreso", dotClassName: "bg-status-progress" },
  pending: { label: "Pendiente", dotClassName: "bg-status-pending" },
};

export function RecentUploads({ uploads }: RecentUploadsProps) {
  if (uploads.length === 0) {
    return (
      <div className="flex h-32 items-center justify-center rounded-2xl border border-border bg-surface text-sm text-ink-faint">
        Aún no hay fotografías subidas.
      </div>
    );
  }

  return (
    <div className="rounded-2xl border border-border bg-surface p-5 shadow-card">
      <h3 className="text-sm font-medium text-ink">Subidas recientes</h3>
      <div className="mt-4 flex gap-4 overflow-x-auto pb-1">
        {uploads.map((upload) => {
          const config = statusConfig[upload.status];
          return (
            <div key={upload.id} className="relative shrink-0">
              <img
                src={resolveBackendUrl(upload.thumbnailUrl)}
                alt={`Fotografía ${upload.id} — ${config.label}`}
                className="h-20 w-20 rounded-xl border border-border object-cover"
              />
              <span
                title={config.label}
                className={`absolute -bottom-1 -right-1 h-4 w-4 rounded-full border-2 border-surface ${config.dotClassName}`}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}
