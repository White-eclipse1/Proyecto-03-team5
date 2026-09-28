import { Download, Search, Upload } from "lucide-react";
import { Link } from "react-router-dom";
import { buildApiUrl } from "../api/client";
import { AnnotationProgressDonut } from "../components/dashboard/AnnotationProgressDonut";
import { DashboardSkeleton } from "../components/dashboard/DashboardSkeleton";
import { ObjectsPerClassChart } from "../components/dashboard/ObjectsPerClassChart";
import { RecentUploads } from "../components/dashboard/RecentUploads";
import { StatCard } from "../components/dashboard/StatCard";
import { ErrorState } from "../components/ui/ErrorState";
import { useDashboardSummary } from "../hooks/useDashboardSummary";

export function DashboardPage() {
  const summary = useDashboardSummary();

  return (
    <main className="flex-1 px-6 py-6 lg:px-10 lg:py-8">
      <div className="mx-auto flex max-w-6xl flex-col gap-6">
        <header className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h1 className="text-xl font-semibold text-ink">Tablero</h1>
            <p className="mt-1 text-sm text-ink-muted">
              Visión general del estado de tus fotografías.
            </p>
          </div>
          <div className="flex gap-2">
            <a
              href={buildApiUrl("/export/coco")}
              className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-surface px-3 py-2 text-sm font-medium text-ink transition-colors hover:bg-sidebar"
            >
              <Download className="h-4 w-4" aria-hidden />
              Exportar COCO
            </a>
            <Link
              to="/search"
              className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-surface px-3 py-2 text-sm font-medium text-ink transition-colors hover:bg-sidebar"
            >
              <Search className="h-4 w-4" aria-hidden />
              Buscar fotografías
            </Link>
            <Link
              to="/upload"
              className="inline-flex items-center gap-1.5 rounded-lg bg-accent-lilac px-3 py-2 text-sm font-medium text-white shadow-sm transition-colors hover:bg-accent-lilac/90"
            >
              <Upload className="h-4 w-4" aria-hidden />
              Subir fotografías
            </Link>
          </div>
        </header>

        {summary.status === "loading" && <DashboardSkeleton />}

        {summary.status === "error" && (
          <ErrorState
            title="No se pudo cargar el tablero."
            message={summary.message}
            onRetry={summary.reload}
          />
        )}

        {summary.status === "success" && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <StatCard
                label="Fotografías subidas"
                value={summary.data.imagesUploaded}
                accent="lilac"
              />
              <StatCard
                label="Fotografías anotadas"
                value={summary.data.imagesAnnotated}
                accent="mint"
              />
              <StatCard
                label="Objetos identificados"
                value={summary.data.boundingBoxes}
                accent="peach"
              />
              <StatCard label="Categorías" value={summary.data.categoriesCount} accent="blue" />
            </div>

            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              <ObjectsPerClassChart data={summary.data.objectsPerClass} />
              <AnnotationProgressDonut progress={summary.data.annotationProgress} />
            </div>

            <RecentUploads uploads={summary.data.recentUploads} />
          </div>
        )}
      </div>
    </main>
  );
}
