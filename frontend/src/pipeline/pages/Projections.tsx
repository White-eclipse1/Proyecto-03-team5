import { useState } from "react";
import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getImageFileUrl } from "@/lib/api/images";
import { PageHeader } from "../components/PageHeader";
import { ReportBoundary } from "../components/ReportBoundary";
import { useProjectionsReport } from "../dataSource";
import type { ProjectionPoint, ProjectionsReport } from "../projectionSchemas";

const colors = ["#60a5fa", "#f472b6", "#34d399", "#fbbf24", "#a78bfa", "#fb923c"];

export function ProjectionTooltip({
  point,
  categories,
}: Readonly<{ point: ProjectionPoint; categories: ProjectionsReport["categories"] }>) {
  const names = point.category_ids.map(
    (id) => categories.find((category) => category.id === id)?.name
  );
  return (
    <div className="rounded-lg border border-border bg-surface p-3 text-sm text-ink">
      <img
        src={getImageFileUrl(point.image_id)}
        alt={`Vista previa de ${point.file_name}`}
        className="mb-2 h-24 w-24 rounded object-cover"
      />
      <p>{point.file_name}</p>
      <p>COCO image_id: {point.image_id}</p>
      <p>Categorías: {names.length ? names.join(", ") : "Sin etiqueta"}</p>
      <p>
        x: {point.x.toFixed(4)} · y: {point.y.toFixed(4)}
      </p>
    </div>
  );
}

function ProjectionView({ report }: Readonly<{ report: ProjectionsReport }>) {
  const [method, setMethod] = useState<"pca" | "tsne">("pca");
  const [selectedCategory, setSelectedCategory] = useState("all");
  const selectedCategoryId = selectedCategory === "all" ? null : Number(selectedCategory);
  const matchesCategory = (point: ProjectionPoint) =>
    selectedCategoryId === null || point.category_ids.includes(selectedCategoryId);
  const groups = [
    ...[...report.categories]
      .sort((a, b) => a.id - b.id)
      .map((category, index) => ({
        key: `category-${category.id}`,
        label: category.name,
        color: colors[index % colors.length],
        points: report[method].points.filter(
          (point) =>
            point.category_ids.length === 1 &&
            point.category_ids[0] === category.id &&
            matchesCategory(point)
        ),
      })),
    {
      key: "multi",
      label: "Multietiqueta",
      color: "#c084fc",
      points: report[method].points.filter(
        (point) => point.category_ids.length > 1 && matchesCategory(point)
      ),
    },
    {
      key: "unlabeled",
      label: "Sin etiqueta",
      color: "#94a3b8",
      points: report[method].points.filter(
        (point) => point.category_ids.length === 0 && matchesCategory(point)
      ),
    },
  ];
  return (
    <div className="mt-6 space-y-4">
      <p>
        {report.total_images} imágenes · {report.dataset_version}
      </p>
      <p>RGB reducido 16×16 · 768 features</p>
      <p className="text-sm text-ink-muted">
        La proyección usa píxeles RGB reducidos, no embeddings de un modelo.
      </p>
      <label className="block" htmlFor="projection-method">
        Método
      </label>
      <select
        id="projection-method"
        value={method}
        onChange={(event) => setMethod(event.target.value === "tsne" ? "tsne" : "pca")}
      >
        <option value="pca">PCA</option>
        <option value="tsne">t-SNE</option>
      </select>
      <label className="block" htmlFor="projection-category">
        Categoría
      </label>
      <select
        id="projection-category"
        value={selectedCategory}
        onChange={(event) => setSelectedCategory(event.target.value)}
      >
        <option value="all">Todas</option>
        {report.categories.map((category) => (
          <option key={category.id} value={category.id}>
            {category.name}
          </option>
        ))}
      </select>
      {method === "pca" ? (
        <p>
          Varianza explicada:{" "}
          {report.pca.explained_variance_ratio
            .map((value) => `${(value * 100).toFixed(2)}%`)
            .join(" / ")}
        </p>
      ) : (
        <p>
          Perplexity: {report.tsne.parameters.perplexity} · random_state:{" "}
          {report.tsne.parameters.random_state}
        </p>
      )}
      <ul className="flex flex-wrap gap-4" aria-label="Leyenda de categorías">
        {groups.map((group) => (
          <li key={group.key}>
            <span style={{ color: group.color }} aria-hidden>
              ●{" "}
            </span>
            {group.label} ({group.points.length})
          </li>
        ))}
      </ul>
      <section
        className="h-[460px] w-full min-w-0"
        aria-label={`Proyección ${method === "pca" ? "PCA" : "t-SNE"}`}
      >
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" dataKey="x" name="x" />
            <YAxis type="number" dataKey="y" name="y" />
            <Tooltip
              content={({ active, payload }) =>
                active && payload?.[0]?.payload ? (
                  <ProjectionTooltip
                    point={payload[0].payload as ProjectionPoint}
                    categories={report.categories}
                  />
                ) : null
              }
            />
            {groups.map((group) => (
              <Scatter
                key={group.key}
                name={group.label}
                data={group.points}
                fill={group.color}
                isAnimationActive={false}
              />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      </section>
      <p className="break-all text-xs text-ink-muted">Fingerprint: {report.dataset_fingerprint}</p>
    </div>
  );
}

export function ProjectionsPage() {
  const state = useProjectionsReport();
  return (
    <main className="flex-1 px-6 py-6 lg:px-10">
      <PageHeader title="PCA / t-SNE" subtitle="Estructura visual del dataset" />
      <ReportBoundary state={state}>
        {(report) => <ProjectionView report={report} />}
      </ReportBoundary>
    </main>
  );
}
