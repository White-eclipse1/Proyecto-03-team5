import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, expect, it, vi } from "vitest";
import { App } from "../src/App";
import { ProjectionTooltip } from "../src/pipeline/pages/Projections";
import { projectionsReportSchema } from "../src/pipeline/projectionSchemas";
import fixture from "./fixtures/projections.json";

// Recharts needs layout dimensions, which jsdom does not calculate.
vi.mock("recharts", async (importOriginal) => {
  const original = await importOriginal<typeof import("recharts")>();
  return { ...original, ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <original.ResponsiveContainer width={700} height={460}>{children}</original.ResponsiveContainer> };
});

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
function open() {
  render(<MemoryRouter initialEntries={["/pipeline/projections"]}><App /></MemoryRouter>);
}
function serve(doc: unknown = fixture, status = 200) {
  const fetcher = vi.fn(async () => new Response(JSON.stringify(doc), { status }));
  vi.stubGlobal("fetch", fetcher);
  return fetcher;
}

it("loads PCA metadata and real scatter points, switches to t-SNE without a second request", async () => {
  const fetcher = serve(); open();
  expect(await screen.findByText("4 imágenes · test-v1")).toBeInTheDocument();
  expect(screen.getByText("RGB reducido 16×16 · 768 features")).toBeInTheDocument();
  expect(screen.getByText("Varianza explicada: 60.00% / 30.00%")).toBeInTheDocument();
  expect(screen.getByText("Multietiqueta (1)")).toBeInTheDocument();
  expect(screen.getByText("Sin etiqueta (1)")).toBeInTheDocument();
  expect(document.querySelectorAll(".recharts-scatter-symbol")).toHaveLength(4);
  fireEvent.change(screen.getByLabelText("Método"), { target: { value: "tsne" } });
  expect(screen.getByText("Perplexity: 2 · random_state: 42")).toBeInTheDocument();
  expect(screen.getByLabelText("Proyección t-SNE")).toBeInTheDocument();
  expect(document.querySelectorAll(".recharts-scatter-symbol")).toHaveLength(4);
  expect(fetcher).toHaveBeenCalledTimes(1);
  expect(fetcher).toHaveBeenCalledWith("/reports/projections.json");
  expect(document.querySelector('a[href^="/annotate/"]')).toBeNull();
});

it("shows informative tooltip including multilabel categories and no links", () => {
  render(<ProjectionTooltip point={fixture.pca.points[2]!} categories={fixture.categories} />);
  expect(screen.getByText("mixed.jpg")).toBeInTheDocument();
  expect(screen.getByText("COCO image_id: 3")).toBeInTheDocument();
  expect(screen.getByText("Categorías: dog, cat")).toBeInTheDocument();
  expect(screen.getByText("x: 0.3000 · y: 0.6000")).toBeInTheDocument();
  expect(screen.queryByRole("link")).not.toBeInTheDocument();
});

it("shows loading without invented data", () => {
  vi.stubGlobal("fetch", vi.fn(() => new Promise(() => {}))); open();
  expect(screen.queryByLabelText("Método")).not.toBeInTheDocument();
  expect(screen.queryByText(/imágenes ·/)).not.toBeInTheDocument();
});

it.each([404, 500])("reports HTTP %s and retries without fixture fallback", async (status) => {
  const fetcher = serve(null, status); open();
  expect(await screen.findByText("No se pudo cargar el reporte.")).toBeInTheDocument();
  expect(screen.queryByLabelText("Método")).not.toBeInTheDocument();
  fetcher.mockImplementation(async () => new Response(JSON.stringify(fixture)));
  fireEvent.click(screen.getByRole("button", { name: /reintentar/i }));
  expect(await screen.findByText("4 imágenes · test-v1")).toBeInTheDocument();
});

it("rejects malformed JSON without a fallback", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => new Response("{broken"))); open();
  expect(await screen.findByText("No se pudo cargar el reporte.")).toBeInTheDocument();
  expect(screen.queryByLabelText("Método")).not.toBeInTheDocument();
});

it("rejects an empty report in the page", async () => {
  serve({ ...fixture, total_images: 0 }); open();
  expect(await screen.findByText("No se pudo cargar el reporte.")).toBeInTheDocument();
});

it("accepts only a full report with required effective parameters", () => {
  expect(projectionsReportSchema.parse(fixture)).toEqual(fixture);
  const doc = structuredClone(fixture) as Record<string, unknown>;
  doc.features = {};
  expect(projectionsReportSchema.safeParse(doc).success).toBe(false);
});

it.each(["coverage", "duplicate", "filename", "absolute", "category", "labels", "count", "nan", "inf", "variance", "perplexity"])("rejects %s contract violations", (kind) => {
  const doc = structuredClone(fixture);
  const point = doc.tsne.points[0]!;
  if (kind === "coverage") point.image_id = 999;
  if (kind === "duplicate") doc.tsne.points[1] = { ...point };
  if (kind === "filename") point.file_name = "other.jpg";
  if (kind === "absolute") point.file_name = "/private/image.jpg";
  if (kind === "category") point.category_ids = [999];
  if (kind === "labels") point.category_ids = [];
  if (kind === "count") doc.total_images = 5;
  if (kind === "nan") point.x = Number.NaN;
  if (kind === "inf") point.y = Number.POSITIVE_INFINITY;
  if (kind === "variance") doc.pca.explained_variance_ratio = [0.8, 0.8];
  if (kind === "perplexity") doc.tsne.parameters.perplexity = 4;
  expect(projectionsReportSchema.safeParse(doc).success).toBe(false);
});

it.each([0, 1.0, Number.MAX_SAFE_INTEGER])("accepts interoperable image/category ID %s", (value) => {
  const doc = structuredClone(fixture);
  const previous = doc.categories[0]!.id;
  doc.categories[0]!.id = value;
  for (const method of [doc.pca, doc.tsne]) {
    method.points[0]!.image_id = value;
    for (const point of method.points) point.category_ids = point.category_ids.map((id) => id === previous ? value : id);
  }
  expect(projectionsReportSchema.safeParse(doc).success).toBe(true);
});

it.each([Number.MAX_SAFE_INTEGER + 1, -1, 1.5, true, false, "1"])("rejects invalid IDs %s in all identifier positions", (value) => {
  for (const field of ["image_id", "category_ids", "category.id"]) {
    const doc = structuredClone(fixture);
    if (field === "category.id") Object.assign(doc.categories[0]!, { id: value });
    else Object.assign(doc.pca.points[0]!, { [field]: field === "category_ids" ? [value] : value });
    expect(projectionsReportSchema.safeParse(doc).success).toBe(false);
  }
});

it.each([true, false])("accepts real boolean whiten=%s", (value) => {
  const doc = structuredClone(fixture);
  doc.pca.parameters.whiten = value;
  expect(projectionsReportSchema.safeParse(doc).success).toBe(true);
});

it.each([0, 1, "false", "true"])("rejects coerced boolean whiten=%s", (value) => {
  const doc = structuredClone(fixture);
  Object.assign(doc.pca.parameters, { whiten: value });
  expect(projectionsReportSchema.safeParse(doc).success).toBe(false);
});
