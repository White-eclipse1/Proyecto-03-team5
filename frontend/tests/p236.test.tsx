import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, expect, it, vi } from "vitest";
import quality from "./fixtures/quality.json";
import splits from "./fixtures/splits.json";
import catalog from "./fixtures/catalog.json";
import { App } from "../src/App";
import { versionsReportSchema } from "../src/pipeline/schemas";

function serve(entries: Record<string, unknown>) {
  const fetcher = vi.fn(
    async (url: string) =>
      new Response(JSON.stringify(entries[url] ?? null), { status: url in entries ? 200 : 404 })
  );
  vi.stubGlobal("fetch", fetcher);
  return fetcher;
}
function open(page: string) {
  render(
    <MemoryRouter initialEntries={[`/pipeline/${page}`]}>
      <App />
    </MemoryRouter>
  );
}
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});
const files = {
  "/reports/quality.json": quality,
  "/reports/versions.json": catalog,
  "/reports/releases/v0.1.0/quality.json": quality,
  "/reports/releases/v0.1.0/splits.json": splits,
};

it("shows all six real criteria with distinct detector and compliance semantics", async () => {
  const custom = structuredClone(quality);
  custom.checks[0]!.details.criterion!.threshold = 777;
  serve({ ...files, "/reports/quality.json": custom });
  open("analyzers");
  await screen.findByText("min_images_per_class");
  expect(screen.getAllByRole("row")).toHaveLength(7);
  expect(screen.getByText(">= 777")).toBeInTheDocument();
  expect(screen.getByText("Indefinido: hay categorías sin imágenes")).toBeInTheDocument();
  expect(screen.getByText("failed")).toBeInTheDocument();
  expect(screen.getByText("Caja pequeña: ancho < 32 px y alto < 32 px.")).toBeInTheDocument();
  const minimum = screen.getByText("min_images_per_class").closest("tr")!;
  expect(within(minimum).getAllByText("fail")).toHaveLength(2);
  const imbalance = screen.getByText("max_imbalance_ratio").closest("tr")!;
  expect(within(imbalance).getAllByText("warn")).toHaveLength(2);
  const phash = screen.getByText("duplicate_similarity_threshold").closest("tr")!;
  expect(within(phash).getByText("== 0")).toBeInTheDocument();
  expect(within(phash).getByText(/Detección pHash: similitud ≥ 0.94/)).toBeInTheDocument();
});

it("selects a catalog release and loads its real split reference", async () => {
  const fetcher = serve(files);
  open("splits");
  fireEvent.change(await screen.findByLabelText("Release"), { target: { value: "v0.1.0" } });
  expect(await screen.findByText("600 imágenes · v0.1.0")).toBeInTheDocument();
  expect(screen.getByText("420")).toBeInTheDocument();
  expect(screen.getAllByText("90")).toHaveLength(2);
  for (const label of ["train (70%)", "validation (15%)", "test (15%)"])
    expect(screen.getByText(label)).toBeInTheDocument();
  expect(fetcher).toHaveBeenCalledWith("/reports/releases/v0.1.0/splits.json");
});

it("loads multiple releases without inventing dates or active status", async () => {
  serve({
    ...files,
    "/reports/versions.json": {
      ...catalog,
      releases: [
        ...catalog.releases,
        {
          dataset_version: "v0.2.0",
          quality_file: "releases/v0.2.0/quality.json",
          splits_file: "releases/v0.2.0/splits.json",
        },
      ],
    },
    "/reports/releases/v0.2.0/quality.json": { ...quality, dataset_version: "v0.2.0" },
    "/reports/releases/v0.2.0/splits.json": { ...splits, dataset_version: "v0.2.0" },
  });
  open("versions");
  await screen.findByText("600 imágenes · v0.2.0");
  expect(screen.getAllByText("failed")).toHaveLength(8);
  expect(screen.getAllByText("420")).toHaveLength(2);
  expect(screen.queryByText(/activo|autor|fecha/i)).not.toBeInTheDocument();
  expect(screen.getAllByRole("link", { name: "Reporte de splits" })[0]).toHaveAttribute(
    "href",
    "/reports/releases/v0.1.0/splits.json"
  );
});

it.each([
  "https://evil.test/quality.json",
  "//evil.test/quality.json",
  "../quality.json",
  "releases/%2e%2e/quality.json",
  "releases\\quality.json",
])("rejects unsafe reference %s before fetching", async (reference) => {
  const invalid = { ...catalog, releases: [{ ...catalog.releases[0], quality_file: reference }] };
  expect(versionsReportSchema.safeParse(invalid).success).toBe(false);
  const fetcher = serve({ "/reports/versions.json": invalid });
  open("splits");
  await screen.findByText("El reporte no tiene el formato esperado.");
  expect(fetcher).toHaveBeenCalledTimes(1);
});

it("rejects a referenced report belonging to a different version", async () => {
  serve({
    ...files,
    "/reports/releases/v0.1.0/splits.json": { ...splits, dataset_version: "other" },
  });
  open("splits");
  fireEvent.change(await screen.findByLabelText("Release"), { target: { value: "v0.1.0" } });
  await screen.findByText("El reporte no tiene el formato esperado.");
});

it.each(["network", "json", "404"])("shows %s errors for release reports", async (failure) => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      if (url.endsWith("versions.json")) return new Response(JSON.stringify(catalog));
      if (failure === "network") throw new Error("Sin conexión");
      return new Response("invalid", { status: failure === "404" ? 404 : 200 });
    })
  );
  open("versions");
  expect(await screen.findAllByText("No se pudo cargar el reporte.")).toHaveLength(2);
});


it.each(["splits", "versions"])("handles an empty catalog in %s", async (page) => {
  serve({ "/reports/versions.json": { schema_version: "1.0", releases: [] } });
  open(page);
  expect(await screen.findByText(/No hay releases publicadas|Todavía no hay releases publicadas/)).toBeInTheDocument();
});

it("shows a contract error for an invalid referenced QualityReport", async () => {
  serve({ ...files, "/reports/releases/v0.1.0/quality.json": { ...quality, status: "invalid" } });
  open("versions");
  expect(await screen.findByText("El reporte no tiene el formato esperado.")).toBeInTheDocument();
  expect(await screen.findByText("600 imágenes · v0.1.0")).toBeInTheDocument();
});

it("renders seven checks including real cross-split diagnostics while accepting the historical six", async () => {
  const leakage = {
    check_name: "cross_split_leakage", passed: false, metric_value: 1, action: "fail",
    details: {
      criterion: { metric: "metric_value", threshold: 0, operator: "<=" },
      similarity_threshold: 0.94, total_pairs_evaluated: 1,
      image_pairs: [{ image_id_a: 41, image_id_b: 42, split_a: "train", split_b: "test", hamming_distance: 2, similarity: 0.96875 }],
    },
  };
  serve({ ...files, "/reports/quality.json": { ...quality, checks: [...quality.checks, leakage] } });
  open("analyzers");
  const row = (await screen.findByText("cross_split_leakage")).closest("tr")!;
  expect(screen.getAllByRole("row")).toHaveLength(8);
  expect(within(row).getByText("1")).toBeInTheDocument();
  expect(within(row).getByText("<= 0")).toBeInTheDocument();
  expect(within(row).getAllByText("fail")).toHaveLength(2);
  expect(within(row).getByText("Pares detectados (1)")).toBeInTheDocument();
  expect(within(row).getByText(/train → test/)).toBeInTheDocument();
  expect(within(row).getByText(/Hamming: 2/)).toBeInTheDocument();
  expect(within(row).getByText(/similitud 0.96875/)).toBeInTheDocument();
});
