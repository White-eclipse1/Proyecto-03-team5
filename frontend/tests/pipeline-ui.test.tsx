import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "../src/App";
import settingsFixture from "./fixtures/settings.json";

const QUALITY_FIXTURE = {
  schema_version: "1.0",
  dataset_version: "demo-v1.0.0",
  status: "warning",
  checks: [
    {
      check_name: "min_images_per_class",
      passed: true,
      metric_value: 400.0,
      details: {},
      action: "fail",
    },
    {
      check_name: "max_imbalance_ratio",
      passed: true,
      metric_value: 1.0,
      details: {
        images_per_category: [{ category_name: "fixture", image_count: 1200 }],
      },
      action: "warn",
    },
    {
      check_name: "max_small_object_ratio",
      passed: false,
      metric_value: 0.45,
      details: {},
      action: "warn",
    },
    {
      check_name: "degenerate_boxes",
      passed: true,
      metric_value: 0.0,
      details: { total_annotations: 668 },
      action: "fail",
    },
    {
      check_name: "duplicate_similarity_threshold",
      passed: true,
      metric_value: 0.0,
      details: {},
      action: "warn",
    },
  ],
};

const SPLITS_FIXTURE = {
  schema_version: "1.0",
  dataset_version: "demo-v1.0.0",
  total_images: 1200,
  splits: {
    train: { image_count: 840, ratio: 0.7 },
    validation: { image_count: 180, ratio: 0.15 },
    test: { image_count: 180, ratio: 0.15 },
  },
};

const VERSIONS_FIXTURE = {
  schema_version: "1.0",
  releases: [
    { dataset_version: "demo-v1.0.0", quality_file: "quality.json", splits_file: "splits.json" },
  ],
};

function jsonResponse(body: unknown, status = 200) {
  return Promise.resolve(
    new Response(status === 404 ? null : JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    }),
  );
}

function mockAllReportsAvailable() {
  vi.stubGlobal(
    "fetch",
    vi.fn((url: string) => {
      if (url === "/api/settings") return jsonResponse(settingsFixture);
      if (url === "/reports/quality.json") return jsonResponse(QUALITY_FIXTURE);
      if (url === "/reports/splits.json") return jsonResponse(SPLITS_FIXTURE);
      if (url === "/reports/versions.json") return jsonResponse(VERSIONS_FIXTURE);
      return jsonResponse(null, 404);
    }),
  );
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

function renderPipelineAt(path: string) {
  render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  );
}

describe("SPEC-PIPELINE-UI-001 - UI del pipeline contra los reportes reales (P2-14/22/23/24)", () => {
  it("Overview muestra métricas coherentes del mismo quality.json, no hardcodeadas", async () => {
    mockAllReportsAvailable();
    renderPipelineAt("/pipeline/overview");

    expect(await screen.findByRole("heading", { name: "Overview" })).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText(/demo-v1\.0\.0/)).toBeInTheDocument());
    // total_images de quality.json (separador de miles en formato "es")
    expect(await screen.findByText((1200).toLocaleString("es"))).toBeInTheDocument();
    // checks.length de quality.json
    expect(screen.getByText("5")).toBeInTheDocument();
  });

  it("navega a las otras 5 pantallas desde el nav del pipeline, todas con datos reales", async () => {
    mockAllReportsAvailable();
    renderPipelineAt("/pipeline/overview");
    await screen.findByRole("heading", { name: "Overview" });

    fireEvent.click(screen.getByRole("link", { name: "Analyzers" }));
    expect(await screen.findByRole("heading", { name: "Analyzers" })).toBeInTheDocument();
    // check_name de quality.json, no un texto inventado en el componente
    expect(await screen.findByText("max_small_object_ratio")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("link", { name: "Splits" }));
    expect(await screen.findByRole("heading", { name: "Splits" })).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Release"), { target: { value: "demo-v1.0.0" } });
    // image_count de splits.json
    expect(await screen.findByText("840")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("link", { name: "Versions" }));
    expect(await screen.findByRole("heading", { name: "Versions" })).toBeInTheDocument();
    expect(await screen.findByRole("heading", { name: "demo-v1.0.0" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("link", { name: "Copilot" }));
    expect(screen.getByRole("heading", { name: "Copilot" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("link", { name: "Settings" }));
    expect(screen.getByRole("heading", { name: "Settings" })).toBeInTheDocument();
    expect(await screen.findByLabelText("Seed reproducible")).toHaveValue(42);
    expect(screen.queryByText("Dataset activo")).not.toBeInTheDocument();
  });

  it("/pipeline redirige a /pipeline/overview", async () => {
    mockAllReportsAvailable();
    renderPipelineAt("/pipeline");
    expect(await screen.findByRole("heading", { name: "Overview" })).toBeInTheDocument();
  });

  it("Overview no se rompe si versions.json todavía no existe (404)", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn((url: string) => {
        if (url === "/reports/quality.json") return jsonResponse(QUALITY_FIXTURE);
        return jsonResponse(null, 404);
      }),
    );
    renderPipelineAt("/pipeline/overview");

    expect(await screen.findByRole("heading", { name: "Overview" })).toBeInTheDocument();
    expect(await screen.findAllByText("No disponible")).toHaveLength(1);
    // Los datos que sí llegaron (quality.json) igual se muestran.
    expect(screen.getByText("5")).toBeInTheDocument();
  });

  it("Analyzers muestra un error legible si quality.json no existe todavía", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => jsonResponse(null, 404)),
    );
    renderPipelineAt("/pipeline/analyzers");

    expect(await screen.findByText("No se pudo cargar el reporte.")).toBeInTheDocument();
    expect(screen.getByText("Este reporte todavía no existe.")).toBeInTheDocument();
  });
});
