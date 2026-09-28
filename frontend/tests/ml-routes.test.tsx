import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "../src/App";
import { InferenceResult } from "../src/ml/pages/Inference";
import { inferenceResponseSchema } from "../src/ml/schemas";
import { loadMlExample } from "./mlCorpus";

// APP-01: las 5 pantallas de modelos viven en el mismo portal (AppLayout/GlobalNav)
// y consumen los contratos de src/ml/schemas.ts. Los ejemplos vienen del corpus
// compartido con Python; ninguna pantalla trae datos propios.

const SCREENS = [
  { label: "Training", path: "/ml/training", endpoint: "/api/ml/training/jobs" },
  { label: "Experiments", path: "/ml/experiments", endpoint: "/api/ml/runs" },
  { label: "Evaluation", path: "/ml/evaluation", endpoint: "/api/ml/evaluations" },
  { label: "Models", path: "/ml/models", endpoint: "/api/ml/models" },
  { label: "Inference", path: "/ml/inference", endpoint: "/api/ml/models" },
] as const;

const EXAMPLES: Record<string, unknown> = {
  "/api/ml/training/jobs": loadMlExample("training_jobs"),
  "/api/ml/runs": loadMlExample("runs"),
  "/api/ml/evaluations": loadMlExample("evaluations"),
  "/api/ml/models": loadMlExample("models"),
};

function json(body: unknown, status = 200) {
  return Promise.resolve(
    new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    })
  );
}

function serve(handler: (url: string) => Promise<Response>) {
  const fetcher = vi.fn((url: string) => handler(url));
  vi.stubGlobal("fetch", fetcher);
  return fetcher;
}

function serveExamples() {
  return serve((url) => (url in EXAMPLES ? json(EXAMPLES[url]) : json({}, 404)));
}

function openAt(path: string) {
  render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>
  );
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("APP-01 rutas de modelos dentro del portal existente", () => {
  it.each(SCREENS)("abre $path directamente dentro del mismo portal", async (screenDef) => {
    const fetcher = serveExamples();
    openAt(screenDef.path);

    expect(await screen.findByRole("heading", { level: 1, name: screenDef.label })).toBeVisible();
    // Mismo GlobalNav que el resto del portal, no una app/shell aparte.
    expect(screen.getByText("Portal de Anotación")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Tablero" })).toHaveAttribute("href", "/dashboard");
    expect(screen.getByRole("link", { name: "Overview" })).toHaveAttribute(
      "href",
      "/pipeline/overview"
    );
    expect(screen.getByRole("link", { name: screenDef.label })).toHaveAttribute(
      "aria-current",
      "page"
    );
    expect(fetcher).toHaveBeenCalledWith(screenDef.endpoint);
  });

  it("navega entre las 5 pantallas desde el nav global", async () => {
    serveExamples();
    openAt("/dashboard");

    for (const screenDef of SCREENS) {
      fireEvent.click(screen.getByRole("link", { name: screenDef.label }));
      expect(
        await screen.findByRole("heading", { level: 1, name: screenDef.label })
      ).toBeVisible();
    }
  });

  it("redirige /ml a Training", async () => {
    serveExamples();
    openAt("/ml");
    expect(await screen.findByRole("heading", { level: 1, name: "Training" })).toBeVisible();
  });
});

describe("APP-01 estados de carga y error", () => {
  it.each(SCREENS)("$label muestra loading mientras la fuente responde", async (screenDef) => {
    serve(() => new Promise<Response>(() => {}));
    openAt(screenDef.path);
    expect(await screen.findByRole("status")).toHaveTextContent("Cargando");
  });

  it.each(SCREENS)("$label indica que la fuente no está conectada (404)", async (screenDef) => {
    serve(() => json({}, 404));
    openAt(screenDef.path);
    expect(
      await screen.findByText("Esta fuente de datos todavía no está conectada.")
    ).toBeVisible();
    expect(screen.queryByText(/0a1b2c3d4e5f60718293a4b5c6d7e8f9/)).not.toBeInTheDocument();
  });

  it("muestra el ErrorResponse del backend y reintenta si es retryable", async () => {
    const fetcher = serve(() => json(loadMlExample("error"), 503));
    openAt("/ml/training");

    expect(await screen.findByText("El servicio de entrenamiento no respondió.")).toBeVisible();
    expect(screen.getByText("training_backend_unavailable")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Reintentar" }));
    expect(fetcher).toHaveBeenCalledTimes(2);
  });

  it("no ofrece reintentar cuando el error no es retryable", async () => {
    serve(() =>
      json(
        {
          schema_version: "1.0",
          error: { code: "invalid_request", message: "Solicitud inválida.", retryable: false },
        },
        400
      )
    );
    openAt("/ml/models");
    expect(await screen.findByText("Solicitud inválida.")).toBeVisible();
    expect(screen.queryByRole("button", { name: "Reintentar" })).not.toBeInTheDocument();
  });

  it("rechaza un payload que rompe el contrato en vez de renderizarlo", async () => {
    const models = structuredClone(EXAMPLES["/api/ml/models"]) as {
      models: { model_version: string }[];
    };
    models.models[0]!.model_version = "demo-v1.0.0";
    serve(() => json(models));
    openAt("/ml/models");

    expect(
      await screen.findByText("La respuesta no cumple el contrato esperado.")
    ).toBeVisible();
    expect(screen.queryByText("pet-detector")).not.toBeInTheDocument();
  });

  it.each([
    ["/ml/training", "jobs", "Todavía no hay jobs de entrenamiento."],
    ["/ml/experiments", "runs", "Todavía no hay runs registrados en MLflow."],
    ["/ml/evaluation", "evaluations", "Todavía no hay evaluaciones."],
    ["/ml/models", "models", "Todavía no hay modelos registrados."],
    ["/ml/inference", "models", "No hay modelos READY disponibles para inferencia."],
  ])("%s muestra un estado vacío explícito", async (path, key, message) => {
    serve(() => json({ schema_version: "1.0", [key]: [] }));
    openAt(path);
    expect(await screen.findByText(message)).toBeVisible();
  });
});

describe("APP-01 consumidores de los contratos (IDs reales)", () => {
  it("Training muestra job, dataset_version, manifest_hash, run_id y checkpoint", async () => {
    serveExamples();
    openAt("/ml/training");

    const row = (await screen.findByText("job-0001")).closest("tr")!;
    expect(within(row).getByText("succeeded")).toBeInTheDocument();
    expect(within(row).getByText("demo-v1.0.0")).toBeInTheDocument();
    expect(
      within(row).getByText("md5:5d41402abc4b2a76b9719d911017c592")
    ).toBeInTheDocument();
    expect(within(row).getByText("0a1b2c3d4e5f60718293a4b5c6d7e8f9")).toBeInTheDocument();
    expect(
      within(row).getByText("runs:/0a1b2c3d4e5f60718293a4b5c6d7e8f9/checkpoints/best.pt")
    ).toBeInTheDocument();

    const failed = screen.getByText("job-0002").closest("tr")!;
    expect(within(failed).getByText("El entrenamiento se quedó sin memoria de GPU.")).toBeVisible();
  });

  it("Experiments muestra los runs MLflow con su run_id y métricas", async () => {
    serveExamples();
    openAt("/ml/experiments");

    const row = (await screen.findByText("yolo-imgsz640")).closest("tr")!;
    expect(within(row).getByText("0a1b2c3d4e5f60718293a4b5c6d7e8f9")).toBeInTheDocument();
    expect(within(row).getByText("FINISHED")).toBeInTheDocument();
    expect(within(row).getByText("val_map50: 0.781")).toBeInTheDocument();
  });

  it("Evaluation separa model_version de dataset_version", async () => {
    serveExamples();
    openAt("/ml/evaluation");

    const row = (await screen.findByText("eval-0002")).closest("tr")!;
    expect(within(row).getByText("pet-detector v3")).toBeInTheDocument();
    expect(within(row).getByText("demo-v1.0.0")).toBeInTheDocument();
    expect(within(row).getByText("0.772")).toBeInTheDocument();
    const unregistered = screen.getByText("eval-0001").closest("tr")!;
    expect(within(unregistered).getByText("Sin registrar")).toBeInTheDocument();
  });

  it("Models muestra versión del registry, aliases y dataset de origen", async () => {
    serveExamples();
    openAt("/ml/models");

    const row = (await screen.findByText("champion")).closest("tr")!;
    expect(within(row).getByText("pet-detector")).toBeInTheDocument();
    expect(within(row).getByText("3")).toBeInTheDocument();
    expect(within(row).getByText("demo-v1.0.0")).toBeInTheDocument();
    expect(within(row).getByText("READY")).toBeInTheDocument();
  });

  it("Inference solo ofrece modelos READY", async () => {
    const models = structuredClone(EXAMPLES["/api/ml/models"]) as {
      models: { status: string }[];
    };
    models.models[1]!.status = "PENDING_REGISTRATION";
    serve(() => json(models));
    openAt("/ml/inference");

    expect(await screen.findByText("pet-detector v3")).toBeVisible();
    expect(screen.queryByText("pet-detector v2")).not.toBeInTheDocument();
  });

  it("InferenceResult muestra predicciones trazables al modelo y dataset", () => {
    const response = inferenceResponseSchema.parse(loadMlExample("inference"));
    render(<InferenceResult response={response} />);

    expect(screen.getByText("pet-detector v3")).toBeInTheDocument();
    expect(screen.getByText("demo-v1.0.0")).toBeInTheDocument();
    expect(screen.getByText("0a1b2c3d4e5f60718293a4b5c6d7e8f9")).toBeInTheDocument();
    const row = screen.getByText("dog").closest("tr")!;
    expect(within(row).getByText("91.2%")).toBeInTheDocument();
    expect(within(row).getByText("[32, 40.5, 210, 300]")).toBeInTheDocument();
  });
});
