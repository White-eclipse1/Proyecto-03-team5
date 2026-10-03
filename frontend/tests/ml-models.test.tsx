import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "../src/App";
import { loadMlExample } from "./mlCorpus";

// APP-06 (issue #23): la pantalla Models lee `GET /api/ml/models` (registro de OPS-06 y
// publicación en S3 de OPS-07). Los datos vienen del corpus compartido con Python.

type Json = Record<string, unknown>;
type Model = Json & {
  model_version: string;
  checkpoint_sha256: string;
  publication: Json & { status: string; objects: Json[] };
};

const SHA_CURRENT = "84d6c88b4bd84c3e6f0229dd23f7f188ff85622bbb39e5c3988a97598fa90d52";
const SHA_PREVIOUS = "5940f4ee0c1f2a63d4c8e1b7a9f6d2e3c4b5a69788f9e0d1c2b3a4f5e6d7c8b9";

function json(body: unknown, status = 200) {
  return Promise.resolve(
    new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    })
  );
}

function example(): { schema_version: "1.0"; models: Model[] } {
  return structuredClone(loadMlExample("models")) as { schema_version: "1.0"; models: Model[] };
}

function serve(body: unknown = example()) {
  vi.stubGlobal(
    "fetch",
    vi.fn((url: string) => (url === "/api/ml/models" ? json(body) : json({}, 404)))
  );
}

function openModels() {
  render(
    <MemoryRouter initialEntries={["/ml/models"]}>
      <App />
    </MemoryRouter>
  );
}

async function versionsTable() {
  return screen.findByRole("table", { name: /versiones del modelo/i });
}

function detail(version: string) {
  return screen.getByRole("region", { name: `dog-cat-resnet18 ${version}` });
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("APP-06 Models: versiones", () => {
  it("lista todas las versiones, también la anterior, con su release, run y estado", async () => {
    serve();
    openModels();

    const table = await versionsTable();
    const current = within(table).getByRole("row", { name: /1\.0\.0/ });
    expect(within(current).getByText("demo-v1.0.0")).toBeInTheDocument();
    expect(within(current).getByText("0a1b2c3d4e5f60718293a4b5c6d7e8f9")).toBeInTheDocument();
    expect(within(current).getByText("md5:5d41402abc4b2a76b9719d911017c592")).toBeInTheDocument();
    expect(within(current).getByText("Publicado en S3")).toBeInTheDocument();
    expect(within(current).getByText("Paquete disponible")).toBeInTheDocument();
    const previous = within(table).getByRole("row", { name: /0\.9\.0/ });
    expect(within(previous).getByText("demo-v0.9.0")).toBeInTheDocument();
    expect(within(previous).getByText("No publicado")).toBeInTheDocument();
    expect(within(previous).getByText("Paquete no descargado")).toBeInTheDocument();
  });

  it("ordena por versión semántica, la más nueva primero", async () => {
    const body = example();
    const newer = structuredClone(body.models[1]!);
    newer.model_version = "0.10.0";
    body.models = [body.models[1]!, newer, body.models[0]!];
    serve(body);
    openModels();

    const rows = within(await versionsTable()).getAllByRole("row").slice(1);
    expect(rows.map((row) => within(row).getAllByRole("cell")[0]!.textContent)).toEqual([
      "1.0.0",
      "0.10.0",
      "0.9.0",
    ]);
  });

  it("muestra un estado vacío si el registro no tiene versiones", async () => {
    serve({ schema_version: "1.0", models: [] });
    openModels();

    expect(await screen.findByText("Todavía no hay versiones registradas del modelo.")).toBeVisible();
  });
});

describe("APP-06 Models: detalle de una versión", () => {
  it("muestra checkpoint, métricas de test, model card, archivos y su publicación en S3", async () => {
    serve();
    openModels();
    await versionsTable();

    const version = detail("1.0.0");
    // También aparece en la tabla de S3: aquí importa el campo del checkpoint.
    expect(within(version).getByText("sha256 del checkpoint").nextElementSibling).toHaveTextContent(
      SHA_CURRENT
    );
    expect(
      within(version).getByText("runs:/0a1b2c3d4e5f60718293a4b5c6d7e8f9/checkpoints/best.pt")
    ).toBeInTheDocument();
    expect(within(version).getByText("resnet18 · 128×128")).toBeInTheDocument();
    expect(within(version).getByText("0.8000")).toBeInTheDocument();
    expect(within(version).getByText("0.7917")).toBeInTheDocument();
    expect(within(version).getByText("Clasificar recortes de cajas COCO en dog o cat.")).toBeInTheDocument();
    expect(within(version).getByText("Solo distingue dog y cat.")).toBeInTheDocument();
    expect(within(version).getByText("ResNet18_Weights.IMAGENET1K_V1")).toBeInTheDocument();
    expect(
      within(version).getByRole("link", { name: "Descargar model-card.md" })
    ).toHaveAttribute("href", "/api/ml/models/1.0.0/files/model-card.md");
    expect(
      within(version).getByRole("link", { name: "Descargar checkpoint/best.pt" })
    ).toHaveAttribute("href", "/api/ml/models/1.0.0/files/checkpoint/best.pt");
    const s3 = within(version).getByRole("table", { name: /objetos en s3/i });
    expect(within(s3).getByText("models/dog-cat-resnet18/1.0.0/checkpoint/best.pt")).toBeInTheDocument();
    expect(within(version).getByText("mlops-p2-dvc-cache-280764207006")).toBeInTheDocument();
    expect(within(version).getByRole("link", { name: /usar en inference/i })).toHaveAttribute(
      "href",
      "/ml/inference?model_version=1.0.0"
    );
  });

  it("Agent Test: elegir otra versión cambia el checkpoint y el hash que se muestran", async () => {
    serve();
    openModels();

    fireEvent.click(await screen.findByRole("button", { name: "Ver 0.9.0" }));

    const version = detail("0.9.0");
    expect(within(version).getByText(SHA_PREVIOUS)).toBeInTheDocument();
    expect(within(version).queryByText(SHA_CURRENT)).not.toBeInTheDocument();
    expect(
      within(version).getByText("runs:/1234567890abcdef1234567890abcdef/checkpoints/best.pt")
    ).toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "dog-cat-resnet18 1.0.0" })).not.toBeInTheDocument();
  });

  it("una versión anterior sin evaluación de test lo dice en vez de mostrar métricas (OPS-10)", async () => {
    const body = example();
    const previous = body.models.find((model) => model.model_version === "0.9.0")!;
    previous.test_metrics = null;
    serve(body);
    openModels();

    fireEvent.click(await screen.findByRole("button", { name: "Ver 0.9.0" }));

    const version = detail("0.9.0");
    expect(within(version).getAllByText("Sin evaluación en test")).toHaveLength(2);
    expect(within(version).queryByText(/^0\.\d{4}$/)).not.toBeInTheDocument();
  });

  it("una versión sin paquete no ofrece descargas ni Inference", async () => {
    serve();
    openModels();

    fireEvent.click(await screen.findByRole("button", { name: "Ver 0.9.0" }));

    const version = detail("0.9.0");
    expect(within(version).queryByRole("link", { name: /descargar/i })).not.toBeInTheDocument();
    expect(within(version).getAllByText("No está en este servidor")).toHaveLength(3);
    expect(within(version).queryByRole("link", { name: /usar en inference/i })).not.toBeInTheDocument();
    expect(within(version).getByText(/dvc pull data\/models\.dvc/)).toBeInTheDocument();
  });

  it("una publicación inconsistente explica el problema y no muestra keys", async () => {
    const body = example();
    body.models[0]!.publication = {
      status: "inconsistent",
      bucket: null,
      region: null,
      published_at: null,
      objects: [],
      problem: "checkpoint/best.pt: el ChecksumSHA256 de S3 no coincide con el archivo.",
    };
    serve(body);
    openModels();

    const table = await versionsTable();
    expect(within(table).getByText("Inconsistente")).toBeInTheDocument();
    const version = detail("1.0.0");
    expect(within(version).getByText(/el ChecksumSHA256 de S3 no coincide/)).toBeVisible();
    expect(within(version).queryByRole("table", { name: /objetos en s3/i })).not.toBeInTheDocument();
    expect(within(version).queryByText(/models\/dog-cat-resnet18/)).not.toBeInTheDocument();
  });

  it("una publicación que no se pudo verificar en S3 dice No verificable y por qué, sin keys", async () => {
    const body = example();
    body.models[0]!.publication = {
      status: "unverifiable",
      bucket: null,
      region: null,
      published_at: null,
      objects: [],
      problem: "No se pudo consultar S3: ml-api no tiene credenciales de AWS.",
    };
    serve(body);
    openModels();

    const table = await versionsTable();
    const current = within(table).getByRole("row", { name: /1\.0\.0/ });
    expect(within(current).getByText("No verificable")).toBeInTheDocument();
    expect(within(current).queryByText("Publicado en S3")).not.toBeInTheDocument();
    const version = detail("1.0.0");
    expect(within(version).getByRole("alert")).toHaveTextContent(
      /no se pudo confirmar en S3.*ml-api no tiene credenciales de AWS/i
    );
    expect(within(version).queryByRole("table", { name: /objetos en s3/i })).not.toBeInTheDocument();
    expect(within(version).queryByText(/models\/dog-cat-resnet18/)).not.toBeInTheDocument();
    expect(within(version).queryByText("mlops-p2-dvc-cache-280764207006")).not.toBeInTheDocument();
  });
});
