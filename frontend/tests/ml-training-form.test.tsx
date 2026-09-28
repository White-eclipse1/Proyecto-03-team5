import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "../src/App";
import qualityFixture from "./fixtures/quality.json";
import { loadMlExample } from "./mlCorpus";

// APP-02: pantalla Training. Los releases, el gate, las clases y la procedencia
// salen de los reportes publicados de P2 (/reports); los parámetros se validan
// con el mismo contrato que Python antes de cualquier POST.

const CATALOG = {
  schema_version: "1.0",
  releases: [
    {
      dataset_version: "v0.1.0",
      quality_file: "releases/v0.1.0/quality.json",
      splits_file: "releases/v0.1.0/splits.json",
    },
    {
      dataset_version: "v0.1.1",
      quality_file: "releases/v0.1.1/quality.json",
      splits_file: "releases/v0.1.1/splits.json",
    },
  ],
};

const QUALITY: Record<string, unknown> = {
  "v0.1.0": { ...qualityFixture, dataset_version: "v0.1.0", status: "failed" },
  "v0.1.1": { ...qualityFixture, dataset_version: "v0.1.1", status: "warning" },
};

const PROVENANCE = {
  ...(loadMlExample("provenance") as object),
  dataset_version: "v0.1.1",
};

const MANIFEST = { ...(loadMlExample("manifest") as object), dataset_version: "v0.1.1" };

const CREATED_JOB = {
  ...(loadMlExample("training_jobs") as { jobs: Record<string, unknown>[] }).jobs[0],
  job_id: "job-0005",
  dataset_version: "v0.1.1",
};

function json(body: unknown, status = 200) {
  return Promise.resolve(
    new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    })
  );
}

interface ServeOptions {
  manifest?: boolean;
  provenance?: boolean;
  post?: () => Promise<Response>;
}

function serve({ manifest = false, provenance = true, post }: ServeOptions = {}) {
  const posts: unknown[] = [];
  const fetcher = vi.fn((url: string, init?: RequestInit) => {
    if (url === "/api/ml/training/jobs" && init?.method === "POST") {
      posts.push(JSON.parse(String(init.body)));
      return post
        ? post()
        : json({ ...CREATED_JOB, params: (posts.at(-1) as { params: unknown }).params }, 201);
    }
    if (url === "/api/ml/training/jobs") return json(loadMlExample("training_jobs"));
    if (url === "/reports/versions.json") return json(CATALOG);
    const release = /^\/reports\/releases\/(v[\d.]+)\/(\w+)\.json$/.exec(url);
    if (release) {
      const [, version = "", file] = release;
      if (file === "quality") return json(QUALITY[version]);
      if (file === "provenance" && provenance && version === "v0.1.1") return json(PROVENANCE);
      if (file === "manifest" && manifest && version === "v0.1.1") return json(MANIFEST);
    }
    return json({}, 404);
  });
  vi.stubGlobal("fetch", fetcher);
  return { fetcher, posts };
}

function openTraining() {
  render(
    <MemoryRouter initialEntries={["/ml/training"]}>
      <App />
    </MemoryRouter>
  );
}

async function selectRelease(version: string) {
  const select = await screen.findByLabelText("Release del dataset");
  await waitFor(() =>
    expect(within(select).getByRole("option", { name: version })).toBeInTheDocument()
  );
  fireEvent.change(select, { target: { value: version } });
}

function setParam(name: string, value: string) {
  fireEvent.change(screen.getByLabelText(name), { target: { value } });
}

function submit() {
  fireEvent.click(screen.getByRole("button", { name: "Crear training job" }));
}

function postCalls(fetcher: ReturnType<typeof serve>["fetcher"]) {
  return fetcher.mock.calls.filter(([, init]) => init?.method === "POST");
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("APP-02 release del Proyecto 2", () => {
  it("lista los releases publicados y muestra versión, gate, hash DVC y clases", async () => {
    serve();
    openTraining();
    await selectRelease("v0.1.1");

    const details = await screen.findByRole("region", { name: "Release seleccionado" });
    expect(within(details).getByText("v0.1.1")).toBeInTheDocument();
    expect(within(details).getByText("warning")).toBeInTheDocument();
    expect(
      await within(details).findByText("images: 8f14e45fceea167a5a36dedd4bea2543.dir")
    ).toBeInTheDocument();
    expect(
      within(details).getByText("annotations: c9f0f895fb98ab9159f51fd0297e236d.dir")
    ).toBeInTheDocument();
    expect(within(details).getByText("dog (300)")).toBeInTheDocument();
    expect(within(details).getByText("cat (301)")).toBeInTheDocument();
    // Categorías sin imágenes no son clases entrenables.
    expect(within(details).queryByText(/person/)).not.toBeInTheDocument();
  });

  it("dice explícitamente cuando el release no tiene hash DVC registrado", async () => {
    serve({ provenance: false });
    openTraining();
    await selectRelease("v0.1.1");

    expect(
      await screen.findByText("Hash DVC no registrado para este release.")
    ).toBeInTheDocument();
  });

  it("muestra el manifiesto 70/20/10 solo cuando existe", async () => {
    serve({ manifest: true });
    openTraining();
    await selectRelease("v0.1.1");

    const manifest = await screen.findByRole("list", { name: "Manifiesto de splits" });
    expect(within(manifest).getByText("train: 420 imágenes (70%)")).toBeInTheDocument();
    expect(within(manifest).getByText("validation: 120 imágenes (20%)")).toBeInTheDocument();
    expect(within(manifest).getByText("test: 60 imágenes (10%)")).toBeInTheDocument();
    const details = screen.getByRole("region", { name: "Release seleccionado" });
    expect(within(details).getByText("md5:5d41402abc4b2a76b9719d911017c592")).toBeInTheDocument();
  });

  it("indica que el manifiesto todavía no está disponible", async () => {
    serve();
    openTraining();
    await selectRelease("v0.1.1");

    expect(
      await screen.findByText("El manifiesto 70/20/10 de este release todavía no está disponible.")
    ).toBeInTheDocument();
  });

  it("un Quality Gate fallido impide iniciar el entrenamiento", async () => {
    const { fetcher } = serve();
    openTraining();
    await selectRelease("v0.1.0");

    expect(
      await screen.findByText(
        "El release no pasó el Quality Gate (failed); no se puede entrenar con él."
      )
    ).toBeVisible();
    expect(screen.getByRole("button", { name: "Crear training job" })).toBeDisabled();
    submit();
    expect(postCalls(fetcher)).toHaveLength(0);
  });
});

describe("APP-02 parámetros del entrenamiento", () => {
  it("incluye los 10 hiperparámetros requeridos", async () => {
    serve();
    openTraining();
    await screen.findByLabelText("Release del dataset");

    for (const name of [
      "optimizer",
      "batch_size",
      "max_epochs",
      "learning_rate",
      "image_size",
      "hidden_layers",
      "dropout",
      "seed",
      "patience",
      "min_delta",
    ]) {
      expect(screen.getByLabelText(name)).toBeInTheDocument();
    }
  });

  // Agent Test del ticket: los tres deben rechazarse antes de crear un job.
  it.each([
    ["batch_size", "-1", "batch_size debe ser al menos 1."],
    ["learning_rate", "abc", "learning_rate debe ser un número."],
    ["learning_rate", "0", "learning_rate debe ser mayor que 0."],
    ["dropout", "1.5", "dropout debe ser menor que 1."],
    ["max_epochs", "0", "max_epochs debe ser al menos 1."],
    ["image_size", "225", "image_size debe ser múltiplo de 32."],
    ["hidden_layers", "512, x", "hidden_layers debe ser una lista de enteros separados por comas."],
    ["patience", "80", "patience no puede ser mayor que max_epochs."],
    ["min_delta", "-0.1", "min_delta no puede ser negativo."],
    ["seed", "-3", "seed no puede ser negativo."],
  ])("rechaza %s = %s antes de crear el job", async (name, value, message) => {
    const { fetcher } = serve();
    openTraining();
    await selectRelease("v0.1.1");
    await screen.findByText("warning");

    setParam(name, value);
    submit();

    expect(await screen.findByText(message)).toBeVisible();
    expect(screen.getByLabelText(name)).toHaveAttribute("aria-invalid", "true");
    expect(
      screen.getByText("Revisa los parámetros marcados; no se creó ningún job.")
    ).toBeVisible();
    expect(postCalls(fetcher)).toHaveLength(0);
  });

  it("exige elegir un release antes de enviar", async () => {
    const { fetcher } = serve();
    openTraining();
    await screen.findByLabelText("Release del dataset");

    submit();

    expect(await screen.findByText("Selecciona un release aprobado.")).toBeVisible();
    expect(postCalls(fetcher)).toHaveLength(0);
  });

  it("envía un TrainingJobRequest válido y muestra el job creado", async () => {
    const { posts } = serve();
    openTraining();
    await selectRelease("v0.1.1");
    await screen.findByText("warning");

    setParam("optimizer", "sgd");
    setParam("batch_size", "64");
    setParam("hidden_layers", "1024, 128");
    submit();

    expect(await screen.findByText("Job job-0005 creado (queued).")).toBeVisible();
    expect(posts).toEqual([
      {
        schema_version: "1.0",
        dataset_version: "v0.1.1",
        params: {
          optimizer: "sgd",
          batch_size: 64,
          max_epochs: 50,
          learning_rate: 0.001,
          image_size: 224,
          hidden_layers: [1024, 128],
          dropout: 0.2,
          seed: 42,
          patience: 5,
          min_delta: 0.001,
        },
      },
    ]);
  });

  it("muestra el error del backend de forma comprensible", async () => {
    serve({
      post: () =>
        json(
          {
            schema_version: "1.0",
            error: {
              code: "release_not_found",
              message: "El release v0.1.1 no existe en el catálogo.",
              retryable: false,
            },
          },
          422
        ),
    });
    openTraining();
    await selectRelease("v0.1.1");
    await screen.findByText("warning");

    submit();

    expect(
      await screen.findByText("El release v0.1.1 no existe en el catálogo.")
    ).toBeVisible();
  });

  it("avisa que el servicio no está conectado si el POST responde 404", async () => {
    serve({ post: () => json({}, 404) });
    openTraining();
    await selectRelease("v0.1.1");
    await screen.findByText("warning");

    submit();

    expect(
      await screen.findByText(
        "El servicio de entrenamiento todavía no está conectado; no se creó ningún job."
      )
    ).toBeVisible();
  });
});
