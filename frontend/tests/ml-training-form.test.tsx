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

/** Por defecto el release v0.1.1 está listo: procedencia DVC y manifiesto 70/20/10 válidos. */
interface ServeOptions {
  manifest?: unknown;
  provenance?: unknown;
  post?: () => Promise<Response>;
}

function serve({ manifest = MANIFEST, provenance = PROVENANCE, post }: ServeOptions = {}) {
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
      if (file === "provenance" && provenance !== null && version === "v0.1.1") {
        return json(provenance);
      }
      if (file === "manifest" && manifest !== null && version === "v0.1.1") return json(manifest);
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

function trainButton() {
  return screen.getByRole("button", { name: "Crear training job" });
}

/** El botón se habilita cuando gate, procedencia y manifiesto ya se verificaron. */
async function waitForReadyRelease() {
  await waitFor(() => expect(trainButton()).toBeEnabled());
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

  it("sin provenance.json DVC no deja entrenar", async () => {
    const { fetcher } = serve({ provenance: null });
    openTraining();
    await selectRelease("v0.1.1");

    expect(
      await screen.findByText("Hash DVC no registrado para este release.")
    ).toBeInTheDocument();
    expect(
      await screen.findByText(
        "El release no tiene un provenance.json de DVC válido; no se puede entrenar de forma reproducible."
      )
    ).toBeVisible();
    expect(trainButton()).toBeDisabled();
    submit();
    expect(postCalls(fetcher)).toHaveLength(0);
  });

  it("muestra el manifiesto 70/20/10 cuando existe", async () => {
    serve();
    openTraining();
    await selectRelease("v0.1.1");

    const manifest = await screen.findByRole("list", { name: "Manifiesto de splits" });
    expect(within(manifest).getByText("train: 420 imágenes (70%)")).toBeInTheDocument();
    expect(within(manifest).getByText("validation: 120 imágenes (20%)")).toBeInTheDocument();
    expect(within(manifest).getByText("test: 60 imágenes (10%)")).toBeInTheDocument();
    const details = screen.getByRole("region", { name: "Release seleccionado" });
    expect(within(details).getByText("md5:5d41402abc4b2a76b9719d911017c592")).toBeInTheDocument();
  });

  it("sin manifest.json no deja entrenar", async () => {
    const { fetcher } = serve({ manifest: null });
    openTraining();
    await selectRelease("v0.1.1");

    expect(
      await screen.findByText("El manifiesto 70/20/10 de este release todavía no está disponible.")
    ).toBeInTheDocument();
    expect(
      await screen.findByText(
        "El release no tiene un manifest.json 70/20/10 válido; no se puede entrenar de forma reproducible."
      )
    ).toBeVisible();
    expect(trainButton()).toBeDisabled();
    submit();
    expect(postCalls(fetcher)).toHaveLength(0);
  });

  it.each([
    [
      "no es 70/20/10",
      {
        ...MANIFEST,
        splits: {
          train: { image_count: 360, ratio: 0.6 },
          validation: { image_count: 180, ratio: 0.3 },
          test: { image_count: 60, ratio: 0.1 },
        },
      },
    ],
    ["es de otro release", { ...MANIFEST, dataset_version: "v0.1.0" }],
  ])("un manifest.json que %s no deja entrenar", async (_case, manifest) => {
    const { fetcher } = serve({ manifest });
    openTraining();
    await selectRelease("v0.1.1");

    expect(await screen.findByText(/No se pudo leer el manifiesto/)).toBeInTheDocument();
    expect(
      await screen.findByText(
        "El release no tiene un manifest.json 70/20/10 válido; no se puede entrenar de forma reproducible."
      )
    ).toBeVisible();
    expect(trainButton()).toBeDisabled();
    submit();
    expect(postCalls(fetcher)).toHaveLength(0);
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
    await waitForReadyRelease();

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

  it("con procedencia DVC y manifiesto 70/20/10 válidos, envía un request ligado al manifest_hash", async () => {
    const { posts } = serve({ provenance: PROVENANCE, manifest: MANIFEST });
    openTraining();
    await selectRelease("v0.1.1");
    await waitForReadyRelease();

    setParam("optimizer", "sgd");
    setParam("batch_size", "64");
    setParam("hidden_layers", "1024, 128");
    submit();

    expect(await screen.findByText("Job job-0005 creado (queued).")).toBeVisible();
    expect(posts).toEqual([
      {
        schema_version: "1.0",
        dataset_version: "v0.1.1",
        manifest_hash: "md5:5d41402abc4b2a76b9719d911017c592",
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
    await waitForReadyRelease();

    submit();

    expect(
      await screen.findByText("El release v0.1.1 no existe en el catálogo.")
    ).toBeVisible();
  });

  it("avisa que el servicio no está conectado si el POST responde 404", async () => {
    serve({ post: () => json({}, 404) });
    openTraining();
    await selectRelease("v0.1.1");
    await waitForReadyRelease();

    submit();

    expect(
      await screen.findByText(
        "El servicio de entrenamiento todavía no está conectado; no se creó ningún job."
      )
    ).toBeVisible();
  });
});
