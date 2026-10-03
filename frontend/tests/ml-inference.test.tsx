import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "../src/App";
import { MAX_UPLOAD_BYTES } from "../src/ml/pages/Inference";
import { loadMlExample } from "./mlCorpus";

// APP-07 (issue #24): la pantalla Inference clasifica una imagen subida o un recorte
// del portal con la model version elegida (`POST /api/ml/inference/upload` y
// `POST /api/ml/inference`). Los datos vienen del corpus compartido con Python.

type Json = Record<string, unknown>;

const MODELS_URL = "/api/ml/models";
const UPLOAD_URL = "/api/ml/inference/upload";
const CROP_URL = "/api/ml/inference";
const SHA = "84d6c88b4bd84c3e6f0229dd23f7f188ff85622bbb39e5c3988a97598fa90d52";

function json(body: unknown, status = 200) {
  return Promise.resolve(
    new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    })
  );
}

function uploadResponse(changes: Json = {}): Json {
  return { ...(structuredClone(loadMlExample("inference_upload")) as Json), ...changes };
}

function cropResponse(changes: Json = {}): Json {
  return { ...(structuredClone(loadMlExample("inference")) as Json), ...changes };
}

type Call = { url: string; init?: RequestInit };

/** Las dos versiones del ejemplo con su paquete en el servidor (en el corpus, 0.9.0 no). */
function bothServable() {
  const models = structuredClone(loadMlExample("models")) as {
    models: { servable: boolean; files: { available: boolean }[] }[];
  };
  for (const model of models.models) {
    model.servable = true;
    for (const file of model.files) file.available = true;
  }
  return models;
}

function serve(handler: (call: Call) => Promise<Response> = defaultHandler) {
  const calls: Call[] = [];
  const fetcher = vi.fn((url: string, init?: RequestInit) => {
    calls.push({ url, init });
    if (url === MODELS_URL) return json(bothServable());
    return handler({ url, init });
  });
  vi.stubGlobal("fetch", fetcher);
  return calls;
}

function defaultHandler({ url, init }: Call) {
  if (url === UPLOAD_URL) {
    const version = (init?.body as FormData).get("model_version");
    return json(uploadResponse({ model_version: version }));
  }
  if (url === CROP_URL) return json(cropResponse());
  return json({}, 404);
}

function openInference(path = "/ml/inference") {
  render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>
  );
}

function png(name = "mi-gato.png", size?: number) {
  const file = new File([new Uint8Array([0x89, 0x50, 0x4e, 0x47])], name, { type: "image/png" });
  if (size !== undefined) Object.defineProperty(file, "size", { value: size });
  return file;
}

async function chooseFile(file: File) {
  const input = await screen.findByLabelText(/imagen \(png o jpeg\)/i);
  fireEvent.change(input, { target: { files: [file] } });
}

function sentTo(calls: Call[], url: string) {
  return calls.filter((call) => call.url === url);
}

beforeEach(() => {
  vi.stubGlobal("URL", Object.assign(URL, { createObjectURL: () => "blob:preview", revokeObjectURL: () => {} }));
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("APP-07 Inference: imagen subida", () => {
  it("clasifica con la model version elegida y muestra clase, probabilidades y checkpoint", async () => {
    const calls = serve();
    openInference();

    await chooseFile(png());
    fireEvent.click(screen.getByRole("button", { name: "Clasificar" }));

    const result = await screen.findByRole("region", { name: /resultado/i });
    expect(within(result).getByText("Clase predicha").nextElementSibling).toHaveTextContent("cat");
    expect(within(result).getByText("96.9%")).toBeInTheDocument();
    expect(within(result).getByText("3.1%")).toBeInTheDocument();
    expect(within(result).getByText("mi-gato.jpg")).toBeInTheDocument();
    expect(within(result).getByText(SHA)).toBeInTheDocument();
    expect(within(result).getByText("dog-cat-resnet18 v1.0.0")).toBeInTheDocument();

    const [sent] = sentTo(calls, UPLOAD_URL);
    const body = sent!.init!.body as FormData;
    expect(sent!.init!.method).toBe("POST");
    expect(body.get("model_name")).toBe("dog-cat-resnet18");
    expect(body.get("model_version")).toBe("1.0.0");
    expect((body.get("file") as File).name).toBe("mi-gato.png");
  });

  it("Agent Test: cambiar de model version cambia la versión que se envía y la que se muestra", async () => {
    const calls = serve();
    openInference();

    fireEvent.change(await screen.findByLabelText(/model version/i), {
      target: { value: "dog-cat-resnet18:0.9.0" },
    });
    await chooseFile(png());
    fireEvent.click(screen.getByRole("button", { name: "Clasificar" }));

    const result = await screen.findByRole("region", { name: /resultado/i });
    expect(within(result).getByText("dog-cat-resnet18 v0.9.0")).toBeInTheDocument();
    expect((sentTo(calls, UPLOAD_URL)[0]!.init!.body as FormData).get("model_version")).toBe("0.9.0");
  });

  it("rechaza en el navegador un archivo que no es PNG ni JPEG", async () => {
    const calls = serve();
    openInference();

    await chooseFile(new File(["hola"], "notas.txt", { type: "text/plain" }));

    expect(await screen.findByText("Solo se aceptan imágenes PNG o JPEG.")).toBeVisible();
    expect(screen.getByRole("button", { name: "Clasificar" })).toBeDisabled();
    expect(sentTo(calls, UPLOAD_URL)).toHaveLength(0);
  });

  it("rechaza en el navegador un archivo demasiado grande", async () => {
    const calls = serve();
    openInference();

    await chooseFile(png("enorme.png", MAX_UPLOAD_BYTES + 1));

    expect(await screen.findByText(/el archivo pesa más de 10 MB/i)).toBeVisible();
    expect(screen.getByRole("button", { name: "Clasificar" })).toBeDisabled();
    expect(sentTo(calls, UPLOAD_URL)).toHaveLength(0);
  });

  it("muestra el mensaje del servidor si la imagen no se puede leer", async () => {
    serve(() =>
      json(
        {
          schema_version: "1.0",
          error: {
            code: "invalid_image",
            message: "La imagen está dañada o incompleta y no se pudo leer.",
            retryable: false,
          },
        },
        422
      )
    );
    openInference();

    await chooseFile(png());
    fireEvent.click(screen.getByRole("button", { name: "Clasificar" }));

    expect(
      await screen.findByText("La imagen está dañada o incompleta y no se pudo leer.")
    ).toBeVisible();
    expect(screen.getByText("invalid_image")).toBeInTheDocument();
    expect(screen.queryByRole("region", { name: /resultado/i })).not.toBeInTheDocument();
  });

  it("no muestra una respuesta de otro modelo como si fuera la del elegido", async () => {
    serve(({ url }) =>
      url === UPLOAD_URL ? json(uploadResponse({ model_version: "0.9.0" })) : json({}, 404)
    );
    openInference();

    await chooseFile(png());
    fireEvent.click(screen.getByRole("button", { name: "Clasificar" }));

    expect(
      await screen.findByText("La respuesta no corresponde a la model version elegida.")
    ).toBeVisible();
    expect(screen.queryByRole("region", { name: /resultado/i })).not.toBeInTheDocument();
  });
});

describe("APP-07 Inference: recorte del portal", () => {
  it("clasifica un recorte por release, image_id y annotation_id", async () => {
    const calls = serve();
    openInference();

    fireEvent.click(await screen.findByRole("button", { name: "Recorte del portal" }));
    fireEvent.change(screen.getByLabelText("image_id"), { target: { value: "42" } });
    fireEvent.change(screen.getByLabelText("annotation_id"), { target: { value: "1007" } });
    fireEvent.click(screen.getByRole("button", { name: "Clasificar" }));

    const result = await screen.findByRole("region", { name: /resultado/i });
    expect(within(result).getByText("demo-v1.0.0 · img 42 · ann 1007")).toBeInTheDocument();
    const [sent] = sentTo(calls, CROP_URL);
    expect(JSON.parse(sent!.init!.body as string)).toEqual({
      schema_version: "1.0",
      model_name: "dog-cat-resnet18",
      model_version: "1.0.0",
      crop: { dataset_version: "demo-v1.0.0", image_id: 42, annotation_id: 1007 },
    });
    expect(screen.getByRole("img", { name: "Recorte img42-ann1007" })).toHaveAttribute(
      "src",
      "/api/ml/crops/img42-ann1007"
    );
  });

  it("abre con la model version indicada en la URL (enlace desde Models)", async () => {
    serve();
    openInference("/ml/inference?model_version=0.9.0");

    expect(await screen.findByLabelText(/model version/i)).toHaveValue("dog-cat-resnet18:0.9.0");
  });

  it("abre con el recorte indicado en la URL", async () => {
    serve();
    openInference("/ml/inference?dataset_version=v0.1.1&image_id=12&annotation_id=11");

    expect(await screen.findByLabelText("image_id")).toHaveValue(12);
    expect(screen.getByLabelText("annotation_id")).toHaveValue(11);
    expect(screen.getByLabelText("Release del recorte")).toHaveValue("v0.1.1");
  });

  it("no envía un recorte incompleto", async () => {
    const calls = serve();
    openInference();

    fireEvent.click(await screen.findByRole("button", { name: "Recorte del portal" }));

    expect(screen.getByRole("button", { name: "Clasificar" })).toBeDisabled();
    await waitFor(() => expect(sentTo(calls, CROP_URL)).toHaveLength(0));
  });
});

describe("APP-08 Inference → annotation queue", () => {
  it("envía una inferencia real a la cola y conserva la trazabilidad", async () => {
    const calls = serve(({ url, init }) => {
      if (url === UPLOAD_URL) {
        const version = (init?.body as FormData).get("model_version");
        return json(uploadResponse({ model_version: version }));
      }

      if (url === "/api/images/from-inference") {
        return json(
          {
            imageId: 321,
            created: true,
            idempotencyKey: "a".repeat(64),
          },
          201
        );
      }

      return json({}, 404);
    });

    openInference();

    await chooseFile(png());
    fireEvent.click(screen.getByRole("button", { name: "Clasificar" }));

    await screen.findByRole("region", { name: /resultado/i });

    fireEvent.click(
      screen.getByRole("button", {
        name: /enviar a cola de anotación/i,
      })
    );

    expect(await screen.findByText(/enviado a la cola de anotación/i)).toBeVisible();

    const queueCalls = sentTo(calls, "/api/images/from-inference");
    expect(queueCalls).toHaveLength(1);

    const body = queueCalls[0]!.init!.body as FormData;
    expect(body.get("image")).toBeInstanceOf(File);

    const metadata = JSON.parse(String(body.get("metadata")));

    expect(metadata).toMatchObject({
      sourceKind: "upload",
      modelName: "dog-cat-resnet18",
      modelVersion: "1.0.0",
      runId: "0a1b2c3d4e5f60718293a4b5c6d7e8f9",
      predictedClass: "cat",
    });

    expect(metadata.sourceRef).toMatch(/^sha256:/);
    expect(metadata.probabilities).toEqual({
      dog: 0.031,
      cat: 0.969,
    });
  });

  it("doble click no crea dos requests mientras el primero sigue activo", async () => {
    let resolveQueue!: (response: Response) => void;

    const queuePromise = new Promise<Response>((resolve) => {
      resolveQueue = resolve;
    });

    const calls = serve(({ url, init }) => {
      if (url === UPLOAD_URL) {
        const version = (init?.body as FormData).get("model_version");
        return json(uploadResponse({ model_version: version }));
      }

      if (url === "/api/images/from-inference") {
        return queuePromise;
      }

      return json({}, 404);
    });

    openInference();

    await chooseFile(png());
    fireEvent.click(screen.getByRole("button", { name: "Clasificar" }));

    await screen.findByRole("region", { name: /resultado/i });

    const button = screen.getByRole("button", {
      name: /enviar a cola de anotación/i,
    });

    fireEvent.click(button);

    await waitFor(() => expect(button).toBeDisabled());

    fireEvent.click(button);

    expect(sentTo(calls, "/api/images/from-inference")).toHaveLength(1);

    resolveQueue(
      new Response(
        JSON.stringify({
          imageId: 321,
          created: true,
          idempotencyKey: "a".repeat(64),
        }),
        {
          status: 201,
          headers: { "Content-Type": "application/json" },
        }
      )
    );

    expect(await screen.findByText(/enviado a la cola de anotación/i)).toBeVisible();
  });

  it("muestra al usuario un fallo del backend", async () => {
    serve(({ url, init }) => {
      if (url === UPLOAD_URL) {
        const version = (init?.body as FormData).get("model_version");
        return json(uploadResponse({ model_version: version }));
      }

      if (url === "/api/images/from-inference") {
        return json(
          {
            error: "No se pudo crear la entrada.",
          },
          500
        );
      }

      return json({}, 404);
    });

    openInference();

    await chooseFile(png());
    fireEvent.click(screen.getByRole("button", { name: "Clasificar" }));

    await screen.findByRole("region", { name: /resultado/i });

    fireEvent.click(
      screen.getByRole("button", {
        name: /enviar a cola de anotación/i,
      })
    );

    expect(await screen.findByRole("alert")).toBeVisible();
  });
});
