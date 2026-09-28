import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { CopilotPage } from "../src/pipeline/pages/Copilot";

const GROUNDED_REPLY = {
  answer: "Hay 600 imágenes en el dataset.",
  tool_calls: [
    {
      name: "get_dataset_summary",
      arguments: {},
      result: { dataset_version: "local-dev", total_images: 600 },
      is_error: false,
    },
  ],
  dataset_versions: ["local-dev"],
};

function jsonResponse(body: unknown, status = 200) {
  return Promise.resolve(
    new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    })
  );
}

function stubChat(...replies: Array<() => Promise<Response>>) {
  const fetchMock = vi.fn();
  for (const reply of replies) fetchMock.mockImplementationOnce(reply);
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function sentMessages(fetchMock: ReturnType<typeof vi.fn>, call = 0) {
  const init = fetchMock.mock.calls[call]?.[1] as RequestInit;
  return JSON.parse(init.body as string).messages as Array<{ role: string; content: string }>;
}

function ask(question: string) {
  fireEvent.change(screen.getByLabelText("Pregunta para el Copilot"), {
    target: { value: question },
  });
  fireEvent.click(screen.getByRole("button", { name: "Enviar" }));
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("SPEC-COPILOT-001 - chat del Copilot (P2-52)", () => {
  it("ya no es un placeholder: muestra el chat y preguntas de ejemplo", () => {
    render(<CopilotPage />);

    expect(screen.getByRole("heading", { name: "Copilot" })).toBeInTheDocument();
    expect(screen.queryByText(/todavía no está implementado/)).not.toBeInTheDocument();
    expect(screen.getByLabelText("Pregunta para el Copilot")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "¿Cuántas imágenes hay por categoría?" })
    ).toBeInTheDocument();
  });

  it("no envía preguntas vacías", () => {
    const fetchMock = stubChat();
    render(<CopilotPage />);

    expect(screen.getByRole("button", { name: "Enviar" })).toBeDisabled();
    fireEvent.change(screen.getByLabelText("Pregunta para el Copilot"), {
      target: { value: "   " },
    });
    expect(screen.getByRole("button", { name: "Enviar" })).toBeDisabled();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("muestra la respuesta con las herramientas consultadas y la versión del dataset", async () => {
    const fetchMock = stubChat(() => jsonResponse(GROUNDED_REPLY));
    render(<CopilotPage />);

    ask("¿Cuántas imágenes hay?");

    expect(await screen.findByText("Hay 600 imágenes en el dataset.")).toBeInTheDocument();
    expect(fetchMock.mock.calls[0]?.[0]).toBe("/copilot-api/chat");
    expect(sentMessages(fetchMock)).toEqual([{ role: "user", content: "¿Cuántas imágenes hay?" }]);
    expect(screen.getByText("get_dataset_summary()")).toBeInTheDocument();
    expect(screen.getByText(/Versión del dataset consultada/)).toBeInTheDocument();
    expect(screen.getAllByText("local-dev").length).toBeGreaterThan(0);
    // El resultado crudo de la herramienta queda a la vista para auditar la cifra.
    expect(screen.getByText(/"total_images": 600/)).toBeInTheDocument();
  });

  it("marca las respuestas que no consultaron ninguna herramienta", async () => {
    stubChat(() =>
      jsonResponse({ answer: "No puedo saberlo.", tool_calls: [], dataset_versions: [] })
    );
    render(<CopilotPage />);

    ask("¿Quién ganó el mundial?");

    expect(await screen.findByText("No puedo saberlo.")).toBeInTheDocument();
    expect(screen.getByText(/Sin consultas a herramientas/)).toBeInTheDocument();
  });

  it("reenvía el historial completo en la pregunta siguiente", async () => {
    const fetchMock = stubChat(
      () => jsonResponse(GROUNDED_REPLY),
      () => jsonResponse({ answer: "Sí.", tool_calls: [], dataset_versions: [] })
    );
    render(<CopilotPage />);

    ask("¿Cuántas imágenes hay?");
    await screen.findByText("Hay 600 imágenes en el dataset.");
    ask("¿Seguro?");
    await screen.findByText("Sí.");

    expect(sentMessages(fetchMock, 1)).toEqual([
      { role: "user", content: "¿Cuántas imágenes hay?" },
      { role: "assistant", content: "Hay 600 imágenes en el dataset." },
      { role: "user", content: "¿Seguro?" },
    ]);
  });

  it("una pregunta de ejemplo se envía al hacer clic", async () => {
    const fetchMock = stubChat(() => jsonResponse(GROUNDED_REPLY));
    render(<CopilotPage />);

    fireEvent.click(screen.getByRole("button", { name: "¿Cuántas imágenes hay por categoría?" }));

    await screen.findByText("Hay 600 imágenes en el dataset.");
    expect(sentMessages(fetchMock)[0]?.content).toBe("¿Cuántas imágenes hay por categoría?");
    expect(screen.queryByRole("button", { name: /¿Cómo quedaron los splits/ })).toBeNull();
  });

  it("si el servicio falla muestra su mensaje, sin trazas, y permite reintentar sin duplicar la pregunta", async () => {
    const fetchMock = stubChat(
      () =>
        jsonResponse(
          { error: "El proveedor de IA no respondió. Intenta de nuevo en unos minutos." },
          502
        ),
      () => jsonResponse(GROUNDED_REPLY)
    );
    render(<CopilotPage />);

    ask("¿Cuántas imágenes hay?");

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("El proveedor de IA no respondió");
    expect(alert).not.toHaveTextContent(/traceback|stack/i);
    // La pregunta del usuario sigue en pantalla.
    expect(screen.getByText("¿Cuántas imágenes hay?")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Reintentar" }));

    expect(await screen.findByText("Hay 600 imágenes en el dataset.")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(sentMessages(fetchMock, 1)).toEqual([{ role: "user", content: "¿Cuántas imágenes hay?" }]);
    expect(screen.getAllByText("¿Cuántas imágenes hay?")).toHaveLength(1);
  });

  it("explica que falta la API key cuando el servicio responde 503", async () => {
    stubChat(() =>
      jsonResponse({ error: "El Copilot no está configurado: falta ANTHROPIC_API_KEY." }, 503)
    );
    render(<CopilotPage />);

    ask("hola");

    expect(await screen.findByRole("alert")).toHaveTextContent("falta ANTHROPIC_API_KEY");
  });

  it("si no hay red, lo dice sin romper la pantalla", async () => {
    stubChat(() => Promise.reject(new TypeError("Failed to fetch")));
    render(<CopilotPage />);

    ask("hola");

    expect(await screen.findByRole("alert")).toHaveTextContent("No se pudo contactar al Copilot");
    expect(screen.getByRole("heading", { name: "Copilot" })).toBeInTheDocument();
  });

  it("un 504 de nginx (respuesta que no es JSON) se traduce a un mensaje claro", async () => {
    stubChat(() => Promise.resolve(new Response("<html>Gateway Time-out</html>", { status: 504 })));
    render(<CopilotPage />);

    ask("hola");

    expect(await screen.findByRole("alert")).toHaveTextContent("tardó demasiado");
  });

  it("rechaza una respuesta con formato inesperado en vez de mostrarla", async () => {
    stubChat(() => jsonResponse({ respuesta: "sin contrato" }));
    render(<CopilotPage />);

    ask("hola");

    expect(await screen.findByRole("alert")).toHaveTextContent("formato esperado");
    expect(screen.queryByText("sin contrato")).not.toBeInTheDocument();
  });

  it("no manda una segunda petición mientras la primera sigue en curso", async () => {
    let finish: (response: Response) => void = () => {};
    const pending = new Promise<Response>((resolve) => {
      finish = resolve;
    });
    const fetchMock = stubChat(() => pending);
    render(<CopilotPage />);

    ask("primera");
    await waitFor(() => expect(screen.getByText(/consultando el dataset/)).toBeInTheDocument());
    ask("segunda");

    expect(fetchMock).toHaveBeenCalledTimes(1);
    finish(new Response(JSON.stringify(GROUNDED_REPLY), { status: 200 }));
    expect(await screen.findByText("Hay 600 imágenes en el dataset.")).toBeInTheDocument();
  });
});
