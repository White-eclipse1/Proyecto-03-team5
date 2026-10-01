import "@testing-library/jest-dom/vitest";
import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "../src/App";
import { RUNS_POLL_MS } from "../src/ml/pages/Experiments";
import { loadMlExample } from "./mlCorpus";

// APP-04: la pantalla Experiments lee runs reales de MLflow vía ml-api. Los datos
// vienen del corpus compartido con Python; la pantalla no trae filas propias.

type Run = Record<string, unknown> & { run_id: string; status: string };
type Runs = { schema_version: "1.0"; runs: Run[] };

const RUNS_URL = "/api/ml/runs";
const FINISHED = "0a1b2c3d4e5f60718293a4b5c6d7e8f9";
const RUNNING = "f1e2d3c4b5a697887766554433221100";
const FAILED = "9c8b7a6d5e4f30211203f4e5d6c7b8a9";
const EXTRA = "abcdefabcdefabcdefabcdefabcdef01";

function json(body: unknown, status = 200) {
  return Promise.resolve(
    new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    })
  );
}

function runsExample(): Runs {
  return structuredClone(loadMlExample("runs")) as Runs;
}

function withExtraFinishedRun(runs: Runs): Runs {
  const extra = structuredClone(runs.runs.find((run) => run.run_id === FINISHED)!);
  extra.run_id = EXTRA;
  extra.run_name = "resnet18-sgd";
  (extra.params as Record<string, string>).optimizer = "sgd";
  (extra.metrics as Record<string, number>).val_accuracy_top1 = 0.91;
  runs.runs.push(extra);
  return runs;
}

function onlyFinished(): Runs {
  const runs = runsExample();
  runs.runs = runs.runs.filter((run) => run.status !== "RUNNING");
  return runs;
}

function curvesFor(runId: string) {
  const curves = structuredClone(loadMlExample("run_curves")) as { run_id: string };
  curves.run_id = runId;
  return curves;
}

function serve(runs: () => Runs = runsExample, curves = (id: string) => json(curvesFor(id))) {
  const fetcher = vi.fn((url: string) => {
    if (url === RUNS_URL) return json(runs());
    const match = /^\/api\/ml\/runs\/([0-9a-f]{32})\/curves$/.exec(url);
    if (match) return curves(match[1]!);
    return json({}, 404);
  });
  vi.stubGlobal("fetch", fetcher);
  return fetcher;
}

const callsTo = (fetcher: ReturnType<typeof serve>, url: string) =>
  fetcher.mock.calls.filter(([called]) => called === url);

function openAt(path: string) {
  render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>
  );
}

const runsTable = async () => screen.findByRole("table", { name: "Runs de MLflow" });
const rowOf = async (text: string) =>
  (await within(await runsTable()).findByText(text)).closest("tr")!;
const runNames = () =>
  within(screen.getByRole("table", { name: "Runs de MLflow" }))
    .getAllByRole("row")
    .slice(1)
    .map((row) => within(row).getAllByRole("cell")[1]!.textContent);

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("APP-04 tabla de runs", () => {
  it("muestra run_id, estado, procedencia, commit, los 7 parámetros y métricas de validación", async () => {
    serve();
    openAt("/ml/experiments");

    const row = await rowOf("mlp-512-256-adam");
    for (const text of [
      FINISHED,
      "FINISHED",
      "demo-v1.0.0",
      "md5:5d41402abc4b2a76b9719d911017c592",
      "9e8d7c6",
      "adam",
      "32",
      "50",
      "0.001",
      "224",
      "512,256",
      "0.2",
    ]) {
      expect(within(row).getAllByText(text).length).toBeGreaterThan(0);
    }
    expect(within(row).getByText("0.875")).toBeVisible();
  });

  it("un run sin commit lo muestra como faltante, no inventado", async () => {
    serve();
    openAt("/ml/experiments");

    const row = await rowOf("mlp-512-256-adam-img512");
    expect(within(row).getByTitle("Commit no registrado")).toBeInTheDocument();
  });

  it("abrir un run lleva exactamente al mismo run_id en MLflow", async () => {
    serve();
    openAt("/ml/experiments");

    const row = await rowOf("mlp-512-256-adam");
    expect(within(row).getByRole("link", { name: "Abrir mlp-512-256-adam en MLflow" })).toHaveAttribute(
      "href",
      `http://localhost:5000/#/experiments/1/runs/${FINISHED}`
    );
  });
});

describe("APP-04 valores no finitos de MLflow (revisión de #41)", () => {
  it("un run con val_loss NaN sigue en la tabla y la métrica se marca como no finita", async () => {
    serve();
    openAt("/ml/experiments");

    const row = await rowOf("mlp-512-256-adamw-bs16");
    expect(within(row).getByTitle("MLflow registró un valor no finito (NaN o ±inf)")).toHaveTextContent(
      "no finito"
    );
  });

  it("al ordenar, una métrica no finita queda al final en ambas direcciones", async () => {
    serve(() => {
      const runs = withExtraFinishedRun(runsExample());
      (runs.runs.find((r) => r.run_id === FINISHED)!.metrics as Record<string, number>).val_loss =
        0.4;
      (runs.runs.find((r) => r.run_id === EXTRA)!.metrics as Record<string, number>).val_loss = 0.3;
      return runs;
    });
    openAt("/ml/experiments");
    await screen.findByText("resnet18-sgd");
    const header = screen.getByRole("button", { name: /^val_loss/ });

    fireEvent.click(header);
    expect(runNames().slice(0, 2)).toEqual(["mlp-512-256-adam", "resnet18-sgd"]);
    expect(runNames().slice(2)).toContain("mlp-512-256-adamw-bs16");
    fireEvent.click(header);
    expect(runNames().slice(0, 2)).toEqual(["resnet18-sgd", "mlp-512-256-adam"]);
    expect(runNames().slice(2)).toContain("mlp-512-256-adamw-bs16");
  });

  it("la tabla de la curva marca la época no finita en vez de inventar un valor", async () => {
    serve();
    openAt(`/ml/experiments?runs=${FINISHED}`);

    const panel = await screen.findByRole("region", { name: "Curvas de mlp-512-256-adam" });
    const table = await within(panel).findByRole("table", { name: "val_loss por época" });
    const epoch3 = within(table).getAllByRole("row")[3]!;
    expect(within(epoch3).getByText("no finito")).toBeInTheDocument();
  });
});

describe("APP-04 orden por fecha (revisión de #41)", () => {
  function sameSecondRuns(secondStart: string) {
    return () => {
      const runs = withExtraFinishedRun(onlyFinished());
      runs.runs.find((r) => r.run_id === FINISHED)!.start_time = "2026-01-14T12:00:30Z";
      runs.runs.find((r) => r.run_id === EXTRA)!.start_time = secondStart;
      return runs;
    };
  }

  it("dentro del mismo segundo, el run con milisegundos posteriores va primero", async () => {
    serve(sameSecondRuns("2026-01-14T12:00:30.500000Z"));
    openAt("/ml/experiments");
    await screen.findByText("resnet18-sgd");

    const names = runNames();
    expect(names.indexOf("resnet18-sgd")).toBeLessThan(names.indexOf("mlp-512-256-adam"));
  });

  it("con la misma hora exacta, desempata por run_id como la API", async () => {
    serve(sameSecondRuns("2026-01-14T12:00:30Z"));
    openAt("/ml/experiments");
    await screen.findByText("resnet18-sgd");

    const names = runNames();
    // FINISHED (0a1b…) < EXTRA (abcd…)
    expect(names.indexOf("mlp-512-256-adam")).toBeLessThan(names.indexOf("resnet18-sgd"));
  });
});

describe("APP-04 ordenar y filtrar", () => {
  it("ordena por val_accuracy_top1 y alterna la dirección", async () => {
    serve(() => withExtraFinishedRun(runsExample()));
    openAt("/ml/experiments");
    await screen.findByText("resnet18-sgd");

    const header = screen.getByRole("button", { name: /val_accuracy_top1/ });
    fireEvent.click(header);
    expect(header.closest("th")).toHaveAttribute("aria-sort", "descending");
    const descending = runNames();
    fireEvent.click(header);
    expect(header.closest("th")).toHaveAttribute("aria-sort", "ascending");
    expect(runNames()).not.toEqual(descending);
  });

  it("por defecto muestra los runs más recientes primero", async () => {
    serve();
    openAt("/ml/experiments");
    await screen.findByText("mlp-512-256-adam");

    expect(screen.getByRole("button", { name: /Inicio/ }).closest("th")).toHaveAttribute(
      "aria-sort",
      "descending"
    );
  });

  it("filtra por estado", async () => {
    serve();
    openAt("/ml/experiments");
    await screen.findByText("mlp-512-256-adam");

    fireEvent.change(screen.getByLabelText("Estado"), { target: { value: "FAILED" } });

    expect(runNames()).toEqual(["mlp-512-256-adam-img512"]);
  });

  it("busca por nombre, run_id, dataset o commit", async () => {
    serve();
    openAt("/ml/experiments");
    await screen.findByText("mlp-512-256-adam");

    fireEvent.change(screen.getByLabelText("Buscar"), { target: { value: "adamw" } });
    expect(runNames()).toEqual(["mlp-512-256-adamw-bs16"]);

    fireEvent.change(screen.getByLabelText("Buscar"), { target: { value: "9e8d7c6" } });
    expect(runNames()).toEqual(["mlp-512-256-adam"]);
  });

  it("dice cuando ningún run coincide con el filtro", async () => {
    serve();
    openAt("/ml/experiments");
    await screen.findByText("mlp-512-256-adam");

    fireEvent.change(screen.getByLabelText("Buscar"), { target: { value: "no-existe" } });

    expect(screen.getByText("Ningún run coincide con el filtro.")).toBeVisible();
  });
});

describe("APP-04 comparar runs", () => {
  const select = async (name: string) => {
    fireEvent.click(within(await rowOf(name)).getByRole("checkbox", { name: `Comparar ${name}` }));
  };

  it("compara dos runs: diferencias de parámetros y curvas reales de ambos", async () => {
    const fetcher = serve();
    openAt("/ml/experiments");
    await select("mlp-512-256-adam");
    await select("mlp-512-256-adamw-bs16");

    const panel = await screen.findByRole("region", { name: "Comparación de runs" });
    const optimizer = within(panel).getByRole("rowheader", { name: "optimizer" }).closest("tr")!;
    expect(within(optimizer).getByText("Distinto")).toBeVisible();
    expect(within(optimizer).getByText("adamw")).toBeVisible();
    const imageSize = within(panel).getByRole("rowheader", { name: "image_size" }).closest("tr")!;
    expect(within(imageSize).queryByText("Distinto")).not.toBeInTheDocument();

    expect(callsTo(fetcher, `/api/ml/runs/${FINISHED}/curves`)).toHaveLength(1);
    expect(callsTo(fetcher, `/api/ml/runs/${RUNNING}/curves`)).toHaveLength(1);
    expect(await within(panel).findAllByRole("figure")).toHaveLength(4);
  });

  it("la vista de tabla trae los valores por época de las curvas", async () => {
    serve();
    openAt(`/ml/experiments?runs=${FINISHED}`);

    const panel = await screen.findByRole("region", { name: "Curvas de mlp-512-256-adam" });
    const table = await within(panel).findByRole("table", { name: "val_accuracy por época" });
    expect(within(table).getAllByRole("row")).toHaveLength(6);
    expect(within(table).getByText("0.875")).toBeInTheDocument();
  });

  it("al refrescar con ?runs= vuelve a mostrar la misma comparación", async () => {
    serve();
    openAt(`/ml/experiments?runs=${FINISHED},${RUNNING}`);

    expect(await screen.findByRole("region", { name: "Comparación de runs" })).toBeVisible();
    expect(
      within(await rowOf("mlp-512-256-adam")).getByRole("checkbox", { name: "Comparar mlp-512-256-adam" })
    ).toBeChecked();
  });

  it("compara como máximo 3 runs", async () => {
    serve(() => withExtraFinishedRun(runsExample()));
    openAt(`/ml/experiments?runs=${FINISHED},${RUNNING},${FAILED}`);

    const extra = await rowOf("resnet18-sgd");
    expect(within(extra).getByRole("checkbox", { name: "Comparar resnet18-sgd" })).toBeDisabled();
  });

  it("quitar un run de la comparación no cambia el color de los demás", async () => {
    serve();
    openAt(`/ml/experiments?runs=${FINISHED},${RUNNING},${FAILED}`);
    const panel = await screen.findByRole("region", { name: "Comparación de runs" });
    const slotOf = (name: string) =>
      within(panel).getByText(name, { selector: "[data-series-slot] *" }).closest("[data-series-slot]")!
        .getAttribute("data-series-slot");
    expect(slotOf("mlp-512-256-adam-img512")).toBe("3");

    fireEvent.click(
      within(await rowOf("mlp-512-256-adamw-bs16")).getByRole("checkbox", {
        name: "Comparar mlp-512-256-adamw-bs16",
      })
    );

    expect(slotOf("mlp-512-256-adam")).toBe("1");
    expect(slotOf("mlp-512-256-adam-img512")).toBe("3");
  });

  it("muestra el error si las curvas de un run no se pueden leer", async () => {
    serve(runsExample, () => json(loadMlExample("error"), 503));
    openAt(`/ml/experiments?runs=${FINISHED}`);

    const panel = await screen.findByRole("region", { name: "Curvas de mlp-512-256-adam" });
    expect(await within(panel).findByText("El servicio de entrenamiento no respondió.")).toBeVisible();
  });
});

describe("APP-04 runs en curso", () => {
  it("sondea MLflow mientras hay runs RUNNING, sin volver a 'Cargando'", async () => {
    const fetcher = serve();
    openAt("/ml/experiments");
    await screen.findByText("mlp-512-256-adam");

    await act(() => vi.advanceTimersByTimeAsync(RUNS_POLL_MS * 2));

    expect(callsTo(fetcher, RUNS_URL).length).toBeGreaterThanOrEqual(3);
    expect(screen.queryByText("Cargando…")).not.toBeInTheDocument();
  });

  it("no sondea cuando todos los runs terminaron", async () => {
    const fetcher = serve(onlyFinished);
    openAt("/ml/experiments");
    await screen.findByText("mlp-512-256-adam");

    await act(() => vi.advanceTimersByTimeAsync(RUNS_POLL_MS * 3));

    expect(callsTo(fetcher, RUNS_URL)).toHaveLength(1);
  });
});
