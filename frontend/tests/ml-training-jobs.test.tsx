import "@testing-library/jest-dom/vitest";
import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "../src/App";
import { JOBS_POLL_MS } from "../src/ml/pages/Training";
import { loadMlExample } from "./mlCorpus";

// APP-03: la pantalla Training sigue jobs que corren fuera del request HTTP. El
// estado vive en el servidor (cola en MariaDB), así que refrescar la página vuelve
// a mostrar lo mismo; la página solo sondea mientras haya jobs activos.

type Jobs = { schema_version: "1.0"; jobs: Record<string, unknown>[] };

const JOBS_URL = "/api/ml/training/jobs";
const LOGS_URL = "/api/ml/training/jobs/job-0003/logs";

function json(body: unknown, status = 200) {
  return Promise.resolve(
    new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    })
  );
}

function jobsExample(): Jobs {
  return structuredClone(loadMlExample("training_jobs")) as Jobs;
}

function onlyTerminalJobs(): Jobs {
  const jobs = jobsExample();
  jobs.jobs = jobs.jobs.filter((job) => job.status === "succeeded" || job.status === "failed");
  return jobs;
}

function serve(routes: Record<string, () => Promise<Response>>) {
  const fetcher = vi.fn((url: string) => {
    const route = Object.keys(routes).find((key) => url === key || url.startsWith(`${key}?`));
    return route ? routes[route]!() : json({}, 404);
  });
  vi.stubGlobal("fetch", fetcher);
  return fetcher;
}

const callsTo = (fetcher: ReturnType<typeof serve>, url: string) =>
  fetcher.mock.calls.filter(([called]) => called === url || called.startsWith(`${url}?`));

function openAt(path: string) {
  render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>
  );
}

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("APP-03 progreso de los jobs", () => {
  it("muestra la época y la métrica de un job running", async () => {
    serve({ [JOBS_URL]: () => json(jobsExample()) });
    openAt("/ml/training");

    const row = (await screen.findByText("job-0003")).closest("tr")!;
    const bar = within(row).getByRole("progressbar", { name: "Progreso de job-0003" });
    expect(bar).toHaveAttribute("aria-valuenow", "12");
    expect(bar).toHaveAttribute("aria-valuemax", "50");
    expect(within(row).getByText("Época 12/50")).toBeVisible();
    expect(within(row).getByText("val_accuracy 0.812")).toBeVisible();
  });

  it("un job queued no inventa progreso", async () => {
    serve({ [JOBS_URL]: () => json(jobsExample()) });
    openAt("/ml/training");

    const row = (await screen.findByText("job-0004")).closest("tr")!;
    expect(within(row).getByText("queued")).toBeVisible();
    expect(within(row).queryByRole("progressbar")).not.toBeInTheDocument();
  });

  it("muestra código y mensaje del error de un job failed", async () => {
    serve({ [JOBS_URL]: () => json(jobsExample()) });
    openAt("/ml/training");

    const row = (await screen.findByText("job-0002")).closest("tr")!;
    expect(within(row).getByText("out_of_memory")).toBeVisible();
    expect(within(row).getByText("El entrenamiento se quedó sin memoria de GPU.")).toBeVisible();
  });
});

describe("APP-03 seguimiento sin recargar", () => {
  it("sondea los jobs mientras alguno sigue activo, sin volver a 'Cargando'", async () => {
    const fetcher = serve({ [JOBS_URL]: () => json(jobsExample()) });
    openAt("/ml/training");
    await screen.findByText("job-0003");

    await act(() => vi.advanceTimersByTimeAsync(JOBS_POLL_MS * 2));

    expect(callsTo(fetcher, JOBS_URL).length).toBeGreaterThanOrEqual(3);
    expect(screen.queryByText("Cargando…")).not.toBeInTheDocument();
    expect(screen.getByText("job-0003")).toBeVisible();
  });

  it("refleja el nuevo estado que reporta el servidor", async () => {
    const running = jobsExample();
    const advanced = jobsExample();
    const job = advanced.jobs.find((j) => j.job_id === "job-0003")!;
    (job.progress as { epoch: number }).epoch = 13;
    let responses = 0;
    serve({ [JOBS_URL]: () => json(responses++ === 0 ? running : advanced) });
    openAt("/ml/training");
    await screen.findByText("Época 12/50");

    await act(() => vi.advanceTimersByTimeAsync(JOBS_POLL_MS));

    expect(await screen.findByText("Época 13/50")).toBeVisible();
  });

  it("no sondea cuando todos los jobs terminaron", async () => {
    const fetcher = serve({ [JOBS_URL]: () => json(onlyTerminalJobs()) });
    openAt("/ml/training");
    await screen.findByText("job-0001");

    await act(() => vi.advanceTimersByTimeAsync(JOBS_POLL_MS * 3));

    expect(callsTo(fetcher, JOBS_URL)).toHaveLength(1);
  });
});

describe("APP-03 logs del job", () => {
  const logs = () => loadMlExample("training_logs");

  it("abre los logs de un job y los deja en la URL", async () => {
    const fetcher = serve({ [JOBS_URL]: () => json(jobsExample()), [LOGS_URL]: () => json(logs()) });
    openAt("/ml/training");

    const row = (await screen.findByText("job-0003")).closest("tr")!;
    fireEvent.click(within(row).getByRole("button", { name: "Ver logs" }));

    const panel = await screen.findByRole("region", { name: "Logs de job-0003" });
    expect(await within(panel).findByText("val_loss no mejoró en 2 épocas (patience 5).")).toBeVisible();
    expect(within(panel).getAllByRole("listitem")).toHaveLength(4);
    expect(callsTo(fetcher, LOGS_URL)[0]![0]).toBe(`${LOGS_URL}?after=0`);
  });

  it("al refrescar con ?job= vuelve a mostrar los logs del mismo job", async () => {
    serve({ [JOBS_URL]: () => json(jobsExample()), [LOGS_URL]: () => json(logs()) });
    openAt("/ml/training?job=job-0003");

    const panel = await screen.findByRole("region", { name: "Logs de job-0003" });
    expect(await within(panel).findByText(/Época 12\/50: train_loss=0.412/)).toBeVisible();
  });

  it("pide solo las líneas nuevas mientras el job corre", async () => {
    const more = {
      schema_version: "1.0",
      job_id: "job-0003",
      entries: [
        {
          seq: 5,
          timestamp: "2026-01-15T09:43:10Z",
          level: "info",
          message: "Época 13/50: train_loss=0.401",
        },
      ],
    };
    let calls = 0;
    const fetcher = serve({
      [JOBS_URL]: () => json(jobsExample()),
      [LOGS_URL]: () => json(calls++ === 0 ? logs() : more),
    });
    openAt("/ml/training?job=job-0003");
    const panel = await screen.findByRole("region", { name: "Logs de job-0003" });
    await within(panel).findByText("val_loss no mejoró en 2 épocas (patience 5).");

    await act(() => vi.advanceTimersByTimeAsync(JOBS_POLL_MS));

    expect(await within(panel).findByText("Época 13/50: train_loss=0.401")).toBeVisible();
    expect(within(panel).getAllByRole("listitem")).toHaveLength(5);
    expect(callsTo(fetcher, LOGS_URL)[1]![0]).toBe(`${LOGS_URL}?after=4`);
  });

  it("indica cuando el job todavía no tiene logs", async () => {
    serve({
      [JOBS_URL]: () => json(jobsExample()),
      "/api/ml/training/jobs/job-0004/logs": () =>
        json({ schema_version: "1.0", job_id: "job-0004", entries: [] }),
    });
    openAt("/ml/training?job=job-0004");

    const panel = await screen.findByRole("region", { name: "Logs de job-0004" });
    expect(await within(panel).findByText("Este job todavía no tiene logs.")).toBeVisible();
  });

  it("muestra el error si los logs no se pueden leer", async () => {
    serve({
      [JOBS_URL]: () => json(jobsExample()),
      [LOGS_URL]: () => json(loadMlExample("error"), 503),
    });
    openAt("/ml/training?job=job-0003");

    const panel = await screen.findByRole("region", { name: "Logs de job-0003" });
    expect(await within(panel).findByText("El servicio de entrenamiento no respondió.")).toBeVisible();
  });
});
