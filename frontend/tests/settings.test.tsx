import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { SettingsPage } from "../src/pipeline/pages/Settings";
import { pipelineSettingsSchema } from "../src/lib/api/settings";

import fixture from "./fixtures/settings.json";
const success = "Guardado. Se aplicará en la próxima ejecución del pipeline/release.";
function response(value: unknown, status = 200) { return new Response(JSON.stringify(value), { status }); }
function serve() {
  const fetcher = vi.fn(async (_url: string, init?: RequestInit) =>
    response(init?.method === "PUT" ? JSON.parse(String(init.body)) : fixture));
  vi.stubGlobal("fetch", fetcher);
  return fetcher;
}
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
async function open() {
  render(<SettingsPage />);
  await screen.findByRole("form", { name: "Política de calidad" });
}
it("loads contractual settings in two forms without active release or hidden fields", async () => {
  expect(pipelineSettingsSchema.safeParse(fixture).success).toBe(true);
  const fetcher = serve();
  await open();
  expect(screen.getAllByRole("form")).toHaveLength(2);
  expect(screen.getByLabelText("Mínimo de imágenes distintas por clase")).toHaveValue(300);
  expect(screen.getByLabelText("Seed reproducible")).toHaveValue(42);
  expect(screen.queryByText(/Dataset activo/i)).not.toBeInTheDocument();
  expect(screen.queryByText(/cross_split_leakage/)).not.toBeInTheDocument();
  expect(fetcher).toHaveBeenCalledWith("/api/settings", { signal: expect.any(AbortSignal) });
});
it.each([
  ["Mínimo de imágenes distintas por clase", "-1"],
  ["Similitud mínima pHash para detectar pares (0–1)", "1.1"],
  ["Ancho límite de objeto pequeño (px)", "0"],
  ["Dispersión espacial mínima (0–0.5)", "0.6"],
])("validates quality %s", async (label, value) => {
  const fetcher = serve(); await open();
  fireEvent.change(screen.getByLabelText(label), { target: { value } });
  fireEvent.submit(screen.getByRole("form", { name: "Política de calidad" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Revisa");
  expect(fetcher).toHaveBeenCalledTimes(1);
});
it("validates ratio sum before PUT", async () => {
  const fetcher = serve(); await open();
  fireEvent.change(screen.getByLabelText("Train"), { target: { value: "0.5" } });
  fireEvent.submit(screen.getByRole("form", { name: "Configuración de splits" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("100%");
  expect(fetcher).toHaveBeenCalledTimes(1);
});
it.each([
  ["Política de calidad", "/api/settings/quality", "Mínimo de imágenes distintas por clase", "301", "quality"],
  ["Configuración de splits", "/api/settings/splits", "Seed reproducible", "71", "splits"],
] as const)("saves only %s via its endpoint", async (form, endpoint, label, value, section) => {
  const fetcher = serve(); await open();
  fireEvent.change(screen.getByLabelText(label), { target: { value } });
  fireEvent.submit(screen.getByRole("form", { name: form }));
  expect(await screen.findByRole("status")).toHaveTextContent(success);
  const [, init] = fetcher.mock.calls[1]!;
  expect(fetcher.mock.calls[1]![0]).toBe(endpoint);
  expect(init?.method).toBe("PUT");
  const body = JSON.parse(String(init?.body));
  expect(Object.keys(body)).toEqual(Object.keys(fixture[section]));
  expect(fetcher.mock.calls.map(([url]) => url)).toEqual(["/api/settings", endpoint]);
});
it("shows backend save error without claiming success", async () => {
  const fetcher = serve(); await open();
  fetcher.mockResolvedValueOnce(response({ error: "No se pudo guardar." }, 500));
  fireEvent.click(screen.getByRole("button", { name: "Guardar calidad" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("No se pudo guardar.");
  expect(screen.queryByText(success)).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Guardar calidad" })).not.toBeDisabled();
});
it("shows loading, disables saving form, then succeeds", async () => {
  const fetcher = serve();
  render(<SettingsPage />);
  expect(screen.getByRole("status")).toHaveTextContent("Cargando");
  await screen.findByRole("button", { name: "Guardar splits" });
  let finish!: (value: Response) => void;
  fetcher.mockImplementationOnce(() => new Promise(resolve => { finish = resolve; }));
  fireEvent.click(screen.getByRole("button", { name: "Guardar splits" }));
  expect(screen.getByRole("button", { name: "Guardando splits…" })).toBeDisabled();
  expect(screen.getByLabelText("Train")).toBeDisabled();
  expect(screen.getByRole("button", { name: "Guardar calidad" })).not.toBeDisabled();
  finish(response(fixture.splits));
  await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent(success));
});
it("shows GET failure and supports retry", async () => {
  const fetcher = serve();
  fetcher.mockResolvedValueOnce(response({ error: "Configuración no disponible." }, 500));
  render(<SettingsPage />);
  expect(await screen.findByRole("alert")).toHaveTextContent("Configuración no disponible.");
  fireEvent.click(screen.getByRole("button", { name: "Reintentar" }));
  await screen.findByRole("form", { name: "Política de calidad" });
});

it.each([
  ["Política de calidad", "Mínimo de imágenes distintas por clase", "301"],
  ["Política de calidad", "Acción: Mínimo de imágenes distintas por clase", "warn"],
  ["Política de calidad", "Ancho límite de objeto pequeño (px)", "33"],
  ["Política de calidad", "Alto límite de objeto pequeño (px)", "33"],
  ["Configuración de splits", "Train", "0.6"],
  ["Configuración de splits", "Validation", "0.2"],
  ["Configuración de splits", "Test", "0.2"],
  ["Configuración de splits", "Seed reproducible", "73"],
])("clears only %s success when editing %s", async (formName, label, value) => {
  serve(); await open();
  const qualityForm = screen.getByRole("form", { name: "Política de calidad" });
  const splitsForm = screen.getByRole("form", { name: "Configuración de splits" });
  fireEvent.submit(qualityForm);
  await within(qualityForm).findByText(success);
  fireEvent.submit(splitsForm);
  await within(splitsForm).findByText(success);
  const edited = screen.getByRole("form", { name: formName });
  const other = edited === qualityForm ? splitsForm : qualityForm;
  fireEvent.change(screen.getByLabelText(label), { target: { value } });
  expect(within(edited).queryByText(success)).not.toBeInTheDocument();
  expect(within(other).getByText(success)).toBeInTheDocument();
});

it("failed splits PUT keeps the edited draft and displays the error", async () => {
  const fetcher = serve(); await open();
  fireEvent.change(screen.getByLabelText("Seed reproducible"), { target: { value: "73" } });
  fetcher.mockResolvedValueOnce(response({ error: "No se pudo guardar splits." }, 500));
  fireEvent.click(screen.getByRole("button", { name: "Guardar splits" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("No se pudo guardar splits.");
  expect(screen.getByLabelText("Seed reproducible")).toHaveValue(73);
  expect(screen.getByRole("button", { name: "Guardar splits" })).not.toBeDisabled();
  expect(screen.queryByText(success)).not.toBeInTheDocument();
});

it.each([
  ["Política de calidad", "Mínimo de imágenes distintas por clase", "quality"],
  ["Configuración de splits", "Seed reproducible", "splits"],
] as const)("adopts validated server values after saving %s", async (formName, label, section) => {
  const fetcher = serve(); await open();
  fireEvent.change(screen.getByLabelText(label), { target: { value: "73" } });
  const returned = structuredClone(fixture);
  returned.quality.min_images_per_class.threshold = 74;
  returned.splits.seed = 74;
  fetcher.mockResolvedValueOnce(response(returned[section]));
  fireEvent.submit(screen.getByRole("form", { name: formName }));
  await screen.findByText(success);
  expect(screen.getByLabelText(label)).toHaveValue(74);
  const sent = JSON.parse(String(fetcher.mock.calls[1]![1]?.body));
  expect(section === "quality" ? sent.min_images_per_class.threshold : sent.seed).toBe(73);
});

it.each([
  ["Política de calidad", "/api/settings/quality"],
  ["Configuración de splits", "/api/settings/splits"],
])("saving %s preserves the other unsaved draft", async (formName, endpoint) => {
  const fetcher = serve(); await open();
  fireEvent.change(screen.getByLabelText("Mínimo de imágenes distintas por clase"), { target: { value: "301" } });
  fireEvent.change(screen.getByLabelText("Seed reproducible"), { target: { value: "73" } });
  fireEvent.submit(screen.getByRole("form", { name: formName }));
  await screen.findByText(success);
  expect(screen.getByLabelText("Mínimo de imágenes distintas por clase")).toHaveValue(301);
  expect(screen.getByLabelText("Seed reproducible")).toHaveValue(73);
  expect(fetcher.mock.calls.map(([url]) => url)).toEqual(["/api/settings", endpoint]);
});
