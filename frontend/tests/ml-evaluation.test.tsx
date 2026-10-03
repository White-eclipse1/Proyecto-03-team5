import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "../src/App";
import { loadMlExample } from "./mlCorpus";

// APP-05 (issue #22): la pantalla Evaluation lee `GET /api/ml/evaluation`
// (EvaluationOverview). TDD de los tres estados: candidate_not_frozen,
// candidate_frozen y evaluated. Los datos vienen del corpus compartido con Python.

type Prediction = {
  image_id: number;
  annotation_id: number;
  true_class: string;
  predicted_class: string;
  probabilities: Record<string, number>;
};
type Evaluation = Record<string, unknown> & {
  class_names: string[];
  confusion_matrix: number[][];
  predictions: Prediction[];
};
type Overview = {
  schema_version: "1.0";
  state: string;
  candidate: Record<string, unknown> | null;
  evaluation: Evaluation | null;
  problem: string | null;
};

const OVERVIEW_URL = "/api/ml/evaluation";
const RUN_ID = "0a1b2c3d4e5f60718293a4b5c6d7e8f9";

function json(body: unknown, status = 200) {
  return Promise.resolve(
    new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    })
  );
}

function evaluated(): Overview {
  return structuredClone(loadMlExample("evaluation_overview")) as Overview;
}

function frozen(problem: string | null = null): Overview {
  return { ...evaluated(), state: "candidate_frozen", evaluation: null, problem };
}

function notFrozen(problem: string | null = null): Overview {
  return { ...evaluated(), state: "candidate_not_frozen", candidate: null, evaluation: null, problem };
}

/**
 * Evaluación coherente con el contrato a partir de una matriz 2×2 (dog, cat). Las
 * métricas guardadas van redondeadas a 3 decimales, como las deja ML-09.
 */
function withMatrix(overview: Overview, matrix: number[][]): Overview {
  const names = ["dog", "cat"];
  const round = (value: number) => Math.round(value * 1000) / 1000;
  const total = matrix.flat().reduce((a, b) => a + b, 0);
  const perClass = names.map((name, index) => {
    const hits = matrix[index]![index]!;
    const support = matrix[index]!.reduce((a, b) => a + b, 0);
    const predicted = matrix.reduce((a, row) => a + row[index]!, 0);
    const precision = hits / predicted;
    const recall = hits / support;
    const f1 = (2 * precision * recall) / (precision + recall);
    return { class_name: name, precision, recall, f1, support };
  });
  const predictions: Prediction[] = [];
  names.forEach((trueClass, row) => {
    names.forEach((predictedClass, column) => {
      for (let i = 0; i < matrix[row]![column]!; i++) {
        const id = predictions.length + 1;
        const other = names.find((name) => name !== predictedClass)!;
        predictions.push({
          image_id: 5000 + id,
          annotation_id: 9000 + id,
          true_class: trueClass,
          predicted_class: predictedClass,
          probabilities: { [predictedClass]: 0.9, [other]: 0.1 },
        });
      }
    });
  });
  overview.evaluation = {
    ...overview.evaluation!,
    class_names: names,
    confusion_matrix: matrix,
    per_class: perClass.map((entry) => ({
      ...entry,
      precision: round(entry.precision),
      recall: round(entry.recall),
      f1: round(entry.f1),
    })),
    metrics: {
      accuracy_top1: round((matrix[0]![0]! + matrix[1]![1]!) / total),
      f1_macro: round((perClass[0]!.f1 + perClass[1]!.f1) / 2),
    },
    predictions,
  };
  return overview;
}

function serve(overview: unknown, status = 200) {
  const fetcher = vi.fn((url: string) =>
    url === OVERVIEW_URL ? json(overview, status) : json({}, 404)
  );
  vi.stubGlobal("fetch", fetcher);
  return fetcher;
}

function openEvaluation() {
  render(
    <MemoryRouter initialEntries={["/ml/evaluation"]}>
      <App />
    </MemoryRouter>
  );
}

/** Nada del test a la vista: ni métricas, ni matriz, ni ejemplos, ni exportación. */
function expectNoTestResults() {
  expect(screen.queryByText(/accuracy/i)).not.toBeInTheDocument();
  expect(screen.queryByText(/F1 macro/i)).not.toBeInTheDocument();
  expect(screen.queryByRole("table", { name: /matriz de confusión/i })).not.toBeInTheDocument();
  expect(screen.queryByRole("img")).not.toBeInTheDocument();
  expect(screen.queryByRole("link", { name: /predicciones/i })).not.toBeInTheDocument();
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("APP-05 Evaluation: estados", () => {
  it("candidate_not_frozen: avisa que falta congelar y no muestra nada del test", async () => {
    serve(notFrozen());
    openEvaluation();

    expect(await screen.findByText(/todavía no hay candidato congelado/i)).toBeVisible();
    expectNoTestResults();
  });

  it("Agent Test: si el backend explica por qué oculta un test, se ve el motivo y ningún número", async () => {
    serve(notFrozen("Hay una evaluación de test sin candidato congelado; no se muestra."));
    openEvaluation();

    expect(await screen.findByText(/evaluación de test sin candidato congelado/)).toBeVisible();
    expectNoTestResults();
  });

  it("Agent Test: una respuesta que revela el test antes del congelamiento no se muestra", async () => {
    serve({ ...notFrozen(), evaluation: evaluated().evaluation });
    openEvaluation();

    expect(await screen.findByText("La respuesta no cumple el contrato esperado.")).toBeVisible();
    expectNoTestResults();
  });

  it("candidate_frozen: muestra el candidato y espera la evaluación final", async () => {
    serve(frozen());
    openEvaluation();

    expect(await screen.findByText(/falta la evaluación final de test/i)).toBeVisible();
    const candidate = screen.getByRole("region", { name: /candidato congelado/i });
    expect(within(candidate).getByText(RUN_ID)).toBeInTheDocument();
    expect(within(candidate).getByText("mlp-512-256-adam")).toBeInTheDocument();
    expect(within(candidate).getByText(`runs:/${RUN_ID}/checkpoints/best.pt`)).toBeInTheDocument();
    expect(within(candidate).getByText("demo-v1.0.0")).toBeInTheDocument();
    expect(
      within(candidate).getByText("md5:5d41402abc4b2a76b9719d911017c592")
    ).toBeInTheDocument();
    expect(within(candidate).getByText(/84d6c88b/)).toBeInTheDocument();
    expectNoTestResults();
  });

  it("candidate_frozen: la métrica de selección se presenta como validation, nunca como test", async () => {
    serve(frozen());
    openEvaluation();

    const candidate = await screen.findByRole("region", { name: /candidato congelado/i });
    expect(within(candidate).getByText(/selección \(validation\)/i)).toBeInTheDocument();
    expect(within(candidate).getByText("best_val_loss = 0.241")).toBeInTheDocument();
    expect(within(candidate).queryByText(/test/i)).not.toBeInTheDocument();
  });

  it.each(["train_loss", "test_accuracy"])(
    "un candidato elegido con %s no se muestra ni revela el test",
    async (metric) => {
      const overview = evaluated();
      overview.candidate = { ...overview.candidate, selection_metric: metric };
      serve(overview);
      openEvaluation();

      expect(
        await screen.findByText("La respuesta no cumple el contrato esperado.")
      ).toBeVisible();
      expectNoTestResults();
      expect(screen.queryByText(metric, { exact: false })).not.toBeInTheDocument();
    }
  );

  it("candidate_frozen: si la evaluación no cuadra, explica por qué se oculta", async () => {
    serve(frozen("Hay 2 evaluaciones de test; el test se evalúa una sola vez."));
    openEvaluation();

    expect(await screen.findByText(/el test se evalúa una sola vez/)).toBeVisible();
    expectNoTestResults();
  });
});

describe("APP-05 Evaluation: evaluated", () => {
  it("muestra el candidato y las métricas finales de test", async () => {
    serve(evaluated());
    openEvaluation();

    const metrics = await screen.findByRole("region", { name: /evaluación final en test/i });
    expect(within(metrics).getByText("eval-0002")).toBeInTheDocument();
    expect(within(metrics).getByText("0.8000")).toBeInTheDocument();
    expect(within(metrics).getByText("8 de 10 recortes de test")).toBeInTheDocument();
    expect(within(metrics).getByText("0.7917")).toBeInTheDocument();
    const candidate = screen.getByRole("region", { name: /candidato congelado/i });
    expect(within(candidate).getByText(RUN_ID)).toBeInTheDocument();
  });

  it("compara con 0.85 desde la matriz, sin redondear", async () => {
    // 1699 / 2000 = 0.8495: accuracy_top1 guardado (3 decimales) diría 0.85.
    serve(withMatrix(evaluated(), [[900, 100], [201, 799]]));
    openEvaluation();

    const metrics = await screen.findByRole("region", { name: /evaluación final en test/i });
    expect(within(metrics).getByText("0.8495")).toBeInTheDocument();
    expect(within(metrics).getByText("1699 de 2000 recortes de test")).toBeInTheDocument();
    expect(within(metrics).getByText(/meta 0\.85: no alcanzada/i)).toBeInTheDocument();
  });

  it("exactamente 0.85 alcanza la meta", async () => {
    serve(withMatrix(evaluated(), [[900, 100], [200, 800]]));
    openEvaluation();

    const metrics = await screen.findByRole("region", { name: /evaluación final en test/i });
    expect(within(metrics).getByText(/meta 0\.85: alcanzada/i)).toBeInTheDocument();
  });

  it("si la evaluación no trae la versión, muestra la registrada con el mismo run y checkpoint", async () => {
    // ML-09 evalúa antes de que OPS-06 registre la versión: su JSON trae model_name null.
    const overview = evaluated();
    overview.evaluation = { ...overview.evaluation!, model_name: null, model_version: null };
    vi.stubGlobal(
      "fetch",
      vi.fn((url: string) => {
        if (url === OVERVIEW_URL) return json(overview);
        if (url === "/api/ml/models") return json(loadMlExample("models"));
        return json({}, 404);
      })
    );
    openEvaluation();

    const metrics = await screen.findByRole("region", { name: /evaluación final en test/i });
    const link = await within(metrics).findByRole("link", { name: "dog-cat-resnet18 v1.0.0" });
    expect(link).toHaveAttribute("href", "/ml/models");
    expect(within(metrics).queryByText("Sin registrar")).not.toBeInTheDocument();
  });

  it("sin una versión registrada del mismo checkpoint dice Sin registrar", async () => {
    const overview = evaluated();
    overview.evaluation = { ...overview.evaluation!, model_name: null, model_version: null };
    const models = structuredClone(loadMlExample("models")) as {
      models: {
        checkpoint_sha256: string;
        files: { name: string; sha256: string }[];
        publication: unknown;
      }[];
    };
    const other = "1".repeat(64);
    models.models[0]!.checkpoint_sha256 = other;
    models.models[0]!.files[0]!.sha256 = other;
    models.models[0]!.publication = {
      status: "not_published",
      bucket: null,
      region: null,
      published_at: null,
      objects: [],
      problem: null,
    };
    vi.stubGlobal(
      "fetch",
      vi.fn((url: string) => {
        if (url === OVERVIEW_URL) return json(overview);
        if (url === "/api/ml/models") return json(models);
        return json({}, 404);
      })
    );
    openEvaluation();

    const metrics = await screen.findByRole("region", { name: /evaluación final en test/i });
    expect(await within(metrics).findByText("Sin registrar")).toBeInTheDocument();
    expect(within(metrics).queryByRole("link", { name: /dog-cat-resnet18/ })).not.toBeInTheDocument();
  });

  it("muestra el baseline de clase mayoritaria sobre el mismo test", async () => {
    serve(evaluated());
    openEvaluation();

    const metrics = await screen.findByRole("region", { name: /evaluación final en test/i });
    expect(within(metrics).getByText("0.6000")).toBeInTheDocument();
    expect(within(metrics).getByText(/siempre «dog»/)).toBeInTheDocument();
  });

  it("muestra precision, recall, f1 y support por clase", async () => {
    serve(evaluated());
    openEvaluation();

    const table = await screen.findByRole("table", { name: /métricas por clase/i });
    const dog = within(table).getByRole("row", { name: /dog/ });
    expect(within(dog).getAllByText("0.8333")).toHaveLength(3);
    expect(within(dog).getByText("6")).toBeInTheDocument();
    const cat = within(table).getByRole("row", { name: /cat/ });
    expect(within(cat).getAllByText("0.7500")).toHaveLength(3);
    expect(within(cat).getByText("4")).toBeInTheDocument();
  });

  it("la matriz de confusión tiene filas reales, columnas predichas y suma el total del test", async () => {
    serve(evaluated());
    openEvaluation();

    const table = await screen.findByRole("table", { name: /matriz de confusión/i });
    const headers = within(table)
      .getAllByRole("columnheader")
      .map((th) => th.textContent);
    expect(headers).toEqual(["Real \\ Predicha", "dog", "cat", "Total real"]);
    const dog = within(table).getByRole("row", { name: /^dog/ });
    expect(within(dog).getAllByRole("cell").map((td) => td.textContent)).toEqual(["5", "1", "6"]);
    const cat = within(table).getByRole("row", { name: /^cat/ });
    expect(within(cat).getAllByRole("cell").map((td) => td.textContent)).toEqual(["1", "3", "4"]);
    const totals = within(table).getByRole("row", { name: /total predicho/i });
    expect(within(totals).getAllByRole("cell").map((td) => td.textContent)).toEqual([
      "6",
      "4",
      "10",
    ]);
    expect(screen.getByText("La matriz suma 10 = 10 predicciones de test.")).toBeVisible();
  });

  it("reporta la confusión más frecuente y avisa del recall bajo aunque se alcance 0.85", async () => {
    serve(withMatrix(evaluated(), [[1000, 0], [300, 700]]));
    openEvaluation();

    const errors = await screen.findByRole("region", { name: /interpretación de errores/i });
    expect(within(errors).getByText(/cat → dog: 300 recortes/)).toBeInTheDocument();
    expect(within(errors).getByText(/recall de cat es 0\.7000/i)).toBeInTheDocument();
    expect(within(errors).getByText(/el accuracy global oculta/i)).toBeInTheDocument();
  });

  it("explica que el accuracy no oculta recall bajo y nombra la clase más débil (OPS-10, rúbrica 4.4)", async () => {
    // Matriz real de ML-09: dog 43/43, cat 25/28 (recall 0.8929 ≥ 0.85).
    serve(withMatrix(evaluated(), [[43, 0], [3, 25]]));
    openEvaluation();

    const errors = await screen.findByRole("region", { name: /interpretación de errores/i });
    expect(
      within(errors).getByText(
        "Ninguna clase tiene recall por debajo de 0.85: el accuracy global no oculta un recall bajo."
      )
    ).toBeInTheDocument();
    expect(within(errors).getByText("La clase más débil es cat: recall 0.8929 (25 de 28).")).toBeInTheDocument();
    expect(within(errors).queryByText(/oculta recall bajo: alcanza/i)).not.toBeInTheDocument();
  });

  it("muestra ejemplos incorrectos y correctos con su recorte, clase real y predicha", async () => {
    serve(evaluated());
    openEvaluation();

    const examples = await screen.findByRole("region", { name: /ejemplos del test/i });
    const wrong = within(examples).getAllByRole("figure");
    expect(wrong).toHaveLength(2);
    const first = wrong[0]!;
    expect(within(first).getByRole("img")).toHaveAttribute(
      "src",
      expect.stringMatching(/^\/api\/ml\/crops\/img\d+-ann\d+$/)
    );
    expect(within(first).getByText(/real: (dog|cat)/i)).toBeInTheDocument();
    expect(within(first).getByText(/predicha: (dog|cat)/i)).toBeInTheDocument();

    fireEvent.click(within(examples).getByRole("button", { name: /correctos \(8\)/i }));
    expect(within(examples).getAllByRole("figure")).toHaveLength(8);
  });

  it("si el recorte no está disponible lo dice en vez de una imagen rota", async () => {
    serve(evaluated());
    openEvaluation();

    const examples = await screen.findByRole("region", { name: /ejemplos del test/i });
    const figure = within(examples).getAllByRole("figure")[0]!;
    fireEvent.error(within(figure).getByRole("img"));
    expect(within(figure).getByText("Recorte no disponible")).toBeInTheDocument();
    expect(within(figure).queryByRole("img")).not.toBeInTheDocument();
  });

  it("los ejemplos se navegan por páginas", async () => {
    serve(withMatrix(evaluated(), [[900, 100], [200, 800]]));
    openEvaluation();

    const examples = await screen.findByRole("region", { name: /ejemplos del test/i });
    expect(within(examples).getAllByRole("figure")).toHaveLength(12);
    fireEvent.click(within(examples).getByRole("button", { name: /mostrar más/i }));
    expect(within(examples).getAllByRole("figure")).toHaveLength(24);
  });

  it("permite exportar las predicciones por recorte", async () => {
    serve(evaluated());
    openEvaluation();

    const link = await screen.findByRole("link", { name: /descargar predicciones \(csv\)/i });
    expect(link).toHaveAttribute("href", "/api/ml/evaluation/predictions.csv");
  });
});
