import { useId, useState } from "react";
import { API_BASE_URL } from "@/lib/api/client";
import { Cell, DataTable } from "../components/DataTable";
import { MlPage } from "../components/MlPage";
import { MlResourceBoundary } from "../components/MlResourceBoundary";
import { cropUrl, ML_ENDPOINTS, useEvaluationOverview } from "../dataSource";
import { type EvaluationSummary, summarizeEvaluation } from "../evaluationSummary";
import type { Evaluation, EvaluationOverview, FrozenCandidate } from "../schemas";

/** Cuatro decimales: 0.8495 no debe leerse como 0.85. */
const metric = (value: number) => value.toFixed(4);
export const EXAMPLES_PAGE_SIZE = 12;

function Section({ title, children }: Readonly<{ title: string; children: React.ReactNode }>) {
  const id = useId();
  return (
    <section
      aria-labelledby={id}
      className="rounded-2xl border border-border bg-surface p-6 shadow-card"
    >
      <h2 id={id} className="text-base font-semibold text-ink">
        {title}
      </h2>
      <div className="mt-4">{children}</div>
    </section>
  );
}

function Field({ label, children }: Readonly<{ label: string; children: React.ReactNode }>) {
  return (
    <div>
      <dt className="text-xs uppercase text-ink-muted">{label}</dt>
      <dd className="mt-1 break-all text-sm text-ink">{children}</dd>
    </div>
  );
}

function Notice({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <div className="rounded-2xl border border-dashed border-border-strong bg-surface px-6 py-10 text-center text-sm text-ink-muted">
      {children}
    </div>
  );
}

function Problem({ text }: Readonly<{ text: string | null }>) {
  if (text === null) return null;
  return (
    <p role="alert" className="rounded-xl bg-status-pending-soft px-4 py-3 text-sm text-ink">
      <span className="font-medium text-status-pending">No se muestran resultados: </span>
      {text}
    </p>
  );
}

/** Lo que ML-08 eligió con validation y congeló. Aquí no aparece nada del test. */
function CandidateCard({ candidate }: Readonly<{ candidate: FrozenCandidate }>) {
  return (
    <Section title="Candidato congelado (ML-08)">
      <dl className="grid gap-4 sm:grid-cols-2">
        <Field label="Run">{candidate.run_name}</Field>
        <Field label="run_id">
          <span className="font-mono text-xs">{candidate.run_id}</span>
        </Field>
        <Field label="Checkpoint">
          <span className="font-mono text-xs">{candidate.checkpoint}</span>
        </Field>
        <Field label="sha256 del checkpoint">
          <span className="font-mono text-xs">{candidate.checkpoint_sha256}</span>
        </Field>
        <Field label="Dataset release">
          <span className="font-mono text-xs">{candidate.dataset_version}</span>
        </Field>
        <Field label="Manifest hash">
          <span className="font-mono text-xs">{candidate.manifest_hash}</span>
        </Field>
        <Field label="Métrica de selección (validation)">
          <span className="font-mono text-xs">
            {candidate.selection_metric} = {candidate.selection_value}
          </span>
        </Field>
        <Field label="Congelado">
          <span className="font-mono text-xs">{candidate.frozen_at}</span>
        </Field>
      </dl>
    </Section>
  );
}

function Stat({
  label,
  value,
  children,
}: Readonly<{ label: string; value: string; children?: React.ReactNode }>) {
  return (
    <div className="rounded-xl border border-border px-4 py-3">
      <p className="text-xs uppercase text-ink-muted">{label}</p>
      <p className="mt-1 font-mono text-2xl text-ink">{value}</p>
      {children && <div className="mt-1 text-xs text-ink-muted">{children}</div>}
    </div>
  );
}

function FinalMetrics({
  evaluation,
  summary,
}: Readonly<{ evaluation: Evaluation; summary: EvaluationSummary }>) {
  return (
    <Section title="Evaluación final en test (ML-09)">
      <dl className="mb-4 grid gap-4 sm:grid-cols-3">
        <Field label="Evaluación">
          <span className="font-mono text-xs">{evaluation.evaluation_id}</span>
        </Field>
        <Field label="Modelo">
          {evaluation.model_name === null ? (
            <span className="text-ink-muted">Sin registrar</span>
          ) : (
            `${evaluation.model_name} v${evaluation.model_version}`
          )}
        </Field>
        <Field label="Evaluada">
          <span className="font-mono text-xs">{evaluation.created_at}</span>
        </Field>
      </dl>
      <div className="grid gap-4 sm:grid-cols-3">
        <Stat label="Accuracy top-1 (test)" value={metric(summary.accuracy)}>
          <p>
            {summary.correct} de {summary.total} recortes de test
          </p>
          <p className={summary.meetsTarget ? "text-status-done" : "text-status-pending"}>
            Meta 0.85: {summary.meetsTarget ? "alcanzada" : "no alcanzada"}
          </p>
        </Stat>
        <Stat label="F1 macro (test)" value={metric(summary.f1Macro)} />
        <Stat label="Baseline de clase mayoritaria" value={metric(summary.baseline.accuracy)}>
          Predecir siempre «{summary.baseline.className}» en el mismo test
        </Stat>
      </div>
    </Section>
  );
}

function PerClassTable({ summary }: Readonly<{ summary: EvaluationSummary }>) {
  return (
    <DataTable
      caption="Métricas por clase (test)"
      headers={["Clase", "Precision", "Recall", "F1", "Support"]}
    >
      {summary.perClass.map((entry) => (
        <tr key={entry.className}>
          <Cell>{entry.className}</Cell>
          <Cell mono>{metric(entry.precision)}</Cell>
          <Cell mono>{metric(entry.recall)}</Cell>
          <Cell mono>{metric(entry.f1)}</Cell>
          <Cell mono>{entry.support}</Cell>
        </tr>
      ))}
    </DataTable>
  );
}

/** Tinte de un solo tono según el conteo; el número siempre va escrito. */
function cellTint(count: number, max: number): string {
  const alpha = max === 0 ? 0 : 0.06 + 0.34 * (count / max);
  return `rgba(124, 111, 234, ${alpha.toFixed(3)})`;
}

function ConfusionMatrix({
  evaluation,
  summary,
}: Readonly<{ evaluation: Evaluation; summary: EvaluationSummary }>) {
  const names = evaluation.class_names;
  const max = Math.max(...evaluation.confusion_matrix.flat());
  const header = "whitespace-nowrap px-4 py-3 font-medium";
  const number = "px-4 py-3 text-right font-mono text-xs";
  return (
    <div className="flex flex-col gap-2">
      <div className="overflow-x-auto rounded-2xl border border-border bg-surface shadow-card">
        <table className="text-sm">
          <caption className="sr-only">Matriz de confusión (test)</caption>
          <thead className="border-b border-border text-xs text-ink-muted">
            <tr>
              <th scope="col" className={`${header} text-left`}>
                Real \ Predicha
              </th>
              {names.map((name) => (
                <th key={name} scope="col" className={`${header} text-right`}>
                  {name}
                </th>
              ))}
              <th scope="col" className={`${header} text-right`}>
                Total real
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {names.map((name, row) => (
              <tr key={name}>
                <th scope="row" className={`${header} text-left text-ink`}>
                  {name}
                </th>
                {evaluation.confusion_matrix[row]?.map((count, column) => (
                  <td
                    key={names[column]}
                    className={`${number} ${row === column ? "font-semibold" : ""}`}
                    style={{ backgroundColor: cellTint(count, max) }}
                  >
                    {count}
                  </td>
                ))}
                <td className={number}>{summary.rowTotals[row]}</td>
              </tr>
            ))}
            <tr>
              <th scope="row" className={`${header} text-left text-ink-muted`}>
                Total predicho
              </th>
              {summary.columnTotals.map((count, column) => (
                <td key={names[column]} className={number}>
                  {count}
                </td>
              ))}
              <td className={`${number} font-semibold`}>{summary.total}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p className="text-xs text-ink-muted">
        La matriz suma {summary.total} = {evaluation.predictions.length} predicciones de test.
      </p>
    </div>
  );
}

function ErrorAnalysis({ summary }: Readonly<{ summary: EvaluationSummary }>) {
  const low = summary.perClass.filter((entry) => entry.lowRecall);
  const confused = summary.mostConfused;
  return (
    <Section title="Interpretación de errores">
      <ul className="flex list-disc flex-col gap-2 pl-5 text-sm text-ink">
        <li>
          {confused === null
            ? "Sin confusiones entre clases."
            : `Confusión más frecuente (real → predicha): ${confused.trueClass} → ${confused.predictedClass}: ${confused.count} recortes.`}
        </li>
        {low.map((entry) => (
          <li key={entry.className}>
            El recall de {entry.className} es {metric(entry.recall)}, por debajo de 0.85.
          </li>
        ))}
        {summary.meetsTarget && low.length > 0 && (
          <li className="font-medium text-status-pending">
            El accuracy global oculta recall bajo: alcanza 0.85, pero no en todas las clases.
          </li>
        )}
      </ul>
    </Section>
  );
}

type Prediction = Evaluation["predictions"][number];

function CropImage({ prediction }: Readonly<{ prediction: Prediction }>) {
  const [missing, setMissing] = useState(false);
  const cropId = `img${prediction.image_id}-ann${prediction.annotation_id}`;
  if (missing) {
    return (
      <div className="flex aspect-square items-center justify-center rounded-lg bg-sidebar text-xs text-ink-muted">
        Recorte no disponible
      </div>
    );
  }
  return (
    <img
      src={cropUrl(prediction.image_id, prediction.annotation_id)}
      alt={`Recorte ${cropId}`}
      loading="lazy"
      onError={() => setMissing(true)}
      className="aspect-square w-full rounded-lg bg-sidebar object-contain"
    />
  );
}

function ExampleCard({ prediction }: Readonly<{ prediction: Prediction }>) {
  const confidence = prediction.probabilities[prediction.predicted_class];
  return (
    <figure className="flex flex-col gap-2 rounded-xl border border-border p-2">
      <CropImage prediction={prediction} />
      <figcaption className="flex flex-col gap-0.5 text-xs">
        <span className="text-ink">Real: {prediction.true_class}</span>
        <span className="text-ink">Predicha: {prediction.predicted_class}</span>
        {confidence !== undefined && (
          <span className="font-mono text-ink-muted">p = {confidence.toFixed(3)}</span>
        )}
        <span className="font-mono text-ink-faint">
          img{prediction.image_id}-ann{prediction.annotation_id}
        </span>
      </figcaption>
    </figure>
  );
}

function Examples({ evaluation }: Readonly<{ evaluation: Evaluation }>) {
  const [kind, setKind] = useState<"wrong" | "right">("wrong");
  const [shown, setShown] = useState(EXAMPLES_PAGE_SIZE);
  const wrong = evaluation.predictions.filter((p) => p.true_class !== p.predicted_class);
  const right = evaluation.predictions.filter((p) => p.true_class === p.predicted_class);
  const list = kind === "wrong" ? wrong : right;
  const choose = (next: "wrong" | "right") => {
    setKind(next);
    setShown(EXAMPLES_PAGE_SIZE);
  };
  const tab = (active: boolean) =>
    `rounded-lg px-3 py-1.5 text-sm ${active ? "bg-accent-lilac-soft text-ink" : "text-ink-muted"}`;

  return (
    <Section title="Ejemplos del test">
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <button
          type="button"
          aria-pressed={kind === "wrong"}
          className={tab(kind === "wrong")}
          onClick={() => choose("wrong")}
        >
          Incorrectos ({wrong.length})
        </button>
        <button
          type="button"
          aria-pressed={kind === "right"}
          className={tab(kind === "right")}
          onClick={() => choose("right")}
        >
          Correctos ({right.length})
        </button>
        <a
          href={`${API_BASE_URL}${ML_ENDPOINTS.predictionsCsv}`}
          download
          className="ml-auto text-sm font-medium text-accent-lilac underline-offset-2 hover:underline"
        >
          Descargar predicciones (CSV)
        </a>
      </div>
      {list.length === 0 ? (
        <p className="text-sm text-ink-muted">No hay ejemplos de este tipo.</p>
      ) : (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-6">
          {list.slice(0, shown).map((prediction) => (
            <ExampleCard key={prediction.annotation_id} prediction={prediction} />
          ))}
        </div>
      )}
      {shown < list.length && (
        <button
          type="button"
          className="mt-4 rounded-lg border border-border px-3 py-1.5 text-sm text-ink"
          onClick={() => setShown((count) => count + EXAMPLES_PAGE_SIZE)}
        >
          Mostrar más ({list.length - shown} restantes)
        </button>
      )}
    </Section>
  );
}

function Overview({ overview }: Readonly<{ overview: EvaluationOverview }>) {
  const { candidate, evaluation } = overview;
  if (candidate === null) {
    return (
      <div className="flex flex-col gap-4">
        <Notice>
          Todavía no hay candidato congelado. ML-08 lo elige solo con validation; hasta que lo
          congele no se muestra ningún resultado del test.
        </Notice>
        <Problem text={overview.problem} />
      </div>
    );
  }
  if (evaluation === null) {
    return (
      <div className="flex flex-col gap-4">
        <CandidateCard candidate={candidate} />
        <Notice>
          Falta la evaluación final de test del candidato (ML-09). El test se evalúa una sola vez,
          después de congelar.
        </Notice>
        <Problem text={overview.problem} />
      </div>
    );
  }
  const summary = summarizeEvaluation(evaluation);
  return (
    <div className="flex flex-col gap-6">
      <CandidateCard candidate={candidate} />
      <FinalMetrics evaluation={evaluation} summary={summary} />
      <PerClassTable summary={summary} />
      <ConfusionMatrix evaluation={evaluation} summary={summary} />
      <ErrorAnalysis summary={summary} />
      <Examples evaluation={evaluation} />
    </div>
  );
}

export function EvaluationPage() {
  const overview = useEvaluationOverview();

  return (
    <MlPage
      title="Evaluation"
      subtitle="Evaluación final del candidato congelado: el test solo se muestra después de elegirlo con validation."
    >
      <MlResourceBoundary state={overview} emptyMessage="">
        {(response) => <Overview overview={response} />}
      </MlResourceBoundary>
    </MlPage>
  );
}
