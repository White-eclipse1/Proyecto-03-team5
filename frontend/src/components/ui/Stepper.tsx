interface StepperProps {
  steps: string[];
  currentIndex: number;
}

/**
 * Indicador de progreso para flujos guiados de varios pasos (p. ej.
 * Subir fotografías → Identificar objetos).
 */
export function Stepper({ steps, currentIndex }: StepperProps) {
  return (
    <ol className="flex items-center gap-2">
      {steps.map((step, index) => {
        const isDone = index < currentIndex;
        const isCurrent = index === currentIndex;
        return (
          <li key={step} className="flex items-center gap-2">
            {index > 0 && <span className="h-px w-6 shrink-0 bg-border" aria-hidden />}
            <span className="flex items-center gap-1.5">
              <span
                className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[11px] font-semibold ${
                  isDone
                    ? "bg-accent-mint text-white"
                    : isCurrent
                      ? "bg-accent-lilac text-white"
                      : "bg-border text-ink-faint"
                }`}
              >
                {isDone ? "✓" : index + 1}
              </span>
              <span className={`text-xs font-medium ${isCurrent ? "text-ink" : "text-ink-faint"}`}>
                {step}
              </span>
            </span>
          </li>
        );
      })}
    </ol>
  );
}
