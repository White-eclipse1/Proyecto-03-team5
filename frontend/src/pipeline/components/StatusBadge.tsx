const TONE_CLASSES = {
  done: "bg-status-done-soft text-status-done",
  pending: "bg-status-pending-soft text-status-pending",
  blocked: "bg-ink text-white",
} as const;

type Tone = keyof typeof TONE_CLASSES;

/** Traduce los literales de los contratos JSON (P2-12) a un tono visual. */
function toneFor(value: string): Tone {
  if (value === "passed" || value === "done") return "done";
  if (value === "failed" || value === "fail") return "blocked";
  return "pending";
}

export function StatusBadge({ label }: Readonly<{ label: string }>) {
  return (
    <span
      className={`inline-flex items-center rounded-full px-2.5 py-1 text-xs font-medium capitalize ${TONE_CLASSES[toneFor(label)]}`}
    >
      {label}
    </span>
  );
}
