type Accent = "lilac" | "mint" | "peach" | "blue";

type StatCardProps = {
  label: string;
  value: number;
  accent: Accent;
};

const dotColor: Record<Accent, string> = {
  lilac: "bg-accent-lilac",
  mint: "bg-accent-mint",
  peach: "bg-status-pending",
  blue: "bg-status-progress",
};

export function StatCard({ label, value, accent }: StatCardProps) {
  return (
    <div className="rounded-2xl border border-border bg-surface p-5 shadow-card">
      <div className="flex items-center gap-2">
        <span className={`h-2 w-2 rounded-full ${dotColor[accent]}`} aria-hidden="true" />
        <p className="text-sm text-ink-muted">{label}</p>
      </div>
      <p className="mt-3 text-3xl font-semibold tracking-tight text-ink">
        {value.toLocaleString("es")}
      </p>
    </div>
  );
}
