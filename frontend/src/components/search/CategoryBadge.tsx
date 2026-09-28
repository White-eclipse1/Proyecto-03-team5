interface CategoryBadgeProps {
  name: string;
  color: string;
}

export function CategoryBadge({ name, color }: CategoryBadgeProps): JSX.Element {
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-sidebar px-2 py-0.5 text-[10px] font-medium text-ink-muted">
      <span
        className="h-1.5 w-1.5 shrink-0 rounded-full"
        style={{ backgroundColor: color }}
        aria-hidden="true"
      />
      <span className="truncate">{name}</span>
    </span>
  );
}
