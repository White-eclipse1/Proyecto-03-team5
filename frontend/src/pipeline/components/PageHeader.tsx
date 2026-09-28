import type { ReactNode } from "react";

export function PageHeader({
  title,
  subtitle,
  children,
}: Readonly<{
  title: string;
  subtitle: string;
  children?: ReactNode;
}>) {
  return (
    <header className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
      <div>
        <h1 className="text-xl font-semibold text-ink">{title}</h1>
        <p className="mt-1 text-sm text-ink-muted">{subtitle}</p>
      </div>
      {children && <div className="flex items-center gap-2">{children}</div>}
    </header>
  );
}
