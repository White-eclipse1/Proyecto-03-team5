import type { ReactNode } from "react";

/** Tabla simple compartida por las pantallas de modelos. */
export function DataTable({
  caption,
  headers,
  children,
}: Readonly<{ caption: string; headers: string[]; children: ReactNode }>) {
  return (
    <div className="overflow-x-auto rounded-2xl border border-border bg-surface shadow-card">
      <table className="w-full text-left text-sm">
        <caption className="sr-only">{caption}</caption>
        <thead className="border-b border-border text-xs uppercase text-ink-muted">
          <tr>
            {headers.map((header) => (
              <th key={header} scope="col" className="whitespace-nowrap px-4 py-3 font-medium">
                {header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-border">{children}</tbody>
      </table>
    </div>
  );
}

export function Cell({
  children,
  mono = false,
}: Readonly<{ children: ReactNode; mono?: boolean }>) {
  return (
    <td className={`whitespace-nowrap px-4 py-3 align-top ${mono ? "font-mono text-xs" : ""}`}>
      {children}
    </td>
  );
}

/** Texto para campos opcionales del contrato (`null` todavía no existe). */
export function Missing() {
  return <span className="text-ink-faint">—</span>;
}
