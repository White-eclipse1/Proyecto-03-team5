import type { ReactNode } from "react";
import { PageHeader } from "@/pipeline/components/PageHeader";

/** Contenedor común de las pantallas de modelos (mismo layout que el pipeline). */
export function MlPage({
  title,
  subtitle,
  children,
}: Readonly<{ title: string; subtitle: string; children: ReactNode }>) {
  return (
    <main className="flex-1 px-6 py-6 lg:px-10 lg:py-8">
      <div className="mx-auto flex max-w-6xl flex-col gap-6">
        <PageHeader title={title} subtitle={subtitle} />
        {children}
      </div>
    </main>
  );
}
