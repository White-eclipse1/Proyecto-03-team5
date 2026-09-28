import type { ReactNode } from "react";
import { GlobalNav } from "./GlobalNav";

/**
 * Shell compartido por Tablero, Buscar y Subir fotografías: mismo nav en
 * las tres pantallas, siempre en el mismo lugar. Ver GlobalNav para por qué
 * Annotate queda fuera de este shell.
 *
 * `sidebarExtra` permite que una pantalla agregue contenido propio (p. ej.
 * los filtros de Búsqueda) debajo del nav global, en el mismo sidebar en
 * vez de uno separado.
 */
export function AppLayout({
  children,
  sidebarExtra,
}: {
  children: ReactNode;
  sidebarExtra?: ReactNode;
}) {
  return (
    <div className="flex min-h-screen flex-col bg-canvas lg:flex-row">
      <GlobalNav>{sidebarExtra}</GlobalNav>
      <div className="flex min-w-0 flex-1 flex-col">{children}</div>
    </div>
  );
}
