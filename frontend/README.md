# Search / Gallery — Frontend (Fase 3)

Pantalla de búsqueda/galería del portal de anotación de imágenes. React + TypeScript
(`strict: true`), Tailwind CSS, `react-router` y Zod para validar toda respuesta del backend.

Este directorio corresponde a `repo/front` dentro del monorepo (junto a `repo/back`,
la Fase 2 ya aprobada).

## Setup

```bash
cp .env.example .env   # opcional: define VITE_API_BASE_URL si el backend no corre en :3000
npm install
npm run dev
```

`npm run dev` levanta Vite en `:5173` con un proxy `/api → http://localhost:3000`
(ver `vite.config.ts`), así que el backend de la Fase 2 debe estar corriendo
localmente (`npm run dev` dentro de `repo/back`, según su propio README de handoff).

```bash
npm run typecheck   # tsc --noEmit
npm run build        # build de producción
```

## ⚠️ Nota importante: el contrato de `GET /images/search` es un supuesto

El backend de la Fase 2 (ver `Fase_2_-_Backend_Data_Layer` — documentación técnica)
**no expone todavía** `GET /images/search` ni `GET /categories`. Este frontend fue
construido asumiendo el contrato descrito en el prompt de la Fase 3 (`data/schemas.ts`
refleja exactamente esa forma). Si el backend termina implementando una forma distinta
de respuesta, solo hay que ajustar:

- `src/api/schemas.ts` — los schemas Zod (fuente de verdad de los tipos)
- `src/lib/searchFilters.ts` — el mapeo de filtros → query params

El resto de la app (hooks, componentes) no necesita cambios porque todo el tipado
sale de `z.infer` de esos schemas.

## Decisiones de arquitectura

- **Cliente HTTP**: `src/api/client.ts` es un wrapper mínimo sobre `fetch`, sin
  librerías externas de data-fetching (según lo pedido). Cada llamada valida la
  respuesta con Zod antes de devolverla; si el parseo falla, se lanza `ApiParseError`
  y la UI muestra un estado de error legible, nunca un crash silencioso.
- **Filtros en la URL**: `src/lib/searchFilters.ts` serializa/deserializa el estado
  de búsqueda (`q`, `categories`, `status`, `dateFrom`, `dateTo`, `page`, `pageSize`)
  contra el querystring de la propia app (`useSearchParams` de react-router), de
  modo que la búsqueda es compartible y recargable. Ese mismo estado se traduce a
  los query params que espera el backend (`filtersToBackendQuery`) — nunca se
  filtra ni pagina en memoria del lado del cliente.
- **Colores de categoría**: siempre vienen de `GET /categories` (campo `color`,
  hex `#RRGGBB`). No hay ningún color de categoría hardcodeado ni generado
  aleatoriamente en el frontend.
- **Imágenes**: `thumbnailUrl` se usa tal cual la entrega el backend. El frontend
  nunca construye URLs de MinIO a mano ni asume que el bucket es público.
- **Ruta de anotación**: cada card navega a `/annotate/:imageId` (`AnnotatePage`
  es un placeholder — el canvas de bounding boxes se implementa en otra parte de
  la Fase 3, pero la ruta ya existe y no rompe la navegación).

## Estructura

```
src/
├── api/
│   ├── client.ts       # wrapper de fetch + validación Zod + errores tipados
│   └── schemas.ts       # única fuente de verdad de los tipos (z.infer)
├── hooks/
│   ├── useCategories.ts
│   └── useImageSearch.ts
├── lib/
│   └── searchFilters.ts # estado de filtros ↔ URL ↔ query del backend
├── components/
│   ├── ui/               # Skeleton, EmptyState, ErrorState (genéricos)
│   └── search/           # Sidebar, FiltersPanel, SearchBar, FilterChips,
│                          # ResultsGrid, ImageCard, Pagination, StatusBadge
├── pages/
│   ├── SearchPage.tsx    # orquesta todo lo anterior
│   └── AnnotatePage.tsx  # placeholder de /annotate/:imageId
├── App.tsx               # rutas
└── main.tsx              # entry point
```

## Checklist de la rúbrica

- [x] `strict: true` + `noUncheckedIndexedAccess`, sin `any` injustificado, sin `@ts-ignore`
- [x] Toda respuesta HTTP se valida con Zod; los tipos de los componentes vienen de `z.infer`
- [x] La búsqueda con `AND` se envía como query param `q`; se asume resuelta en SQL del backend
- [x] Filtros combinables entre sí y reflejados en la URL (compartible/recargable)
- [x] Paginación real vía `page`/`pageSize` del backend (no se pagina en memoria)
- [x] Loading (skeleton), error (con reintentar) y empty state cubiertos
- [x] Responsive: por debajo de `sm` el sidebar se apila arriba del contenido (no lo
      comprime); el grid pasa de 2 columnas en mobile a 5 en desktop.
