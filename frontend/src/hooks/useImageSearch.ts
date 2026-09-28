import { useEffect, useState } from "react";
import { getJson } from "@/api/client";
import { type ImageSearchItem, imageSearchResponseSchema, type Pagination } from "@/api/schemas";
import { filtersToBackendQuery, type SearchFilters } from "@/lib/searchFilters";

interface UseImageSearchResult {
  images: ImageSearchItem[];
  pagination: Pagination | null;
  isLoading: boolean;
  error: string | null;
  retry: () => void;
}

/**
 * Ejecuta la búsqueda contra GET /images/search con los filtros actuales.
 * El filtrado, el operador AND del texto libre y la paginación se resuelven
 * en el backend (SQL) — este hook nunca filtra ni pagina en memoria, solo
 * refleja page/pageSize en la query enviada.
 */
export function useImageSearch(filters: SearchFilters): UseImageSearchResult {
  const [images, setImages] = useState<ImageSearchItem[]>([]);
  const [pagination, setPagination] = useState<Pagination | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);

  const queryKey = filtersToBackendQuery(filters);

  // queryKey ya representa todo el estado relevante de `filters`; attempt solo fuerza el reintento.
  // biome-ignore lint/correctness/useExhaustiveDependencies: ver comentario anterior.
  useEffect(() => {
    const controller = new AbortController();

    setIsLoading(true);
    setError(null);

    getJson(`/images/search?${queryKey}`, imageSearchResponseSchema, {
      signal: controller.signal,
    })
      .then((data) => {
        setImages(data.data);
        setPagination(data.pagination);
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        setImages([]);
        setPagination(null);
        setError(err instanceof Error ? err.message : "No se pudo completar la búsqueda.");
      })
      .finally(() => {
        if (!controller.signal.aborted) setIsLoading(false);
      });

    return () => controller.abort();
  }, [queryKey, attempt]);

  return {
    images,
    pagination,
    isLoading,
    error,
    retry: () => setAttempt((n) => n + 1),
  };
}
