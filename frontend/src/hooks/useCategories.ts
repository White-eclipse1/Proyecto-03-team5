import { useEffect, useState } from "react";
import { getCategories } from "../lib/api/categories";
import type { Category } from "../types/schemas";

interface UseCategoriesResult {
  categories: Category[];
  isLoading: boolean;
  error: string | null;
  retry: () => void;
}

export function useCategories(): UseCategoriesResult {
  const [categories, setCategories] = useState<Category[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);

  // biome-ignore lint/correctness/useExhaustiveDependencies: attempt solo se usa para forzar el reintento.
  useEffect(() => {
    let cancelled = false;
    setIsLoading(true);
    setError(null);

    getCategories()
      .then((result) => {
        if (!cancelled) setCategories(result);
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "No se pudieron cargar las categorías.");
        }
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [attempt]);

  return { categories, isLoading, error, retry: () => setAttempt((n) => n + 1) };
}
