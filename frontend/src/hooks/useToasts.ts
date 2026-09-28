import { useCallback, useRef, useState } from "react";

export interface Toast {
  id: number;
  variant: "error" | "success" | "info";
  message: string;
}

let toastCounter = 0;

/**
 * Manejo simple de toasts en memoria de componente. Se monta una vez por
 * pantalla (Upload, Canvas de anotación) y se pasa `showToast` hacia abajo.
 */
export function useToasts(autoDismissMs = 4000) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const timers = useRef<Map<number, ReturnType<typeof setTimeout>>>(new Map());

  const dismissToast = useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
    const timer = timers.current.get(id);
    if (timer) {
      clearTimeout(timer);
      timers.current.delete(id);
    }
  }, []);

  const showToast = useCallback(
    (message: string, variant: Toast["variant"] = "info") => {
      const id = ++toastCounter;
      setToasts((prev) => [...prev, { id, variant, message }]);
      const timer = setTimeout(() => dismissToast(id), autoDismissMs);
      timers.current.set(id, timer);
    },
    [autoDismissMs, dismissToast]
  );

  return { toasts, showToast, dismissToast };
}
