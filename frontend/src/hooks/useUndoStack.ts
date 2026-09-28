import { useCallback, useRef, useState } from "react";

export interface UndoableAction {
  /** Descripción corta, útil para depurar o mostrar en un toast. */
  label: string;
  undo: () => Promise<void> | void;
}

/**
 * Pila de acciones deshacer-únicamente (no hay redo por requisito). Cada
 * acción push()eada ya conoce cómo revertirse a sí misma.
 */
export function useUndoStack() {
  const stackRef = useRef<UndoableAction[]>([]);
  const [canUndo, setCanUndo] = useState(false);

  const push = useCallback((action: UndoableAction) => {
    stackRef.current.push(action);
    setCanUndo(true);
  }, []);

  const undo = useCallback(async (): Promise<string | null> => {
    const action = stackRef.current.pop();
    setCanUndo(stackRef.current.length > 0);
    if (!action) return null;
    await action.undo();
    return action.label;
  }, []);

  const clear = useCallback(() => {
    stackRef.current = [];
    setCanUndo(false);
  }, []);

  return { push, undo, clear, canUndo };
}
