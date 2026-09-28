import { type ReactNode, useEffect } from "react";

interface ModalProps {
  title: string;
  description?: ReactNode;
  onClose: () => void;
  children?: ReactNode;
  /** Ancho máximo del panel. Por defecto suficiente para diálogos de confirmación. */
  maxWidthClassName?: string;
}

/**
 * Modal base del design system: backdrop + panel centrado + cierre con Esc.
 * El contenido de las acciones (footer) se pasa como children para que cada
 * caso de uso controle sus propios botones (ver LeaveConfirmModal).
 */
export function Modal({
  title,
  description,
  onClose,
  children,
  maxWidthClassName = "max-w-sm",
}: ModalProps) {
  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  return (
    // biome-ignore lint/a11y/useKeyWithClickEvents: cerrar con teclado ya lo cubre el listener de Escape de arriba.
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="modal-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-ink/40 px-4 backdrop-blur-[2px]"
      onClick={onClose}
    >
      {/* biome-ignore lint/a11y/noStaticElementInteractions: solo evita que el click se propague al backdrop, no es una acción. */}
      {/* biome-ignore lint/a11y/useKeyWithClickEvents: no es una acción de teclado, solo frena la propagación del click. */}
      <div
        onClick={(e) => e.stopPropagation()}
        className={`w-full ${maxWidthClassName} animate-popover-in rounded-2xl border border-border bg-surface p-5 shadow-popover`}
      >
        <h2 id="modal-title" className="text-sm font-semibold text-ink">
          {title}
        </h2>
        {description && <div className="mt-1.5 text-sm text-ink-muted">{description}</div>}
        {children && <div className="mt-5">{children}</div>}
      </div>
    </div>
  );
}
