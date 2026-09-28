import { type ButtonHTMLAttributes, forwardRef } from "react";

export type ButtonVariant = "primary" | "secondary" | "ghost" | "destructive" | "icon";
export type ButtonSize = "sm" | "md";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  isLoading?: boolean;
}

const BASE =
  "inline-flex items-center justify-center gap-1.5 whitespace-nowrap rounded-lg font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-lilac/40 disabled:cursor-not-allowed disabled:opacity-50";

const VARIANT_CLASSES: Record<ButtonVariant, string> = {
  primary: "bg-accent-lilac text-white shadow-sm hover:bg-accent-lilac/90 disabled:shadow-none",
  secondary: "border border-border bg-surface text-ink hover:bg-sidebar",
  ghost: "text-ink-muted hover:bg-sidebar hover:text-ink",
  destructive: "bg-red-600 text-white shadow-sm hover:bg-red-500 disabled:shadow-none",
  icon: "text-ink-faint hover:bg-sidebar hover:text-ink",
};

const SIZE_CLASSES: Record<Exclude<ButtonVariant, "icon">, Record<ButtonSize, string>> = {
  primary: { sm: "px-3 py-1.5 text-xs", md: "px-4 py-2 text-sm" },
  secondary: { sm: "px-3 py-1.5 text-xs", md: "px-4 py-2 text-sm" },
  ghost: { sm: "px-2.5 py-1.5 text-xs", md: "px-3 py-2 text-sm" },
  destructive: { sm: "px-3 py-1.5 text-xs", md: "px-4 py-2 text-sm" },
};

const ICON_SIZE_CLASSES: Record<ButtonSize, string> = {
  sm: "h-8 w-8",
  md: "h-9 w-9",
};

function Spinner() {
  return (
    <svg className="h-3.5 w-3.5 animate-spin" viewBox="0 0 24 24" fill="none" aria-hidden>
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 0 1 8-8v4a4 4 0 0 0-4 4H4Z" />
    </svg>
  );
}

/**
 * Botón base del design system. Variantes: primary (acción principal),
 * secondary (acción alterna), ghost (baja énfasis), destructive (borrar/
 * irreversible), icon (solo ícono, cuadrado).
 */
export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    { variant = "secondary", size = "md", isLoading, disabled, className = "", children, ...props },
    ref
  ) => {
    const sizeClass = variant === "icon" ? ICON_SIZE_CLASSES[size] : SIZE_CLASSES[variant][size];

    return (
      <button
        ref={ref}
        disabled={disabled || isLoading}
        className={`${BASE} ${VARIANT_CLASSES[variant]} ${sizeClass} ${className}`}
        {...props}
      >
        {isLoading && <Spinner />}
        {children}
      </button>
    );
  }
);
Button.displayName = "Button";
