import type { Config } from "tailwindcss";

// Tokens de diseño — paleta neutra con dos acentos (lila / menta),
// pensada para el dominio: anotación de imágenes / datasets COCO.
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
      },
      colors: {
        canvas: "#FAFAF9",
        surface: "#FFFFFF",
        sidebar: "#F5F4F1",
        border: {
          DEFAULT: "#E7E5E1",
          strong: "#D8D5D0",
        },
        ink: {
          DEFAULT: "#1C1B1A",
          muted: "#8A8782",
          faint: "#B5B2AC",
        },
        accent: {
          lilac: "#7C6FEA",
          "lilac-soft": "#EFECFD",
          mint: "#2FAF87",
          "mint-soft": "#E3F5EE",
        },
        status: {
          pending: "#B08900",
          "pending-soft": "#FBF2D9",
          progress: "#4A6FE0",
          "progress-soft": "#E7ECFC",
          done: "#2FAF87",
          "done-soft": "#E3F5EE",
        },
      },
      borderRadius: {
        xl: "0.875rem",
        "2xl": "1.25rem",
      },
      boxShadow: {
        card: "0 1px 2px rgba(28,27,26,0.04), 0 1px 8px rgba(28,27,26,0.04)",
        popover: "0 8px 24px rgba(28,27,26,0.10)",
      },
      keyframes: {
        "popover-in": {
          from: { opacity: "0", transform: "scale(0.96) translateY(-2px)" },
          to: { opacity: "1", transform: "scale(1) translateY(0)" },
        },
      },
      animation: {
        "popover-in": "popover-in 0.12s ease-out",
      },
    },
  },
  plugins: [],
} satisfies Config;
