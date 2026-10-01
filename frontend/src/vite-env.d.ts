/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string;
  /** APP-04: UI de MLflow para "Abrir en MLflow" (por defecto http://localhost:5000). */
  readonly VITE_MLFLOW_UI_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
