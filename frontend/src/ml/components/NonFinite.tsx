export const NON_FINITE_TITLE = "MLflow registró un valor no finito (NaN o ±inf)";

/**
 * APP-04: un valor que MLflow registró como NaN o ±inf (llega como null). Se muestra
 * así en vez de esconder el run o inventar un número.
 */
export function NonFinite() {
  return (
    <span title={NON_FINITE_TITLE} className="text-status-pending">
      no finito
    </span>
  );
}
