import { dashboardSummarySchema } from "../types/dashboard";
import { useValidatedFetch } from "./useValidatedFetch";

// Assumed endpoint per the Fase 3 spec — not implemented in the backend yet.
const DASHBOARD_SUMMARY_URL = "/dashboard/summary";

export function useDashboardSummary() {
  return useValidatedFetch(DASHBOARD_SUMMARY_URL, dashboardSummarySchema);
}
