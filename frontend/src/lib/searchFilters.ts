import type { ImageStatus } from "@/api/schemas";

export const DEFAULT_PAGE_SIZE = 15;

export interface SearchFilters {
  q: string;
  categoryIds: number[];
  statuses: ImageStatus[];
  dateFrom: string; // ISO date (yyyy-mm-dd) o ""
  dateTo: string; // ISO date (yyyy-mm-dd) o ""
  page: number;
  pageSize: number;
}

export const EMPTY_FILTERS: SearchFilters = {
  q: "",
  categoryIds: [],
  statuses: [],
  dateFrom: "",
  dateTo: "",
  page: 1,
  pageSize: DEFAULT_PAGE_SIZE,
};

const VALID_STATUSES: ImageStatus[] = ["pending", "in_progress", "completed"];

/** Lee los filtros actuales desde un URLSearchParams. Nunca lanza. */
export function filtersFromSearchParams(params: URLSearchParams): SearchFilters {
  const categoryIds = (params.get("categories") ?? "")
    .split(",")
    .map((s) => Number(s))
    .filter((n) => Number.isFinite(n) && n > 0);

  const statuses = (params.get("status") ?? "")
    .split(",")
    .filter((s): s is ImageStatus => (VALID_STATUSES as string[]).includes(s));

  const page = Number(params.get("page") ?? "1");
  const pageSize = Number(params.get("pageSize") ?? String(DEFAULT_PAGE_SIZE));

  return {
    q: params.get("q") ?? "",
    categoryIds,
    statuses,
    dateFrom: params.get("dateFrom") ?? "",
    dateTo: params.get("dateTo") ?? "",
    page: Number.isFinite(page) && page > 0 ? page : 1,
    pageSize: Number.isFinite(pageSize) && pageSize > 0 ? pageSize : DEFAULT_PAGE_SIZE,
  };
}

/** Escribe los filtros a un URLSearchParams, omitiendo los que están vacíos. */
export function filtersToSearchParams(filters: SearchFilters): URLSearchParams {
  const params = new URLSearchParams();

  if (filters.q.trim() !== "") params.set("q", filters.q.trim());
  if (filters.categoryIds.length > 0) params.set("categories", filters.categoryIds.join(","));
  if (filters.statuses.length > 0) params.set("status", filters.statuses.join(","));
  if (filters.dateFrom !== "") params.set("dateFrom", filters.dateFrom);
  if (filters.dateTo !== "") params.set("dateTo", filters.dateTo);
  if (filters.page !== 1) params.set("page", String(filters.page));
  if (filters.pageSize !== DEFAULT_PAGE_SIZE) params.set("pageSize", String(filters.pageSize));

  return params;
}

/** Construye el querystring que se envía al backend en GET /images/search. */
export function filtersToBackendQuery(filters: SearchFilters): string {
  const params = new URLSearchParams();

  if (filters.q.trim() !== "") params.set("q", filters.q.trim());
  if (filters.categoryIds.length > 0) params.set("categories", filters.categoryIds.join(","));
  if (filters.statuses.length > 0) params.set("status", filters.statuses.join(","));
  if (filters.dateFrom !== "") params.set("dateFrom", filters.dateFrom);
  if (filters.dateTo !== "") params.set("dateTo", filters.dateTo);
  params.set("page", String(filters.page));
  params.set("pageSize", String(filters.pageSize));

  return params.toString();
}

export function hasActiveFilters(filters: SearchFilters): boolean {
  return (
    filters.q.trim() !== "" ||
    filters.categoryIds.length > 0 ||
    filters.statuses.length > 0 ||
    filters.dateFrom !== "" ||
    filters.dateTo !== ""
  );
}
