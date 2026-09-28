import { type Category, categoriesResponseSchema } from "../../types/schemas";
import { apiRequest } from "./client";

export function getCategories(): Promise<Category[]> {
  return apiRequest("/categories", categoriesResponseSchema);
}
