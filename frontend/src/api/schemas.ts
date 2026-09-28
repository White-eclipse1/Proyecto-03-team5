import { z } from "zod";

/**
 * Todo lo que llega del backend se valida aquí. Nunca se confía en el tipo
 * que TypeScript infiere del JSON de un fetch — se parsea con estos schemas
 * y el tipo real de cada componente sale de z.infer<typeof schema>.
 */

export const imageStatusSchema = z.enum(["pending", "in_progress", "completed"]);
export type ImageStatus = z.infer<typeof imageStatusSchema>;

export const categorySchema = z.object({
  id: z.number(),
  name: z.string(),
  color: z.string().regex(/^#[0-9A-Fa-f]{6}$/, "color debe ser hex #RRGGBB"),
});
export type Category = z.infer<typeof categorySchema>;

export const categoriesResponseSchema = z.array(categorySchema);

export const searchResultCategorySchema = z.object({
  id: z.number(),
  name: z.string(),
  color: z.string(),
});

export const imageSearchItemSchema = z.object({
  id: z.number(),
  filename: z.string(),
  thumbnailUrl: z.string(),
  width: z.number(),
  height: z.number(),
  status: imageStatusSchema,
  annotationsCount: z.number(),
  categories: z.array(searchResultCategorySchema),
  createdAt: z.string(),
});
export type ImageSearchItem = z.infer<typeof imageSearchItemSchema>;

export const paginationSchema = z.object({
  page: z.number(),
  pageSize: z.number(),
  total: z.number(),
  totalPages: z.number(),
});
export type Pagination = z.infer<typeof paginationSchema>;

export const imageSearchResponseSchema = z.object({
  data: z.array(imageSearchItemSchema),
  pagination: paginationSchema,
});
export type ImageSearchResponse = z.infer<typeof imageSearchResponseSchema>;
