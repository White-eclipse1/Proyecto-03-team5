import { type Category, listCategories } from '../data/index.js';

/**
 * Devuelve todas las categorías disponibles para anotar.
 */
export async function getCategories(): Promise<Category[]> {
  return listCategories();
}
