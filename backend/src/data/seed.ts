import 'dotenv/config';

import { eq } from 'drizzle-orm';

import { db, pool } from './db/client.js';
import { categories } from './db/schema.js';
import { ensureMinioBucket } from './storage/minio.storage.js';

/**
 * Categorías de ejemplo.
 * El nombre es UNIQUE, por lo que el seeder puede ejecutarse varias veces
 * sin crear duplicados.
 */
const seedCategories = [
  { name: 'dog', color: '#2ECC71' },
  { name: 'cat', color: '#9B59B6' },
] as const;

/**
 * Inserta o actualiza las categorías sin duplicarlas.
 */
async function seedCategoryData(): Promise<void> {
  for (const category of seedCategories) {
    const existing = await db
      .select({ id: categories.id })
      .from(categories)
      .where(eq(categories.name, category.name))
      .limit(1);

    if (existing.length === 0) {
      await db.insert(categories).values(category);
    } else {
      await db
        .update(categories)
        .set({ color: category.color })
        .where(eq(categories.name, category.name));
    }
  }
}

/**
 * Ejecuta el seeder completo.
 */
async function seed(): Promise<void> {
  await ensureMinioBucket();

  await seedCategoryData();

  console.log('Seeder completado correctamente.');
}

seed()
  .catch((error: unknown) => {
    console.error('Error al ejecutar el seeder:', error);
    process.exitCode = 1;
  })
  .finally(async () => {
    await pool.end();
  });
