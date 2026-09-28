import { sql } from 'drizzle-orm';
import { db } from '../data/index.js';

export type HealthStatus = {
  status: 'ok' | 'error';
  database: 'connected' | 'unreachable';
  timestamp: string;
};

/**
 * Verifica que la aplicación esté arriba y que la conexión a MariaDB
 * funcione. Es la única pieza de "lógica de negocio" que existe en la
 * Fase 1: sirve para validar el esqueleto UI → Logic → Data, no para
 * ninguna funcionalidad de fases posteriores.
 */
export async function checkHealth(): Promise<HealthStatus> {
  const timestamp = new Date().toISOString();

  try {
    await db.execute(sql`SELECT 1`);
    return { status: 'ok', database: 'connected', timestamp };
  } catch {
    return { status: 'error', database: 'unreachable', timestamp };
  }
}
