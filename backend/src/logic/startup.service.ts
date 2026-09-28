import { ensureMinioBucket } from '../data/index.js';

/**
 * Inicializa los servicios necesarios para arrancar la aplicación.
 */
export async function initializeApplication(): Promise<void> {
  await ensureMinioBucket();
}
