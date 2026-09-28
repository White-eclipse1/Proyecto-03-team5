/**
 * SPEC-VALID-001 ÔÇö Errores tipados de la capa Logic.
 *
 * La capa UI mapea la clase del error a un c├│digo HTTP. De este modo no
 * necesita inspeccionar mensajes de texto para decidir el status, lo que
 * ser├¡a fr├ígil ante cualquier cambio de redacci├│n.
 */

/**
 * Dato inv├ílido o regla de negocio violada. La capa UI responde 400.
 */
export class ValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ValidationError';
  }
}

/**
 * El recurso solicitado no existe en la base de datos. La capa UI responde 404.
 */
export class NotFoundError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'NotFoundError';
  }
}
