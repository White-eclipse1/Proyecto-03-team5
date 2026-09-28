/**
 * SPEC-SEARCH-001 ÔÇö Parseo de consultas de b├║squeda por clases.
 *
 * M├│dulo puro: no importa la capa de datos ni la configuraci├│n,
 * por lo que puede probarse en aislamiento total.
 */

export type SearchOperator = 'AND' | 'OR';

export interface ParsedSearchQuery {
  /** Nombres de clase normalizados a min├║sculas. */
  terms: string[];
  /** Operador l├│gico que une los t├®rminos. */
  operator: SearchOperator;
}

/** Operadores reconocidos: solo en may├║sculas y como palabra completa. */
const OPERATORS = ['AND', 'OR'] as const;

function isOperator(token: string): token is SearchOperator {
  return (OPERATORS as readonly string[]).includes(token);
}

/**
 * Convierte una consulta de texto en t├®rminos + operador.
 *
 * Devuelve `null` cuando la consulta est├í vac├¡a (no hay b├║squeda que aplicar).
 * Lanza `Error` cuando la sintaxis es inv├ílida.
 *
 * Ejemplos:
 *   'car'              ÔåÆ { terms: ['car'], operator: 'AND' }
 *   'car AND person'   ÔåÆ { terms: ['car', 'person'], operator: 'AND' }
 *   'car OR person'    ÔåÆ { terms: ['car', 'person'], operator: 'OR' }
 *   'android'          ÔåÆ { terms: ['android'], operator: 'AND' }
 */
export function parseSearchQuery(raw: string): ParsedSearchQuery | null {
  const tokens = raw
    .trim()
    .split(/\s+/)
    .filter((token) => token.length > 0);

  // Consulta vac├¡a: no se aplica ning├║n filtro de b├║squeda.
  if (tokens.length === 0) {
    return null;
  }

  const terms: string[] = [];
  const operatorsFound = new Set<SearchOperator>();

  // Se espera la secuencia: t├®rmino (operador t├®rmino)*
  // Las posiciones pares deben ser t├®rminos; las impares, operadores.
  for (const [index, token] of tokens.entries()) {
    const expectsOperator = index % 2 === 1;

    if (expectsOperator) {
      if (!isOperator(token)) {
        throw new Error(
          `Se esperaba AND u OR en la posici├│n ${index + 1} de la b├║squeda, se recibi├│ "${token}".`,
        );
      }
      operatorsFound.add(token);
      continue;
    }

    if (isOperator(token)) {
      throw new Error(`Se esperaba un nombre de clase, pero se recibi├│ el operador "${token}".`);
    }

    terms.push(token.toLowerCase());
  }

  // Un operador al final queda sin t├®rmino a la derecha.
  const lastToken = tokens[tokens.length - 1];
  if (lastToken !== undefined && isOperator(lastToken)) {
    throw new Error(`La b├║squeda no puede terminar con el operador "${lastToken}".`);
  }

  // Mezclar AND con OR har├¡a la precedencia ambigua.
  if (operatorsFound.size > 1) {
    throw new Error('No se puede combinar AND y OR en la misma b├║squeda.');
  }

  // Con un solo t├®rmino el operador es irrelevante; AND es el valor por defecto.
  const [operator = 'AND'] = [...operatorsFound];

  return { terms, operator };
}
