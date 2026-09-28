import 'dotenv/config';
import { drizzle } from 'drizzle-orm/mysql2';
import mysql from 'mysql2/promise';
import * as schema from './schema.js';

const databaseUrl = process.env.DATABASE_URL;

if (!databaseUrl) {
  throw new Error('DATABASE_URL no está definida. Copia .env.example a .env y configúrala.');
}

/**
 * Pool de conexiones a MariaDB. Se reutiliza en toda la app (data layer)
 * para evitar abrir una conexión nueva por query.
 */
export const pool = mysql.createPool({
  uri: databaseUrl,
  connectionLimit: 10,
});

/**
 * Instancia de Drizzle ORM, tipada con el schema del proyecto.
 * Este es el único punto de acceso a la base de datos: la capa `logic`
 * debe consumir repositorios/queries de `data`, nunca este cliente
 * directamente desde `ui`.
 */
export const db = drizzle(pool, { schema, mode: 'default' });
