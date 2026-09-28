import 'dotenv/config';
import { defineConfig } from 'drizzle-kit';

const databaseUrl = process.env.DATABASE_URL;

if (!databaseUrl) {
  throw new Error('DATABASE_URL no está definida. Copia .env.example a .env y configúrala.');
}

export default defineConfig({
  dialect: 'mysql',
  schema: './src/data/db/schema.ts',
  out: './src/data/db/migrations',
  dbCredentials: {
    url: databaseUrl,
  },
  strict: true,
  verbose: true,
});
