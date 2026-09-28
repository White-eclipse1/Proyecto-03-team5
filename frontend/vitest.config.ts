import path from 'node:path';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';

// Espeja el alias "@" de vite.config.ts: sin esto, cualquier módulo que
// importe con "@/..." (como App.tsx) falla a resolver bajo Vitest aunque
// funcione con `vite dev`/`vite build`.
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  test: {
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
  },
});