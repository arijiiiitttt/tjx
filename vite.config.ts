import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import { fileURLToPath } from 'node:url';

export default defineConfig({
  plugins: [react()],
  resolve: { alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) } },
  worker: { format: 'es' },
  build: { target: 'es2022', chunkSizeWarningLimit: 4000 },
  test: { include: ['tests/**/*.test.{ts,tsx}'], testTimeout: 120_000, hookTimeout: 60_000 },
});
