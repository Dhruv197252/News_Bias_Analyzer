import { defineConfig } from 'vite'

export default defineConfig({
  // Vite 8 has native JSX support via rolldown without any plugin needed.
  // Just tell it which JSX runtime to use.
  jsx: {
    mode: 'automatic',
    importSource: 'react',
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      }
    }
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
  }
})
