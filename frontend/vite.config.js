import { defineConfig } from 'vite'

export default defineConfig({
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
