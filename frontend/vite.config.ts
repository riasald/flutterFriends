import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // If something else is already on 5173, fail instead of silently using
    // 5174 — opening localhost:5173 would then show the *other* app ("old UI").
    strictPort: true,
  },
})
