import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server on 5173 (the backend's default CORS origin).
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, strictPort: true },
});
