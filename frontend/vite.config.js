import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// mode "demo": static build for GitHub Pages (relative paths, no backend)
export default defineConfig(({ mode }) => ({
  base: mode === "demo" ? "./" : "/",
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": process.env.VITE_API_TARGET || "http://localhost:8000" },
  },
  build: { chunkSizeWarningLimit: 6000 },
}));
