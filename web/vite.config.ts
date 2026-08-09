import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The bundle lands in `server/static/`, which stage 4 serves and which is
// gitignored — `npm --prefix web run build` is a documented prerequisite for
// running the server with a UI. The dev server proxies to a local FORGE server
// so `sse` and `ws` work from `npm run dev`; the `mock` transport needs neither.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../server/static",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/ws": { target: "ws://127.0.0.1:8000", ws: true },
    },
  },
});
