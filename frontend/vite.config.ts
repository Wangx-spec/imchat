import { fileURLToPath, URL } from "node:url";

import vue from "@vitejs/plugin-vue";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [vue()],
  base: "/",
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
    },
  },
  build: {
    outDir: "../src/web/static",
    emptyOutDir: false,
    assetsDir: "frontend/assets",
    rollupOptions: {
      output: {
        entryFileNames: "frontend/assets/[name]-[hash].js",
        chunkFileNames: "frontend/assets/[name]-[hash].js",
        assetFileNames: "frontend/assets/[name]-[hash][extname]",
      },
    },
  },
});
