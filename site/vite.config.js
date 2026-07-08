import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base: './' — относительные пути, работает на github.io из любой подпапки.
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: { outDir: "dist", assetsInlineLimit: 0 },
});
