#!/usr/bin/env node

import { fileURLToPath } from "node:url";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";

const siteRoot = fileURLToPath(new URL("..", import.meta.url));
const server = await createServer({
  root: siteRoot,
  appType: "custom",
  logLevel: "error",
  server: { middlewareMode: true, hmr: false, ws: false },
  ssr: { noExternal: ["@dmitrymake/rk-ui"] },
});

try {
  const { default: App } = await server.ssrLoadModule("/src/App.jsx");
  const html = renderToStaticMarkup(React.createElement(App));
  for (const marker of [
    "Стилометрия",
    "Три среза корпуса",
    "Исторический LOBO headline отозван",
  ]) {
    if (!html.includes(marker)) {
      throw new Error(`site render smoke: missing marker ${JSON.stringify(marker)}`);
    }
  }
  console.log("site render smoke: OK");
} finally {
  await server.close();
}
