#!/usr/bin/env node

import { fileURLToPath } from "node:url";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";

const siteRoot = fileURLToPath(new URL("..", import.meta.url));
const CHAPTER_MARKERS = {
  framework: ["Стилометрия", "Три среза корпуса", "Исторический LOBO headline отозван"],
  sholokhov: ["«Шолохов вообще не писатель»?"],
  ilfpetrov: ["Ильф и Петров: писал ли дилогию Булгаков?"],
  nikolai: ["Николай II: писал ли он свои дневники?"],
  hohol: ["Кто дописал «Тараса Бульбу»?"],
};

function verifyReferenceErrorSensitivity() {
  function BrokenNondefaultBranch() {
    return React.createElement("div", null, NONDEFAULT_FREE_IDENTIFIER);
  }
  try {
    renderToStaticMarkup(React.createElement(BrokenNondefaultBranch));
  } catch (error) {
    if (error instanceof ReferenceError) return;
    throw error;
  }
  throw new Error("site render smoke did not catch an undefined branch identifier");
}

const server = await createServer({
  root: siteRoot,
  appType: "custom",
  logLevel: "error",
  server: { middlewareMode: true, hmr: false, ws: false },
  ssr: { noExternal: ["@dmitrymake/rk-ui"] },
});

try {
  verifyReferenceErrorSensitivity();
  const { default: App, CHAPTER_IDS } = await server.ssrLoadModule("/src/App.jsx");
  const expectedChapters = Object.keys(CHAPTER_MARKERS);
  if (JSON.stringify(CHAPTER_IDS) !== JSON.stringify(expectedChapters)) {
    throw new Error(
      `site render smoke chapter mismatch: ${JSON.stringify(CHAPTER_IDS)} != ${JSON.stringify(expectedChapters)}`
    );
  }
  for (const chapter of expectedChapters) {
    const html = renderToStaticMarkup(
      React.createElement(App, { initialChapter: chapter })
    );
    for (const marker of CHAPTER_MARKERS[chapter]) {
      if (!html.includes(marker)) {
        throw new Error(
          `site render smoke: chapter ${chapter} missing marker ${JSON.stringify(marker)}`
        );
      }
    }
  }
  console.log(`site render smoke: OK (${expectedChapters.length} chapters)`);
} finally {
  await server.close();
}
