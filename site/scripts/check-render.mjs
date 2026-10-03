#!/usr/bin/env node

import { fileURLToPath } from "node:url";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";

const siteRoot = fileURLToPath(new URL("..", import.meta.url));
// Stable chapter/root identifiers are navigation and render contracts; prose is editorial.
const CHAPTER_ROOTS = {
  framework: "top",
  ilfpetrov: "ilfpetrov",
  sholokhov: "sholokhov",
  nikolai: "nikolai",
  hohol: "hohol",
};
const textContent = (html) => html.replace(/<[^>]*>/g, "").trim();

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
  const { default: App, CHAPTER_IDS, loadChapterForRender } = await server.ssrLoadModule("/src/App.jsx");
  const expectedChapters = Object.keys(CHAPTER_ROOTS);
  if (new Set(CHAPTER_IDS).size !== CHAPTER_IDS.length ||
      JSON.stringify([...CHAPTER_IDS].sort()) !== JSON.stringify([...expectedChapters].sort())) {
    throw new Error(
      `site render smoke chapter mismatch: ${JSON.stringify(CHAPTER_IDS)} != ${JSON.stringify(expectedChapters)}`
    );
  }
  for (const chapter of expectedChapters) {
    const chapterComponent = await loadChapterForRender(chapter);
    const html = renderToStaticMarkup(
      React.createElement(App, { initialChapter: chapter, chapterComponent })
    );
    const main = html.match(/<main\b[^>]*>([\s\S]*?)<\/main>/)?.[1];
    if (!main || !textContent(main)) {
      throw new Error(`site render smoke: chapter ${chapter} has no main content`);
    }
    const heading = main.match(/<h[12]\b[^>]*>([\s\S]*?)<\/h[12]>/)?.[1];
    if (!heading || !textContent(heading)) {
      throw new Error(`site render smoke: chapter ${chapter} has no heading`);
    }
    if (!main.includes(`id="${CHAPTER_ROOTS[chapter]}"`)) {
      throw new Error(`site render smoke: chapter ${chapter} did not render its content root`);
    }
  }
  console.log(`site render smoke: OK (${expectedChapters.length} chapters)`);
} finally {
  await server.close();
}
