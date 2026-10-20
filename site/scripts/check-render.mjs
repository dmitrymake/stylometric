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
  const { default: App, CHAPTER_IDS, loadChapterForRender } = await server.ssrLoadModule("/src/App.jsx");
  if (!CHAPTER_IDS.length || new Set(CHAPTER_IDS).size !== CHAPTER_IDS.length) {
    throw new Error("site render smoke requires nonempty, unique chapter choices");
  }
  for (const chapter of CHAPTER_IDS) {
    const chapterComponent = await loadChapterForRender(chapter);
    // Exercise each lazy branch without freezing markup or editorial content.
    const content = chapterComponent
      ? renderToStaticMarkup(React.createElement(chapterComponent))
      : renderToStaticMarkup(React.createElement(App, { initialChapter: chapter }));
    if (!content.trim()) {
      throw new Error(`site render smoke: chapter ${chapter} rendered nothing`);
    }
    renderToStaticMarkup(
      React.createElement(App, { initialChapter: chapter, chapterComponent })
    );
  }
  console.log(`site render smoke: OK (${CHAPTER_IDS.length} chapters)`);
} finally {
  await server.close();
}
