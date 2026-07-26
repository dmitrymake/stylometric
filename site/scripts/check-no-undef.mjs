#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parse } from "@babel/parser";
import traverseModule from "@babel/traverse";

const traverse = traverseModule.default;
const siteRoot = fileURLToPath(new URL("..", import.meta.url));
const sourceRoot = path.join(siteRoot, "src");
const BROWSER_GLOBALS = new Set(["document", "window", "IntersectionObserver"]);

function sourceFiles(directory) {
  return fs
    .readdirSync(directory, { withFileTypes: true })
    .flatMap((entry) => {
      const target = path.join(directory, entry.name);
      if (entry.isDirectory()) return sourceFiles(target);
      return /\.(?:js|jsx)$/.test(entry.name) ? [target] : [];
    })
    .sort();
}

function unresolvedIdentifiers(source, filename) {
  const ast = parse(source, {
    sourceType: "module",
    sourceFilename: filename,
    plugins: ["jsx"],
  });
  const unresolved = [];
  traverse(ast, {
    ReferencedIdentifier(identifierPath) {
      const { name } = identifierPath.node;
      if (
        !identifierPath.scope.hasBinding(name) &&
        !BROWSER_GLOBALS.has(name)
      ) {
        unresolved.push({
          name,
          line: identifierPath.node.loc?.start.line ?? 0,
          column: identifierPath.node.loc?.start.column ?? 0,
        });
      }
    },
  });
  return unresolved;
}

const negative = unresolvedIdentifiers(
  "export function hidden() { return NONDEFAULT_FREE_IDENTIFIER; }",
  "<negative-self-test>"
);
if (
  negative.length !== 1 ||
  negative[0].name !== "NONDEFAULT_FREE_IDENTIFIER"
) {
  throw new Error("site no-undef negative self-test did not detect its fixture");
}

const failures = sourceFiles(sourceRoot).flatMap((filename) =>
  unresolvedIdentifiers(fs.readFileSync(filename, "utf8"), filename).map(
    (failure) => ({
      ...failure,
      filename: path.relative(siteRoot, filename),
    })
  )
);
if (failures.length) {
  for (const failure of failures) {
    console.error(
      `${failure.filename}:${failure.line}:${failure.column + 1}: undefined identifier ${failure.name}`
    );
  }
  process.exit(1);
}

console.log(`site no-undef: OK (${sourceFiles(sourceRoot).length} source files)`);
