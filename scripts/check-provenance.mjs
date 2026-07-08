// Гейт провенанса: КАЖДЫЙ источник, читаемый генераторами (gen-site-data/gen-readme),
// должен быть в git-индексе — иначе свежий клон не пересоберёт и не проверит числа на сайте/README,
// а это и есть заявленный дифференциатор (clean-clone reproducibility). Запуск: node scripts/check-provenance.mjs
// Код выхода 1, если хоть один источник untracked. Безопасно гонять в CI/pre-commit.
import { readFileSync } from "node:fs";
import { execSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const GENS = ["scripts/gen-site-data.mjs", "scripts/gen-readme.mjs"];

const sources = new Set();
for (const g of GENS) {
  const s = readFileSync(resolve(ROOT, g), "utf-8");
  // load("x.json") / jload("docs/x.json") / load("cases/x.json") — путь может содержать подпапку (cases/)
  for (const m of s.matchAll(/\b(?:load|jload)\(\s*"(?:docs\/)?([A-Za-z0-9_/]+\.json)"/g)) sources.add("docs/" + m[1]);
  // CSV-источники (final_comparison.csv и т.п.)
  for (const m of s.matchAll(/"(?:docs\/)?([A-Za-z0-9_/]+\.csv)"/g)) sources.add("docs/" + m[1]);
}

const tracked = new Set(execSync("git ls-files docs/", { cwd: ROOT }).toString().trim().split("\n"));
const missing = [...sources].filter((s) => !tracked.has(s)).sort();

if (missing.length) {
  console.error("ГЕЙТ ПРОВЕНАНСА: источники генераторов НЕ в git-индексе (свежий клон сломается):\n  " + missing.join("\n  "));
  console.error("Исправить: git add " + missing.join(" "));
  process.exit(1);
}
console.log(`✓ провенанс: все ${sources.size} источников генераторов в git-индексе`);
