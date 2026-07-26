// Verify the typed site-generation registry, every consumed byte, and every generated output.
// Checkout: node scripts/check-provenance.mjs
// Git-free release archive: node scripts/check-provenance.mjs --archive
// Tests may point at a synthetic tree with: --root PATH --skip-tracked
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { lstatSync, readFileSync } from "node:fs";
import { dirname, isAbsolute, normalize, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const DEFAULT_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const MANIFEST_PATH = "site/src/generated/manifest.json";
const SCHEMA = "stylo.site_generation_provenance.v1";
const HEX64 = /^[0-9a-f]{64}$/;

function fail(message) {
  console.error(`PROVENANCE GATE: ${message}`);
  process.exit(1);
}

function parseArgs(argv) {
  let root = DEFAULT_ROOT;
  let trackingMode = "checkout";
  let trackingModeExplicit = false;
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === "--root" && i + 1 < argv.length) {
      root = resolve(argv[++i]);
    } else if (argv[i] === "--skip-tracked") {
      if (trackingModeExplicit) fail("tracking mode may be specified only once");
      trackingMode = "test-skip";
      trackingModeExplicit = true;
    } else if (argv[i] === "--archive") {
      if (trackingModeExplicit) fail("tracking mode may be specified only once");
      trackingMode = "archive";
      trackingModeExplicit = true;
    } else {
      fail(`unknown or incomplete argument ${JSON.stringify(argv[i])}`);
    }
  }
  return { root, trackingMode };
}

function hasGitMetadata(root) {
  const gitPath = resolve(root, ".git");
  try {
    const metadata = lstatSync(gitPath);
    if (!metadata.isDirectory() && !metadata.isFile()) {
      fail(`${gitPath} is not regular Git metadata`);
    }
    return true;
  } catch (error) {
    if (error?.code === "ENOENT") return false;
    fail(`cannot inspect Git metadata at ${gitPath}: ${error.message}`);
  }
}

function exactKeys(value, keys, where) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    fail(`${where} must be an object`);
  }
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    fail(`${where} keys must be exactly ${expected.join(", ")}`);
  }
}

function safeRelativePath(value, where) {
  if (typeof value !== "string" || value.length === 0 || isAbsolute(value)) {
    fail(`${where} must be a non-empty repository-relative path`);
  }
  const canonical = normalize(value);
  if (
    canonical !== value ||
    value === ".." ||
    value.startsWith(`..${sep}`) ||
    value.includes("\\")
  ) {
    fail(`${where} is not a canonical repository-relative path: ${JSON.stringify(value)}`);
  }
  return value;
}

function digest(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function verifyBinding(root, binding, where) {
  exactKeys(binding, ["path", "sha256"], where);
  const path = safeRelativePath(binding.path, `${where}.path`);
  if (typeof binding.sha256 !== "string" || !HEX64.test(binding.sha256)) {
    fail(`${where}.sha256 must be lowercase hex64`);
  }
  let bytes;
  try {
    bytes = readFileSync(resolve(root, path));
  } catch (error) {
    fail(`${where} cannot read ${path}: ${error.message}`);
  }
  const actual = digest(bytes);
  if (actual !== binding.sha256) {
    fail(`${path} digest mismatch: registered=${binding.sha256} actual=${actual}`);
  }
  return path;
}

const { root, trackingMode } = parseArgs(process.argv.slice(2));
const gitMetadataPresent = hasGitMetadata(root);
if (trackingMode === "checkout" && !gitMetadataPresent) {
  fail("Git metadata is absent; a verified source archive must use --archive");
}
if (trackingMode === "archive" && gitMetadataPresent) {
  fail("--archive requires a Git-free root and cannot bypass checkout trackedness");
}
let registry;
try {
  registry = JSON.parse(readFileSync(resolve(root, MANIFEST_PATH), "utf-8"));
} catch (error) {
  fail(`cannot parse ${MANIFEST_PATH}: ${error.message}`);
}

exactKeys(registry, ["schema", "generator", "sources", "outputs", "entries"], "registry");
if (registry.schema !== SCHEMA) fail(`schema must be ${SCHEMA}`);
if (!Array.isArray(registry.sources) || registry.sources.length === 0) {
  fail("sources must be a non-empty array");
}
if (!Array.isArray(registry.outputs) || registry.outputs.length === 0) {
  fail("outputs must be a non-empty array");
}
if (!Array.isArray(registry.entries)) fail("entries must be an array");

const paths = [];
paths.push(verifyBinding(root, registry.generator, "generator"));
for (const [index, binding] of registry.sources.entries()) {
  paths.push(verifyBinding(root, binding, `sources[${index}]`));
}
for (const [index, binding] of registry.outputs.entries()) {
  paths.push(verifyBinding(root, binding, `outputs[${index}]`));
}

if (new Set(paths).size !== paths.length) fail("generator/source/output paths must be unique");
const sourcePaths = registry.sources.map((item) => item.path);
if (JSON.stringify(sourcePaths) !== JSON.stringify([...sourcePaths].sort())) {
  fail("sources must be sorted by path");
}
if (registry.generator.path !== "scripts/gen-site-data.mjs") {
  fail("generator path must be scripts/gen-site-data.mjs");
}
if (
  registry.outputs.length !== 1 ||
  registry.outputs[0].path !== "site/src/generated/site-data.json"
) {
  fail("outputs must bind exactly site/src/generated/site-data.json");
}

if (trackingMode === "checkout") {
  for (const path of [...paths, MANIFEST_PATH]) {
    try {
      execFileSync("git", ["ls-files", "--error-unmatch", "--", path], {
        cwd: root,
        stdio: "ignore",
      });
    } catch {
      fail(`${path} is not tracked; a clean clone cannot reproduce the site`);
    }
  }
}

console.log(
  `✓ provenance: ${registry.sources.length} source digests and ` +
  `${registry.outputs.length} output digest verified`,
);
