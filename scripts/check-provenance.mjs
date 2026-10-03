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
const SCHEMA = "stylo.site_generation_provenance.v2";
const HEX64 = /^[0-9a-f]{64}$/;
const ENTRY_KEY = /^[A-Za-z][A-Za-z0-9]*(?:\.[A-Za-z][A-Za-z0-9]*)*$/;

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
if (!Array.isArray(registry.entries) || registry.entries.length === 0) {
  fail("entries must be a non-empty array");
}

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
const outputPaths = registry.outputs.map((item) => item.path);
if (!outputPaths.includes("site/src/generated/site-data.json")) {
  fail("outputs must bind site/src/generated/site-data.json");
}

let siteData;
try {
  siteData = JSON.parse(
    readFileSync(resolve(root, "site/src/generated/site-data.json"), "utf-8"),
  );
} catch (error) {
  fail(`cannot parse site/src/generated/site-data.json: ${error.message}`);
}
if (siteData === null || typeof siteData !== "object" || Array.isArray(siteData)) {
  fail("site/src/generated/site-data.json must contain an object");
}

const sourcePathSet = new Set(sourcePaths);
const topicSource = "research/evidence/topic_validity_lobo_v1/aggregate.json";
const measurementOutput = "site/public/measurement/topic-validity-aggregate.json";
const pairedSource = "research/evidence/topic_validity_lobo_v1/paired_summary.json";
const pairedScript = "scripts/evaluation/summarize_topic_validity.py";
const authorRegistrySource = "src/stylo/resources/authors.json";
const pairedOutput = "site/public/measurement/topic-validity-paired-summary.json";
const hasPaired = sourcePathSet.has(pairedSource) || siteData.measurement?.pairedAnalysis !== undefined;
if (sourcePathSet.has(topicSource) || siteData.measurement !== undefined) {
  if (JSON.stringify([...outputPaths].sort()) !== JSON.stringify(
    ["site/src/generated/site-data.json", measurementOutput, ...(hasPaired ? [pairedOutput] : [])].sort())) {
    fail("measurement outputs must bind exactly site-data and the downloadable sources");
  }
  if (!sourcePathSet.has(topicSource) || !outputPaths.includes(measurementOutput)) {
    fail("measurement requires its canonical source and downloadable output bindings");
  }
  const sourceBytes = readFileSync(resolve(root, topicSource));
  if (!sourceBytes.equals(readFileSync(resolve(root, measurementOutput)))) {
    fail("downloadable measurement must be byte-identical to the canonical aggregate");
  }
  const artifact = JSON.parse(sourceBytes.toString("utf-8"));
  const measurement = siteData.measurement;
  exactKeys(measurement, ["source", "sourceSelfHash", "studyIdentity", "publicArtifact", "unit",
    "testedAuthors", "candidateClasses", "works", "fits", "cells", ...(hasPaired ? ["pairedAnalysis"] : [])], "measurement");
  const expectedScope = {
    source: topicSource, sourceSelfHash: artifact.self_hash, studyIdentity: artifact.study_identity,
    publicArtifact: "measurement/topic-validity-aggregate.json", unit: artifact.design.unit,
    testedAuthors: artifact.design.tested_author_count,
    candidateClasses: artifact.design.probability_class_count, works: artifact.design.fold_count,
    fits: artifact.design.fold_count * artifact.design.cells.length * artifact.design.arms.length,
  };
  for (const [key, expected] of Object.entries(expectedScope)) {
    if (measurement[key] !== expected) fail(`measurement.${key} differs from the canonical source`);
  }
  if (!Array.isArray(measurement.cells) || measurement.cells.length !== artifact.cells.length) {
    fail("measurement cells differ from the canonical source");
  }
  for (const [index, expected] of artifact.cells.entries()) {
    const cell = measurement.cells[index];
    if (cell.cell !== expected.cell) fail("measurement cell identity drift");
    for (const arm of artifact.design.arms) {
      const { correct, total } = expected.accuracy[arm];
      if (cell.accuracy[arm].correct !== correct || cell.accuracy[arm].total !== total ||
          cell.accuracy[arm].value !== correct / total) {
        fail(`measurement ${cell.cell}/${arm} accuracy differs from the canonical source`);
      }
    }
    const delta = expected.delta_accuracy;
    if (cell.delta.numerator !== delta.numerator || cell.delta.denominator !== delta.denominator ||
        cell.delta.value !== delta.numerator / delta.denominator) {
      fail(`measurement ${cell.cell} delta differs from the canonical source`);
    }
    for (const key of artifact.design.transition_categories) {
      const total = expected.per_author_transitions.reduce((sum, row) => sum + row[key], 0);
      if (cell.transitions[key] !== total) fail(`measurement ${cell.cell}/${key} transition count drift`);
    }
  }
  if (hasPaired) {
    if (!sourcePathSet.has(pairedSource) || !sourcePathSet.has(pairedScript) || !sourcePathSet.has(authorRegistrySource)) {
      fail("paired analysis requires source and computation script bindings");
    }
    const bytes = readFileSync(resolve(root, pairedSource));
    if (!bytes.equals(readFileSync(resolve(root, pairedOutput)))) fail("downloadable paired summary differs from source");
    const summary = JSON.parse(bytes.toString("utf-8"));
    if (summary.canonical_self_hash !== artifact.self_hash ||
        summary.inputs?.aggregate?.sha256 !== digest(sourceBytes) ||
        summary.script_path !== pairedScript ||
        summary.script_sha256 !== digest(readFileSync(resolve(root, pairedScript)))) {
      fail("paired summary source binding mismatch");
    }
    const authorRegistry = JSON.parse(readFileSync(resolve(root, authorRegistrySource), "utf-8"));
    const expected = { source: pairedSource, publicArtifact: "measurement/topic-validity-paired-summary.json",
      sourceSelfHash: summary.self_hash,
      authorNames: Object.fromEntries(summary.arms[0].per_author.map(({ author }) => [author, authorRegistry[author]?.name || author])),
      arms: summary.arms, comparisons: summary.comparisons };
    if (JSON.stringify(measurement.pairedAnalysis) !== JSON.stringify(expected)) {
      fail("paired analysis differs from source metrics/transitions");
    }
  }
} else if (JSON.stringify(outputPaths) !== JSON.stringify(["site/src/generated/site-data.json"])) {
  fail("unexpected generated output without a measurement source");
}
const citedSources = new Set();
const entryKeys = new Set();
const coveredRoots = new Set();
for (const [index, entry] of registry.entries.entries()) {
  const where = `entries[${index}]`;
  exactKeys(entry, ["key", "sources", "note"], where);
  if (typeof entry.key !== "string" || !ENTRY_KEY.test(entry.key)) {
    fail(`${where}.key must be a canonical dotted site-data path`);
  }
  if (entryKeys.has(entry.key)) fail(`${where}.key is duplicated: ${entry.key}`);
  entryKeys.add(entry.key);
  if (typeof entry.note !== "string" || entry.note.trim().length === 0) {
    fail(`${where}.note must be a non-empty string`);
  }
  if (!Array.isArray(entry.sources) || entry.sources.length === 0) {
    fail(`${where}.sources must be a non-empty array`);
  }
  const entrySources = entry.sources.map((source, sourceIndex) => {
    const path = safeRelativePath(source, `${where}.sources[${sourceIndex}]`);
    if (!sourcePathSet.has(path)) {
      fail(`${where}.sources[${sourceIndex}] is not digest-verified: ${path}`);
    }
    citedSources.add(path);
    return path;
  });
  if (
    new Set(entrySources).size !== entrySources.length ||
    JSON.stringify(entrySources) !== JSON.stringify([...entrySources].sort())
  ) {
    fail(`${where}.sources must be unique and sorted`);
  }

  let value = siteData;
  for (const segment of entry.key.split(".")) {
    if (
      value === null ||
      typeof value !== "object" ||
      !Object.prototype.hasOwnProperty.call(value, segment)
    ) {
      fail(`${where}.key does not resolve in site-data: ${entry.key}`);
    }
    value = value[segment];
  }
  coveredRoots.add(entry.key.split(".", 1)[0]);
}
if (
  JSON.stringify([...citedSources].sort()) !== JSON.stringify(sourcePaths)
) {
  fail("entries must cite every digest-verified source exactly by registered path");
}
if (
  JSON.stringify([...coveredRoots].sort()) !==
  JSON.stringify(Object.keys(siteData).sort())
) {
  fail("entries must cover every top-level site-data key");
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
