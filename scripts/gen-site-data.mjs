// ════════════════════════════════════════════════════════════════════
//  Генератор данных сайта: docs/*.json (выходы прогонов) → site/src/generated/.
//  data.js/corpus.js/segdata.js берут числа ТОЛЬКО отсюда (без литералов). Единств. исключения
//  в segdata: ccat50Valla (внешняя цитата обзора Valla, не наш прогон) и nSegments=0 (сегмент-
//  прогон 12 стульев, чей JSON не сохранён). Пересчёт прогона → этот скрипт → числа на сайте
//  обновляются сами (prebuild-хук); сгенерированное — под coverage-гейтом ниже. Значения, что
//  живут лишь в прозе прогона, тянутся якорным grab()-regex с null-фоллбэком (дрейф ловит гейт).
//
//  Запуск: node scripts/gen-site-data.mjs   (из корня репозитория)
//  Пишет:  site/src/generated/site-data.json + manifest.json (провенанс).
//  Падает с кодом 1, если обязательный источник отсутствует или поле пустое.
// ════════════════════════════════════════════════════════════════════
import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const DOCS = join(ROOT, "docs");
const OUT = join(ROOT, "site", "src", "generated");

const manifest = []; // провенанс: какой ключ из какого файла/поля
const consumedSources = new Set();
function track(key, file, note) { manifest.push({ key, source: file, note: note || "" }); }
function sha256(bytes) { return createHash("sha256").update(bytes).digest("hex"); }

function parseStrictJson(raw, label) {
  try {
    return JSON.parse(raw);
  } catch (error) {
    throw new Error(`${label}: invalid strict JSON (${error.message})`);
  }
}

function finiteCsvNumber(raw, label, { required = true } = {}) {
  if (raw === undefined || raw === null || String(raw).trim() === "") {
    if (required) throw new Error(`${label}: required finite number is blank`);
    return null;
  }
  const value = Number(raw);
  if (!Number.isFinite(value)) throw new Error(`${label}: expected a finite number`);
  return value;
}

function finiteCsvInterval(raw, label) {
  if (raw === undefined || String(raw).trim() === "") {
    throw new Error(`${label}: required finite interval is blank`);
  }
  const values = raw.replace(/[[\]]/g, "").split(",").map(
    (value, index) => finiteCsvNumber(value, `${label}[${index}]`)
  );
  if (values.length !== 2 || values[0] > values[1]) {
    throw new Error(`${label}: expected an ordered two-number interval`);
  }
  return values;
}

if (process.argv.includes("--self-test")) {
  const assert = (condition, message) => {
    if (!condition) throw new Error(`site generator self-test failed: ${message}`);
  };
  assert(parseStrictJson('{"note":"NaN"}', "fixture").note === "NaN", "prose was mutated");
  for (const malformed of ['{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}']) {
    let rejected = false;
    try { parseStrictJson(malformed, "fixture"); } catch { rejected = true; }
    assert(rejected, `accepted ${malformed}`);
  }
  for (const malformed of ["", "bogus", "Infinity", "NaN"]) {
    let rejected = false;
    try { finiteCsvNumber(malformed, "metric"); } catch { rejected = true; }
    assert(rejected, `accepted required metric ${JSON.stringify(malformed)}`);
  }
  assert(finiteCsvNumber("", "optional", { required: false }) === null, "optional blank");
  assert(finiteCsvNumber("0.5", "metric") === 0.5, "finite metric");
  console.log("site generator strict-input self-test: OK");
  process.exit(0);
}

function load(name) {
  const p = join(DOCS, name);
  if (!existsSync(p)) { console.error(`ОТСУТСТВУЕТ обязательный источник: docs/${name}`); process.exit(1); }
  consumedSources.add(`docs/${name}`);
  return parseStrictJson(readFileSync(p, "utf-8"), `docs/${name}`);
}

function loadRepositoryJson(relativePath) {
  const p = join(ROOT, relativePath);
  if (!existsSync(p)) {
    console.error(`ОТСУТСТВУЕТ обязательный источник: ${relativePath}`);
    process.exit(1);
  }
  consumedSources.add(relativePath);
  return parseStrictJson(readFileSync(p, "utf-8"), relativePath);
}

// final_comparison.csv: model,accuracy,acc_ci,macro_f1,top2,ece,vs_stylo_dacc,vs_stylo_mcnemar_p
// acc_ci — поле в кавычках с запятой внутри: "[0.849,0.924]".
function loadModelsCsv() {
  const p = join(DOCS, "final_comparison.csv");
  if (!existsSync(p)) { console.error("ОТСУТСТВУЕТ docs/final_comparison.csv"); process.exit(1); }
  consumedSources.add("docs/final_comparison.csv");
  const lines = readFileSync(p, "utf-8").trim().split("\n");
  const rows = [];
  for (const [rowIndex, line] of lines.slice(1).entries()) {
    // сплит по запятым ВНЕ кавычек
    const c = line.split(/,(?=(?:[^"]*"[^"]*")*[^"]*$)/).map(s => s.replace(/^"|"$/g, "").trim());
    const [model, accuracy, acc_ci, macro_f1, top2, ece, dacc, p_mc] = c;
    if (!model) throw new Error(`docs/final_comparison.csv row ${rowIndex + 2}: model is blank`);
    const prefix = `docs/final_comparison.csv row ${rowIndex + 2}`;
    rows.push({
      id: model,
      acc: finiteCsvNumber(accuracy, `${prefix}.accuracy`),
      ci: finiteCsvInterval(acc_ci, `${prefix}.acc_ci`),
      f1: finiteCsvNumber(macro_f1, `${prefix}.macro_f1`),
      top2: finiteCsvNumber(top2, `${prefix}.top2`),
      ece: finiteCsvNumber(ece, `${prefix}.ece`, { required: false }),
      p: finiteCsvNumber(p_mc, `${prefix}.p`, { required: false }),
    });
  }
  track("models", "docs/final_comparison.csv", "stylo/bow_lr/char_cos/delta/majority 1:1");
  return rows;
}

const val = load("validation.json");
const valPd = load("validation_pd.json");
const rec = load("model_recall.json");
const corpusVal = load("corpus_validation.json");
const ccat = load("ccat50.json");
const seg = load("segment_recall.json");
const lobo = load("lobo_fast.json");
const attrib = load("sholokhov_attrib.json");
const tomskFinal = load("tomsk_final.json");
const tomskFull = load("tomsk_full.json");
const ineligibleCorpus = loadRepositoryJson(
  "research/evidence/ineligible_corpus_registrations_v1.json"
);
if (
  ineligibleCorpus.status !== "ineligible_for_new_scientific_runs" ||
  !Array.isArray(ineligibleCorpus.affected) ||
  ineligibleCorpus.affected.length !== 1
) {
  throw new Error("ineligible corpus registry does not carry the expected fail-closed status");
}
const historicalCorpusRegistration = ineligibleCorpus.affected[0];

// ── корпус (три среза) ──
// источник LOBO-размеров (пул/тестированные/книги) — docs/stylo_lobo_authorci.json, не литералы (защита от дрейфа)
const styloCI = load("stylo_lobo_authorci.json");
const corpus = {
  research: {
    authors: corpusVal.summary.n_authors, books: corpusVal.summary.n_books,
    words: corpusVal.summary.total_words, imbalanceRatio: Math.round(corpusVal.summary.imbalance_ratio),
  },
  benchmark: { authors: val.n_authors, books: val.n_books, chunks: val.n_chunks },
  // Исторический ineligible headline-срез: полный per-book LOBO (final.py → final_comparison.csv).
  // пул/тестированные/книги — из stylo_lobo_authorci.json (4 автора с одной книгой в LOBO не тестируются — нет train-примера).
  lobo: { pool_authors: styloCI.n_authors_dataset, tested_authors: styloCI.n_authors_tested, books: styloCI.n_books },
  pd: {
    authors: valPd.n_authors, books: valPd.n_books, chunks: valPd.n_chunks,
    ensembleTop1: valPd.channels["АНСАМБЛЬ (равновес.)"].top1, ensembleMacroF1: valPd.channels["АНСАМБЛЬ (равновес.)"].macro_f1, ci: valPd.macro_f1_authorclustered_CI,
  },
};
track("corpus.research", "docs/corpus_validation.json", "summary");
track("corpus.benchmark", "docs/validation.json", "n_authors/n_books/n_chunks");
track("corpus.pd", "docs/validation_pd.json", "channels['АНСАМБЛЬ (равновес.)'].top1/macro_f1 + macro_f1_authorclustered_CI");

// ── Исторические stylo-LR LOBO артефакты из ineligible corpus snapshot ──
// acc/macroF1/top2/ece — сохранённая арифметика final_comparison.csv (251 книга);
// macroF1CI — отозванный author-clustered интервал (docs/stylo_lobo_authorci.json).
// Равновесный ансамбль каналов (LinearSVC + StratifiedGroupKFold(5)) — ДРУГОЙ классификатор/протокол,
// отдельная историческая диагностика (ensemble* ниже), не действующий headline.
const stylo = loadModelsCsv().find(m => m.id === "stylo");
const headline = {
  // macroF1CI ОТОЗВАН (null): author-clustered bootstrap ресэмпла авторов меняет набор классов
  // macro-усреднения → это не CI фиксированной 43-классовой функции. Статус/прежнее значение/ссылка ниже.
  accuracy: stylo.acc, macroF1: stylo.f1, macroF1CI: styloCI.macro_f1_authorclustered_CI,
  macroF1CIStatus: styloCI.macro_f1_authorclustered_interval_status,
  macroF1CISuperseded: styloCI.macro_f1_authorclustered_superseded_interval,
  macroF1CIErratumRef: styloCI.macro_f1_authorclustered_erratum_ref,
  macroF1BootstrapMedian: styloCI.macro_f1_bootstrap_median,
  accCIAuthor: styloCI.accuracy_authorclustered_CI,               // author-clustered 95% CI accuracy
  accBootstrapMedian: styloCI.accuracy_bootstrap_median,
  styloMacroF1: stylo.f1,   // = headline.macroF1 (историческая stylo-LR LOBO)
  top2: stylo.top2, ece: stylo.ece, ci: stylo.ci,
  ensembleTop1: val.ensemble_top1, ensembleMacroF1: val.headline_macro_f1, ensembleTop3: val.ensemble_top3,
  // train-side взвешивание — по чанкам; work-balanced пересчёт ещё не проведён (§1.2 плана)
  trainingWeighting: styloCI.training_weighting, claimStatus: styloCI.claim_status,
  corpusEligibilityStatus: ineligibleCorpus.status,
  corpusEligibilityReason: historicalCorpusRegistration.reason,
  requiredCorpusMigration: historicalCorpusRegistration.required_migration,
};
track(
  "headline",
  "docs/final_comparison.csv + docs/stylo_lobo_authorci.json + research/evidence/ineligible_corpus_registrations_v1.json",
  "историческая арифметика; corpus snapshot непригоден для новых scientific claims"
);

// ── историческая таблица моделей на ineligible corpus snapshot ──
const models = loadModelsCsv();

// ── каналы (одиночные SVM) из validation.json ──
const channels = {
  ensembleTop1: val.ensemble_top1, ensembleMacroF1: val.headline_macro_f1,
  rows: Object.entries(val.channels)
    .filter(([k]) => !k.startsWith("АНСАМБЛЬ"))
    .map(([id, v]) => ({ id, top1: v.top1, f1: v.macro_f1 })),
};
track("channels", "docs/validation.json", "channels{} без ансамбля");

// ── per-author recall + путаницы (числа; имена/заметки — в data.js) ──
const authorRecall = rec.per_author_recall.map(r => ({ id: r.id, recall: r.recall, books: r.books }));
const confusions = rec.top_confusions.map(c => ({ trueId: c.true, predId: c.pred, n: c.n }));
track("authorRecall", "docs/model_recall.json", "per_author_recall");
track("confusions", "docs/model_recall.json", "top_confusions");

// ── сегментный детектор «чужой руки» ──
const spliceShare = (p) => ({ host: p.host, intruder: p.intruder, foreignShare: +(1 - p.splice_at / p.n_chunks).toFixed(3), detected: p.detected });
const segment = {
  recallDissimilar: { detected: seg.recall_dissimilar.detected, total: seg.recall_dissimilar.n_pairs },
  recallSimilar: { detected: seg.recall_similar_ceiling.detected, total: seg.recall_similar_ceiling.n_pairs },
  fpr: { falseBooks: seg.fpr_single_author.false_positive_books, totalBooks: seg.fpr_single_author.n_books },
  admixture: seg.recall_at_admixture.map(a => ({ pct: a.admix_pct, foreign: a.foreign_fraction, detected: a.detected })),
  // минимальный обнаруженный процент подмеса (нижний край кривой чувствительности)
  minDetectedAdmixPct: Math.min(...seg.recall_at_admixture.filter((a) => a.detected).map((a) => a.admix_pct)),
  // честный пол обнаружения для ПОХОЖИХ авторов: наименьшая доля чужого текста, при которой шов
  // между близкими (донскими) авторами реально пойман (ниже — за пределом мощности, не тестировано)
  similarDetectionFloorPct: Math.round(Math.min(...seg.recall_similar_ceiling.pairs.map((p) => 1 - p.splice_at / p.n_chunks)) * 100),
  // доля «чужой» (подсаженной) части в каждой склейке = где сидит шов (intruder_chunks / all_chunks)
  splices: {
    dissimilar: seg.recall_dissimilar.pairs.map(spliceShare),
    similar: seg.recall_similar_ceiling.pairs.map(spliceShare),
  },
};
track("segment", "docs/segment_recall.json", "recall_dissimilar/similar_ceiling/fpr/admixture + minDetectedAdmixPct + доля шва в парах");

// ── истинный LOBO (строгий пол на голом признаке) ──
const loboStrict = {
  trueLoboTop1: lobo.top1, trueLoboTop2: lobo.top2, trueLoboTop3: lobo.top3, // char-Delta (голый признак) под истинным LOBO — отдельный строгий ПОЛ
  trueLoboAuthors: lobo.corpus.authors, trueLoboBooks: lobo.corpus.books_lobo,
  styloFullLobo: stylo.acc,          // исторический полный stylo per-book LOBO; не текущая оценка
  proxyTop1: rec.headline.accuracy,  // исторический 5-fold GKF stylo; не текущая оценка
};
track("loboStrict", "docs/lobo_fast.json + final_comparison.csv + model_recall.json", "char-Delta LOBO-пол + полный stylo LOBO + GKF-прокси");

// ── внешний CCAT50 ──
const ccat50 = {
  ensembleTop1: ccat.ensemble_top1, ensembleMacroF1: ccat.ensemble_macro_f1,
  channels: Object.entries(ccat.channels)
    .filter(([k]) => !k.startsWith("АНСАМБЛЬ") && !k.startsWith("headline"))
    .map(([id, v]) => ({ id, top1: v.top1, f1: v.macro_f1 })),
};
track("ccat50", "docs/ccat50.json", "ensemble_top1 + channels");

// ── внешний proza.ru: сравнение методов (наш прогон) — docs/proza_compare.json ──
const prozaC = load("proza_compare.json").comparison_top1_macroF1;
const pk = (sub) => +Object.entries(prozaC).find(([k]) => k.includes(sub))[1][0];
const prozaBench = {
  leader: pk("char-SVM"), word: pk("word-SVM"), equalEnsemble: pk("равновесный"), neuro: pk("ruBERT"),
  ensemble: pk("reliability"), // reliability^6 (test-favoured: степень 6 — лучшая из свипа [2,4,6] НА ТЕСТЕ; веса train-OOF leak-free, степень — нет). leader=char-SVM.
};
track("prozaBench", "docs/proza_compare.json", "char-SVM / word / равновесный / нейро / reliability^6-ансамбль");

// ── атрибуция «Поднятой целины» (негативный контроль) ──
const pod = attrib.podnyataya;
const shareFull = pod.share_full, shareTopic = pod.share_topic || {};
const SHOL = "Михаил Шолохов", KRUK = "Фёдор Крюков";
const disputed = {
  podnyataya: {
    fragments: pod.n_chunks,
    margin: +(shareFull[SHOL] - (shareFull[KRUK] ?? 0)).toFixed(3), // отрыв формулой, не руками
    agreement: pod.agreement_lr_delta,
    candidates: [
      { name: SHOL, full: shareFull[SHOL], topic: shareTopic[SHOL] },
      { name: KRUK, full: shareFull[KRUK], topic: shareTopic[KRUK] ?? 0 },
      { name: "Александр Серафимович", full: shareFull.serafimovich ?? 0, topic: shareTopic.serafimovich ?? 0 },
    ],
  },
};
track("disputed.podnyataya", "docs/sholokhov_attrib.json", "share_full/share_topic; margin=Шолохов−Крюков формулой");

// ── ТУСУР head-to-head (leak premium) ──
const byK = tomskFull.by_K;
const tomsk = {
  theirAcc: +(tomskFinal.tomsk_ref_literary / 100).toFixed(3),
  ourAcc: tomskFinal.results?.char_S10000_K3?.acc ?? null,
  ourStd: tomskFinal.results?.char_S10000_K3?.std ?? null,
  headroom: [
    { k: 12, acc: tomskFinal.results?.char_S10000_K12?.acc ?? null },
    { k: 20, acc: tomskFinal.results?.char_S10000_K20?.acc ?? null },
  ],
  headToHead: Object.entries(byK)
    .map(([k, v]) => ({ k: Number(k), rand: v.fixed_random, grouped: v.fixed_grouped, prem: v.leak_premium_pp }))
    .filter(t => t.grouped !== null && t.prem !== null), // K=273: grouped не посчитался (NaN) → не показываем
};
track("tomsk", "docs/tomsk_final.json + tomsk_full.json", "ref_literary + char_S10000 + by_K leak premium");

// ── исторический PD-only срез из ineligible snapshot — docs/validation_pd.json ──
// Веса равновесного ансамбля не зависят от test, но upstream content leakage
// делает весь сохранённый PD-результат непригодным для текущего claim.
const pdEqual = valPd.channels["АНСАМБЛЬ (равновес.)"];
// худший по узнаваемости автор PD-среза — динамически из per_author_recall (имя резолвится по id в data.js/segdata)
const [worstPdId, worstPdRecall] = Object.entries(valPd.per_author_recall).sort((a, b) => a[1] - b[1])[0];
// книги автора в PD-срезе: число ошибочно отнесённых его книг (top_confusions) / (1 − recall); фоллбэк — счёт книг полного среза
const worstPdMissed = valPd.top_confusions
  .filter((c) => c.startsWith(worstPdId + "->"))
  .reduce((s, c) => s + Number((c.match(/x(\d+)$/) || [0, 1])[1]), 0);
let worstPdBooks = worstPdRecall < 1 && worstPdMissed > 0 ? Math.round(worstPdMissed / (1 - worstPdRecall)) : null;
if (worstPdBooks == null) { const fr = rec.per_author_recall.find((r) => r.id === worstPdId); worstPdBooks = fr ? fr.books : null; }
const benchPd = {
  nAuthors: valPd.n_authors, nBooks: valPd.n_books, chance: +(1 / valPd.n_authors).toFixed(3),
  topTop1: pdEqual.top1, topMacroF1: pdEqual.macro_f1, top3: valPd.ensemble_top3,
  ci: valPd.macro_f1_authorclustered_CI,
  channels: Object.entries(valPd.channels).map(([id, v]) => ({ id, top1: v.top1 })),
  worstRecall: { id: worstPdId, recall: worstPdRecall, books: worstPdBooks },
};
track("benchPd", "docs/validation_pd.json", "PD ансамбль + каналы + худший по узнаваемости автор (id/recall/книги, динамически)");

// ── панель «жанрового регистра»: кто кластеризуется как монарх по служебным словам — docs/royal_register.json ──
const royalReg = load("royal_register.json");
const ROYAL_NAMES_RU = {
  nikolas2: "Николай II", alexander3: "Александр III",
  andrei_vlad: "вел. кн. Андрей Владимирович", konstantin_kr: "вел. кн. Константин Константинович (К. Р.)",
  tolstoy: "Л. Толстой", bunin: "Бунин", chehov: "Чехов", pushkin: "Пушкин",
  suvorin: "Суворин", valuev: "Валуев", kuropatkin: "Куропаткин",
};
const dynastyPanel = {
  permP: royalReg.perm_p, silhouette: royalReg.silhouette, registerGap: royalReg.register_gap,
  reigning: Object.keys(royalReg.reigning || {}).map((n) => ROYAL_NAMES_RU[n] || n),
  grandDukes: Object.keys(royalReg.grand_dukes || {}).map((n) => ROYAL_NAMES_RU[n] || n),
  dukesWriterside: (royalReg.dukes_writerside || []).length,
  dukesTotal: Object.keys(royalReg.grand_dukes || {}).length,
  rows: royalReg.per_author.map((r) => ({
    name: ROYAL_NAMES_RU[r.name] || r.name,
    group: r.group,                              // престольная линия / великий князь / писатель
    period: r.period || "",                      // период дневника (Александр III — наследник, 1880)
    isRoyal: r.group !== "писатель",
    isThrone: r.group === "престольная линия",
    dRoyal: r.d_royal, dWriter: r.d_writer,
    closer: r.closer,                            // монарх / писатель (к какому центроиду ближе)
    axis: +(r.d_royal / (r.d_royal + r.d_writer)).toFixed(3),   // 0 = как монарх, 1 = как писатель
  })),
};
track("dynastyPanel", "docs/royal_register.json", "per_author кластеризация монарх/писатель + perm_p");

// ── тест «одна канцелярия»: одна ли рука писала дневники обоих императоров — docs/nikolai_scribe.json ──
const scr = load("nikolai_scribe.json");
const scribe = {
  selfMin: scr.self_min, selfMax: scr.self_max, selfMedian: scr.self_median,
  dMonarchs: scr.d_monarchs, betweenWritersMedian: scr.between_writers_median,
  ratioToSelf: scr.ratio_to_self_median, oneHandRejected: scr.one_hand_rejected,
};
track("scribe", "docs/nikolai_scribe.json", "внутри-авторская vs Николай-Александр vs между писателями");

// ── кросс-регистр Николая дневник↔письма (SIZE-MATCHED, русский оригинал) — docs/nikolai_crossreg.json ──
const nikCr = load("nikolai_crossreg.json");
const nikolaiCrossreg = {
  controls: nikCr.controls, nikolas: nikCr.nikolas, z: nikCr.z, zAll: nikCr.zAll,
  alexander3: nikCr.alexander3, controlsMedian: nikCr.controlsMedian, nTokens: nikCr.N_tokens,
};
track("nikolaiCrossreg", "docs/nikolai_crossreg.json", "size-matched дневник-письма Николая, русский оригинал");

// ── тест «менялся ли почерк при восшествии»: дневник Николая до/после воцарения — docs/nikolai_accession.json ──
const acc = load("nikolai_accession.json");
const accession = {
  dPrePost: acc.d_prepost, baseline: acc.baseline_max, diffAuthorRef: acc.diff_author_ref,
  ratioBaseline: acc.ratio_to_baseline, ratioDiffAuthor: acc.ratio_to_diff_author,
  dPostLate: acc.d_post_to_late_diary, dPreLate: acc.d_pre_to_late_diary,
  classification: acc.classification, tokensPre: acc.tokens.pre, tokensPost: acc.tokens.post,
};
track("accession", "docs/nikolai_accession.json", "d(до воцарения, после) vs внутри-блоковая baseline");

// ── гетерогенность «Ильф-Петров» (след двух рук?) — docs/ilfpetrov_heterogeneity.json ──
const ilfHet = load("ilfpetrov_heterogeneity.json");
const ilfHeterogeneity = {
  targetSil: ilfHet.targetSil, controlMean: ilfHet.controlMean, controlStd: ilfHet.controlStd,
  z: ilfHet.z, controls: ilfHet.controls,
};
track("ilfHeterogeneity", "docs/ilfpetrov_heterogeneity.json", "силуэт k=2 цели + контроли + z");

// ── тематическая атрибуция Шолохова (топик-инвариантный стиль) — docs/sholokhov_thematic.json ──
const themat = load("sholokhov_thematic.json");
const sholokhovThematic = {
  tihiyDon: themat["«Тихий Дон»"], podnyataya: themat["«Поднятая целина»"], donskie: themat["Донские рассказы Шолохова"],
};
track("sholokhovThematic", "docs/sholokhov_thematic.json", "ТД/ПЦ/Донские рассказы");

// ── рукопись ТД: глубина авторской правки vs классики (оценка через VertexAI) — docs/sholokhov_manuscript.json ──
const ms = load("sholokhov_manuscript.json");
const MS_NAMES = { sholokhov: "Шолохов · «Тихий Дон»", tolstoy: "Толстой", dostoevsky: "Достоевский", pushkin: "Пушкин" };
const sholokhovManuscript = {
  rows: ["sholokhov", "tolstoy", "dostoevsky", "pushkin"].map((k) => ({
    name: MS_NAMES[k], isTarget: k === "sholokhov", n: ms.per_author[k].n,
    mean: ms.per_author[k].mean, max: ms.per_author[k].max,
    structFrac: ms.per_author[k].struct_frac, schemes: ms.per_author[k].schemes_frac,
    copyErr: ms.per_author[k].copyist_err_frac,
  })),
  test: {
    shMean: ms.sholokhov_vs_controls.sholokhov_mean, shN: ms.sholokhov_vs_controls.sholokhov_n,
    ctrlMean: ms.sholokhov_vs_controls.controls_mean, ctrlN: ms.sholokhov_vs_controls.controls_n,
    diff: ms.sholokhov_vs_controls.mean_diff, cohenD: ms.sholokhov_vs_controls.cohen_d,
    p: ms.sholokhov_vs_controls.p_value, significant: ms.sholokhov_vs_controls.significant_05,
  },
};
track("sholokhovManuscript", "docs/sholokhov_manuscript.json", "глубина правки Шолохов vs Толстой/Достоевский/Пушкин");

// ── консистентность Шолохова (within-author силуэт) — docs/consistency.json ──
const cons = load("consistency.json");
const CONS_NAMES = { chehov: "Чехов", bunin: "Бунин", dostoevsky: "Достоевский", tolstoy: "Толстой",
  turgenev: "Тургенев", tolstoy_an: "А.Н. Толстой", saltykov: "Салтыков", sholohov: "ШОЛОХОВ", leskov: "Лесков", prutkov: "Прутков (коллектив)" };
const consistency = {
  sholokhovSil: cons.sholokhov_silhouette, sholokhovRank: cons.sholokhov_rank, nPanel: cons.n_panel,
  sholokhovSelf: (cons.per_author.find((p) => p.author === "sholohov") || {}).self_rate,
  singleRange: cons.indisputable_range, singleMedian: cons.indisputable_median,
  prutkov: +cons.collective_prutkov.toFixed(3),
  prutkovRatio: +(cons.collective_prutkov / cons.sholokhov_silhouette).toFixed(1),
  scale: cons.per_author.filter((p) => CONS_NAMES[p.author]).map((p) => ({
    a: CONS_NAMES[p.author], v: p.within_silhouette,
    hi: p.author === "sholohov", coll: p.author === "prutkov",
  })),
};
track("consistency", "docs/consistency.json", "within-author силуэт + per_author панель");

// якорный экстрактор числа из прозы прогона; null при несовпадении → coverage-гейт упадёт (защита от тихого дрейфа)
const grab = (str, re) => { const m = String(str).match(re); return m ? Number(m[1].replace(",", ".")) : null; };

// ── много рук Шолохова: отделимость книг + скрытый позитив — multiple_hands.json + hidden_positive.json ──
const mh = load("multiple_hands.json");
const hp = load("hidden_positive.json");
const smh = load("sholokhov_multihand.json"); // supervised pairwise-AV: z=−5.26 против «много рук»
const pf = mh.power_limit_fake_similar;
const MH_AUT_RU = { krukov: "Крюков", bunin: "Бунин", dostoevsky: "Достоевский", gorky: "Горький", serafimovich: "Серафимович", platonov: "Платонов" };
const multihands = {
  groupsToSelf: mh.sholokhov_groups_to_self,
  groups: Object.entries(mh.sholokhov_groups).map(([g, o]) => ({ g, to: "Шолохов", frac: o.frac_self })),
  fakeDifferentCaught: mh.fake_groups_to_self,
  // внутр. отделимость книг (ВЫШЕ = разнороднее/смесь): контроли pairwise + FAKE/Шолохов из power_limit
  sep: [
    { a: "Достоевский (1 автор)", v: mh.pairwise.controls.dostoevsky },
    { a: "Тургенев", v: mh.pairwise.controls.turgenev },
    { a: "Толстой", v: mh.pairwise.controls.tolstoy },
    { a: "FAKE: похожие донские (Крюков+Сераф.+Севский)", v: pf.fake_similar_don, similar: true },
    { a: "Бунин (1 автор)", v: mh.pairwise.controls.bunin },
    { a: "ШОЛОХОВ", v: pf.sholokhov, hi: true },
    { a: "FAKE: разные (Набоков+Лесков+Салтыков)", v: pf.fake_different, fake: true },
  ],
  sholokhovSep: pf.sholokhov, fakeSimilar: pf.fake_similar_don, fakeDifferent: pf.fake_different,
  hiddenPositive: {
    // спайк-кривая: чистый Шолохов frac_self ~0.76, каждые +25% похожего (Крюков) ≈ −0.15; флаг при ~50%
    spikeKrukov: hp.with_coauthor_in_panel.krukov.series.map((s) => ({ r: Math.round(s.r * 100), fs: s.frac_self })),
    flagThreshold: Math.round(hp.flag_threshold_similar.coauthor_in_panel * 100),
    // калибровка реальных групп: frac_self → эквивалент подмеса похожей руки
    calib: [
      { g: "ранние", pct: Math.round(hp.calibration_equiv_admixture["ранние"] * 100) },
      { g: "Поднятая целина", pct: Math.round(hp.calibration_equiv_admixture["ПоднятаяЦелина"] * 100) },
      { g: "Тихий Дон", pct: Math.round(hp.calibration_equiv_admixture["ТихийДон"] * 100) },
      { g: "война", pct: Math.round(hp.calibration_equiv_admixture["война"] * 100) },
    ],
    // дискриминатор: КУДА утекают не-свои фрагменты (концентрация = гострайтер; рассыпание = широкий одиночка)
    discriminator: [{ g: "война", key: "война" }, { g: "Тихий Дон", key: "ТихийДон" }].map(({ g, key }) => {
      const d = hp.real_group_distribution[key];
      return { g, top: d.non_self_top.map(([a, v]) => [MH_AUT_RU[a] || a, v]), donShare: d.frac_don_of_nonself };
    }),
    runNoise: grab(hp.run_noise_caveat, /±\s*(\d\.\d+)/), // "дрожит ~±0.04 между прогонами" -> 0.04
  },
  // РЕШАЮЩИЙ довод против «много литнегров» (supervised pairwise-AV, author-disjoint, equal-token 3000 слов):
  // multi_hand_score Шолохова 0.387 — на уровне одиночек (neg 0.399), ДАЛЕКО от псевдонимной смеси (pos 0.698);
  // z_vs_pseudobyline=−5.26 (~5σ), AUC 0.964 [0.902,1.0], permutation p=0.0001.
  avMultiHand: {
    zPseudo: smh.sholokhov.z_vs_pseudobyline,     // -5.26
    zSingles: smh.sholokhov.z_vs_singles,          // -0.1 (Шолохов ≈ одиночки)
    score: smh.sholokhov.multi_hand_score,          // 0.387
    auc: smh.calibration.auc,                        // 0.964
    aucCi: smh.calibration.auc_ci95,                 // [0.902, 1.0]
    permP: smh.calibration.permutation_p,             // 0.0001
    nPos: smh.calibration.n_pos, nNeg: smh.calibration.n_neg,   // 60 / 15
    negMean: smh.calibration.neg_mean, posMean: smh.calibration.pos_mean,  // 0.399 / 0.698
  },
};
track("multihands", "docs/multiple_hands.json + hidden_positive.json + sholokhov_multihand.json", "отделимость книг + спайк-кривая скрытого позитива + runNoise + pairwise-AV z=−5.26");

// ── Ильф и Петров: соло-различимость + open-set таймлайн «12 стульев» — ilf_vs_petrov.json + ilfpetrov_timeline.json ──
const ivp = load("ilf_vs_petrov.json");
const ipTlAll = load("ilfpetrov_timeline.json");
const ipTl = ipTlAll["двенадцать_стульев"];
const ipGold = ipTlAll["золотой_телёнок"];
// закрытое сравнение только 4 подозреваемых (дуэт/Булгаков/Катаев/Олеша) — docs/disputed_ilfpetrov.json
const dispIP = load("disputed_ilfpetrov.json");
const closedShare = (novel) => { const s = dispIP.novels[novel].restricted_suspects_share;
  return { ipShare: s["ilf-petrov"], bulgakov: s.bulgakov, kataev: s.kataev, olesha: s.olesha }; };
const IP_NAME_FIX = { kuprin: "Куприн", korolenko: "Короленко" };
const topF = (tf) => tf.slice(0, 3).map(([a, v]) => [IP_NAME_FIX[a] || a, v]);
const ilfPetrov = {
  solo: {
    ilfWords: ivp.ilf_words, petrovWords: ivp.petrov_words,
    soloAuc: ivp.solo_separability_auc, fwAuc: ivp.solo_fw_only_auc,
    // проекция спорных романов на ось Ильф/Петров: доля чанков, отнесённых к Петрову
    projP12: +ivp.projection["12 стульев"].frac_petrov_chunks.toFixed(2),
    projPgt: +ivp.projection["Золотой телёнок"].frac_petrov_chunks.toFixed(2),
  },
  dvenadtsat: {
    nChunks: ipTl.n_chunks, foreign: ipTl.foreign, ipShare: ipTl.ip_share, bulgakovShare: ipTl.bulgakov_share,
    nForeign: ipTl.n_foreign_segments,  // отрезков к ближайшим внешним авторам (для контраст-ряда)
    topForeign: topF(ipTl.top_foreign),
    timeline: ipTl.timeline,
    closed: closedShare("двенадцать_стульев"),  // круг сужен до 4 подозреваемых → доля каждого
  },
  gold: {  // «Золотой телёнок» — собственная карта авторства (вторая книга дилогии)
    nChunks: ipGold.n_chunks, foreign: ipGold.foreign, ipShare: ipGold.ip_share, bulgakovShare: ipGold.bulgakov_share,
    nForeign: ipGold.n_foreign_segments,
    topForeign: topF(ipGold.top_foreign),
    timeline: ipGold.timeline,
    closed: closedShare("золотой_телёнок"),
  },
};
track("ilfPetrov", "docs/ilf_vs_petrov.json + ilfpetrov_timeline.json + disputed_ilfpetrov.json", "соло-AUC + проекция + open-set таймлайн обеих книг дилогии + закрытое сравнение 4 подозреваемых");

// ── казусы авторства на чистом dependency — cases_attribution + more_cases + grin_control + hyp_tests ──
const ca = load("cases_attribution.json");
const mc = load("more_cases.json");
const grin = load("grin_control.json");
const ht = load("hyp_tests.json");
const mapBulg = (o) => ({ ipRef: o.ip_ref, buRef: o.bu_ref, mid: o.mid, dilogy: o.dilogy,
  b12: o.dilogy_books["двенадцать_стульев"], gold: o.dilogy_books["золотой_телёнок"] });
const ta = ca.tolstoy_an;
const TA_SIL = [["А.Н.Толстой", "А. Н. Толстой", true], ["turgenev", "Тургенев"], ["tolstoy", "Лев Толстой"], ["bunin", "Бунин"], ["dostoevsky", "Достоевский"]];
const sp = ht.serafimovich_potok, pl = ht.platonov;
const cases = {
  bulgakov: { dep: mapBulg(ca.bulgakov_dependency), ens: mapBulg(ca.bulgakov_ensemble) },
  tolstoyAn: {
    sil: TA_SIL.map(([k, label, hi]) => ({ a: label, v: ta.within_silhouette[k], ...(hi ? { hi: true } : {}) })),
    nSelf: ta.n_self, nBooks: ta.n_books, confusedLev: ta.confused_with_lev,
  },
  chapaev: {
    nearest: "Фурманов", pFurmanov: mc.chapaev.chapaev_P_furmanov, internalSil: mc.chapaev.chapaev_internal_silhouette,
    otherSil: [["мятеж", "Мятеж"], ["в_восемнадцатом_году", "В восемнадцатом году"], ["красный_десант", "Красный десант"]]
      .map(([k, b]) => ({ b, v: +mc.chapaev.chapaev_vs_other_internal_sil[k].toFixed(3) })),
  },
  prutkov: { sil: +mc.prutkov.within_silhouette.toFixed(3), nTexts: mc.prutkov.n_texts },
  controls: [["Пильняк", mc.controls["Пильняк"]], ["Бабель", mc.controls["Бабель"]], ["Фурманов", mc.controls["Фурманов"]], ["Грин", grin.grin]]
    .map(([a, o]) => ({ a, v: o.silhouette ?? o.within_silhouette, self: `${o.n_self}/${o.n_books}` })),
  serafimovich: { pVsGorky: sp.P_serafimovich_vs_gorky, nearest: "Серафимович",
    internalSil: sp.internal_silhouette, dGorky: +sp.dists.gorky.toFixed(2), dSelf: +sp.dists["serafimovich(ранний)"].toFixed(1) },
  platonov: { chevengurSil: +pl.chevengur.internal_silhouette.toFixed(3), kotlovanSil: +pl.kotlovan.internal_silhouette.toFixed(3),
    pChevengur: pl.chevengur.P_platonov, pKotlovan: pl.kotlovan.P_platonov },
  donSchool: { sevskyKrukovAuc: ht.don_school.sevsky_vs_krukov },
};
track("cases", "docs/cases_attribution.json + more_cases.json + grin_control.json + hyp_tests.json", "казусы на dependency: Булгаков/АНТолстой/Чапаев/Прутков/Серафимович/Платонов/донская школа");

// ── РИГОР Шолохова: агрегат прогонов rigor3/5/6/7/8/9/10/11/12 + homogeneity/downsample/
//    clean_attribution/td_candidates/dsp/feature_audit2/genre_matched_lr/hyp_tests2/embedding_robustness/audit ──
const homog = load("sholokhov_homogeneity.json");
const ibAuc = homog.inter_book_auc;
const dsamp = load("attrib_downsample.json");
const fa = load("feature_audit2.json");
const cleanA = load("clean_attribution.json");
const dep = cleanA.dependency, syn = cleanA.clean_syntactic;
const tdc = load("td_candidates.json");
const dsp = load("dsp_attribution.json");
const gml = load("genre_matched_lr.json");
const r3s = load("sholokhov_rigor3.json").strict;
const r5 = load("sholokhov_rigor5.json");
const r6 = load("sholokhov_rigor6.json");
const r7 = load("sholokhov_rigor7.json");
const r8 = load("sholokhov_rigor8.json");
const r9 = load("sholokhov_rigor9.json").p7_panel;
const r10 = load("sholokhov_rigor10.json");
const r11 = load("sholokhov_rigor11.json");
const r12 = load("sholokhov_rigor12.json");
const shOpenset = load("sholokhov_openset.json");   // open-set 48-way + инъекция аутсайдера + блочная перестановка ТД-1
const shVerify = load("sholokhov_verify.json");     // 3-4-е семейства: unmasking + imposters на ТД (топик-чувствительны, контроль честности)
const slob = load("sholokhov_lobo.json"); // нециркулярный disputed-TD LOBO: градиент 0.455→0.017, p=0.0001, 4/4→Шолохову
const ser = load("hyp_tests2.json").serafimovich_editor_td;
const emb = load("embedding_robustness.json");
const crossGenre = load("audit_genre_crossauthor.json").TOPIC_INVARIANT_strict.cross_author_genre_auc;
const rnd = (v, d) => Math.round(v * 10 ** d + 1e-9) / 10 ** d; // half-up с поправкой на float-погрешность
const FEAT_RU = { dependency: "синтакс. связи (dependency)", pos_ngrams: "POS-n-граммы", syntax: "синтаксис (17 призн.)",
  "char_ngrams (baseline)": "символьные n-граммы", function_words: "служебные слова", morphology: "морфология",
  punctuation_ngrams: "пунктуация", length_dist: "распред. длин", "DSP (суффиксы)": "DSP (суффиксы)" };
const TD_LABEL = { "Шолохов-Дон(ранний)": "Шолохов-Дон (ранний)" };
const BK = { tihiy_don_1: "Тихий Дон кн.1", tihiy_don_2: "кн.2", tihiy_don_3: "кн.3", tihiy_don_4: "кн.4", podnyataya_celina_1: "Поднятая целина" };
const BK_TD = { tihiy_don_1: "Тихий Дон кн.1", tihiy_don_2: "кн.2", tihiy_don_3: "кн.3", tihiy_don_4: "кн.4" };
const ABK = { tihiy_don_1: "ТД-1", tihiy_don_2: "ТД-2", tihiy_don_3: "ТД-3", tihiy_don_4: "ТД-4" };
const BK_FULL = { aleshkino_serdce: "Алёшкино сердце", batraki: "Батраки", chuzhaya_krov: "Чужая кровь", nauka_nenavisti: "Наука ненависти",
  oni_srazhalis: "Они сражались за Родину", podnyataya_celina_2: "Поднятая целина кн.2", sudba_cheloveka: "Судьба человека",
  tihiy_don_1: "Тихий Дон кн.1", tihiy_don_2: "Тихий Дон кн.2", tihiy_don_3: "Тихий Дон кн.3", tihiy_don_4: "Тихий Дон кн.4", podnyataya_celina_1: "Поднятая целина кн.1" };
const NEAR = { sholohov_rest: "Шолохов", bulgakov: "Булгаков", krukov: "Крюков" };
const caBooks = (h) => [["кн.1", "tihiy_don_1"], ["кн.2", "tihiy_don_2"], ["кн.3", "tihiy_don_3"], ["кн.4", "tihiy_don_4"]].map(([b, k]) => ({ book: b, p: h.td_books[k] }));
// имена для открытого режима (48 авторов) + инъекции аутсайдера — читаемые, без слагов в тексте секции
const OPENSET_NAMES = { tolstoy_an: "А.Н. Толстой", platonov: "Платонов", sholohov: "Шолохов", krukov: "Крюков", serafimovich: "Серафимович" };
// таблица долей «Тихого Дона»/«Поднятой целины» к Шолохову (полная модель / со сниженной лексикой / согласие LR·Delta)
const ATTRIB_LABELS = [["tihiy_don_1", "Тихий Дон кн.1"], ["tihiy_don_2", "кн.2"], ["tihiy_don_3", "кн.3"], ["tihiy_don_4", "кн.4"], ["podnyataya", "Поднятая целина"], ["tihiy_don_all", "Тихий Дон (всё)"]];
const cx = r11.period_cluster_crosstab;
const rigor = {
  homFloor: ibAuc.floor[0], homSholohov: ibAuc.sholohov_median, homCeil: ibAuc.ceiling[0], homCtrlMed: ibAuc.control_median,
  homCtrls: [["Бунин", "bunin"], ["Горький", "gorky"], ["Тургенев", "turgenev"], ["Достоевский", "dostoevsky"]].map(([a, k]) => ({ a, auc: ibAuc.controls[k][0] })),
  homTdInternal: homog.tihiy_don_internal_auc[0], homNStay: homog.n_stay, homNWorks: homog.n_works,
  bcTdDiffMed: r10["Тихий Дон"].diff_median, bcTdCiLo: r10["Тихий Дон"].diff_ci_bookclustered[0], bcTdCiHi: r10["Тихий Дон"].diff_ci_bookclustered[1],
  bcTdFracPos: r10["Тихий Дон"].frac_pos, bcTdExcl0: r10["Тихий Дон"].ci_excludes_0, bcTdNbooks: r10["Тихий Дон"].n_books_target,
  bcPcFracPos: r10["Поднятая целина"].frac_pos,
  dispRank: r9.rank, dispPanelN: r9.n_panel + 1, dispPct: r9.percentile, // +1: панель контролей + сам Шолохов
  cxEarly0: cx.crosstab.early.cl0_pct, cxTd0: cx.crosstab.td.cl0_pct, cxPc1: cx.crosstab.pc.cl1_pct, cxWar1: cx.crosstab.war.cl1_pct,
  cxSilObs: cx.silhouette_obs, cxSilNull: rnd(r11.silhouette_balanced.gauss_p95, 3),
  cxAriNovel: cx.ari_novel_vs_nonnovel, cxAriTd: cx.ari_td_vs_rest, cxAriDonskoy: cx.ari_donskoy_vs_soviet,
  circLate: r12.circularity.late_ref, circKr: r12.circularity.kr_ref, circMid: r12.circularity.mid,
  circEarly: r12.circularity.early_to_late, circTd: r12.circularity.td_to_late, circPc1: r12.circularity.pc1,
  silCtrl: [["Шолохов", "sholohov", true], ["Тургенев", "turgenev"], ["Достоевский", "dostoevsky"], ["Бунин", "bunin"], ["Горький", "gorky"]]
    .map(([a, k, hi]) => ({ a, v: rnd(r12.silhouette_oneauthor_control[k], 3), ...(hi ? { hi: true } : {}) })),
  tdLoboAttributed: slob.td_attributed_to_sholokhov,   // ЕДИНЫЙ источник вердикта = docs/sholokhov_lobo.json: "3/4" тома ТД → Шолохову (ТД-1 по большинству кусков уходит Крюкову)
  tdLoboP: r12.test_registry.confirmatory.permutation_p,            // перестановка по чанкам: доля «чужих» чанков ТД-1 vs донской FPR-нуль (НЕ тренд по томам)
  tdLoboBlockP: shOpenset.td1_block_permutation.block_perm_p,       // блочная перестановка (чанки внутри книги связаны) — честнее точечной
  tdLoboSurvives: r12.test_registry.confirmatory["survives_0.05"],
  tdExploratoryN: r12.test_registry.exploratory.length,             // направленные наблюдения (доли бутстрепов/дескриптивы)
  gmlrTdShFull: gml.tihiy_don.full["Михаил Шолохов"], gmlrTdKrFull: gml.tihiy_don.full["Фёдор Крюков"],
  gmlrTdShTopic: gml.tihiy_don.topic["Михаил Шолохов"], gmlrPcKrFull: gml.podnyataya.full["Фёдор Крюков"],
  floorRandom: r8.noise_floor_random, crossGenreAuc: crossGenre, earlyPoolN: r8.early_pool_n, krFicN: r8.kr_fiction_n,
  tdDonSh: r8.works[0].d_shol_don, tdDonShCi: r8.works[0].d_shol_don_ci, tdDonKr: r8.works[0].d_krukov_don, tdDonKrCi: r8.works[0].d_krukov_don_ci, tdFracSh: r8.works[0].frac_closer_sholohov,
  pcDonSh: r8.works[1].d_shol_don, pcDonKr: r8.works[1].d_krukov_don, pcFracSh: r8.works[1].frac_closer_sholohov,
  dsNmin: dsamp.N_min, dsTdFullMed: dsamp.td_full_median, dsTdFullLo: dsamp.td_full_range[0], dsTdFullHi: dsamp.td_full_range[1], dsTdTopic: dsamp.td_topic_median,
  noiseFloor: r7.noise_floor,
  tdNearestPost1930: r7.td_nearest_post1930, tdNotKrukov: r7.td_not_krukov, nTd: r7.n_td,
  antiCirc: r7.p2_anticircular.map((x) => ({ book: ABK[x.book], post: x.d_post1930, early: x.d_early_don, kr: x.d_krukov })),
  embRobustConfigs: emb.n_configs, embRobustN: Array.isArray(emb.all_sholohov_configs) ? emb.all_sholohov_configs.length : emb.all_sholohov_configs,
  embZmin: Math.min(...emb.grid.map((g) => g.disp_z)), embZmax: Math.max(...emb.grid.map((g) => g.disp_z)),
  symBoot: r6.p5p6_symmetric_boot.map((x) => ({ book: BK[x.book], frac: x.frac_closer_sholohov })),
  genreMatched: r6.p6_genre_matched.map((x) => ({ book: BK_TD[x.book], dSh: x.d_undisp, dKr: x.d_krukov_don })),
  candKrukovOverlap: r5.p6_candidate_boot.map((x) => ({ book: x.book, frac: x.frac_closer_krukov })),
  sepFloor: r5.p9_separability.anchor_floor_halfTD1, sepCeiling: r5.p9_separability.anchor_ceiling_TD1_vs_dostoevsky,
  dispSholohov: r3s.dispersion_sholohov, dispControl: r3s.control_mean, dispControlStd: r3s.control_std,
  b2Stay: r3s.b2_stay, b2N: r3s.b2_n,
  perBook: r3s.b2.map((x) => ({ book: BK_FULL[x.book], nearest: NEAR[x.nearest], stays: x.stays })),
  power: r3s.power_frac_krukov.map((x) => ({ k: x.k, frac: x.frac_krukov_nn })), powerDetectK: r3s.power_detect_k,
  serafEdShDon: ser.distances["Шолохов-Дон(ранний)"], serafEdSeraf: rnd(ser.distances["Серафимович"], 2), serafEdKrukov: rnd(ser.distances["Крюков"], 2), serafEdP: ser.P_sholokhov_vs_serafimovich,
  tdCandDist: tdc.td_full.order.map((a) => ({ a: TD_LABEL[a] || a, d: tdc.td_full.dists[a], ...(a === tdc.td_full.nearest ? { self: true } : {}) })),
  tdCandGm: Object.entries(tdc.gm_vs_sholokhov).sort((x, y) => x[1] - y[1]).map(([a, p]) => ({ a, p })),
  dspAuthorAuc: dsp.dsp_author_auc, dspGenreAucWithin: dsp.dsp_genre_auc_within_author,
  dspGmShRef: dsp.gm_lr_holdout.sh_ref, dspGmKrRef: dsp.gm_lr_holdout.kr_ref, dspGmMid: dsp.gm_lr_holdout.mid, dspGmPc: dsp.gm_lr_holdout.podnyataya, dspGmTd: dsp.gm_lr_holdout.tihiy_don,
  faWarAuthors: fa.war_authors.length, faRuralAuthors: fa.rural_authors.length,
  fa2: fa.ranked.map((rr) => ({ feat: FEAT_RU[rr.feat], author: rr.author_auc, genreXA: rr.genre_xauthor_auc, idi: rr.idiolect })),
  caDepShRef: dep.holdout.sh_ref, caDepKrRef: dep.holdout.kr_ref, caDepMid: dep.holdout.mid, caDepTd: dep.holdout.td, caDepPc: dep.holdout.pc,
  caDepTdBooks: caBooks(dep.holdout), caDepCentTdMed: dep.centroid["ТД"].med, caDepCentTdFracPos: rnd(dep.centroid["ТД"].frac_pos, 2),
  caEnsShRef: syn.holdout.sh_ref, caEnsKrRef: syn.holdout.kr_ref, caEnsMid: syn.holdout.mid, caEnsTd: syn.holdout.td, caEnsPc: syn.holdout.pc,
  caEnsTdBooks: caBooks(syn.holdout), caEnsCentTdMed: syn.centroid["ТД"].med, caEnsCentTdCiLo: syn.centroid["ТД"].ci[0], caEnsCentTdCiHi: syn.centroid["ТД"].ci[1],
  caEnsCentTdFracPos: rnd(syn.centroid["ТД"].frac_pos, 2), caEnsPcCentMed: syn.centroid["ПЦ"].med,
  // НЕЦИРКУЛЯРНЫЙ disputed-TD LOBO (sholokhov_lobo.json): все спорные + Don-control вне обучения за один
  // ретрейн; якорь solo-in-train = rodinka/zherebenok/batraki. 3/4 тома → Шолохову (ТД-1 по большинству
  // кусков уходит Крюкову); градиент чужой доли ТД-1 0.595 → ТД-4 0.035 значим выше FPR-нуля (permutation p=0.0001).
  loboTd: {
    gradient: slob.disputed_td.map((b) => ({ book: ({ tihiy_don_1: "ТД кн.1", tihiy_don_2: "ТД кн.2", tihiy_don_3: "ТД кн.3", tihiy_don_4: "ТД кн.4" })[b.book] || b.book, ff: b.foreign_fraction, ci: b.ff_ci95, segs: b.n_foreign_segments })),
    td1PermP: slob.td1_vs_null_permutation_p,        // 0.0001
    donFpr: slob.fpr_null_don_control.pooled_ff,      // 0.0 (контрольные донские → 100% Шолохову)
    tdAttrib: slob.td_attributed_to_sholokhov,         // "3/4" (ТД-1 по большинству кусков → Крюкову; тот же источник, что tdLoboAttributed)
    valid: slob.procedure_valid,                        // true
    minAdmix: slob.lobo_power_curve_krukov.min_detectable_admixture_pct,  // 25
  },
  // открытый режим (48 авторов, без короткого списка) — sholokhov_openset.json: поздние тома → Шолохову
  // уверенно, ранние — нет (регистр-сосед А.Н. Толстой тянет эпику к романисту-эпику)
  openSetTd: (() => {
    const byVol = (v) => shOpenset.openset_td_full_argmax.find((x) => x.vol === v);
    const td1 = byVol("tihiy_don_1"), td4 = byVol("tihiy_don_4"), td1Top = td1.top[0];
    return { td4Share: td4.sholokhov_share_open, td1Share: td1.sholokhov_share_open,
      td1TopName: OPENSET_NAMES[td1Top[0]] || td1Top[0], td1TopShare: td1Top[1] };
  })(),
  // контроль «список не засасывает чужого»: вброшенный аутсайдер уходит к себе, не к Шолохову
  platonovInject: {
    name: OPENSET_NAMES[shOpenset.openset_injection.outsider] || shOpenset.openset_injection.outsider,
    selfShare: shOpenset.openset_injection.open_to_self, toSholokhovShare: shOpenset.openset_injection.open_to_sholokhov,
  },
  // верификация «этот ли автор вообще» (unmasking + imposters) на ТД: 0/4, но метод смещён (контроль честности)
  verifTd: { tdAttributed: shVerify.imposters_verified_sholokhov, unmaskAttributed: shVerify.unmasking_td_to_sholokhov, biased: true },
  // таблица долей ТД/ПЦ к Шолохову (Тест №1): full/topic/agree из docs/sholokhov_attrib.json (не литералы в segdata)
  attribTable: ATTRIB_LABELS.map(([k, book]) => ({ book, full: attrib[k].share_full[SHOL], topic: attrib[k].share_topic[SHOL], agree: attrib[k].agreement_lr_delta })),
};
track("rigor", "docs/sholokhov_rigor3/5/6/7/8/9/10/11/12 + homogeneity/downsample/clean_attribution/td_candidates/dsp/feature_audit2/genre_matched_lr/hyp_tests2/embedding_robustness/audit_genre_crossauthor + sholokhov_lobo/openset/verify/attrib.json", "агрегат ригор-прогонов Шолохова + нециркулярный disputed-TD LOBO + открытый режим/верификация/таблица атрибуции");

// ── кейс Николая II: вспомог. тесты — nikolas2_authorship.json. Часть значений живёт ТОЛЬКО в прозе
//    прогона (removal_curve, kamerfurier и т.п.) → тянем якорным regex; null при несовпадении → coverage-гейт упадёт. ──
const nik = load("nikolas2_authorship.json");
const insTest = nik.inserted_entries_test, kam = nik.kamerfurier_test;
const kamOv = grab(kam.overlap_4grams, /\((\d\.\d+)%\)/); // "5 из 987 (0.5%)" -> 0.5
// кросс-жанровый перенос атрибуции (обучение на прозе → дневники/письма) — docs/crossgenre_recall.json;
// калибровочная точность (LOO) — nikolas2_authorship.json.positive_control_LOO_acc.
const cgr = load("crossgenre_recall.json");
const cgAgg = cgr.aggregate;
const nikCrossGenre = {
  docTop1: cgAgg.all.doc_top1,                                          // целый документ: top-1 = 0.885
  chunkTop1: cgAgg.all.chunk_top1,                                       // отдельные куски: 0.58
  diaryDocs: cgAgg.diary.n_docs,                                          // дневников = 10
  diaryCorrect: Math.round(cgAgg.diary.doc_top1 * cgAgg.diary.n_docs),   // верно на дневниках = 10/10
  letterDocs: cgAgg.letters.n_docs,                                       // писем = 16
  letterCorrect: Math.round(cgAgg.letters.doc_top1 * cgAgg.letters.n_docs), // верно на письмах = 13/16
  candidates: cgr.train.n_authors,                                        // кандидатов в модели = 47
  calibrationAcc: nik.positive_control_LOO_acc,                           // контрольная точность калибровки = 0.74
  calibrationAuthors: cgr.test.n_authors,                                 // на пяти авторах с обоими жанрами
};
const nikolaiCase = {
  zWithRoyal: nik.dynasty_control.nikolas2_z_with_royal_control_CORRECTED, // чистое поле (CORRECTED, size-matched)
  insHetZ: insTest.internal_heterogeneity_z,                              // чистое поле
  officialVal: grab(nik.official_recalibration.control, /=\s*(\d\.\d+)/),  // "= 0.0270" -> 0.027
  insBimodality: grab(insTest.bimodality, /^(\d+)%/),                     // "0% записей" -> 0
  insRemovalFrom: grab(insTest.removal_curve, /(\d\.\d+)→/),              // "0.267->0.231"
  insRemovalTo: grab(insTest.removal_curve, /→(\d\.\d+)/),
  insRemovalFracPct: grab(insTest.removal_curve, /удаление\s*(\d+)%/),    // "удаление 70% самых далёких записей" -> 70
  insNormTo: grab(insTest.removal_curve, /нормы\s*(\d\.\d+)/),            // "нормы 0.07"
  kamOverlap: kamOv === null ? null : kamOv / 100,                        // 0.5% -> 0.005
  kamControl: grab(kam.control, /=\s*(\d+)/),                             // "...=0" -> 0
  controlsN: grab(nik.controls_total, /^(\d+)/),                         // "11 личных-документных" -> 11
  crAucAll: nik.cross_register_auc.all_function_words,                   // AUC узнавания автора между жанрами: все служебные слова
  crAucInvariant: nik.cross_register_auc.topic_invariant_LOAO,          // ...тематически инвариантные (leave-one-author-out)
  crossGenre: nikCrossGenre,                                             // перенос атрибуции проза→дневники/письма (crossgenre_recall.json)
};
track("nikolaiCase", "docs/nikolas2_authorship.json + crossgenre_recall.json", "вспомог. тесты Николая (часть значений якорным regex) + crossGenre-перенос");

// ── прямая проверка «одиозных» записей дневника (охота/стрельба) — docs/nikolai_cats.json ──
const ncats = load("nikolai_cats.json");
const nikolaiCats = {
  share: +(ncats.cat_windows / ncats.n_windows * 100).toFixed(1), // % окон «одиозных» записей (12.8)
  p: +ncats.dist_to_letters.p.toFixed(2),                         // к письмам не ближе остального дневника (0.41)
};
track("nikolaiCats", "docs/nikolai_cats.json", "доля окон одиозных записей + p близости к письмам");

// ── чистота: ни одно обязательное число не должно быть null ──
// ── Честный протокол: метрический урок + карта режимов (где метод делит руки, где честно отказывает) ──
const kol = load("cases/kolokol_herzen_ogaryov.json");
const sov = load("cases/sovremennik.json");
const nek = load("cases/nekrasov_panaeva.json");
const pet = load("cases/dostoevsky_petersburg_chronicle.json");
const ch15 = load("cases/chekhonte_15_micro.json");
const cal = load("cases/calibration_reference.json");
const chp = load("cases/nekrasov_panaeva_chapters.json");
const workBalancedAudit = load("cases/work_balanced_audit/summary.json");
const kolokolWorkAudit = load("cases/work_balanced_audit/custom/kolokol_herzen_ogaryov.work_balanced.json");
const sovremennikWorkAudit = load("cases/work_balanced_audit/custom/sovremennik.work_balanced.json");
const nekrasovWorkAudit = load("cases/work_balanced_audit/custom/nekrasov_panaeva.work_balanced.json");
const workAuditCase = (stem) => workBalancedAudit.cases.find((row) => row.source_spec.endsWith(`/${stem}.yaml`));
const workAuditPrimary = (stem) => workAuditCase(stem).work_balanced;
const petersburgWorkAudit = workAuditPrimary("petersburg_nn_fourway_fw_2000");
const chekhonteWorkAudit = workAuditPrimary("chekhonte_budilnik_sredi_milykh");
const limits = {
  threshold: 0.80,
  // (1) метрический урок: один корпус, две единицы голоса. work — один удержанный текст = один
  //     голос (корректно); chunk — каждый отрывок = голос (длинные работы весят больше по числу
  //     кусков, эффективный размер выборки иной; в этих кейсах доля ниже). Единица голоса определяет вердикт.
  metric: [
    { id: "sovremennik", label: "Современник: две школы критиков",
      work: sovremennikWorkAudit.axis_school_radical_vs_aesthete.fw.macro_recall, chunk: sovremennikWorkAudit.axis_school_radical_vs_aesthete.fw.chunk_weighted_recall },
    { id: "kolokol", label: "Колокол: Герцен и Огарёв",
      work: kolokolWorkAudit.fw_only.macro_recall, chunk: kolokolWorkAudit.fw_only.chunk_weighted_recall },
    { id: "nekrasov", label: "Некрасов и Панаева (служебные слова)",
      work: nekrasovWorkAudit.fw_only.macro_recall, chunk: nekrasovWorkAudit.fw_only.chunk_weighted_recall },
  ],
  // (2) калибровочная линейка чтения: две опорные точки на известных авторах
  calibration: {
    easy: { macro: cal.pairs.easy_diff_author_register.macro_recall, cos: cal.pairs.easy_diff_author_register.cross_author_centroid_cos, label: cal.pairs.easy_diff_author_register.label },
    medium: { macro: cal.pairs.medium_diff_author_same_era.macro_recall, cos: cal.pairs.medium_diff_author_same_era.cross_author_centroid_cos, label: cal.pairs.medium_diff_author_same_era.label },
  },
  // (3) карта режимов — где метод делит руки (work-level ≥ 0.80, значимая перестановка)
  separates: [
    { id: "sovremennik", title: "Современник: две школы критиков",
      question: "Различает ли метод критиков-радикалов и эстетиков?",
      candidates: "радикалы (Чернышевский, Добролюбов) ↔ эстетики (Дружинин, Анненков, Боткин)",
      macro: sovremennikWorkAudit.axis_school_radical_vs_aesthete.fw.macro_recall, perm: sovremennikWorkAudit.axis_school_radical_vs_aesthete.fw.perm_p,
      cos: sovremennikWorkAudit.axis_school_radical_vs_aesthete.fw.centroid_cos,
      caveat: "Это разделение конкретных критиков двух школ, не перенос на «школу как класс». Боткин — одна работа (~14k слов), входит только в эстетиков. Значительная доля атрибуций «Современника» закрыта по гонорарным ведомостям (Боград), поэтому часть прогона — калибровка." },
    { id: "petersburg", title: "Фельетоны Ф.Д. «Петербургской летописи»",
      question: "Уходят ли фельетоны 1847 года к Достоевскому?",
      candidates: "Достоевский (публицистика), Соллогуб, Плещеев, Панаев",
      macro: petersburgWorkAudit.work_macro_recall, perm: petersburgWorkAudit.permutation_p,
      caveat: "Панель остаётся выше порога после равного веса работ, но спорный Н.Н. расколот 1:1 между публицистикой Достоевского и Панаевым. Это выполнимость панели, не положительная атрибуция Н.Н." },
  ],
  // (4) карта режимов — где метод честно отказывает, и почему именно
  limitsCases: [
    { id: "nekrasov", title: "Романы Некрасова и Панаевой",
      question: "Делятся соавторы по личному почерку или по теме?",
      candidates: "Некрасов, Панаева",
      reason: "автор ≡ тема",
      fwMacro: nekrasovWorkAudit.fw_only.macro_recall, fwPerm: nekrasovWorkAudit.fw_only.work_level_permutation_p,
      char3Macro: nekrasovWorkAudit.fw_char3.macro_recall, char3Perm: nekrasovWorkAudit.fw_char3.work_level_permutation_p,
      cos: nekrasovWorkAudit.fw_only.cross_author_centroid_cos, kappa: chp.summary.char3_fw_kappa },
    { id: "pair", title: "Современник: учитель и ученик",
      question: "Делится ли пара Чернышевский ↔ Добролюбов внутри одной школы?",
      candidates: "Чернышевский, Добролюбов",
      reason: "сросшиеся руки у границы",
      macro: sovremennikWorkAudit.axis_pair_chernyshevsky_vs_dobrolyubov.fw.macro_recall, perm: sovremennikWorkAudit.axis_pair_chernyshevsky_vs_dobrolyubov.fw.perm_p,
      cos: sovremennikWorkAudit.axis_pair_chernyshevsky_vs_dobrolyubov.fw.centroid_cos },
    { id: "kolokol", title: "Передовые «Колокола»",
      question: "Делятся ли руки Герцена и Огарёва?",
      candidates: "Герцен, Огарёв",
      reason: "сросшиеся руки · train-side audit",
      macro: kolokolWorkAudit.fw_only.macro_recall,
      perm: kolokolWorkAudit.fw_only.work_level_permutation_p,
      cos: kolokolWorkAudit.fw_only.cross_author_centroid_cos },
    { id: "chekhonte", title: "«Среди милых москвичей»",
      question: "Однородна ли чеховская подборка из колонки «Будильника»?",
      candidates: "Чехонте, Билибин, Лейкин, Александр Чехов",
      reason: "панель у случайности · source-check доступен",
      macro: chekhonteWorkAudit.work_macro_recall,
      perm: chekhonteWorkAudit.permutation_p,
      status: chekhonteWorkAudit.status },
  ],
};
track("limits", "docs/cases/{kolokol,sovremennik,nekrasov_panaeva,nekrasov_panaeva_chapters,dostoevsky_petersburg_chronicle,chekhonte_15_micro,calibration_reference,work_balanced_audit/{summary.json,custom/{kolokol_herzen_ogaryov,sovremennik,nekrasov_panaeva}.work_balanced.json}}", "post-audit метрический урок + карта режимов честного протокола");

// ── Taras Bulba hardened case: паспорта gate-first слоя ──
const tarasStrict = load("cases/taras_hardened/passports/taras_bulba_additions_strict_fw_2000.passport.json");
const tarasLoose = load("cases/taras_hardened/passports/taras_bulba_additions_loose_fw_2000.passport.json");
const tarasSpeech = load("cases/taras_hardened/passports/taras_bulba_tovarishchestvo_fw_2000.passport.json");
const tarasExtStrict = load("cases/taras_hardened/passports/taras_bulba_additions_strict_extended_fw_2000.passport.json");
const tarasExtLoose = load("cases/taras_hardened/passports/taras_bulba_additions_loose_extended_fw_2000.passport.json");
const tarasManifest = load("cases/taras_hardened/target_manifest.json");
// Панели подозреваемых и контроли (hardened v2, 2026-07-07)
const tarasAnnBinary = load("cases/taras_hardened/passports/taras_bulba_additions_strict_annenkov_binary_fw_2000.passport.json");
const tarasSusV2Strict = load("cases/taras_hardened/passports/taras_bulba_additions_strict_suspects_v2_fw_2000.passport.json");
const tarasSusV2Loose = load("cases/taras_hardened/passports/taras_bulba_additions_loose_suspects_v2_fw_2000.passport.json");
const tarasSameStrict = load("cases/taras_hardened/passports/taras_bulba_additions_strict_sameperiod_fw_2000.passport.json");
const tarasSameLoose = load("cases/taras_hardened/passports/taras_bulba_additions_loose_sameperiod_fw_2000.passport.json");
const tarasCtlAnn = load("cases/taras_hardened/passports/taras_control_annenkov_holdout_v2_fw_2000.passport.json");
const tarasCtlShinel = load("cases/taras_hardened/passports/taras_control_shinel_holdout_v2_fw_2000.passport.json");
const tarasCtl1835 = load("cases/taras_hardened/passports/taras_control_gogol1835_base_v2_fw_2000.passport.json");
const tarasCtlTurg = load("cases/taras_hardened/passports/taras_control_turgenev_holdout_v2_fw_2000.passport.json");
const tarasTopicStrict = load("cases/taras_hardened/passports/taras_bulba_additions_strict_topic_cossack_fw_2000.passport.json");
const tarasTopic1835 = load("cases/taras_hardened/passports/taras_control_gogol1835_base_topic_fw_2000.passport.json");
const tarasSomovBinary = load("cases/taras_hardened/passports/taras_bulba_additions_strict_somov_binary_fw_2000.passport.json");
const tarasPeriod = load("cases/taras_hardened/passports/taras_bulba_additions_strict_period_fw_2000.passport.json");
const tarasProkopovich = load("cases/taras_hardened/passports/taras_bulba_diag_prokopovich_letters_fw_2000.passport.json");
const tarasDelta = load("cases/taras_hardened/reports/delta_replication.json");
const tarasExtraction = load("cases/taras_hardened/reports/extraction_audit.json");
const tarasWorkAudit = workBalancedAudit;
const tarasDeltaAudit = load("cases/work_balanced_audit/custom/taras_delta_full_refit_work_balanced.json");
const firstGate = (p) => p.gates?.[0] || {};
const firstAttr = (p) => p.attributions?.[0] || {};
const caseRow = (p) => {
  const g = firstGate(p), a = firstAttr(p);
  return {
    id: p.case_id, title: p.title, target: p.target_description, status: p.status,
    confidence: p.confidence, score: p.evidence_score, gatePass: p.gate_pass,
    verdict: p.verdict, claim: p.claim, limitations: p.limitations, failureModes: p.failure_modes,
    gate: g.work_macro_recall, chunkRecall: g.chunk_weighted_recall, p: g.permutation_p,
    works: g.n_works, chunks: a.n_chunks ?? 0, targetChunks: p.data?.target_chunks ?? 0,
    top: a.top || "", second: a.second || "", margin: a.margin ?? 0, ci: a.margin_ci95 || [],
    winnerShare: a.winner_share || {}, perChunk: a.per_chunk_winners || {},
    worksPerAuthor: p.data?.works_per_author || {},
  };
};
const deltaFw = tarasDelta.panels.suspects.modes.delta_fw;
const tarasCase = {
  hypothesis: tarasStrict.hypothesis,
  claim: "После work-balanced аудита уникальная рука крупных добавлений не установлена: multi-candidate панели не проходят gate, а валидные бинарные панели и Delta-режимы меняют направление вместе с составом кандидатов и признаков.",
  headline: [caseRow(tarasSusV2Strict), caseRow(tarasSusV2Loose)],
  basePanel: [caseRow(tarasStrict), caseRow(tarasLoose)],
  annenkovBinary: caseRow(tarasAnnBinary),
  samePeriod: [caseRow(tarasSameStrict), caseRow(tarasSameLoose)],
  controls: [caseRow(tarasCtlAnn), caseRow(tarasCtlShinel), caseRow(tarasCtl1835), caseRow(tarasCtlTurg)],
  topic: {
    additions: caseRow(tarasTopicStrict),
    base1835: caseRow(tarasTopic1835),
    somovBinary: caseRow(tarasSomovBinary),
    note: "Служебные слова несут и регистр повествования: на казачьей панели добавления читаются как сомовская сказовая манера (малый запас), при этом бесспорный текст 1835 на той же панели идёт к Гоголю.",
  },
  period: caseRow(tarasPeriod),
  prokopovich: caseRow(tarasProkopovich),
  speech: caseRow(tarasSpeech),
  extended: [caseRow(tarasExtStrict), caseRow(tarasExtLoose)],
  replication: {
    deltaFwGate: deltaFw.gate.work_macro_recall,
    deltaFwP: deltaFw.gate.permutation_p,
    deltaFwStrictTop: deltaFw.targets.strict_additions.top,
    deltaFwStrictShare: deltaFw.targets.strict_additions.winner_share,
    deltaFwLooseTop: deltaFw.targets.loose_additions.top,
    deltaFwLooseShare: deltaFw.targets.loose_additions.winner_share,
    deltaFwBaseTop: deltaFw.targets.gogol1835_base_control.top,
    deltaFwBaseShare: deltaFw.targets.gogol1835_base_control.winner_share,
  },
  postAudit: {
    date: tarasWorkAudit.date,
    status: tarasWorkAudit.status,
    centroidWeighting: tarasWorkAudit.estimands.work_balanced,
    suspectsStrict: workAuditPrimary("taras_bulba_additions_strict_suspects_v2_fw_2000"),
    samePeriodStrict: workAuditPrimary("taras_bulba_additions_strict_sameperiod_fw_2000"),
    topicStrict: workAuditPrimary("taras_bulba_additions_strict_topic_cossack_fw_2000"),
    annenkovBinary: workAuditPrimary("taras_bulba_additions_strict_annenkov_binary_fw_2000"),
    somovBinary: workAuditPrimary("taras_bulba_additions_strict_somov_binary_fw_2000"),
    delta: {
      suspectsFw: tarasDeltaAudit.panels.suspects.modes.delta_fw,
      suspectsMfw: tarasDeltaAudit.panels.suspects.modes.delta_mfw,
      somovBinaryFw: tarasDeltaAudit.panels.somov_binary.modes.delta_fw,
      somovBinaryMfw: tarasDeltaAudit.panels.somov_binary.modes.delta_mfw,
      topicFw: tarasDeltaAudit.panels.topic.modes.delta_fw,
      topicMfw: tarasDeltaAudit.panels.topic.modes.delta_mfw,
    },
    conclusion: "Многоавторные панели не проходят порог 0.80; две валидные бинарные панели дают разные ближайшие руки. Уникальная атрибуция крупных добавлений не установлена.",
  },
  extraction: {
    strictIn1842: tarasExtraction.containment.strict_in_1842,
    strictIn1835: tarasExtraction.containment.strict_in_1835,
    looseIn1842: tarasExtraction.containment.loose_in_1842,
    looseIn1835: tarasExtraction.containment.loose_in_1835,
    allChecksPass: tarasExtraction.all_checks_pass,
    edition1842Words: tarasExtraction.word_counts.edition_1842,
    edition1835Words: tarasExtraction.word_counts.edition_1835,
  },
  manifest: {
    strictWords: tarasManifest.targets.additions1842_strict.words,
    looseWords: tarasManifest.targets.additions1842_loose.words,
    speechWords: tarasManifest.targets.tovarishchestvo_speech.words,
    strictSha: tarasManifest.targets.additions1842_strict.sha256,
    looseSha: tarasManifest.targets.additions1842_loose.sha256,
    speechSha: tarasManifest.targets.tovarishchestvo_speech.sha256,
    rawPolicy: tarasManifest.raw_policy,
  },
};
track("tarasCase", "docs/cases/{taras_hardened/{passports/*.json,target_manifest.json,reports/{delta_replication,extraction_audit}.json},work_balanced_audit/{summary.json,custom/taras_delta_full_refit_work_balanced.json}}", "Taras: historical hardened artifacts plus 2026-07-11 work-balanced and Delta full-refit adversarial audits");

// ── воспроизводимость gate-кейсов: перепрогон бит-в-бит — docs/repro_gates.json ──
const rg = load("repro_gates.json");
const rgLongest = rg.gates.slice().sort((a, b) => b.seconds - a.seconds)[0];
const repro = {
  gatesTotal: rg.gates.length,
  gatesBitExact: rg.gates.filter((g) => g.bit_exact).length,
  longestGateName: rgLongest.label_ru, longestGateSeconds: rgLongest.seconds,
};
track("repro", "docs/repro_gates.json", "перепрогон gate-кейсов бит-в-бит + самый долгий gate");

const data = { corpus, headline, models, channels, authorRecall, confusions, segment, loboStrict, ccat50, disputed, tomsk, benchPd, sholokhovThematic, ilfHeterogeneity, dynastyPanel, scribe, accession, sholokhovManuscript, nikolaiCrossreg, consistency, prozaBench, multihands, ilfPetrov, cases, rigor, nikolaiCase, nikolaiCats, limits, tarasCase, repro };
const holes = [];
(function scan(o, path) {
  if (o === null || o === undefined) { holes.push(path); return; } // undefined ловим ДО JSON.stringify (он молча выкидывает ключ)
  if (typeof o === "number" && !Number.isFinite(o)) { holes.push(path + " (NaN/Infinity)"); return; } // нефинитные ловим ДО stringify (он пишет их как null)
  if (Array.isArray(o)) return o.forEach((x, i) => scan(x, `${path}[${i}]`));
  if (typeof o === "object") return Object.entries(o).forEach(([k, v]) => scan(v, `${path}.${k}`));
})(data, "");
const NULLABLE_PATHS = new Set([
  ".headline.macroF1CI",
  ".tomsk.headroom[0].acc",
  ".tomsk.headroom[1].acc",
]);
models.forEach((model, index) => {
  // The stylo reference has no comparison p-value; historical non-stylo
  // baselines did not record calibration ECE.  No subtree is exempted.
  NULLABLE_PATHS.add(
    model.id === "stylo" ? `.models[${index}].p` : `.models[${index}].ece`
  );
});
const realHoles = holes.filter(h => !NULLABLE_PATHS.has(h));
if (realHoles.length) { console.error("COVERAGE-ГЕЙТ: пустые числа без источника:\n  " + realHoles.join("\n  ")); process.exit(1); }

if (!existsSync(OUT)) mkdirSync(OUT, { recursive: true });
const siteDataBytes = Buffer.from(JSON.stringify(data, null, 2) + "\n", "utf-8");
writeFileSync(join(OUT, "site-data.json"), siteDataBytes);
const generatorPath = "scripts/gen-site-data.mjs";
const provenance = {
  schema: "stylo.site_generation_provenance.v1",
  generator: {
    path: generatorPath,
    sha256: sha256(readFileSync(join(ROOT, generatorPath))),
  },
  sources: [...consumedSources].sort().map((source) => ({
    path: source,
    sha256: sha256(readFileSync(join(ROOT, source))),
  })),
  outputs: [{
    path: "site/src/generated/site-data.json",
    sha256: sha256(siteDataBytes),
  }],
  entries: manifest,
};
writeFileSync(join(OUT, "manifest.json"), JSON.stringify(provenance, null, 2) + "\n", "utf-8");
console.log(`✓ site/src/generated/site-data.json (${manifest.length} ключей, источники в manifest.json)`);
console.log(`  headline stylo=${headline.accuracy} ансамбль=${headline.ensembleTop1} | корпус ${corpus.research.authors}/${corpus.research.books} | бенчмарк ${corpus.benchmark.authors}/${corpus.benchmark.books} | CCAT50 ${ccat50.ensembleTop1}`);
