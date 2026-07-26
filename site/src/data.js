// ════════════════════════════════════════════════════════════════════
//  Тонкий маппинг-слой: числа приходят из site/src/generated/site-data.json
//  (генерируется scripts/gen-site-data.mjs из docs/*.json — выходов прогонов).
//  Здесь только имена строк, порядок, цвет-токены и нарративные подписи —
//  ни одного метрик-литерала. Пересчёт прогона → генератор → числа обновятся.
// ════════════════════════════════════════════════════════════════════
import D from "./generated/site-data.json";
import { CORPUS } from "./corpus.js";

// HEADLINE = продакшен-модель stylo (LogReg по всем блокам фич), полный per-book LOBO.
// authors/books — срез LOBO (43 тестированных автора / 251 книга).
export const HEADLINE = {
  ...D.headline,
  authors: CORPUS.lobo.tested_authors,
  books: CORPUS.lobo.books,
  words: CORPUS.research.words,
};
const ruBooks = (n) => {
  const mod100 = Math.abs(n) % 100;
  const mod10 = mod100 % 10;
  if (mod100 >= 11 && mod100 <= 14) return "книг";
  if (mod10 === 1) return "книга";
  if (mod10 >= 2 && mod10 <= 4) return "книги";
  return "книг";
};

// Финальное сравнение моделей — ИСТИННЫЙ пер-книжный LOBO (final.py/lobo.py → final_comparison.csv).
// p = McNemar по книгам против stylo. p===0 → машинный нуль (≪0.0001).
const MODEL_META = {
  stylo: { name: "stylo (все блоки)", kind: "ours" },
  bow_lr: { name: "Мешок слов + логрег", kind: "baseline" },
  char_cos: { name: "char-3gram косинус", kind: "baseline" },
  "delta:150": { name: "Burrows Delta · 150 MFW", kind: "classic" },
  "delta:300": { name: "Burrows Delta · 300 MFW", kind: "classic" },
  "delta:500": { name: "Burrows Delta · 500 MFW", kind: "classic" },
  "delta_cos:150": { name: "Cosine Delta · 150 частых слов", kind: "classic" },
  "delta_cos:300": { name: "Cosine Delta · 300 частых слов", kind: "classic" },
  "delta_cos:500": { name: "Cosine Delta · 500 частых слов", kind: "classic" },
  majority: { name: "majority (нижняя граница)", kind: "floor" },
};
export const MODELS = D.models
  .map((m) => ({ ...m, ...MODEL_META[m.id] }))
  .sort((a, b) => b.acc - a.acc);

// Худший (наименее значимый, самый большой) McNemar-p среди классических опорных методов
// (Burrows Delta + Cosine Delta): честная верхняя граница для утверждения «stylo обходит классику».
export const WORST_CLASSIC_P = Math.max(
  ...MODELS.filter((m) => m.kind === "classic").map((m) => m.p).filter((p) => p != null),
);

// Вклад блоков признаков: каждый КАНАЛ в одиночку (один SVM, тот же leak-free срез).
// Равновесный ансамбль (leak-free) ≥ лучшего одиночного канала → блоки взаимодополняемы.
const CHANNEL_META = {
  "char (2-5)": { name: "char-n-граммы 2–5", kind: "base" },
  function_words: { name: "служебные слова", kind: "base" },
  "syntax (dep+pos+syn)": { name: "синтаксис (связи + POS + метрики)", kind: "base" },
  "word (1-2)": { name: "мешок слов (1–2)", kind: "bow" },
  dependency: { name: "синтакс. связи (чистый идиолект)", kind: "base" },
  morphology: { name: "морфология", kind: "base" },
  "DSP (suffixes)": { name: "словообразование · суффиксы (DSP)", kind: "weak" },
};
const CHANNEL_ROWS = D.channels.rows
  .map((r) => ({ ...CHANNEL_META[r.id], id: r.id, top1: r.top1, f1: r.f1 }))
  .sort((a, b) => b.top1 - a.top1);
export const CHANNELS = {
  ensembleTop1: D.channels.ensembleTop1,
  ensembleF1: D.channels.ensembleMacroF1,
  rows: CHANNEL_ROWS,
  byId: (id) => CHANNEL_ROWS.find((r) => r.id === id),
};

// Per-author recall (stylo, book-level CV) — числа из docs/model_recall.json (генератор).
// Здесь только отображаемые имена авторов; recall/books приходят из прогона.
export const AUTHOR_NAMES = {
  kataev: "Валентин Катаев", olesha: "Юрий Олеша", prutkov: "Козьма Прутков",
  volcheck: "Дмитрий Волчек", zoshenko: "Михаил Зощенко", leskov: "Николай Лесков",
  gumilev: "Николай Гумилёв", victor_erofeev: "Виктор Ерофеев", nabokov: "Владимир Набоков",
  elizarov: "Михаил Елизаров", sasha_sokolov: "Саша Соколов", sevsky: "Виктор Севский",
  kumov: "Роман Кумов", furmanov: "Дмитрий Фурманов", gorky: "Максим Горький",
  serafimovich: "Александр Серафимович", garshin: "Всеволод Гаршин", prokhanov: "Александр Проханов",
  gazdanov: "Гайто Газданов", mamleev: "Юрий Мамлеев", platonov: "Андрей Платонов",
  radov: "Егор Радов", babel: "Исаак Бабель", kuprin: "Александр Куприн", akunun: "Борис Акунин",
  pilnyak: "Борис Пильняк", sorokin: "Владимир Сорокин", uspensky: "Глеб Успенский",
  andreev: "Леонид Андреев", bulgakov: "Михаил Булгаков", grin: "Александр Грин",
  pushkin: "Александр Пушкин", tolstoy_an: "Алексей Н. Толстой", novikov_priboy: "Алексей Новиков-Прибой",
  korolenko: "Владимир Короленко", saltykov: "Михаил Салтыков-Щедрин", krukov: "Фёдор Крюков",
  tolstoy: "Лев Толстой", dostoevsky: "Фёдор Достоевский", bunin: "Иван Бунин",
  gogol: "Николай Гоголь", turgenev: "Иван Тургенев", chehov: "Антон Чехов",
  sholohov: "Михаил Шолохов",
};
export const AUTHOR_RECALL = D.authorRecall.map((r) => ({
  id: r.id, name: AUTHOR_NAMES[r.id] || r.id, recall: r.recall, books: r.books,
}));

// Атрибуция спорного текста «Поднятая целина» (исследовательский предикт, отдельно
// от бенчмарка). Это негативный контроль: ПЦ — бесспорно Шолохов, и метод это подтверждает.
// Доли фрагментов из docs/sholokhov_attrib.json (podnyataya): full = модель с лексикой,
// topic = тематически-инвариантный стиль. Булгакова среди кандидатов нет — он лишь
// конспирологическая версия, которую данные не подтверждают.
export const DISPUTED = {
  podnyataya: {
    title: "«Поднятая целина»",
    winner: "Михаил Шолохов",
    verdict: "Бесспорный Шолохов — и метод это подтверждает (негативный контроль: проверяем себя на заведомо известном ответе). Ближайший след — Фёдор Крюков, но на топик-инвариантном стиле отпадает и он. Никакой «близости к Булгакову» в данных нет — его в кандидатах просто нет.",
    ...D.disputed.podnyataya, // fragments, margin (Шолохов−Крюков формулой), agreement, candidates
  },
};

// Самые показательные ошибки book-level CV (docs/model_recall.json, генератор) — для графа путаницы.
// Имена и пояснения — здесь; пары и частоты приходят из прогона.
const CONFUSION_NOTES = {
  "prutkov>uspensky": "коллективная маска размывается в очеркистов",
  "gumilev>grin": "поэт в прозе — тонкий профиль",
  "kataev>bulgakov": "одесская школа, мало текста",
  "garshin>sholohov": "единственная ошибка в сторону Шолохова",
  "furmanov>krukov": "донская проза",
};
export const CONFUSIONS = D.confusions.map((c) => ({
  trueAuthor: AUTHOR_NAMES[c.trueId] || c.trueId,
  predicted: AUTHOR_NAMES[c.predId] || c.predId,
  n: c.n,
  note: CONFUSION_NOTES[`${c.trueId}>${c.predId}`],
}));

// Находки валидатора корпуса (docs/corpus_validation.json).
export const CORPUS_FINDINGS = [
  { severity: "warn", title: "Внутриавторские near-duplicate", text: "Прутков (cos 0.95–0.96 между сборниками афоризмов) и Гаршин — пересекающиеся фрагменты. Источник утечки train↔test; учитывается при чтении силуэта коллектива." },
  { severity: "warn", title: `Дисбаланс ${D.corpus.research.imbalanceRatio}×`, text: "Достоевский (≈1.15M слов) против Волошина (≈1.8k). Accuracy завышается «толстыми» классами — поэтому рядом приводим macro-F1. Интервал macro-F1 по авторам отозван; интервал accuracy сохранён." },
  { severity: "info", title: "1-книжные авторы вне оценки", text: "Гончаров, Григорович, Решетников, Волошин — книгу нельзя оставить, не лишив автора профиля; в book-level CV не участвуют." },
  { severity: "info", title: "Дуэт и дневники — отдельно", text: "Ильф-Петров (соавторство) и дневники Николая II (не проза) исключены из headline-бенчмарка и разбираются отдельными кейсами." },
];

// Детектор «чужой руки»: rolling-атрибуция чанков + поиск контигуальных «чужих» сегментов,
// с ОБЯЗАТЕЛЬНЫМ нуль-контролем. Позитив-контроль = синтетическая склейка двух авторов
// (детектор обязан увидеть точку склейки); негатив-контроль = одноавторские книги (ложных
// «чужих» сегментов быть не должно). Воспроизводимо: docs/segment_recall.json.
export const SEGMENT = {
  ...D.segment, // recallDissimilar, recallSimilar, fpr, admixture, similarDetectionFloorPct — из docs/segment_recall.json
  ceiling:
    `Потолок мощности: стилистически ПОХОЖИХ авторов (донская школа) при малой доле подмеса детектор не разделяет — ` +
    `наименьшая пойманная доля чужого текста около ${D.segment.similarDetectionFloorPct}%; честный предел, решающий в кейсе «Тихого Дона».`,
};

// Истинный LOBO для голого признака (trueLoboBooks, каждая книга держится вне по очереди)
// vs 5-fold book-level прокси; канонический stylo LOBO берётся из final_comparison.csv.
export const LOBO_STRICT = {
  ...D.loboStrict, // trueLoboTop1/2/3, trueLoboAuthors/Books, proxyTop1 — из docs/lobo_fast.json + headline
  trueLoboMethod: "Cosine-Delta, char_wb 3–5 (HashingVectorizer без обучаемого словаря → leak-free по построению)",
  proxyCv: "5-fold book-level StratifiedGroupKFold (прокси для sweep/ablation)",
  note:
    `Канонический HEADLINE — полный per-book LOBO (leave-one-book-out; пересчётов: ${HEADLINE.books}, каждая книга по очереди отложена; final.py/lobo.py → final_comparison.csv): top-1 ${D.loboStrict.styloFullLobo} (${HEADLINE.authors} тестированных автора / ${HEADLINE.books} ${ruBooks(HEADLINE.books)}). StratifiedGroupKFold(5) (top-1 ${D.loboStrict.proxyTop1}) — лишь быстрый прокси для sweep/ablation, не headline; сходится с LOBO в пределах шума. Голый char-косинус (Cosine-Delta) под LOBO даёт строгий ПОЛ на признаке без обучаемого словаря (${D.loboStrict.trueLoboTop1}) — отдельный нижний ориентир.`,
};

// Заметка о воспроизводимости демо-кода группы ТУСУР (Romanov/Kurtukova/Fedotova/Shelupanov).
// Полный корпус — по запросу; демо (300 авторов) + код открыты. docs/tomsk_final.json.
export const TOMSK = {
  theirAcc: D.tomsk.theirAcc, // их заявленное 80.4% (без группировки по книге)
  // Источник 80.4% (SVM+GA, 50 авторов, лит. тексты): группа ТУСУР (Факультет безопасности, каф. КИБЭВС).
  ref: {
    cite: "Fedotova, Romanov, Kurtukova, Shelupanov · Future Internet 2022, 14(1), 4",
    url: "https://doi.org/10.3390/fi14010004",
    group: "группа ТУСУР (Факультет безопасности, каф. КИБЭВС), Томск",
    baseCite: "Базовая серия: Romanov, Kurtukova, Shelupanov, Fedotova, Goncharov · Future Internet 2021, 13(1), 3",
    baseUrl: "https://doi.org/10.3390/fi13010003",
  },
  // Их код и демо-данные открыты; полный корпус — по запросу у авторов.
  data: {
    repo: "github.com/afedotowaa/authorship_attribution",
    repoUrl: "https://github.com/afedotowaa/authorship_attribution",
    note: "Открыты ноутбуки (SVM+GA, BERT, fastText) и демо-корпус (300 авторов, лит.); классика бралась с lib.ru — моего же источника. Полный корпус статьи — по запросу у авторов.",
  },
  // Честный book-grouped пересчёт на ИХ открытых данных и признаках — docs/tomsk_full.json.
  headToHead: {
    note: "Опубликовано 80.4% (50 авторов, SVM+GA). В их открытом демо-коде нет группировки train/test по книге — куски одного романа учатся и тестируются вместе (GroupKFold/GroupShuffleSplit в репозитории нет). На их же данных и их признаках честный book-grouped пересчёт даёт около 56% на 50 авторах (0.561 против 0.654 при случайном сплите); разрыв в 8-11 п.п. между сплитами держится на всех масштабах. Это не пересчёт их непубличного полного корпуса, а указание на риск утечки в открытом коде.",
    table: D.tomsk.headToHead, // {k,rand,grouped,prem} из docs/tomsk_full.json by_K (K=273 с NaN отфильтрован)
    prCite: "PR с book-grouped оценкой и багфиксами (merged)",
    prUrl: "https://github.com/afedotowaa/authorship_attribution/pull/1",
  },
};

export const FEATURES = [
  { id: "char", name: "char-n-граммы 3–5", note: "+ topic-bleaching (маскировка POS)", kind: "base" },
  { id: "fw", name: "функциональные слова", note: "MFW-300 / список 405 слов", kind: "base" },
  { id: "syntax", name: "синтаксис · 17 метрик", note: "длины, POS-доли, TTR/Hapax/Yule, прямая речь", kind: "base" },
  { id: "pos", name: "POS-n-граммы", note: "синтаксический скелет, тематически нейтрален", kind: "base" },
  { id: "punct", name: "пунктуационные n-граммы", note: "авторская привычка знаков", kind: "base" },
  { id: "dep", name: "dependency", note: "типы связей, глубина дерева", kind: "base" },
  { id: "morph", name: "морфология", note: "падеж/время/вид (spaCy morph)", kind: "base" },
  { id: "len", name: "распределения длин", note: "слов (Менденхолл) и предложений", kind: "base" },
  { id: "emb", name: "ruBERT-эмбеддинги", note: "off — риск утечки темы", kind: "opt" },
];
