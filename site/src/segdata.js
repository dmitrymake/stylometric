import D from "./generated/site-data.json";
import { AUTHOR_NAMES } from "./data.js";

// имя худшего по узнаваемости автора PD-среза резолвится по id из генератора (не литерал).
const worstPdName = AUTHOR_NAMES[D.benchPd.worstRecall.id] || D.benchPd.worstRecall.id;

// Тематическая атрибуция (топик-инвариантный стиль) — docs/sholokhov_thematic.json (генератор).
// Полный пул кандидатов; topic = тематически-инвариантный стиль (тема замаскирована).
export const SHOLOKHOV = { thematic: D.sholokhovThematic, manuscript: D.sholokhovManuscript };

// Каталог пределов (честный протокол): 5 gate-кейсов + калибровочная шкала косинуса.
export const LIMITS = D.limits;

// Кейс «Тараса Бульбы»: паспорта gate-first слоя + manifest target hashes.
export const TARAS = D.tarasCase;

// Воспроизводимость gate-кейсов: перепрогон бит-в-бит (docs/repro_gates.json через генератор).
export const REPRO = D.repro;

export const ILF_PETROV = {
  candidates: ["Илья Ильф и Евгений Петров", "Михаил Булгаков", "Валентин Катаев", "Юрий Олеша"],
  // Числа -> генератор (D.ilfPetrov) из ilf_vs_petrov.json + ilfpetrov_timeline.json.
  dvenadtsat: { label: "«12 стульев»", ...D.ilfPetrov.dvenadtsat,
    nSegments: 0, // сегментная атрибуция: 0 чужих отрезков (отдельный прогон, не таймлайн)
  },
  gold: { label: "«Золотой телёнок»", ...D.ilfPetrov.gold }, // своя карта: nForeign 2, к Булгакову 0
  solo: { ...D.ilfPetrov.solo,
    ilfSource: "Записные книжки (1925–1937), сольный Ильф",
    petrovSource: "военная публицистика + мемуар об Ильфе (1940–42), сольный Петров" },
  heterogeneity: D.ilfHeterogeneity, // docs/ilfpetrov_heterogeneity.json (генератор)
};

// Исторический бенчмарк (docs/validation.json): book-id grouping и fit внутри
// фолда соблюдены, но весь snapshot позднее признан ineligible из-за cross-work
// content leakage. Числа ниже — диагностика, не текущие claims.
// Исторический PD-only срез (НЕ «редистрибутируемый» — у Гумилёва/Пильняка охрана в РФ продлена
// после реабилитации, тексты докачиваются по манифесту) — числа из docs/validation_pd.json (генератор).
// Метки каналов — здесь; точность каждого канала из прогона.
const PD_CH_LABELS = {
  "АНСАМБЛЬ (равновес.)": "АНСАМБЛЬ (равновесный)",
  "char (2-5)": "char-ngrams (2-5)",
  function_words: "function words",
  "syntax (dep+pos+syn)": "синтаксис dep+pos+syn",
  dependency: "dependency (чистый идиолект)",
  morphology: "morphology",
  "word (1-2)": "word-ngrams (1-2)",
  "DSP (suffixes)": "словообразование (DSP)",
};
export const BENCH = {
  pdOnly: true,
  publicationStatus: "historical_ineligible_corpus_snapshot",
  nAuthors: D.benchPd.nAuthors, nBooks: D.benchPd.nBooks, chance: D.benchPd.chance,
  topMacroF1: D.benchPd.topMacroF1, topTop1: D.benchPd.topTop1, top3: D.benchPd.top3, ci: D.benchPd.ci,
  // худший по узнаваемости автор PD-среза — динамически из генератора (имя, recall, число книг), не литерал
  lowRecall: [worstPdName],
  worstRecall: { id: D.benchPd.worstRecall.id, name: worstPdName, recall: D.benchPd.worstRecall.recall, books: D.benchPd.worstRecall.books },
  channels: D.benchPd.channels
    .map((c) => ({ c: PD_CH_LABELS[c.id] || c.id, v: c.top1, hi: c.id.startsWith("АНСАМБЛЬ") }))
    .sort((a, b) => b.v - a.v),
  caveat:
    `PD-only (${D.corpus.pd.authors} автора, ансамбль ${D.corpus.pd.ensembleTop1}) и полный ` +
    `(${D.corpus.benchmark.authors} автора, ${D.headline.ensembleTop1}) срезы сохранены как ` +
    `историческая диагностика ineligible corpus snapshot. Их нельзя сравнивать или использовать ` +
    `для текущего claim до content-safe миграции и полного пересчёта.`,
};

// Внешние бенчмарки + сравнение с современными инструментами — docs/ccat50.json, proza_compare.json
export const BENCH_EXT = {
  // Полный стек на стандартном CCAT50 (Reuters, 50 авт., fixed train/test) — docs/ccat50.json (генератор).
  // Каналы — реальный Tfidf-словарь (без hash-коллизий); равновесный ансамбль (веса не зависят от теста → leak-free).
  ccat50Ensemble: D.ccat50.ensembleTop1,
  ccat50Published: "0.74–0.78", // внешняя цитата: опубл. диапазон char-n-gram SVM на CCAT50 (обзор Valla 2022), не наш прогон
  ccat50Channels:
    D.ccat50.channels.filter((c) => !c.id.startsWith("summary")).map((c) => `${c.id} ${c.top1}`).join(" · ") +
    ` → равновесный ансамбль ${D.ccat50.ensembleTop1}`,
  // Контекст из Valla (канонический обзор SOTA, та же CCAT50): n-gram-эталон Ngram_A 0.767, BERT_A 0.657.
  // На фикс-сплите мой ансамбль чуть выше n-gram-эталона — паритет в этом классе, не значимое превосходство;
  // абсолютный рекорд 0.832 (syntax-CNN) получен на ДРУГОМ протоколе (60/20/20 + CV), несопоставим и не заявляется.
  ccat50Valla: { ngramA: 0.767, bertA: 0.657, record: 0.832 },
  // Proza.ru (внешний русский, 50 авт) — сравнение методов на ИДЕНТИЧНОМ сплите.
  // Reliability^6-ансамбль: веса ∝ точности каналов на train-OOF (leak-free), НО степень 6 — лучшая из
  // свипа [2,4,6] на тесте (тест-благоприятна, мягкий HARKing) → 0.887, +0.006 над char-SVM 0.881 (шум).
  // Консервативный честный лидер — один char-SVM. Наивный равновесный ансамбль (0.700) НАМНОГО хуже.
  prozaCompare: [
    { m: "char-SVM (консервативный лидер)", v: D.prozaBench.leader, hi: true },
    { m: "reliability^6 (test-favoured diagnostic)", v: D.prozaBench.ensemble },
    { m: "word-SVM (мой)", v: D.prozaBench.word },
    { m: "равновесный ансамбль (наивный)", v: D.prozaBench.equalEnsemble, old: true },
    { m: "ruBERT-tiny2 (pretrained нейро)", v: D.prozaBench.neuro, neuro: true },
  ],
  prozaEnsemble: D.prozaBench.ensemble,
  prozaLeader: D.prozaBench.leader, prozaEqualEnsemble: D.prozaBench.equalEnsemble, prozaNeuro: D.prozaBench.neuro,
};

// Гипотеза «разные авторы»: разные работы Шолохова писали разные люди? — docs/multiple_hands.json, fake_similar.json
// Числа -> генератор из docs/multiple_hands.json + hidden_positive.json (D.multihands).
// В segdata остаётся только нарратив: вердикт, флаги-метки, runNoise (из run_noise_caveat как прозаич. ±0.04).
export const MULTIHANDS = {
  groupsToSelf: D.multihands.groupsToSelf,         // все работы Шолохова → он сам (даже с Крюковым в панели)
  groups: D.multihands.groups,
  fakeDifferentCaught: D.multihands.fakeDifferentCaught, // позитивный контроль (3 разных автора) — метод ловит
  sep: D.multihands.sep,                           // внутр. отделимость книг (ВЫШЕ = разнороднее/смесь)
  sholokhovSep: D.multihands.sholokhovSep, fakeSimilar: D.multihands.fakeSimilar, fakeDifferent: D.multihands.fakeDifferent,
  powerLimit: true, // метод НЕ различает смесь ПОХОЖИХ донских авторов от одиночки
  // Решающий тест (синтетический скрытый позитив): подмешиваем РЕАЛЬНОГО автора к Шолохову,
  // меряем порог обнаружения, калибруем реальные группы, смотрим КУДА утекают не-свои фрагменты.
  hiddenPositive: {
    spikeKrukov: D.multihands.hiddenPositive.spikeKrukov,
    flagThreshold: D.multihands.hiddenPositive.flagThreshold,
    calib: D.multihands.hiddenPositive.calib,
    discriminator: D.multihands.hiddenPositive.discriminator,
    runNoise: D.multihands.hiddenPositive.runNoise,
    scattered: true, // не-свои фрагменты РАССЫПАНЫ, не сконцентрированы → нет концентрированной второй руки
    verdict: "Сильная версия (отдельный/концентрированный чужой автор) не подтверждается: группы → Шолохов, не-свои фрагменты рассыпаны (война: Крюков≈Бунин≈Достоевский; ТД ведёт Горький). НО война (≈45% подмеса-эквив.) чуть ниже порога обнаружения этого теста (~50%) — частичный вклад ПОХОЖЕГО донского соавтора метод исключить не может.",
  },
  // РЕШАЮЩИЙ: supervised pairwise-AV (author-disjoint, equal-token 3000-слов чанки) против 60 псевдонимных
  // смесей: multi_hand_score Шолохова 0.387 ≈ одиночки (0.399), z_vs_pseudobyline=−5.26 (~5σ), AUC 0.964, p=0.0001.
  // Источник: docs/sholokhov_multihand.json.
  avMultiHand: D.multihands.avMultiHand,
};

// Консистентность Шолохова: внутр. разнородность vs неоспоримые одиночки — docs/consistency.json
// CONSISTENCY — числа из генератора (docs/consistency.json), здесь только нарративный label.
export const CONSISTENCY = {
  ...D.consistency,
  moreHeterogeneous: ["Лесков"], // единственный одиночка разнороднее Шолохова (нарратив)
};

// Казусы авторства на ЧИСТОМ признаке dependency (+ансамбль dep+pos+syntax).
// Числа -> генератор (D.cases): cases_attribution + more_cases + grin_control + hyp_tests.
// В segdata остаётся нарратив: жанры А.Н.Толстого, leak-флаг циркулярности, rank Чапаева.
export const CASES = {
  // гипотеза Булгакова: «12 стульев»+«Золотой телёнок» → Ильф-Петров или Булгаков?
  bulgakov: { ...D.cases.bulgakov, leak: false }, // dep/ens из генератора; leak = флаг циркулярности (нарратив)
  // А.Н. Толстой — валидация: держится ли один автор сквозь жанры (НФ/историч./эмиграция)?
  tolstoyAn: { ...D.cases.tolstoyAn,
    genres: "НФ (Аэлита, Гиперболоид) · историч. (Граф Калиостро) · усадебная (Хромой барин) · эмиграция (Эмигранты)" },
  // Фурманов «Чапаев» — автор vs документы (rank=1: Фурманов = nearest)
  chapaev: { ...D.cases.chapaev, rank: 1 },
  // Козьма Прутков — коллектив (разнороден, но конфаунд формы)
  prutkov: D.cases.prutkov,
  // контроли (одиночные авторы, силуэт «цельности»)
  controls: D.cases.controls,
  // Серафимович «Железный поток» — автор vs правка Горького (аналог Островского/Шолохова)
  serafimovich: D.cases.serafimovich,
  // Платонов — калибровочный negative control (идиолект переживает редактуру)
  platonov: D.cases.platonov,
  // донская школа разделима (инфра-контроль для атрибуций, вкл. Шолохова)
  donSchool: D.cases.donSchool,
};


// Авто-сгенерировано из docs/sholokhov_rigor3.json (нейтральный симметричный базис, leak-free).
// Базис: 10/12 (не утечные 12/12), genre-AUC, корректная мощность.
// РИГОР Шолохова: ВСЕ числа -> генератор (D.rigor) из rigor3/5/6/7/8/9/10/11/12 +
// homogeneity/downsample/clean_attribution/td_candidates/dsp/feature_audit2/genre_matched_lr/
// hyp_tests2/embedding_robustness/audit_genre_crossauthor. В segdata — только нарратив.
export const RIGOR = {
  ...D.rigor,
  // единств. нечисловой нарратив: кандидаты без доступной PD-прозы (не тестируемы)
  tdCandUntestable: "Голоушев, Громославский, Цыганков, Родионов — без доступной худож. прозы (PD-источников нет)",
  // per-book доли к Шолохову (full vs сниж.лексика) + согласие LR/Delta (agree) + надёжность.
  // Нужен Sholokhov.jsx «Тест №1». Числа — из генератора (D.rigor.attribTable, docs/sholokhov_attrib.json).
  // reliable — порог отображения (согласие ≥ 0.5 = две модели совпадают по большинству), не результат-число.
  attrib: D.rigor.attribTable.map((r) => ({ ...r, reliable: r.agree >= 0.5 })),
};

// ─────────────────────────── Кейс «дневники Николая II» ───────────────────────────
// Источник: docs/nikolas2_authorship.json. Подлог НЕ подтверждён; аномалия реальна, но
// объяснена тремя проверками (манускрипт, не-копипаст, династический регистр).
export const NIKOLAI = {
  crossReg: {
    // SIZE-MATCHED русский ОРИГИНАЛ (письма матери). Числа из генератора (docs/nikolai_crossreg.json),
    // log/crossreg_sizematched.py: и дневник, и письма каждого автора субсэмплированы до объёма писем
    // Николая с бутстрапом — устраняет раздувание дистанции малой выборкой. Переписка с женой исключена (перевод).
    controls: D.nikolaiCrossreg.controls,
    nikolas: D.nikolaiCrossreg.nikolas, z: D.nikolaiCrossreg.z, zAll: D.nikolaiCrossreg.zAll,
    controlsMedian: D.nikolaiCrossreg.controlsMedian, nTokens: D.nikolaiCrossreg.nTokens,
    translationNote: "Честные ограничения данных по письмам Николая. (1) Его письма жене Александре написаны по-английски — русские издания это перевод (мерят переводчика), поэтому исключены. (2) Единственный свободно доступный русский ОРИГИНАЛ — фрагменты писем матери 1905 года, извлечённые из веб-публикации, где они переплетены с редакторским и авторским текстом; отделить их полностью не удалось (полное чистое издание — Индрик-2017, печатное, в открытом доступе нет). Поэтому конкретная величина z (около 4) НЕНАДЁЖНА и подаётся как ориентир, не как точное число. (3) Что устойчиво: при уравнивании объёмов (size-matched, бутстрап) направление сохраняется — дневник Николая дальше от писем, чем у писателей; и то же видно на ЧИСТОМ русском оригинале Александра III (письма жене Минни, разрыв " + D.nikolaiCrossreg.alexander3.toFixed(2) + "). Величина z чувствительна к переводу и к неуравненным объёмам, поэтому подаётся как ориентир (около 4); устойчивость по частям речи и пунктуации на такой малой выборке оценить нельзя — опускаем.",
  },
  dynasty: { alexander3: D.nikolaiCrossreg.alexander3, zWithRoyal: D.nikolaiCase.zWithRoyal, panel: D.dynastyPanel,
    // Панель регистра (docs/royal_register.json): кто кластеризуется как монарх по служебным словам.
    // Числа — из генератора (D.dynastyPanel), здесь только формулировки.
    royalPanel: {
      emperors: D.dynastyPanel.reigning, grandDukes: D.dynastyPanel.grandDukes,
      silhouette: D.dynastyPanel.silhouette, permP: D.dynastyPanel.permP, registerGap: D.dynastyPanel.registerGap,
      dukesWriterside: D.dynastyPanel.dukesWriterside, dukesTotal: D.dynastyPanel.dukesTotal,
      note: "На служебных словах два дневника главной линии престолонаследия — Николай II (как император) и Александр III (как наследник, 1880) — оказываются взаимно ближайшими и отделяются от прочих; перестановочный тест p=" + D.dynastyPanel.permP.toFixed(3) + " при силуэте " + D.dynastyPanel.silhouette + ". Ключевой контроль на «кровь»: оба великих князя той же династии — Андрей Владимирович и Константин Константинович (К. Р.), чьи дневники нарративны, — идут к писателям, а не к этим двум. Сухой логбук — скорее жанр придворного дневника престольной линии, чем царская кровь. Оговорки честные: точек всего две, p=" + D.dynastyPanel.permP.toFixed(3) + " — это пол перестановочного теста для двух точек, и значимость держится целиком на этой паре, поэтому регистр как общий класс на таком n не установить. Дневник Александра III здесь наследнический (1880): как император он связного дневника не вёл вовсе, так что строго «регистр правящего монарха» данные не доказывают. Наконец, повышенный разрыв дневник↔письма у второго лица главной линии симметрично совместим и с жанром, и с версией о редакторской переработке обеих публикаций.",
    } },
  scribe: D.scribe,   // тест «одна канцелярия»: внутри-авторская vs Николай-Александр vs между писателями
  accession: D.accession,   // тест «менялся ли почерк при восшествии»: до/после воцарения (OCR через VertexAI)
  paleography: "Палеографическая прикидка автографа 1896 года (мультимодальной моделью Gemini 3.1 Pro через VertexAI, не экспертная экспертиза): почерк ВЫГЛЯДИТ как личная скоропись, а не писарский протокольный; на развороте ПОХОЖЕ одна рука, без явных признаков смены; модель не нашла следов переписывания набело. Это эвристика по одной странице, а не заключение почерковеда.",
  official: { name: "Победоносцев", val: D.nikolaiCase.officialVal },
  // Числа -> генератор (D.nikolaiCase) из nikolas2_authorship.json. removalFrom 0.267 — дистанция
  // дневник↔письма на ПОЛНОМ якоре (до size-match); в заголовке кросс-регистра — уравненный 0.25/z≈4.
  inserted: { hetZ: D.nikolaiCase.insHetZ, bimodality: D.nikolaiCase.insBimodality,
    removalFrom: D.nikolaiCase.insRemovalFrom, removalTo: D.nikolaiCase.insRemovalTo, normTo: D.nikolaiCase.insNormTo,
    removalFracPct: D.nikolaiCase.insRemovalFracPct },
  manuscript: {
    page: "запись 19 февраля 1896 (смерть П. А. Черевина)",
    result: "на этой странице печать дословно совпадает с автографом — её не переписывали под чужой стиль",
    caveat: "это единственная сверенная страница, к тому же личная запись (смерть близкого человека); оспариваемые сухие записи 1916 года с автографом не сверялись",
    fond: "ГАРФ, ф. 601 (личный фонд Николая II); скан автографа — на Wikimedia Commons, PD",
  },
  kamerfurier: { overlap: D.nikolaiCase.kamOverlap, control: D.nikolaiCase.kamControl },
  // AUC узнавания автора между жанрами (log/invariant_av.py): все служебные слова -> тематически инвариантные (LOAO).
  // 0.588 в прозе authorship_cases.json — ПОТОЛОК другого протокола (Вырубова), не базовый уровень для 0.794.
  crossRegAuc: { all: D.nikolaiCase.crAucAll, invariant: D.nikolaiCase.crAucInvariant },
  // кросс-жанровый перенос атрибуции (обучение на прозе → дневники/письма) — docs/crossgenre_recall.json (генератор).
  // Числа: целый документ top-1, куски, дневники/письма верно, кандидаты, контрольная точность калибровки.
  crossGenre: D.nikolaiCase.crossGenre,
  cats: D.nikolaiCats, // прямая проверка «одиозных» записей: доля окон + p близости к письмам

  controlsN: D.nikolaiCase.controlsN,
  refs: [
    { cite: "Автограф дневника, запись 19.02.1896 о смерти П. А. Черевина — Wikimedia Commons, PD", url: "https://commons.wikimedia.org/wiki/File:%D0%94%D0%BD%D0%B5%D0%B2%D0%BD%D0%B8%D0%BA_%D0%B8%D0%BC%D0%BF%D0%B5%D1%80%D0%B0%D1%82%D0%BE%D1%80%D0%B0_%D0%9D%D0%B8%D0%BA%D0%BE%D0%BB%D0%B0%D1%8F_II_%D0%B7%D0%B0_1895-1896_%D0%B3%D0%B3..jpg" },
    { cite: "Сканы дневника Николая II — Wikimedia Commons (категория, PD)", url: "https://commons.wikimedia.org/wiki/Category:Diary_of_Nicholas_II" },
    { cite: "Дневники императора Николая II (1894–1918) / отв. ред. С. В. Мироненко. РОССПЭН, 2011–2013 — ГАРФ", url: "https://statearchive.ru/1632" },
    { cite: "Дневники Николая II (выборка, Берлин: «Слово», 1923) — militera.lib.ru", url: "http://militera.lib.ru/db/nikolay-2_02/index.html" },
    { cite: "Дневник Николая II — проект «Прожито» (точка входа, PD)", url: "https://corpus.prozhito.org/person/165" },
    { cite: "Военный дневник вел. кн. Андрея Владимировича (1914–1917), ГАРФ — militera.lib.ru (для панели жанрового регистра)", url: "http://militera.lib.ru/db/romanov_av/index.html" },
    { cite: "Дневник вел. кн. Константина Константиновича (К. Р.), 1877–1915 — az.lib.ru (по изд.: «Искусство», 1998; PD); контроль на «царскую кровь» в панели регистра", url: "http://az.lib.ru/k/kr_k_k/text_1915_iz_dnevnikov.shtml" },
  ],
};
