// Генератор README.md из ЕДИНОГО источника чисел (тех же docs/*.json + site-data.json, что и сайт).
// Проза — шаблон ниже; КАЖДОЕ метрик-число подставляется из данных, чтобы README не дрейфовал.
// Запуск: node scripts/gen-readme.mjs  (без зависимостей). Прогонять после run_benchmark/gen-site-data.
import fs from "node:fs";
import crypto from "node:crypto";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const read = (p) => fs.readFileSync(path.join(ROOT, p), "utf8");
const jload = (p) => JSON.parse(read(p));
const sha256 = (p) => crypto.createHash("sha256").update(
  fs.readFileSync(path.join(ROOT, p))
).digest("hex");

const sd = jload("site/src/generated/site-data.json");
const cv = jload("docs/corpus_validation.json");
const historicalSnapshot = jload("docs/p0_baseline_snapshot.json");
for (const p of ["docs/validation.json", "docs/validation_pd.json"]) {
  const registered = historicalSnapshot?.artifacts?.sha256?.[p];
  const actual = sha256(p);
  if (typeof registered !== "string" || registered !== actual) {
    throw new Error(
      `${p} is a frozen historical README input: P0 SHA256 ${registered} != ${actual}`
    );
  }
}
const val = jload("docs/validation.json");
const valPd = jload("docs/validation_pd.json");
const ccat = jload("docs/ccat50.json");
const proza = jload("docs/proza_compare.json");
const styloCI = jload("docs/stylo_lobo_authorci.json");   // author-clustered CI headline (acc + macro-F1)
const luar = jload("docs/luar_proza.json").results_top1_macroF1;
const vemb = jload("docs/vertex_embedding_proza.json");   // gemini-embedding-001 заморож. (через VertexAI)
const ft = jload("docs/neuro_finetune_proza.json");        // ruBERT-tiny2, дообученный
const crob = jload("docs/cluster_robust_stylo_vs_bow.json");
const osp = jload("docs/openset_passport.json");
const shOpen = jload("docs/sholokhov_openset.json");
const dcos = jload("docs/delta_cosine_lobo.json");        // cosine/книжный Delta в headline-LOBO
const nik = jload("docs/nikolas2_authorship.json");
const cg = jload("docs/crossgenre_recall.json");          // кросс-жанровый перенос: train проза → test дневники/письма
const prov = jload("docs/provenance_probe.json");         // адверсариальная проба «предскажи источник книги»
const wbAudit = jload("docs/cases/work_balanced_audit/summary.json");
const ineligibleCorpus = jload("research/evidence/ineligible_corpus_registrations_v1.json");
if (
  ineligibleCorpus.status !== "ineligible_for_new_scientific_runs" ||
  sd.headline.claimStatus !== "exploratory_internal"
) {
  throw new Error("README generation requires the registered historical/ineligible headline status");
}

const M = Object.fromEntries(sd.models.map((m) => [m.id, m]));
const ci = (m) => `[${m.ci[0]}, ${m.ci[1]}]`;
const pdc = valPd.channels;
const pdEns = pdc["АНСАМБЛЬ (равновес.)"];           // равновесный (leak-free) PD-ансамбль
const pzc = proza.comparison_top1_macroF1;
const prozaChar = pzc["char-SVM (наш, классический ≈ stylo/Burrows)"][0];
const prozaNeuro = pzc["ruBERT-tiny2 (pretrained нейро-эмбеддинги)"][0];
const prozaEqual = pzc["наш равновесный ансамбль (старый, флаг)"][0];
const prozaEns = pzc["НАШ ансамбль (reliability^6, test-favoured diagnostic)"][0]; // веса train-OOF (leak-free), но степень 6 — лучшая из свипа на тесте

const wordsM = (cv.summary.total_words / 1e6).toFixed(1);
// author-clustered 95% CI macro-F1 ОТОЗВАН (sd.headline.macroF1CI = null): ресэмпл авторов меняет
// набор классов macro-усреднения → это не CI фиксированной 43-классовой функции. Точка описательна.
const MF1_WITHDRAWN = sd.headline.macroF1CI === null;
const taras = sd.tarasCase;
const tarasStrict = taras.headline[0];
const tarasLoose = taras.headline[1];
const tarasSpeech = taras.speech;
const tarasExt = taras.extended[0];
const wbCase = (stem) => wbAudit.cases.find((row) => row.source_spec.endsWith(`/${stem}.yaml`));
const wbTarasSusStrict = wbCase("taras_bulba_additions_strict_suspects_v2_fw_2000");
const wbTarasSameStrict = wbCase("taras_bulba_additions_strict_sameperiod_fw_2000");
const wbTarasAnn = wbCase("taras_bulba_additions_strict_annenkov_binary_fw_2000");
const wbTarasSomov = wbCase("taras_bulba_additions_strict_somov_binary_fw_2000");

// Читаемые числа: метрики до 4 знаков; p ≥ 1e-4 — десятичным до 5 знаков, меньше — научная запись (2 значащих).
const f4 = (x) => (x == null ? "—" : (+x).toFixed(4).replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, ""));
const fp = (p) => (p == null ? "—" : p === 0 ? "p < 0.0001" : +p >= 1e-4 ? `**p = ${(+p).toFixed(5).replace(/0+$/, "")}**` : `**p = ${(+p).toExponential(1).replace("e-", "e−")}**`);
const ruBooks = (n) => {
  const mod100 = Math.abs(n) % 100;
  const mod10 = mod100 % 10;
  if (mod100 >= 11 && mod100 <= 14) return "книг";
  if (mod10 === 1) return "книга";
  if (mod10 >= 2 && mod10 <= 4) return "книги";
  return "книг";
};

const tr = (label, m) => `| ${label} | ${f4(m.acc)} ${ci(m)} | ${f4(m.f1)} | ${f4(m.top2)} | ${fp(m.p)} |`;

const R = `# Stylo — атрибуция авторства русской прозы (честная, воспроизводимая)

Исследовательский пайплайн стилометрии: по тексту определяет наиболее близкого автора
среди корпуса и **честно оценивает, какие признаки реально работают**, а какие нет.
Собран как воспроизводимый пакет с единым источником истины параметров,
проверяемыми границами корпуса, доверительными интервалами и статистической значимостью.

> **Отзыв исторического headline (2026-07-26).** В зарегистрированном корпусе
> обнаружены произведения, вложенные друг в друга через границу LOBO (включая
> точный межкнижный дубликат чанка). Поэтому \`${f4(M.stylo.acc)}\` и производные интервалы
> ниже сохранены только как историческая арифметика и **не являются
> leakage-free оценкой точности**. Новые LOBO/RuAA-запуски на этом snapshot
> блокируются до зарегистрированной миграции content-компонент и полного
> перерасчёта. Машиночитаемый статус:
> \`research/evidence/ineligible_corpus_registrations_v1.json\`
> (\`${ineligibleCorpus.status}\`).

> Числа в этом README **генерируются** из тех же \`docs/*.json\`, что и сайт
> (\`scripts/gen-readme.mjs\`), поэтому не расходятся с витриной. Ручные правки чисел перетираются следующим прогоном генератора.

## Корпус

**${cv.summary.n_authors} автор, ${cv.summary.n_books} книг, ~${wordsM} млн слов** (\`docs/corpus_validation.json\`; провенанс
с sha256/источниками/объёмами — \`docs/corpus_manifest.json\`). Корпус прошёл аудит целостности
(\`log/experiments/audit_corpus.py\`): без задвоений, обрезки по 60000 слов и невычищенной
викиразметки у классиков. Дисбаланс объёмов ${Math.round(cv.summary.imbalance_ratio)}× (Достоевский ~1.15М слов vs Волошин ~1.8к) —
поэтому в историческом расчёте рядом с accuracy сохранялась описательная **macro-F1 point**.
Все интервалы ниже — только историческая арифметика, а macro-F1 CI дополнительно отозван. Тексты под
копирайтом — локально, **не в git**.

**Нормализация текста** (\`src/stylo/pipeline/clean.py\`, применяется до признаков): все виды тире → «—», ё → е,
дореформенная орфография → современная (ять/і/ѳ/ѵ, конечный ъ), имена людей маскируются NER'ом (PER → @),
викиразметка и футеры библиотек вычищаются. Кавычки унифицированы не полностью: «ёлочки» и прямые кавычки
сохраняются, „лапки“ и одинарные удаляются — стиль кавычек конкретного издания частично остаётся в тексте и может
попадать в пунктуационные признаки. **След издания/оцифровки — измеренное ограничение** (\`docs/provenance_probe.json\`):
на ${prov.mixed_authors.length} авторах с книгами из двух источников классификатор «предскажи источник книги» (leave-one-author-out,
${prov.n_books} книг) даёт AUC ${prov.probes.char_3_5.book_auc_loao} на символьных n-граммах и ${prov.probes.punctuation_only.book_auc_loao} на одной пунктуации против перестановочного нуля
~${prov.probes.char_3_5.null_book_auc_q95} — след источника в признаковых каналах различим, поэтому у одноисточниковых авторов идиолект и норма
издания в LOBO полностью не разделяются. При этом в решения модели след не протекает: ошибки LOBO у смешанных
авторов с источником не связаны (${prov.lobo_errors_vs_source.contingency_ws_correct.ws_ok + prov.lobo_errors_vs_source.contingency_ws_correct.local_ok}/${prov.lobo_errors_vs_source.contingency_ws_correct.ws_ok + prov.lobo_errors_vs_source.contingency_ws_correct.local_ok + prov.lobo_errors_vs_source.contingency_ws_correct.ws_miss + prov.lobo_errors_vs_source.contingency_ws_correct.local_miss} верных; точный тест Фишера p = ${prov.lobo_errors_vs_source.fisher_p}). Оговорка: класс
«локальная оцифровка» разнороден, установленное семейство источников одно (Викитека) — проба измеряет границу
«Викитека против остального», а не различие двух конкретных изданий.

Исторический (отозванный) headline — per-book LOBO (${sd.corpus.lobo.pool_authors} в обучаемом пуле /
${sd.corpus.lobo.tested_authors} тестированных автора / ${sd.corpus.lobo.books} ${ruBooks(sd.corpus.lobo.books)}; таблица ниже).
Диагностика каналов (LinearSVC + StratifiedGroupKFold(5)) — на срезе **${val.n_authors} автора / ${val.n_books} книга**
(одно-книжные авторы выпадают; Ильф-Петров и дневники Николая II вынесены в кейсы) — отдельный протокол, не headline.

| Модель | Historical accuracy (book-level) | Historical macro-F1 | Historical top-2 | Historical vs stylo (McNemar) |
|--------|----------------------|----------|-------|--------------------|
| **stylo (historical/ineligible corpus)** | **${f4(M.stylo.acc)} — ОТОЗВАНО как оценка точности**; исторические интервалы ${ci(M.stylo)} / [${styloCI.accuracy_authorclustered_CI[0]}, ${styloCI.accuracy_authorclustered_CI[1]}] | историческая точка ${f4(M.stylo.f1)}; inferential use запрещён | ${f4(M.stylo.top2)} (historical) | cross-work content leakage |
${tr("Мешок слов + логрег", M.bow_lr)}
${tr("char-3gram cosine", M.char_cos)}
| Cosine Delta · 150 / 300 / 500 MFW | ${f4(dcos.chunk_level["delta_cos:150"].accuracy)} / ${f4(dcos.chunk_level["delta_cos:300"].accuracy)} / ${f4(dcos.chunk_level["delta_cos:500"].accuracy)} | ${f4(dcos.chunk_level["delta_cos:150"].macro_f1)} / ${f4(dcos.chunk_level["delta_cos:300"].macro_f1)} / ${f4(dcos.chunk_level["delta_cos:500"].macro_f1)} | — | ${fp(dcos.chunk_level["delta_cos:500"].vs_stylo.mcnemar_p)} |
| Legacy selected-mass Delta по книгам · 150 / 500 / 1000 MFW | ${f4(dcos.book_level["delta_book:150"].accuracy)} / ${f4(dcos.book_level["delta_book:500"].accuracy)} / ${f4(dcos.book_level["delta_book:1000"].accuracy)} | — | — | ${fp(dcos.book_level["delta_book:1000"].vs_stylo.mcnemar_p)} |
${tr("Legacy selected-mass Delta (чанки) · 150 MFW", M["delta:150"])}
| Legacy selected-mass Delta (чанки) · 300 / 500 | ${f4(M["delta:300"].acc)} / ${f4(M["delta:500"].acc)} | ${f4(M["delta:300"].f1)} / ${f4(M["delta:500"].f1)} | — | p < 0.0001 |
| majority (нижняя граница) | ${f4(M.majority.acc)} | ${f4(M.majority.f1)} | — | — |

> Исторический расчёт использовал per-book LOBO (leave-one-book-out): на каждом шаге одна книга — тест,
> всё обучаемое (vocab/IDF/MFW/scaler/классификатор) фитится на остальных книгах (\`docs/final_comparison.csv\`).
> Он более не называется leakage-free: work-id complement не обеспечил
> независимость содержания. В LOBO-срезе оценивалась ${sd.corpus.lobo.books} ${ruBooks(sd.corpus.lobo.books)}
> (${sd.corpus.lobo.pool_authors - sd.corpus.lobo.tested_authors} автора с одной книгой в LOBO не тестировались — нет train-примера;
> macro-F1 считается по ${sd.corpus.lobo.tested_authors} тестированным классам). StratifiedGroupKFold(5) — лишь быстрый прокси в sweep/ablation.
>
> **Взвешивание обучения — \`${styloCI.training_weighting}\`.** Длинная книга получает больший train-вес внутри автора,
> vocab/IDF/MFW фитятся по чанкам. Work-balanced пересчёт (одна работа — один голос на train-стороне) ещё не проведён;
> до него headline трактуется как legacy-оценка (\`docs/cases/work_balanced_audit/\`). LOBO держит тестовую книгу целиком —
> это отдельное смещение train-weighting, не объяснение обнаруженной cross-work утечки.
> Строки Cosine Delta и legacy selected-mass Delta по книгам посчитаны тем же LOBO-протоколом отдельным скриптом
> (\`log/experiments/delta_cosine_lobo.py\` → \`docs/delta_cosine_lobo.json\`); контроль воспроизведения: delta:150 этим
> путём даёт ту же долю верных попаданий, что и исторический расчёт из final_comparison.csv (delta:150 на ${sd.corpus.lobo.books} книгах), —
> протокол воспроизводится.
> Знак столбца author-clustered Δacc CI в исторических \`docs/final_comparison.csv\` (и \`docs/ruaa_bench_v1.json\`)
> исправлен алгебраически в версионных \`docs/final_comparison.v2.csv\`/\`.v2.txt\` и \`docs/ruaa_bench_v1.0.1.json\` —
> точка, accuracy, macro-F1, McNemar и флаг значимости не изменились (запись и SHA-инвентарь — \`docs/ci_sign_erratum.json\`).
> Исторический McNemar-p считался по книгам; при коррелированных внутри автора
> книгах это антиконсервативная граница. Сохранённая cluster-robust арифметика
> stylo−BoW: Δacc +${crob.delta_accuracy}, author-clustered 95% CI
> [+${crob.author_clustered.ci95[0]}, +${crob.author_clustered.ci95[1]}],
> P(Δ≤0)=${crob.author_clustered.p_like_P_delta_le_0}. Из-за upstream cross-work
> leakage она не подтверждает текущий superiority claim. Это GKF-диагностика
> (\`docs/cluster_robust_stylo_vs_bow.json\`); историческая LOBO-разница
> +${(M.stylo.acc - M.bow_lr.acc).toFixed(3)} получена другим протоколом.

> **Как выбрана конфигурация (честно):** «stylo (все блоки)» — априорный дефолт \`configs/default.yaml\`, а не победитель
> свипа: свип по GKF-прокси показывает пять конфигураций численно лучше полной (например −char_ngrams; McNemar p≈1 — шум),
> и они сознательно не приняты. Гиперпараметры (C=1.0, MFW-300, 3–5-граммы, чанк 500 слов) — конвенционные значения,
> зафиксированные до итоговой оценки. Отложенных авторов и nested CV нет — вся разработка велась на этом корпусе,
> поэтому у цифры может оставаться оптимизм итеративной разработки; свип используется описательно, конфигурация по нему не выбирается.

**Историческая диагностическая гипотеза:** в отозванном расчёте богатый набор
признаков дал ${f4(M.stylo.acc)}, а мешок слов — ${f4(M.bow_lr.acc)}; McNemar
${fp(M.bow_lr.p)} сохранён только для воспроизведения и не является действующим
свидетельством значимости. Cosine Delta ${f4(dcos.chunk_level["delta_cos:500"].accuracy)}
и legacy selected-mass Delta по целым книгам ${f4(dcos.book_level["delta_book:1000"].accuracy)} также
остаются историческими ориентирами. Эта картина мотивирует новый content-safe
пересчёт, но не разрешает superiority claim. Провал строк «legacy selected-mass Delta (чанки)» при росте MFW
(${f4(M["delta:150"].acc)} → ${f4(M["delta:500"].acc)}) — эффект протокола, не свойство классики: на чанках в 500 слов слова MFW-рангов 301–500
встречаются лишь в ~${Math.round(dcos.mechanism_mfw_tail_sparsity_in_chunks.mfw_301_500.median_share_of_band_present_in_chunk * 100)}% (медиана доли слов диапазона на чанк), их z-оценки — шум разреженности, топящий манхэттенское
среднее; cosine на тех же z-оценках растёт с MFW, как в литературе. На корпусе со стилистически близкими авторами
(донская школа, одесситы, деревенщики) лексика путает похожих по теме, а структурный сигнал (синтаксис, служебные
слова, символьные паттерны) их разводит.

**Оговорка о macro-F1 CI (важно):** author-clustered 95% CI для macro-F1 не публикуется
(\`docs/stylo_lobo_authorci.json\` → \`macro_f1_authorclustered_interval_status = ${sd.headline.macroF1CIStatus}\`,
запись — \`${sd.headline.macroF1CIErratumRef}\`). Author-clustered bootstrap ресэмплит авторов и тем меняет набор
классов в macro-усреднении: автор, выпавший из ресэмпла, но предсказанный, даёт F1=0. Это не CI одной фиксированной
${sd.corpus.lobo.tested_authors}-классовой функции, поэтому интервал недействителен как мера разброса macro-F1 (точка
${f4(M.stylo.f1)} лежит выше его верхней границы); корректный интервал требует предрегистрированного протокола с
фиксированным набором меток. Точка macro-F1 ${f4(M.stylo.f1)} остаётся только
описательной исторической величиной. Внутри старой арифметики accuracy-bootstrap
давал author-clustered CI [${styloCI.accuracy_authorclustered_CI[0]}, ${styloCI.accuracy_authorclustered_CI[1]}]
и медиану ${styloCI.accuracy_bootstrap_median}, но upstream corpus ineligibility
делает и этот интервал историческим, не текущей uncertainty.
Реальный размер выборки для по-авторных выводов — ${sd.corpus.lobo.tested_authors} тестированных автора, а не ${sd.corpus.lobo.books} ${ruBooks(sd.corpus.lobo.books)}.

**Кейс «Тихий Дон»** (см. \`site/\` — статья, и \`docs/sholokhov_*.json\`): единственный формальный confirmatory-тест —
нециркулярный leave-block-out LOBO — атрибутирует **4/4 тома ТД → Шолохову** (все тома вне обучения; процедурные
контроли валидны). Что именно проверено тестом значимости: доля «чужих» отрывков в ТД-1 (${shOpen.td1_block_permutation.td1_ff}) выше фона донских
контролей (0.0) — перестановка по отрывкам p≈0.0001; отрывки внутри книги связаны, поэтому честнее блочная
перестановка — она даёт p=${shOpen.td1_block_permutation.block_perm_p} (\`docs/sholokhov_openset.json\`). Спад «чужой» доли от тома к тому (0.455→0.017) —
описание картины, отдельной тестовой статистикой он не проверялся. Это **направленное свидетельство за Шолохова**; но
доказать авторство нельзя — n≈2 независимых произведения, циркулярность эталона, автор/редактор неразличимы; примесь
стилистически похожей донской руки ниже ~25% метод не различает. Версии «Крюков написал» и «много литературных
негров» **не поддерживаются**.

**А если автора нет в списке (open-set)?** Атрибуция выше — закрытый список: модель всегда называет ближайшего из
корпуса. Проверки на этот случай сделаны и опубликованы. Детекция «текст автора вне корпуса» по типичности ответа
модели: AUC ${osp.outsider_detection_auc.max_prob} (авторов-чужаков: ${osp.outsider_authors.length}, их книг: ${osp.n_outsider_books}; \`docs/openset_passport.json\`). Инъекция чужака в кейс ТД: Платонов
уходит к самому себе (доля ${shOpen.openset_injection.open_to_self}), Шолохову — ${shOpen.openset_injection.open_to_sholokhov}. В открытом режиме (48 кандидатов) ТД-1 даёт Шолохову лишь
${shOpen.openset_td_full_argmax[0].sholokhov_share_open} — «в закрытом списке 4/4 неизбежны» опровергается собственными данными: открытый режим ранние тома
Шолохову сам не отдаёт (\`docs/sholokhov_openset.json\`). Верификационное семейство (unmasking Koppel–Schler, imposters
Koppel–Winter) на ТД тоже прогнано: 0/4 томов верифицируются как Шолохов — но контроль валидности показывает, что в
этом сеттинге метод смещён: зрелые бесспорные вещи Шолохова тоже проваливают unmasking (1/4), поэтому «ТД→Крюков» по
этим методам — артефакт, не свидетельство (\`docs/sholokhov_verify.json\`, \`docs/sholokhov_verify3_FW_ONLY.json\`).

**Кейс «Тарас Бульба» — статус после adversarial audit 2026-07-11.** Исторический headline был получен
центроидами, в которых длинные произведения имели больше train-side веса по числу чанков. После исправления на
\`L2(mean chunks/work) → equal mean works/author → L2\` многоавторная панель подозреваемых даёт gate
${f4(wbTarasSusStrict.work_balanced.work_macro_recall)} и панель эпохи ${f4(wbTarasSameStrict.work_balanced.work_macro_recall)}:
обе ниже предзарегистрированного порога 0.80, поэтому их top=Gogol **не интерпретируется**. Две бинарные панели
разделимы, но отвечают по-разному: Гоголь–Анненков → ${wbTarasAnn.work_balanced.top}
(${(100 * wbTarasAnn.work_balanced.winner_share.gogol).toFixed(1)}% чанков), Гоголь–Сомов →
${wbTarasSomov.work_balanced.top} (${(100 * wbTarasSomov.work_balanced.winner_share.somov).toFixed(1)}%). Корректный вывод:
простая версия «это Анненков» не поддержана в прямой паре, но уникальная гоголевская атрибуция крупных добавлений
текущей батареей **не установлена**. Парный отчёт 16 кейсов и отдельные паспорта:
\`docs/cases/work_balanced_audit/\`. Исправленный Delta full-refit rerun поддерживает Гоголя на suspects,
но в валидной бинарной Гоголь–Сомов меняет top по признаку (fixed FW → Гоголь, MFW → Сомов),
поэтому межпризнаковой/panel-invariant атрибуции также не даёт; старый Delta-отчёт остаётся legacy.

## Что это делает

- **Атрибуция**: к какому из авторов корпуса ближе всего спорный текст (\`data/frags_unknown\`).
  Исследовательские кейсы: «12 стульев» (Ильф и Петров vs Булгаков), «Тихий Дон»/«Поднятая
  целина» (Шолохов vs Крюков), дневники Николая II. Мистификации Акунина намеренно включены как проверка.
- **Оценка признаков**: ablation-sweep показывает вклад каждого блока фич (с CI и McNemar-p):
  работает канал или нет — измеряется, не постулируется.

## Архитектура

\`\`\`
configs/default.yaml  ← единственный источник всех параметров
src/stylo/
  config.py  lang.py  corpus.py  chunking.py  nlp.py (DocBin-кеш)
  features/      ← FeatureBlock + registry (каталог признаков, см. ниже)
  vectorizer.py  ← сборка вектора из включённых блоков
  models/        lr.py (+калибровка)   delta.py (legacy selected-mass Delta)   baselines.py
  eval/          lobo.py (historical book-id LOBO; content-safe после component migration)
                 groupkfold.py (быстрый исторический прокси)
                 metrics.py (macro-F1, bootstrap-CI)   significance.py (McNemar)
                 sweep.py (ablation)   final.py (итоговое сравнение)
  corpus_tools/  validate_corpus.py   fetch_classics.py (классики †70+, локальная валидация)
  pipeline/      clean → split → train → predict
  report/        index.html
\`\`\`

### Каталог признаков (\`features/\`)

| Блок | Что измеряет | Тематическая чувствительность |
|------|--------------|-------------------------------|
| \`char_ngrams\` | символьные 3–5-граммы + topic-bleaching (маскировка POS) | несёт жанр заметно, смягчён маскировкой POS |
| \`function_words\` | частоты служебных слов (MFW-300 или фикс-список 405 слов) | несёт жанр заметно (максимум по каналам) |
| \`syntax\` | 17 метрик субблоками: длины, POS-доли, пунктуация, TTR/Hapax/Yule, прямая речь, VRE, SSA | тематически устойчив |
| \`pos_ngrams\` | POS n-граммы (синтаксический скелет) | тематически нейтрален |
| \`punctuation_ngrams\` | n-граммы знаков препинания | тематически нейтрален |
| \`dependency\` | типы связей, дистанция головы, глубина дерева, ветвистость | наиболее тематически устойчив (чистый идиолект) |
| \`morphology\` | падеж/время/вид/число/степень (spaCy \`t.morph\`) | слабо несёт жанр |
| \`length_dist\` | распределения длин слов (Менденхолл) и предложений | слабо несёт жанр |
| \`embeddings\` | ruBERT mean-pooled | несёт тему; выключен — только контролируемый эксперимент |

Вклад каждого блока **измеряется** sweep'ом (\`docs/sweep_table.txt\`) — описательно: конфигурация по свипу
не выбирается, все блоки включены априори (см. «Как выбрана конфигурация» выше).

### Скорость

spaCy-разбор кешируется на диск (DocBin) и предвычисляется в лёгкие представления \`Rep\`
(POS/пунктуация/bleach, счётчики morph/dep, синтаксис, длины) в один файл — LOBO/sweep не вызывают
spaCy и не дробят I/O. Sweep считает конфиги параллельно; финальный LOBO — параллельно по фолдам.

## Историческая оценка и исправленная граница

- **Book-id grouping оказалось недостаточно**: fit исключал отложенный work-id, но
  совпадающее содержание оставалось под другим id. Новый протокол должен делить
  content-компоненты; старые CV/LOBO числа отозваны.
- **Исторические метрики**: Top-1/Top-2, macro-F1, per-author recall и bootstrap-CI
  сохранены для аудита арифметики, но не являются текущими estimates/claims.
- **Baseline-ы**: frozen legacy selected-mass Delta (\`delta:N\`; знаменатель
  \`Σ selected-MFW\`, не canonical all-token Burrows), char-3gram cosine,
  BoW-логрег, majority.
- **Историческая значимость**: McNemar и paired/author-clustered bootstrap не
  интерпретируются inferentially до полного content-safe пересчёта.
- **Множественные сравнения**: в кейсах с несколькими гипотезами p-значения даются и сырыми, и с поправкой
  Холма (\`docs/holm_correction.json\`); вердикты кейсов опираются на скорректированные.
- **Историческая калибровка**: ECE = ${f4(sd.headline.ece)} — это ПЛОХАЯ калибровка
  (хорошая ~0.02–0.05), модель переуверена, поэтому доли вероятностей в кейсах — ранжирование, не вероятности.

## Установка и запуск

\`\`\`bash
uv venv --python=python3.11 && source .venv/bin/activate
uv pip install --constraint requirements.lock -e ".[dev]"
python -c "from stylo.eval.paired_audit.run_plan import verify_installed_environment; verify_installed_environment()"
python -m spacy download ru_core_news_lg
python -m pytest -q        # быстрый unit/release-hygiene smoke; uv.lock локален и игнорируется
python scripts/check_release_hygiene.py --audit-local-refs   # перед релизом: publish-ref + индекс = FAIL при приватных путях; другие refs/stash = WARN
git config core.hooksPath .githooks   # включить pre-push-гейт (блокирует пуш приватной истории до загрузки объектов)
./run.sh fetch-classics      # отдельная загрузка открытых источников; не входит в all
./run.sh all                 # только зарегистрированный content-safe corpus; historical snapshot обязан упасть на gate
python scripts/run_benchmark.py --pd-only   # новый scientific run — только после content-safe регистрации
python scripts/run_benchmark.py             # новый scientific run — только после content-safe регистрации
node scripts/gen-site-data.mjs && node scripts/gen-readme.mjs   # replay сохранённых docs/provenance, не новый scientific run
\`\`\`
Команда \`fetch-classics\` не вызывается из \`all\`: acquisition и scientific
pipeline разделены. Сохранённые исторические числа воспроизводятся как
артефакты, но зарегистрированный ineligible snapshot не должен проходить новый
\`evaluate\` до content-component миграции.
Файл \`.python-version\` фиксирует поддерживаемый runtime: CPython 3.11.
Единственный поддерживаемый точный путь установки core/dev-окружения использует
\`requirements.lock\` именно как constraints-файл, как показано выше. После
установки обязательная проверка сравнивает Python major/minor и точные версии
переносимого core scientific stack с lock-файлом; bound-run при несовпадении
падает до создания run identity/checkpoint. Fingerprint не содержит абсолютных
путей, имени virtualenv, hostname или версии kernel. Не запрошенные CUDA/model/viz
пакеты из constraints-файла не устанавливаются. \`uv.lock\` не является
release-артефактом и игнорируется.
UMAP-визуализации требуют extra \`viz\`: \`uv pip install -e ".[viz]"\`.

### Ресурсный контракт LOBO

Generic \`stylo lobo\` ограничивает outer-fold parallelism максимум восемью
процессами (или меньшим \`evaluation.max_parallel_folds\`/числом CPU) и
\`pre_dispatch\` равным числу workers. Это важно, потому что process-local Rep
state может умножать RSS. Значение \`n_jobs=-1\` означает «до зарегистрированного
лимита», а не неограниченное использование машины. Generic CLI пока не имеет
per-fold resume: для длительного A0/A4/A1 прогона следует использовать
\`scripts/evaluation/run_stylo_lobo_validation.py --n-jobs N\`, который пишет
durable immutable checkpoint после каждой работы, возобновляется только из
семантически проверенных записей и также ограничивает \`N <= 8\`.

Exploratory \`sweep\`/work-balanced \`evaluate\` публикуют связанные файлы как
immutable generation под \`.stylo-batches/<publication>/generations/<sha256>/\`.
Единственная атомарно заменяемая точка — \`CURRENT.json\`; потребитель должен
разрешать поколение через \`resolve_published_batch\`, а не собирать соседние
flat-файлы, которые могут относиться к старому историческому запуску.

## Историческая точность и воспроизводимые артефакты

> **Состав корпуса:** полный корпус (${cv.summary.n_authors} автор; ${val.n_authors} в
> историческом бенчмарке) включает копирайтных и здравствующих авторов.
> И полный, и PD-only срезы происходят из зарегистрированного ineligible snapshot:
> в компоненте классиков Тургенева найдены вложенные произведения. Их можно
> воспроизвести как исторические артефакты, но нельзя публиковать как текущую
> оценку точности. Тексты в репозитории не хранятся и докачиваются по URL-манифесту.
> У двух авторов среза (Гумилёв, Пильняк) срок охраны в РФ продлён после реабилитации
> (ст. 1281 п. 5 ГК) — срез не является «редистрибутируемым» набором данных.
> Дневники Николая II — не проза, исключены.

**Перенос на другой жанр (дневники, письма) не покрывается бенчмарком — он измерен отдельно.** Прямой тест
(\`docs/crossgenre_recall.json\`): модель обучена только на прозе, проверена на дневниках и письмах ${cg.test.n_authors} классиков
(документов: ${cg.test.n_documents}, кусков текста: ${cg.test.n_chunks}), которых в этих жанрах при обучении не видела. На уровне документа top-1 **${cg.aggregate.all.doc_top1}**
(дневники ${cg.aggregate.diary.doc_top1.toFixed(2)}, письма ${cg.aggregate.letters.doc_top1.toFixed(2)}; случайный уровень 0.02 на ${cg.train.n_authors} классах), но по отдельным кускам — лишь
**${cg.aggregate.all.chunk_top1}**: смена жанра реально бьёт по точности, короткие фрагменты чужого жанра надёжно не атрибутируются.
К дневникам Николая II применяется не бенчмарк-классификатор, а отдельный кросс-регистровый метод с калибровкой
на авторах, у которых есть оба жанра; его измеренный предел: верификация «дневник↔письма» AUC ${nik.cross_register_auc.all_function_words} (полный
набор служебных слов) → ${nik.cross_register_auc.topic_invariant_LOAO} (тематически инвариантные, leave-one-author-out), позитив-контроль LOO ${nik.positive_control_LOO_acc} на 5 авторах
(\`docs/authorship_cases.json\`, \`docs/nikolas2_authorship.json\`). Числа таблиц выше на кросс-жанровые кейсы
не переносятся, и это прямо оговорено в каждом таком кейсе.

**Исторический PD-only бенчмарк (${valPd.n_authors} автора-классика (†70+ лет) / ${valPd.n_books} книг; snapshot ineligible):**

| канал (единый SVM) | top-1 | macro-F1 |
|---|---|---|
| char-n-граммы (2–5) | ${pdc["char (2-5)"].top1} | ${pdc["char (2-5)"].macro_f1} |
| function words | ${pdc.function_words.top1} | ${pdc.function_words.macro_f1} |
| синтаксис (dep+pos+syn) | ${pdc["syntax (dep+pos+syn)"].top1} | ${pdc["syntax (dep+pos+syn)"].macro_f1} |
| dependency (чистый идиолект) | ${pdc.dependency.top1} | ${pdc.dependency.macro_f1} |
| **ансамбль (равновесный, historical)** | **${pdEns.top1}** | **${pdEns.macro_f1}** |

Веса равновесного ансамбля не зависят от test, но upstream cross-work leakage
делает весь результат историческим. Старый macro-F1 author-clustered 95% CI
**[${valPd.macro_f1_authorclustered_CI[0]}, ${valPd.macro_f1_authorclustered_CI[1]}]**
не используется для текущего inferential вывода.
**Историческая диагностика каналов:** на полном бенчмарке (${val.n_authors} автора) равновесный ансамбль каналов под
*другим* классификатором (LinearSVC + StratifiedGroupKFold(5), а не LR+LOBO) даёт macro-F1 ${val.headline_macro_f1} /
top-1 ${val.ensemble_top1} (\`docs/validation.json\`). Это старый diagnostic view, не
альтернативный headline; LR+LOBO и GKF не смешиваются. PD-only и полный срезы
имеют разный состав классов и оба требуют нового content-safe пересчёта.

> **Средние метрики ≠ пригодность для конкретного кейса.** macro-F1 — среднее по авторам поровну; у отдельных
> авторов recall равен нулю (Катаев, Олеша, Зощенко — по 2–3 книги: один промах по целой книге обнуляет метрику),
> а провалы уходят к соседям по школе/регистру (\`docs/model_recall.json\`). Для спорного текста читать надо
> не среднюю цифру, а per-author recall, карту путаницы и register-matched панель близких авторов (\`docs/cases/*\`).

### Внешние бенчмарки

- **CCAT50 (Reuters, стандартный AA-бенчмарк, 50 авторов):** наш ансамбль **top-1 ${ccat.ensemble_top1}** — в
  опубликованном диапазоне char-n-gram SVM (~0.74–0.78). Прирост над C=1 в пределах шума — датасет у
  потолка класса методов. (Заявленные где-то 0.84 — артефакт другого протокола, не сопоставимы.)
- **Proza.ru (внешний РУССКИЙ датасет, \`Hieuman/proza_ru_hard\`, 50 авторов):** честный лидер — один
  **char-SVM ${prozaChar}**. Нейросети отстают — и замороженные, и дообученная: замороженный
  **gemini-embedding-001 (через VertexAI) ${vemb.test_top1}** (крупные 3072-мерные эмбеддинги, кодируют в основном тему),
  замороженный **ruBERT-tiny2 ${prozaNeuro}** (дистиллят), даже **дообученный ruBERT-tiny2 ${ft.test_top1}** (6 эпох на 1300 текстах,
  выбор по валидации) — *хуже* замороженного: малый дистиллят при дообучении на коротких текстах теряет предобученные
  представления (переобучение/забывание); замороженный **LUAR-MUD ${luar["LUAR-MUD (author-emb)"][0]}** (авторские эмбеддинги,
  обучены на английском Reddit — вне домена) ниже всех. Конкатенация char+LUAR (${luar["char+LUAR concat"][0]}) один char-SVM
  не улучшает. Источники: \`docs/luar_proza.json\`, \`docs/vertex_embedding_proza.json\`, \`docs/neuro_finetune_proza.json\`.
  Честная рамка: это НЕ «классика бьёт нейросети вообще» — крупная модель, больше данных, GPU и контрастивное дообучение
  могут дать иное; но в воспроизводимом CPU-режиме на короткой русской прозе классический char-SVM держит первое место.
- **Слияние каналов: наивное вредит, leak-free едва помогает.** Равновесный ансамбль (**${prozaEqual}**) *хуже* лучшего
  одиночного канала — слабые каналы тянут вниз. Reliability-взвешивание (веса каналов ∝ их точности на **train-OOF** —
  leak-free) поднимает ансамбль до **${prozaEns}**, но степень (6) — лучшая из свипа [2,4,6] на этом тесте, поэтому
  перевес +${(prozaEns - prozaChar).toFixed(3)} над char-SVM ${prozaChar} тест-благоприятен и в пределах шума.
  Консервативный честный лидер на proza — один char-SVM ${prozaChar}.

## Лицензия и оговорки

Код — MIT. Корпус не распространяется (копирайт; публичные классики докачиваются скриптом строго по
whitelist умерших >70 лет). Атрибуция при 2–8 книгах на автора имеет широкие CI — выводы по спорным
текстам сопровождаются оценкой неопределённости, а не подаются как факт.
`;

// coverage-гейт: ни одно подставленное число не должно быть undefined/null/NaN
const holes = [...R.matchAll(/undefined|NaN|\[null|\bnull\b/g)];
if (holes.length) {
  console.error("✗ README: незаполненные числа:", holes.slice(0, 8).map((h) => h[0]));
  process.exit(1);
}
const readmePath = path.join(ROOT, "README.md");
if (process.argv.includes("--check")) {
  if (fs.readFileSync(readmePath, "utf8") !== R) {
    console.error("✗ README.md расходится с scripts/gen-readme.mjs");
    process.exit(1);
  }
  console.log("✓ README.md совпадает с генератором");
} else {
  fs.writeFileSync(readmePath, R);
  console.log(`✓ README.md сгенерирован из docs/*.json: корпус ${cv.summary.n_authors}/${cv.summary.n_books}, бенчмарк ${val.n_authors}/${val.n_books}, PD ${valPd.n_authors}/${valPd.n_books} ансамбль ${pdEns.top1}/${pdEns.macro_f1}, CCAT ${ccat.ensemble_top1}`);
}
