# Исполнение ревью Stylo

- Владелец: Dmitry Purtov.
- Разрешение: после отчёта ревью пользователь указал «делай, запускай sol 6.1 агентов для исполнения».
- Baseline: `f312efb8f5fa291a961c337e6efa3b67cc247be4`.
- На входе единственный untracked файл — отчёт `2026-10-02-codebase-research-review.md`.
- Цели исследования: Булгаков / «Двенадцать стульев», Шолохов / «Тихий Дон».

## Объём

Исправить воспроизводимые R1–R8, проверить регрессии и механические гейты; проверить
наличие/совместимость входов ранее разрешённого corrected topic-validity исследования;
продолжать независимую подготовку исследовательских методов и метаданных по доступности данных.
Публичные научные заявления, immutable evidence, push и публикация требуют отдельного
непосредственного разрешения и в этот этап не включены.

Изменения исходников завершаются до любого реального bound run. Старый checkpoint нельзя
перепривязать к изменённой реализации. Приватные тексты используются только внутри локальных
вычислений при наличии нужных входов, не попадают в Git, логи, отчёты и ответы.

## Исполнение

- Два агента GPT-6.1 Sol: статистика R3/R4; corpus/clean/split/chunking R2/R5/R8.
- Основной агент: inference/config R1/R7, embeddings R6, интеграция, проверка prerequisites.
- Python: `/tmp/stylo-review-venv/bin/python`, CPython 3.11, core из `requirements.lock`.

## Результат

R1–R8 реализованы и прошли интеграционную проверку. LZMA baseline и metadata census
добавлены. Следующий исполняемый этап — полный ранее разрешённый topic-validity run;
результат научного измерения ещё не получен.

| Пункт | Изменение |
|---|---|
| R1 | Train требует явную deployment-панель независимо от benchmark exclusions. Predict возвращает diagnostic_closed_set_top, winner=null и abstained=true при отсутствующей калибровке применимости |
| R2 | Общие word shingles делятся на полный объём меньшего произведения; отсутствие пересечений обрабатывается без ошибки |
| R3 | Неверный macro-F1 CI заменён точечной оценкой и явной причиной отсутствия CI; accuracy CI сохранён |
| R4 | Одноавторские work controls определяются по всей разметке произведения, включая все переданные главы |
| R5 | Clean v2 сохраняет actual NER/preprocessing identity и hashes; новый split проверяет receipt и байты |
| R6 | Embedding cache v2 требует immutable revisions, учитывает настройки и runtime, проверяет dimension/finite и пишет атомарно |
| R7 | External bundle token исключён из config hash; остальные настройки связаны. Конфликт CLI/config token отвергается |
| R8 | Малые хвосты пропускаются, большие предложения сохраняются; safe splitter больше не теряет символы |

## Изменения пользовательского контракта

Для train/predict нужен явный список минимум двух кандидатов: YAML-поле
`deployment.candidate_authors` либо повторяемый `--candidate-author`. Один и тот же список
передаётся обеим командам; сохраняются полные проверки состава/порядка классов bundle.
Пример после подготовки отдельного корпуса эталонов с исключённой спорной дилогией:

```bash
stylo train --config case.yaml --candidate-author bulgakov --candidate-author ilf-petrov
stylo predict --config case.yaml --candidate-author bulgakov --candidate-author ilf-petrov --model-bundle-token TOKEN_FROM_TRAIN
```

Token можно сохранить в `deployment.expected_bundle_token` после обучения; это не меняет
identity обучающей конфигурации. Изменение модели, кандидатов и остальных настроек по-прежнему
делает bundle несовместимым. Старые bare configs без панели нужно дополнить и переобучить
bundle, поскольку панель — часть identity. Формат исторических bundle не переписывается.

Рейтинг описывает только относительную близость внутри панели. Вердикт об авторстве
остаётся `abstained`, даже при большом margin: открытая верификация ещё не откалибрована.
Явная панель не заменяет проверку независимости эталонных произведений от target.

Новый split требует clean manifest v2. Старые clean roots без actual preprocessing receipt
требуют повторного `clean`; существующие замороженные fragment snapshots остаются читаемыми
с записанной исторической identity. Новые chunker/normalization contracts имеют версию v2.
Исторические данные не пересоздавались.
Новый deployment train также проверяет stored work manifests до warm/fit: старые, смешанные
или отсутствующие chunker identities не маркируются текущей v2. Его metadata использует
проверенный recorded hash. Это не меняет исторические loaders и frozen topic evaluator.

Для дискового embedding cache нужны полные 40-hex commit IDs в `features.embeddings.revision`
и, при необходимости, `tokenizer_revision`. Ветки, теги и локальные модели работают без
дискового кеша. Старое cache namespace не переиспользуется и не удаляется.

## Данные для продолжения исследования

Metadata-only проверка 2026-10-02: corrected bundle и historical parent доступны локально,
агрегат topic validity отсутствует. Текущий local checkpoint содержит 70 A0/current fits,
остальные arms пусты; его commit — baseline `f312efb`. Это отличается от 260/992 в старом
ledger/handoff. Исторические записи не переписываются по этому наблюдению.
Источник — поля schema/completed/git_commit/run_identity checkpoint, без чтения исходных
текстов в транскрипт. После изменения исходников эти 70 fits нельзя продолжать под новой
identity; старый checkpoint сохранён. Проверяется новый scoped запуск с отдельным checkpoint.

Русская модель `ru_core_news_lg==3.8.0` установлена в изолированное окружение для проверок,
которые были пропущены во время ревью. Frozen core dependencies не изменены.

## Новый метод и корпусные входы

Добавлен `src/stylo/models/compression.py`: research-only LZMA conditional code length
`C(reference + target) - C(reference)`, exact UTF-8, фиксированные параметры raw LZMA1.
Нормировка — байты сжатия на байт target; при ранжировании каждому reference work дан
равный вес. Панель, work IDs и точные повторы проверяются до сжатия; near duplicates,
жанр и библиографическая независимость требуют отдельной проверки. Результаты —
диагностические scores, не вероятности/вердикты. PPM, статистическая репликация статьи,
calibrated open-set gate и интеграция в frozen evaluator не заявлены.

Проверены 26 новых regression tests и 11 соседних Delta/metrics tests. Источник метода:
[Ryabko/Savina, 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8534409/).
Реального прироста качества этого метода пока не измерено.

[Metadata census](2026-10-02-case-corpus-census.md) разделяет исторический каталог H,
audit parent A и corrected bundle C. Root независимо пересчитал восемь author rows:
в C нет `ilf-petrov` и `sholohov`; библиографические пробелы H не устранены corrected
content hashes. Подготовлены списки приоритетных сольных фельетонов, изданий и источников.
Автоматически сертифицированной независимой case panel ещё нет.

## Проверки и открытые ограничения

- Первый полный прогон: 1281 passed, 6 failed, 20 skipped. Четыре failure связаны с
  отсутствовавшим pymorphy3, один с synthetic clean fixture старого формата, один с raw
  JSON serialization нового embedding key. Причины устранены.
- Итоговый полный regression: **1333 passed, 4 skipped, 2 warnings**, 110.95 s, exit 0.
  Пропуски: live-golden module, два ru_core_news_md tests и один opt-in real context test.
  Последний затем отдельно запущен с доступным локальным corrected bundle.
  Warnings: два прежних `invalid escape sequence` в frozen-writer tests.
- Установлены pymorphy3 и словари ровно в версиях `requirements.lock`; real-NLP targeted
  после этого: 132 passed, 2 skipped (нет ru_core_news_md).
- Inference/config/cache/release targeted: 108 passed. Corpus-builder fixture: 40 passed.
- Отдельно включённый live-golden replay: 15 passed, 2 failed. Не совпали runtime и
  BLAS/OpenMP fingerprints: fixture требует NumPy 2.4.6 / SciPy 1.17.1 / spaCy 3.8.14,
  а supported requirements.lock фиксирует 2.3.5 / 1.16.3 / 3.8.11. Численные replay nodes
  прошли, но полный capture-environment PASS не заявляется; fixture и guards не менялись.
- Topic no-fit preflight на промежуточном дереве прошёл: 248 folds, 43 metric labels,
  47 probability classes. Добавление compression-модуля изменило source identity, поэтому
  перед реальными fits нужен новый preflight на окончательно зафиксированном дереве.
- Source inventory: 249 paths, digest
  `e92636278dab698974043c391549ec22b7348da7eb35559ad91852e9de592602`, OK.
  Release hygiene HEAD/index и provenance 93/1 прошли; regenerated site JSON/manifest
  из отдельного patched archive побайтно совпадают с tracked файлами.
- Sdist и wheel собраны из копии изменённого дерева; warning setuptools об устаревшей
  TOML-таблице license сохраняется. Source/site/governance protected claims не переписывались.
- Cross-review проверил совместимость corpus и inference. Дополнительная гипотеза о
  принятии изменённых chunk bytes снята после полного synthetic loader/disk-gate опыта:
  существующие проверки отклоняют изменённые, отсутствующие и лишние файлы.

## Запуск долгого измерения после фиксации исходников

Проверенная реализация фиксируется локальным commit; push/публикация не выполняются.
Далее повторяется no-fit preflight и запускается полный A0/A4 × current/topic_strict
LOBO-248 (992 fits), восемь workers. Старый checkpoint не удаляется и не перепривязывается.
Новые состояния выполнения находятся только в игнорируемом `research/local/`:

- `topic_validity_lobo_v1.20261002.repaired.checkpoint.json` — checkpoint;
- `topic_validity_lobo_v1.20261002.repaired.log` — журнал без исходных текстов;
- `topic_validity_lobo_v1.20261002.repaired.process.json` — PID/commit/source identity.

Полная команда (тот же checkpoint для resume, без `--fresh`):

```bash
env PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /tmp/stylo-review-venv/bin/python scripts/evaluation/run_topic_validity_lobo_v1.py \
  --execute \
  --bundle-root research/local/v3_2_bundle_20260824/paired_audit_v3_2_bundles/ff620b05f20b81c21732014b553aa739a393c74fe344e6d9f2bd8d80996cef21 \
  --historical-parent-root data/audit_corpus/15d265e0878dbf1acd9224e2558598ff7266fd6fc650585d1433fbd65a717029 \
  --ruaa-selection-manifest data/ruaa_bench_v1/manifest.json \
  --checkpoint research/local/topic_validity_lobo_v1.20261002.repaired.checkpoint.json \
  --output research/evidence/topic_validity_lobo_v1/aggregate.json \
  --workers 8
```

Это exploratory topic study, не confirmatory execution и не новая атрибуция романов.
Во время работы нельзя менять `src/stylo`, runner, модель/окружение и HEAD: их identities
связаны с checkpoint. SIGTERM/SIGINT сохраняет завершённые fits штатным runner.
После агрегата нужны отдельная проверка результата и решение о model semantics;
публикация и исправление R9 на сайте этим запуском не разрешаются.
