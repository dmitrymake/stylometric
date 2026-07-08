# Case Gate Specs

Новый `stylo case`-слой нужен для первичного строгого просеивания исторических
гипотез: сначала work-level feasibility gate, потом атрибуция спорного текста,
затем единый JSON-паспорт.

Текущий handoff по реализованному слою и главному hardened-кейсу:
`docs/cases/HANDOFF.md`.

Hardened case families:

- `docs/cases/taras_hardened/` - positive public claim: large 1842 additions to
  «Тарас Бульба» go to Gogol, not to a foreign editorial hand.
- `docs/cases/petersburg_hardened/` - documented near-miss: `Н.Н.` in
  «Петербургской летописи» points toward Dostoevsky publicistic prose by centroid
  but does not reach strong attribution because chunk evidence is unstable.

Минимальный `case.yaml`:

```yaml
case_id: chekhonte_example
title: "Chekhonte example"
language: ru
feature_sets: [fw_fixed, char3]
unit: work
chunk_words: 1500
min_work_words: 300

candidates:
  chehov: input_cases/chekhonte/cand_chehov
  gogol:
    path: input_clean/gogol
    exclude:
      - input_clean/gogol/тарас_бульба.txt
  herzen:
    paths:
      - input_cases/kolokol_herzen_ogaryov/herzen_publicistic
      - input_cases/kolokol_herzen_ogaryov/herzen_kolokol

distractors:
  bilibin:
    path: input_cases/chekhonte/cand_bilibin
    label: "В. Билибин"

target: input_cases/chekhonte_dubia/texts/some_target.txt
forbidden_sources:
  - input_cases/chekhonte_dubia/texts/some_target.txt
sources:
  - "source URL or bibliographic note"
hypothesis: "What the case is testing"
target_description: "short target label for reports"
claim: "bounded public claim if the gate passes"
limitations:
  - "scope limit to keep the claim narrow"
provenance:
  analysis_command: "stylo case run path/to/case.yaml --out docs/cases/<case>.passport.json"
```

Команды:

```bash
stylo case run path/to/case.yaml --out docs/cases/<case>.passport.json
stylo case rank docs/cases/*.passport.json --out docs/cases/ranking.json
stylo case report docs/cases/*.passport.json --out docs/cases/ranking.md
stylo case dossier docs/cases/*.passport.json --out docs/cases/dossier.md
```

Статус `strong` возможен только если primary feature set проходит gate:
`work_macro_recall >= 0.80` и work-level permutation `p <= 0.05`. Без этого
паспорт возвращает отказ и прямо пишет, что атрибуцию давать нельзя.

`paths` позволяет собрать один author_id из нескольких папок. `exclude` обязателен,
если candidate-папка содержит спорный target или другой циркулярный/заражённый текст.
Если declared candidate не загрузился или после фильтров имеет меньше двух работ,
gate не запускается на уменьшенной панели.

Если target даёт только один chunk, паспорт может быть диагностическим, но не
получает `strong`: в failure modes добавляется
`target_single_chunk_no_strong_verdict`.

Ограничение v1: старые `docs/cases/*.json` остаются прежними bespoke-отчётами.
Для ранжирования их нужно пересчитать или вручную мигрировать в passport-формат с
полями `case_id`, `status`, `verdict`, `confidence`, `evidence_score`, `gates`.
