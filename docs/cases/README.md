# Case Gate Specs

| Classification | Boundary |
|---|---|
| Current v2 | abstain/inconclusive: no registered calibrated open-set gate; work-level feasibility remains, and `diagnostic_closed_set_top` is descriptive only. Old passports are rejected. |
| Taras hardened v1 | Historical family; the former positive target claim is withdrawn and not a current verdict. |
| Petersburg hardened v1 | Historical closed-set diagnostics only; not an attribution verdict. |

Detailed handoff: `docs/cases/HANDOFF.md`.

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

Feasibility gate требует `work_macro_recall >= 0.80` и work-level permutation
`p <= 0.05`, но этот gate проверяет только различимость самой candidate-панели.
Он не является target p-value и не разрешает атрибуцию. Для target автоматически
добавляется точное имя `target_open_set_applicability_gate_v1`; пока gate
зарегистрирован как `unavailable`, паспорт возвращает `status=inconclusive`,
`top=null`, `abstained=true`. Относительный closed-set победитель сохраняется
только как `diagnostic_closed_set_top`.

`paths` позволяет собрать один author_id из нескольких папок. `exclude` обязателен,
если candidate-папка содержит спорный target или другой циркулярный/заражённый текст.
Если declared candidate не загрузился или после фильтров имеет меньше двух работ,
gate не запускается на уменьшенной панели.

Target-чанки сохраняют parent `work_id`. CI строится только work-cluster
bootstrap по независимым target works; iid chunk bootstrap запрещён. При менее
чем двух независимых target works `margin_ci95=null`, а в failure modes
добавляется `target_lt2_independent_works_ci_unavailable`. Число чанков одного
work не создаёт новых независимых единиц.

Старые `docs/cases/*.json` и паспорта без `schema_version=stylo.case-passport.v2`
сохраняются только для исторической воспроизводимости. `load_passport` отвергает
их, чтобы старые `strong`/`moderate` вердикты нельзя было смешать с v2.
