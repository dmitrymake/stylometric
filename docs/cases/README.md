# Case Gate Specs

`stylo case` выполняет work-level feasibility gate и описательные closed-set
диагностики. Научная атрибуция target сейчас всегда воздерживается: в проекте
ещё нет зарегистрированного калиброванного open-set/negative-control gate,
который мог бы установить применимость панели к спорному тексту.

> **WITHDRAWAL (case-passport v1):** все ранее сохранённые паспорта со статусами
> `strong`/`moderate` получены до work-level target uncertainty и без
> обязательного open-set gate. Они остаются историческими артефактами, но их
> target-вердикты отозваны и не могут ранжироваться/публиковаться текущим CLI.
> Текущий формат — `stylo.case-passport.v2`.

Текущий handoff по реализованному слою и главному hardened-кейсу:
`docs/cases/HANDOFF.md`.

Hardened case families:

- `docs/cases/taras_hardened/` — историческая v1-серия; прежний positive public
  target claim отозван до появления calibrated open-set gate.
- `docs/cases/petersburg_hardened/` — исторические v1 closed-set diagnostics,
  не научные атрибуционные решения.

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
