# Taras Bulba Hardened Case

> **CASE-PASSPORT v1 WITHDRAWAL.** Все приведённые ниже target-статусы
> `strong`/`moderate`, включая positive controls, являются историческими
> closed-set diagnostics. Их iid chunk CI и отсутствие calibrated open-set
> applicability gate не соответствуют v2; научные атрибуционные вердикты сняты.

> **POST-AUDIT NOTICE — 2026-07-11.** Прежний headline о панель-устойчивом
> притяжении добавлений к Гоголю **снят** после устранения train-side
> псевдорепликации. В work-balanced аудите multi-candidate panels не проходят
> порог: suspects v2 = 0.7958, same-period = 0.7876, topic-cossack = 0.7875
> (везде p=0.0005, но gate < 0.80 запрещает чтение target). Две бинарные панели
> проходят, однако конфликтуют: Гоголь↔Анненков даёт Гоголя (gate 1.000), а
> Гоголь↔Сомов — Сомова (gate 0.925). Следовательно, panel-invariant
> атрибуционного вывода сейчас нет. Актуальные артефакты:
> [docs/cases/work_balanced_audit/README.md](../work_balanced_audit/README.md) и
> [docs/cases/work_balanced_audit/summary.json](../work_balanced_audit/summary.json).
> Исторический `reports/delta_replication.json` — superseded selected-mass
> legacy-аудит, а не каноническая Delta Бэрроуза и не основание для headline.

Гипотеза (ревизионистская): крупные добавления редакции «Тараса Бульбы» 1842
года написал не Гоголь, а редактор/чужая рука (исторические подозреваемые:
Н. Я. Прокопович — редактор издания 1842, П. В. Анненков — переписчик под
диктовку в Риме, 1841).

Проверяются только **крупные добавленные пассажи**, выделенные сравнением
полных академических редакций 1835/1842 (ФЭБ, изд. АН СССР). Точечная
редактура, русификация отдельных слов и правки набора не проверяются.

## Data chain (полностью воспроизводимая)

1. `scripts/fetch_taras_editions.py` — обе редакции с ФЭБ целиком
   (1842: 36 601 слово, 12 глав; 1835: 20 559 слов, 9 глав), постраничные
   разрывы ФЭБ склеены.
2. `scripts/extract_taras_additions.py` — добавления по 4-граммам предложений:
   strict (overlap < 0.10 к 1835) = 23 640 слов, loose (< 0.20) = 24 303.
3. `scripts/audit_taras_extraction.py` → `reports/extraction_audit.json`:
   strict/loose найдены в 1842 на 100.0%, в 1835 — 0.0004/0.0007 (8-словные
   шинглы, пофрагментно); утечка в остальные работы анкора Гоголя = 0.
4. `scripts/build_taras_panel.py` → панель подозреваемых и топик-контролей с
   az.lib.ru (`panel_manifest.json`: SHA256, слова, источники, факт
   модернизации орфографии).

## Protocol

- Primary feature: `fw_fixed` (закрытый список служебных слов);
  диагностический канал — char3 по симметрично NER-маскированным текстам.
- Gate: leave-one-work-out, один изъятый текст = один голос; на обучении
  центроид автора есть равновесное среднее L2-нормированных поработных
  профилей. Порог work_macro_recall >= 0.80; значимость: перестановка ярлыков
  работ, random_2000.
- Gogol anchor: `input_clean/gogol` без `тарас_бульба.txt`.
- Кандидатская панель подозреваемых (suspects v2): Гоголь, **Анненков
  (проза 1840-х, 5 работ, 239k слов — «Письма из-за границы» 1841-43 и др.)**,
  Тургенев, Достоевский. Пушкин исключён по gate-диагностике: его
  документальная проза коллидирует с травелог-регистром Анненкова (recall
  3/6); решение принято по confusion матрице gate, таргет не участвовал.
- Прокопович напрямую немоделируем (прозаического корпуса нет) — статус
  documented-but-unmodelled; его письма 1843 г. — отдельный диагностический
  target.

## Results: work-balanced post-audit

| target | панель | gate | p | направление target | решение |
|---|---|---:|---:|---|---|
| strict additions | Гоголь vs Анненков | 1.0000 | 0.0005 | gogol 14/16 | проходит, бинарная диагностика |
| strict additions | Гоголь vs Сомов | 0.9250 | 0.0005 | somov 13/16 | проходит, бинарная диагностика |
| strict / loose additions | suspects v2 | 0.7958 | 0.0005 | gogol 14/16; 14/17 | **fail; target не интерпретируется** |
| strict / loose additions | same-period | 0.7876 | 0.0005 | gogol 11/16; 11/17 | **fail; target не интерпретируется** |
| strict / loose additions | topic-cossack | 0.7875 | 0.0005 | somov 13/16; 16/17 | **fail; target не интерпретируется** |

### Corrected selected-mass Delta full-refit audit

Старый Delta JSON не был независимым подтверждением: его permutation null
переставлял истинные метки при фиксированных предсказаниях. Исправленный
legacy selected-mass equal-work/full-refit отчёт сохранён отдельно в
`../work_balanced_audit/custom/taras_delta_full_refit_work_balanced.json`.
На его suspects-панели с Пушкиным оба режима проходят (`delta_fw=0.9500`,
`delta_mfw=0.9467`) и дают Гоголя. Но на также прошедшей binary
Гоголь–Сомов режимы расходятся:
fixed FW → Гоголь 13/16, learned MFW → Сомов 9/16. Topic fixed-FW не проходит
(`0.7312`), topic MFW проходит (`0.9344`) и даёт Гоголя. Все p=0.0005.
Итог: corrected selected-mass Delta даёт exploratory перевес Гоголя на части
панелей, но не восстанавливает cross-feature/panel-invariant headline.

## Claim

Work-balanced данные **не дают положительной атрибуции** крупных добавлений
1842 года. Узкая бинарная проверка не поддерживает Анненкова против Гоголя,
но другая прошедшая бинарная проверка предпочитает Сомова, а все три более
широкие панели остаются ниже зарегистрированного порога. Поэтому прежний
headline «добавления устойчиво ближе к Гоголю» отозван; остаётся только
эксплораторная карта panel sensitivity.

Ограничения: анализ относится к крупным пассажам, не к точечной правке;
Прокопович проверен лишь косвенно (кандидатского корпуса не существует);
результат зависит от состава панели; multi-candidate gate не достигнут.

## Artifacts

- [`target_manifest.json`](target_manifest.json) и
  [`panel_manifest.json`](panel_manifest.json)
- [`specs/`](specs/) и [`passports/`](passports/) — панели, контроли,
  диагностики и исторические v1-прогоны
- [`reports/extraction_audit.json`](reports/extraction_audit.json) и
  исторический selected-mass
  [`reports/delta_replication.json`](reports/delta_replication.json)
- [work-balanced паспорта и сводка](../work_balanced_audit/) и
  [corrected selected-mass Delta JSON](../work_balanced_audit/custom/taras_delta_full_refit_work_balanced.json)
