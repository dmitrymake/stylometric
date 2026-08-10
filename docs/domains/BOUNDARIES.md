# Domain Boundaries

Этот документ описывает операционную изоляцию доменов Stylo. Нормативные требования: `DOM-*` в `docs/agentic/STANDARD.md`.

## 1. Известные домены

| Домен | Фактические entry points | Владеет | Не владеет |
|---|---|---|---|
| corpus/data | `src/stylo/corpus/`, corpus manifests, acquisition/validation scripts | provenance, normalization, content grouping, corpus eligibility | model/evaluation claims |
| feature-extraction | `src/stylo/features/`, configs/resources | linguistic feature contracts and vector construction | attribution verdicts |
| evaluation/paired-audit | `src/stylo/eval/`, `src/stylo/domain/`, evaluation scripts/tests | estimands, split/control contracts, scoring, run identities | publication wording |
| reporting/publication | `src/stylo/report/`, `docs/`, generated public data | report artifacts, bounded claims and provenance links | raw private corpora |
| site | `site/`, `scripts/gen-site-data.mjs`, provenance checks | interactive article build and public presentation | scientific source-of-truth registries |
| research-governance | `research/governance/`, protocols and evidence registries | current status, requirements, topology, scientific gates | implementation ownership of every producer |

До создания конкретных domain docs owner каждого домена — repository owner. Имя каталога само по себе не доказывает ownership; authoritative source определяется кодом, executable configuration, accepted governance record и task evidence.

## 2. Primary domain задачи

Каждый task объявляет один primary domain. Несколько primary domains допустимы только для явной cross-domain L-задачи. По умолчанию агент читает domain doc и direct producers/consumers, не сканирует все соседние домены и не меняет соседние models/tests/selectors/site/docs.

## 2.1. Capability entry points и local context packet

Для типовой локальной задачи capability должен иметь один obvious authoritative entry point и direct verification path. Целевое состояние:

- изменение не требует чтения чужого домена без explicit contract;
- обычно достаточно не более двух обоснованных boundary transitions;
- central config не заставляет читать unrelated capabilities;
- один invariant не дублируется в docs/config/macros;
- test объясняет production/scientific contract, а не test framework internals.

File count сам по себе не metric; splitting оправдан, только если representative packet уменьшается.

## 3. Разрешённые зависимости

```text
corpus/data -> feature-extraction -> evaluation/paired-audit
     |                                  |
     v                                  v
research-governance <---------- reporting/publication -> site
```

Стрелка означает разрешённый explicit contract, не право переносить semantics. Cross-domain связь использует published API/artifact, content-addressed manifest/receipt, accepted governance registry, reconciliation contract или другой утверждённый interface.

Прямое соединение artifacts/claims с разным grain или estimand запрещено без cardinality и fan-out/identity checks.

## 4. Cross-domain contract

| Поле | Вопрос |
|---|---|
| Producer / consumer | Кто владеет source semantics и кто использует? |
| Grain | Что означает одна row/work/chunk/run/claim? |
| Key/cardinality | Stable identity и 1:1/1:N/N:M; где dedup/weighting? |
| Time | Event/effective/load/publication time; current vs historical? |
| NULL/default | Как представлены unknown/missing/not-applicable? |
| Delete/history | Withdrawal, supersession, tombstone, immutable artifact? |
| Late arrival | Replay/restatement windows? |
| Ownership | Где исправляется ошибка? |
| Verification | Identity, contract, fan-out, leakage, reconciliation tests? |

## 5. Shared/conformed limits

Допустимы canonical identifiers, technical envelopes, content hashes, neutral normalization/time helpers, agreed bridges и generic quality primitives. Нельзя помещать domain-specific corpus eligibility, attribution thresholds, estimands, publication wording или governance status в shared layer только ради reuse. Shared code принимает semantics явно или ссылается на accepted contract.

## 6. Типичные leakage для Stylo

- **Identity leakage:** filename/book ID принимается за content identity, хотя дубли/вложенные тексты требуют content grouping.
- **Split leakage:** chunks или related works одного content group попадают по разные стороны train/evaluation.
- **Estimand leakage:** chunk-weighted и work-balanced results сравниваются как одна метрика.
- **Feature leakage:** topical/content features трактуются как authorial-style evidence без control.
- **Status leakage:** historical/withdrawn result публикуется как current claim вопреки governance ledger.
- **Selector leakage:** test/runner не входит в canonical CI or requirements map.
- **Publication leakage:** generated site data расходится с canonical artifacts/provenance gate.
- **Privacy/copyright leakage:** raw/private corpus text попадает в Git, evidence или transcript.
- **Default leakage:** unknown/not-tested интерпретируется как fail/pass или real zero.

## 7. Процедура cross-domain изменения

1. Объявить domains/risk и прочитать оба contracts/ADR.
2. Проверить physical artifact/schema, identity, distributions/cardinality и current governance status.
3. Сформулировать interface contract и scientific invariants.
4. Добавить relevant identity/leakage/fan-out/reconciliation/provenance tests.
5. Получить owner review либо зафиксировать unresolved approval.
6. Обновить domain docs/ADR, если изменилось долговечное содержание.
7. Проверить CI selectors, generated site/publication surfaces и status lineage.
8. Для maintainability claim измерить context packet и entry points/sources before/after.

## 8. Review checklist

- [ ] Domains совпадают с изменёнными files/artifacts.
- [ ] Нет неутверждённой semantics в shared layer.
- [ ] Grain/identity/key/cardinality/estimand доказаны.
- [ ] Content/train/test/future/current-state leakage проверены.
- [ ] NULL/unknown/withdrawal/supersession semantics явны.
- [ ] Contract tests реально входят в canonical selectors/CI.
- [ ] Producer, consumer, governance и publication docs согласованы.
- [ ] Локальная задача имеет obvious entry point и bounded packet.
- [ ] Central config не требует unrelated context.
- [ ] Simplification уменьшает packet/source count, а не перераспределяет строки.

## 9. Встроенный шаблон domain document

Создавай `docs/domains/<domain>.md` только для реального домена с owner и канонической семантикой.

```markdown
# Domain: <name>

- Status: draft | active | deprecated
- Owners:
- Last verified:
- Related ADR/tasks:

## Purpose and boundaries
- Owns:
- Does not own:
- Upstream producers:
- Downstream consumers:

## Canonical entities and terms
| Term/entity | Definition | Source of truth | Notes |
|---|---|---|---|

## Data/scientific contract
- Grain / estimand:
- Keys/cardinality/content identity:
- Event/effective/load/publication time:
- NULL/default semantics:
- Delete/withdrawal/supersession semantics:
- Late-arrival/restatement:
- PII/copyright/security classification:

## Authoritative entry points and representative local changes
- Primary capability entry points:
- Representative scenarios:
- Minimum required files/contracts:
- Expected domain transitions:
- Central config dependencies:
- Duplicate/conflicting sources:

## Published interfaces
| Interface | Consumers | Contract | Verification |
|---|---|---|---|

## Invariants and quality checks
- ...

## Known unknowns and change procedure
- ...
```
