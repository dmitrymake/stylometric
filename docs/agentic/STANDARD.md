# Стандарт агентной инженерной работы

- Версия: **1.3**
- Статус: **active**
- Владелец: **Dmitry Purtov (repository owner)**
- Дата вступления в силу: **2026-08-10**
- Следующий пересмотр: **2026-09-07**
- Последнее изменение: **2026-08-10**; журнал находится в разделе 19.

## 1. Назначение

Стандарт определяет минимально достаточный процесс, при котором человек управляет coding agents, а результат остаётся проверяемым, воспроизводимым и поддерживаемым без transcript конкретной модели. Он применяется к research, code/data/config changes, review, incidents, pruning и bounded simplification campaigns в Stylo.

## 2. Нормативные слова

- **ОБЯЗАН** — нарушение требует остановки, исправления или waiver.
- **СЛЕДУЕТ** — ожидаемая практика; отклонение объясняется в task result.
- **МОЖЕТ** — допустимый вариант.

Справочные guides не вводят обязательные требования. Нормативные требования существуют только здесь и имеют ID.

## 3. Роли

| Роль | Ответственность |
|---|---|
| Оператор | Определяет intent, принимает риск, утверждает R3a и научные/публикационные решения. |
| Planner | Формирует frozen task contract из intent и проверенного state packet. |
| Исполнитель | Исследует, реализует или выполняет pruning в пределах контракта. |
| Adversarial reviewer | Ищет in-scope нарушения frozen contract и regressions, не расширяя DoD. |
| Deletion reviewer | Ищет безопасные удаления при frozen behavior; обязателен для pruning. |
| Владелец стандарта | Управляет версиями, experiments, metrics и waivers. |

Одна среда может исполнять разные роли только в раздельных контекстах с явным переключением. Независимый review не наследует выводы предыдущей роли до собственного результата.

## 4. Артефакты и ответственность

| Артефакт | Назначение |
|---|---|
| `AGENTS.md` | Точка входа и правила поведения в checkout. |
| Task file | Контракт конкретной работы и result. |
| State packet | Переносимый проверяемый снимок для внешнего контура. |
| Handoff | Короткая маршрутизация следующей сессии. |
| ADR | Долговечное решение и trade-offs. |
| Domain doc | Семантика, ownership, grain, keys и boundaries. |
| Runbook | Повторяемая эксплуатационная процедура. |
| Evidence artifact | Воспроизводимое bounded доказательство. |
| Context packet | Минимальный набор entry points, files/sections и contracts для representative task. |

`ART-01 [ОПЕРАТОР/АГЕНТ]` Один факт или правило ОБЯЗАНЫ иметь один канонический дом; расходящиеся копии запрещены.

`ART-02 [АГЕНТ]` Handoff, state packet и reports НЕ МОГУТ заменять код, task, ADR, domain contract или runbook.

## 5. Классификация работы

Тип: `context-only | research | implementation | pruning | review | incident`.

Размер: `S` — локальный результат; `M` — несколько files/layers; `L` — несколько subsystems/domains с coverage и отдельным review.

Риск: `R1` — локально/обратимо; `R2` — заметное behavior/consumer impact; `R3b` — read-only production, sensitive data или bounded hot-path check; `R3a` — production mutation, irreversibility, money, security, PII, external contract/public interface. Flags: `mutation`, `production`, `money`, `security`, `pii`, `external-contract`, `load`, `irreversible`.

`CLASS-01 [ОПЕРАТОР/АГЕНТ]` До существенной работы задача ОБЯЗАНА иметь type, size, risk, flags и primary domain.

`CLASS-02 [АГЕНТ]` Новый риск ОБЯЗАН остановить работу до reclassification и недостающих gates.

## 6. Контракт задачи

`TASK-01 [ОПЕРАТОР/PLANNER]` Для M/L/R2/R3 task file ОБЯЗАН существовать до mutation. S/R1 МОЖЕТ использовать short contract.

`TASK-02 [PLANNER]` Контракт ОБЯЗАН определить goal, observable result, baseline, scope, domains, invariants, evidence, acceptance, verification, stop conditions и artifacts.

`TASK-03 [АГЕНТ]` Research/context-only/review НЕ МОГУТ переходить в implementation без явного gate.

`TASK-04 [АГЕНТ]` Scope creep ОБЯЗАН быть отклонён либо оформлен изменением контракта.

`TASK-05 [PLANNER]` До implementation/pruning контракт ОБЯЗАН задать max production subsystems, policy на production files/public entry points, production LOC added/net target, test/docs budgets, policy на framework/state/dependency/generic abstraction, context target для maintainability и review correction budget. N/A требует причины.

`TASK-06 [PLANNER/АГЕНТ]` Acceptance и non-goals ОБЯЗАНЫ быть frozen до mutation; reviewer НЕ МОЖЕТ расширять их.

`TASK-07 [АГЕНТ]` Превышение tripwire ОБЯЗАНО остановить mutation; агент не увеличивает budget самостоятельно.

`TASK-08 [PLANNER/АГЕНТ]` Clean restart ОБЯЗАН перечислить rejected/excluded branches, stashes, diffs, plans и reviews; они НЕ МОГУТ быть design/evidence source без operator reinstatement.

## 7. Доменные границы

Домены верхнего уровня: corpus/data, feature-extraction, evaluation/paired-audit, reporting/publication, site, research-governance. Текущий default owner — repository owner до появления конкретного domain doc.

`DOM-01 [PLANNER/АГЕНТ]` Каждая задача ОБЯЗАНА объявить primary и разрешённые external domains.

`DOM-02 [АГЕНТ]` Cross-domain dependency ОБЯЗАНА иметь producer/consumer ownership, grain, key/cardinality, temporal/null/delete semantics и contract verification.

`DOM-03 [АГЕНТ]` Одинаковое имя метрики, field или entity НЕ ЯВЛЯЕТСЯ доказательством общей семантики.

`DOM-04 [REVIEWER]` Review ОБЯЗАН проверить selector/tag/model/test leakage и неявное расширение shared layer.

Подробности: `../domains/BOUNDARIES.md`.

## 7.1. Context locality

`CTX-01 [PLANNER/АГЕНТ]` Maintainability/pruning/simplification task ОБЯЗАНА определить 3–5 representative scenarios и baseline packet: authoritative entry point, required files/sections, approximate tokens/bytes, domain transitions, central-config-only reads, duplicate sources и blurred boundaries.

`CTX-02 [АГЕНТ]` Claim «стало проще» ОБЯЗАН уменьшить хотя бы одну величину: production/test/docs LOC; required files/sections или tokens/bytes; domain transitions; public/operational entry points; authoritative sources; production concepts/duplication — при frozen behavior/safety.

`CTX-03 [АГЕНТ/REVIEWER]` File move/split, wrapper, registry, doc или checker НЕ СЧИТАЮТСЯ simplification, если обязательный packet не уменьшился или sources/entry points выросли.

`CTX-04 [АГЕНТ]` Локальное изменение СЛЕДУЕТ выполнять через один obvious capability entry point без чтения чужого домена, кроме explicit contract. Central-config debt не переписывается автоматически.

`CTX-05 [PLANNER/АГЕНТ]` Новая abstraction в pruning/simplification допустима только если заменяет минимум два механизма, не увеличивает cumulative production LOC без correctness justification, уменьшает packet и не создаёт public entry point/framework/state/dependency; иначе требуется reclassification.

`CTX-06 [АГЕНТ/REVIEWER]` Before/after metrics ОБЯЗАНЫ использовать одну воспроизводимую методику; стабильная оценка bytes/lines/tokens допустима.

`CTX-07 [АГЕНТ]` Новый authoritative doc/checker ОБЯЗАН назвать заменяемый источник. Correctness artifact без замены не засчитывается как reduction.

`CTX-08 [АГЕНТ]` Invariant СЛЕДУЕТ локализовать у owning capability/domain или публиковать одним contract. Параллельные определения требуют owner decision.

## 8. Evidence и confidence

`EVID-01 [АГЕНТ]` Существенный вывод ОБЯЗАН разделять observed fact, source, interpretation, confidence и consequence.

`EVID-02 [АГЕНТ]` Evidence ОБЯЗАНО быть reproducible и bounded: commit/file/line, command+artifact, bounded query+source/period или test node+result.

`EVID-03 [АГЕНТ]` Переписка, docs, handoff, reports и agent conclusions — гипотезы до проверки.

`EVID-04 [АГЕНТ]` Для заметного решения ОБЯЗАТЕЛЬНА попытка опровержения.

`EVID-05 [REVIEWER]` До research decision, merge и close reviewer/operator ОБЯЗАНЫ выборочно аудитировать ключевые claims.

High confidence допускает решения любого класса, но не заменяет gates. Medium — обратимые решения; R3a требует дополнительного evidence/waiver. Low — только hypotheses/questions.

`CONF-01 [АГЕНТ/ОПЕРАТОР]` Confidence ОБЯЗАН влиять на допустимое решение.

Минимальная evidence-запись:

```markdown
- Claim ID:
- Observed fact:
- Source:
- Reproduction:
- Interpretation:
- Confidence: high | medium | low
- Falsifier/negative check:
- Consequence:
```

## 9. Независимость и bounded review

`IND-01 [ОПЕРАТОР/REVIEWER]` Независимая проверка выполняется в clean session и предпочтительно отдельном worktree на baseline.

`IND-02 [ОПЕРАТОР]` Reviewer получает frozen AC/invariants, baseline, final diff, evidence/verification и constraints, но не выводы предыдущего reviewer.

`IND-03 [REVIEWER]` Независимый result фиксируется до сравнения с предыдущими выводами.

`REV-01 [REVIEWER]` Reviewer НЕ МОЖЕТ менять goal, frozen acceptance/non-goals, architecture scope или complexity budget.

`REV-02 [REVIEWER]` Blocker ОБЯЗАН содержать violated AC/invariant, baseline regression, supported unsafe counterexample или safety/approval violation, а также reproduction/evidence.

`REV-03 [REVIEWER]` Pre-existing debt, speculative hardening, style и alternative architecture НЕ МОГУТ блокировать текущую task без основания `REV-02`.

`REV-04 [PLANNER/АГЕНТ]` Default: один adversarial review и максимум один bounded correction. Оставшийся blocker даёт `blocked` или follow-up; новый loop требует operator decision и contract.

`REV-05 [REVIEWER]` Все passes сравниваются с исходным baseline; shifting baseline запрещён.

`REV-06 [REVIEWER]` Reviewer описывает counterexample/constraint, но не превращает finding в обязательный architecture recipe.

`REV-07 [DELETION REVIEWER]` Deletion review получает baseline, frozen acceptance, final diff и evidence и отвечает только, что можно удалить/упростить при сохранении контракта.

`REV-08 [АГЕНТ]` После deletion review не запускается новый architecture review; применённые удаления проходят frozen verification.

## 10. Production и безопасность

`SAFE-01 [АГЕНТ]` Production access по умолчанию read-only; bounded checks; DML/DDL/deploy/publication/external writes запрещены без разрешения.

`SAFE-02 [ОПЕРАТОР]` R3a требует ручного подтверждения непосредственно перед action.

`SAFE-03 [АГЕНТ]` Secrets, raw private/copyrighted corpus content и PII запрещены в Git/evidence/fixtures/reports/task/handoff.

`SAFE-04 [АГЕНТ]` Unbounded production/data scan запрещён без load plan и approval.

`SAFE-05 [АГЕНТ]` Неизвестный user WIP нельзя уничтожать, переписывать или смешивать с task.

Для Stylo production surface включает GitHub Pages publication и публичные scientific claims; governance/evidence changes требуют явного scope и review.

## 11. Реализация и проверка

`IMPL-01 [АГЕНТ]` До code ОБЯЗАТЕЛЬНО проверить physical/source/scientific contract, consumers и baseline behavior.

`IMPL-02 [АГЕНТ]` Изменение ОБЯЗАНО иметь targeted и relevant regression proof; static tests не заменяют runtime/scientific verification.

`IMPL-03 [АГЕНТ]` Checks ОБЯЗАНЫ фиксировать exact command, значимый environment/version и result.

`IMPL-04 [АГЕНТ]` Warning, skipped и cannot-run ОБЯЗАНЫ быть отделены от pass.

`IMPL-05 [АГЕНТ]` Bounded diff ОБЯЗАН измеряться от original baseline: production subsystems/files и production/test LOC added/deleted/net.

## 11.1. Pruning

`PRUNE-01 [PLANNER]` До mutation pruning task ОБЯЗАНА зафиксировать baseline, frozen behavior, paths/non-goals и reduction target.

`PRUNE-02 [АГЕНТ]` Pruning ОБЯЗАН быть deletion-first. Новые production frameworks/layers/state/dependencies/generic abstractions/files запрещены по умолчанию; необходимость требует reclassification.

`PRUNE-03 [АГЕНТ]` Cumulative delta всегда измеряется от original baseline.

`PRUNE-04 [АГЕНТ]` Production LOC net delta по умолчанию отрицательный; иначе честный `partially-done/blocked`, а не код ради cleanup.

`PRUNE-05 [АГЕНТ/REVIEWER]` Удаление tests/assertions/monitoring/safety/canonical docs не засчитывается без доказательства исчезновения контракта; test LOC не вычитается из production target.

`PRUNE-06 [REVIEWER]` Review ограничен frozen AC/regression/unsupported deletion/safety/tripwires; hardening и unrelated bugs — follow-up.

`PRUNE-07 [АГЕНТ]` Максимум два mutation passes: initial и один bounded correction; затем `done | partially-done | blocked`.

`PRUNE-08 [АГЕНТ]` Pre-existing bug не исправляется автоматически при расширении frozen scope.

`PRUNE-09 [DELETION REVIEWER]` Independent deletion review обязателен и не проектирует alternative architecture.

`PRUNE-10 [АГЕНТ]` Stop — достигнут target или больше нет доказуемо безопасных deletions.

`PRUNE-11 [PLANNER]` Repo-wide cleanup ОБЯЗАН раскладываться на отдельно выбранные bounded tasks; backlog не расширяет текущий scope.

`PRUNE-12 [PLANNER/ОПЕРАТОР]` Progress измеряется cumulative production delta, removed files/concepts/duplication и regression status, не количеством review rounds.

`PRUNE-13 [PLANNER/АГЕНТ]` Maintainability pruning target ОБЯЗАН включать representative packet и target по files/sections, tokens/bytes, domain transitions, entry points или sources.

`PRUNE-14 [АГЕНТ]` Context growth, новый public helper, oversized test harness, redesign соседнего capability или сложный rollback — stop/reclassification tripwire.

## 11.2. Bounded simplification campaigns

`SIMPL-01 [PLANNER/АГЕНТ]` Campaign ОБЯЗАНА начинаться с clean baseline и read-only audit; до human selection запрещены mutations files/Git/external systems.

`SIMPL-02 [PLANNER]` Audit contract ОБЯЗАН задать representative packets, baseline LOC/entry points, boundaries, project-specific hazards/non-goals, false positives и limits. Defaults: максимум 15 findings и 6 candidate fixes.

`SIMPL-03 [PLANNER]` Candidate ОБЯЗАН описывать одну problem/effect, 3–7 frozen AC, allowed/forbidden paths, budgets/delta, packet effect, smoke, rollback и tripwire.

`SIMPL-04 [PLANNER/ОПЕРАТОР]` Fix входит в wave только если уменьшает `CTX-02`; correctness positive delta классифицируется отдельно и не считается simplification progress.

`SIMPL-05 [PLANNER]` До selection frozen cumulative defaults: production delta `<= 0`, test `<= 0`, docs `< 0`, new production files/public entry points/frameworks/generic checkers/runtime states `= 0`; deviations require operator justification.

`SIMPL-06 [АГЕНТ]` Каждый fix выполняется в отдельном worktree/branch от общего baseline и сохраняется accepted commit либо `blocked/rejected`.

`SIMPL-07 [REVIEWER/АГЕНТ]` На fix допускаются один adversarial review, один correction pass и independent deletion review; new hardening/architecture — backlog без `REV-02` blocker.

`SIMPL-08 [АГЕНТ/REVIEWER]` Integration применяет только accepted commits, пересчитывает cumulative LOC/context, проверяет cross-fix conflicts, relevant full flow и combined deletion review; новые findings не исправляются в wave, кроме introduced regression.

`SIMPL-09 [PLANNER/ОПЕРАТОР]` Campaign ОБЯЗАНА иметь finite stops: max fixes, exhausted budget, no measurable next improvement, two blocked fixes или cosmetic-only remainder.

`SIMPL-10 [АГЕНТ/ОПЕРАТОР]` Следующая wave НЕ МОЖЕТ запускаться автоматически; нужны новый state/context snapshot, priority selection и contract.

## 12. ADR, domains, runbooks и handoff

`DOC-01 [АГЕНТ]` Долговечное architecture/scientific semantic решение ОБЯЗАНО фиксироваться ADR; accepted ADR меняется superseding ADR.

`DOC-02 [АГЕНТ]` Domain semantics, grain, keys и ownership ОБЯЗАНЫ быть в domain doc.

`DOC-03 [АГЕНТ]` Эксплуатационная процедура ОБЯЗАНА быть в runbook.

`DOC-04 [АГЕНТ]` Handoff ОБЯЗАН быть коротким, связывать verified baseline, branch/worktree state at capture, active task baseline, material changes и stale/revalidation conditions и служить только следующей сессии. Он не обязан и не может заранее содержать собственный будущий commit.

`DOC-05 [АГЕНТ]` Завершённая история НЕ ДОЛЖНА накапливаться в handoff.

## 13. Definition of Done

Задача завершена только если выполнены применимые требования:

- `DOD-01` Контракт удовлетворён либо deviation явно принят.
- `DOD-02` Scope/domain boundaries соблюдены; creep отсутствует или оформлен.
- `DOD-03` Key claims имеют reproducible evidence; audit выполнен.
- `DOD-04` Targeted/regression checks выполнены; skips/warnings/unknowns отделены.
- `DOD-05` Negative scenario/falsifier выполнен для существенного решения.
- `DOD-06` R3 approvals, safety и rollback/abort выполнены.
- `DOD-07` Result фиксирует changes, evidence, residual risk и Git state.
- `DOD-08` ADR/domain/runbook/handoff обновлены только по каноническому назначению.
- `DOD-09` Independent review выполнен для L/R3 и задач, где указан.
- `DOD-10` Result воспроизводим без transcript.
- `DOD-11` Blockers соответствуют `REV-*`, budget соблюдён, hardening/debt не расширили scope.
- `DOD-12` Implementation/pruning tripwires и cumulative diff metrics зафиксированы.
- `DOD-13` Pruning выполнил `PRUNE-*`, deletion review и reduction либо честный partial/block.
- `DOD-14` Maintainability claim имеет одинаково измеренные context packets before/after.
- `DOD-15` Simplification campaign выполнила audit, selection, bounded fixes, integration, metrics и stop без auto-wave.

Операционные checklist могут ссылаться на IDs, но не определять другой DoD.

## 14. Waiver

`WAIV-01 [ОПЕРАТОР]` Невыполненный MUST требует waiver с requirement ID, reason, scope, approver, expiry, accepted risk и compensating controls.

`WAIV-02 [ВЛАДЕЛЕЦ]` Бессрочный waiver запрещён; expired считается отсутствующим.

```markdown
## Waiver <ID>
- Requirement ID:
- Scope/task:
- Reason:
- Accepted risk:
- Compensating controls:
- Approver:
- Approved at:
- Expires at:
- Close condition:
- Status: active | expired | closed
```

## 15. Механизмы контроля

| Требование | Trigger | Control | Ответственный |
|---|---|---|---|
| TASK-01/02 | До M/L/R2/R3 mutation | task contract + review | Исполнитель |
| CLASS-02 | Новый risk flag | metadata diff + gate | Исполнитель/оператор |
| DOM-01/02 | Cross-domain scope | fields + contract tests | Исполнитель/reviewer |
| EVID-05 | Перед decision/merge/close | evidence audit | Reviewer/operator |
| SAFE-01/02 | Production/publication action | read-only + approval | Operator |
| SAFE-03 | Каждый commit/CI | existing release hygiene/secret controls | Repo owner |
| DOC-04 | Session end/material Git change | commit freshness check | Исполнитель |
| TASK-05..08 | До mutation/clean restart | tripwires/frozen AC/exclusions | Planner/исполнитель |
| REV-01..08 | Review/correction | blocker mapping + pass budget | Reviewer/operator |
| PRUNE-01..14 | Pruning | cumulative metrics + deletion review | Planner/agent/reviewer |
| CTX-01..08 | Maintainability claim | representative packets | Planner/agent/reviewer |
| SIMPL-01..10 | Campaign | audit/selection/wave/integration gates | All roles |
| DOD-* | Перед close | task checklist | Исполнитель/reviewer |

Automation проверяет поля, но не качество решения.

## 16. Метрики v1.3

Для L/pruning/simplification initially collect: unreproducible claims; late requirements; reclassifications; rules moved to gates; defects found pre-merge; correction passes; production LOC added/deleted/net; new files/concepts/framework/state/dependencies; deletion-review removals; representative packet files/sections/tokens before/after; domain transitions/entry points/sources before/after; candidate/selected/accepted/blocked fixes и stop reason. Baselines не используются для оценки людей.

## 17. Экспериментальные требования

Experiment MUST имеет owner, hypothesis, максимум 5 применимых задач или 4 недели, cost/prevented errors/workarounds и exit decision.

`GOV-01 [ВЛАДЕЛЕЦ]` Временное правило без expiry и exit criteria запрещено.

## 18. Запрещённые практики

Ritual artifacts; rule без trigger/owner/control; competing DoD; reviewer scope creep; hardening blocker без counterexample; cleanup с architecture growth без reclassification; context laundering; authoritative doc/checker без consolidation; campaign без selection/budget/stop; automatic next wave; rejected artifacts as source; agent count as truth; hidden mutation; domain leakage; confidence вместо approval; handoff-history; MUST в guide; close при stale evidence.

## 19. Встроенные форматы и журнал

### 19.1. Минимальный ADR

Создавать в `docs/adr/<id>-<slug>.md` только по `DOC-01`.

```markdown
# ADR <ID>: <Title>
- Status: proposed | accepted | rejected | superseded
- Date: YYYY-MM-DD
- Owners: <human owners>
- Related task: <path>
- Supersedes / superseded by: <ADR or none>

## Context
<Observed constraints and facts.>
## Decision
<One durable decision and boundaries.>
## Alternatives considered
- <option>: <trade-off>
## Consequences
### Positive
- ...
### Negative / accepted debt
- ...
## Verification and revisit triggers
- Evidence:
- Revisit conditions:
```

### 19.2. Минимальный runbook

Создавать в `docs/runbooks/<slug>.md` только для повторяемой операции. Он содержит prerequisites, permissions, bounded steps, verification, rollback/abort и escalation.

### 19.3. Журнал стандарта

| Версия | Дата | Владелец | Изменение | Основание |
|---|---|---|---|---|
| 1.3 | 2026-08-10 | Dmitry Purtov (repository owner) | Добавлены representative packets, `CTX-*`, bounded campaigns `SIMPL-*`, selection/integration gates и защита от context laundering. | Maintainability измеряется стоимостью безопасного изменения, не только LOC/files. |
| 1.2 | previous | Dmitry Purtov (repository owner) | Frozen acceptance, tripwires, pruning и anti-loop review. | Защита от scope creep и unbounded hardening. |
| 1.1 | previous | Dmitry Purtov (repository owner) | Нормативное ядро, task/domain/handoff/chat и evidence/waiver/ADR formats. | Предыдущая версия процесса. |
