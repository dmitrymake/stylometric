# Repository Agent Instructions

Эти инструкции обязательны для любой coding environment, выполняющей инженерную работу в этом репозитории. Они platform-neutral и не зависят от конкретной модели или поставщика.

Нормативное ядро процесса: `docs/agentic/STANDARD.md`.
Справочники: `docs/agentic/`, `docs/domains/`, `docs/adr/`, `docs/runbooks/`, `docs/handoff/`.

## 0. Профиль проекта

- Проект: **Stylo**
- Назначение: исследовательский инструмент для воспроизводимого сравнения авторской манеры русской прозы. Он выдаёт ранжированный список кандидатов и явно ограничивает область обоснованности результата.
- Тип репозитория: single research project / Python library with a static publication site
- Основной стек: CPython 3.11, setuptools, spaCy, scikit-learn, NumPy/Pandas, pytest; Node.js 22, Vite и React для `site/`
- Установка зависимостей: `uv venv --python 3.11 && uv pip install --constraint requirements.lock -e ".[dev]"`
- Дополнительный NLP runtime: `.venv/bin/python -m spacy download ru_core_news_lg`
- Форматирование/линтинг: `UNKNOWN — отдельный formatter/linter не настроен; не подменять им CI hygiene gates`
- Targeted tests: `.venv/bin/python -m pytest <test-node> -q -p no:cacheprovider`
- Полный regression: `.venv/bin/python -m pytest tests -q -p no:cacheprovider`
- Release hygiene: `.venv/bin/python scripts/check_release_hygiene.py --publish-ref HEAD && .venv/bin/python scripts/check_executable_source_inventory.py`
- Build/compile: `.venv/bin/python -m build --no-isolation`; для сайта: `cd site && npm ci --no-audit --no-fund && npm run gen && npm run build`
- Основная task-директория: `docs/tasks`
- ADR: `docs/adr`
- Domain docs: `docs/domains/`
- Runbooks: `docs/runbooks`
- Защищённые области: raw/private corpora (`input*`, `_staging_corpora`, локальные `data/`), scientific governance/evidence, release/publication artifacts, `.github/workflows/`, `site/` publication surface и внешние публикации
- Production/data restrictions: в репозитории не обнаружен production application runtime; GitHub Pages deploy является production publication. Raw/private/copyrighted тексты и персональные корпуса не помещаются в Git. Production publication, изменение governance claims и внешние writes требуют явного разрешения.

Если команда неизвестна или отсутствует, не выдумывай её. Сначала найди подтверждение в исполняемой конфигурации или запроси владельца.

## 1. Приоритет инструкций

Используй более узкую и более конкретную инструкцию, если она не противоречит более высокому уровню:

1. явная инструкция оператора для текущей работы;
2. утверждённый task-файл;
3. ближайший к изменяемым файлам локальный `AGENTS.md`;
4. этот корневой `AGENTS.md`;
5. `docs/agentic/STANDARD.md`;
6. справочные guides и templates.

Инструкции безопасности, ограничения среды и права доступа не могут быть отменены task-файлом. При конфликте документов зафиксируй конфликт и останови зависимое решение.

## 2. Обязательный старт сессии

До содержательной работы:

1. найди фактический Git root;
2. зафиксируй branch, HEAD и состояние index/worktree;
3. прочитай утверждённый task-файл или явно работай в режиме `context-only`;
4. прочитай `docs/handoff/CURRENT.md`, сравни verified baseline и описанные material changes с HEAD/worktree и выполни указанные revalidation checks;
5. прочитай только релевантные domain docs, ADR и runbooks;
6. найди локальные `AGENTS.md` в изменяемых областях;
7. перечисли доступные и недоступные каналы доказательств.

Handoff с несовпадающим baseline/material-change состоянием является указателем, а не фактом. Если Git root не найден, не запускай команды как будто checkout найден.

## 3. Режим работы

В начале зафиксируй один режим:

- `context-only` — только инвентаризация и state packet; никаких изменений;
- `research` — исследование и evidence; никаких product changes;
- `implementation` — изменения в пределах утверждённого контракта;
- `pruning` — bounded deletion/simplification существующей реализации по frozen behavior;
- `review` — независимая проверка baseline..result; без исправлений, если отдельно не разрешено;
- `incident` — bounded-диагностика и минимальное безопасное восстановление.

Не переходи из research/context-only/review в implementation без явного gate в task-файле или новой инструкции оператора.

## 4. Task contract

Для `M`, `L`, `R2`, `R3b` и `R3a` должен существовать task-файл. Для `S/R1` допускается short contract из `docs/agentic/TASK_TEMPLATE.md`.

До изменения кода контракт обязан определить цель и observable result; baseline; scope/non-goals; domains и invariants; risk; frozen acceptance; verification и negative scenarios; complexity tripwires; review correction budget; stop conditions; требуемые ADR/domain/runbook/handoff updates. Tripwires включают число production subsystems/files, production/test/docs LOC budget, политику на framework/state/dependency/concepts/public entry points и, для maintainability, representative context packets.

После gate acceptance не расширяется reviewer'ом. Изменение риска, scope или tripwire требует остановки и reclassification.

## 5. Доменные границы

Политика: `docs/domains/BOUNDARIES.md`. Известные верхнеуровневые домены: `corpus/data`, `feature-extraction`, `evaluation/paired-audit`, `reporting/publication`, `site`, `research-governance`. Ownership отдельных capability уточняется в первой существенной задаче; до этого owner — repository owner.

Каждая задача и изменяемый артефакт имеют primary domain. Cross-domain работа должна перечислять producer/consumer, ownership, grain, key/cardinality, time, null/delete/late-arrival semantics, направление зависимости и contract verification.

Запрещено переносить метрику по совпадению имени, копировать доменные filters/macros как универсальные, соединять факты без утверждённого bridge, прятать бизнес-семантику в shared layer и расширять selectors/tags с неявной утечкой.

## 5.1. Context locality и simplification

Для локальной задачи загружай минимально достаточный context packet: authoritative capability entry point, владеющий код/конфигурацию/тесты и только обязательные контракты. Repo-wide чтение не является стартовым режимом.

Если цель — поддерживаемость или сокращение сложности:

- выбери 3–5 representative scenarios и измерь required files/sections, примерный tokens/bytes, domain transitions, entry points, authoritative sources и central-config-only reads;
- сначала проведи read-only audit; mutation разрешена только после human selection bounded fixes;
- выбранные fixes выполняй в отдельных worktree/branches от общего baseline и интегрируй только принятые commits;
- новые findings отправляй в backlog; следующая волна автоматически не запускается.

Перенос текста, дробление файла, wrapper, checker или дополнительный документ не являются упрощением без измеримого уменьшения context packet. Excluded/rejected artifacts не читаются и не переиспользуются без нового решения оператора.

## 6. Evidence и выводы

Для существенных выводов разделяй observed fact, source/evidence, interpretation, confidence и consequence. Источник должен быть воспроизводим: file:line/commit, команда, bounded SQL + source/period, test node + result artifact или endpoint + безопасное summary.

Переписка, report, старый handoff и вывод другого агента являются гипотезами до проверки. Для заметного решения выполни попытку опровержения. Для L/R3 или спорной семантики используй независимую проверку в чистом контексте.

## 7. Production и чувствительные данные

- Production access по умолчанию read-only; проверки bounded.
- Production DML/DDL, deploy, publication, secret rotation и external writes запрещены без явного разрешения.
- Raw/private/copyrighted/PII тексты не копируются в Git, task, report, fixture или transcript; результаты агрегируются.
- `.env`, token stores, keys и credentials не читаются и не включаются в evidence.
- R3a требует ручного подтверждения repository owner непосредственно перед необратимым действием.
- Неограниченный scan запрещён; если bounded-проверка невозможна, сначала нужен load plan и approval.

## 8. Git и изоляция работы

Перед изменением зафиксируй baseline; не уничтожай пользовательский WIP; не выполняй reset/rebase/clean/delete без явного разрешения. Для независимого review или параллельной реализации используй отдельный worktree/branch. В simplification campaign каждый fix стартует от общего baseline. Не смешивай unrelated fixes и не называй commit atomic, если evidence появился лишь позже.

При неизвестном WIP в scope останови mutation и предложи отдельный worktree, patch-only, research-only или новый baseline.

## 9. Изменения и тестирование

До кода проверь physical/source/business contract и consumers. Для изменения определи failure mode; добавь падающую до fix или независимую проверку; выполни targeted test, релевантный regression и, при необходимости, реальный runtime; сохрани команды и результаты. Static/unit pass не доказывает scientific/business correctness или production readiness. Warnings/skips фиксируются отдельно.

## 10. ADR

Создай ADR для долговечного решения, меняющего architecture boundary, ownership, storage/transport/materialization, каноническую научную/бизнес-семантику или существенный future constraint. Статус задачи, список файлов, временный workaround, run instruction и доменное определение без архитектурного выбора ADR не являются. Accepted ADR заменяется новым со статусом `supersedes`.

## 11. Handoff

Точка входа: `docs/handoff/CURRENT.md`; политика: `docs/handoff/README.md`. Handoff короток, связывает verified baseline, branch/worktree state at capture, active task baseline и material changes, а также содержит blockers, next actions, revalidation conditions и ссылки. История — в task/Git, решения — в ADR, semantics — в domain doc, operations — в runbook. При закрытии обновляй handoff только если остаётся незавершённое состояние.

## 12. Review

Review проводится против frozen task contract, исходного baseline, итогового diff, evidence и выполненных checks. Blocking finding допустим только при нарушении frozen criterion/invariant, воспроизводимой regression, unsafe counterexample на supported path или явного safety gate. Pre-existing debt, speculative hardening и идеи на будущее — non-blocking follow-up.

По умолчанию разрешён один adversarial review и один bounded correction pass. Для pruning обязателен отдельный deletion review; после него выполняется frozen verification, а не новый architecture-review loop. Пересказ diff не является review.

## 13. Завершение

Единственный нормативный Definition of Done находится в `docs/agentic/STANDARD.md` (`DOD-*`). Перед close выполни evidence audit, зафиксируй result/risks/Git state, обнови назначенные artifacts и покажи status. Не объявляй production-ready без production gates.

## 14. Запрещённые практики

- mutation при запросе изучить/оценить/собрать состояние;
- число агентов или запросов как evidence coverage;
- скрытие unknown за high confidence;
- дублирующий DoD или обязательные MUST вне стандарта;
- reviewer-induced scope creep и correction passes сверх budget;
- pruning через новый framework/state/abstraction;
- context laundering: перенос/дробление без сокращения обязательного context packet;
- новый authoritative source без замены более слабого;
- автоматическая следующая simplification wave;
- использование rejected/stashed AI diff без reinstatement;
- продолжение после нового R3-фактора без reclassification;
- handoff как источник истины.
