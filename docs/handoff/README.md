# Handoff policy

Handoff — точка входа следующей сессии, а не история проекта и не evidence base.

## Каноническая точка входа

`docs/handoff/CURRENT.md`

Он должен связывать проверенный baseline commit с branch/worktree state at capture, baseline активной задачи и material changes после baseline; помещаться примерно в 150 строк / 10 KiB; перечислять только active state/blockers/next gates; ссылаться на task/ADR/domain/runbook/evidence и не содержать secrets, PII, raw private/copyrighted corpus data. Tracked `CURRENT.md` не содержит и не обязан предсказывать собственный будущий HEAD.

## Допустимая структура каталога

```text
docs/handoff/
├── CURRENT.md                 # единственная обязательная точка входа
├── commit-map.md              # только при сложной ветке
├── operations-and-next.md     # только если CURRENT перегружен
└── evidence/                  # bounded evidence artifacts + hashes
```

Дополнительные файлы создаются только реальной задачей. Architecture lineage принадлежит architecture docs; durable decisions — `docs/adr/`; semantics — `docs/domains/`; operations — `docs/runbooks/`.

## Freshness triggers

CURRENT обновляется в конце незавершённой сессии, после material HEAD/branch/worktree change, при изменении blocker/approval/next gate и перед передачей другому исполнителю. Закрытая задача не добавляет историю: она удаляется из active, при необходимости остаётся ссылка на task result.

## Проверка freshness

1. Сравнить verified baseline и active task baseline с Git history/HEAD.
2. Сопоставить описанные material changes и branch/worktree state at capture с текущими HEAD и `git status`.
3. Проверить active task statuses.
4. Выполнить перечисленные в handoff revalidation conditions и воспроизвести key claims либо пометить их stale.
5. Не начинать mutation по stale handoff без обновления context.

## Разнос содержимого

| Содержимое | Каноническое место |
|---|---|
| Completed changes/tests | task result / Git history |
| Durable decision | ADR |
| Grain, metrics, source/scientific semantics | domain doc/governance registry |
| Deploy/backfill/rollback procedure | runbook |
| Detailed audit | report + evidence |
| Context baseline/simplification findings | campaign task result + evidence |
| Active blockers/next gate | CURRENT.md |

## Read-only generation

```bash
git rev-parse --show-toplevel
git branch --show-current
git rev-parse HEAD
git status --short --branch
git log -n 10 --oneline --decorate
```

Git output не доказывает scientific semantics; active tasks, domain/ADR/runbook references и claims проверяются отдельно.
