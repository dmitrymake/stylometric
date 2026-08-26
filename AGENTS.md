# Repository Agent Instructions

Инструкции для любой coding environment, работающей в этом репозитории.

## Проект

**Stylo** — исследовательский инструмент для воспроизводимого сравнения авторской манеры русской
прозы. Выдаёт ранжированный список кандидатов и явно ограничивает область обоснованности результата.

Стек: CPython 3.11, spaCy, scikit-learn, NumPy/Pandas, pytest; Node.js 22, Vite и React для `site/`.

```
uv venv --python 3.11 && uv pip install --constraint requirements.lock -e ".[dev]"
.venv/bin/python -m spacy download ru_core_news_lg

.venv/bin/python -m pytest <test-node> -q -p no:cacheprovider     # targeted
.venv/bin/python -m pytest tests -q -p no:cacheprovider           # полный regression
.venv/bin/python -m build --no-isolation                          # пакет
cd site && npm ci --no-audit --no-fund && npm run gen && npm run build
```

Отдельный formatter/linter не настроен. Если команда неизвестна — не выдумывай её, а найди
подтверждение в исполняемой конфигурации или спроси владельца.

## Три гейта

Это единственные механические защиты; они проверяют байты, а не формулировки:

```
.venv/bin/python scripts/check_release_hygiene.py --publish-ref HEAD   # приватный корпус вне Git
.venv/bin/python scripts/check_executable_source_inventory.py          # состав и хеши исходников
node scripts/gen-site-data.mjs && node scripts/check-provenance.mjs    # числа сайта = источники
git diff --exit-code -- site/src/generated
```

Удаление или добавление `.py` под `scripts/`, `src/stylo/`, `tests/` требует пересчёта
`release_python_file_count` и `release_python_paths_sha256` в `release/executable_sources.json`
(значения печатает `check_executable_source_inventory.py --show-paths`). Правка любого
`research/governance/*.json` требует обновления его хеша в том же файле.

## Что нельзя

- Raw/private/copyrighted тексты и персональные корпуса не попадают в Git, задачи, отчёты, фикстуры
  и транскрипты: `input*`, `_staging_corpora`, `data/frags_*`. Результаты только агрегированные.
- `.env`, ключи и токены не читаются и не включаются в evidence.
- Публикация (GitHub Pages), push, изменение публичных научных заявлений и любые внешние записи —
  только с явного разрешения владельца, запрошенного непосредственно перед действием.
- Не уничтожать пользовательский WIP: reset, rebase, clean, удаление веток и stash — только по
  явному разрешению.

Защищённые области: `input*`, `_staging_corpora`, `data/`, `research/governance/`,
`research/evidence/`, `.github/workflows/`, `site/`.

## Как работать

- Точка входа сессии — `docs/handoff/CURRENT.md`. Это указатель, а не источник истины: сверь
  baseline с HEAD и состоянием дерева, прежде чем на него опираться.
- Задачи живут в `docs/tasks/`. Заводи файл, когда работа переживёт одну сессию или когда нужен
  след «кто разрешил и что измерено». На тривиальную правку файл не нужен.
- Домены и их границы: `docs/domains/BOUNDARIES.md`.
- Текущий научный статус: `research/governance/status_ledger.json`, план — `research/ROADMAP.md`.

## Доказательства

Для существенного вывода разделяй наблюдаемый факт, источник, интерпретацию и следствие. Источник
должен быть воспроизводим: `file:line`, commit, команда, test node. Переписка, старый handoff и
вывод другого агента — гипотезы, пока не проверены. Warnings и пропущенные проверки фиксируются
отдельно от прошедших, а не растворяются в «всё зелёное».

Не выдавай mutation за исследование: если просили изучить или оценить — не меняй код. Число агентов
или запросов не является покрытием. Неизвестное называй неизвестным.
