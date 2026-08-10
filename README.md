# Stylo

Stylo — исследовательский инструмент для сравнения авторской манеры русской прозы. Он измеряет
синтаксис, пунктуацию, употребление служебных слов и другие признаки текста, сводит их в один вектор и
сопоставляет спорный текст с корпусом известных авторов. На выходе — ранжированный список кандидатов и
явно указанные границы, за которыми такое сопоставление уже не обосновано.

[Интерактивная статья](https://stylometry.russkiykod.com/)

## Статус исследования

Результаты первого эксперимента сохранены как исторические.

Поздний аудит корпуса обнаружил одинаковое и вложенное содержание под разными идентификаторами
произведений: один и тот же текст встречался под несколькими названиями, а отдельные произведения
входили в состав сборников. Поэтому прежние accuracy и macro-F1 не являются текущей оценкой качества
модели. Новый расчёт на корпусе, разделённом по содержанию, ещё не опубликован. Прежний интервал
macro-F1 отозван.

Идентификатора книги для разделения обучения и проверки оказалось недостаточно. В новом протоколе
тексты с совпадающим содержанием объединяются в одну группу, и вся группа целиком уходит либо в
обучение, либо в проверку.

Исторические артефакты и их контрольные суммы задним числом не переписываются. Нормативный текущий
статус хранится отдельно от них, в машиночитаемых записях:

- [research/evidence/ineligible_corpus_registrations_v1.json](research/evidence/ineligible_corpus_registrations_v1.json)
  — регистрация корпуса, непригодного для новых научных прогонов.
- [docs/macro_f1_ci_withdrawal.json](docs/macro_f1_ci_withdrawal.json) — запись об отзыве интервала
  macro-F1.
- [research/governance/status_ledger.json](research/governance/status_ledger.json) — текущий статус
  работ.

## Возможности

- Подготовка корпуса и строгая проверка его целостности: кодировка, нормализация, дубли, вложенные
  тексты, происхождение и контрольные суммы файлов.
- Извлечение языковых признаков отдельными блоками и сборка вектора из тех блоков, что включены в
  конфигурации расчёта.
- Остановка вычислений до обучения, когда обучающая и проверочная части пересекаются по содержанию.
- Диагностический анализ спорных текстов: ранжирование кандидатов, оценка неопределённости и явно
  указанные границы вывода. Результат такого анализа — направленное свидетельство при названных
  допущениях, а не доказательство авторства.

## Быстрый старт

```bash
uv venv --python 3.11
uv pip install --constraint requirements.lock -e ".[dev]"
.venv/bin/python -m spacy download ru_core_news_lg
.venv/bin/python -m pytest -q
.venv/bin/stylo --help
```

Поддерживаемый runtime зафиксирован в `.python-version` — CPython 3.11. Файл `requirements.lock`
используется именно как constraints-файл, как показано выше. Модель spaCy ставится отдельно: она не
входит в зависимости пакета. Команда `stylo --help` перечисляет доступные подкоманды: проверку
корпуса, очистку, разбиение, обучение, предсказание, оценку и сборку отчёта.

Прогон `pytest -q` не требует корпуса: он проверяет модульную логику, контракты артефактов и релизную
гигиену репозитория.

Тексты корпуса в Git не входят. Полный научный прогон требует отдельно собранного корпуса, прошедшего
проверку целостности; загрузка исходников и вычисления разделены на отдельные шаги, поэтому чистый
клон репозитория сам по себе исторические результаты не воспроизводит.

## Структура репозитория

- `src/stylo/` — библиотека: корпус, признаки, модели, оценка, пайплайн.
- `scripts/` — исполняемые точки входа, гейты релизной гигиены и генератор данных сайта.
- `configs/` — параметры расчётов.
- `docs/` — машинные результаты и артефакты кейсов.
- `research/` — протоколы, governance и evidence.
- `site/` — исходники интерактивной статьи.
- `tests/` — контрактные и регрессионные тесты.

## Данные и лицензия

Код распространяется по лицензии MIT. Тексты корпуса вместе с репозиторием не распространяются: у
источников собственные лицензионные и правовые ограничения, различающиеся по авторам и юрисдикциям.
Воспроизводимость привязана к манифестам, хеш-суммам и receipt-записям прогонов, а не к включению
текстов в Git.

## Agentic Engineering Kit

В репозитории установлен platform-neutral Agentic Engineering Kit **v1.3** (2026-08-10). Product README выше остаётся канонической картой Stylo; этот раздел описывает только инженерный процесс.

### Точное размещение

```text
<repository-root>/
├── AGENTS.md
├── CHAT_INSTRUCTIONS.md
├── README.md
└── docs/
    ├── agentic/
    │   ├── STANDARD.md
    │   └── TASK_TEMPLATE.md
    ├── domains/
    │   └── BOUNDARIES.md
    └── handoff/
        ├── README.md
        └── CURRENT.md
```

Все пути относительны к фактическому Git root. Ответственность файлов:

| Файл | Назначение |
|---|---|
| `AGENTS.md` | Точка входа для coding sessions: профиль, режимы, safety, evidence, domains, review и handoff. |
| `CHAT_INSTRUCTIONS.md` | Инструкция внешнему управляющему контуру по формированию task contract из intent и state packet. |
| `README.md` | Product overview и карта установленного комплекта. |
| `docs/agentic/STANDARD.md` | Единственный нормативный источник requirement IDs и Definition of Done. |
| `docs/agentic/TASK_TEMPLATE.md` | Short/full/pruning/context-simplification contracts. |
| `docs/domains/BOUNDARIES.md` | Domain isolation, cross-domain contracts и шаблон конкретного domain doc. |
| `docs/handoff/README.md` | Политика freshness и разделения handoff/artifacts. |
| `docs/handoff/CURRENT.md` | Короткое текущее состояние, привязанное к commit. |

### Первый рабочий цикл

1. Открой coding session в Git root и попроси её прочитать `AGENTS.md`.
2. Собери проверяемое состояние в режиме `context-only`; формат state packet встроен в `CHAT_INSTRUCTIONS.md`.
3. Передай внешнему управляющему контуру intent, state packet, `CHAT_INSTRUCTIONS.md` и при необходимости стандарт.
4. Сохрани утверждённый контракт в `docs/tasks/YYYY-MM-DD-<slug>.md`.
5. Выполни работу от зафиксированного baseline, затем review против контракта, diff и evidence.
6. Обнови `docs/handoff/CURRENT.md` только если остаётся незавершённое состояние.

ADR создаются по необходимости в `docs/adr/`, domain contracts — в `docs/domains/<domain>.md`, runbooks — в `docs/runbooks/`. Эти каталоги не требуют пустых файлов.

### Fresh, upgrade и repair

Один bootstrap обслуживает новый комплект (`fresh`), безопасно объединяет старую версию с project-specific правилами (`upgrade`) и восстанавливает неполный manifest (`repair`). Успех требует физического наличия всех восьми путей; legacy `STANDART.md` не заменяет `STANDARD.md`.

Pruning и context simplification не создают девятый постоянный файл. Нормативные `PRUNE-*`, `CTX-*`, `SIMPL-*`, representative context packets и bounded flow `read-only audit → human selection → fixes → integration → stop` находятся в стандарте и task template.

### Проверка комплекта

```bash
missing=0
for f in \
  AGENTS.md CHAT_INSTRUCTIONS.md README.md \
  docs/agentic/STANDARD.md docs/agentic/TASK_TEMPLATE.md \
  docs/domains/BOUNDARIES.md docs/handoff/README.md docs/handoff/CURRENT.md
do
  test -f "$f" || { echo "MISSING: $f" >&2; missing=1; }
done
test "$missing" -eq 0

grep -RInE '\{\{[A-Z0-9_]+\}\}' \
  AGENTS.md CHAT_INSTRUCTIONS.md README.md docs/agentic docs/domains docs/handoff
```

Документы не заменяют существующие CI, scientific governance, release/security gates или непосредственное human approval для R3a. Handoff и state packet не являются runtime truth; reviewer не расширяет frozen acceptance; file splitting не является simplification без уменьшения context packet или authoritative sources; следующая cleanup wave требует нового выбора владельца.
