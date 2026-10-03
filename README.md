# Stylo

Stylo сравнивает авторскую манеру русской прозы и строит диагностический рейтинг заданных
кандидатов. Рейтинг не является вероятностью авторства; применимость зависит от корпуса,
состава панели и условий проверки. [Интерактивная статья](https://stylometry.russkiykod.com/).

## Установка

Проверенное окружение — CPython 3.11 на Linux. Из корня проекта:

```bash
uv venv --python 3.11 --seed
uv pip install --constraint requirements.lock -e ".[dev]"
.venv/bin/python -m spacy download ru_core_news_lg
```

Модель spaCy устанавливается отдельно. Git для работы установленного пакета не требуется.

## Анализ текста

Положите UTF-8 тексты в `input/<author_id>/<work_id>.txt`: одно произведение в одном файле.
Исследуемую книгу сохраните, например, как `input/unknown/target.txt`.
Нужны несколько независимых произведений каждого кандидата; целевая книга, её переиздания
и содержащие её сборники должны быть исключены из эталонов.

Создайте `local/case.yaml`, заменив имена кандидатов на имена каталогов:

```yaml
deployment:
  candidate_authors: [author_a, author_b]
evaluation:
  training_weighting: work_balanced
paths:
  data: local/case/data
  docs: local/case/results
```

```bash
.venv/bin/stylo analyze --config local/case.yaml --target-work unknown/target
```

Команда проверяет корпус, откладывает выбранную цель, обучает модели и сохраняет JSON/HTML-отчёт.
Пересечение цели с эталонами останавливает анализ до обучения. LR и Delta показываются отдельно.
Каждый результат сохраняется самостоятельной версией; команда выводит путь к HTML и `result_id`.
Для повторного чтения сохранённой версии достаточно каталога результатов:

```bash
.venv/bin/stylo report --config local/case.yaml --result-id <result_id>
```

Отдельные этапы и параметры доступны через `stylo --help` и `stylo <команда> --help`.
Пути конфигурации разрешаются относительно рабочего каталога запуска.

## Синтетический пример

```bash
.venv/bin/python scripts/make_demo_corpus.py --output local/demo
.venv/bin/stylo analyze --config local/demo/case.yaml --target-work unknown/target_periodic
```

Генератор создаёт оригинальные тексты вымышленных авторов в новой директории.
Пример проверяет рабочий путь, а не точность на литературе.

## Воспроизвести сравнение методов

Каталог `research/corpora/publication_prose_v1.json` задаёт цифровые источники, заявленные
издания и фиксированное разделение 33 произведений. Тексты скачиваются только локально:

```bash
.venv/bin/python scripts/fetch_publication_prose.py --output local/prose-panel
PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/evaluation/run_publication_comparison.py \
  --catalog research/corpora/publication_prose_v1.json \
  --text-root local/prose-panel/texts --output local/results/comparison.json
```

Сопоставляются восемь фиксированных методов: покнижная проверка на 22 работах и отдельная
оценка 11 отложенных произведений. Машинные агрегаты находятся в
`research/evidence/publication_comparison_v1/`; они нужны статье и проверке источников чисел.
Дополнительный `run_publication_length_check.py` использует только работы для разработки.
Этот небольшой корпус не проверяет неизвестных авторов, тематический перенос или авторство
спорных романов. Черновики статей, планы, отчёты, корпуса, кэши и модели остаются в `local/`
или других игнорируемых каталогах.

## Разработка и результаты

```bash
.venv/bin/python -m pytest tests -q -p no:cacheprovider
cd site
npm ci --no-audit --no-fund
npm run gen
npm run build
```

`src/stylo/` — библиотека и CLI; `configs/` — настройки; `tests/` — проверки;
`scripts/` — исследовательские запуски и генератор данных сайта; `site/` — статья.
Машинные результаты в `docs/` и `research/evidence/` нужны для воспроизведения чисел сайта:
[агрегат сравнения](research/evidence/topic_validity_lobo_v1/aggregate.json),
[парная сводка](research/evidence/topic_validity_lobo_v1/paired_summary.json),
[научные ограничения](research/governance/status_ledger.json).

Корпуса, модели, кэши, черновики, планы и рабочие отчёты хранятся локально и не входят в Git.
Код — MIT. Исходные тексты имеют собственные условия использования и вместе с кодом не распространяются.
