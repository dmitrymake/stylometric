# Исправление UI-кита и потребителей

Статус: завершено локально, 2026-10-03. Владелец разрешил исправить findings
комплексного ревью rk и статьи, с целевой планкой 9/10. Дополнительное замечание:
неоформленная загрузка/ошибка главы. Разрешены Sol 6.1 исполнители и холодное
ревью Astra. Логотип сохранён; публикация и push не выполнялись.

Исходный authorship: `36887a0b089cc27760bb908e9e399061a9d078c1`, чистое дерево.
rk и rk-example содержали пользовательский WIP. Работа велась в копии актуального
дерева, а не только HEAD: `/tmp/stylo-rk-fixes-l29b81q_/{rk,rk-example}`.

## Результат

- Overlay получает фокус после фактического mount Portal, удерживает его в верхнем
  слое и возвращает при закрытии. Tab учитывает disabled/inert, пустой диалог,
  вложенные слои и нативный summary. Наблюдатель mount отключается после фокусировки.
- Combobox выбирает только видимые пункты, согласует loading с ARIA, закрывается
  на Tab/blur и сохраняет контролируемое значение при отказе потребителя от изменения.
- Button объединяет внешние стили по свойствам; пары фон/текст обеспечивают контраст.
  Брендовые gold/cinnabar и восемь исходных файлов логотипа не изменены.
- Public defaults разрешаются общей функцией: определённый явный prop, затем preset,
  затем прежний fallback. Вложенные providers объединяют presets по компоненту и prop.
  Список компонентов, использующих этот механизм, явно документирован в rk.
- Editorial поддерживает class/data themes, светлую область внутри тёмной, aliases
  Panel, slots/root props и ref. Пропы потребителя имеют приоритет над aliases preset.
- Витрина использует компактную Pagination по ширине контейнера; GraphControls
  располагается внутри своего примера.
- DataTable ограничивает первый виртуальный рендер до измерения контейнера,
  выбирает Tab-позицию из видимых строк и сохраняет непустое последнее окно при
  прокрутке за расчётную высоту. Это не виртуализатор произвольной высоты карточек:
  контракт rowHeight по-прежнему требует соответствия реальной геометрии строки.
- В статье окна timeline доступны клавиатурой и касанием через native range/output.
  Перемещение курсора не пересоздаёт полосы. Загрузка и ошибка имеют заголовок,
  сетку статьи, status/alert; стандартная кнопка rk повторно загружает страницу,
  сохраняя маршрут главы.

## Проверки

| Проверка | Наблюдаемый результат |
|---|---|
| Полный rk Vitest, 2 workers | 698 passed, 93 files, 60.94 s |
| rk typecheck / lint / build / pack | PASS |
| Size-limit, minified + Brotli | Button 1.97 kB / 7; Modal 4.84 / 11; barrel 104.36 / 130 |
| Контраст Button, Chromium computed styles | 135 checks, min 4.7288521; focus-visible без пропусков |
| Interaction targeted | 55 passed; browser normal/empty/disabled/nested focus, loading/blur/controlled Combobox |
| Холодное Astra | Собственные 47 tests, browser исходных findings, summary edge; позднее 14 DataTable tests и desktop/mobile browser |
| Оригинальные rk / rk-example после переноса | typecheck/lint/build кита и typecheck/build витрины PASS |
| Stylo build | no-undef 23 source files; SSR 5 chapters; основной bundle 382.51 kB, gzip 98.22 |
| Чистая установка Stylo | npm ci --offline и build/SSR PASS в `/tmp/stylo-rk-clean-2i18klyd` |
| Сайт, production Chromium | 5 глав × 1440/390; без pageerrors/горизонтального overflow, 5 JSON downloads доступны |
| Загрузка/отказ/восстановление | 1440/390, x=160/20 в сетке; retry открывает `#sholokhov`; keyboard range меняет окно |
| Генератор сайта | `tests/test_site_generator_contract.py`: 14 passed |
| Inventory / provenance / deterministic generation | 251 Python paths; 97 sources / 8 outputs; generated diff пуст |
| Release hygiene | HEAD history/index PASS после локального commit |

Предупреждения отдельно от PASS: полный Vitest содержит axe preload-assets timeout
в VideoEmbed; size-limit — esbuild ignored bare imports при sideEffects:false.
Они не объявляются устранёнными. Native Firefox/Safari и реальные AT не проверялись.
Первый чистый npm ci внутри ограниченной песочницы остановился на esbuild EPERM;
повтор той же offline-команды с разрешённым запуском бинарника прошёл.

## Производительность и пределы вывода

Один парный Chromium запуск с одинаковыми 10000 синтетическими строками и onRowClick:
время до passive App effect 1143.8 → 64.6 ms, после scroll-to-end 0 → 1 Tab-позиция.
DOM-count в этом profile снят после effects; ранний взрыв DOM подтверждён исходной
веткой viewportH > 0, а новый первый commit отдельно проверен Astra через layout
measurement: 20–22 строки. В окончательном desktop/CSS-height/mobile сценарии уже
через два кадра есть последняя строка и одна Tab-позиция; Enter вызывает callback.

Timeline: 208 полос, 100 cursor updates, 0 изменений полос и сохранённые DOM-узлы;
median/p95 frame latency 16.7/16.8 ms. Эти локальные измерения не дают общей оценки
производительности всего кита, всех браузеров или всех данных.

## Интеграция и evidence

Применено 38 файловых изменений (37 rk, 1 rk-example) после проверки old/new SHA256.
Проверены 457 baseline-файлов: вне разрешённого delta байты сохранены. Логотипные
файлы неизменны; gold/cinnabar сверены с точной исходной CSS-копией по baseline hash.
Чужой WIP не коммитился; rk/rk-example остаются рабочими деревьями владельца.
Stylo содержит итоговый локальный tarball; lock отличается только integrity, SHA512
совпадает с tarball, прежние platform/libc metadata сохранены.

Холодный отчёт: `research/local/reviews/rk-remediation-cold-20261003.md`.
Все исходные RK-01–RK-07, EX-01–EX-02 и дополнительный CR-01 закрыты в проверенном
срезе. Промежуточная mobile ReferenceError и пустое окно исправлены и повторно
проверены; прежние PASS не выдаются за подтверждение этих промежуточных snapshots.
Astra оценивает чистоту/API и понятность в 9/10, лаконичность в 8.5/10;
равномерная оценка 9/10 всего проекта не заявляется.

Локальные воспроизводимые артефакты:

- `/tmp/stylo-rk-fixes-l29b81q_/{baseline,integration-receipt,preservation-check}.json`;
- `/tmp/rk-{regression,typecheck,lint,build,size,pack}-final-complete.log`;
- `/tmp/rk-api-fixes/contrast.{mjs,jsx,json}`;
- `/tmp/rk-remediation-cold-review/` — самостоятельные browser harness и результаты;
- `/tmp/rk-runtime-profile{,-before-controlled}.json`, `.mjs` и JSX fixtures;
- `/tmp/stylo-load-state-check.mjs`, `/tmp/stylo-load-states/results.json` и screenshots;
- `/tmp/stylo-final-ui-smoke.mjs`, `/tmp/stylo-final-ui/results.json`;
- `/tmp/stylo-{site-final-build,rk-clean-install,rk-clean-build}.log`.

Числа исследований, математический код и корпуса не менялись. Эта задача не даёт
новой атрибуции и не заменяет внешнюю научную валидацию. Локальный сайт:
http://127.0.0.1:4174/; исходная витрина: http://127.0.0.1:5180/.
