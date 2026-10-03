# Публикация текущей версии Stylo

Владелец 2026-10-03 явно разрешил: «пока залей что есть в гит.
Оно вроде автоматически должно доехать на сайт».

Область: текущие коммиты authorship в main репозитория dmitrymake/stylometric
и существующая автоматическая публикация GitHub Pages. Исследовательское содержимое
фиксируется на 0fa934872cd6d1da786e2626b2724bb345843232; дополнительный commit содержит
только эту запись разрешения и указатель handoff. Новая атрибуция и подача в журнал
не являются частью задачи. Локальные корпуса и ignored research/local не публикуются.

Перед отправкой сверена origin/main; она является предком локальной main.
Пройдены release hygiene (история HEAD и index), executable inventory (251 paths),
генерация/provenance (97 sources, 8 outputs) и пустой generated diff.
Сборка сайта: no-undef 23 files, SSR 5 chapters, PASS.

Workflow deploy-pages.yml публикует текущую main после успешного push-run
«CI (release integrity)». Настройки Pages проверены через GitHub API:
build_type=workflow, публичный адрес https://stylometry.russkiykod.com/.

Результат push, ссылки CI/deploy и проверка публичной сборки сохраняются по факту
в ignored `research/local/publish-20261003/receipt.json`. Наличие разрешения
и успешных локальных проверок само по себе не является успешной публикацией.
