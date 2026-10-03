# Временный GCP runner для тестов и topic-validity

## Разрешение и границы

- Владелец после локального сбоя попросил временно поднять сервер GCP, выполнить все
  тесты и расчёт, затем выключить. Отдельно выбрал текущий проект `anime-date-sim`.
- Кодовая база: исправления `9efd6b5`; перед cloud freeze добавлено исправление остановки
  fork-workers и связи checkpoint с runtime/thread identities.
- Только временная VM с отдельным boot disk. Создание отдельных service account и bucket
  было отклонено автоматической проверкой разрешений; они не создаются. Перенос через SSH/SCP.
- Тексты и производные кэши передаются только на эту закрытую VM, не в Git/логи/публичное
  хранилище. Локальные исходники и старые checkpoints сохраняются.
- Публикация и push не разрешены.

## Причина смены исполнения

Локальный журнал содержит повторные SIGTERM и не содержит нового checkpoint. Отдельный
синтетический subprocess подтвердил: fork-workers наследовали обработчик SIGTERM родителя,
поэтому `Pool.terminate()` мог не завершить workers, а `pool.join()` блокировал final save.
Initializer теперь ставит SIGTERM default / SIGINT ignore в workers; родитель сохраняет
завершённые fits. Runtime и thread identities теперь входят в hash checkpoint. Старые
checkpoints не мигрируются.

Три bounded fork subprocess проверки stop/deadline/failure и две проверки identity drift
прошли на доступном локальном Python 3.14. Canonical Python 3.11 tests выполняются на VM.
Deployment synthetic tests также отделены от реального Git workspace, чтобы полный
source-archive CI работал без `.git`; production workspace guard не ослаблялся.

## Конфигурация и завершение

- VM: `stylo-audit-20261003-ea10fc`, `europe-west1-b`, `n2-standard-16` (16 vCPU, 64 GiB).
- Debian 12, boot disk 100 GB pd-balanced с auto-delete при удалении VM.
- Service account/scopes отсутствуют. SSH через IAP; firewall внутри VM разрешает SSH
  только из диапазона IAP. Project-wide SSH keys заблокированы для VM.
- systemd отделяет выполнение от жизни SSH/чат-сессии. ExecStopPost выключает VM после
  задания, включая ошибку; platform deadline `2026-10-04T21:05:10Z` с действием STOP.
- STOP сохраняет results на диске до выгрузки. После копирования и проверки SHA-256
  необходимо удалить только эту VM и её boot disk, проверить отсутствие обоих ресурсов.
- Игнорируемые launcher/config records: `research/local/cloud_20261003/`.

## Проверки и scientific run

Main environment: CPython 3.11.14, `requirements.lock`, spaCy lg/md 3.8.0 и pymorphy.
Полный pytest с доступным corrected context, три mechanical gates, wheel/sdist, clean wheel
smoke, source-archive tests и site build. Capture replay — отдельное окружение по frozen
fixture, не замена main lock и не изменение fixture. Все failures/skips записываются отдельно.

Полный topic run: 248 folds × A0/A4 × current/topic_strict, восемь workers, один BLAS thread,
новый cloud checkpoint. Успех — validated aggregate и 992 records; exit 0 при частичном
checkpoint не означает завершения. Измерение остаётся exploratory; новая атрибуция романов
этим прогоном не выполняется.

## Результат

VM создана, 253 MiB выбранных данных и полный Git bundle перенесены по IAP/SSH с проверкой
SHA-256. Первый cloud pytest: 1339 passed, 1 failure из-за launcher umask 077, 2 skips.
При mask 022 и переданных reference CSV/SHA: **1341 passed, 1 skipped** (отдельный live-golden).
Полный Git-free archive: **1333 passed, 9 expected environment/data skips**. Package/wheel
smoke, site build и три mechanical gates прошли.

Capture environment создан отдельно: кроме NumPy/SciPy/spaCy потребовались совместимые
thinc/confection/weasel/srsly. Replay дал **14 passed, 3 failed**: thread-pool fingerprint
и два Stylo goldens с различиями на уровне последних примерно 11–12 десятичных знаков.
Это не полный capture PASS; fixture и tolerances не менялись, failures сохранены.

На n2-standard-16 измерено около 50% CPU при 8 workers и примерно 11 GiB RAM. Первый
cloud checkpoint содержит 32 A0/current fits, сохранён также локально перед resize.
Владелец затем прямо выбрал **«Сразу используй 128 до завершения»**. Та же VM увеличена
до n2-standard-128 (128 vCPU, 512 GiB); план следующего запуска — 64 workers.

Штатная остановка выявила ещё один дефект: `_summarise` обращался к PredictionDecision
как к dict и падал после сохранения checkpoint. Исправлено на сравнение `decision.top1`
с true label, добавлены partial/complete-992 regression tests с настоящим scorer.
Для изменённого runner используется новая source identity и отдельный checkpoint;
сохранённые 32 fits не перепривязываются к новому коду.

Фактические cloud checks, прогресс и cleanup нужно сверять с ignored execution records;
этот документ не утверждает завершение измерения. Подготовлен одноразовый локальный
collector через user systemd: он ждёт явного enable marker, затем копирует проверенные
результаты, проверяет commit/hash/ресурсную область и удаляет только созданную VM и boot disk.

## Завершение 2026-10-03

Источник bb3760f; n2-standard-128, 64 workers. Выполнены все **992 fits** за
12447 секунд (3 ч 27 мин); topic_execute exit 0 в 01:41:49 UTC. Агрегат проверен и
сохранён в `research/evidence/topic_validity_lobo_v1/aggregate.json`, self-hash
`06b01f7a0fdf4e8e0440bd25ac5b499b943d77336813f78be2497f75d584290e`.

A0 current: 216/248; A0 topic_strict: 218/248. A4: 225/248 в обоих вариантах.
Суммы переходов по авторам согласованы с точностью; canonical aggregate побайтно
совпадает с выгруженным. Интервалы этот runner не рассчитывает. Это сравнение
методов на исправленном корпусе, не новая атрибуция спорных романов.

Фактический последний основной pytest: **1343 passed, 1 skipped**; Git-free archive:
**1335 passed, 9 skipped**. Исторический capture replay: **14 passed, 3 failed**;
эти сбои не скрыты и не исправлены изменением fixture/tolerance. Остальные стадии
сборки и проверок, включая итоговую неизменность исходников, завершились exit 0.

Collector сохранил результаты в `research/local/cloud_20261003/retrieval-lrbk7f_m/results/`,
проверил inventory/commit/completion и удалил временную VM вместе с boot disk.
В 01:45:46 UTC записан phase=complete. Независимые `gcloud compute instances list`
и `gcloud compute disks list` в anime-date-sim с точным именем
`stylo-audit-20261003-ea10fc` вернули пустые списки. VM и диск больше не существуют.
Никаких новых VM, bucket или service account для завершения не создавалось.
