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

Подготовка и создание VM выполняются. Фактические cloud checks, прогресс и cleanup нужно
сверять с ignored execution records; этот документ не утверждает завершение измерения.
