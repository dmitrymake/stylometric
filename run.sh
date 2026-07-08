#!/usr/bin/env bash
# Единый параметризованный entrypoint (заменяет run.sh/run_full.sh/run_validation.sh).
#
#   ./run.sh all            — полный пайплайн end-to-end
#   ./run.sh validate       — валидация корпуса
#   ./run.sh clean          — очистка input -> input_clean (NER-маскировка)
#   ./run.sh split          — нарезка на чанки
#   ./run.sh warm           — прогрев DocBin-кеша spaCy
#   ./run.sh train          — обучение продакшен-модели
#   ./run.sh sweep [--lobo] — ablation-sweep «что работает»
#   ./run.sh predict        — атрибуция unknown
#   ./run.sh report         — собрать HTML-отчёт
#
# Доп. аргументы прокидываются в CLI (напр. ./run.sh sweep --lobo).
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-.venv/bin/python}"
run() { PYTHONPATH=src "$PY" -m stylo.cli "$@"; }

cmd="${1:-all}"; shift || true

case "$cmd" in
  all)
    run validate-corpus || true
    run split
    run warm
    run train
    run sweep "$@"        # скрининг блоков быстрым GKF-прокси
    run evaluate          # финальные цифры полным leakage-free LOBO + baseline + значимость
    run predict || true
    run report
    ;;
  validate)        run validate-corpus "$@" ;;
  clean)           run clean "$@" ;;
  split)           run split "$@" ;;
  warm)            run warm "$@" ;;
  train)           run train "$@" ;;
  sweep)           run sweep "$@" ;;
  lobo)            run lobo "$@" ;;
  predict)         run predict "$@" ;;
  fetch-classics)  run fetch-classics "$@" ;;
  report)          run report "$@" ;;
  *)               run "$cmd" "$@" ;;
esac
