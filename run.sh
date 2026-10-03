#!/usr/bin/env bash
# Convenience aliases for the Python CLI; orchestration lives in stylo analyze.
# ./run.sh all --config case.yaml --target-work unknown/target
set -euo pipefail
cd "$(dirname "$0")"

stylo_python="${PYTHON:-.venv/bin/python}"
if (($# == 0)); then
  set -- --help
fi
case "$1" in
  all) shift; set -- analyze "$@" ;;
  validate) shift; set -- validate-corpus "$@" ;;
esac
PYTHONPATH=src exec "$stylo_python" -m stylo.cli "$@"
