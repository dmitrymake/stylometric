#!/usr/bin/env bash
# Пересчёт корпуса и канонических чисел на нормализации с унифицированными кавычками
# (clean.py: все двойные кавычки → "). Порядок: очистка → нарезка → прогрев кэшей →
# полный LOBO (final, 10 спеков) → производные (author-CI, delta-варианты, провенанс-проба,
# нециркулярный кейс ТД). Кэши Rep/DocBin ключуются хэшем текста — пересчитываются сами.
set -euo pipefail
cd /home/dmake/code/private/authorship
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=.venv/bin/python

echo "=== [1/8] clean: нормализация с унификацией кавычек ==="
./run.sh clean

echo "=== [2/8] split: нарезка на чанки ==="
./run.sh split

echo "=== [3/8] warm: DocBin/Rep кэши ==="
./run.sh warm

echo "=== [4/8] final: полный leakage-free LOBO (все спеки) ==="
PYTHONPATH=src $PY -m stylo.cli evaluate

echo "=== [5/8] author-clustered CI headline ==="
$PY log/lobo_author_ci.py

echo "=== [6/8] cosine/книжный Delta ==="
$PY log/experiments/delta_cosine_lobo.py

echo "=== [7/8] провенанс-проба (эффект унификации кавычек) ==="
nice -n 10 $PY log/experiments/provenance_probe.py

echo "=== [8/8] нециркулярный leave-block-out кейс ТД ==="
$PY log/sholokhov_lobo.py

echo "=== ГОТОВО: сравнить с git-версиями docs/final_comparison.csv, stylo_lobo_authorci.json, delta_cosine_lobo.json, provenance_probe.json, sholokhov_lobo.json ==="
