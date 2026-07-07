#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:./scripts"

############################################
# Конфигурация
############################################
SRC_DIR="input"
CLEAN_DIR="input_clean"
CHUNK_SIZE=500
OVERLAP=0
MIN_WORDS=200

# RESEARCH_TARGET: фиктивный ID, чтобы все книги из input/unknown можно было
# исключить из TRAIN и отправить в UNKNOWN для честного теста.
RESEARCH_TARGET="temp_target"

# Язык анализа (ru, en, fr)
LANG="ru"
export STYLO_LANG="$LANG"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║      СТИЛОМЕТРИЯ ПРОЗЫ (ENHANCED PIPELINE v3)            ║"
echo "║      Язык: $LANG                                         ║"
echo "╚══════════════════════════════════════════════════════════╝"

# ========================================
# 0. Очистка и Базовая подготовка
# ========================================
echo -e "\n[0/6] Очистка и базовая подготовка..."
rm -rf docs
rm -rf data/frags_*
mkdir -p docs data "$CLEAN_DIR"

# Очистка текста (раскомментируйте при первом запуске или после обновления входных данных)
# echo "Запуск очистки текстов..."
# python3 scripts/clean_text.py "$SRC_DIR" "$CLEAN_DIR"

# ========================================
# 1. Нарезка и Обучение (Train)
# ========================================
echo -e "\n[1/6] Нарезка корпуса и обучение моделей..."

# 1.1 Нарезка (Smart Chunking по предложениям)
python3 scripts/split.py \
  --input "$CLEAN_DIR" \
  --chunk "$CHUNK_SIZE" \
  --min-words "$MIN_WORDS" \
  --overlap "$OVERLAP" \
  --lang "$LANG"

# 1.2 Обучение (ЕДИНЫЙ пайплайн: StyloVectorizer + LR, + Delta assets)
# Артефакты:
#   data/model.pkl
#   data/vectorizer.pkl + data/vectorizer_fitted.pkl
#   data/train_vectors.pkl (sparse)
#   data/scaler_delta.pkl
#   data/centroids.npy
#   data/authors.npy
python3 scripts/train.py --lang "$LANG"

# ========================================
# 2. Валидация (LOBO)
# ========================================
echo -e "\n[2/6] Валидация модели (LOBO Cross-Validation, leakage-free)..."
python3 scripts/lobo_cv.py --lang "$LANG"

# ========================================
# 3. Ablation Study (Влияние маскировки)
# ========================================
echo -e "\n[3/6] Ablation Study (Bleaching Check)..."
python3 scripts/ablation.py --lang "$LANG"

# ========================================
# 4. Атрибуция спорного текста (Predict)
# ========================================
echo -e "\n[4/6] Атрибуция спорного текста..."

# 4.1 Перенарезка: Исключаем RESEARCH_TARGET из TRAIN, отправляя его в UNKNOWN.
rm -rf data/frags_train data/frags_unknown
python3 scripts/split.py \
  --input "$CLEAN_DIR" \
  --chunk "$CHUNK_SIZE" \
  --min-words "$MIN_WORDS" \
  --overlap "$OVERLAP" \
  --leave-out "$RESEARCH_TARGET" \
  --lang "$LANG"

# 4.2 Финальное предсказание (использует data/model.pkl + data/scaler_delta.pkl + centroids.npy)
python3 scripts/predict.py --lang "$LANG"

# ========================================
# 5. Аналитика и Визуализация (EDA)
# ========================================
echo -e "\n[5/6] Генерация аналитики..."

python3 scripts/statistic/consistency.py
python3 scripts/statistic/syntax.py
python3 scripts/statistic/entropy.py
python3 scripts/statistic/pos.py
python3 scripts/statistic/morpho_profile.py
python3 scripts/statistic/markers.py
python3 scripts/statistic/anomaly_stats.py
python3 scripts/umap_vis.py
python3 scripts/clustering.py

# ========================================
# 6. Сборка Отчёта
# ========================================
echo -e "\n[6/6] Сборка HTML отчёта..."
python3 scripts/report.py

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                      ГОТОВО!                             ║"
echo "║ Откройте: docs/index.html                                ║"
echo "╚══════════════════════════════════════════════════════════╝"
