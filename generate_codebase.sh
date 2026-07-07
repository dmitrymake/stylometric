#!/usr/bin/env bash
set -euo pipefail

output_file="codebase.txt"

# Собираем все .py и .sh файлы,
# исключаем виртуальные окружения и сам этот скрипт
find . \
  -type f \( -name '*.py' -o -name '*.sh' \) \
  ! -path "./.venv/*" \
  ! -name "generate_codebase.sh" \
  -print0 |
  while IFS= read -r -d '' file; do
    printf '=== %s ===\n' "$file"
    cat -- "$file"
    printf '\n'
  done >"$output_file"

echo "Готово: кодовая база собрана в $output_file"
