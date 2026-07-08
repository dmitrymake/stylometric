"""
Генерация HTML отчета (Report Generator).

Файл читает:
- docs/dual_prediction_report.txt (создает predict.py)
- docs/experiments_summary.txt (создает lobo_cv.py)
- docs/ablation_report.txt (создает ablation.py)
- docs/consistency_stats.txt + графики и CSV
"""

from __future__ import annotations

import datetime
import logging
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DOCS = Path("docs")
DOCS.mkdir(exist_ok=True)


# HELPERS


def safe_read(filename: str) -> str:
    path = DOCS / filename
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def markdown_to_html_table(text: str) -> str:
    """
    Преобразует Markdown-таблицу (с разделителями |) в HTML-таблицу.
    Поддерживает формат tabulate (с пайпами по краям).
    """
    if not text.strip():
        return ""

    lines = [ln.rstrip() for ln in text.split("\n") if ln.strip()]

    # Ищем разделительную строку: |---|---| или ---|---|---
    sep_idx = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if set(stripped) <= {"|", "-", " ", ":"} and "-" in stripped:
            sep_idx = i
            break

    if sep_idx <= 0 or sep_idx >= len(lines) - 1:
        return f"<pre>{text}</pre>"

    header_line = lines[sep_idx - 1]
    headers = [h.strip() for h in header_line.strip().strip("|").split("|")]

    html = ['<div class="table-wrapper"><table>']
    html.append("<thead><tr>")
    for h in headers:
        html.append(f"<th>{h}</th>")
    html.append("</tr></thead>")

    html.append("<tbody>")
    for line in lines[sep_idx + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        # Пропускаем "нижние границы"
        if set(stripped) <= {"|", "-", " ", ":"} and "-" in stripped:
            continue

        cells = [c.strip() for c in stripped.strip("|").split("|")]
        html.append("<tr>")
        for c in cells:
            html.append(f"<td>{c}</td>")
        html.append("</tr>")
    html.append("</tbody></table></div>")

    return "".join(html)


def csv_to_html_table(filename: str, title: str) -> str:
    """Читает CSV и возвращает HTML таблицу."""
    path = DOCS / filename
    if not path.exists():
        return ""

    try:
        df = pd.read_csv(path)
        # Ограничим вывод для читабельности
        if "syntactic_complexity_mean" in df.columns:
            df = df.sort_values("syntactic_complexity_mean", ascending=False).head(15)

        html = f"<h3>{title}</h3>"
        html += '<div class="table-wrapper">'
        html += df.to_html(index=False, border=0, classes="dataframe")
        html += "</div>"
        return html
    except Exception as e:
        logging.warning(f"Ошибка чтения CSV {filename}: {e}")
        return ""


def get_img_block(filename: str, caption: str) -> str:
    path = DOCS / filename
    if not path.exists():
        return ""
    return f"""
    <div class="figure">
        <a href="{filename}" target="_blank" title="Открыть полный размер">
            <img src="{filename}" alt="{caption}">
        </a>
        <div class="caption">Рис: {caption}</div>
    </div>
    """


def parse_prediction(text: str) -> dict:
    """
    Извлекает:
    - winner
    - conf (в процентах, как число)
    - p_value (строкой)
    Поддерживает несколько форматов.
    """
    res = {"winner": "Не определен", "conf": 0.0, "p_value": "N/A"}
    if not text:
        return res

    # Формат: "=== ИТОГОВОЕ ЗАКЛЮЧЕНИЕ: NAME ==="
    m_win = re.search(r"ИТОГОВОЕ\s+ЗАКЛЮЧЕНИЕ:\s*(.+)", text, flags=re.IGNORECASE)
    if m_win:
        winner = m_win.group(1).strip()
        # убираем возможные хвостовые === или лишние символы
        winner = re.sub(r"\s*=+\s*$", "", winner).strip()
        res["winner"] = winner

    # "Уверенность: 83.42%"
    m_conf = re.search(r"Уверенность:\s*([\d\.]+)\s*%", text, flags=re.IGNORECASE)
    if m_conf:
        try:
            res["conf"] = float(m_conf.group(1))
        except Exception:
            pass

    # "p-value: 0.012345"
    m_p = re.search(r"p-value:\s*([0-9\.eE\-\<]+)", text, flags=re.IGNORECASE)
    if m_p:
        res["p_value"] = m_p.group(1).strip()

    return res


def format_prediction_report(text: str) -> str:
    if not text:
        return "<p>Нет данных</p>"
    safe = text.replace("<", "&lt;").replace(">", "&gt;")
    safe = re.sub(r"(=== .+ ===)", r"<strong>\1</strong>", safe)
    safe = re.sub(
        r"(===\s*ИТОГОВОЕ\s+ЗАКЛЮЧЕНИЕ:.*?===)",
        r"<span class='result-highlight'>\1</span>",
        safe,
        flags=re.IGNORECASE,
    )
    return f"<pre>{safe}</pre>"


def generate_conclusion(data: dict) -> str:
    return f"""
    <div class="conclusion-box">
        <h3>7. Заключение и выводы</h3>
        <p>На основании комплексного анализа (Hybrid Stylometry) получен следующий результат:</p>
        <ul>
            <li><strong>Наиболее вероятный автор:</strong> {data["winner"]}</li>
            <li><strong>Уровень уверенности ансамбля:</strong> {data["conf"]:.2f}%</li>
            <li><strong>Статистическая значимость (p-value):</strong> {data["p_value"]}</li>
        </ul>
    </div>
    """


# MAIN


def main() -> None:
    logging.info("Генерация HTML отчета...")

    pred_text = safe_read("dual_prediction_report.txt")
    ex_summary = safe_read("experiments_summary.txt")
    ablation_txt = safe_read("ablation_report.txt")

    consistency_txt = safe_read("consistency_stats.txt")

    pred_data = parse_prediction(pred_text)

    css = """
    body { font-family: sans-serif; line-height: 1.6; color: #333; max-width: 960px; margin: 0 auto; padding: 20px; }
    h1, h2, h3 { color: #2c3e50; }
    h2 { border-left: 5px solid #3498db; padding-left: 10px; background: #f4f6f7; }
    pre { background: #f8f9fa; padding: 10px; border: 1px solid #ddd; overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    th, td { padding: 8px; border: 1px solid #ddd; text-align: left; }
    th { background-color: #f2f2f2; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 20px; align-items: start; }
    .figure { text-align: center; border: 1px solid #eee; padding: 10px; background: white; }
    .caption { font-size: 13px; color: #555; margin-top: 8px; }
    .conclusion-box { background: #e8f6f3; border: 1px solid #1abc9c; padding: 20px; }
    .result-highlight { background-color: #fff3cd; font-weight: bold; }
    .table-wrapper { overflow-x: auto; }
    """

    html: list[str] = [
        f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Отчет: {pred_data["winner"]}</title>
<style>{css}</style>
</head>
<body>
<h1>Стилометрический анализ текста</h1>
<div class="meta">Дата: {datetime.datetime.now():%d.%m.%Y} | Hybrid Stylometry</div>
"""
    ]

    html.append("<h2>1. Результаты Атрибуции</h2>")
    html.append(format_prediction_report(pred_text))

    html.append("<h2>2. Валидация (LOBO)</h2>")
    if ex_summary:
        html.append(f"<pre>{ex_summary}</pre>")
    html.append('<div class="grid">')
    html.append(get_img_block("confusion_heatmap.png", "Confusion Matrix"))
    html.append(get_img_block("rank_distribution.png", "Rank Distribution"))
    html.append("</div>")

    html.append("<h2>3. Ablation Study (Impact of Masking)</h2>")
    if ablation_txt:
        html.append(f"<pre>{ablation_txt}</pre>")
    else:
        html.append("<p>Данные абляции отсутствуют.</p>")

    html.append("<h2>4. Консистентность (Stability)</h2>")
    html.append(get_img_block("consistency_boxplot.png", "Consistency Boxplot"))

    if consistency_txt:
        parts = consistency_txt.split("===")
        for part in parts:
            if not part.strip():
                continue
            lines = part.strip().split("\n")
            title = lines[0].strip()
            table_content = "\n".join(lines[1:])
            html.append(f"<h3>{title}</h3>")
            html.append(markdown_to_html_table(table_content))

    html.append("<h2>5. Структурные аномалии</h2>")
    html.append(
        csv_to_html_table(
            "anomaly_stats_by_book.csv",
            "Топ-15 сложных текстов (Syntactic Complexity)",
        )
    )

    html.append("<h2>6. Лингвистическая статистика</h2>")
    html.append('<div class="grid">')
    html.append(get_img_block("entropy.png", "Entropy"))
    html.append(get_img_block("syntax.png", "Syntax Lengths"))
    html.append(get_img_block("pos.png", "POS Ratios"))
    html.append(get_img_block("umap.png", "UMAP Projection"))
    html.append(get_img_block("clusters_kmeans.png", "Clusters"))
    html.append("</div>")

    html.append(generate_conclusion(pred_data))

    html.append("</body></html>")

    outfile = DOCS / "index.html"
    outfile.write_text("\n".join(html), encoding="utf-8")
    logging.info(f"Отчет сохранен: {outfile.absolute()}")


if __name__ == "__main__":
    main()
