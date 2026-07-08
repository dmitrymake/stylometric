"""Сравнительная таблица LOBO с macro-F1 по тестированным классам + синхронный TXT.

macro-F1 считается по меткам в y_true ∪ y_pred (тестированные классы). stylo — из persisted
per-book LOBO (docs/lobo_books.txt); прочие спеки — пересчёт по соотношению числа классов
n_dataset/n_tested (точный: нетестируемые single-book авторы дают F1=0 и не входят в числитель,
меняется только знаменатель). accuracy/top2/McNemar/ECE от набора меток макро не зависят.

CSV пишется с lineterminator='\n' без trailing-whitespace; TXT генерируется ИЗ тех же строк,
чтобы docs/final_comparison.csv и .txt не расходились. Идемпотентно по значениям.
"""
from __future__ import annotations
import csv, io, json, pathlib, re, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from stylo.lang import display_name
from stylo.eval.metrics import macro_f1
from sklearn.metrics import f1_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
N_AUTHORS_DATASET = 47      # 43 тестированных + 4 single-book (нетестируемы в LOBO)
N_TESTED = 43
FACTOR = N_AUTHORS_DATASET / N_TESTED
SEED, N_BOOT = 42, 10000

HDR = re.compile(r"^\[(OK|MISS)\s*\]\s+(.+?)\s*/\s*\S+\s+\(rank истинного автора:\s*(\d+)\)")


def stylo_authoritative():
    val = json.loads((ROOT / "docs/validation.json").read_text(encoding="utf-8"))
    authors = val["authors"]
    id_of = {display_name(a): i for i, a in enumerate(authors)}
    true_s, pred_s, au_s = [], [], []
    cur = None
    for line in (ROOT / "docs/lobo_books.txt").read_text(encoding="utf-8").splitlines():
        h = HDR.match(line)
        if h:
            cur = h.group(2).strip(); continue
        if cur and line.lstrip().startswith("топ:"):
            pred = line.split("топ:")[1].split("(")[0].strip().rstrip(",").strip()
            if cur in id_of and pred in id_of:
                true_s.append(id_of[cur]); pred_s.append(id_of[pred]); au_s.append(cur)
            cur = None
    y = np.array(true_s); p = np.array(pred_s)
    labels = np.unique(np.concatenate([y, p])).tolist()
    point = macro_f1(y, p, labels)
    au = np.array(au_s); uniq = np.array(sorted(set(au_s)))
    au_to_idx = {a: np.where(au == a)[0] for a in uniq}
    rng = np.random.default_rng(SEED)
    boot = np.empty(N_BOOT)
    for i in range(N_BOOT):
        sa = rng.choice(len(uniq), size=len(uniq), replace=True)
        sel = np.concatenate([au_to_idx[uniq[j]] for j in sa])
        boot[i] = f1_score(y[sel], p[sel], average="macro", zero_division=0)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi), float(np.median(boot)), len(y)


def write_txt(rows, path):
    """Человекочитаемая таблица из тех же строк, что CSV — чтобы артефакты не расходились."""
    def f(x):
        return "" if x is None or x == "" else (f"{float(x):.4f}" if isinstance(x, str) and _isnum(x) else x)
    lines = ["=== ФИНАЛЬНОЕ СРАВНЕНИЕ (полный leakage-free LOBO, book-level; macro-F1 по тестированным классам) ===",
             f"{'model':<12} {'acc':>9} {'acc 95% CI':>16} {'macroF1':>8} {'top2':>6} {'Δvs_stylo':>10} {'McNemar_p':>12} {'ECE':>6}",
             "-" * 95]
    stylo_acc = float(next(r for r in rows if r["model"] == "stylo")["accuracy"])
    for r in rows:
        acc = f"{float(r['accuracy']):.4f}"
        ci = r["acc_ci"]
        mf1 = f"{float(r['macro_f1']):.4f}"
        top2 = f"{float(r['top2']):.4f}" if r.get("top2") else ""
        dacc = "" if r["model"] == "stylo" else f"{float(r['accuracy']) - stylo_acc:+.4f}"
        p = r.get("vs_stylo_mcnemar_p")
        pcol = "" if p is None or p == "" else (f"{float(p):.4g}")
        ece = "" if not r.get("ece") else f"{float(r['ece']):.4f}"
        lines.append(f"{r['model']:<12} {acc:>9} {str(ci):>16} {mf1:>8} {top2:>6} {dacc:>10} {pcol:>12} {ece:>6}")
    path.write_text("\n".join(l.rstrip() for l in lines) + "\n", encoding="utf-8")


def _isnum(s):
    try:
        float(s); return True
    except (ValueError, TypeError):
        return False


def main():
    stylo_f1, slo, shi, smed, n = stylo_authoritative()
    print(f"stylo (per-book LOBO): macro-F1={stylo_f1:.4f} author-CI=[{slo:.4f},{shi:.4f}] медиана={smed:.4f} n={n}", flush=True)

    rows_in = list(csv.DictReader((ROOT / "docs/final_comparison.csv").read_text(encoding="utf-8").splitlines()))
    fields = list(rows_in[0].keys())
    stylo_raw = float(next(r for r in rows_in if r["model"] == "stylo")["macro_f1"])
    already = abs(stylo_raw - stylo_f1) < 0.005
    print(f"final_comparison.csv stylo macro-F1={stylo_raw:.4f} -> {'значения уже корректны' if already else 'пересчёт'}", flush=True)

    out_rows = []
    for r in rows_in:
        raw = float(r["macro_f1"])
        corr = raw if already else (stylo_f1 if r["model"] == "stylo" else raw * FACTOR)
        r2 = dict(r); r2["macro_f1"] = round(corr, 4)
        out_rows.append(r2)

    # CSV: lineterminator='\n', без trailing-whitespace (всегда чистый формат)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n"); w.writeheader(); w.writerows(out_rows)
    csv_text = buf.getvalue().rstrip("\n") + "\n"
    (ROOT / "docs/final_comparison.csv").write_text(csv_text, encoding="utf-8")
    # TXT: из тех же строк
    write_txt(out_rows, ROOT / "docs/final_comparison.txt")
    print("✓ docs/final_comparison.csv (\\n, без trailing-whitespace) + docs/final_comparison.txt (синхронно)", flush=True)

    (ROOT / "docs/stylo_lobo_authorci.json").write_text(json.dumps({
        "method": "author-clustered bootstrap macro-F1 (ресэмпл АВТОРОВ) из persisted per-book stylo-LR LOBO (docs/lobo_books.txt); по тестированным классам",
        "source": "docs/lobo_books.txt (LOBO, 260 книг, 43 тестированных автора)",
        "n_books": int(n), "n_authors_tested": N_TESTED, "n_authors_dataset": N_AUTHORS_DATASET,
        "macro_f1_point": round(float(stylo_f1), 4),
        "macro_f1_authorclustered_CI": [round(slo, 4), round(shi, 4)],
        "macro_f1_bootstrap_median": round(smed, 4),
        "n_boot": N_BOOT,
        "note": "канонический headline stylo-LR LOBO: macro-F1 + author-clustered 95% CI",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("✓ docs/stylo_lobo_authorci.json", flush=True)


if __name__ == "__main__":
    main()
