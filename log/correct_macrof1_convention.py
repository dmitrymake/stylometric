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
    # SUPERSEDED / DISABLED. This script multiplied macro-F1 by a class-count "convention" FACTOR and
    # overwrote the FROZEN docs/final_comparison.* + wrote docs/stylo_lobo_authorci.json with the old
    # (non-withdrawn) macro-F1 CI array. Both are now handled correctly elsewhere and must not be
    # regenerated here: the CI-sign correction is the versioned CI-sign erratum
    # (scripts/apply_ci_sign_erratum.py → docs/final_comparison.v2.*), and the macro-F1
    # author-clustered CI is WITHDRAWN (docs/stylo_lobo_authorci.json → macro_f1_authorclustered_CI=null,
    # docs/macro_f1_ci_withdrawal.json). Fail closed before touching any frozen artifact.
    from stylo.eval.ci_erratum import assert_publish_target_not_frozen
    assert_publish_target_not_frozen(ROOT / "docs" / "final_comparison.csv")   # raises (frozen docs/ path)
    raise SystemExit("superseded: see scripts/apply_ci_sign_erratum.py and docs/macro_f1_ci_withdrawal.json")


if __name__ == "__main__":
    main()
