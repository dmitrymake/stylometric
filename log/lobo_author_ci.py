"""Шаг 2: author-clustered macro-F1 CI для канонического stylo-LR LOBO headline.

final.py считает только book-level accuracy-CI для stylo; macro-F1 author-clustered CI
в final.py отсутствует. Здесь — пересчёт из PERSISTED per-book LOBO-данных (docs/lobo_books.txt),
без ре-рана LOBO.

Парсит per-book (true_author, pred=top1, correct) -> author-clustered bootstrap macro-F1
(ресэмпл АВТОРОВ). Sanity: reconstructed accuracy должна = 0.8923 (232/260).
"""
from __future__ import annotations
import json, pathlib, re, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from sklearn.metrics import f1_score
from stylo.jsonio import dumps_strict

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "lobo_books.txt"
N_BOOT = 10000
SEED = 42

HDR = re.compile(r"^\[(OK|MISS)\s*\]\s+(.+?)\s*/\s*\S+\s+\(rank истинного автора:\s*(\d+)\)")
TOP = re.compile(r"топ:\s*(.+)")


def parse():
    recs, cur_author, cur_rank, cur_status = [], None, None, None
    for line in SRC.read_text(encoding="utf-8").splitlines():
        h = HDR.match(line)
        if h:
            cur_status, cur_author, cur_rank = h.group(1), h.group(2).strip(), int(h.group(3))
            continue
        if cur_author and line.lstrip().startswith("топ:"):
            m = TOP.search(line)
            # pred = имя до первой ' (' в списке кандидатов
            pred = m.group(1).split("(")[0].strip().rstrip(",").strip()
            correct = (cur_status == "OK")
            recs.append((cur_author, pred, correct))
            cur_author = None
    return recs


def canonical_stylo_acc() -> float:
    """Каноническая stylo-accuracy из docs/final_comparison.csv (один источник с README)."""
    import csv
    with open(ROOT / "docs" / "final_comparison.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["model"] == "stylo":
                return float(row["accuracy"])
    raise SystemExit("docs/final_comparison.csv: строка stylo не найдена")


def main():
    recs = parse()
    n = len(recs)
    authors = np.array([r[0] for r in recs])
    preds = np.array([r[1] for r in recs])
    acc = float(np.mean([r[2] for r in recs]))
    uniq = np.array(sorted(set(authors.tolist()) | set(preds.tolist())))
    acc_canon = canonical_stylo_acc()
    print(f"книг распознано={n} | reconstructed accuracy={acc:.4f} (канон из final_comparison.csv: {acc_canon:.4f})", flush=True)
    if abs(acc - acc_canon) > 1e-3:
        raise SystemExit(f"PARSE FAIL: accuracy {acc:.4f} != {acc_canon:.4f} — парсер разошёлся с каноном, не доверять CI")

    def macroF1(idx):
        # idx — индексы книг (после автор-ресэмпла это повторяющийся массив)
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder().fit(authors)
        y = le.transform(authors[idx])
        p = le.transform(preds[idx])
        return f1_score(y, p, average="macro", zero_division=0)

    correct = np.array([r[2] for r in recs], dtype=float)
    full = np.arange(n)
    point = macroF1(full)
    # author-clustered: ресэмпл авторов -> все их книги; в том же цикле — accuracy
    au_to_idx = {a: np.where(authors == a)[0] for a in np.unique(authors)}
    au_list = np.array(sorted(au_to_idx))
    rng = np.random.default_rng(SEED)
    boot = np.empty(N_BOOT)
    boot_acc = np.empty(N_BOOT)
    for i in range(N_BOOT):
        sa = rng.choice(len(au_list), size=len(au_list), replace=True)
        sel = np.concatenate([au_to_idx[au_list[j]] for j in sa])
        boot[i] = macroF1(sel)
        boot_acc[i] = correct[sel].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    median = float(np.median(boot))
    acc_lo, acc_hi = np.percentile(boot_acc, [2.5, 97.5])
    acc_median = float(np.median(boot_acc))
    print(f"macro-F1 точка={point:.4f} | author-clustered 95%CI=[{lo:.4f},{hi:.4f}] | медиана бутстрапа={median:.4f}", flush=True)
    print(f"accuracy точка={acc:.4f} | author-clustered 95%CI=[{acc_lo:.4f},{acc_hi:.4f}] | медиана бутстрапа={acc_median:.4f}", flush=True)

    out = {
        "claim_status": "exploratory_internal",
        "training_weighting": "chunk_weighted_training_legacy",
        "legacy_caveat": (
            "Обучение stylo взвешено по чанкам: длинная книга получает больший train-вес внутри автора, "
            "словари/idf/MFW тоже фитятся по чанкам. Work-balanced пересчёт (одна работа — один голос на "
            "train-стороне) не проведён, поэтому до него этот headline = chunk_weighted_training_legacy "
            "(парный аудит центроидов — docs/cases/work_balanced_audit/). LOBO держит тестовую книгу "
            "целиком; смещение касается train-взвешивания, не утечки."),
        "method": ("author-clustered bootstrap macro-F1 и accuracy (ресэмпл АВТОРОВ) из persisted "
                   "per-book stylo-LR LOBO (docs/lobo_books.txt); без ре-рана"),
        "source": f"docs/lobo_books.txt (final.py LOBO, {int(n)} книг)",
        "n_books": int(n),
        "n_authors_tested": int(len(au_list)),
        "n_authors_dataset": 47,
        "accuracy_point": round(acc, 4),
        "accuracy_authorclustered_CI": [round(float(acc_lo), 4), round(float(acc_hi), 4)],
        "accuracy_bootstrap_median": round(acc_median, 4),
        "macro_f1_point": round(float(point), 4),
        # author-clustered CI macro-F1 ОТОЗВАН (null): ресэмпл авторов меняет набор классов
        # macro-усреднения (выпавший, но предсказанный автор даёт F1=0) → это не CI фиксированной
        # 43-классовой функции, недействителен как мера разброса; прежнее значение — в superseded.
        "macro_f1_authorclustered_CI": None,
        "macro_f1_authorclustered_interval_status": "withdrawn_pending_preregistered_recompute",
        "macro_f1_authorclustered_superseded_interval": [round(float(lo), 4), round(float(hi), 4)],
        "macro_f1_authorclustered_erratum_ref": "docs/macro_f1_ci_withdrawal.json",
        "macro_f1_bootstrap_median": round(median, 4),
        "n_boot": N_BOOT,
        "note": ("канонический headline stylo-LR LOBO: accuracy + author-clustered 95% CI. Точка "
                 "macro-F1 описательная. Author-clustered 95% CI для macro-F1 не публикуется "
                 "(macro_f1_authorclustered_CI=null). Author-clustered bootstrap ресэмплит авторов и "
                 "меняет набор классов macro-усреднения: выпавший, но предсказанный автор даёт F1=0. "
                 "Такой интервал не является CI одной фиксированной функции, поэтому недействителен "
                 "как мера разброса macro-F1 (точка лежит выше его верхней границы); корректный интервал "
                 "требует предрегистрированного протокола с фиксированным набором меток. Accuracy "
                 "устойчива к той же процедуре."),
    }
    out_path = ROOT / "docs" / "stylo_lobo_authorci.json"
    out_path.write_text(dumps_strict(out, indent=2) + "\n", encoding="utf-8")
    print(f"\n✓ {out_path}", flush=True)


if __name__ == "__main__":
    main()
