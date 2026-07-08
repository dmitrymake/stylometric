"""Адверсариальная проба провенанса: предсказуем ли ИСТОЧНИК книги (Викитека vs локальная
оцифровка) по тем же признаковым каналам, через которые «след издания» мог бы протечь в
атрибуцию (символьные n-граммы, пунктуация)?

Постановка: авторы с книгами из двух источников (по docs/corpus_manifest.json), чанки —
те же, что видит бенчмарк (data/frags_train, после pipeline/clean). Автор как конфаунд
убран протоколом leave-one-author-out: классификатор источника обучается на чанках
остальных смешанных авторов и предсказывает книги отложенного. AUC≈0.5 — след издания
в признаках не выражен; AUC≫0.5 — выражен.

Оговорки, зашитые в вывод: класс «local/неизвестно» гетерогенен (это «не Викитека», а не
одно издание); установленный источник в корпусе один (Викитека), поэтому проба измеряет
границу «Викитека vs остальное», а не различие двух конкретных изданий.

Дополнительно: связь LOBO-ошибок headline-модели с источником книги (те же смешанные
авторы; точный тест Фишера).

Выход: docs/provenance_probe.json.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

DOCS = ROOT / "docs"
SEED = 42
RNG = np.random.default_rng(SEED)


def load_labels():
    man = json.loads((DOCS / "corpus_manifest.json").read_text())
    labels = {}   # (author, book) -> 1 если Викитека, 0 если local
    mixed = []
    for author, a in man["authors"].items():
        books = a.get("books", [])
        ws = [b for b in books if str(b.get("source", "")).startswith("ru.wikisource.org")]
        loc = [b for b in books if str(b.get("source", "")) == "local/неизвестно"]
        if ws and loc:
            mixed.append(author)
            for b in ws:
                labels[(author, b["book"])] = 1
            for b in loc:
                labels[(author, b["book"])] = 0
    return mixed, labels


def load_chunks(mixed, labels):
    texts, authors, books, y = [], [], [], []
    root = ROOT / "data" / "frags_train"
    for author in mixed:
        for bdir in sorted((root / author).iterdir()):
            if not bdir.is_dir():
                continue
            key = (author, bdir.name)
            if key not in labels:
                continue
            for fp in sorted(bdir.glob("*.txt")):
                t = fp.read_text(encoding="utf-8").strip()
                if t:
                    texts.append(t)
                    authors.append(author)
                    books.append(f"{author}/{bdir.name}")
                    y.append(labels[key])
    return (np.asarray(texts, dtype=object), np.asarray(authors), np.asarray(books),
            np.asarray(y, dtype=int))


PUNCT_RE = re.compile(r"[^\w\s]")


def punct_view(texts):
    """Текст → только знаки пунктуации (с пробелом-разделителем): изолирует канал пунктуации."""
    return [" ".join(PUNCT_RE.findall(t)) for t in texts]


def build_folds(texts, authors, books, vec_factory, view):
    """LOAO-фолды с ОДНОКРАТНОЙ векторизацией: векторизатор не видит меток источника,
    поэтому в перестановочном нуле фичи фолда переиспользуются, перефитится только LR."""
    folds = []
    for held in sorted(set(authors.tolist())):
        te = authors == held
        tr = ~te
        vec = vec_factory()
        Xtr = vec.fit_transform(view(texts[tr]))
        Xte = vec.transform(view(texts[te]))
        folds.append({"held": held, "tr": tr, "te": te, "Xtr": Xtr, "Xte": Xte,
                      "books_te": books[te]})
    return folds


def loao_book_auc(folds, y, q=None):
    """LOAO по готовым фолдам: книжные score (средняя P(ws) по чанкам) в пул → AUC по книгам."""
    scores, truth = [], []
    for f in folds:
        y_tr = y[f["tr"]]
        if len(set(y_tr.tolist())) < 2:
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=SEED)
        clf.fit(f["Xtr"], y_tr)
        p = clf.predict_proba(f["Xte"])[:, 1]
        y_te = y[f["te"]]
        for b in sorted(set(f["books_te"].tolist())):
            m = f["books_te"] == b
            scores.append(float(p[m].mean()))
            truth.append(int(y_te[m][0]))
    return float(roc_auc_score(truth, scores)), len(truth)


def null_auc_quantile(folds, authors, books, y, n_perm=200, q=95):
    """Нуль: перестановка меток источника МЕЖДУ КНИГАМИ внутри автора (доли сохранены)."""
    book_list = sorted(set(books.tolist()))
    b2a = {b: b.split("/", 1)[0] for b in book_list}
    b2y = {b: int(y[books == b][0]) for b in book_list}
    aucs = []
    for _ in range(n_perm):
        y_perm = np.empty_like(y)
        for a in sorted(set(authors.tolist())):
            bs = [b for b in book_list if b2a[b] == a]
            vals = [b2y[b] for b in bs]
            RNG.shuffle(vals)
            for b, v in zip(bs, vals):
                y_perm[books == b] = v
        try:
            auc, _ = loao_book_auc(folds, y_perm)
            aucs.append(auc)
        except ValueError:
            continue
    return float(np.percentile(aucs, q)), float(np.median(aucs)), len(aucs)


HDR = re.compile(r"^\[(OK|MISS)\s*\]\s+(.+?)\s*/\s*(\S+)\s+\(rank")


def lobo_errors_vs_source(labels):
    """Связь ошибок headline-LOBO с источником книги (смешанные авторы)."""
    from stylo.lang import display_name
    from scipy.stats import fisher_exact
    man_authors = sorted({a for a, _ in labels})
    disp2id = {display_name(a): a for a in man_authors}
    table = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}  # (ws, correct) -> n
    for line in (DOCS / "lobo_books.txt").read_text(encoding="utf-8").splitlines():
        h = HDR.match(line)
        if not h:
            continue
        status, disp, book = h.group(1), h.group(2).strip(), h.group(3)
        aid = disp2id.get(disp)
        if aid is None or (aid, book) not in labels:
            continue
        table[(labels[(aid, book)], int(status == "OK"))] += 1
    m = [[table[(1, 1)], table[(1, 0)]], [table[(0, 1)], table[(0, 0)]]]
    odds, p = fisher_exact(m)
    return {"contingency_ws_correct": {"ws_ok": m[0][0], "ws_miss": m[0][1],
                                       "local_ok": m[1][0], "local_miss": m[1][1]},
            "fisher_odds_ratio": round(float(odds), 3) if np.isfinite(odds) else None,
            "fisher_p": round(float(p), 4)}


def main():
    mixed, labels = load_labels()
    texts, authors, books, y = load_chunks(mixed, labels)
    n_books = len(set(books.tolist()))
    print(f"смешанные авторы: {mixed}; книг {n_books}, чанков {len(texts)}, ws-чанков {int(y.sum())}", flush=True)

    char_vec = lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=30000,
                                       sublinear_tf=True)
    punct_vec = lambda: TfidfVectorizer(analyzer="char", ngram_range=(1, 3), max_features=3000,
                                        sublinear_tf=True)

    res = {}
    for name, vf, view in [("char_3_5", char_vec, lambda x: list(x)),
                           ("punctuation_only", punct_vec, punct_view)]:
        folds = build_folds(texts, authors, books, vf, view)
        print(f"[{name}] фолды векторизованы", flush=True)
        auc, nb = loao_book_auc(folds, y)
        q95, med, nperm = null_auc_quantile(folds, authors, books, y, n_perm=100)
        res[name] = {"book_auc_loao": round(auc, 3), "n_books_scored": nb,
                     "null_book_auc_median": round(med, 3), "null_book_auc_q95": round(q95, 3),
                     "n_null_permutations": nperm,
                     "above_null_q95": bool(auc > q95)}
        print(f"[{name}] AUC={auc:.3f} (нуль: медиана {med:.3f}, q95 {q95:.3f}, n={nperm})", flush=True)

    lobo_link = lobo_errors_vs_source(labels)
    print(f"[lobo↔источник] {lobo_link}", flush=True)

    out = {
        "method": ("Проба «предскажи источник книги (Викитека vs локальная оцифровка) по признакам, "
                   "через которые след издания мог бы протечь»: чанки бенчмарка (data/frags_train, "
                   "после нормализации clean.py), авторы с книгами из двух источников, "
                   "leave-one-author-out (автор как конфаунд исключён), score книги = средняя "
                   "вероятность чанков, AUC по книгам + перестановочный нуль (метки источника "
                   "переставляются между книгами внутри автора)."),
        "mixed_authors": mixed,
        "n_books": n_books,
        "n_chunks": int(len(texts)),
        "probes": res,
        "lobo_errors_vs_source": lobo_link,
        "caveats": ("Класс «local/неизвестно» гетерогенен — это «не Викитека», а не одно издание; "
                    "установленное семейство источников в корпусе одно (Викитека), поэтому проба "
                    "измеряет границу «Викитека vs остальное». AUC выше нуля-q95 означает: след "
                    "источника в признаках выражен и для одноисточниковых авторов не отделим от "
                    "идиолекта средствами LOBO; AUC в пределах нуля — след издания в этих каналах "
                    "не доминирует."),
    }
    (DOCS / "provenance_probe.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"записано: {DOCS / 'provenance_probe.json'}", flush=True)


if __name__ == "__main__":
    main()
