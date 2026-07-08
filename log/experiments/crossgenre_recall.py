"""Кросс-жанровый перенос атрибуции: обучение на художественной прозе,
проверка на дневниках и письмах тех же авторов.

Протокол (без утечек):
  • Train: все прозаические чанки data/frags_train, те же классы и исключения,
    что в основном бенчмарке (exclude_from_benchmark из configs/default.yaml);
    штатный пайплайн make_full_pipeline + StyloVectorizer.from_config.
  • Test: файлы input_personal/{author}_{diary|letters} для авторов, у которых
    есть проза в train-классах. Каждый файл = документ. Тексты проходят ту же
    очистку normalize() (маскировка имён, тире, орфография) и ту же нарезку
    по предложениям (500 слов, минимум 200), что и проза.
  • Метрики: top-1/top-3 recall на уровне документа (среднее вероятностей
    чанков документа, как в eval/lobo._align_proba) и на уровне чанков;
    стоки ошибок = самые частые ложные кандидаты.

Результат: docs/crossgenre_recall.json.
Запуск (щадящий для общей машины):
  cd <repo> && nice -n 10 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 .venv/bin/python log/experiments/crossgenre_recall.py
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import hashlib
import json
import pathlib
import sys
import time
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)  # конфиг использует относительные пути (data/, docs/)

import joblib
import numpy as np
import warnings; warnings.filterwarnings("ignore")

from stylo.config import load_config
from stylo.corpus import load_dataset
from stylo.chunking import CombinedDoc, make_sent_chunks, sentences_for_text
from stylo.features.reps import make_rep_cache
from stylo.models.lr import make_full_pipeline
from stylo.nlp import load_sentencizer
from stylo.pipeline.clean import normalize
from stylo.vectorizer import StyloVectorizer

N_PROC = 4          # общая машина: не больше 4 процессов spaCy
GENRES = ("diary", "letters")
PERSONAL = ROOT / "input_personal"
NORM_CACHE = ROOT / "log" / "experiments" / "_crossgenre_norm_cache"
MODEL_CACHE = ROOT / "log" / "experiments" / "_crossgenre_model.pkl"


def log(*a):
    print(*a, flush=True)


# ── 1. Train: штатный пайплайн на всех прозаических чанках бенчмарк-классов ──
def train_pipeline(cfg):
    exclude = set(cfg.get_path("corpus_policy.exclude_from_benchmark", []) or [])
    ds = load_dataset(ROOT / "data" / "frags_train", exclude_authors=exclude,
                      unknown_name=cfg.get_path("corpus_policy.unknown_dir_name", "unknown"))
    fp = hashlib.sha1(("|".join(ds.authors) + f"#{len(ds)}"
                       + "|".join(sorted(set(map(str, ds.groups))))).encode()).hexdigest()[:12]
    log(f"train: {ds.n_authors} авторов, {len(set(ds.groups))} книг, {len(ds)} чанков "
        f"(исключены: {sorted(exclude)}); отпечаток корпуса {fp}")

    if MODEL_CACHE.exists():
        cached = joblib.load(MODEL_CACHE)
        if cached.get("fingerprint") == fp:
            log("train: модель из кеша эксперимента (тот же корпус)")
            return cached["pipe"], ds
    t = time.time()
    make_rep_cache(cfg).warm(list(ds.texts), n_process=N_PROC)
    log(f"train: rep-кэш прозы готов ({time.time()-t:.0f}s)")
    vec = StyloVectorizer.from_config(cfg)
    pipe = make_full_pipeline(cfg, vec)
    t = time.time()
    pipe.fit(list(ds.texts), ds.y)
    log(f"train: пайплайн обучен ({time.time()-t:.0f}s)")
    joblib.dump({"fingerprint": fp, "pipe": pipe}, MODEL_CACHE)
    return pipe, ds


# ── 2. Test: личные жанры тех же авторов, та же очистка и нарезка ──
def normalize_cached(raw: str, model: str, fallback):
    """normalize() дорогая (NER); результат кешируется по хэшу сырого текста."""
    NORM_CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1((model + "\x00" + raw).encode()).hexdigest()
    fp = NORM_CACHE / f"{key}.txt"
    if fp.exists():
        return fp.read_text("utf-8")
    clean = normalize(raw, model, fallback)
    fp.write_text(clean, "utf-8")
    return clean


def collect_test_docs(cfg, train_authors):
    model = cfg.get_path("language.spacy_model", "ru_core_news_lg")
    fallback = cfg.get_path("language.spacy_fallback", None)
    size = cfg.get_path("chunking.chunk_size", 500)
    min_words = cfg.get_path("chunking.min_words", 200)
    sent_nlp = load_sentencizer(cfg.get_path("language.code", "ru"))

    docs = []  # {author, genre, doc_id, chunks}
    for author in train_authors:
        for genre in GENRES:
            d = PERSONAL / f"{author}_{genre}"
            if not d.is_dir():
                continue
            for f in sorted(d.glob("*.txt")):
                raw = f.read_text("utf-8", errors="ignore").strip()
                if not raw:
                    continue
                t = time.time()
                clean = normalize_cached(raw, model, fallback)
                sents = sentences_for_text(clean, sent_nlp)
                chunks = make_sent_chunks(CombinedDoc(sents), size, min_words, 0.0)
                if not chunks:
                    continue
                docs.append({"author": author, "genre": genre, "doc_id": f.stem,
                             "chunks": chunks})
                log(f"test: {author}/{genre}/{f.stem}: {len(chunks)} чанков "
                    f"({sum(len(c.split()) for c in chunks)} слов, {time.time()-t:.0f}s)")
    return docs


def rank_of(p_row: np.ndarray, true_idx: int) -> int:
    """Tie-aware ранг истинного автора (как в eval/lobo): классы с prob >= p_true."""
    return int((p_row >= p_row[true_idx]).sum())


def main():
    cfg = load_config()
    pipe, ds = train_pipeline(cfg)
    authors = ds.authors
    aidx = {a: i for i, a in enumerate(authors)}

    docs = collect_test_docs(cfg, authors)
    all_chunks = [c for d in docs for c in d["chunks"]]
    log(f"test: {len({d['author'] for d in docs})} авторов, {len(docs)} документов, "
        f"{len(all_chunks)} чанков")

    t = time.time()
    make_rep_cache(cfg).warm(all_chunks, n_process=N_PROC)
    log(f"test: rep-кэш личных текстов готов ({time.time()-t:.0f}s)")

    # вероятности одним батчем, выровненные на полный список авторов
    t = time.time()
    proba = pipe.predict_proba(all_chunks)
    classes = np.asarray(pipe.named_steps["classifier"].classes_, dtype=int)
    full = np.zeros((len(all_chunks), len(authors)))
    full[:, classes] = proba
    log(f"test: вероятности посчитаны ({time.time()-t:.0f}s)")

    # ── 3. Метрики ──
    per_doc = []
    cell = {}  # (author, genre) -> аккумуляторы
    off = 0
    for d in docs:
        n = len(d["chunks"])
        P = full[off:off + n]; off += n
        ti = aidx[d["author"]]
        doc_p = P.mean(axis=0)                      # как _align_proba в eval/lobo
        r = rank_of(doc_p, ti)
        order = np.argsort(-doc_p, kind="stable")
        top3_names = [(authors[int(i)], round(float(doc_p[int(i)]), 4)) for i in order[:3]]
        ch_ranks = np.array([rank_of(P[k], ti) for k in range(n)])
        ch_pred = [authors[int(i)] for i in P.argmax(axis=1)]
        per_doc.append({
            "author": d["author"], "genre": d["genre"], "doc": d["doc_id"],
            "n_chunks": n, "doc_rank": r, "doc_top1": bool(r == 1), "doc_top3": bool(r <= 3),
            "doc_pred": authors[int(order[0])], "doc_top3_candidates": top3_names,
            "chunk_top1": round(float((ch_ranks == 1).mean()), 3),
            "chunk_top3": round(float((ch_ranks <= 3).mean()), 3),
        })
        key = (d["author"], d["genre"])
        c = cell.setdefault(key, {"docs": 0, "doc_top1": 0, "doc_top3": 0, "chunks": 0,
                                  "chunk_top1": 0, "chunk_top3": 0, "sink": Counter()})
        c["docs"] += 1; c["doc_top1"] += (r == 1); c["doc_top3"] += (r <= 3)
        c["chunks"] += n
        c["chunk_top1"] += int((ch_ranks == 1).sum()); c["chunk_top3"] += int((ch_ranks <= 3).sum())
        c["sink"].update(p for p in ch_pred if p != d["author"])

    table = {}
    for (a, g), c in sorted(cell.items()):
        table.setdefault(a, {})[g] = {
            "n_docs": c["docs"], "n_chunks": c["chunks"],
            "doc_top1": round(c["doc_top1"] / c["docs"], 3),
            "doc_top3": round(c["doc_top3"] / c["docs"], 3),
            "chunk_top1": round(c["chunk_top1"] / c["chunks"], 3),
            "chunk_top3": round(c["chunk_top3"] / c["chunks"], 3),
            "false_candidates_top3": [f"{n}×{k}" for n, k in c["sink"].most_common(3)],
        }

    def agg(pred):
        sel = [x for x in per_doc if pred(x)]
        ch = [(k, g) for k, g in cell.items() if pred({"author": k[0], "genre": k[1]})]
        n_ch = sum(g["chunks"] for _, g in ch)
        return {
            "n_docs": len(sel),
            "doc_top1": round(float(np.mean([x["doc_top1"] for x in sel])), 3),
            "doc_top3": round(float(np.mean([x["doc_top3"] for x in sel])), 3),
            "n_chunks": n_ch,
            "chunk_top1": round(sum(g["chunk_top1"] for _, g in ch) / n_ch, 3),
            "chunk_top3": round(sum(g["chunk_top3"] for _, g in ch) / n_ch, 3),
        }

    aggregate = {
        "all": agg(lambda x: True),
        "diary": agg(lambda x: x["genre"] == "diary"),
        "letters": agg(lambda x: x["genre"] == "letters"),
    }
    sink_all = Counter()
    for c in cell.values():
        sink_all.update(c["sink"])

    for genre in ("all", "diary", "letters"):
        a = aggregate[genre]
        log(f"agg {genre:8}: docs {a['n_docs']:3} top1={a['doc_top1']:.3f} top3={a['doc_top3']:.3f} | "
            f"chunks {a['n_chunks']:5} top1={a['chunk_top1']:.3f} top3={a['chunk_top3']:.3f}")
    log(f"стоки ошибок (чанки): {sink_all.most_common(8)}")
    for a, gs in table.items():
        for g, r in gs.items():
            log(f"  {a:12} {g:8} docs {r['n_docs']}: top1={r['doc_top1']} top3={r['doc_top3']} | "
                f"chunks {r['n_chunks']}: top1={r['chunk_top1']} top3={r['chunk_top3']} | "
                f"ложные: {r['false_candidates_top3']}")

    out = {
        "title": "Кросс-жанровый перенос атрибуции: обучение на прозе, проверка на дневниках и письмах",
        "method": (
            "Модель обучена только на художественной прозе (все прозаические чанки основного бенчмарка, "
            f"{ds.n_authors} авторов, штатный пайплайн: стилометрические признаки + логистическая регрессия). "
            "Проверка — на дневниках и письмах тех авторов, чья проза есть в обучении; эти тексты модель "
            "при обучении не видела, и жанр другой. Каждый файл дневника/писем = документ; он проходит ту же "
            "очистку (маскировка имён, унификация тире и орфографии) и ту же нарезку на куски по ~500 слов, "
            "что и проза. top-1 recall = доля случаев, когда модель ставит настоящего автора на первое место "
            "среди всех кандидатов; top-3 — когда он попадает в первую тройку. Оценка на уровне документа "
            "(среднее вероятностей его кусков) и на уровне отдельных кусков."
        ),
        "date": "2026-07-02",
        "train": {
            "source": "data/frags_train (художественная проза основного бенчмарка)",
            "excluded": sorted(cfg.get_path("corpus_policy.exclude_from_benchmark", []) or []),
            "n_authors": ds.n_authors,
            "n_books": len(set(map(str, ds.groups))),
            "n_chunks": len(ds),
            "chance_top1": round(1.0 / ds.n_authors, 4),
        },
        "test": {
            "source": "input_personal/{автор}_{diary|letters}, автор имеет прозу в обучении",
            "n_authors": len({d["author"] for d in docs}),
            "authors": sorted({d["author"] for d in docs}),
            "n_documents": len(docs),
            "n_chunks": len(all_chunks),
            "composition": {f"{a}_{g}": {"n_docs": c["docs"], "n_chunks": c["chunks"]}
                            for (a, g), c in sorted(cell.items())},
        },
        "recall_table": table,
        "aggregate": aggregate,
        "error_sinks": {
            "overall_top": [f"{n}×{k}" for n, k in sink_all.most_common(10)],
            "note_sinks": "Стоки = авторы, которых модель ошибочно называет вместо настоящего (счёт по кускам).",
        },
        "documents": per_doc,
        "note": (
            "Ориентир, не бенчмарк: дневники и письма в открытом доступе есть лишь у 4-5 авторов "
            "обучающего набора — те же пределы данных, что зафиксированы в docs/authorship_cases.json "
            "(~7 авторов с обоими жанрами, включая авторов вне прозаического бенчмарка). "
            "Файлы писем и дневников — редакторские собрания: даты, адресаты и примечания издания "
            "входят в текст. Чехов-письма 1875-1886 включают фрагменты рукой брата Николая. "
            "Кандидатов в модели "
            f"{ds.n_authors}; случайное попадание top-1 = {round(1.0 / ds.n_authors, 4)}."
        ),
    }
    (ROOT / "docs" / "crossgenre_recall.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    log("сохранено: docs/crossgenre_recall.json")


if __name__ == "__main__":
    main()
