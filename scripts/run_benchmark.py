"""Воспроизводимый exploratory-бенчмарк атрибуции авторства.

Гарантии валидности бенчмарка:
  • Один классификатор (LinearSVC) для ВСЕХ каналов — честное сравнение признаков.
  • До любого cache/fit весь uncapped корпус проходит cross-work content-isolation gate.
  • Векторизаторы/idf/scaler фитятся ВНУТРИ train-фолда, не на всех текстах.
  • StratifiedGroupKFold(5) ПО КНИГАМ — тест-книги не видны обучению.
  • Reference-метрика = macro-F1 (микро top-1/top-3 рядом); per-author recall + confusion.
  • Ансамбль — равновесное усреднение softmax (без подбора веса по тесту).
  • Детерминизм: фиксированный seed; результат публикуется атомарно только в
    docs/exploratory/channel_benchmark/{full,pd_only}/{all_channels,fast}.

Исторические ``docs/validation.json`` и ``docs/validation_pd.json`` являются
неизменяемыми входами сайта/README; этот runner их никогда не перезаписывает и
не авторизует новый публичный headline.

Запуск:  python scripts/run_benchmark.py            (все каналы)
         python scripts/run_benchmark.py --fast    (без DSP, быстрее)
Требует: опубликованный `stylo split` fragment snapshot (см. README: fetch → clean → split).
"""
from __future__ import annotations
import sys, time, math, argparse, hashlib, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from stylo.jsonio import dump_strict, dumps_strict  # noqa: E402
from stylo.domain.prediction_contract import (  # noqa: E402
    validate_class_indices,
    validate_score_matrix,
)
from stylo.eval.provenance import (  # noqa: E402
    derive_dataset,
    prepare_scientific_evaluation,
    safe_exploratory_dir,
    safe_write_batch,
)
import numpy as np
from collections import defaultdict, Counter
from scipy.special import softmax
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
import warnings; warnings.filterwarnings("ignore")
from stylo.config import load_config
from stylo.dataset import resolve_dataset, resolve_fragment_roots
from stylo.domain.work_weighting import CHUNK_WEIGHTED_LEGACY
from stylo.features.reps import make_rep_cache
from stylo.pipeline.train import _attestation
from stylo.corpus_tools.fetch_classics import PUBLIC_DOMAIN_CLEAR  # единый источник истины по юр-чистому PD (минус реабилитационные продления)

SEED = 42
CAP = 35           # макс. чанков с книги (баланс по объёму)
MIN_CHUNKS = 3     # мин. чанков в книге
MIN_BOOKS = 2      # мин. книг у автора (иначе LOBO/GroupKFold невозможен)
# ilf-petrov — соавторский дуэт; nikolas2 — дневники Николая II (не проза, не PD);
# sholohov разбирается отдельным disputed-authorship кейсом.
EXCLUDE = {"ilf-petrov", "nikolas2", "sholohov"}
DOCS = pathlib.Path(__file__).resolve().parents[1] / "docs"

def log(*a): print(*a, flush=True)

def resolve_benchmark_data(cfg) -> pathlib.Path:
    """Resolve and validate the one current fragment generation."""
    return resolve_fragment_roots(cfg).train_root


def prepare_benchmark_contexts(cfg, parent, *, pd_only: bool, rng=None):
    """Select eligible works, gate before CAP, then gate the exact capped child.

    Both derived datasets select directly from the disk-anchored parent so
    their provenance chains remain independently verifiable against disk.
    """

    if rng is None:
        rng = np.random.RandomState(SEED)
    group_rows = defaultdict(list)
    for index, raw_group in enumerate(parent.groups):
        group_rows[str(raw_group)].append(index)
    eligible_groups = {
        group
        for group, indexes in group_rows.items()
        if len(indexes) >= MIN_CHUNKS
    }
    works_per_author = Counter(
        group.split("/", 1)[0] for group in eligible_groups
    )
    eligible_authors = {
        author
        for author, count in works_per_author.items()
        if count >= MIN_BOOKS
        and (not pd_only or author in PUBLIC_DOMAIN_CLEAR)
    }
    selected_groups = {
        group
        for group in eligible_groups
        if group.split("/", 1)[0] in eligible_authors
    }
    uncapped_indices = [
        index
        for index, raw_group in enumerate(parent.groups)
        if str(raw_group) in selected_groups
    ]
    if not uncapped_indices:
        raise ValueError("benchmark selection produced no eligible rows")
    uncapped = derive_dataset(parent, uncapped_indices)
    uncapped_context = prepare_scientific_evaluation(
        cfg,
        uncapped,
        CHUNK_WEIGHTED_LEGACY,
    )

    capped_parent_indices = []
    for group in sorted(selected_groups):
        indexes = list(group_rows[group])
        if len(indexes) > CAP:
            chosen = sorted(rng.choice(len(indexes), CAP, replace=False))
            indexes = [indexes[index] for index in chosen]
        capped_parent_indices.extend(indexes)
    capped = derive_dataset(parent, sorted(capped_parent_indices))
    capped_context = prepare_scientific_evaluation(
        cfg,
        capped,
        CHUNK_WEIGHTED_LEGACY,
    )
    return uncapped_context, capped_context


def _benchmark_items(context):
    items = []
    for group in sorted({str(value) for value in context.groups}):
        author, book = group.split("/", 1)
        chunks = [
            str(text)
            for text, raw_group in zip(
                context.texts,
                context.groups,
                strict=True,
            )
            if str(raw_group) == group
        ]
        items.append((author, book, chunks))
    return items


def publish_benchmark_result(
    out,
    *,
    docs=DOCS,
    pd_only: bool,
    fast: bool,
):
    """Atomically publish one exploratory generation, never a historical site input."""

    mode = "pd_only" if pd_only else "full"
    variant = "fast" if fast else "all_channels"
    filename = f"validation.{mode}.{variant}.candidate.json"
    destination = safe_exploratory_dir(
        docs,
        "exploratory",
        "channel_benchmark",
        mode,
        variant,
    )
    candidate = (
        dumps_strict(
            out,
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    run_provenance = {
        "schema_version": "stylo.channel-benchmark-candidate-provenance.v1",
        "claim_status": "exploratory_internal",
        "public_headline_authorized": False,
        "supersedes": None,
        "candidate_file": filename,
        "candidate_sha256": hashlib.sha256(
            candidate.encode("utf-8")
        ).hexdigest(),
        "dataset": out["dataset_identity"],
        "attestation": out["attestation"],
    }
    published = safe_write_batch(
        destination,
        {
            filename: candidate,
            "run_provenance.json": dumps_strict(
                run_provenance,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        },
        publication_id=f"channel-benchmark-{mode}-{variant}-v1",
    )
    return published


# ── DSP: профиль словообразовательных суффиксов (stateless, без fit → без утечки) ──
SUF = sorted(["ость","ение","ание","ние","тель","ник","ниц","ист","изм","ация","ция","ство","еств","чик","щик",
              "арь","ач","ёж","льник","очк","ушк","ишк","ёнок","онок","знь","оват","еват","еньк","оньк","аст",
              "ив","лив","чив","еск","чат","альн","ова","ыва","ива","ничать","ировать","ствова","ани","ени"],
             key=len, reverse=True)
_DSP_NLP = [None]
def dsp_matrix(texts):
    import spacy, hashlib
    from stylo.jsonio import dump_strict, load_strict
    cf = pathlib.Path(__file__).resolve().parents[1] / "data" / "dsp_bench_cache.json"
    cache = load_strict(cf) if cf.exists() else {}
    if type(cache) is not dict:
        raise ValueError("DSP cache must be a strict JSON object")
    expected_width = len(SUF) + 2
    for key, values in cache.items():
        if (
            type(key) is not str
            or len(key) != 40
            or any(ch not in "0123456789abcdef" for ch in key)
            or type(values) is not list
            or len(values) != expected_width
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in values
            )
        ):
            raise ValueError("DSP cache contains a malformed entry")
    h = lambda s: hashlib.sha1(s.encode()).hexdigest()
    todo = [t for t in texts if h(t) not in cache]
    if todo:
        if _DSP_NLP[0] is None:
            _DSP_NLP[0] = spacy.load("ru_core_news_lg", disable=["parser", "ner"])
        for doc, txt in zip(_DSP_NLP[0].pipe(todo, batch_size=64), todo):
            types = set(tk.lemma_.lower() for tk in doc if tk.pos_ in {"NOUN","VERB","ADJ"} and tk.lemma_.isalpha() and len(tk.lemma_) > 3)
            prof = {s: 0 for s in SUF}; m = 0
            for l in types:
                for s in SUF:
                    if l.endswith(s) and len(l) > len(s) + 1:
                        prof[s] += 1; m += 1; break
            N = len(types) + 1; c = np.array([prof[s] for s in SUF], float)
            pp = c / (c.sum() + 1); ent = -sum(x*math.log2(x) for x in pp if x > 0) / math.log2(len(SUF))
            cache[h(txt)] = np.concatenate([c / N, [m / N, ent]]).tolist()
        dump_strict(cache, cf, sort_keys=True)
    return np.asarray([cache[h(t)] for t in texts], dtype=np.float64)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="без DSP-канала (быстрее)")
    ap.add_argument(
        "--pd-only",
        action="store_true",
        help="только public-domain авторы (exploratory-срез)",
    )
    args = ap.parse_args()
    pd_only = args.pd_only
    log(
        "режим корпуса: "
        f"{'PD-only exploratory' if pd_only else 'полный exploratory (вкл. копирайтных, локально)'}"
    )

    cfg = load_config()
    data_root = resolve_benchmark_data(cfg)
    log(f"fragment snapshot: {data_root}")
    configured_exclusions = set(
        cfg.get_path("corpus_policy.exclude_from_benchmark", []) or []
    )
    if configured_exclusions != EXCLUDE:
        raise ValueError(
            "benchmark exclusion policy drift: "
            f"configured={sorted(configured_exclusions)} "
            f"expected={sorted(EXCLUDE)}"
        )
    parent = resolve_dataset(
        cfg,
        CHUNK_WEIGHTED_LEGACY,
        data_root,
        exclude_authors=configured_exclusions,
        unknown_name=cfg.get_path(
            "corpus_policy.unknown_dir_name",
            "unknown",
        ),
    )
    uncapped_context, context = prepare_benchmark_contexts(
        cfg,
        parent,
        pd_only=pd_only,
    )
    attestation_before = _attestation(cfg)
    A = list(context.authors)
    aidx = {author: index for index, author in enumerate(A)}
    items = _benchmark_items(context)
    texts = [t for _, _, ch in items for t in ch]
    ychunk = np.array(
        [aidx[author] for author, _, chunks in items for _ in chunks]
    )
    gchunk = np.array(
        [
            f"{author}/{book}"
            for author, book, chunks in items
            for _ in chunks
        ],
        dtype=object,
    )
    book_author = context.book_to_author()
    ybook = np.array(
        [book_author[f"{author}/{book}"] for author, book, _ in items]
    )
    book_chunks = []
    off = 0
    for _, _, ch in items:
        book_chunks.append(list(range(off, off + len(ch)))); off += len(ch)
    log(f"корпус: авторов={len(A)} книг={len(items)} чанков={len(texts)}")

    t = time.time(); make_rep_cache(cfg).warm(texts, n_process=cfg.get_path("language.parse_n_process", 4))
    log(f"rep-кэш прогрет {time.time()-t:.0f}s")

    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(ychunk)), ychunk, gchunk))

    # ── каналы: единый источник — stylo.models.channels (fit ТОЛЬКО на train);
    #    DSP остаётся локальным (тяжёлый spaCy-lg кэш) ──
    from stylo.models.channels import make_channels
    def ch_dsp(tr, te):
        Etr = dsp_matrix(tr); Ete = dsp_matrix(te)
        sc = StandardScaler().fit(Etr); return sc.transform(Etr), sc.transform(Ete)

    CHANNELS = make_channels(cfg)
    if not args.fast:
        CHANNELS["DSP (suffixes)"] = ch_dsp

    def channel_book_scores(chan_fn):
        """OOF decision_function по чанкам (fit-within-fold), затем mean по книге → (n_books, n_classes)."""
        dfc = np.full((len(ychunk), len(A)), np.nan)
        for tr_i, te_i in splits:
            tr_txt = [texts[i] for i in tr_i]; te_txt = [texts[i] for i in te_i]
            Xtr, Xte = chan_fn(tr_txt, te_txt)
            clf = LinearSVC(C=1.0, class_weight="balanced", max_iter=3000, random_state=SEED).fit(Xtr, ychunk[tr_i])
            df = clf.decision_function(Xte); pres = clf.classes_
            validate_class_indices(
                pres, len(A), name="benchmark fold classifier.classes_"
            )
            if df.ndim == 1:
                dfc[te_i, pres[1]] = df; dfc[te_i, pres[0]] = -df
            else:
                dfc[te_i] = validate_score_matrix(
                    df,
                    rows=len(te_i),
                    n_classes=len(A),
                    name="benchmark fold decision_function",
                )
        validate_score_matrix(
            dfc,
            rows=len(ychunk),
            n_classes=len(A),
            name="benchmark complete OOF scores",
        )
        bs = np.empty((len(items), len(A)), dtype=np.float64)
        for k, ii in enumerate(book_chunks):
            if not ii:
                raise ValueError(f"book {k} has no chunks")
            bs[k] = dfc[np.asarray(ii)].mean(axis=0)
        return validate_score_matrix(
            bs,
            rows=len(items),
            n_classes=len(A),
            name="benchmark book scores",
        )

    def metrics(scores):
        pred = scores.argmax(1)
        top1 = float(np.mean(pred == ybook))
        top3 = float(np.mean([ybook[k] in scores[k].argsort()[::-1][:3] for k in range(len(ybook))]))
        mf1 = float(f1_score(ybook, pred, average="macro"))
        return top1, top3, mf1, pred

    log("\n%-24s %8s %8s %9s" % ("канал (один классификатор SVM)", "top-1", "top-3", "macro-F1"))
    chan_scores = {}; res = {}
    for name, fn in CHANNELS.items():
        t = time.time(); bs = channel_book_scores(fn); chan_scores[name] = bs
        t1, t3, mf, _ = metrics(bs); res[name] = {"top1": round(t1, 3), "top3": round(t3, 3), "macro_f1": round(mf, 3)}
        log("%-24s %8.3f %8.3f %9.3f  (%.0fs)" % (name, t1, t3, mf, time.time() - t))

    # ── ансамбль: равновесный + reliability-взвешенный (вес ∝ (OOF-acc − chance)^p) ──
    from stylo.eval.ensemble import reliability_weighted
    chance0 = 1.0 / len(A)
    ens = sum(softmax(bs, axis=1) for bs in chan_scores.values()) / len(chan_scores)
    t1, t3, mf, _ = metrics(ens)
    res["АНСАМБЛЬ (равновес.)"] = {"top1": round(t1, 3), "top3": round(t3, 3), "macro_f1": round(mf, 3)}
    log("%-24s %8.3f %8.3f %9.3f" % ("АНСАМБЛЬ (равновес.)", t1, t3, mf))
    # reliability-взвешенный ансамбль — ТОЛЬКО ДИАГНОСТИКА: его веса = OOF-точность каждого канала
    # НА ТЕХ ЖЕ тест-книгах, по которым потом отчитывается ансамбль → это test-set leak. Поэтому
    # headline'ом он быть НЕ может; оставляем как справочный вариант, честно помечая источник весов.
    oof_test_acc = {n: res[n]["top1"] for n in chan_scores}   # это TEST-OOF точность, не train
    diag_leaked = {}  # reliability^p — НЕ в channels: его веса выведены по тесту (leak), держим ОТДЕЛЬНО
    for p_ in [2.0, 4.0, 6.0]:
        ensw, _ = reliability_weighted(chan_scores, oof_test_acc, chance0, power=p_)
        wt1, wt3, wmf, _ = metrics(ensw)
        diag_leaked[f"reliability^{int(p_)}"] = {"top1": round(wt1, 3), "top3": round(wt3, 3), "macro_f1": round(wmf, 3)}
        log("%-24s %8.3f %8.3f %9.3f" % (f"reliability^{int(p_)} (диагностика)", wt1, wt3, wmf))
    # Exploratory reference = equal ensemble; its weights do not use test outcomes.
    reference_key = "АНСАМБЛЬ (равновес.)"
    _, _, _, pred = metrics(ens)
    t1, t3, mf = (
        res[reference_key]["top1"],
        res[reference_key]["top3"],
        res[reference_key]["macro_f1"],
    )
    log(
        "EXPLORATORY REFERENCE: %s (macro-F1 %.3f) — равновесный; "
        "веса не зависят от теста" % (reference_key, mf)
    )

    # Per-author recall + confusion. The historical author-resampled macro-F1
    # interval did not remap duplicated sampled labels and is withdrawn; this
    # exploratory runner does not emit a replacement inferential interval.
    byA = defaultdict(lambda: [0, 0]); conf = Counter()
    for k in range(len(ybook)):
        tr = A[ybook[k]]; pr = A[pred[k]]; byA[tr][1] += 1; byA[tr][0] += (tr == pr)
        if tr != pr: conf[(tr, pr)] += 1
    recalls = {a: round(c / n, 2) for a, (c, n) in sorted(byA.items(), key=lambda x: x[1][0] / x[1][1])}
    low = [a for a, r in recalls.items() if r <= 0.5]
    ci = None
    chance = 1.0 / len(A)
    log(
        "\nmacro-F1 author-clustered interval: WITHDRAWN "
        "(no replacement inferential protocol authorized) | "
        f"случайный (1/{len(A)})={chance:.3f}"
    )
    log(f"низкий recall (≤0.5): {low or 'нет'}")
    log(f"топ-путаниц: {[f'{a}->{b}x{c}' for (a,b),c in conf.most_common(8)]}")

    hb = res[reference_key]
    attestation_after = _attestation(cfg)
    if attestation_after != attestation_before:
        raise RuntimeError(
            "code/config/git attestation drifted during benchmark; "
            "refusing candidate publication"
        )
    dataset_identity = {
        "fragment_generation": data_root.parent.name,
        "training_weighting": context.weighting,
        "uncapped_rows_digest": uncapped_context.rows_digest,
        "uncapped_isolation_receipt_sha256": (
            uncapped_context.isolation_receipt_sha256
        ),
        "capped_rows_digest": context.rows_digest,
        "capped_isolation_receipt_sha256": (
            context.isolation_receipt_sha256
        ),
        "isolation_contract_version": (
            context.isolation_contract_version
        ),
        "selection": {
            "pd_only": pd_only,
            "min_chunks_per_work": MIN_CHUNKS,
            "min_works_per_author": MIN_BOOKS,
            "cap_chunks_per_work": CAP,
            "excluded_authors": sorted(configured_exclusions),
        },
    }
    out = {
        "claim_status": "exploratory_internal",
        "public_headline_authorized": False,
        "supersedes": None,
        "dataset_identity": dataset_identity,
        "attestation": attestation_after,
        "content_isolation": (
            "passed_on_uncapped_selection_and_exact_capped_child_"
            "before_cache_or_fit"
        ),
        "method": "LinearSVC (один для всех каналов) + StratifiedGroupKFold(5) book-level; cross-work content isolation проверена до fit; векторизаторы fit ВНУТРИ фолда; reference-ансамбль = РАВНОВЕСНОЕ усреднение softmax каналов (веса не зависят от теста); reliability^p — справочная диагностика с весами, выведенными по тесту",
        "corpus_mode": "pd_only" if pd_only else "full_research",
        "runner_variant": "fast" if args.fast else "all_channels",
        "corpus_note": ("классики, умершие >70 лет назад; тексты докачиваются по URL-манифесту для локальной валидации — "
                        "у Гумилёва и Пильняка охрана в РФ продлена после реабилитации (ст. 1281 п. 5 ГК), "
                        "downstream-редистрибуция — ответственность пользователя"
                        if pd_only else "полный исследовательский корпус ВКЛЮЧАЕТ копирайтных/живых авторов — НЕ редистрибутируемо; для exploratory PD-only среза используйте --pd-only"),
        "reference_ensemble": reference_key,
        "seed": SEED, "cap_chunks_per_book": CAP, "n_authors": len(A), "n_books": len(items), "n_chunks": len(texts),
        "chance_micro": round(chance, 4),
        "reference_macro_f1": hb["macro_f1"],
        "macro_f1_authorclustered_CI": ci,
        "macro_f1_authorclustered_interval_status": (
            "withdrawn_invalid_resampling_labels_not_remapped"
        ),
        "ensemble_top1": hb["top1"], "ensemble_top3": hb["top3"],
        "channels": res, "per_author_recall": recalls, "low_recall_authors": low,
        "_diagnostic_test_leaked": {"_warning": "веса этих ансамблей выведены по ТЕСТУ (leak) — НЕ headline, только справочно", **diag_leaked},
        "top_confusions": [f"{a}->{b}x{c}" for (a, b), c in conf.most_common(12)], "authors": A,
        "notes": [
            "Честное сравнение: ВСЕ каналы под одним классификатором (LinearSVC); смена nearest-centroid→SVM сама по себе даёт прибавку — это вклад классификатора, не признаков.",
            "Топик-инвариантный синтаксис НЕ превосходит char-n-граммы по точности (trade-off, не преимущество по accuracy).",
            "Бенчмарк на собственном корпусе, НЕ на стандартном PAN/RusProfiling — claim 'SOTA' не делается.",
            "exploratory reference-ансамбль = РАВНОВЕСНЫЙ (веса не зависят от теста); reliability^p — справочная диагностика, её веса выведены по тесту.",
            "Воспроизводимо как exploratory-run: python scripts/run_benchmark.py [--pd-only] (требует собранного content-safe корпуса; seed фиксирован).",
        ],
    }
    published = publish_benchmark_result(
        out,
        pd_only=pd_only,
        fast=args.fast,
    )
    log(
        "\n✓ exploratory generation saved: "
        f"{published['run_provenance.json'].parent}"
    )

if __name__ == "__main__":
    main()
