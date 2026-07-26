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
import sys, time, math, argparse, hashlib, pathlib, platform, tempfile
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
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
from stylo.config import load_config, with_overrides
from stylo.dataset import resolve_dataset, resolve_fragment_roots
from stylo.domain.work_weighting import CHUNK_WEIGHTED_LEGACY
from stylo.eval.paired_audit.run_plan import verify_installed_environment
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
DOCS = ROOT / "docs"

_BENCHMARK_CACHE_AUTHORITY = {
    "schema_version": "stylo.channel-benchmark-cache-authority.v1",
    "mode": "fresh_ephemeral_recompute",
    "persistent_cache_reads_allowed": False,
    "representation_cache": "unique_empty_temporary_root",
    "doc_cache": "unique_empty_temporary_root",
    "dsp_cache": "run_local_empty_mapping",
    "process_memory_precondition": (
        "representation_doc_and_nlp_caches_empty"
    ),
}


def _validated_nlp_identity(cfg, raw_identity):
    """Validate and normalise the exact pipeline that produced all NLP features."""

    from importlib.metadata import version

    required = {
        "requested_model",
        "resolved_model",
        "fallback_used",
        "package_version",
        "package_record_sha256",
        "spacy_version",
        "disabled_pipes",
        "active_pipes",
        "max_length",
        "identity_sha256",
    }
    if type(raw_identity) is not dict or set(raw_identity) != required:
        raise RuntimeError("benchmark spaCy identity has an unexpected schema")
    for field in (
        "requested_model",
        "resolved_model",
        "package_version",
        "package_record_sha256",
        "spacy_version",
        "identity_sha256",
    ):
        if type(raw_identity[field]) is not str or not raw_identity[field]:
            raise RuntimeError(f"benchmark spaCy identity {field} is invalid")
    if type(raw_identity["fallback_used"]) is not bool:
        raise RuntimeError("benchmark spaCy fallback flag is invalid")
    if raw_identity["fallback_used"]:
        raise RuntimeError("benchmark refuses a fallback spaCy model")
    configured_model = cfg.get_path("language.spacy_model")
    configured_version = str(cfg.get_path("language.spacy_model_version"))
    if (
        raw_identity["requested_model"] != configured_model
        or raw_identity["resolved_model"] != configured_model
        or raw_identity["package_version"] != configured_version
    ):
        raise RuntimeError(
            "resolved spaCy model/version does not match the benchmark config"
        )
    if raw_identity["spacy_version"] != version("spacy"):
        raise RuntimeError("resolved spaCy runtime version drifted")
    for field in ("package_record_sha256", "identity_sha256"):
        value = raw_identity[field]
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise RuntimeError(f"benchmark spaCy identity {field} is not SHA-256")
    if (
        type(raw_identity["max_length"]) is not int
        or raw_identity["max_length"] <= 0
    ):
        raise RuntimeError("benchmark spaCy max_length is invalid")
    for field in ("disabled_pipes", "active_pipes"):
        value = raw_identity[field]
        if not isinstance(value, (list, tuple)) or any(
            type(item) is not str or not item for item in value
        ):
            raise RuntimeError(f"benchmark spaCy identity {field} is invalid")
        if len(set(value)) != len(value):
            raise RuntimeError(f"benchmark spaCy identity {field} has duplicates")

    normalised = {
        **raw_identity,
        "disabled_pipes": list(raw_identity["disabled_pipes"]),
        "active_pipes": list(raw_identity["active_pipes"]),
    }
    identity_body = {
        key: normalised[key]
        for key in (
            "requested_model",
            "resolved_model",
            "fallback_used",
            "package_version",
            "package_record_sha256",
            "spacy_version",
            "disabled_pipes",
            "active_pipes",
            "max_length",
        )
    }
    expected = hashlib.sha256(
        dumps_strict(
            identity_body,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if normalised["identity_sha256"] != expected:
        raise RuntimeError("benchmark spaCy identity digest does not match its fields")
    return normalised


def snapshot_benchmark_nlp_identity(cfg, nlp):
    """Recompute the installed model/pipeline identity from the live object."""

    from stylo.nlp import _build_nlp_identity

    requested = cfg.get_path("language.spacy_model")
    return _build_nlp_identity(
        requested=requested,
        resolved=requested,
        nlp=nlp,
        max_length=nlp.max_length,
    ).to_dict()


def _validated_environment_contract(root):
    contract = verify_installed_environment(root)
    required = {
        "schema_version",
        "python_implementation",
        "python_major_minor",
        "distributions",
        "environment_lock_identity_sha256",
        "contract_sha256",
    }
    if type(contract) is not dict or set(contract) != required:
        raise RuntimeError("benchmark installed-environment contract is incomplete")
    body = {key: contract[key] for key in required - {"contract_sha256"}}
    expected = hashlib.sha256(
        dumps_strict(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if contract["contract_sha256"] != expected:
        raise RuntimeError("benchmark installed-environment contract digest is invalid")
    return contract


def benchmark_attestation(cfg, *, nlp_identity, root=ROOT):
    """Bind a candidate to clean source, exact environment, NLP and cache policy."""

    from importlib.metadata import version

    root = pathlib.Path(root).resolve()
    attestation = _attestation(cfg)
    required = {
        "git_commit",
        "git_dirty",
        "code_tree_sha256",
        "config_id",
    }
    if set(attestation) != required or any(
        attestation[field] is None
        for field in ("git_commit", "code_tree_sha256", "config_id")
    ):
        raise RuntimeError("benchmark source attestation is incomplete")
    if attestation["git_dirty"] is not False:
        raise RuntimeError(
            "benchmark candidate publication requires a clean Git worktree"
        )

    bound_files = {
        "runner_sha256": root / "scripts" / "run_benchmark.py",
        "requirements_lock_sha256": root / "requirements.lock",
        "pyproject_sha256": root / "pyproject.toml",
    }
    for field, path in bound_files.items():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"benchmark attestation input is missing/unsafe: {path}")
        attestation[field] = hashlib.sha256(path.read_bytes()).hexdigest()
    attestation["runtime"] = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "scikit_learn": version("scikit-learn"),
        "spacy": version("spacy"),
    }
    attestation["installed_environment"] = _validated_environment_contract(root)
    attestation["nlp_model_identity"] = _validated_nlp_identity(
        cfg,
        nlp_identity,
    )
    attestation["cache_authority"] = dict(_BENCHMARK_CACHE_AUTHORITY)
    return attestation


def isolated_benchmark_config(cfg, workspace):
    """Route all disk-backed feature caches to one verified empty temp root."""

    from stylo import nlp as nlp_module
    from stylo.features import reps as reps_module

    workspace = pathlib.Path(workspace)
    if workspace.is_symlink() or not workspace.is_dir():
        raise RuntimeError("benchmark cache workspace is missing or unsafe")
    workspace = workspace.resolve(strict=True)
    if any(workspace.iterdir()):
        raise RuntimeError("benchmark cache workspace must start empty")
    if (
        reps_module._MEM_REPS
        or nlp_module._MEM_DOCS
        or nlp_module._NLP_CACHE
        or nlp_module._NLP_IDENTITIES
    ):
        raise RuntimeError(
            "benchmark requires empty process-local representation/Doc/NLP caches"
        )
    return with_overrides(
        cfg,
        {
            "paths.data": str(workspace / "representation"),
            "paths.doc_cache": str(workspace / "doc"),
        },
    )


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


def dsp_matrix(texts, *, nlp, nlp_identity_sha256, cache):
    """Build DSP features through an explicitly run-local, identity-scoped cache."""

    if type(cache) is not dict:
        raise ValueError("DSP cache must be an exact run-local dict")
    if (
        type(nlp_identity_sha256) is not str
        or len(nlp_identity_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in nlp_identity_sha256)
    ):
        raise ValueError("DSP spaCy identity must be an exact SHA-256")
    expected_width = len(SUF) + 2
    for key, values in cache.items():
        if (
            type(key) is not str
            or len(key) != 64
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
    def cache_key(text):
        return hashlib.sha256(
            nlp_identity_sha256.encode("ascii")
            + b"\0"
            + text.encode("utf-8")
        ).hexdigest()

    todo = [
        (cache_key(text), text)
        for text in texts
        if cache_key(text) not in cache
    ]
    if todo:
        todo_texts = [text for _key, text in todo]
        for doc, (key, txt) in zip(
            nlp.pipe(todo_texts, batch_size=64),
            todo,
            strict=True,
        ):
            if doc.text != txt:
                raise RuntimeError("DSP spaCy pipeline changed input text")
            types = set(tk.lemma_.lower() for tk in doc if tk.pos_ in {"NOUN","VERB","ADJ"} and tk.lemma_.isalpha() and len(tk.lemma_) > 3)
            prof = {s: 0 for s in SUF}; m = 0
            for l in types:
                for s in SUF:
                    if l.endswith(s) and len(l) > len(s) + 1:
                        prof[s] += 1; m += 1; break
            N = len(types) + 1; c = np.array([prof[s] for s in SUF], float)
            pp = c / (c.sum() + 1); ent = -sum(x*math.log2(x) for x in pp if x > 0) / math.log2(len(SUF))
            cache[key] = np.concatenate([c / N, [m / N, ent]]).tolist()
    matrix = np.asarray(
        [cache[cache_key(text)] for text in texts],
        dtype=np.float64,
    )
    if matrix.shape != (len(texts), expected_width) or not np.isfinite(matrix).all():
        raise RuntimeError("DSP matrix is incomplete or non-finite")
    return matrix

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
    cache_workspace = tempfile.TemporaryDirectory(
        prefix="stylo-benchmark-cache-",
    )
    cache_root = pathlib.Path(cache_workspace.name)
    runtime_cfg = isolated_benchmark_config(cfg, cache_root)
    rep_cache = make_rep_cache(runtime_cfg)
    benchmark_nlp = rep_cache.doc_cache.nlp
    nlp_identity = snapshot_benchmark_nlp_identity(cfg, benchmark_nlp)
    attestation_before = benchmark_attestation(
        cfg,
        nlp_identity=nlp_identity,
    )
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

    t = time.time(); rep_cache.warm(texts, n_process=cfg.get_path("language.parse_n_process", 4))
    log(f"rep-кэш прогрет {time.time()-t:.0f}s")

    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(ychunk)), ychunk, gchunk))

    # ── каналы: единый источник — stylo.models.channels (fit ТОЛЬКО на train);
    #    DSP остаётся локальным и использует только run-local memory ──
    from stylo.models.channels import make_channels
    dsp_cache = {}
    dsp_nlp = benchmark_nlp
    dsp_nlp_identity_sha256 = attestation_before["nlp_model_identity"][
        "identity_sha256"
    ]

    def ch_dsp(tr, te):
        Etr = dsp_matrix(
            tr,
            nlp=dsp_nlp,
            nlp_identity_sha256=dsp_nlp_identity_sha256,
            cache=dsp_cache,
        )
        Ete = dsp_matrix(
            te,
            nlp=dsp_nlp,
            nlp_identity_sha256=dsp_nlp_identity_sha256,
            cache=dsp_cache,
        )
        sc = StandardScaler().fit(Etr); return sc.transform(Etr), sc.transform(Ete)

    CHANNELS = make_channels(runtime_cfg)
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
    cache_workspace.cleanup()
    if cache_root.exists():
        raise RuntimeError("ephemeral benchmark cache cleanup failed")
    attestation_after = benchmark_attestation(
        cfg,
        nlp_identity=snapshot_benchmark_nlp_identity(cfg, benchmark_nlp),
    )
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
