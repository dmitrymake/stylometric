"""B4-B1: the frozen external A0/A4 goldens are a strict, read-only golden contract.

Captured from the committed `f1b8e165` (B3) by `scripts/capture_b4_goldens.py` BEFORE any B4-B
estimator-axis refactor. This test only READS the fixture (never regenerates it, never fits an
estimator) and STRICTLY validates EVERY field: full-SHA pin; config/lock/tool SHAs against the live
authoritative bytes; a per-kind schema for every fit-trace entry; per-model/corner state schemas with
exact keys, non-empty typed feature order and `{shape,dtype,64-hex digest}` arrays for A0 AND A4; the
real inner+full fit weights (18 SVC for the stack); both calibration paths; and mutation-negative
tests that reproduce the previously-accepted false-greens.
"""
from __future__ import annotations

import copy
import hashlib
import numbers
import pathlib
from collections import Counter

import pytest

from stylo import jsonio

_STYLO_BLOCKS = {f"block:{b}": 1 for b in ("CharNgramBlock", "DependencyBlock", "FunctionWordBlock",
                                          "LengthDistBlock", "MorphologyBlock", "PosNgramBlock",
                                          "PunctNgramBlock", "SyntaxBlock")}
_STACK_BLOCKS = {"block:DependencyBlock": 6, "block:FunctionWordBlock": 3, "block:MorphologyBlock": 3,
                 "block:PosNgramBlock": 3, "block:SyntaxBlock": 3}
# exact per-(model, corner) fit-trace kind→multiplicity signature (frozen; empty/truncated traces fail)
TRACE_SIG = {
    ("delta_cos:500", "A0"): {"count_vec": 2},
    ("delta_cos:500", "A4"): {"count_vec": 1, "work_vec": 1},
    ("delta_cos:12", "A0"): {"count_vec": 2},
    ("delta_cos:12", "A4"): {"count_vec": 1, "work_vec": 1},
    ("char_cos", "A0"): {"count_vec": 1, "tfidf": 1},
    ("char_cos", "A4"): {"count_vec": 1, "work_vec": 1},
    ("bow_lr", "A0"): {"count_vec": 1, "lr": 1},
    ("bow_lr", "A4"): {"count_vec": 1, "lr": 1, "work_vec": 1},
    ("stylo", "A0"): {**_STYLO_BLOCKS, "count_vec": 5, "lr": 1, "tfidf": 3},
    ("stylo", "A4"): {**_STYLO_BLOCKS, "count_vec": 4, "lr": 1, "work_vec": 4},
    ("stylo_stack", "A0"): {**_STACK_BLOCKS, "count_vec": 9, "lr": 9, "svc": 18, "tfidf": 9},
    ("stylo_stack", "A4"): {**_STACK_BLOCKS, "count_vec": 6, "lr": 21, "svc": 18, "tfidf": 6, "work_vec": 6},
    ("stylo_stack__faildisabled", "A0"): {**_STACK_BLOCKS, "count_vec": 9, "lr": 9, "svc": 18, "tfidf": 9},
    # fail-closed calibration → equal ensemble, NO meta-CV/meta-LR fits (no "lr" entries)
    ("stylo_stack__faildisabled", "A4"): {**_STACK_BLOCKS, "count_vec": 6, "svc": 18, "tfidf": 6, "work_vec": 6},
}

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIX_DIR = ROOT / "tests" / "fixtures"
FIXTURE = FIX_DIR / "b4_goldens_v1.json"
INVENTORY = FIX_DIR / "b4_goldens_v1.SHA256SUMS"

FIXTURE_SHA256 = "c66e63e7e8af36b03b6aaa28f3d8cb23b71ef8e7a433ddd0c85322463b993a25"
SOURCE_COMMIT = "f1b8e165fad4a1ae7ce30c0a613ffcf0deaa7b3d"
CLASS_ORDER = [0, 1]
MODEL_PANEL = {"delta_cos:500": "P1", "delta_cos:12": "P1", "char_cos": "P1",
               "bow_lr": "P1", "stylo": "P1", "stylo_stack": "P1", "stylo_stack__faildisabled": "P2"}
CW_ALLOWED = {None, "balanced", "dict"}


def _fixture() -> dict:
    return jsonio.load_strict(FIXTURE)


def _exact_int(x) -> bool:
    return isinstance(x, numbers.Integral) and not isinstance(x, bool)


def _is_digest(s) -> bool:
    return isinstance(s, str) and len(s) == 64 and all(c in "0123456789abcdef" for c in s)


def _v_arr(a) -> None:
    if not (isinstance(a, dict) and set(a) == {"shape", "dtype", "digest"}):
        raise ValueError("array record must be exactly {shape,dtype,digest}")
    if a["dtype"] != "float64" or not (isinstance(a["shape"], list) and all(_exact_int(v) for v in a["shape"])):
        raise ValueError("bad array shape/dtype")
    if not _is_digest(a["digest"]):
        raise ValueError("array digest must be a 64-hex sha256")


def _v_feature_names(fn) -> None:
    if not (isinstance(fn, list) and fn and all(isinstance(x, str) and x for x in fn)):
        raise ValueError("feature_names must be a non-empty list of non-empty strings")


def _v_trace_entry(t) -> None:
    if not (isinstance(t, dict) and isinstance(t.get("kind"), str)):
        raise ValueError("trace entry needs a string kind")
    k = t["kind"]
    if not (t.get("n_rows") is None or _exact_int(t["n_rows"])):
        raise ValueError("n_rows must be int or null")
    if k in ("lr", "svc"):
        allowed, required = {"kind", "n_rows", "class_weight", "coef_", "sample_weight"}, {"kind", "n_rows", "class_weight", "coef_"}
        if not (required <= set(t) <= allowed):
            raise ValueError(f"{k} entry keys {set(t)}")
        if t["class_weight"] not in CW_ALLOWED:
            raise ValueError("bad class_weight")
        _v_arr(t["coef_"])
        if "sample_weight" in t:
            _v_arr(t["sample_weight"])
    elif k == "tfidf":
        if set(t) != {"kind", "n_rows", "idf_"}:
            raise ValueError("tfidf entry keys")
        _v_arr(t["idf_"])
    elif k == "count_vec":
        if set(t) != {"kind", "n_rows", "vocab_digest"} or not _is_digest(t["vocab_digest"]):
            raise ValueError("count_vec entry")
    elif k == "work_vec":
        if not ({"kind", "n_rows", "vocab_digest"} <= set(t) <= {"kind", "n_rows", "vocab_digest", "idf_"}):
            raise ValueError("work_vec entry keys")
        if not _is_digest(t["vocab_digest"]):
            raise ValueError("work_vec vocab digest")
        if "idf_" in t:
            _v_arr(t["idf_"])
    elif k.startswith("block:"):
        if set(t) != {"kind", "n_rows", "vocab_digest"} or not _is_digest(t["vocab_digest"]):
            raise ValueError("block entry")
    else:
        raise ValueError(f"unknown trace kind {k!r}")


def _v_proba(proba, classes, n) -> None:
    if classes != CLASS_ORDER or not all(type(c) is int for c in classes):
        raise ValueError("classes must be exactly [0,1] as plain int (not bool)")
    if not (isinstance(proba, list) and len(proba) == n and all(isinstance(r, list) for r in proba)):
        raise ValueError("proba must be a 2-D list of exactly n_chunks rows")
    for row in proba:
        if len(row) != len(classes):
            raise ValueError("proba width")
        for p in row:
            if isinstance(p, bool) or not isinstance(p, numbers.Real) or not (0.0 <= float(p) <= 1.0):
                raise ValueError("proba entries must be real non-bool in [0,1]")
        if abs(sum(row) - 1.0) > 1e-9:
            raise ValueError("proba row sum != 1")


def _v_corner(name, corner, g, n) -> None:
    spec = "stylo_stack" if name.startswith("stylo_stack") else name
    ra = g.get("requested_axes")
    on = corner == "A4"
    if not (isinstance(ra, dict) and ra == {"W": on, "F": on, "R": on}
            and all(type(v) is bool for v in ra.values())):
        raise ValueError("requested_axes must be exact bools (A0=F/F/F, A4=T/T/T)")
    if Counter(t["kind"] for t in g["fit_trace"]) != TRACE_SIG[(name, corner)]:
        raise ValueError(f"{name} {corner}: fit_trace signature mismatch (empty/truncated/extra)")
    _v_proba(g["proba"], g["classes_"], n)
    if spec.startswith("delta"):
        if set(g) != {"requested_axes", "fit_trace", "classes_", "proba", "group_weighting_",
                      "feature_names", "mean_", "std_", "centroids_"}:
            raise ValueError("delta corner keys")
        _v_feature_names(g["feature_names"])
        for a in ("mean_", "std_", "centroids_"):
            _v_arr(g[a])
    elif spec == "char_cos":
        if set(g) != {"requested_axes", "fit_trace", "classes_", "proba", "group_weighting_",
                      "feature_names", "centroids"}:
            raise ValueError("char corner keys")
        _v_feature_names(g["feature_names"]); _v_arr(g["centroids"])
    elif spec == "stylo_stack":
        if set(g) != {"requested_axes", "fit_trace", "classes_", "proba", "passport", "mode_", "meta_is_none"}:
            raise ValueError("stack corner keys")
        if not isinstance(g["passport"], dict):
            raise ValueError("passport")
    else:  # stylo, bow_lr
        req = {"requested_axes", "fit_trace", "classes_", "proba", "vectorizer_feature_names_digest", "block_feature_digests"}
        if spec == "bow_lr":
            req = {"requested_axes", "fit_trace", "classes_", "proba"}
        if set(g) != req:
            raise ValueError(f"{spec} corner keys {set(g)}")
        if spec == "stylo":
            if not _is_digest(g["vectorizer_feature_names_digest"]):
                raise ValueError("stylo vectorizer digest")
            if not (g["block_feature_digests"] and all(_is_digest(v) for v in g["block_feature_digests"].values())):
                raise ValueError("stylo block digests")
    for t in g["fit_trace"]:
        _v_trace_entry(t)


# ── the whole fixture validates strictly ────────────────────────────────────────
def test_full_fixture_is_strictly_valid():
    d = _fixture()
    assert set(d["models"]) == set(MODEL_PANEL)
    for name, m in d["models"].items():
        assert m["panel"] == MODEL_PANEL[name] and set(m) == {"panel", "A0", "A4"}
        n = d["panels"][m["panel"]]["n_chunks"]
        for corner in ("A0", "A4"):
            _v_corner(name, corner, m[corner], n)


# ── pin, inventory, provenance, environment (authoritative-byte verification) ────
def test_fixture_matches_pinned_full_sha():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256


def test_inventory_is_exactly_one_valid_line():
    lines = INVENTORY.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    digest, name = lines[0].split()
    assert name == FIXTURE.name and _is_digest(digest) == True
    assert digest == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()


def test_provenance_contract_and_authoritative_hashes():
    d = _fixture()
    assert d["source_commit"] == SOURCE_COMMIT and d["clean_tree"] is True
    assert d["numeric_contract"] == {"kind": "quantized_exact", "round_decimals": 12} and "tolerances" not in d
    e = d["environment"]
    assert e["spacy_model"]["name"] and e["spacy_model"]["version"] and e["threadpool"]
    assert e["pythonhashseed"] and e["omp_num_threads"]
    # recorded hashes must equal the LIVE authoritative bytes
    assert e["config_sha256"] == hashlib.sha256((ROOT / "configs" / "default.yaml").read_bytes()).hexdigest()
    assert e["lock_sha256"] == hashlib.sha256((ROOT / "requirements.lock").read_bytes()).hexdigest()
    assert e["capture_tool_sha256"] == hashlib.sha256((ROOT / "scripts" / "capture_b4_goldens.py").read_bytes()).hexdigest()


def test_panels_self_contained():
    d = _fixture()
    for pk in ("P1", "P2"):
        p = d["panels"][pk]
        assert len(p["texts"]) == p["n_chunks"] == len(p["y"]) == len(p["groups"]) > 0
        assert p["authors"] == sorted({g.split("/", 1)[0] for g in p["groups"]}) and all("/" not in a for a in p["authors"])
        assert hashlib.sha256(("␟".join(p["texts"]) + "␞" + ",".join(map(str, p["y"])) + "␞" + ",".join(map(str, p["groups"]))).encode()).hexdigest() == p["panel_sha256"]


# ── the substantive golden properties ──────────────────────────────────────────
def test_real_18_svc_and_weight_routing():
    d = _fixture()
    sk4 = d["models"]["stylo_stack"]["A4"]["fit_trace"]
    assert sum(t["kind"] == "svc" for t in sk4) == 18       # 12 inner + 6 full-train (instrumentation open thru predict)
    assert any("sample_weight" in t and t["kind"] == "svc" for t in sk4)   # work-supervised SVC
    for spec in ("stylo", "bow_lr"):
        a4 = [t for t in d["models"][spec]["A4"]["fit_trace"] if t["kind"] == "lr"]
        a0 = [t for t in d["models"][spec]["A0"]["fit_trace"] if t["kind"] == "lr"]
        assert a4[-1]["class_weight"] is None and "sample_weight" in a4[-1]
        assert a0[-1]["class_weight"] == "balanced" and all("sample_weight" not in t for t in a0)


def test_wb_idf_and_bow_vocab_and_channel_blocks_captured():
    d = _fixture()
    sk = d["models"]["stylo_stack"]["A4"]["fit_trace"]
    assert any(t["kind"] == "work_vec" and "idf_" in t for t in sk)        # WB IDF present
    assert any(t["kind"] == "count_vec" for t in d["models"]["bow_lr"]["A0"]["fit_trace"])  # bow A0 vocab
    blocks = {t["kind"] for t in sk if t["kind"].startswith("block:")}
    assert {"block:DependencyBlock", "block:MorphologyBlock"} <= blocks    # ephemeral channel vocabs


def test_A0_A4_distinct_and_delta_F_axis():
    d = _fixture()
    for name, m in d["models"].items():
        assert m["A0"]["proba"] != m["A4"]["proba"], name
    for spec in ("delta_cos:500", "delta_cos:12"):
        assert d["models"][spec]["A0"]["feature_names"] != d["models"][spec]["A4"]["feature_names"]


def test_stack_two_calibration_paths():
    d = _fixture()
    assert d["models"]["stylo_stack"]["A4"]["passport"]["calibration_disabled"] is False
    p2 = d["models"]["stylo_stack__faildisabled"]["A4"]
    assert p2["passport"]["calibration_disabled"] is True and p2["mode_"] == "equal" and p2["meta_is_none"]


# ── mutation-negative: the previously-accepted false-greens now fail ─────────────
def test_mutation_negatives_are_rejected():
    d = _fixture()
    n = d["panels"]["P1"]["n_chunks"]
    # malformed idf digest inside a char fit_trace work_vec entry
    bad = copy.deepcopy(d["models"]["char_cos"]["A4"])
    for t in bad["fit_trace"]:
        if t["kind"] == "work_vec" and "idf_" in t:
            t["idf_"]["digest"] = "xyz"
            break
    with pytest.raises(ValueError):
        _v_corner("char_cos", "A4", bad, n)
    # broken SVC coef digest in the stack
    bs = copy.deepcopy(d["models"]["stylo_stack"]["A4"])
    for t in bs["fit_trace"]:
        if t["kind"] == "svc":
            t["coef_"]["digest"] = "nothex" * 8
            break
    with pytest.raises(ValueError):
        _v_corner("stylo_stack", "A4", bs, n)
    # empty feature_names
    ec = copy.deepcopy(d["models"]["char_cos"]["A4"]); ec["feature_names"] = []
    with pytest.raises(ValueError):
        _v_corner("char_cos", "A4", ec, n)
    # empty fit_trace (false-green reproduced on char_cos.A0)
    et = copy.deepcopy(d["models"]["char_cos"]["A0"]); et["fit_trace"] = []
    with pytest.raises(ValueError):
        _v_corner("char_cos", "A0", et, n)
    # bool classes masquerading as [0,1], and non-exact requested_axes
    bc = copy.deepcopy(d["models"]["char_cos"]["A0"]); bc["classes_"] = [False, True]
    with pytest.raises(ValueError):
        _v_corner("char_cos", "A0", bc, n)
    ra = copy.deepcopy(d["models"]["stylo"]["A0"]); ra["requested_axes"] = {"W": True, "F": False, "R": False}
    with pytest.raises(ValueError):
        _v_corner("stylo", "A0", ra, n)
    # extra key / missing key
    xk = copy.deepcopy(d["models"]["stylo"]["A0"]); xk["surprise"] = 1
    with pytest.raises(ValueError):
        _v_corner("stylo", "A0", xk, n)
    mk = copy.deepcopy(d["models"]["delta_cos:500"]["A0"]); del mk["mean_"]
    with pytest.raises(ValueError):
        _v_corner("delta_cos:500", "A0", mk, n)
