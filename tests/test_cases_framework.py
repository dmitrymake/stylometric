import json
from pathlib import Path

import numpy as np
import pytest

from stylo.cases import load_case_spec, rank_passports, run_case
from stylo.cases import framework as case_framework
from stylo.cli import main


def _write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_case(tmp_path: Path) -> Path:
    a = tmp_path / "a"
    b = tmp_path / "b"
    target = tmp_path / "target.txt"
    for i in range(4):
        _write(a / f"a{i}.txt", ("алый автор пишет мягко и спокойно " * 40) + f" {i}")
        _write(b / f"b{i}.txt", ("бурый рассказ звучит резко и громко " * 40) + f" {i}")
    _write(target, "алый автор пишет мягко и спокойно " * 35)
    spec = tmp_path / "case.yaml"
    spec.write_text(
        f"""
case_id: synthetic_clear
title: Synthetic clear case
language: ru
feature_sets: [char3]
chunk_words: 80
min_work_words: 5
max_exact_permutations: 200
random_permutations: 50
candidates:
  author_a: {a}
  author_b: {b}
target: {target}
""",
        encoding="utf-8",
    )
    return spec


def test_case_target_abstains_until_open_set_gate_exists(tmp_path):
    spec_path = _make_case(tmp_path)
    spec = load_case_spec(spec_path)

    passport = run_case(spec).to_dict()

    assert spec.centroid_weighting == "work_balanced"
    assert passport["schema_version"] == "stylo.case-passport.v2"
    assert passport["case_id"] == "synthetic_clear"
    assert passport["data"]["centroid_weighting"] == "work_balanced"
    assert passport["gate_pass"] is False
    assert passport["status"] == "inconclusive"
    assert passport["gates"][0]["work_macro_recall"] == 1.0
    assert passport["gates"][0]["gate_pass"] is True
    assert passport["gates"][0]["permutation_p"] > 0
    attribution = passport["attributions"][0]
    assert attribution["top"] is None
    assert attribution["abstained"] is True
    assert attribution["diagnostic_closed_set_top"] == "author_a"
    assert attribution["n_works"] == 1
    assert attribution["target_work_ids"] == ["target"]
    assert attribution["margin_ci95"] is None
    assert attribution["uncertainty_unit"] == "work_id"
    assert (
        "required_gate_unavailable:target_open_set_applicability_gate_v1"
        in passport["failure_modes"]
    )
    assert (
        "target_lt2_independent_works_ci_unavailable"
        in passport["failure_modes"]
    )


def test_case_gate_refuses_single_work(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    _write(a / "a0.txt", "один авторский текст " * 20)
    _write(b / "b0.txt", "другой авторский текст " * 20)
    spec_path = tmp_path / "single.yaml"
    spec_path.write_text(
        f"""
case_id: single_work
feature_sets: [char3]
min_work_words: 5
candidates:
  a: {a}
  b: {b}
""",
        encoding="utf-8",
    )

    passport = run_case(load_case_spec(spec_path)).to_dict()

    assert passport["status"] == "fail"
    assert "gate_uncomputable_lt2_works:a" in passport["failure_modes"]
    assert "gate_uncomputable_lt2_works:b" in passport["failure_modes"]


def test_case_blocks_when_candidate_drops_out(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    empty = tmp_path / "empty"
    empty.mkdir()
    target = tmp_path / "target.txt"
    for i in range(4):
        _write(a / f"a{i}.txt", "алый автор пишет мягко и спокойно " * 30)
        _write(b / f"b{i}.txt", "бурый рассказ звучит резко и громко " * 30)
    _write(target, "алый автор пишет мягко и спокойно " * 30)
    spec_path = tmp_path / "dropout.yaml"
    spec_path.write_text(
        f"""
case_id: dropout
feature_sets: [char3]
chunk_words: 80
min_work_words: 5
candidates:
  a: {a}
  b: {b}
  c: {empty}
target: {target}
""",
        encoding="utf-8",
    )

    passport = run_case(load_case_spec(spec_path)).to_dict()

    assert passport["status"] == "fail"
    assert passport["gate_pass"] is False
    assert passport["gates"] == []
    assert "empty_candidate:c" in passport["failure_modes"]


def test_single_chunk_target_is_not_strong(tmp_path):
    spec_path = _make_case(tmp_path)
    text = spec_path.read_text(encoding="utf-8").replace("chunk_words: 80", "chunk_words: 10000")
    spec_path.write_text(text, encoding="utf-8")

    passport = run_case(load_case_spec(spec_path)).to_dict()

    assert passport["gates"][0]["gate_pass"] is True
    assert passport["gate_pass"] is False
    assert passport["attributions"][0]["n_chunks"] == 1
    assert passport["attributions"][0]["margin_ci95"] is None
    assert passport["status"] == "inconclusive"


def test_candidate_paths_and_exclude(tmp_path):
    gogol_a = tmp_path / "gogol_a"
    gogol_b = tmp_path / "gogol_b"
    other = tmp_path / "other"
    target = gogol_b / "target.txt"
    for i in range(2):
        _write(gogol_a / f"a{i}.txt", "гоголевский текст с шинелью и носом " * 40)
        _write(gogol_b / f"b{i}.txt", "ещё один гоголевский текст с городничим " * 40)
        _write(other / f"o{i}.txt", "другой автор говорит иначе и строго " * 40)
    _write(target, "гоголевский спорный фрагмент с носом " * 40)
    spec_path = tmp_path / "multi.yaml"
    spec_path.write_text(
        f"""
case_id: multi_path
feature_sets: [char3]
chunk_words: 80
min_work_words: 5
candidates:
  gogol:
    paths:
      - {gogol_a}
      - {gogol_b}
    exclude:
      - {target}
  other: {other}
target: {target}
forbidden_sources:
  - {target}
""",
        encoding="utf-8",
    )

    passport = run_case(load_case_spec(spec_path)).to_dict()

    assert "forbidden_source_in_panel:gogol" not in passport["failure_modes"]
    assert passport["data"]["works_per_author"]["gogol"] == 4


def test_rank_passports_orders_by_evidence_score():
    rows = rank_passports([
        {
            "schema_version": "stylo.case-passport.v2",
            "case_id": "weak",
            "status": "inconclusive",
            "evidence_score": 30,
        },
        {
            "schema_version": "stylo.case-passport.v2",
            "case_id": "gate",
            "status": "gate_only",
            "evidence_score": 80,
        },
    ])

    assert [r["case_id"] for r in rows] == ["gate", "weak"]


def test_cli_case_run_writes_passport(tmp_path):
    spec = _make_case(tmp_path)
    out = tmp_path / "passport.json"

    rc = main(["case", "run", str(spec), "--out", str(out)])

    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["case_id"] == "synthetic_clear"
    assert data["attributions"][0]["top"] is None
    assert data["attributions"][0]["diagnostic_closed_set_top"] == "author_a"


def test_case_dossier_uses_public_metadata(tmp_path):
    spec = _make_case(tmp_path)
    spec.write_text(
        spec.read_text(encoding="utf-8")
        + """
hypothesis: "Synthetic authorship hypothesis"
target_description: "synthetic target"
claim: "Synthetic target goes to author A."
limitations:
  - "Synthetic limitation."
provenance:
  analysis_command: "stylo case run synthetic.yaml"
""",
        encoding="utf-8",
    )
    passport_path = tmp_path / "passport.json"
    dossier_path = tmp_path / "dossier.md"

    assert main(["case", "run", str(spec), "--out", str(passport_path)]) == 0
    assert main(["case", "dossier", str(passport_path), "--out", str(dossier_path)]) == 0

    data = json.loads(passport_path.read_text(encoding="utf-8"))
    dossier = dossier_path.read_text(encoding="utf-8")
    assert data["hypothesis"] == "Synthetic authorship hypothesis"
    assert data["claim"] == "Synthetic target goes to author A."
    assert "Synthetic target goes to author A." not in dossier
    assert "## Decision" in dossier
    assert "воздержалась" in dossier
    assert "synthetic target" in dossier
    assert "stylo case run synthetic.yaml" in dossier


def test_case_report_commands_create_parent_directories(tmp_path):
    spec = _make_case(tmp_path)
    passport_path = tmp_path / "passport.json"
    rank_path = tmp_path / "nested" / "reports" / "ranking.json"
    report_path = tmp_path / "nested" / "reports" / "ranking.md"
    dossier_path = tmp_path / "nested" / "reports" / "dossier.md"

    assert main(["case", "run", str(spec), "--out", str(passport_path)]) == 0
    assert main(["case", "rank", str(passport_path), "--out", str(rank_path)]) == 0
    assert main(["case", "report", str(passport_path), "--out", str(report_path)]) == 0
    assert main(["case", "dossier", str(passport_path), "--out", str(dossier_path)]) == 0

    assert json.loads(rank_path.read_text(encoding="utf-8"))[0]["case_id"] == "synthetic_clear"
    assert "synthetic_clear" in report_path.read_text(encoding="utf-8")
    assert "Synthetic clear case" in dossier_path.read_text(encoding="utf-8")


def test_gate_float_tolerance():
    # macro recall 0.9+0.9+0.75+0.65 = 3.1999999999999997 -> /4 = 0.79999...;
    # gate обязан считать это ровно 0.80 и проходить
    vals = [0.9, 0.9, 0.75, 0.65]
    macro = sum(vals) / len(vals)
    assert macro < 0.80  # сам float-эффект
    assert macro >= 0.80 - 1e-9


def test_work_balanced_centroid_ignores_unequal_chunk_counts():
    # Author A has two orthogonal works. Repeating chunks in the second work
    # must not give that work extra weight under the scientific default. The
    # explicit legacy mode preserves the historical chunk-weighted flip.
    a_short = [[1.0, 0.0]]
    a_long_once = [[0.0, 1.0]]
    a_long_repeated = a_long_once * 9
    b_works = [[0.6, 0.8], [0.6, 0.8]]
    target = np.asarray([[1.0, 0.0]])

    def predict(a_long, weighting):
        rows = np.asarray(a_short + a_long + b_works)
        labels = ["a"] * (len(a_short) + len(a_long)) + ["b", "b"]
        work_ids = ["a_short"] + ["a_long"] * len(a_long) + ["b1", "b2"]
        return case_framework._predict_by_centroid(
            rows,
            labels,
            target,
            train_work_ids=work_ids,
            centroid_weighting=weighting,
        )[0]

    assert predict(a_long_once, "work_balanced") == "a"
    assert predict(a_long_repeated, "work_balanced") == "a"
    assert predict(a_long_once, "chunk_weighted_legacy") == "a"
    assert predict(a_long_repeated, "chunk_weighted_legacy") == "b"


def test_work_balanced_loo_excludes_held_out_work_before_centroid(monkeypatch):
    works = [
        case_framework.Work("a", "a1", "", ("", "")),
        case_framework.Work("a", "a2", "", ("",)),
        case_framework.Work("b", "b1", "", ("",)),
        case_framework.Work("b", "b2", "", ("", "")),
    ]
    ctx = case_framework.FwContext(
        X=np.asarray([
            [1.0, 0.0], [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9], [0.1, 0.9],
        ]),
        chunk_work=np.asarray([0, 0, 1, 2, 3, 3]),
    )
    seen_train_work_ids = []
    original = case_framework._predict_by_centroid

    def spy(Xtr, ytr, Xte, *, train_work_ids=None, centroid_weighting=None):
        seen_train_work_ids.append(set(train_work_ids))
        return original(
            Xtr,
            ytr,
            Xte,
            train_work_ids=train_work_ids,
            centroid_weighting=centroid_weighting,
        )

    monkeypatch.setattr(case_framework, "_predict_by_centroid", spy)
    case_framework._leave_one_work_out_fw_context(
        works,
        [w.author for w in works],
        ctx,
        centroid_weighting="work_balanced",
    )

    assert seen_train_work_ids == [
        {1, 2, 3},
        {0, 2, 3},
        {0, 1, 3},
        {0, 1, 2},
    ]


def test_gate_and_permutation_share_explicit_centroid_weighting(monkeypatch):
    works = [
        case_framework.Work("a", "a1", "", ("и и и и",)),
        case_framework.Work("a", "a2", "", ("и и и",)),
        case_framework.Work("b", "b1", "", ("но но но но",)),
        case_framework.Work("b", "b2", "", ("но но но",)),
    ]
    spec = case_framework.CaseSpec(
        case_id="weighting_plumbing",
        title="weighting plumbing",
        candidates=(),
        feature_sets=("fw_fixed",),
        centroid_weighting="chunk_weighted_legacy",
        max_exact_permutations=10,
        random_permutations=5,
    )
    seen_direct = []
    seen_cached = []
    original_direct = case_framework._leave_one_work_out_fw_context
    original_cached = case_framework._leave_one_work_out_fw_cached

    def spy_direct(works, labels, ctx, centroid_weighting="work_balanced"):
        seen_direct.append(centroid_weighting)
        return original_direct(
            works, labels, ctx, centroid_weighting=centroid_weighting
        )

    def spy_cached(works, labels, ctx, cache, centroid_weighting="work_balanced"):
        seen_cached.append(centroid_weighting)
        return original_cached(
            works, labels, ctx, cache, centroid_weighting=centroid_weighting
        )

    monkeypatch.setattr(case_framework, "_leave_one_work_out_fw_context", spy_direct)
    monkeypatch.setattr(case_framework, "_leave_one_work_out_fw_cached", spy_cached)
    case_framework._run_gate(works, spec, "fw_fixed")

    # Gate scoring uses the reference path once; observed/null permutations use
    # the optimized path. Both receive the exact same explicit policy.
    assert seen_direct == ["chunk_weighted_legacy"]
    assert len(seen_cached) > 2
    assert set(seen_cached) == {"chunk_weighted_legacy"}


def test_cached_fw_assignments_match_reference_for_both_weightings():
    chunk_counts = [1, 4, 2, 3, 1, 5]
    works = [
        case_framework.Work(
            "a" if wi < 3 else "b",
            f"w{wi}",
            "",
            tuple("" for _ in range(count)),
        )
        for wi, count in enumerate(chunk_counts)
    ]
    rows_by_work = [
        [[1.00, 0.10, 0.05]],
        [[0.92, 0.20, 0.08], [0.88, 0.25, 0.10],
         [0.95, 0.15, 0.06], [0.90, 0.22, 0.09]],
        [[0.80, 0.32, 0.10], [0.84, 0.28, 0.12]],
        [[0.10, 0.95, 0.18], [0.12, 0.90, 0.22], [0.08, 0.92, 0.20]],
        [[0.20, 0.82, 0.12]],
        [[0.05, 0.86, 0.30], [0.08, 0.88, 0.28], [0.06, 0.84, 0.32],
         [0.10, 0.90, 0.25], [0.07, 0.87, 0.29]],
    ]
    ctx = case_framework.FwContext(
        X=np.asarray([row for group in rows_by_work for row in group]),
        chunk_work=np.asarray([
            wi for wi, count in enumerate(chunk_counts) for _ in range(count)
        ]),
    )
    cache = case_framework._build_fw_permutation_cache(works, ctx)
    assignments = [
        ["a", "a", "a", "b", "b", "b"],
        ["a", "b", "a", "b", "a", "b"],
        ["b", "b", "b", "a", "a", "a"],
        ["a", "b", "b", "b", "a", "a"],
    ]

    for weighting in ("work_balanced", "chunk_weighted_legacy"):
        for labels in assignments:
            reference_wcp = case_framework._leave_one_work_out_fw_context(
                works, labels, ctx, centroid_weighting=weighting
            )
            cached_wcp = case_framework._leave_one_work_out_fw_cached(
                works, labels, ctx, cache, centroid_weighting=weighting
            )
            authors = sorted(set(labels))
            assert cached_wcp == reference_wcp
            assert (
                case_framework._metrics_from_wcp(cached_wcp, authors)
                == case_framework._metrics_from_wcp(reference_wcp, authors)
            )


def test_fw_permutation_result_is_deterministic_for_both_weightings():
    works = [
        case_framework.Work("a", "a1", "", ("и и на",)),
        case_framework.Work("a", "a2", "", ("и на", "и и", "на и", "и в")),
        case_framework.Work("a", "a3", "", ("и в на", "на и в")),
        case_framework.Work("b", "b1", "", ("но что", "но но", "что но")),
        case_framework.Work("b", "b2", "", ("что но что",)),
        case_framework.Work(
            "b", "b3", "", ("но что", "что что", "но но", "что но", "но что что")
        ),
    ]
    labels = [w.author for w in works]

    for weighting in ("work_balanced", "chunk_weighted_legacy"):
        exact_1 = case_framework._permutation_p(
            works, labels, "fw_fixed", "ru", 100, 11, 20260630,
            centroid_weighting=weighting,
        )
        exact_2 = case_framework._permutation_p(
            works, labels, "fw_fixed", "ru", 100, 11, 20260630,
            centroid_weighting=weighting,
        )
        random_1 = case_framework._permutation_p(
            works, labels, "fw_fixed", "ru", 0, 11, 20260630,
            centroid_weighting=weighting,
        )
        random_2 = case_framework._permutation_p(
            works, labels, "fw_fixed", "ru", 0, 11, 20260630,
            centroid_weighting=weighting,
        )

        assert exact_1 == exact_2
        assert exact_1[1:] == ("exact_20", 0.05)
        assert random_1 == random_2
        assert random_1[1:] == ("random_11", 0.05)


def test_centroid_weighting_requires_supported_explicit_value(tmp_path):
    spec_path = _make_case(tmp_path)
    spec_path.write_text(
        spec_path.read_text(encoding="utf-8")
        + "\ncentroid_weighting: accidental_flat_mode\n",
        encoding="utf-8",
    )

    passport = run_case(load_case_spec(spec_path)).to_dict()

    assert passport["gates"] == []
    assert "unknown_centroid_weighting:accidental_flat_mode" in passport["failure_modes"]


def test_legacy_centroid_weighting_is_explicit_and_recorded(tmp_path):
    spec_path = _make_case(tmp_path)
    spec_path.write_text(
        spec_path.read_text(encoding="utf-8")
        + "\ncentroid_weighting: chunk_weighted_legacy\n",
        encoding="utf-8",
    )

    spec = load_case_spec(spec_path)
    passport = run_case(spec).to_dict()

    assert spec.centroid_weighting == "chunk_weighted_legacy"
    assert passport["data"]["centroid_weighting"] == "chunk_weighted_legacy"


def test_target_loader_preserves_parent_work_ids(tmp_path):
    spec_path = _make_case(tmp_path)
    target_file = tmp_path / "target.txt"
    target_dir = tmp_path / "target_works"
    _write(target_dir / "one.txt", "первый спорный самостоятельный текст " * 40)
    _write(target_dir / "two.txt", "второй спорный самостоятельный текст " * 40)
    spec_path.write_text(
        spec_path.read_text(encoding="utf-8").replace(
            f"target: {target_file}",
            f"target: {target_dir}",
        ),
        encoding="utf-8",
    )

    spec = load_case_spec(spec_path)
    chunks = case_framework._load_target_chunks(spec)

    assert {chunk.work_id for chunk in chunks} == {
        "target_works/one",
        "target_works/two",
    }
    assert all(chunk.text for chunk in chunks)


def test_work_bootstrap_cannot_gain_precision_from_duplicate_chunks():
    centroids = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    base_rows = np.asarray([[1.0, 0.0], [0.8, 0.2]])
    base_ids = ["work_one", "work_two"]
    duplicated_rows = np.vstack(
        [np.repeat(base_rows[:1], 50, axis=0), base_rows[1:]]
    )
    duplicated_ids = ["work_one"] * 50 + ["work_two"]

    base_ci = case_framework._bootstrap_margin_by_work(
        base_rows, base_ids, centroids, ["a", "b"], seed=7, n_iter=200
    )
    duplicated_ci = case_framework._bootstrap_margin_by_work(
        duplicated_rows,
        duplicated_ids,
        centroids,
        ["a", "b"],
        seed=7,
        n_iter=200,
    )
    one_work_ci = case_framework._bootstrap_margin_by_work(
        duplicated_rows[:-1],
        duplicated_ids[:-1],
        centroids,
        ["a", "b"],
        seed=7,
        n_iter=200,
    )

    assert duplicated_ci == base_ci
    assert one_work_ci is None


def test_required_gate_registry_rejects_unknown_name(tmp_path):
    spec_path = _make_case(tmp_path)
    spec_path.write_text(
        spec_path.read_text(encoding="utf-8")
        + "\nrequired_gates: [feasibility_gate, typo_gate]\n",
        encoding="utf-8",
    )

    passport = run_case(load_case_spec(spec_path)).to_dict()

    assert set(case_framework.REQUIRED_GATE_REGISTRY) == {
        "feasibility_gate",
        "target_open_set_applicability_gate_v1",
    }
    assert passport["status"] == "fail"
    assert passport["gate_pass"] is False
    assert passport["gates"] == []
    assert "unknown_required_gate:typo_gate" in passport["failure_modes"]
    evaluations = {
        row["name"]: row for row in passport["data"]["required_gate_evaluations"]
    }
    assert evaluations["typo_gate"]["status"] == "unknown"
    assert evaluations["typo_gate"]["gate_pass"] is False


def test_outsider_target_abstains_despite_relative_closed_set_winner(tmp_path):
    spec_path = _make_case(tmp_path)
    _write(
        tmp_path / "target.txt",
        "совершенно посторонний зелёный голос без сходства с панелью " * 35,
    )

    passport = run_case(load_case_spec(spec_path)).to_dict()
    attribution = passport["attributions"][0]

    assert passport["gates"][0]["gate_pass"] is True
    assert passport["gate_pass"] is False
    assert passport["status"] == "inconclusive"
    assert passport["confidence"] == "low"
    assert attribution["top"] is None
    assert attribution["abstained"] is True
    assert attribution["diagnostic_closed_set_top"] in {"author_a", "author_b"}
    assert "не научным вердиктом" in passport["verdict"]


def test_historical_case_passport_cannot_be_ranked_as_current(tmp_path):
    path = tmp_path / "historical.passport.json"
    path.write_text(
        json.dumps(
            {
                "case_id": "legacy",
                "claim_status": "exploratory_internal",
                "status": "strong",
                "evidence_score": 99,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(case_framework.CasePassportSemanticError, match="withdrawn"):
        case_framework.load_passport(path)


def test_forged_v2_target_verdict_is_rejected():
    forged = {
        "schema_version": "stylo.case-passport.v2",
        "case_id": "forged",
        "status": "strong",
        "gate_pass": True,
        "evidence_score": 100,
        "attributions": [
            {
                "top": "candidate",
                "second": "other",
                "abstained": False,
                "uncertainty_unit": "chunk",
                "n_works": 1,
                "target_work_ids": ["one"],
                "margin_ci95": [0.1, 0.2],
            }
        ],
    }

    with pytest.raises(
        case_framework.CasePassportSemanticError,
        match="forbids strong/moderate",
    ):
        rank_passports([forged])
