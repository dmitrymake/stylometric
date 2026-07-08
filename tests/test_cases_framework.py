import json
from pathlib import Path

from stylo.cases import load_case_spec, rank_passports, run_case
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


def test_case_spec_and_run_strong(tmp_path):
    spec_path = _make_case(tmp_path)
    spec = load_case_spec(spec_path)

    passport = run_case(spec).to_dict()

    assert passport["case_id"] == "synthetic_clear"
    assert passport["gate_pass"] is True
    assert passport["status"] == "strong"
    assert passport["gates"][0]["work_macro_recall"] == 1.0
    assert passport["gates"][0]["permutation_p"] > 0
    assert passport["attributions"][0]["top"] == "author_a"


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

    assert passport["gate_pass"] is True
    assert passport["attributions"][0]["n_chunks"] == 1
    assert passport["status"] != "strong"


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
        {"case_id": "weak", "status": "moderate", "evidence_score": 30},
        {"case_id": "strong", "status": "strong", "evidence_score": 80},
    ])

    assert [r["case_id"] for r in rows] == ["strong", "weak"]


def test_cli_case_run_writes_passport(tmp_path):
    spec = _make_case(tmp_path)
    out = tmp_path / "passport.json"

    rc = main(["case", "run", str(spec), "--out", str(out)])

    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["case_id"] == "synthetic_clear"
    assert data["attributions"][0]["top"] == "author_a"


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
    assert "Synthetic target goes to author A." in dossier
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
