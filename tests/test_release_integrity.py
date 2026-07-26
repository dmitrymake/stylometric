"""Release-integrity guarantees: strict JSON, claim-status vocabulary, and repository hygiene.

These tests encode the persistent release gate:
* every committed JSON artifact parses under a strict parser (no NaN/Infinity);
* the claim-status enum carries exactly the mandated evidence tiers;
* the publish gate blocks private corpus paths in the shipped tree while the
  local-repo audit only warns about private history on other refs.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

import pytest

from stylo import jsonio
from stylo.claims import BenchmarkRole, ClaimStatus, parse_claim_status
from stylo.release import hygiene

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# strict JSON writer / reader
# --------------------------------------------------------------------------- #
class TestStrictJson:
    def test_non_finite_floats_serialise_as_null(self):
        text = jsonio.dumps_strict({"a": float("nan"), "b": float("inf"), "c": float("-inf"), "d": 1.5})
        assert '"a": null' in text and '"b": null' in text and '"c": null' in text
        assert json.loads(text) == {"a": None, "b": None, "c": None, "d": 1.5}

    def test_dumps_strict_never_emits_nan_token(self):
        assert "NaN" not in jsonio.dumps_strict({"x": float("nan")})
        assert "Infinity" not in jsonio.dumps_strict({"x": float("inf")})

    def test_loads_strict_rejects_nan(self):
        with pytest.raises(jsonio.StrictJSONError):
            jsonio.loads_strict('{"x": NaN}')

    def test_loads_strict_rejects_infinity(self):
        with pytest.raises(jsonio.StrictJSONError):
            jsonio.loads_strict('{"x": Infinity}')

    def test_loads_strict_rejects_overflow_to_inf(self):
        # 1e999 has no NaN/Infinity token but parses to inf via parse_float
        with pytest.raises(jsonio.StrictJSONError):
            jsonio.loads_strict('{"x": 1e999}')
        with pytest.raises(jsonio.StrictJSONError):
            jsonio.loads_strict('{"x": -1e999}')

    def test_singleton_numpy_array_keeps_shape(self):
        np = pytest.importorskip("numpy")
        # .item() would collapse [2.0] to 2.0; shape must survive
        assert json.loads(jsonio.dumps_strict({"v": np.array([2.0])})) == {"v": [2.0]}
        assert json.loads(jsonio.dumps_strict({"v": np.array([[1.0, 2.0]])})) == {"v": [[1.0, 2.0]]}

    def test_longdouble_sanitised(self):
        np = pytest.importorskip("numpy")
        ld = getattr(np, "float128", np.longdouble)
        assert json.loads(jsonio.dumps_strict({"a": ld(2.5), "b": ld("nan")})) == {"a": 2.5, "b": None}

    def test_zero_d_object_array_recurses(self):
        np = pytest.importorskip("numpy")
        # a 0-d object array wrapping a dict must serialise, not raise
        wrapped = np.array({"a": float("nan"), "b": 2}, dtype=object)
        assert json.loads(jsonio.dumps_strict({"v": wrapped})) == {"v": {"a": None, "b": 2}}

    def test_object_array_wrapping_numpy_scalar_recurses(self):
        np = pytest.importorskip("numpy")
        # 0-d object array wrapping a numpy scalar: item() yields the scalar, which
        # must be re-sanitised (nan -> null), not passed through unserialisable.
        wrapped = np.array(np.float64("nan"), dtype=object)
        assert json.loads(jsonio.dumps_strict({"v": wrapped})) == {"v": None}

    def test_longdouble_overflow_raises_not_silent_null(self):
        np = pytest.importorskip("numpy")
        ld = getattr(np, "float128", np.longdouble)
        big = ld("1e400")  # finite in extended precision, exceeds float64
        if not (bool(np.isfinite(big)) and not math.isfinite(float(big))):
            pytest.skip("platform longdouble does not exceed float64 range")
        with pytest.raises(jsonio.StrictJSONError):
            jsonio.dumps_strict({"v": big})

    def test_loads_strict_rejects_duplicate_keys(self):
        with pytest.raises(jsonio.StrictJSONError):
            jsonio.loads_strict('{"x": 1, "x": 2}')

    def test_round_trip_finite(self):
        payload = {"m": [1, 2.5, "t", True, None], "n": {"k": 0.1}}
        assert jsonio.loads_strict(jsonio.dumps_strict(payload)) == payload

    def test_dump_strict_writes_atomically(self, tmp_path):
        target = tmp_path / "sub" / "out.json"
        jsonio.dump_strict({"v": float("nan"), "w": 3}, target)
        assert jsonio.load_strict(target) == {"v": None, "w": 3}
        assert target.read_text(encoding="utf-8").endswith("\n")

    def test_dump_strict_file_mode_respects_umask(self, tmp_path):
        import os
        import stat
        target = tmp_path / "out.json"
        jsonio.dump_strict({"a": 1}, target)
        umask = os.umask(0)
        os.umask(umask)
        assert stat.S_IMODE(target.stat().st_mode) == (0o666 & ~umask)

    def test_dump_strict_trailing_newline_optional(self, tmp_path):
        target = tmp_path / "out.json"
        jsonio.dump_strict({"a": 1}, target, trailing_newline=False)
        assert not target.read_bytes().endswith(b"\n")

    def test_numpy_scalars_sanitised(self):
        np = pytest.importorskip("numpy")
        text = jsonio.dumps_strict({"a": np.float64(2.0), "b": np.float64("nan"), "c": np.int64(4)})
        assert json.loads(text) == {"a": 2.0, "b": None, "c": 4}

    def test_sanitise_leaves_finite_untouched(self):
        assert jsonio.sanitise(1.25) == 1.25
        assert jsonio.sanitise(math.pi) == math.pi

    def test_colliding_keys_raise_not_drop(self):
        # int 1 and str "1" both coerce to "1"; dropping one silently is data loss
        with pytest.raises(jsonio.StrictJSONError):
            jsonio.dumps_strict({1: "a", "1": "b"})

    def test_set_serialisation_is_sorted_deterministic(self):
        assert jsonio.sanitise({"c", "a", "b"}) == ["a", "b", "c"]
        assert jsonio.loads_strict(jsonio.dumps_strict({"s": {3, 1, 2}}))["s"] == [1, 2, 3]


# --------------------------------------------------------------------------- #
# every versioned or newly added JSON artifact must be strict
# --------------------------------------------------------------------------- #
def _versioned_json_files() -> list[str]:
    if (REPO_ROOT / ".git").exists():
        out = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "*.json"],
            cwd=REPO_ROOT,
            text=True,
        )
        return sorted({
            line
            for line in out.splitlines()
            if line and (REPO_ROOT / line).is_file()
        })
    # ``git archive`` intentionally has no .git directory. Its filesystem is
    # already the release inventory, so enumerate it directly.
    excluded_parts = {".git", ".venv", "build", "dist", "__pycache__"}
    return sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path in REPO_ROOT.rglob("*.json")
        if path.is_file() and not excluded_parts.intersection(path.relative_to(REPO_ROOT).parts)
    )


def _raw_json_dump_calls(tree: "ast.Module") -> list[int]:
    """Line numbers of json.dump/json.dumps calls, alias-aware (import json as j,
    from json import dumps, dotted or bare, multiline)."""
    import ast
    module_aliases = set()   # names bound to the json module
    bound_dumpers = set()    # names bound directly to json.dump / json.dumps
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "json":
                    module_aliases.add(alias.asname or "json")
        elif isinstance(node, ast.ImportFrom) and node.module == "json":
            for alias in node.names:
                if alias.name in ("dump", "dumps"):
                    bound_dumpers.add(alias.asname or alias.name)
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (isinstance(func, ast.Attribute) and func.attr in ("dump", "dumps")
                and isinstance(func.value, ast.Name) and func.value.id in module_aliases):
            hits.append(func.lineno)
        elif isinstance(func, ast.Name) and func.id in bound_dumpers:
            hits.append(func.lineno)
    return hits


def test_no_raw_json_dump_in_production_code():
    # production code (src/ and scripts/) must write JSON only through the strict
    # writer, so an artifact can never carry a literal NaN/Infinity. Alias-aware AST
    # check: `import json as j; j.dumps(...)` and `from json import dumps` are caught.
    import ast
    immutable_driver = "scripts/evaluation/run_stack_class_coverage_repair_smoke.py"
    evidence = jsonio.load_strict(
        REPO_ROOT
        / "research/evidence/stack_class_coverage_repair_smoke_v1/source_manifest.json"
    )
    immutable_driver_sha = evidence["source_files"][immutable_driver]["sha256"]
    offenders = []
    for root in ("src", "scripts"):
        for path in (REPO_ROOT / root).rglob("*.py"):
            if path.name == "jsonio.py":
                continue
            relative = path.relative_to(REPO_ROOT).as_posix()
            # This audit driver predates jsonio and is bound byte-for-byte to the
            # completed smoke. Its exact implementation uses allow_nan=False;
            # changing even one byte removes this evidence-backed exception.
            if (
                relative == immutable_driver
                and hashlib.sha256(path.read_bytes()).hexdigest() == immutable_driver_sha
            ):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for line in _raw_json_dump_calls(tree):
                offenders.append(f"{relative}:{line}")
    assert not offenders, "raw json.dump call in production code:\n" + "\n".join(offenders)


def test_all_versioned_json_is_strict():
    offenders = []
    for rel in _versioned_json_files():
        path = REPO_ROOT / rel
        try:
            jsonio.load_strict(path)
        except jsonio.StrictJSONError as exc:
            offenders.append(f"{rel}: {exc}")
        except UnicodeDecodeError as exc:
            offenders.append(f"{rel}: not UTF-8: {exc}")
    assert not offenders, "non-strict versioned JSON:\n" + "\n".join(offenders)


# --------------------------------------------------------------------------- #
# claim-status vocabulary
# --------------------------------------------------------------------------- #
class TestClaimStatus:
    def test_exact_claim_status_values(self):
        assert {s.value for s in ClaimStatus} == {
            "engineering",
            "exploratory_internal",
            "development_gold",
            "external_blind",
            "independently_replicated",
        }

    def test_benchmark_role_has_legacy_marker(self):
        assert BenchmarkRole.REPRODUCIBLE_CV_LEGACY_NOT_BLIND.value == "reproducible_cv_legacy_not_blind"

    def test_parse_rejects_unknown(self):
        with pytest.raises(ValueError):
            parse_claim_status("headline")

    def test_status_serialises_as_string(self):
        assert jsonio.dumps_strict({"s": ClaimStatus.EXPLORATORY_INTERNAL}) == '{"s": "exploratory_internal"}'


def test_ruaa_v1_marked_legacy_not_blind():
    data = jsonio.load_strict(REPO_ROOT / "docs" / "ruaa_bench_v1.json")
    assert data["benchmark_role"] == BenchmarkRole.REPRODUCIBLE_CV_LEGACY_NOT_BLIND.value
    assert data["claim_status"] == ClaimStatus.EXPLORATORY_INTERNAL.value


def test_headline_marked_chunk_weighted_training_legacy():
    data = jsonio.load_strict(REPO_ROOT / "docs" / "stylo_lobo_authorci.json")
    assert data["training_weighting"] == "chunk_weighted_training_legacy"


def test_ruaa_manifest_marked_legacy():
    m = jsonio.load_strict(REPO_ROOT / "docs" / "ruaa_bench_manifest.json")
    assert m["benchmark_role"] == BenchmarkRole.REPRODUCIBLE_CV_LEGACY_NOT_BLIND.value
    assert m["claim_status"] == ClaimStatus.EXPLORATORY_INTERNAL.value
    assert m["training_weighting"] == "chunk_weighted_training_legacy"


class TestPassportClaimStatus:
    def _minimal(self, claim_status):
        from stylo.cases.framework import CasePassport
        return CasePassport(
            case_id="c", title="t", status="fail", verdict="v", confidence="low",
            evidence_score=0.0, gate_pass=False, primary_feature_set="fw",
            gates=[], attributions=[], failure_modes=[], data={}, claim_status=claim_status,
        )

    def test_write_passport_rejects_bad_claim_status(self, tmp_path):
        from stylo.cases.framework import write_passport
        with pytest.raises(ValueError):
            write_passport(self._minimal("headline"), tmp_path / "p.json")

    def test_write_then_load_passport_roundtrips(self, tmp_path):
        from stylo.cases.framework import write_passport, load_passport
        path = tmp_path / "p.json"
        write_passport(self._minimal(ClaimStatus.EXPLORATORY_INTERNAL.value), path)
        loaded = load_passport(path)
        assert loaded["claim_status"] == ClaimStatus.EXPLORATORY_INTERNAL.value

    def test_load_passport_rejects_bad_claim_status(self, tmp_path):
        from stylo.cases.framework import load_passport
        path = tmp_path / "p.json"
        jsonio.dump_strict({"case_id": "c", "claim_status": "bogus"}, path)
        with pytest.raises(ValueError):
            load_passport(path)

    def test_load_passport_rejects_nan(self, tmp_path):
        from stylo.cases.framework import load_passport
        path = tmp_path / "p.json"
        path.write_text('{"case_id": "c", "data": {"m": NaN}}', encoding="utf-8")
        with pytest.raises(jsonio.StrictJSONError):
            load_passport(path)

    def test_read_mapping_rejects_nonstrict_json_spec(self, tmp_path):
        from stylo.cases.framework import _read_mapping
        path = tmp_path / "spec.json"
        path.write_text('{"case_id": "c", "unsafe_metric": 1e999}', encoding="utf-8")
        with pytest.raises(jsonio.StrictJSONError):
            _read_mapping(path)

    def test_read_mapping_rejects_yaml_nan_spec(self, tmp_path):
        from stylo.cases.framework import _read_mapping
        path = tmp_path / "spec.yaml"
        path.write_text("case_id: c\nunsafe_metric: .nan\n", encoding="utf-8")
        with pytest.raises(jsonio.StrictJSONError):
            _read_mapping(path)

    @pytest.mark.parametrize("body", [
        "case_id: c\n.inf: null\n",              # non-finite float as a KEY
        "case_id: c\nk: !!set {.nan: null}\n",   # non-finite float inside a SET
    ])
    def test_read_mapping_rejects_yaml_nonfinite_in_keys_and_sets(self, tmp_path, body):
        from stylo.cases.framework import _read_mapping
        path = tmp_path / "spec.yaml"
        path.write_text(body, encoding="utf-8")
        with pytest.raises(jsonio.StrictJSONError):
            _read_mapping(path)

    def test_case_rank_out_writes_strict_json(self, tmp_path, monkeypatch):
        # `stylo case rank --out` must not emit literal NaN into the artifact
        from stylo.cases import cli as ccli
        monkeypatch.setattr(ccli, "load_or_run_many", lambda paths: [])
        monkeypatch.setattr(ccli, "rank_passports", lambda rows: [{"case_id": "c", "m": float("nan")}])
        out = tmp_path / "ranked.json"
        ccli.rank(["ignored"], out=str(out))
        assert "NaN" not in out.read_text(encoding="utf-8")
        assert jsonio.load_strict(out)[0]["m"] is None


# --------------------------------------------------------------------------- #
# release hygiene: publish gate blocks, local audit warns
# --------------------------------------------------------------------------- #
def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _init_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")


class TestReleaseHygiene:
    def test_is_private_path(self):
        assert hygiene.is_private_path("input_clean/a.txt")
        assert hygiene.is_private_path("data/frags_train/x/y.txt")
        assert not hygiene.is_private_path("docs/report.json")

    def test_archive_content_gate_rejects_private_host_layout(self, tmp_path):
        archive = tmp_path / "archive"
        docs = archive / "docs"
        docs.mkdir(parents=True)
        private_home = "/" + "home" + "/example/private/project"
        (docs / "report.json").write_text(
            json.dumps({"source": f"{private_home}/input_clean/author/work.txt"}),
            encoding="utf-8",
        )
        (docs / "escape").symlink_to(f"{private_home}/input_cases")

        issues = hygiene.check_archive_content(archive)
        assert any("report.json" in issue for issue in issues)
        assert any("escape" in issue and "absolute symlink" in issue for issue in issues)

        (docs / "report.json").write_text(
            json.dumps({"source": "docs/public/input.json"}),
            encoding="utf-8",
        )
        (docs / "escape").unlink()
        assert hygiene.check_archive_content(archive) == []

    def test_release_archive_export_excludes_internal_path_bearing_evidence(self):
        attributes = {
            line.strip()
            for line in (REPO_ROOT / ".gitattributes").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        assert {
            "/docs/cases/work_balanced_audit/passports/ export-ignore",
            "/log/experiments/requote_recompute.sh export-ignore",
            "/research/reviews/stylometry_codebase_inventory.json export-ignore",
        }.issubset(attributes)

    def test_publish_gate_flags_private_paths_in_tree(self, tmp_path):
        repo = tmp_path / "r"
        _init_repo(repo)
        (repo / "input_clean").mkdir()
        (repo / "input_clean" / "secret.txt").write_text("x")
        (repo / "README.md").write_text("pub")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "with private")
        assert "input_clean/secret.txt" in hygiene.check_publish_ref("HEAD", cwd=str(repo))
        assert hygiene.check_index(cwd=str(repo)) == ["input_clean/secret.txt"]

    def test_publish_gate_catches_deleted_but_still_in_history(self, tmp_path):
        # a push sends the whole reachable history: a secret that was committed and
        # then deleted is gone from the tip tree but still travels in the push.
        repo = tmp_path / "r"
        _init_repo(repo)
        (repo / "README.md").write_text("pub")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "init")
        (repo / "input_clean").mkdir()
        (repo / "input_clean" / "leak.txt").write_text("secret")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "add secret")
        _git(repo, "rm", "-q", "-r", "input_clean")
        _git(repo, "commit", "-q", "-m", "delete secret")
        # tip tree is clean, but the object is still reachable -> gate must FAIL
        tip = subprocess.check_output(
            ["git", "-C", str(repo), "ls-tree", "-r", "--name-only", "HEAD"], text=True)
        assert "input_clean" not in tip
        assert "input_clean/leak.txt" in hygiene.check_publish_ref("HEAD", cwd=str(repo))

    def test_gate_catches_pathological_filename(self, tmp_path):
        # tab/newline/quote/backslash in a name must not let a private path evade
        # a string-splitting matcher. git's pathspec + NUL output make it byte-safe.
        repo = tmp_path / "r"
        _init_repo(repo)
        name = 'input_clean/a\tb\nc"d\\e.txt'
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("secret")
        (repo / "README.md").write_text("pub")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "pathological")
        assert hygiene.check_index(cwd=str(repo)) == [name]
        assert hygiene.check_publish_ref("HEAD", cwd=str(repo))  # non-empty => blocked

    def test_audit_surfaces_replace_refs(self, tmp_path):
        # a refs/replace entry can swap a private tree for a clean one; the audit
        # must surface it, and the gate uses --no-replace-objects so it is unaffected.
        repo = tmp_path / "r"
        _init_repo(repo)
        (repo / "README.md").write_text("v1")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "c1")
        (repo / "README.md").write_text("v2")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "c2")
        first = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD~1"], text=True).strip()
        second = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        _git(repo, "replace", second, first)
        assert hygiene.replace_refs(cwd=str(repo))
        audit = hygiene.audit_local_refs("HEAD", cwd=str(repo))
        assert audit.has_replace_refs

    def test_publish_gate_clean_but_audit_warns_on_other_ref(self, tmp_path):
        repo = tmp_path / "r"
        _init_repo(repo)
        (repo / "README.md").write_text("pub")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "clean public")
        # a separate branch carries private data; it is never the publish ref
        _git(repo, "checkout", "-q", "-b", "private_local")
        (repo / "input_personal").mkdir()
        (repo / "input_personal" / "diary.txt").write_text("secret")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "private history")
        _git(repo, "checkout", "-q", "main")

        assert hygiene.check_publish_ref("HEAD", cwd=str(repo)) == []
        assert hygiene.check_index(cwd=str(repo)) == []
        audit = hygiene.audit_local_refs("HEAD", cwd=str(repo))
        assert audit.has_private_history
        assert any(r.ref.endswith("private_local") for r in audit.refs)

    def test_real_repo_publish_ref_is_clean(self):
        # the release/publish tree must never carry private corpus paths
        if not (REPO_ROOT / ".git").exists():
            pytest.skip("publish-history gate requires Git; archive contents are gated separately")
        assert hygiene.check_publish_ref("HEAD", cwd=str(REPO_ROOT)) == []
        assert hygiene.check_index(cwd=str(REPO_ROOT)) == []

    def test_publish_gate_catches_unicode_private_path(self, tmp_path):
        # git quotes non-ASCII paths by default ("input_personal/\\320\\264...");
        # the gate must still match the private prefix. Corpus is Cyrillic-named.
        repo = tmp_path / "r"
        _init_repo(repo)
        (repo / "input_personal").mkdir()
        (repo / "input_personal" / "дневник.txt").write_text("secret", encoding="utf-8")
        (repo / "README.md").write_text("pub")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "cyrillic private")
        assert "input_personal/дневник.txt" in hygiene.check_publish_ref("HEAD", cwd=str(repo))
        assert hygiene.check_index(cwd=str(repo)) == ["input_personal/дневник.txt"]

    def test_local_audit_raises_on_invalid_publish_ref(self, tmp_path):
        repo = tmp_path / "r"
        _init_repo(repo)
        (repo / "README.md").write_text("pub")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "c")
        with pytest.raises(hygiene.HygieneError):
            hygiene.audit_local_refs("no_such_ref", cwd=str(repo))

    def test_gate_catches_miscased_private_path(self, tmp_path):
        # a private path in non-canonical case must not evade the gate
        repo = tmp_path / "r"
        _init_repo(repo)
        (repo / "Input_Clean").mkdir()
        (repo / "Input_Clean" / "Secret.txt").write_text("x")
        (repo / "README.md").write_text("pub")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "miscased")
        assert hygiene.check_index(cwd=str(repo)) == ["Input_Clean/Secret.txt"]
        assert "Input_Clean/Secret.txt" in hygiene.check_publish_ref("HEAD", cwd=str(repo))
        assert hygiene.is_private_path("INPUT/leak.txt")

    def test_publish_gate_catches_merged_then_deleted(self, tmp_path):
        # add+delete on a feature branch, then --no-ff merge: history simplification
        # would hide the blob from a pathspec walk, but a push still ships it.
        repo = tmp_path / "r"
        _init_repo(repo)
        (repo / "README.md").write_text("pub")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "init")
        _git(repo, "checkout", "-q", "-b", "feat")
        (repo / "input_clean").mkdir()
        (repo / "input_clean" / "leak.txt").write_text("secret")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "add secret")
        _git(repo, "rm", "-q", "input_clean/leak.txt")
        _git(repo, "commit", "-q", "-m", "delete on feat")
        _git(repo, "checkout", "-q", "main")
        _git(repo, "merge", "-q", "--no-ff", "--no-edit", "feat")
        assert "input_clean/leak.txt" in hygiene.check_publish_ref("HEAD", cwd=str(repo))

    def test_gate_catches_content_aliased_private_path(self, tmp_path):
        # a private path whose blob is shared with a public path (identical content):
        # rev-list --objects emits the blob once with only the public hint, hiding
        # the private path. Full per-commit tree listing must still surface it.
        repo = tmp_path / "r"
        _init_repo(repo)
        (repo / "aaa").mkdir()
        (repo / "input_clean").mkdir()
        (repo / "aaa" / "secret.txt").write_text("SECRET-CONTENT")
        (repo / "input_clean" / "secret.txt").write_text("SECRET-CONTENT")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "both paths")
        _git(repo, "rm", "-q", "-r", "input_clean")
        _git(repo, "commit", "-q", "-m", "remove private path, public alias stays")
        assert "input_clean/secret.txt" in hygiene.check_publish_ref("HEAD", cwd=str(repo))

    def test_gate_catches_exact_root_file(self, tmp_path):
        # a blob named exactly like a protected root (no trailing slash) counts
        repo = tmp_path / "r"
        _init_repo(repo)
        (repo / "input_clean").write_text("secret")
        (repo / "README.md").write_text("pub")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "root file")
        assert hygiene.check_index(cwd=str(repo)) == ["input_clean"]
        assert hygiene.check_publish_ref("HEAD", cwd=str(repo)) == ["input_clean"]

    def test_gate_returns_exact_control_char_path(self, tmp_path):
        # a path with control / line-boundary bytes must be returned byte-exact
        # (rev-list --objects -z), not over-split by splitlines()
        repo = tmp_path / "r"
        _init_repo(repo)
        name = "input_clean/a\x1cb\x0bc\nd.txt"  # FS, VT, LF
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("secret")
        (repo / "README.md").write_text("pub")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "ctrl chars")
        assert hygiene.check_index(cwd=str(repo)) == [name]
        assert name in hygiene.check_publish_ref("HEAD", cwd=str(repo))

    def test_sibling_root_is_not_a_false_positive(self, tmp_path):
        repo = tmp_path / "r"
        _init_repo(repo)
        (repo / "input_cleanX").mkdir()
        (repo / "input_cleanX" / "ok.txt").write_text("ok")
        (repo / "README.md").write_text("pub")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "sibling")
        assert hygiene.check_publish_ref("HEAD", cwd=str(repo)) == []
        assert hygiene.check_index(cwd=str(repo)) == []

    @pytest.mark.parametrize("bad_ref", ["--max-count=0", "--no-walk", "--all"])
    def test_publish_gate_rejects_option_like_ref(self, tmp_path, bad_ref):
        repo = tmp_path / "r"
        _init_repo(repo)
        (repo / "README.md").write_text("pub")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "c")
        with pytest.raises(hygiene.HygieneError):
            hygiene.check_publish_ref(bad_ref, cwd=str(repo))

    def test_audit_detects_tree_and_blob_tags(self, tmp_path):
        # a tag pointing at a tree/blob publishes objects too; the audit must not
        # silently skip it
        repo = tmp_path / "r"
        _init_repo(repo)
        (repo / "README.md").write_text("pub")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "clean")
        (repo / "input_clean").mkdir()
        (repo / "input_clean" / "x.txt").write_text("s")
        _git(repo, "add", "input_clean")
        tree = subprocess.check_output(["git", "-C", str(repo), "write-tree"], text=True).strip()
        _git(repo, "tag", "treetag", tree)
        _git(repo, "reset", "-q")
        blob = subprocess.run(
            ["git", "-C", str(repo), "hash-object", "-w", "--stdin"],
            input=b"secretblob", stdout=subprocess.PIPE, check=True,
        ).stdout.decode().strip()
        _git(repo, "tag", "blobtag", blob)
        audit = hygiene.audit_local_refs("HEAD", cwd=str(repo))
        assert any("input_clean/x.txt" in r.sample for r in audit.refs)
        assert any("blobtag" in nsr for nsr in audit.nonstandard_refs)

    def test_audit_detects_untracked_stash_third_parent(self, tmp_path):
        # `git stash -u` stores untracked files in a third parent commit
        repo = tmp_path / "r"
        _init_repo(repo)
        (repo / "README.md").write_text("pub")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "c")
        (repo / "input_personal").mkdir()
        (repo / "input_personal" / "diary.txt").write_text("secret")
        _git(repo, "stash", "-u")
        assert hygiene.check_publish_ref("HEAD", cwd=str(repo)) == []
        audit = hygiene.audit_local_refs("HEAD", cwd=str(repo))
        assert audit.has_private_history
        assert any("input_personal/diary.txt" in s.sample for s in audit.stashes)

    def test_publish_gate_checks_arbitrary_branch_not_just_head(self, tmp_path):
        # a push of `priv:priv` publishes a branch that is not HEAD; the gate must
        # be able to certify any pushed sha, which is what the pre-push hook relies on
        repo = tmp_path / "r"
        _init_repo(repo)
        (repo / "README.md").write_text("pub")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "clean head")
        _git(repo, "checkout", "-q", "-b", "priv")
        (repo / "input_clean").mkdir()
        (repo / "input_clean" / "leak.txt").write_text("secret")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "private branch")
        _git(repo, "checkout", "-q", "main")
        assert hygiene.check_publish_ref("HEAD", cwd=str(repo)) == []
        assert "input_clean/leak.txt" in hygiene.check_publish_ref("priv", cwd=str(repo))

    def test_prepush_hook_blocks_private_branch(self, tmp_path):
        # end-to-end: the hook reads stdin and checks each pushed sha
        repo = tmp_path / "r"
        _init_repo(repo)
        (repo / "README.md").write_text("pub")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "clean head")
        _git(repo, "checkout", "-q", "-b", "priv")
        (repo / "input_clean").mkdir()
        (repo / "input_clean" / "leak.txt").write_text("secret")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "private branch")
        _git(repo, "checkout", "-q", "main")
        # symlink the real script into the tmp repo so the hook's repo-root resolves
        (repo / "scripts").mkdir()
        (repo / "scripts" / "check_release_hygiene.py").symlink_to(REPO_ROOT / "scripts" / "check_release_hygiene.py")
        (repo / "src").symlink_to(REPO_ROOT / "src")
        priv = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "priv"], text=True).strip()
        zero = "0" * 40
        stdin = f"refs/heads/priv {priv} refs/heads/priv {zero}\n"
        result = subprocess.run(
            ["bash", str(REPO_ROOT / ".githooks" / "pre-push"), "origin", "url"],
            input=stdin, cwd=str(repo), capture_output=True, text=True,
        )
        assert result.returncode == 1, result.stderr

        # a deletion (all-zero local sha) must be skipped for BOTH sha widths
        for width in (40, 64):
            deletion = f"refs/heads/priv {'0' * width} refs/heads/priv {priv}\n"
            skipped = subprocess.run(
                ["bash", str(REPO_ROOT / ".githooks" / "pre-push"), "origin", "url"],
                input=deletion, cwd=str(repo), capture_output=True, text=True,
            )
            assert skipped.returncode == 0, f"width={width}: {skipped.stderr}"

    def test_publish_gate_fails_closed_on_shallow_clone(self, tmp_path):
        # a shallow clone truncates the history walk; the gate must refuse to
        # certify it rather than emit a false green.
        origin = tmp_path / "origin"
        _init_repo(origin)
        (origin / "README.md").write_text("pub")
        _git(origin, "add", "-A")
        _git(origin, "commit", "-q", "-m", "c1")
        (origin / "more.txt").write_text("x")
        _git(origin, "add", "-A")
        _git(origin, "commit", "-q", "-m", "c2")
        shallow = tmp_path / "shallow"
        subprocess.run(
            ["git", "clone", "--depth", "1", "--quiet", origin.as_uri(), str(shallow)],
            check=True, capture_output=True,
        )
        assert subprocess.check_output(
            ["git", "-C", str(shallow), "rev-parse", "--is-shallow-repository"], text=True
        ).strip() == "true"
        with pytest.raises(hygiene.HygieneError):
            hygiene.check_publish_ref("HEAD", cwd=str(shallow))


def test_default_ruaa_specs_reproduce_committed_rows():
    # a default rerun must not silently drop a committed leaderboard model.
    # Parse DEFAULT_SPECS via AST rather than importing the heavy RuAA pipeline,
    # so this test runs on a CI runner without spaCy/torch installed.
    import ast
    tree = ast.parse((REPO_ROOT / "scripts" / "run_ruaa_baselines.py").read_text(encoding="utf-8"))
    default_specs = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "DEFAULT_SPECS" for t in node.targets
        ):
            default_specs = [ast.literal_eval(elt) for elt in node.value.elts]
    assert default_specs is not None, "DEFAULT_SPECS not found"
    committed = {row["model"] for row in jsonio.load_strict(REPO_ROOT / "docs" / "ruaa_bench_v1.json")["leaderboard"]}
    assert committed.issubset(set(default_specs))
