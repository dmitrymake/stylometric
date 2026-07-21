"""Path-guarded transient store and the verified content-addressed publisher (§4.4/§8).

The run's transient output (per-fold checkpoints, full per-work vectors) lives under the **gitignored**
``docs/exploratory/work_balanced/audit/runs/<run_id>/`` (a path-aware guard, no headline path). The
final **verified** artifact is promoted to the unignored, committed ``docs/work_balanced_paired_audit_v1.json``
(matched by ``!docs/*.json``), and the full per-work probability vectors are durably committed as a
**content-addressed archive** ``docs/work_balanced_paired_audit_v1/`` (each per-cell per-work file named
by its content hash) with a committed ``SHA256SUMS`` the summary references by hash, published via the
immutable version-directory + single atomic ``current.json``/``COMPLETE`` pointer pattern.

The publisher can write ONLY inside the transient run namespace and the two published audit paths; it
**refuses** to write any headline/frozen artifact (an allowlist plus a headline-basename denylist),
and rejects path traversal or a symlinked path chain. It never touches ``0.8805``, the frozen baseline
snapshot, or the README/site/PAPER headline artifacts.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import re
import shutil
import tempfile
from typing import Mapping

from ...jsonio import dump_strict, dumps_strict, load_strict
from ...pipeline.bundle import (BundleError, _real_within, _safe_name,
                                _sha256_file, _verify_real_dir_chain)

RUNS_SUBPATH = pathlib.PurePosixPath("exploratory/work_balanced/audit/runs")
SUMMARY_NAME = "work_balanced_paired_audit_v1.json"
ARCHIVE_DIRNAME = "work_balanced_paired_audit_v1"
VERSIONS_DIR = "versions"
CURRENT_NAME = "current.json"
COMPLETE_NAME = "COMPLETE"
SHA256SUMS_NAME = "SHA256SUMS"
SUMMARY_IN_VERSION = "summary.json"
_PUBLISH_SCHEMA = "paired_audit.publish.v1"

# frozen/headline artifacts the publisher must NEVER write (denylist, belt-and-suspenders on top of
# the allowlist)
_HEADLINE_BASENAMES = frozenset({
    "final_comparison.txt", "final_comparison.v2.txt", "final_comparison.csv", "final_comparison.v2.csv",
    "lobo_books.txt", "screening_panel_v1.json", "p0_baseline_snapshot.json",
    "corpus_manifest.json", "corpus_validation.json", "README.md", "PAPER.md", "index.html",
})


class PublisherError(RuntimeError):
    """Fail-closed: a write escapes the allowed audit namespace, targets a headline path, or the
    published archive is inconsistent."""


def _docs_root(docs_root: pathlib.Path | str) -> pathlib.Path:
    return pathlib.Path(docs_root).resolve()


def assert_writable_audit_path(path: pathlib.Path | str, *, docs_root: pathlib.Path | str,
                               run_id: str | None = None, allow_published: bool = False) -> pathlib.Path:
    """Fail-closed unless ``path`` is inside an allowed audit namespace and is not a headline artifact.

    Allowed: the transient run dir ``<docs>/exploratory/work_balanced/audit/runs/<run_id>/…`` (when a
    ``run_id`` is given) and — only when ``allow_published`` — the published summary
    ``<docs>/work_balanced_paired_audit_v1.json`` and the archive dir ``<docs>/work_balanced_paired_audit_v1/…``.
    """
    droot = _docs_root(docs_root)
    p = pathlib.Path(path)
    if p.name in _HEADLINE_BASENAMES:
        raise PublisherError(f"refusing to write a headline/frozen artifact: {p.name}")
    # the existing part of the parent chain must be symlink-free
    anchor = p
    while not anchor.exists():
        anchor = anchor.parent
    try:
        _verify_real_dir_chain(anchor)
    except BundleError as exc:
        raise PublisherError(f"unsafe path chain: {exc}") from exc
    # NORMALIZE the (possibly non-existent) tail before any containment test — a tail with `..`
    # segments must not string-prefix-match an allowed base while its realpath escapes.
    raw = (anchor.resolve() / p.relative_to(anchor)) if anchor != p else anchor.resolve()
    resolved = pathlib.Path(os.path.normpath(str(raw)))

    def _within(base: pathlib.Path) -> bool:
        base = base.resolve()
        return resolved == base or str(resolved).startswith(str(base) + os.sep)

    if not _within(droot):
        raise PublisherError(f"audit write escapes docs root: {p}")
    allowed = False
    if run_id is not None:
        if not _safe_name(run_id):
            raise PublisherError("run_id is not a safe path token")
        if _within(droot / RUNS_SUBPATH / run_id):
            allowed = True
    if allow_published:
        if resolved == (droot / SUMMARY_NAME) or _within(droot / ARCHIVE_DIRNAME):
            allowed = True
    if not allowed:
        raise PublisherError(f"path is not in an allowed audit namespace: {p}")
    return p


def write_transient(run_id: str, relname: str, obj: Mapping, *,
                    docs_root: pathlib.Path | str) -> pathlib.Path:
    """Atomically write a transient run artifact under the gitignored run namespace. The path is
    guarded BEFORE any mkdir (so a symlinked run-namespace ancestor is caught before it can be created
    through) and re-checked AFTER the mkdir."""
    if not _safe_name(relname):
        raise PublisherError(f"unsafe transient artifact name: {relname!r}")
    droot = _docs_root(docs_root)
    path = droot / RUNS_SUBPATH / run_id / relname
    assert_writable_audit_path(path, docs_root=docs_root, run_id=run_id)   # guard BEFORE mkdir
    path.parent.mkdir(parents=True, exist_ok=True)
    assert_writable_audit_path(path, docs_root=docs_root, run_id=run_id)   # re-check AFTER mkdir
    dump_strict(dict(obj), path, trailing_newline=True)
    return path


# ── content-addressed per-work archive ───────────────────────────────────────
def _stage_archive(staging: pathlib.Path, per_work_vectors: Mapping[str, object]) -> dict:
    """Write each per-cell per-work payload to ``<content-sha256>.json`` and a SHA256SUMS inventory.

    Returns ``{key: {"filename", "sha256"}}`` for the summary to reference by hash.
    """
    inventory: dict[str, dict] = {}
    digest_to_name: dict[str, str] = {}
    for key in sorted(per_work_vectors):
        payload = per_work_vectors[key]
        content = dumps_strict(payload, sort_keys=True).encode("utf-8") + b"\n"
        digest = hashlib.sha256(content).hexdigest()
        fname = f"{digest}.json"
        fpath = staging / fname
        if digest not in digest_to_name:                 # identical content dedups to one file
            fpath.write_bytes(content)
            digest_to_name[digest] = fname
        inventory[key] = {"filename": fname, "sha256": digest}
    lines = sorted(f"{digest}  {name}" for digest, name in digest_to_name.items())
    (staging / SHA256SUMS_NAME).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return inventory


def _self_hash(body: Mapping) -> str:
    return hashlib.sha256(dumps_strict(body, sort_keys=True).encode("utf-8")).hexdigest()


def _archive_token(staging: pathlib.Path) -> str:
    """Deterministic token over every file in the staged version dir (name + content sha)."""
    h = hashlib.sha256()
    h.update(_PUBLISH_SCHEMA.encode("utf-8"))
    for entry in sorted(os.scandir(staging), key=lambda e: e.name):
        if entry.is_symlink():
            raise PublisherError(f"symlink in staged archive (rejected): {entry.path}")
        h.update(len(entry.name).to_bytes(8, "big") + entry.name.encode("utf-8"))
        h.update(bytes.fromhex(_sha256_file(pathlib.Path(entry.path))))
    return h.hexdigest()[:32]


def _version_complete(versioned: pathlib.Path, token: str) -> bool:
    if not _real_within(versioned, versioned.parent, must_dir=True) or versioned.name != token:
        return False
    try:
        return _archive_token(versioned) == token
    except PublisherError:
        return False


_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEADLINE_DECISIONS = frozenset({"relabel", "keep_legacy", "inconclusive"})
# the exact top-level shape of a candidate summary (no injected/decorative extra field survives)
_SUMMARY_KEYS = frozenset({"run_id", "claim_status", "run_plan", "universes", "continuous_tolerances",
                           "attestation", "cells", "holm", "headline", "result_audit", "run_id_source"})


def verify_final_assembly(summary: Mapping, per_work_vectors: Mapping) -> None:
    """Fail-closed unless the summary is a COMPLETE verified assembly (§4.1/§4.4): sha256 run_id,
    exploratory claim, the full 30-cell applicability matrix for BOTH datasets with valid records, the
    exact 15-member Holm family per dataset, the registered headline endpoint/decision, an attestation,
    and per-work vectors covering exactly the applied cells. An arbitrary/partial Mapping is rejected."""
    from .applicability import (CELLS, MODELS, ApplicabilityError, assert_cell_record,
                                assert_holm_family_complete, assert_matrix_invariants, registered_cells)
    from .headline import HEADLINE_ENDPOINT

    from .run_plan import run_id as _recompute_run_id

    if set(summary) != _SUMMARY_KEYS:               # exact top-level shape — no injected/extra field
        raise PublisherError(f"summary top-level keys must be exactly {sorted(_SUMMARY_KEYS)}")
    if not (isinstance(summary.get("run_id"), str) and _HEX64.match(summary["run_id"])):
        raise PublisherError("summary.run_id must be a sha256 hex string")
    if summary.get("claim_status") != "exploratory_internal":
        raise PublisherError("summary.claim_status must be exploratory_internal")
    if summary.get("run_id_source") != "canonical_run_plan_sha256":
        raise PublisherError("summary.run_id_source must be canonical_run_plan_sha256")

    # the embedded canonical RunPlan must RECOMPUTE to the summary run_id (independent identity), and
    # the completeness sections (both class orders / universes / tolerances / full attestation) present
    plan = summary.get("run_plan")
    if not isinstance(plan, dict):
        raise PublisherError("summary must embed the canonical run_plan")
    if _recompute_run_id(plan) != summary["run_id"]:
        raise PublisherError("summary.run_id does not recompute from the embedded run_plan")
    # re-apply EVERY build invariant to the embedded plan (frozen stats/tolerances for a confirmatory
    # plan, clean tree, spaCy, a registered evaluator, hex digests) — recomputing the id is not enough:
    # a forged confirmatory plan is self-consistent with its own id.
    from .run_plan import RunPlanError, assert_wellformed_run_plan
    try:
        assert_wellformed_run_plan(plan)
    except (RunPlanError, TypeError, ValueError) as exc:         # any malformed plan field -> fail closed
        raise PublisherError(f"embedded run_plan is not well-formed: {exc}") from exc
    universes = summary.get("universes")
    if not isinstance(universes, Mapping) or set(universes) != {"lobo", "ruaa"}:
        raise PublisherError("summary.universes must cover lobo and ruaa")
    # every universes field is bound to the identity-bound run_plan (no forgeable decorative copy):
    # the class orders drive the auditor's metric_idx, the digests anchor the dataset/manifest.
    _UNI_KEYS = ("dataset_digest", "fold_manifest_digest", "probability_class_order", "metric_label_order")
    for ds, u in universes.items():
        if set(u) != set(_UNI_KEYS):
            raise PublisherError(f"summary.universes[{ds}] must carry exactly {_UNI_KEYS}")
        if any(u[k] != plan[ds][k] for k in _UNI_KEYS):
            raise PublisherError(f"summary.universes[{ds}] != the run_plan dataset binding")
    if summary.get("continuous_tolerances") != plan["tolerances"]:
        raise PublisherError("summary.continuous_tolerances != the run_plan tolerances")
    # the attestation must EQUAL the run_plan-bound values (every field lives in the run_id, so a forged
    # provenance stamp cannot diverge from the identity)
    att = summary.get("attestation")
    expected_att = {"git_commit": plan["git_commit"], "run_kind": plan["run_kind"],
                    "audit_version": plan["audit_version"],
                    "execution_source_sha256": plan["execution_source_sha256"],
                    "env_lock_sha256": plan["env_lock_sha256"], "config_id": plan["config_id"],
                    "golden_fixture_inventory_sha": plan["golden_fixture_inventory_sha"]}
    if att != expected_att:
        raise PublisherError("summary.attestation != the run_plan-bound attestation")

    cells = summary.get("cells")
    if not isinstance(cells, Mapping) or set(cells) != {"lobo", "ruaa"}:
        raise PublisherError("summary.cells must cover exactly lobo and ruaa")
    grid_keys = {f"{m}/{c}" for m in MODELS for c in CELLS}
    holm = summary.get("holm")
    if not isinstance(holm, Mapping) or set(holm) != {"lobo", "ruaa"}:
        raise PublisherError("summary.holm must cover lobo and ruaa")
    try:                                                 # surface applicability failures as PublisherError
        assert_matrix_invariants()
        for ds, grid in cells.items():
            if not isinstance(grid, Mapping) or set(grid) != grid_keys:
                raise PublisherError(f"summary.cells[{ds}] must carry all 30 (model,cell) records")
            for m in MODELS:
                for c in CELLS:
                    assert_cell_record(m, c, grid[f"{m}/{c}"])
        for ds, fam in holm.items():
            if not isinstance(fam, Mapping):
                raise PublisherError(f"summary.holm[{ds}] must be a mapping")
            assert_holm_family_complete([tuple(k.split("/", 1)) for k in fam])
    except ApplicabilityError as exc:
        raise PublisherError(f"invalid applicability/Holm content in the assembly: {exc}") from exc

    hl = summary.get("headline")
    if not isinstance(hl, Mapping) or hl.get("endpoint") != HEADLINE_ENDPOINT:
        raise PublisherError("summary.headline must use the registered stylo A4-A0 endpoint")
    if hl.get("decision") not in _HEADLINE_DECISIONS:
        raise PublisherError("summary.headline decision must be relabel/keep_legacy/inconclusive")

    # §8: the candidate must carry the fixed passing result-audit stamp (re-derived below at publish)
    if summary.get("result_audit") != {"passed": True, "auditor": "independent_recompute_v1"}:
        raise PublisherError("summary.result_audit must be the fixed passing independent-audit stamp")
    # Holm <-> cell verdict consistency across both datasets
    for ds, fam in holm.items():
        for key, hp in fam.items():
            vs = cells.get(ds, {}).get(key, {}).get("vs_A0")
            if not isinstance(vs, Mapping) or vs.get("holm_p") != hp.get("holm_p") \
                    or bool(vs.get("significant")) != bool(hp.get("significant")):
                raise PublisherError(f"Holm<->cell verdict inconsistency at {ds}/{key}")
    # headline <-> CI consistency: the margin MUST equal the run_id-bound frozen protocol margin (a
    # publish-boundary craft cannot decide under a softer δ), the decision is the gate on the stored
    # difference CI, and that CI equals the stylo A4-A0 cell difference CI in the lobo grid.
    from .headline import HeadlineError, headline_gate
    frozen_margin = plan.get("stats", {}).get("noninferiority_margin")
    if hl.get("margin") != frozen_margin:
        raise PublisherError("headline margin != the run_plan frozen noninferiority margin")
    dci = hl.get("diff_ci") or {}
    try:
        gate = headline_gate(dci["lo"], dci["hi"], margin=frozen_margin)
    except (KeyError, TypeError, HeadlineError) as exc:
        raise PublisherError(f"headline diff CI/margin is malformed: {exc}") from exc
    if gate != hl.get("decision"):
        raise PublisherError("headline decision != the gate applied to its own difference CI")
    a4_vs = cells.get("lobo", {}).get("stylo/A4", {}).get("vs_A0", {})
    if [dci.get("lo"), dci.get("hi")] != list(a4_vs.get("dacc_authorclustered_ci", [])):
        raise PublisherError("headline diff CI != the stylo A4-A0 cell difference CI")

    if not isinstance(per_work_vectors, Mapping) or not per_work_vectors:
        raise PublisherError("per_work_vectors must be a non-empty mapping")
    applied_keys = {f"{ds}/{m}/{c}" for ds in ("lobo", "ruaa") for (m, c) in registered_cells()}
    if set(per_work_vectors) != applied_keys:
        raise PublisherError("per_work_vectors must cover exactly the applied cells across both datasets")
    for k, vec in per_work_vectors.items():
        if not isinstance(vec, list) or not vec:
            raise PublisherError(f"per_work_vectors[{k}] must be a non-empty list")
    # the cell record's per_work is a redundant copy of the content-addressed archive vector — reconcile
    # them so a forged in-cell per_work cannot diverge from the audited archive.
    for ds in ("lobo", "ruaa"):
        for (m, c) in registered_cells():
            if cells[ds][f"{m}/{c}"].get("per_work") != per_work_vectors[f"{ds}/{m}/{c}"]:
                raise PublisherError(f"cells[{ds}][{m}/{c}].per_work != the archived per-work vector")


def _archive_root(droot: pathlib.Path, run_id: str, *, confirmatory: bool) -> pathlib.Path:
    """The confirmatory publish targets the committed content-addressed archive; a smoke/dry run
    targets the gitignored TRANSIENT run namespace so it can never write the production artifact."""
    return droot / ARCHIVE_DIRNAME if confirmatory else droot / RUNS_SUBPATH / run_id / ARCHIVE_DIRNAME


def publish_audit(summary: Mapping, per_work_vectors: Mapping[str, object], *,
                  docs_root: pathlib.Path | str, run_kind: str = "confirmatory") -> dict:
    """Verified publication: validate the COMPLETE audited assembly, stage the content-addressed
    per-work archive, bind it into the summary self-hash, publish the immutable version dir + atomic
    ``current.json``/``COMPLETE`` pointer — all path-guarded, never a headline path. A CONFIRMATORY run
    additionally writes the committed ``docs/*.json`` production artifact; a smoke/dry run publishes
    ONLY under the gitignored transient run namespace and never touches the committed artifact."""
    verify_final_assembly(summary, per_work_vectors)     # reject an arbitrary/partial assembly first
    # the publish target must match the run's OWN kind (a smoke/dry summary can never be published as
    # the confirmatory committed artifact), and the publisher RE-DERIVES every metric from the vectors
    # rather than trusting the caller-set result_audit.passed flag.
    plan = summary["run_plan"]
    if plan.get("run_kind") != run_kind:
        raise PublisherError(
            f"publish run_kind {run_kind!r} != the embedded run_plan run_kind {plan.get('run_kind')!r}")
    from .result_audit import ResultAuditError, audit_results
    try:
        audit_results(summary, per_work_vectors, plan)
    except ResultAuditError as exc:
        raise PublisherError(f"publish-time independent result audit failed: {exc}") from exc
    except (KeyError, ValueError, TypeError, IndexError) as exc:   # malformed vectors -> fail closed
        raise PublisherError(f"publish-time result audit could not run on the vectors: {exc}") from exc
    confirmatory = run_kind == "confirmatory"

    droot = _docs_root(docs_root)
    run_id = summary["run_id"]
    guard = {"allow_published": True} if confirmatory else {"run_id": run_id}
    if confirmatory:
        assert_archive_committable(droot)                # §4.4 durability: refuse if the archive is gitignored
    archive_root = _archive_root(droot, run_id, confirmatory=confirmatory)
    versions = archive_root / VERSIONS_DIR
    assert_writable_audit_path(versions, docs_root=docs_root, **guard)  # guard BEFORE mkdir
    versions.mkdir(parents=True, exist_ok=True)
    _verify_real_dir_chain(versions)

    staging = pathlib.Path(tempfile.mkdtemp(dir=versions, prefix=".staging_"))
    published: dict = {}
    try:
        inventory = _stage_archive(staging, per_work_vectors)
        sha256sums_digest = _sha256_file(staging / SHA256SUMS_NAME)
        full_summary = dict(summary)
        full_summary["per_work_archive"] = inventory
        full_summary["archive_sha256sums_digest"] = sha256sums_digest
        full_summary["schema"] = _PUBLISH_SCHEMA
        full_summary["self_hash"] = _self_hash({k: v for k, v in full_summary.items()
                                                if k != "self_hash"})
        dump_strict(full_summary, staging / SUMMARY_IN_VERSION, trailing_newline=True)

        token = _archive_token(staging)
        versioned = versions / token
        if versioned.exists() or versioned.is_symlink():
            if not _version_complete(versioned, token):
                raise PublisherError(f"archive version {token} exists with different content (conflict)")
            shutil.rmtree(staging)
        else:
            os.replace(staging, versioned)
        staging = None
        published = {"version": token, "versioned_dir": versioned, "summary": full_summary}
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    # atomic pointer + COMPLETE marker (dump_strict itself does a secure mkstemp + os.replace)
    for name, body in ((CURRENT_NAME, {"schema": _PUBLISH_SCHEMA, "version": published["version"]}),
                       (COMPLETE_NAME, {"schema": _PUBLISH_SCHEMA, "version": published["version"],
                                        "run_id": summary["run_id"]})):
        target = archive_root / name
        assert_writable_audit_path(target, docs_root=docs_root, **guard)
        dump_strict(body, target, trailing_newline=True)

    # the committed production artifact (matched by !docs/*.json) is written ONLY for a confirmatory run
    if confirmatory:
        summary_path = droot / SUMMARY_NAME
        assert_writable_audit_path(summary_path, docs_root=docs_root, allow_published=True)
        dump_strict(published["summary"], summary_path, trailing_newline=True)
        published["summary_path"] = summary_path
    published["run_kind"] = run_kind
    return published


def _git_toplevel(path: pathlib.Path):
    import subprocess
    try:
        out = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=path,
                             capture_output=True, text=True)
        return pathlib.Path(out.stdout.strip()) if out.returncode == 0 else None
    except Exception:
        return None


def assert_archive_committable(docs_root: pathlib.Path | str, *, repo_root=None) -> None:
    """Fail-closed unless the content-addressed archive subtree is committable (§4.4).

    §4.4 promises the per-work vectors are DURABLY committed, so a confirmatory publish must verify the
    archive path is NOT git-ignored (a ``!docs/work_balanced_paired_audit_v1/`` whitelist must exist)
    before claiming durable publication. No-op outside a git repository (e.g. a tmp test root).
    """
    import subprocess
    docs_root = pathlib.Path(docs_root)
    probe = docs_root / ARCHIVE_DIRNAME / VERSIONS_DIR / "token" / "x.json"
    root = pathlib.Path(repo_root) if repo_root is not None else _git_toplevel(docs_root)
    if root is None:
        return
    res = subprocess.run(["git", "check-ignore", str(probe)], cwd=root,
                         capture_output=True, text=True)
    if res.returncode == 0:                              # git names the ignoring rule => path IS ignored
        raise PublisherError(
            f"content-addressed archive subtree is git-ignored ({res.stdout.strip()}); add a "
            f"!docs/{ARCHIVE_DIRNAME}/ whitelist before claiming durable publication (§4.4)")


def load_published_audit(docs_root: pathlib.Path | str, *, run_kind: str = "confirmatory",
                         run_id: str | None = None) -> dict:
    """Load and verify the published audit: pointer → immutable version dir → token integrity →
    summary self-hash → run_id recompute → archive files match the summary inventory by content hash.
    A confirmatory load additionally asserts the committed root summary equals the version-dir summary;
    a smoke/dry load reads the transient run namespace (a safe ``run_id`` is required)."""
    droot = _docs_root(docs_root)
    confirmatory = run_kind == "confirmatory"
    if confirmatory:
        archive_root = droot / ARCHIVE_DIRNAME
    else:
        if not (isinstance(run_id, str) and _safe_name(run_id)):
            raise PublisherError("a transient (smoke/dry) load requires a safe run_id")
        archive_root = droot / RUNS_SUBPATH / run_id / ARCHIVE_DIRNAME
    _verify_real_dir_chain(archive_root)
    ptr = archive_root / CURRENT_NAME
    if not _real_within(ptr, archive_root, must_file=True):
        raise PublisherError("current pointer missing, a symlink, or escapes the archive root")
    pointer = load_strict(ptr)
    if pointer.get("schema") != _PUBLISH_SCHEMA:
        raise PublisherError("current pointer schema mismatch")
    token = pointer.get("version")
    if not isinstance(token, str) or not _safe_name(token):
        raise PublisherError("current pointer version is not a safe token")
    versioned = archive_root / VERSIONS_DIR / token
    if not _version_complete(versioned, token):
        raise PublisherError("published archive version is partial, tampered, or conflicting")
    complete = archive_root / COMPLETE_NAME
    if not _real_within(complete, archive_root, must_file=True):
        raise PublisherError("COMPLETE marker missing, a symlink, or escapes the archive root")
    if load_strict(complete).get("version") != token:
        raise PublisherError("COMPLETE marker version does not match the current pointer")

    summary = load_strict(versioned / SUMMARY_IN_VERSION)
    recorded = {k: v for k, v in summary.items() if k != "self_hash"}
    if summary.get("self_hash") != _self_hash(recorded):
        raise PublisherError("published summary self-hash mismatch")
    # the run_id must RECOMPUTE from the embedded canonical run_plan, and the plan must re-apply every
    # build invariant (a forged confirmatory plan is self-consistent with its own id)
    from .run_plan import RunPlanError, assert_wellformed_run_plan
    from .run_plan import run_id as _recompute_run_id
    if not isinstance(summary.get("run_plan"), dict) \
            or _recompute_run_id(summary["run_plan"]) != summary.get("run_id"):
        raise PublisherError("loaded run_id does not recompute from the embedded run_plan")
    try:
        assert_wellformed_run_plan(summary["run_plan"])
    except (RunPlanError, TypeError, ValueError) as exc:
        raise PublisherError(f"loaded run_plan is not well-formed: {exc}") from exc
    for key, ref in summary.get("per_work_archive", {}).items():
        fpath = versioned / ref["filename"]
        if not _real_within(fpath, versioned, must_file=True) or _sha256_file(fpath) != ref["sha256"]:
            raise PublisherError(f"archive file for {key} missing or content-hash mismatch")

    # SHA256SUMS must parse, match the summary digest, and cover EXACTLY the per-work files
    sums_path = versioned / SHA256SUMS_NAME
    if not _real_within(sums_path, versioned, must_file=True):
        raise PublisherError("archive SHA256SUMS missing or a symlink")
    if _sha256_file(sums_path) != summary.get("archive_sha256sums_digest"):
        raise PublisherError("archive_sha256sums_digest does not match the SHA256SUMS file")
    listed = {}
    for raw in sums_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        digest, _, name = raw.partition("  ")
        listed[name.strip()] = digest.strip()
    referenced = {r["filename"]: r["sha256"] for r in summary.get("per_work_archive", {}).values()}
    if listed != referenced:
        raise PublisherError("SHA256SUMS does not match the summary per-work inventory")
    # exact directory inventory: no extra or missing file in the immutable version dir
    on_disk = {e.name for e in os.scandir(versioned)}
    expected = set(referenced) | {SHA256SUMS_NAME, SUMMARY_IN_VERSION}
    if on_disk != expected:
        raise PublisherError(
            f"version dir inventory mismatch (extra {sorted(on_disk - expected)[:3]}, "
            f"missing {sorted(expected - on_disk)[:3]})")
    # §8: the committed root summary must equal the immutable version-dir summary (a confirmatory run)
    if confirmatory:
        root_summary = load_strict(droot / SUMMARY_NAME)
        if root_summary != summary:
            raise PublisherError("committed root summary != the version-dir summary")
    return {"version": token, "versioned_dir": versioned, "summary": summary}
