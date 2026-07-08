"""Reusable gate-first protocol for historical attribution cases.

The module is deliberately lightweight: it uses fixed function-word and char
profile channels, so a case can be screened without spaCy caches or the full
benchmark corpus. Heavy, case-specific scripts can still exist, but their final
results should fit the same passport shape.
"""
from __future__ import annotations

import dataclasses
import itertools
import json
import math
import pathlib
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize

from ..lang import function_words

WORD_RE = re.compile(r"[\w\-]+", re.U)
DEFAULT_FEATURE_SETS = ("fw_fixed", "char3")


@dataclasses.dataclass(frozen=True)
class CorpusSource:
    """One author/candidate corpus in a case panel."""

    author_id: str
    path: pathlib.Path
    paths: Tuple[pathlib.Path, ...] = ()
    exclude: Tuple[pathlib.Path, ...] = ()
    role: str = "candidate"  # candidate | distractor
    label: Optional[str] = None

    def all_paths(self) -> Tuple[pathlib.Path, ...]:
        return (self.path,) + tuple(self.paths)


@dataclasses.dataclass(frozen=True)
class CaseSpec:
    """Decision-complete machine-readable description of a case."""

    case_id: str
    title: str
    candidates: Tuple[CorpusSource, ...]
    target: Optional[pathlib.Path] = None
    hypothesis: str = ""
    target_description: str = ""
    claim: str = ""
    limitations: Tuple[str, ...] = ()
    provenance: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    unit: str = "work"
    language: str = "ru"
    feature_sets: Tuple[str, ...] = DEFAULT_FEATURE_SETS
    required_gates: Tuple[str, ...] = ("feasibility_gate",)
    forbidden_sources: Tuple[pathlib.Path, ...] = ()
    sources: Tuple[str, ...] = ()
    notes: str = ""
    chunk_words: int = 1500
    min_work_words: int = 30
    max_exact_permutations: int = 400
    random_permutations: int = 500
    seed: int = 42


@dataclasses.dataclass
class GateResult:
    feature_set: str
    status: str
    gate_pass: bool
    work_macro_recall: float
    chunk_weighted_recall: float
    work_recall: Dict[str, str]
    chunk_recall: Dict[str, str]
    permutation_p: Optional[float]
    permutation_method: Optional[str]
    permutation_exact_floor: Optional[float]
    confusion: Dict[str, Dict[str, int]]
    n_works: int
    n_chunks: int
    failure_modes: List[str]


@dataclasses.dataclass
class AttributionResult:
    feature_set: str
    top: Optional[str]
    second: Optional[str]
    margin: Optional[float]
    margin_ci95: Optional[Tuple[float, float]]
    winner_share: Dict[str, float]
    similarities: List[Dict[str, float]]
    per_chunk_winners: Dict[str, int]
    n_chunks: int


@dataclasses.dataclass
class CasePassport:
    case_id: str
    title: str
    status: str
    verdict: str
    confidence: str
    evidence_score: float
    gate_pass: bool
    primary_feature_set: str
    gates: List[GateResult]
    attributions: List[AttributionResult]
    failure_modes: List[str]
    data: Dict[str, Any]
    notes: str = ""
    hypothesis: str = ""
    target_description: str = ""
    claim: str = ""
    limitations: Tuple[str, ...] = ()
    provenance: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "title": self.title,
            "hypothesis": self.hypothesis,
            "target_description": self.target_description,
            "claim": self.claim,
            "limitations": list(self.limitations),
            "status": self.status,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "evidence_score": round(float(self.evidence_score), 2),
            "gate_pass": bool(self.gate_pass),
            "primary_feature_set": self.primary_feature_set,
            "gates": [dataclasses.asdict(g) for g in self.gates],
            "attributions": [_attr_to_dict(a) for a in self.attributions],
            "failure_modes": list(self.failure_modes),
            "data": self.data,
            "provenance": dict(self.provenance),
            "notes": self.notes,
        }


@dataclasses.dataclass(frozen=True)
class Work:
    author: str
    work_id: str
    text: str
    chunks: Tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class FwContext:
    X: Any
    chunk_work: np.ndarray


def load_case_spec(path: str | pathlib.Path) -> CaseSpec:
    """Load a case spec from YAML or JSON.

    Supported compact candidate shape:

        candidates:
          chehov: input_cases/.../cand_chehov
        distractors:
          bilibin: {path: input_cases/.../cand_bilibin, label: "Bilibin"}
    """
    p = pathlib.Path(path)
    raw = _read_mapping(p)
    base = p.parent
    candidates = list(_parse_sources(raw.get("candidates", {}), base, "candidate"))
    candidates.extend(_parse_sources(raw.get("distractors", {}), base, "distractor"))
    if not candidates:
        raise ValueError(f"{p}: case spec must define at least one candidate")
    target = _optional_path(raw.get("target"), base)
    forbidden = tuple(_as_path_list(raw.get("forbidden_sources", []), base))
    feature_sets = tuple(raw.get("feature_sets") or DEFAULT_FEATURE_SETS)
    required_gates = tuple(raw.get("required_gates") or ("feasibility_gate",))
    return CaseSpec(
        case_id=str(raw.get("case_id") or p.stem),
        title=str(raw.get("title") or raw.get("case_id") or p.stem),
        candidates=tuple(candidates),
        target=target,
        hypothesis=str(raw.get("hypothesis", "")),
        target_description=str(raw.get("target_description", "")),
        claim=str(raw.get("claim", "")),
        limitations=_as_str_tuple(raw.get("limitations", [])),
        provenance=_as_mapping(raw.get("provenance", {})),
        unit=str(raw.get("unit", "work")),
        language=str(raw.get("language", "ru")),
        feature_sets=feature_sets,
        required_gates=required_gates,
        forbidden_sources=forbidden,
        sources=tuple(str(x) for x in raw.get("sources", []) or []),
        notes=str(raw.get("notes", "")),
        chunk_words=int(raw.get("chunk_words", 1500)),
        min_work_words=int(raw.get("min_work_words", 30)),
        max_exact_permutations=int(raw.get("max_exact_permutations", 400)),
        random_permutations=int(raw.get("random_permutations", 500)),
        seed=int(raw.get("seed", 42)),
    )


def run_case(spec: CaseSpec) -> CasePassport:
    """Run feasibility gates and, if target is provided, target attribution."""
    validity_failures = _validate_spec(spec)
    works, load_failures = _load_panel_works(spec)
    failures = validity_failures + load_failures
    primary_feature = spec.feature_sets[0] if spec.feature_sets else DEFAULT_FEATURE_SETS[0]

    gates: List[GateResult] = []
    attributions: List[AttributionResult] = []
    # If a declared panel member is missing/empty/too short, the panel is not the
    # panel described by the spec. Do not silently run a smaller easier case.
    if not validity_failures and not load_failures and works:
        for feature_set in spec.feature_sets:
            gates.append(_run_gate(works, spec, feature_set))
        if spec.target is not None:
            target_chunks = _load_target_chunks(spec)
            if target_chunks:
                for feature_set in spec.feature_sets:
                    attributions.append(_attribute_target(works, target_chunks, spec, feature_set))
                if len(target_chunks) < 2:
                    failures.append("target_single_chunk_no_strong_verdict")
            else:
                failures.append("target_has_no_eligible_chunks")

    primary_gate = gates[0] if gates else None
    primary_attr = attributions[0] if attributions else None
    gate_pass = bool(primary_gate and primary_gate.gate_pass)
    status, confidence, verdict = _decide(spec, primary_gate, primary_attr, failures)
    evidence_score = _evidence_score(primary_gate, primary_attr, attributions, failures)
    data = {
        "unit": spec.unit,
        "language": spec.language,
        "feature_sets": list(spec.feature_sets),
        "required_gates": list(spec.required_gates),
        "candidate_panel": [
            {"id": c.author_id, "role": c.role,
             "paths": [str(p) for p in c.all_paths()],
             "exclude": [str(p) for p in c.exclude],
             "label": c.label}
            for c in spec.candidates
        ],
        "target": str(spec.target) if spec.target else None,
        "sources": list(spec.sources),
        "n_candidate_works": len(works),
        "n_candidate_chunks": sum(len(w.chunks) for w in works),
        "works_per_author": dict(sorted(Counter(w.author for w in works).items())),
        "target_chunks": attributions[0].n_chunks if attributions else 0,
    }
    return CasePassport(
        case_id=spec.case_id,
        title=spec.title,
        status=status,
        verdict=verdict,
        confidence=confidence,
        evidence_score=evidence_score,
        gate_pass=gate_pass,
        primary_feature_set=primary_feature,
        gates=gates,
        attributions=attributions,
        failure_modes=sorted(set(failures + ([]
                                             if not primary_gate else primary_gate.failure_modes))),
        data=data,
        notes=spec.notes,
        hypothesis=spec.hypothesis,
        target_description=spec.target_description,
        claim=spec.claim,
        limitations=spec.limitations,
        provenance=spec.provenance,
    )


def write_passport(passport: CasePassport, path: str | pathlib.Path) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(passport.to_dict(), ensure_ascii=False, indent=2) + "\n",
                 encoding="utf-8")


def load_passport(path: str | pathlib.Path) -> Dict[str, Any]:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def rank_passports(passports: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Sort passports by evidence score, then by status strength."""
    status_rank = {"strong": 4, "moderate": 3, "gate_only": 2, "inconclusive": 1, "fail": 0}
    rows = [dict(p) for p in passports]
    rows.sort(key=lambda p: (
        float(p.get("evidence_score", 0.0)),
        status_rank.get(str(p.get("status", "")), 0),
        bool(p.get("gate_pass", False)),
    ), reverse=True)
    return rows


def passport_markdown(passports: Sequence[Mapping[str, Any]]) -> str:
    rows = rank_passports(passports)
    out = [
        "| score | status | confidence | case | verdict |",
        "|---:|---|---|---|---|",
    ]
    for p in rows:
        verdict = str(p.get("verdict", "")).replace("\n", " ")
        if len(verdict) > 180:
            verdict = verdict[:177] + "..."
        out.append(
            f"| {float(p.get('evidence_score', 0.0)):.2f} | "
            f"{p.get('status', '')} | {p.get('confidence', '')} | "
            f"{p.get('case_id', '')} | {verdict} |"
        )
    return "\n".join(out)


def dossier_markdown(passports: Sequence[Mapping[str, Any]]) -> str:
    """Render a publication-facing auditable dossier for one case family."""
    rows = rank_passports(passports)
    if not rows:
        return "# Case Dossier\n\nNo passports supplied."
    title = _family_title(rows)
    out = [f"# {title}", ""]
    lead = _first_nonempty(rows, "claim") or _first_nonempty(rows, "verdict")
    hypothesis = _first_nonempty(rows, "hypothesis")
    if hypothesis:
        out.extend(["## Hypothesis", "", str(hypothesis), ""])
    if lead:
        out.extend(["## Claim", "", str(lead), ""])

    out.extend([
        "## Results",
        "",
        "| target | status | score | gate | p | top | chunks | margin CI |",
        "|---|---|---:|---:|---:|---|---|---|",
    ])
    for p in rows:
        gate = _primary_gate(p)
        attr = _primary_attr(p)
        target = p.get("target_description") or p.get("case_id", "")
        if p.get("gate_pass", False):
            top = attr.get("top", "") if attr else ""
            chunks = _chunk_summary(attr)
            ci = _margin_ci(attr)
        else:
            top = "not interpreted"
            chunks = ""
            ci = ""
        pval = gate.get("permutation_p", "") if gate else ""
        gate_score = gate.get("work_macro_recall", "") if gate else ""
        out.append(
            f"| {target} | {p.get('status', '')} | {float(p.get('evidence_score', 0.0)):.2f} | "
            f"{gate_score} | {pval} | {top} | {chunks} | {ci} |"
        )
    out.append("")

    limitations = []
    for p in rows:
        limitations.extend(str(x) for x in (p.get("limitations") or []) if x)
        limitations.extend(str(x) for x in (p.get("failure_modes") or [])
                           if "single_chunk" in str(x))
    if limitations:
        out.extend(["## Limitations", ""])
        for item in sorted({_public_limitation(item) for item in limitations}):
            out.append(f"- {item}")
        out.append("")

    out.extend(["## Reproduction", ""])
    for p in rows:
        prov = p.get("provenance") or {}
        cmd = prov.get("analysis_command") if isinstance(prov, Mapping) else None
        if cmd:
            out.extend(["```bash", str(cmd), "```", ""])
    out.extend(["## Passport Verdicts", ""])
    for p in rows:
        out.append(f"- **{p.get('case_id', '')}**: {p.get('verdict', '')}")
    return "\n".join(out).rstrip() + "\n"


def _family_title(passports: Sequence[Mapping[str, Any]]) -> str:
    titles = [str(p.get("title") or p.get("case_id") or "Case Dossier") for p in passports]
    if len(titles) == 1:
        return titles[0]
    common = _common_prefix_words(titles)
    return common if common else "Case Dossier"


def _common_prefix_words(values: Sequence[str]) -> str:
    split = [v.split() for v in values if v]
    if not split:
        return ""
    prefix = []
    for words in zip(*split):
        if len(set(words)) != 1:
            break
        prefix.append(words[0])
    return " ".join(prefix).rstrip(":")


def _first_nonempty(rows: Sequence[Mapping[str, Any]], key: str) -> str:
    for row in rows:
        value = row.get(key)
        if value:
            return str(value)
    return ""


def _primary_gate(passport: Mapping[str, Any]) -> Mapping[str, Any]:
    gates = passport.get("gates") or []
    return gates[0] if gates else {}


def _primary_attr(passport: Mapping[str, Any]) -> Mapping[str, Any]:
    attrs = passport.get("attributions") or []
    return attrs[0] if attrs else {}


def _chunk_summary(attr: Mapping[str, Any]) -> str:
    if not attr:
        return ""
    winners = attr.get("per_chunk_winners") or {}
    total = attr.get("n_chunks", 0)
    parts = [f"{k} {v}/{total}" for k, v in winners.items()]
    return ", ".join(parts) if parts else str(total)


def _margin_ci(attr: Mapping[str, Any]) -> str:
    if not attr:
        return ""
    ci = attr.get("margin_ci95")
    if not ci:
        return ""
    return f"[{ci[0]}, {ci[1]}]"


def _public_limitation(item: str) -> str:
    if item == "target_single_chunk_no_strong_verdict":
        return "Single-chunk targets are diagnostic only and cannot receive a strong verdict."
    return item


def _read_mapping(path: pathlib.Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        raw = json.loads(text)
    else:
        raw = yaml.safe_load(text)
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path}: expected mapping")
    return raw


def _parse_sources(raw: Any, base: pathlib.Path, role: str) -> Iterable[CorpusSource]:
    if raw is None:
        return []
    rows = []
    if isinstance(raw, Mapping):
        for author_id, value in raw.items():
            if isinstance(value, Mapping):
                paths_raw = value.get("paths")
                if paths_raw is not None:
                    paths = _as_path_list(paths_raw, base)
                    if not paths:
                        raise ValueError(f"{author_id}.paths must not be empty")
                    primary, extra = paths[0], tuple(paths[1:])
                else:
                    primary, extra = _required_path(value.get("path"), base, f"{author_id}.path"), ()
                rows.append(CorpusSource(str(author_id), primary, paths=extra,
                                         exclude=tuple(_as_path_list(value.get("exclude", []), base)),
                                         role=role, label=_maybe_str(value.get("label"))))
            else:
                rows.append(CorpusSource(str(author_id), _required_path(value, base, str(author_id)),
                                         role=role))
        return rows
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for idx, item in enumerate(raw):
            if not isinstance(item, Mapping):
                raise ValueError(f"{role}[{idx}] must be a mapping")
            author_id = item.get("id") or item.get("author_id")
            if not author_id:
                raise ValueError(f"{role}[{idx}] missing id")
            paths_raw = item.get("paths")
            if paths_raw is not None:
                paths = _as_path_list(paths_raw, base)
                if not paths:
                    raise ValueError(f"{role}[{idx}].paths must not be empty")
                primary, extra = paths[0], tuple(paths[1:])
            else:
                primary, extra = _required_path(item.get("path"), base, str(author_id)), ()
            rows.append(CorpusSource(str(author_id), primary, paths=extra,
                                     exclude=tuple(_as_path_list(item.get("exclude", []), base)),
                                     role=role, label=_maybe_str(item.get("label"))))
        return rows
    raise ValueError(f"{role}: expected mapping or list")


def _optional_path(value: Any, base: pathlib.Path) -> Optional[pathlib.Path]:
    if value in (None, ""):
        return None
    if isinstance(value, Mapping):
        value = value.get("path")
    return _required_path(value, base, "path")


def _required_path(value: Any, base: pathlib.Path, field: str) -> pathlib.Path:
    if value in (None, ""):
        raise ValueError(f"missing path for {field}")
    p = pathlib.Path(str(value))
    return p if p.is_absolute() else (base / p)


def _as_path_list(values: Any, base: pathlib.Path) -> List[pathlib.Path]:
    if values in (None, ""):
        return []
    if isinstance(values, (str, bytes)) or isinstance(values, Mapping):
        values = [values]
    return [_required_path(v.get("path") if isinstance(v, Mapping) else v, base, "forbidden_sources")
            for v in values]


def _maybe_str(value: Any) -> Optional[str]:
    return None if value is None else str(value)


def _as_str_tuple(values: Any) -> Tuple[str, ...]:
    if values in (None, ""):
        return ()
    if isinstance(values, (str, bytes)):
        values = [values]
    if not isinstance(values, Sequence):
        values = [values]
    return tuple(str(v) for v in values)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _validate_spec(spec: CaseSpec) -> List[str]:
    failures: List[str] = []
    ids = [c.author_id for c in spec.candidates]
    if len(ids) != len(set(ids)):
        failures.append("duplicate_candidate_ids")
    if len(ids) < 2:
        failures.append("need_at_least_two_panel_authors")
    for c in spec.candidates:
        for p in c.all_paths():
            if not p.exists():
                failures.append(f"missing_candidate_path:{c.author_id}:{p}")
        for p in c.exclude:
            if not p.exists():
                failures.append(f"missing_candidate_exclude:{c.author_id}:{p}")
    if spec.target is not None and not spec.target.exists():
        failures.append("missing_target_path")
    if spec.target is not None:
        target_res = _safe_resolve(spec.target)
        for c in spec.candidates:
            for p in c.all_paths():
                cres = _safe_resolve(p)
                if cres == target_res or _is_relative_to(target_res, cres):
                    excluded = any(
                        target_res == _safe_resolve(e) or _is_relative_to(target_res, _safe_resolve(e))
                        for e in c.exclude
                    )
                    if not excluded:
                        failures.append(f"circular_target_in_candidate:{c.author_id}")
        for f in spec.forbidden_sources:
            fres = _safe_resolve(f)
            for c in spec.candidates:
                for p in c.all_paths():
                    cres = _safe_resolve(p)
                    if cres == fres or _is_relative_to(cres, fres) or _is_relative_to(fres, cres):
                        excluded = any(
                            fres == _safe_resolve(e) or _is_relative_to(fres, _safe_resolve(e))
                            for e in c.exclude
                        )
                        if not excluded:
                            failures.append(f"forbidden_source_in_panel:{c.author_id}")
    for fs in spec.feature_sets:
        if _canonical_feature(fs) not in {"fw_fixed", "char3", "char3_fw"}:
            failures.append(f"unknown_feature_set:{fs}")
    return failures


def _safe_resolve(path: pathlib.Path) -> pathlib.Path:
    try:
        return path.resolve()
    except FileNotFoundError:
        return path.absolute()


def _is_relative_to(path: pathlib.Path, parent: pathlib.Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _load_panel_works(spec: CaseSpec) -> Tuple[List[Work], List[str]]:
    works: List[Work] = []
    failures: List[str] = []
    for src in spec.candidates:
        docs = _read_text_docs(src.all_paths(), exclude=src.exclude)
        if not docs:
            failures.append(f"empty_candidate:{src.author_id}")
            continue
        for work_id, text in docs:
            if _word_count(text) < spec.min_work_words:
                continue
            chunks = tuple(_chunk_words(text, spec.chunk_words))
            if not chunks:
                continue
            works.append(Work(src.author_id, work_id, text, chunks))
        if not any(w.author == src.author_id for w in works):
            failures.append(f"no_eligible_works:{src.author_id}")
    counts = Counter(w.author for w in works)
    for author_id in sorted({c.author_id for c in spec.candidates}):
        if counts[author_id] < 2:
            failures.append(f"gate_uncomputable_lt2_works:{author_id}")
    return works, failures


def _load_target_chunks(spec: CaseSpec) -> List[str]:
    if spec.target is None:
        return []
    chunks: List[str] = []
    for _wid, text in _read_text_docs(spec.target):
        if _word_count(text) < spec.min_work_words:
            continue
        chunks.extend(_chunk_words(text, spec.chunk_words))
    return chunks


def _read_text_docs(
    path: pathlib.Path | Sequence[pathlib.Path],
    exclude: Sequence[pathlib.Path] = (),
) -> List[Tuple[str, str]]:
    paths = [path] if isinstance(path, pathlib.Path) else list(path)
    excluded = [_safe_resolve(p) for p in exclude]
    docs: List[Tuple[str, str]] = []
    for base in paths:
        if base.is_file():
            if _path_is_excluded(base, excluded):
                continue
            docs.append((base.stem, base.read_text(encoding="utf-8", errors="ignore")))
        elif base.is_dir():
            files = sorted(p for p in base.rglob("*.txt") if p.is_file())
            prefix = base.name
            for p in files:
                if _path_is_excluded(p, excluded):
                    continue
                work_id = f"{prefix}/{p.relative_to(base).with_suffix('')}"
                docs.append((work_id, p.read_text(encoding="utf-8", errors="ignore")))
    return docs


def _path_is_excluded(path: pathlib.Path, excluded: Sequence[pathlib.Path]) -> bool:
    resolved = _safe_resolve(path)
    for ex in excluded:
        if resolved == ex or _is_relative_to(resolved, ex):
            return True
    return False


def _word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def _chunk_words(text: str, chunk_words: int) -> List[str]:
    words = WORD_RE.findall(text)
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]
    chunks = []
    for start in range(0, len(words), chunk_words):
        piece = words[start:start + chunk_words]
        if piece:
            chunks.append(" ".join(piece))
    return chunks


def _run_gate(works: List[Work], spec: CaseSpec, feature_set: str) -> GateResult:
    authors = sorted({w.author for w in works})
    failure_modes: List[str] = []
    if any(sum(1 for w in works if w.author == a) < 2 for a in authors):
        failure_modes.append("gate_requires_at_least_two_works_per_author")
        return GateResult(feature_set, "uncomputable", False, 0.0, 0.0, {}, {}, None, None,
                          None, {}, len(works), sum(len(w.chunks) for w in works), failure_modes)

    wcp = _leave_one_work_out(works, [w.author for w in works], feature_set, spec.language)
    metrics = _metrics_from_wcp(wcp, authors)
    p, method, floor = _permutation_p(
        works, [w.author for w in works], feature_set, spec.language,
        max_exact=spec.max_exact_permutations,
        n_random=spec.random_permutations,
        seed=spec.seed,
    )
    # Толеранс покрывает float-суммирование recall-ов: панель с точным средним
    # 0.80 (например, 0.9+0.9+0.75+0.65) не должна падать из-за 0.7999999....
    _GATE_EPS = 1e-9
    gate_ok = metrics["work_macro_recall"] >= 0.80 - _GATE_EPS
    gate_pass = gate_ok and (p is not None and p <= 0.05)
    if not gate_ok:
        failure_modes.append("work_macro_recall_below_0_80")
    if p is None or p > 0.05:
        failure_modes.append("work_permutation_not_significant")
    status = "pass" if gate_pass else "fail"
    return GateResult(
        feature_set=feature_set,
        status=status,
        gate_pass=gate_pass,
        work_macro_recall=round(metrics["work_macro_recall"], 4),
        chunk_weighted_recall=round(metrics["chunk_weighted_recall"], 4),
        work_recall=metrics["work_recall"],
        chunk_recall=metrics["chunk_recall"],
        permutation_p=p,
        permutation_method=method,
        permutation_exact_floor=floor,
        confusion=metrics["confusion"],
        n_works=len(works),
        n_chunks=sum(len(w.chunks) for w in works),
        failure_modes=failure_modes,
    )


def _leave_one_work_out(
    works: List[Work],
    labels: Sequence[str],
    feature_set: str,
    language: str,
) -> List[Tuple[str, str, List[str]]]:
    if _canonical_feature(feature_set) == "fw_fixed":
        return _leave_one_work_out_fw_fixed(works, labels, language)
    out: List[Tuple[str, str, List[str]]] = []
    labels = list(labels)
    for i, work in enumerate(works):
        train_idx = [j for j in range(len(works)) if j != i]
        train_labels = [labels[j] for j in train_idx]
        if labels[i] not in train_labels:
            continue
        if len(set(train_labels)) < 2:
            continue
        train_texts = [ch for j in train_idx for ch in works[j].chunks]
        train_chunk_labels = [labels[j] for j in train_idx for _ch in works[j].chunks]
        test_texts = list(work.chunks)
        Xtr, Xte = _fit_transform_feature(train_texts, test_texts, feature_set, language)
        preds = _predict_by_centroid(Xtr, train_chunk_labels, Xte)
        out.append((labels[i], work.work_id, preds))
    return out


def _leave_one_work_out_fw_fixed(
    works: List[Work],
    labels: Sequence[str],
    language: str,
) -> List[Tuple[str, str, List[str]]]:
    """Fast LOO path for fixed function words.

    The vocabulary is closed and language-level, so vectorizing all chunks once
    does not fit anything on the held-out work. Only author centroids are rebuilt
    per fold/per permutation.
    """
    return _leave_one_work_out_fw_context(works, labels, _build_fw_context(works, language))


def _build_fw_context(works: List[Work], language: str) -> FwContext:
    chunk_texts: List[str] = []
    chunk_work = []
    for wi, work in enumerate(works):
        for ch in work.chunks:
            chunk_texts.append(ch)
            chunk_work.append(wi)
    if not chunk_texts:
        return FwContext(X=None, chunk_work=np.array([], dtype=int))
    vec = CountVectorizer(vocabulary=sorted(function_words(language)), lowercase=True,
                          token_pattern=r"(?u)\b[\w\-]+\b")
    X = normalize(vec.fit_transform(chunk_texts), norm="l2")
    return FwContext(X=X, chunk_work=np.asarray(chunk_work, dtype=int))


def _leave_one_work_out_fw_context(
    works: List[Work],
    labels: Sequence[str],
    ctx: FwContext,
) -> List[Tuple[str, str, List[str]]]:
    labels = list(labels)
    if ctx.X is None or len(ctx.chunk_work) == 0:
        return []
    chunk_work_arr = ctx.chunk_work
    labels_arr = np.asarray(labels, dtype=object)
    chunk_labels = labels_arr[chunk_work_arr]
    out: List[Tuple[str, str, List[str]]] = []
    for wi, work in enumerate(works):
        train_mask = chunk_work_arr != wi
        train_labels = labels_arr[[j for j in range(len(works)) if j != wi]]
        if labels[wi] not in set(train_labels.tolist()):
            continue
        if len(set(train_labels.tolist())) < 2:
            continue
        test_mask = chunk_work_arr == wi
        preds = _predict_by_centroid(ctx.X[train_mask], chunk_labels[train_mask], ctx.X[test_mask])
        out.append((labels[wi], work.work_id, preds))
    return out


def _metrics_from_wcp(
    wcp: Sequence[Tuple[str, str, List[str]]],
    authors: Sequence[str],
) -> Dict[str, Any]:
    cw_c = {a: 0 for a in authors}
    cw_t = {a: 0 for a in authors}
    wl_c = {a: 0 for a in authors}
    wl_t = {a: 0 for a in authors}
    conf = {a: {b: 0 for b in authors} for a in authors}
    for truth, _work_id, preds in wcp:
        if not preds:
            continue
        for pred in preds:
            cw_c[truth] += pred == truth
            cw_t[truth] += 1
            if pred in conf[truth]:
                conf[truth][pred] += 1
        top = Counter(preds).most_common(1)[0][0]
        wl_c[truth] += top == truth
        wl_t[truth] += 1
    wl_vals = [wl_c[a] / wl_t[a] for a in authors if wl_t[a]]
    cw_vals = [cw_c[a] / cw_t[a] for a in authors if cw_t[a]]
    return {
        "work_macro_recall": float(np.mean(wl_vals)) if wl_vals else 0.0,
        "chunk_weighted_recall": float(np.mean(cw_vals)) if cw_vals else 0.0,
        "work_recall": {a: f"{wl_c[a]}/{wl_t[a]}" for a in authors},
        "chunk_recall": {a: f"{cw_c[a]}/{cw_t[a]}" for a in authors},
        "confusion": conf,
    }


def _permutation_p(
    works: List[Work],
    labels: Sequence[str],
    feature_set: str,
    language: str,
    max_exact: int,
    n_random: int,
    seed: int,
) -> Tuple[Optional[float], Optional[str], Optional[float]]:
    labels = list(labels)
    fw_ctx = _build_fw_context(works, language) if _canonical_feature(feature_set) == "fw_fixed" else None

    def _wcp_for(lab: Sequence[str]):
        if fw_ctx is not None:
            return _leave_one_work_out_fw_context(works, lab, fw_ctx)
        return _leave_one_work_out(works, lab, feature_set, language)

    observed = _metrics_from_wcp(_wcp_for(labels), sorted(set(labels)))["work_macro_recall"]
    uniq = sorted(set(labels))
    counts = Counter(labels)
    if len(uniq) < 2:
        return None, None, None
    assignments: List[List[str]]
    floor = None
    method = None
    if len(uniq) == 2:
        n0 = counts[uniq[0]]
        total = math.comb(len(labels), n0)
        floor = round(1.0 / total, 5)
        if total <= max_exact:
            assignments = []
            idx = range(len(labels))
            for combo in itertools.combinations(idx, n0):
                s = set(combo)
                assignments.append([uniq[0] if i in s else uniq[1] for i in idx])
            method = f"exact_{len(assignments)}"
        else:
            assignments = _random_assignments(labels, n_random, seed)
            method = f"random_{n_random}"
    else:
        assignments = _random_assignments(labels, n_random, seed)
        method = f"random_{n_random}"
    ge = 0
    for lab in assignments:
        score = _metrics_from_wcp(_wcp_for(lab), sorted(set(lab)))["work_macro_recall"]
        if score >= observed - 1e-12:
            ge += 1
    if method and method.startswith("exact_"):
        p = ge / len(assignments)
    else:
        p = (ge + 1) / (len(assignments) + 1)
    return round(float(p), 4), method, floor


def _random_assignments(labels: Sequence[str], n: int, seed: int) -> List[List[str]]:
    rng = np.random.default_rng(seed)
    labels_arr = np.array(list(labels), dtype=object)
    return [list(rng.permutation(labels_arr)) for _ in range(max(1, n))]


def _fit_transform_feature(
    train_texts: Sequence[str],
    test_texts: Sequence[str],
    feature_set: str,
    language: str,
):
    fs = _canonical_feature(feature_set)
    if fs == "fw_fixed":
        vec = CountVectorizer(vocabulary=sorted(function_words(language)), lowercase=True,
                              token_pattern=r"(?u)\b[\w\-]+\b")
        Xtr = vec.fit_transform(train_texts)
        Xte = vec.transform(test_texts)
        return normalize(Xtr, norm="l2"), normalize(Xte, norm="l2")
    if fs == "char3":
        vec = CountVectorizer(analyzer="char_wb", ngram_range=(3, 3), max_features=800,
                              lowercase=True)
        tf = TfidfTransformer(sublinear_tf=True)
        Xtr = tf.fit_transform(vec.fit_transform(train_texts))
        Xte = tf.transform(vec.transform(test_texts))
        return normalize(Xtr, norm="l2"), normalize(Xte, norm="l2")
    if fs == "char3_fw":
        Xtr_fw, Xte_fw = _fit_transform_feature(train_texts, test_texts, "fw_fixed", language)
        Xtr_ch, Xte_ch = _fit_transform_feature(train_texts, test_texts, "char3", language)
        from scipy.sparse import hstack
        return normalize(hstack([Xtr_fw, Xtr_ch]), norm="l2"), normalize(hstack([Xte_fw, Xte_ch]), norm="l2")
    raise ValueError(f"unknown feature set: {feature_set}")


def _canonical_feature(feature_set: str) -> str:
    fs = feature_set.lower().strip()
    aliases = {
        "function_words": "fw_fixed",
        "fw": "fw_fixed",
        "topic_strict": "fw_fixed",
        "char": "char3",
        "char_3gram": "char3",
        "char3+fw": "char3_fw",
        "fw+char3": "char3_fw",
    }
    return aliases.get(fs, fs)


def _predict_by_centroid(Xtr, ytr: Sequence[str], Xte) -> List[str]:
    authors = sorted(set(ytr))
    centroids = []
    yarr = np.array(list(ytr), dtype=object)
    for a in authors:
        rows = Xtr[yarr == a]
        c = np.asarray(rows.mean(axis=0)).ravel()
        norm = np.linalg.norm(c)
        centroids.append(c / (norm + 1e-12))
    C = np.vstack(centroids)
    scores = Xte @ C.T
    scores = np.asarray(scores)
    return [authors[int(i)] for i in scores.argmax(axis=1)]


def _attribute_target(
    works: List[Work],
    target_chunks: Sequence[str],
    spec: CaseSpec,
    feature_set: str,
) -> AttributionResult:
    train_texts = [ch for w in works for ch in w.chunks]
    train_labels = [w.author for w in works for _ch in w.chunks]
    Xtr, Xte = _fit_transform_feature(train_texts, target_chunks, feature_set, spec.language)
    authors = sorted(set(train_labels))
    yarr = np.array(train_labels, dtype=object)
    centroids = []
    for a in authors:
        c = np.asarray(Xtr[yarr == a].mean(axis=0)).ravel()
        centroids.append(c / (np.linalg.norm(c) + 1e-12))
    C = np.vstack(centroids)
    target_vec = np.asarray(Xte.mean(axis=0)).ravel()
    target_vec = target_vec / (np.linalg.norm(target_vec) + 1e-12)
    sims_arr = C @ target_vec
    order = np.argsort(-sims_arr, kind="stable")
    similarities = [{"candidate": authors[int(i)], "cos": round(float(sims_arr[int(i)]), 6)}
                    for i in order]
    preds = [authors[int(i)] for i in np.asarray(Xte @ C.T).argmax(axis=1)]
    counts = Counter(preds)
    margin = None
    margin_ci = None
    top = similarities[0]["candidate"] if similarities else None
    second = similarities[1]["candidate"] if len(similarities) > 1 else None
    if len(similarities) > 1:
        margin = round(float(similarities[0]["cos"] - similarities[1]["cos"]), 6)
        margin_ci = _bootstrap_margin(Xte, C, authors, spec.seed)
    return AttributionResult(
        feature_set=feature_set,
        top=top,
        second=second,
        margin=margin,
        margin_ci95=margin_ci,
        winner_share={k: round(v / len(preds), 3) for k, v in counts.most_common()} if preds else {},
        similarities=similarities,
        per_chunk_winners=dict(counts.most_common()),
        n_chunks=len(target_chunks),
    )


def _bootstrap_margin(Xte, C: np.ndarray, authors: Sequence[str], seed: int, n_iter: int = 1000):
    if Xte.shape[0] < 2 or len(authors) < 2:
        return None
    rng = np.random.default_rng(seed)
    margins = []
    for _ in range(n_iter):
        idx = rng.integers(0, Xte.shape[0], size=Xte.shape[0])
        v = np.asarray(Xte[idx].mean(axis=0)).ravel()
        v = v / (np.linalg.norm(v) + 1e-12)
        s = C @ v
        order = np.argsort(-s, kind="stable")
        margins.append(float(s[order[0]] - s[order[1]]))
    lo, hi = np.quantile(margins, [0.025, 0.975])
    return (round(float(lo), 6), round(float(hi), 6))


def _decide(
    spec: CaseSpec,
    gate: Optional[GateResult],
    attr: Optional[AttributionResult],
    failures: Sequence[str],
) -> Tuple[str, str, str]:
    if failures and not gate:
        return "fail", "low", "Кейс не вычислим: " + ", ".join(sorted(set(failures)))
    if gate is None:
        return "fail", "low", "Нет вычислимого feasibility gate."
    if not gate.gate_pass:
        return (
            "fail",
            "low",
            f"Feasibility gate не пройден: work_macro_recall={gate.work_macro_recall}, "
            f"permutation_p={gate.permutation_p}. Атрибуцию давать нельзя.",
        )
    if spec.target is None:
        return (
            "gate_only",
            "moderate",
            f"Feasibility gate пройден ({gate.feature_set}): work_macro_recall="
            f"{gate.work_macro_recall}, permutation_p={gate.permutation_p}.",
        )
    if attr is None or attr.top is None:
        return "inconclusive", "low", "Gate пройден, но target attribution не вычислена."
    win_share = attr.winner_share.get(attr.top, 0.0)
    if attr.n_chunks < 2:
        if attr.margin is not None and attr.margin > 0:
            return (
                "moderate",
                "medium",
                f"Gate пройден; target из одного chunk диагностически ближе к {attr.top}, "
                f"но single-chunk target не может получить strong verdict "
                f"(winner_share={win_share}, margin={attr.margin}).",
            )
        return (
            "inconclusive",
            "low",
            f"Gate пройден, но single-chunk target неустойчив: top={attr.top}, "
            f"winner_share={win_share}, margin={attr.margin}.",
        )
    ci_ok = attr.margin_ci95 is not None and attr.margin_ci95[0] > 0
    if win_share >= 0.90 and (attr.margin is not None and attr.margin > 0) and ci_ok:
        return (
            "strong",
            "high",
            f"Gate пройден; target устойчиво ближе к {attr.top} "
            f"(winner_share={win_share}, margin={attr.margin}).",
        )
    if win_share >= 0.65 and attr.margin is not None and attr.margin > 0:
        return (
            "moderate",
            "medium",
            f"Gate пройден; target чаще ближе к {attr.top}, но запас/бутстрап недостаточны "
            f"для strong verdict (winner_share={win_share}, margin={attr.margin}).",
        )
    return (
        "inconclusive",
        "low",
        f"Gate пройден, но target attribution неустойчива: top={attr.top}, "
        f"winner_share={win_share}, margin={attr.margin}.",
    )


def _evidence_score(
    gate: Optional[GateResult],
    attr: Optional[AttributionResult],
    attrs: Sequence[AttributionResult],
    failures: Sequence[str],
) -> float:
    if gate is None:
        return 0.0
    recall = max(0.0, min(1.0, gate.work_macro_recall))
    if gate.permutation_p is None:
        sig = 0.0
    else:
        sig = max(0.0, min(1.0, -math.log10(max(gate.permutation_p, 1e-12)) / 3.0))
    if attr is None or attr.top is None:
        consistency = 0.0
        margin = 0.0
    else:
        tops = [a.top for a in attrs if a.top]
        consistency = tops.count(attr.top) / len(tops) if tops else 0.0
        margin = max(0.0, min(1.0, (attr.margin or 0.0) / 0.05))
    penalty = min(0.3, 0.05 * len(set(failures)))
    score = 100.0 * (0.45 * recall + 0.25 * sig + 0.20 * consistency + 0.10 * margin - penalty)
    return max(0.0, min(100.0, score))


def _attr_to_dict(a: AttributionResult) -> Dict[str, Any]:
    out = dataclasses.asdict(a)
    if a.margin_ci95 is not None:
        out["margin_ci95"] = list(a.margin_ci95)
    return out
