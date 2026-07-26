#!/usr/bin/env python3
"""Paired adversarial rerun of headline cases under two centroid estimands.

The historical framework scored held-out works with one vote each but formed
training centroids by pooling every chunk.  This audit changes only that
train-side estimand and runs both policies through the same current code:

``chunk_weighted_legacy``
    Mean all chunks of an author; long works have more influence.

``work_balanced``
    L2(mean chunks per work), then mean works per author; every work has one
    equal directional vote.

Outputs are deliberately separate from the historical passports.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
from typing import Any

from stylo.cases import load_case_spec, run_case, write_passport
from stylo.jsonio import dump_strict, dumps_strict  # noqa: E402


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "cases" / "work_balanced_audit"
WEIGHTINGS = ("chunk_weighted_legacy", "work_balanced")

CASE_SPECS = (
    "docs/cases/autoscreen/specs/kolokol_herzen_ogaryov.yaml",
    "docs/cases/autoscreen/specs/sovremennik_fourway_gate.yaml",
    "docs/cases/autoscreen/specs/chekhonte_budilnik_sredi_milykh.yaml",
    "docs/cases/taras_hardened/specs/taras_bulba_additions_strict_annenkov_binary_fw_2000.yaml",
    "docs/cases/taras_hardened/specs/taras_bulba_additions_strict_somov_binary_fw_2000.yaml",
    "docs/cases/taras_hardened/specs/taras_bulba_additions_strict_suspects_v2_fw_2000.yaml",
    "docs/cases/taras_hardened/specs/taras_bulba_additions_strict_sameperiod_fw_2000.yaml",
    "docs/cases/taras_hardened/specs/taras_bulba_additions_strict_topic_cossack_fw_2000.yaml",
    "docs/cases/taras_hardened/specs/taras_bulba_additions_loose_suspects_v2_fw_2000.yaml",
    "docs/cases/taras_hardened/specs/taras_bulba_additions_loose_sameperiod_fw_2000.yaml",
    "docs/cases/taras_hardened/specs/taras_bulba_additions_loose_topic_cossack_fw_2000.yaml",
    "docs/cases/petersburg_hardened/specs/petersburg_nn_fourway_fw_2000.yaml",
    "docs/cases/petersburg_hardened/specs/petersburg_fd_1847_04_27_fourway_fw_2000.yaml",
    "docs/cases/petersburg_hardened/specs/petersburg_fd_1847_05_11_fourway_fw_2000.yaml",
    "docs/cases/petersburg_hardened/specs/petersburg_fd_1847_06_01_fourway_fw_2000.yaml",
    "docs/cases/petersburg_hardened/specs/petersburg_fd_1847_06_15_fourway_fw_2000.yaml",
)


def _primary(result: dict[str, Any]) -> dict[str, Any]:
    gate = (result.get("gates") or [{}])[0]
    attribution = (result.get("attributions") or [{}])[0]
    return {
        "status": result["status"],
        "required_gates_pass": result["gate_pass"],
        "panel_feasibility_gate_pass": gate.get("gate_pass", False),
        "work_macro_recall": gate.get("work_macro_recall"),
        "permutation_p": gate.get("permutation_p"),
        "top": attribution.get("top"),
        "diagnostic_closed_set_top": attribution.get(
            "diagnostic_closed_set_top"
        ),
        "abstained": attribution.get("abstained", True),
        "winner_share": attribution.get("winner_share", {}),
        "margin": attribution.get("margin"),
        "n_chunks": attribution.get("n_chunks", 0),
    }


def _paired_row(source_spec: str, results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    legacy = _primary(results["chunk_weighted_legacy"])
    balanced = _primary(results["work_balanced"])
    legacy_gate = legacy["work_macro_recall"]
    balanced_gate = balanced["work_macro_recall"]
    delta = None
    if legacy_gate is not None and balanced_gate is not None:
        delta = round(float(balanced_gate) - float(legacy_gate), 4)
    return {
        "source_spec": source_spec,
        "legacy": legacy,
        "work_balanced": balanced,
        "gate_delta_work_balanced_minus_legacy": delta,
        "panel_gate_decision_changed": (
            legacy["panel_feasibility_gate_pass"]
            != balanced["panel_feasibility_gate_pass"]
        ),
        "diagnostic_target_top_changed": (
            legacy["diagnostic_closed_set_top"]
            != balanced["diagnostic_closed_set_top"]
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Work-balanced centroid adversarial audit",
        "",
        "Paired rerun under identical features, chunks, labels, seeds, and target texts. "
        "Only the train-side author centroid changes.",
        "The registered feasibility threshold is 0.80; a score below it fails even when "
        "the permutation p-value is small.",
        "",
        "| case | legacy gate | work-balanced gate | delta | legacy p | balanced p | legacy diagnostic top | balanced diagnostic top |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in report["cases"]:
        legacy = row["legacy"]
        balanced = row["work_balanced"]
        lines.append(
            "| {case} | {lg} | {bg} | {delta} | {lp} | {bp} | {lt} | {bt} |".format(
                case=pathlib.Path(row["source_spec"]).stem,
                lg=legacy["work_macro_recall"],
                bg=balanced["work_macro_recall"],
                delta=row["gate_delta_work_balanced_minus_legacy"],
                lp=legacy["permutation_p"],
                bp=balanced["permutation_p"],
                lt=legacy["diagnostic_closed_set_top"] or "",
                bt=balanced["diagnostic_closed_set_top"] or "",
            )
        )
    lines.extend(
        [
            "",
            "All target labels in this table are withdrawn closed-set diagnostics. "
            "A feasibility pass does not establish target applicability; current v2 "
            "passports abstain until a calibrated open-set gate exists. Historical "
            "passports remain available only as withdrawn legacy artifacts.",
            "",
            "The historical bespoke Kolokol panel is audited separately because its corpus and "
            "600-word window differ from the framework spec. Under work-balanced function-word "
            "centroids it falls from the historical 0.8667 (p=0.0015) to 0.6857 (p=0.0755), "
            "so that feasibility claim is withdrawn. See "
            "`custom/kolokol_herzen_ogaryov.work_balanced.json`.",
            "",
            "The historical Taras Delta report also used an invalid fixed-prediction null. "
            "Its corrected equal-work/full-refit rerun is exploratory: suspects passes in "
            "both modes and points to Gogol, but the separable Gogol-Somov binary reverses "
            "from Gogol under fixed function words to Somov under learned MFW. The fixed-FW "
            "topic panel fails its gate. Cross-feature/panel robustness is therefore absent. "
            "See `custom/taras_delta_full_refit_work_balanced.json`.",
            "",
            "Supplemental bespoke reruns complete the public limits map. The Sovremennik "
            "school axis survives at 1.000 (p=0.0005), while the "
            "Chernyshevsky-Dobrolyubov pair falls to 0.700 (p=0.2222). "
            "Nekrasov-Panaeva function words fall to 0.650 (p=0.2273), whereas "
            "the content-sensitive char-3gram channel remains 0.950 (p=0.0152). "
            "See `custom/sovremennik.work_balanced.json` and "
            "`custom/nekrasov_panaeva.work_balanced.json`.",
            "",
            "Frozen summary/custom artifact hashes are listed in `SHA256SUMS`.",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    out: pathlib.Path,
    *,
    min_random_permutations: int,
    force_random_permutations: int | None = None,
) -> dict[str, Any]:
    passports = out / "passports"
    passports.mkdir(parents=True, exist_ok=True)
    rows = []
    for relative in CASE_SPECS:
        source_path = ROOT / relative
        source = load_case_spec(source_path)
        paired: dict[str, dict[str, Any]] = {}
        n_random = (
            force_random_permutations
            if force_random_permutations is not None
            else max(source.random_permutations, min_random_permutations)
        )
        for weighting in WEIGHTINGS:
            current = dataclasses.replace(
                source,
                case_id=f"{source.case_id}__{weighting}",
                centroid_weighting=weighting,
                random_permutations=n_random,
            )
            passport = run_case(current)
            destination = passports / f"{source.case_id}__{weighting}.passport.json"
            write_passport(passport, destination)
            paired[weighting] = passport.to_dict()
        rows.append(_paired_row(relative, paired))

    report = {
        "audit": "work_balanced_train_centroid_v2_fail_closed_targets",
        "status": "exploratory_adversarial_rerun",
        "date": "2026-07-11",
        "estimands": {
            "chunk_weighted_legacy": "L2(mean all author chunks)",
            "work_balanced": "L2(mean over L2(mean chunks within each work))",
        },
        "minimum_random_permutations": min_random_permutations,
        "forced_random_permutations": force_random_permutations,
        "n_cases": len(rows),
        "cases": rows,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        dumps_strict(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out / "README.md").write_text(_markdown(report), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--min-random-permutations", type=int, default=2_000)
    parser.add_argument(
        "--force-random-permutations",
        type=int,
        default=None,
        help="Testing-only exact override; omit for the registered >=2000 audit.",
    )
    args = parser.parse_args()
    report = run(
        args.out,
        min_random_permutations=args.min_random_permutations,
        force_random_permutations=args.force_random_permutations,
    )
    print(dumps_strict(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
