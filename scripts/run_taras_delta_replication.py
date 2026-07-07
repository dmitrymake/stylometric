"""Method-independent replication of the Taras Bulba additions attribution.

Burrows Delta (Manhattan over z-scored frequencies — a different classifier
family than the fw_fixed cosine-centroid channel) in two modes:
  - delta_fw   — vocabulary fixed to function words (theme-neutral);
  - delta_mfw  — classic top-300 MFW of the train corpus.

Panel mirrors the suspects spec: Gogol anchor (without Taras Bulba),
Annenkov 1840s prose, Pushkin, Turgenev, Dostoevsky. Case texts are taken
from input_cases/taras_bulba/masked/ (symmetric NER masking) when present.

Protocol per mode: leave-one-work-out gate (held-out work classified by the
majority of its chunks, one work = one vote), exact-or-random work-label
permutation p, then target attribution by chunk majority with the margin
between the two closest authors.

Output: docs/cases/taras_hardened/reports/delta_replication.json
Run: .venv/bin/python scripts/run_taras_delta_replication.py
"""
from __future__ import annotations

import json
import pathlib
import random
import sys
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stylo.lang import function_words  # noqa: E402
from stylo.models.delta import BurrowsDelta  # noqa: E402

IC = ROOT / "input_clean"
CASE = ROOT / "input_cases" / "taras_bulba"
MASKED = CASE / "masked"
OUT = ROOT / "docs" / "cases" / "taras_hardened" / "reports" / "delta_replication.json"

WIN = 1500
MFW = 300
GATE_THRESHOLD = 0.80
N_PERM = 2000
SEED = 42
FW = sorted(function_words("ru"))


def masked_path(rel: str) -> pathlib.Path:
    p = MASKED / rel
    return p if p.exists() else CASE / rel

_GOGOL = [p for p in sorted((IC / "gogol").glob("*.txt"))
          if p.name != "тарас_бульба.txt"]

PANELS: Dict[str, Dict[str, List[pathlib.Path]]] = {
    "suspects": {
        "gogol": _GOGOL,
        "annenkov_1840s": sorted(masked_path("cand_annenkov_1840s").glob("*.txt")),
        "pushkin": sorted((IC / "pushkin").glob("*.txt")),
        "turgenev": sorted((IC / "turgenev").glob("*.txt")),
        "dostoevsky": sorted((IC / "dostoevsky").glob("*.txt")),
    },
    "somov_binary": {
        "gogol": _GOGOL,
        "somov": sorted(masked_path("cand_somov").glob("*.txt")),
    },
    "topic": {
        "gogol": _GOGOL,
        "somov": sorted(masked_path("cand_somov").glob("*.txt")),
        "narezhny": sorted(masked_path("cand_narezhny").glob("*.txt")),
        "grebenka": sorted(masked_path("cand_grebenka").glob("*.txt")),
    },
}

TARGETS = {
    "strict_additions": masked_path("dobavleniya1842_strict.txt"),
    "loose_additions": masked_path("dobavleniya1842_loose.txt"),
    "gogol1835_base_control": masked_path("gogol1835_mirgorod.txt"),
}


def chunk_words(text: str, win: int = WIN) -> List[str]:
    w = text.split()
    return [" ".join(w[i:i + win]) for i in range(0, len(w), win)
            if len(w[i:i + win]) >= win // 2]


def load_panel(panel: Dict[str, List[pathlib.Path]]) -> List[Tuple[str, str, str]]:
    """[(author, work, chunk_text)]"""
    rows = []
    for author, files in panel.items():
        for f in files:
            for ch in chunk_words(f.read_text("utf-8", "ignore")):
                rows.append((author, f.stem, ch))
    return rows


def make_model(mode: str) -> BurrowsDelta:
    if mode == "delta_fw":
        return BurrowsDelta(metric="manhattan", vocabulary=FW)
    return BurrowsDelta(mfw_count=MFW, metric="manhattan")


def work_loo(rows, mode: str) -> Tuple[float, Dict[str, str], List[Tuple[str, str]]]:
    works = sorted({(a, w) for a, w, _ in rows})
    votes: List[Tuple[str, str]] = []  # (true_author, predicted)
    for author, work in works:
        train = [(a, c) for a, w, c in rows if w != work]
        test = [c for a, w, c in rows if w == work]
        model = make_model(mode)
        model.fit([c for _, c in train], [a for a, _ in train])
        preds = model.predict(test)
        top = Counter(preds).most_common(1)[0][0]
        votes.append((author, top))
    per_author: Dict[str, List[int]] = {}
    for true, pred in votes:
        per_author.setdefault(true, []).append(int(true == pred))
    recalls = {a: sum(v) / len(v) for a, v in per_author.items()}
    macro = sum(recalls.values()) / len(recalls)
    detail = {a: f"{sum(v)}/{len(v)}" for a, v in per_author.items()}
    return macro, detail, votes


def permutation_p(votes: List[Tuple[str, str]], macro: float, n: int = N_PERM) -> float:
    """Random work-label permutation of the observed macro recall."""
    rng = random.Random(SEED)
    trues = [t for t, _ in votes]
    preds = [p for _, p in votes]
    hits = 0
    for _ in range(n):
        shuffled = trues[:]
        rng.shuffle(shuffled)
        per: Dict[str, List[int]] = {}
        for t, p in zip(shuffled, preds):
            per.setdefault(t, []).append(int(t == p))
        m = sum(sum(v) / len(v) for v in per.values()) / len(per)
        if m >= macro:
            hits += 1
    return (hits + 1) / (n + 1)


def attribute(rows, mode: str, target_text: str) -> Dict[str, object]:
    model = make_model(mode)
    model.fit([c for _, _, c in rows], [a for a, _, _ in rows])
    chunks = chunk_words(target_text)
    if not chunks:
        chunks = [target_text]
    counts = Counter(model.predict(chunks))
    top = counts.most_common(1)[0][0]
    # Margin between the two closest authors by mean Delta distance.
    mean_d = model.distances(chunks).mean(axis=0)
    order = np.argsort(mean_d)
    return {
        "top": top,
        "per_chunk_winners": dict(counts),
        "n_chunks": len(chunks),
        "winner_share": round(counts[top] / len(chunks), 3),
        "mean_distances": {str(model.classes_[i]): round(float(mean_d[i]), 6)
                           for i in order},
        "margin": round(float(mean_d[order[1]] - mean_d[order[0]]), 6),
    }


def main() -> int:
    report = {
        "case_id": "taras_bulba_1842_additions",
        "protocol": {
            "unit": "work", "chunk_words": WIN, "gate_threshold": GATE_THRESHOLD,
            "permutation": f"random_{N_PERM}", "seed": SEED,
            "masked_inputs": MASKED.exists(),
        },
        "panels": {},
    }
    for panel_name, panel in PANELS.items():
        rows = load_panel(panel)
        panel_entry = {"composition": {a: len(fs) for a, fs in panel.items()},
                       "modes": {}}
        for mode in ("delta_fw", "delta_mfw"):
            macro, detail, votes = work_loo(rows, mode)
            p = permutation_p(votes, macro)
            entry = {
                "gate": {
                    "work_macro_recall": round(macro, 4),
                    "gate_pass": macro >= GATE_THRESHOLD - 1e-9,
                    "work_recall": detail,
                    "permutation_p": round(p, 4),
                },
                "targets": {},
            }
            for name, path in TARGETS.items():
                attr = attribute(rows, mode, path.read_text("utf-8", "ignore"))
                attr["interpreted"] = bool(macro >= GATE_THRESHOLD - 1e-9)
                entry["targets"][name] = attr
            panel_entry["modes"][mode] = entry
            print(f"[{panel_name}/{mode}] gate={macro:.4f} p={p:.4f} " +
                  " ".join(f"{n}:{t['top']}({t['winner_share']})"
                           for n, t in entry["targets"].items()), flush=True)
        report["panels"][panel_name] = panel_entry
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
