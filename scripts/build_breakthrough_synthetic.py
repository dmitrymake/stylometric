#!/usr/bin/env python3
"""Build a deterministic integration-only SPOOF/IDIOSHIFT benchmark package.

The generated corpus is deliberately synthetic and may only be used to test
schemas, split logic, models, and scorers.  It is not scientific evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "src"))
from stylo.jsonio import dump_strict, dumps_strict  # noqa: E402


AUTHORS = {
    "author_a": ["ибо", "однако", "который", "между", "тихо", "ровно"],
    "author_b": ["ведь", "словно", "чтобы", "около", "резко", "громко"],
    "author_c": ["уж", "зато", "когда", "внутри", "медленно", "сухо"],
}
SOURCES = {
    "scan": ["ocrскан", "твёрдыйзнак", "страница"],
    "web": ["htmlweb", "абзац", "ссылка"],
    "epub": ["epubbook", "глава", "раздел"],
}
EDITIONS = {
    "critical": ["примечание", "вариант", "редакция"],
    "popular": ["собрание", "текст", "издание"],
}
TOPICS = {
    0: ["город", "улица", "дом", "фонарь"],
    1: ["поле", "река", "лес", "дорога"],
}


def _opaque(prefix: str, value: str, length: int = 20) -> str:
    return f"{prefix}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:length]}"


def _tokens(
    author: str,
    work_no: int,
    source: str,
    edition: str,
    *,
    seed: int,
    length: int = 240,
) -> list[str]:
    rng = random.Random(seed)
    style = AUTHORS[author]
    topic = TOPICS[work_no]
    source_markers = SOURCES[source]
    edition_markers = EDITIONS[edition]
    tokens = []
    for i in range(length):
        bucket = i % 12
        if bucket < 6:
            tokens.append(rng.choice(style))
        elif bucket < 9:
            tokens.append(rng.choice(topic))
        elif bucket < 11:
            tokens.append(rng.choice(source_markers))
        else:
            tokens.append(rng.choice(edition_markers))
    return tokens


def _write_text(root: pathlib.Path, doc_id: str, tokens: list[str]) -> tuple[str, str]:
    relative = f"texts/{doc_id}.txt"
    payload = (" ".join(tokens) + "\n").encode("utf-8")
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return relative, hashlib.sha256(payload).hexdigest()


def build_package(root: pathlib.Path, seed: int = 42) -> dict:
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"output directory must be absent or empty: {root}")
    root.mkdir(parents=True, exist_ok=True)

    documents = []
    truth_records = []
    predictions = []
    counter = 0

    for author_no, author in enumerate(AUTHORS):
        for work_no in TOPICS:
            work = _opaque("work", f"{author}:{work_no}")
            for source_no, source in enumerate(SOURCES):
                for edition_no, edition in enumerate(EDITIONS):
                    semantic = f"train:{author}:{work_no}:{source}:{edition}"
                    doc_id = _opaque("doc", semantic)
                    tokens = _tokens(
                        author,
                        work_no,
                        source,
                        edition,
                        seed=seed + counter * 17,
                    )
                    counter += 1
                    text_path, digest = _write_text(root, doc_id, tokens)
                    documents.append(
                        {
                            "doc_id": doc_id,
                            "source": {
                                "source_id": source,
                                "provenance": f"generated:{semantic}",
                                "revision": f"seed:{seed}",
                                "sha256": digest,
                            },
                            "split": "train" if work_no == 0 else "validation",
                            "task_types": ["idio_shift"],
                            "text_path": text_path,
                            "author_label": author,
                            "work": work,
                            "edition": edition,
                            "period": "early" if work_no == 0 else "late",
                            "genre": "story" if work_no == 0 else "essay",
                            "topic": f"topic_{work_no}",
                            "register": "literary" if edition == "critical" else "popular",
                            "spans": [
                                {
                                    "start": 0,
                                    "end": len(tokens),
                                    "label": author,
                                    "ground_truth_known": True,
                                    "evidence": f"synthetic-generator:{semantic}",
                                }
                            ],
                        }
                    )

    # Blind mixed-author positive.
    mixed_id = _opaque("doc", "blind:mixed:a:b")
    first = _tokens("author_a", 0, "scan", "critical", seed=seed + 10_001, length=120)
    second = _tokens("author_b", 1, "scan", "critical", seed=seed + 10_002, length=120)
    mixed_tokens = first + second
    mixed_path, mixed_hash = _write_text(root, mixed_id, mixed_tokens)
    mixed_work = _opaque("work", "blind:mixed:a:b")
    documents.append(
        {
            "doc_id": mixed_id,
            "source": {
                "source_id": "scan",
                "provenance": f"sealed:{mixed_id}",
                "revision": "sealed:v1",
                "sha256": mixed_hash,
            },
            "split": "blind",
            "task_types": ["mixed_authorship"],
            "text_path": mixed_path,
            "spans": [],
        }
    )
    mixed_truth_spans = [
        {
            "start": 0,
            "end": len(first),
            "label": "author_a",
            "evidence": "synthetic-generator:first-half",
        },
        {
            "start": len(first),
            "end": len(mixed_tokens),
            "label": "author_b",
            "evidence": "synthetic-generator:second-half",
        },
    ]
    truth_records.append({"doc_id": mixed_id, "spans": mixed_truth_spans})
    predictions.append(
        {
            "doc_id": mixed_id,
            "spans": [{k: v for k, v in span.items() if k != "evidence"} for span in mixed_truth_spans],
        }
    )

    # Blind single-author negative control, scored by the segmentation FPR.
    # It is also the registered no-shift observation for the idio_shift
    # endpoint; every top-level task must have at least one blind observation.
    control_id = _opaque("doc", "blind:single:c")
    control_tokens = _tokens(
        "author_c", 1, "web", "popular", seed=seed + 10_003, length=240
    )
    control_path, control_hash = _write_text(root, control_id, control_tokens)
    control_work = _opaque("work", "blind:single:c")
    documents.append(
        {
            "doc_id": control_id,
            "source": {
                "source_id": "web",
                "provenance": f"sealed:{control_id}",
                "revision": "sealed:v1",
                "sha256": control_hash,
            },
            "split": "blind",
            "task_types": ["mixed_authorship", "idio_shift", "spoof"],
            "text_path": control_path,
            "spans": [],
        }
    )
    control_truth_span = {
        "start": 0,
        "end": len(control_tokens),
        "label": "author_c",
        "evidence": "synthetic-generator:single-author-control",
    }
    truth_records.append(
        {
            "doc_id": control_id,
            "author_label": "author_c",
            "author_evidence": "synthetic-generator:known-author-c",
            "document_label": "genuine",
            "document_evidence": "synthetic-generator:single-author-control",
            "spans": [control_truth_span],
        }
    )
    predictions.append(
        {
            "doc_id": control_id,
            "author_label": "author_c",
            "document_label": "genuine",
            "spans": [{k: v for k, v in control_truth_span.items() if k != "evidence"}],
        }
    )

    manifest = {
        "schema_version": "1.0",
        "dataset": {
            "name": "SPOOF-RU synthetic integration control",
            "version": "0.1.0",
            "license": "CC0-1.0",
            "language": "ru",
            "description": "Synthetic integration control; never scientific evidence",
            "offset_unit": "token",
            "tokenizer": "stylo_unicode_word_punct_v1",
        },
        "task_types": ["spoof", "idio_shift", "mixed_authorship"],
        "documents": documents,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        dumps_strict(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    truth = {
        "schema_version": "1.0",
        "dataset_name": manifest["dataset"]["name"],
        "dataset_version": manifest["dataset"]["version"],
        "manifest_sha256": manifest_digest,
        "records": truth_records,
    }
    submission = {
        "schema_version": "1.0",
        "dataset_name": manifest["dataset"]["name"],
        "dataset_version": manifest["dataset"]["version"],
        "predictions": predictions,
    }
    (root / "truth.synthetic-public.json").write_text(
        dumps_strict(truth, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "submission.reference.json").write_text(
        dumps_strict(submission, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        "# Synthetic integration control\n\n"
        "Generated data only. It validates the pipeline and must never be cited as "
        "historical or state-of-the-art evidence.\n",
        encoding="utf-8",
    )
    return {
        "root": str(root.resolve()),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_digest,
        "n_documents": len(documents),
        "n_blind": len(truth_records),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = build_package(pathlib.Path(args.out), seed=args.seed)
    print(dumps_strict(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
