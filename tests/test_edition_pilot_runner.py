import importlib.util
from pathlib import Path

import numpy as np


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "run_edition_invariance_pilot.py"
    spec = importlib.util.spec_from_file_location("run_edition_invariance_pilot", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


pilot = _load_script()


def test_identifiability_audit_detects_author_period_confound():
    labels = np.asarray(["chehov", "chehov", "gogol", "gogol"], dtype=object)
    metadata = {
        "period": np.asarray(["1890s", "1890s", "1830s", "1830s"], dtype=object),
        "source": np.asarray(["main", "version2", "main", "version2"], dtype=object),
        "edition": np.asarray(["main", "version2", "main", "version2"], dtype=object),
    }

    report = pilot.identifiability_audit(labels, metadata)

    assert report["perfect_author_confounds"] == ["period"]
    assert report["factors"]["period"]["authors_by_factor"] == {
        "1830s": ["gogol"],
        "1890s": ["chehov"],
    }
    assert report["factors"]["source"]["perfect_confound"] is False


def test_purged_report_uses_actual_cells_and_has_svd_control():
    texts = []
    authors = []
    works = []
    editions = []
    for author, phrase in (("a", "алый мягкий голос"), ("b", "бурый резкий голос")):
        for work_no in range(3):
            work = f"{author}_work_{work_no}"
            for edition in ("main", "version2", "local"):
                texts.append(
                    ((phrase + " ") * 35)
                    + ((f"тема{work_no} ") * 10)
                    + ((f"редакция{edition} ") * 12)
                )
                authors.append(author)
                works.append(work)
                editions.append(edition)

    labels = np.asarray(authors, dtype=object)
    metadata = {
        "author": labels.copy(),
        "work": np.asarray(works, dtype=object),
        "source": np.asarray(editions, dtype=object),
        "edition": np.asarray(editions, dtype=object),
    }
    report = pilot.purged_edition_attribution(
        np.asarray(texts, dtype=object), labels, metadata, bootstrap_iters=0
    )

    assert report["plan"]["n_splits"] == 18
    assert report["actual_cell_fit_sizes"] == {
        "min_train": 10,
        "max_train": 10,
        "min_test": 1,
        "max_test": 1,
    }
    assert "svd_only_control" in report
    for model_name in (
        "baseline_char_lr",
        "svd_only_control",
        "paired_residualizer",
    ):
        model_report = report[model_name]
        assert model_report["factors"] == {}
        assert set(model_report["by_edition"]) == {"local", "main", "version2"}
