"""Synthetic HTML only: no literary text fixtures are redistributed."""
from collections import Counter
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("publication_acquisition", ROOT / "scripts/fetch_publication_prose.py")
acq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acq)


def html_page(text):
    return f'<div id="text"><div class="author"><z>HEADER</z></div><h1>TITLE</h1><z>{text}</z><div id="tbd"></div><div><z>FOOTER</z></div></div>'


def metadata_page():
    return '<div class="tabout">Даты написания: 1899 г.. Источник: Author. Volume 1. 1990. Добавлено в библиотеку: 01.01.2000.</div><a href="/text/1/p.1/index.html">FIRST</a><a href="/text/1/p.2/index.html">SECOND</a>'


def test_provider_paragraph_extraction_excludes_editorial_nodes():
    markup = '<div id="text"><div class="author"><z>HEADER</z></div><h1>TITLE</h1><h2>CHAPTER</h2><z> alpha <i>beta</i><sup>99</sup><fn>NOTE</fn> gamma </z><z>delta&nbsp;epsilon</z><div id="tbd"></div><div class="note"><z>COMMENT</z></div><script>NOISE</script></div>'
    assert acq.extract_body(markup) == 'alpha beta gamma\n\ndelta epsilon\n'


@pytest.mark.parametrize("markup", ["<div id='text'><z>alpha</z></div>", "<div id='tbd'></div>", html_page("alpha") + "<div id='tbd'></div>"])
def test_provider_dom_drift_is_not_silently_accepted(markup):
    with pytest.raises(ValueError):
        acq.extract_body(markup)


def test_word_count_and_nfc_are_explicit():
    assert acq.word_count("first-word 123 next_word l’amour е\u0308") == 5
    assert acq.extract_body(html_page("е\u0308")) == "ё\n"


def test_multi_page_reconstruction_checks_metadata_and_body(tmp_path):
    metadata = metadata_page().encode("cp1251")
    pages = [html_page("alpha beta").encode("cp1251"), html_page("gamma delta").encode("cp1251")]
    work = {"work_id": "author/work", "source_url": "https://ilibrary.ru/text/1/index.html",
            "metadata_html_sha256": acq.digest(metadata), **acq.bibliography(metadata.decode("cp1251")),
            "pages": [{"url": f"https://ilibrary.ru/text/1/p.{i}/index.html", "html_sha256": acq.digest(raw)} for i, raw in enumerate(pages, 1)],
            "text_sha256": acq.digest(b"alpha beta\n\ngamma delta\n"), "word_count": 4}
    (tmp_path / "metadata.html").write_bytes(metadata)
    for i, raw in enumerate(pages, 1):
        (tmp_path / f"{i:03d}.html").write_bytes(raw)
    assert acq.reconstruct(work, tmp_path)[0] == b"alpha beta\n\ngamma delta\n"
    (tmp_path / "002.html").write_bytes(html_page("changed input").encode())
    with pytest.raises(ValueError, match="Text checksum mismatch"):
        acq.reconstruct(work, tmp_path)
    (tmp_path / "002.html").write_bytes(pages[1])
    (tmp_path / "metadata.html").write_bytes(metadata + b" ")
    assert acq.reconstruct(work, tmp_path)[1][0]["rendering_matches_acquired"] is False
    (tmp_path / "metadata.html").write_bytes(metadata + b'<a href="/text/1/p.3/index.html">THIRD</a>')
    with pytest.raises(ValueError, match="chapter inventory mismatch"):
        acq.reconstruct(work, tmp_path)
    (tmp_path / "metadata.html").write_bytes(metadata.replace(b"1990", b"1991"))
    with pytest.raises(ValueError, match="bibliography mismatch"):
        acq.reconstruct(work, tmp_path)


def test_acquisition_rejects_undeclared_providers_without_network():
    with pytest.raises(ValueError, match="declared HTTPS provider"):
        acq.fetch("https://example.org/work")


def test_catalog_roles_and_scope_are_fixed_before_fit():
    catalog = json.loads((ROOT / "research/corpora/publication_prose_v1.json").read_text())
    works = catalog["works"]
    assert len(works) == 33
    assert len({w["work_id"] for w in works}) == len(works)
    assert len({w["text_sha256"] for w in works}) == len(works)
    assert catalog["selection"]["model_results_seen_during_selection"] is False
    for role, counts_key in [("development", "development_counts"), ("locked_test", "locked_test_counts")]:
        assert Counter(w["author_id"] for w in works if w["role"] == role) == catalog["selection"][counts_key]
    for work in works:
        assert 1890 <= work["date_year_min"] <= work["date_year_max"] <= 1904
        assert work["word_count"] >= 1000
        assert work["edition_label"] and work["pages"]
        assert work["bibliographic_status"] == "provider_declared_print_source_not_independently_collated"
        if work["role"] == "locked_test":
            assert work["historical_work_id"] is None
            assert work["prior_historical_exposure"] is False


def test_content_audit_detects_new_id_for_historical_copy(tmp_path):
    # Generated alphabetic tokens have no literary source and exceed min_shingles.
    text = " ".join(f"token{chr(97+i//26)}{chr(97+i%26)}" for i in range(50))
    text_root = tmp_path / "texts"
    for work_id in ["author/development", "author/locked"]:
        target = text_root / f"{work_id}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    historical_text = tmp_path / "input_clean" / "author" / "different_title.txt"
    historical_text.parent.mkdir(parents=True)
    historical_text.write_text(text)
    low_text = "alpha beta " * 100
    low_path = historical_text.parent / "low_diversity.txt"
    low_path.write_text(low_text)
    historical = {"corrected_content_inventory": [{"path": "input_clean/author/different_title.txt", "sha256": acq.digest(text.encode())},
                                                 {"path": "input_clean/author/low_diversity.txt", "sha256": acq.digest(low_text.encode())}]}
    historical_path = tmp_path / "historical.json"
    historical_path.write_text(json.dumps(historical))
    catalog = {"works": [{"work_id": f"author/{name}", "role": role, "text_sha256": acq.digest(text.encode())} for name, role in [("development", "development"), ("locked", "locked_test")]],
               "historical_exposure_check": {"corrected_manifest_sha256": acq.digest(historical_path.read_bytes()), "historical_n_works": 2}}
    audit = acq.audit_content(catalog, text_root, historical_path)
    assert audit["within_panel"]["development_locked_max_containment"] == 1
    assert len(audit["within_panel"]["flags"]) == 1
    assert len(audit["locked_vs_historical"]["flags"]) == 1
    assert audit["locked_vs_historical"]["per_locked_max"][0]["historical_work_id"] == "author/different_title"
    assert audit["locked_vs_historical"]["n_historical_inputs_verified"] == 2
    assert audit["locked_vs_historical"]["n_historical_inputs_eligible"] == 1
    assert audit["locked_vs_historical"]["n_pairs_below_minimum"] == 1
    assert audit["locked_vs_historical"]["historical_inputs_below_minimum"][0]["historical_work_id"] == "author/low_diversity"
