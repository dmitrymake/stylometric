from __future__ import annotations

import json
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts" / "gen-site-data.mjs"


def test_site_generator_strict_input_self_test():
    completed = subprocess.run(
        ["node", str(GENERATOR), "--self-test"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert "strict-input self-test: OK" in completed.stdout


def test_site_generator_has_no_token_rewrite_or_subtree_null_allowlist():
    source = GENERATOR.read_text(encoding="utf-8")
    assert r".replace(/\bNaN\b/g" not in source
    assert "h.includes(a)" not in source
    assert "NULLABLE_PATHS.has(h)" in source


def test_site_build_includes_render_and_undefined_identifier_checks():
    package = json.loads((ROOT / "site" / "package.json").read_text(encoding="utf-8"))
    assert package["scripts"]["test:render"] == "node ./scripts/check-render.mjs"
    assert "npm run test:render" in package["scripts"]["build"]
    assert package["scripts"]["check:undef"] == "node ./scripts/check-no-undef.mjs"
    assert "npm run check:undef" in package["scripts"]["build"]
    assert package["devDependencies"]["@babel/parser"] == "7.29.7"
    assert package["devDependencies"]["@babel/traverse"] == "7.29.7"


def test_site_lock_contains_every_declared_optional_platform_package():
    lock = json.loads((ROOT / "site" / "package-lock.json").read_text(encoding="utf-8"))
    packages = lock["packages"]
    for dependency in ("@babel/parser", "@babel/traverse"):
        assert packages[""]["devDependencies"][dependency] == "7.29.7"
        assert packages[f"node_modules/{dependency}"]["version"] == "7.29.7"
    optional = {
        dependency
        for package in ("node_modules/esbuild", "node_modules/rollup")
        for dependency in packages[package]["optionalDependencies"]
    }
    missing = sorted(
        dependency for dependency in optional if f"node_modules/{dependency}" not in packages
    )
    assert missing == []
    incomplete_registry_records = sorted(
        path
        for path, record in packages.items()
        if path and ("resolved" not in record or "integrity" not in record)
    )
    assert incomplete_registry_records == []


def test_first_experiment_machine_status_matches_source():
    registry = json.loads(
        (
            ROOT
            / "research"
            / "evidence"
            / "ineligible_corpus_registrations_v1.json"
        ).read_text(encoding="utf-8")
    )
    site_data = json.loads(
        (ROOT / "site" / "src" / "generated" / "site-data.json").read_text(
            encoding="utf-8"
        )
    )
    headline = site_data["headline"]
    assert registry["status"] == "ineligible_for_new_scientific_runs"
    assert headline["corpusEligibilityStatus"] == registry["status"]
    assert headline["claimStatus"] == "exploratory_internal"


def test_sholokhov_claim_is_bound_to_registered_lobo_source():
    registered = json.loads(
        (ROOT / "docs" / "sholokhov_lobo.json").read_text(encoding="utf-8")
    )
    site_data = json.loads(
        (ROOT / "site" / "src" / "generated" / "site-data.json").read_text(
            encoding="utf-8"
        )
    )
    rigor = site_data["rigor"]

    assert rigor["tdLoboAttributed"] == registered["td_attributed_to_sholokhov"]
    assert [step["ff"] for step in rigor["loboTd"]["gradient"]] == [
        step["foreign_fraction"] for step in registered["disputed_td"]
    ]
    assert rigor["tdLoboP"] == registered["td1_vs_null_permutation_p"]
    assert rigor["tdLoboSurvives"] == registered["don_source_signal_significant"]


def test_sholokhov_registered_reference_and_heldout_worksets():
    registered = json.loads(
        (ROOT / "docs" / "sholokhov_lobo.json").read_text(encoding="utf-8")
    )
    assert registered["anchor_solo_in_train"] == ["rodinka", "zherebenok", "batraki"]
    heldout_td = {work for work in registered["heldout"] if work.startswith("tihiy_don_")}
    assert heldout_td == {f"tihiy_don_{index}" for index in range(1, 5)}
    assert registered["td_attributed_to_sholokhov"] == "3/4"


def _measurement_source():
    return json.loads((ROOT / 'research/evidence/topic_validity_lobo_v1/aggregate.json').read_text())


def test_completed_measurement_is_distinct_from_historical_headline_and_source_bound():
    data = json.loads((ROOT / 'site/src/generated/site-data.json').read_text())
    artifact = _measurement_source()
    measured = data['measurement']
    assert measured['sourceSelfHash'] == artifact['self_hash']
    assert measured['works'] == artifact['design']['fold_count']
    assert measured['testedAuthors'] == artifact['design']['tested_author_count']
    assert measured['candidateClasses'] == artifact['design']['probability_class_count']
    assert measured['fits'] == measured['works'] * len(artifact['design']['cells']) * len(artifact['design']['arms'])
    for cell, source in zip(measured['cells'], artifact['cells'], strict=True):
        assert cell['cell'] == source['cell']
        for arm in artifact['design']['arms']:
            correct, total = source['accuracy'][arm]['correct'], source['accuracy'][arm]['total']
            assert cell['accuracy'][arm] == {'correct': correct, 'total': total, 'value': correct / total}
        assert cell['delta']['numerator'] == source['delta_accuracy']['numerator']
        for key in artifact['design']['transition_categories']:
            assert cell['transitions'][key] == sum(row[key] for row in source['per_author_transitions'])
    assert 'macroF1' not in measured and 'ci' not in measured
    assert data['headline']['macroF1CI'] is None
    assert data['headline']['claimStatus'] == 'exploratory_internal'
    registry = json.loads((ROOT / 'site/src/generated/manifest.json').read_text())
    entry = next(row for row in registry['entries'] if row['key'] == 'measurement')
    assert entry['sources'] == [measured['source']]
    assert (ROOT / 'site/public' / measured['publicArtifact']).read_bytes() == (ROOT / measured['source']).read_bytes()


def _copy_site_provenance_tree(destination):
    import shutil

    registry = json.loads((ROOT / 'site/src/generated/manifest.json').read_text())
    paths = {row['path'] for row in [registry['generator'], *registry['sources'], *registry['outputs']]}
    paths.update({'site/src/generated/manifest.json', 'scripts/check-provenance.mjs'})
    for relative in sorted(paths):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    return registry


def test_measurement_provenance_rejects_forged_count_even_if_output_digest_is_updated(tmp_path):
    import hashlib

    registry = _copy_site_provenance_tree(tmp_path)
    path = tmp_path / 'site/src/generated/site-data.json'
    data = json.loads(path.read_text())
    data['measurement']['cells'][0]['accuracy']['current']['correct'] += 1
    path.write_text(json.dumps(data))
    output = next(row for row in registry['outputs'] if row['path'] == 'site/src/generated/site-data.json')
    output['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / 'site/src/generated/manifest.json').write_text(json.dumps(registry))
    result = subprocess.run(['node', str(tmp_path / 'scripts/check-provenance.mjs'),
                             '--root', str(tmp_path), '--skip-tracked'], text=True, capture_output=True)
    assert result.returncode != 0
    assert 'accuracy differs from the canonical source' in result.stderr


def test_measurement_generator_rejects_inconsistent_transition_arithmetic(tmp_path):
    import hashlib

    _copy_site_provenance_tree(tmp_path)
    path = tmp_path / 'research/evidence/topic_validity_lobo_v1/aggregate.json'
    artifact = json.loads(path.read_text())
    artifact['cells'][0]['accuracy']['current']['correct'] += 1
    unsigned = {key: value for key, value in artifact.items() if key != 'self_hash'}
    artifact['self_hash'] = hashlib.sha256(json.dumps(unsigned, ensure_ascii=False,
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    path.write_text(json.dumps(artifact, ensure_ascii=False))
    result = subprocess.run(['node', str(tmp_path / 'scripts/gen-site-data.mjs')],
                             cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode != 0
    assert 'accuracy/transition arithmetic mismatch' in result.stderr


def test_paired_analysis_preserves_source_metrics_and_download_bytes():
    from stylo.jsonio import artifact_self_hash

    source = ROOT / 'research/evidence/topic_validity_lobo_v1/paired_summary.json'
    artifact = json.loads(source.read_text())
    assert artifact_self_hash(artifact) == artifact['self_hash']
    data = json.loads((ROOT / 'site/src/generated/site-data.json').read_text())
    paired = data['measurement']['pairedAnalysis']
    assert paired['arms'] == artifact['arms']
    assert paired['comparisons'] == artifact['comparisons']
    assert (ROOT / 'site/public' / paired['publicArtifact']).read_bytes() == source.read_bytes()


def test_paired_provenance_rejects_forged_macro_recall_with_updated_output_hash(tmp_path):
    import hashlib

    registry = _copy_site_provenance_tree(tmp_path)
    path = tmp_path / 'site/src/generated/site-data.json'
    data = json.loads(path.read_text())
    data['measurement']['pairedAnalysis']['arms'][0]['macro_author_recall'] += 0.01
    path.write_text(json.dumps(data))
    output = next(row for row in registry['outputs'] if row['path'] == 'site/src/generated/site-data.json')
    output['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / 'site/src/generated/manifest.json').write_text(json.dumps(registry))
    result = subprocess.run(['node', str(tmp_path / 'scripts/check-provenance.mjs'),
                             '--root', str(tmp_path), '--skip-tracked'], text=True, capture_output=True)
    assert result.returncode != 0
    assert 'paired analysis differs from source metrics/transitions' in result.stderr


def test_reader_downloads_contain_the_displayed_aggregates_and_bind_sources():
    data = json.loads((ROOT / 'site/src/generated/site-data.json').read_text())
    registry = json.loads((ROOT / 'site/src/generated/manifest.json').read_text())
    bound_sources = {row['path']: row['sha256'] for row in registry['sources']}
    assert set(data['caseDownloads']) == {'sholokhov', 'ilfpetrov', 'nikolai', 'hohol', 'controls'}
    for chapter, descriptor in data['caseDownloads'].items():
        payload = json.loads((ROOT / 'site/public' / descriptor['publicArtifact']).read_text())
        assert payload['schema'] == 'stylo.reader-data.v1'
        assert payload['chapter'] == chapter
        assert payload['datasets'] == {key: data[key] for key in descriptor['keys']}
        assert payload['sources']
        assert all(bound_sources[row['path']] == row['sha256'] for row in payload['sources'])


def test_provenance_rejects_forged_reader_download_with_updated_file_digest(tmp_path):
    import hashlib

    registry = _copy_site_provenance_tree(tmp_path)
    relative = 'site/public/evidence/controls.json'
    path = tmp_path / relative
    payload = json.loads(path.read_text())
    payload['datasets']['limits']['threshold'] = 0.99
    path.write_text(json.dumps(payload))
    next(row for row in registry['outputs'] if row['path'] == relative)['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / 'site/src/generated/manifest.json').write_text(json.dumps(registry))
    result = subprocess.run(['node', str(tmp_path / 'scripts/check-provenance.mjs'),
                             '--root', str(tmp_path), '--skip-tracked'], text=True, capture_output=True)
    assert result.returncode != 0
    assert 'reader download differs from displayed chapter data' in result.stderr
