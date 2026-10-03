"""Immutable prediction publication, historical selection, and crash contracts."""
from __future__ import annotations

import hashlib
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

from stylo.config import load_config, with_overrides
from stylo.jsonio import load_strict
from stylo.pipeline import _snapshot, predict
from stylo.pipeline.bundle import load_bundle, publish_bundle
from stylo.report import build, evidence

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture
def report_case(tmp_path, monkeypatch):
    cfg = with_overrides(load_config(), {
        'paths.data': str(tmp_path / 'data'), 'paths.docs': str(tmp_path / 'docs'),
        'deployment.candidate_authors': ['alpha', 'beta'],
        'evaluation.training_weighting': 'work_balanced',
    })
    monkeypatch.setattr(evidence, '_code_tree_sha256', lambda: 'a' * 64)
    catalog = tmp_path / 'targets'
    for name in ('one', 'two'):
        path = catalog / 'unknown' / name / '0.txt'
        path.parent.mkdir(parents=True)
        path.write_text(f'Отдельный синтетический учебный пример {name}.', encoding='utf-8')
    return cfg, catalog


def _model(cfg):
    meta = {
        'training_weighting': 'work_balanced', 'dataset_contract': 'work_balanced_manifest',
        'rows_digest': 'b' * 64, 'chunker_config_hash': 'c' * 64,
        'code_tree_sha256': evidence._code_tree_sha256(), 'config_id': evidence._config_id(cfg),
        'git_commit': None, 'git_dirty': None,
    }
    root = pathlib.Path(cfg.get_path('paths.data')) / 'deployment' / 'work_balanced'
    receipt = publish_bundle(root, {
        'model.pkl': lambda path: path.write_bytes(b'synthetic model fixture'),
        'delta.pkl': lambda path: path.write_bytes(b'synthetic delta fixture'),
        'authors.json': lambda path: path.write_text('["alpha","beta"]', encoding='utf-8'),
    }, meta)
    full_meta, _ = load_bundle(root, expected_token=receipt['bundle_token'])
    return receipt['bundle_token'], full_meta


def _publish(cfg, catalog, work='unknown/one', model=None):
    token, meta = model or _model(cfg)
    target = predict.resolve_prediction_target(cfg, work, unknown_root=catalog)
    result = {'schema_version': 'stylo.target-prediction.v2', 'target_work_id': work,
              'candidate_authors': ['alpha', 'beta'], 'training_weighting': 'work_balanced',
              'fixture': 'synthetic saved result; no accuracy measured'}
    directory = evidence.publish_prediction(
        cfg, unknown_root=target.root, target=target, selected_identity=predict.target_identity(target),
        structured=result, report=f'Synthetic saved result for {work}; model {token}.',
        bundle_token=token, bundle_meta=meta,
    )
    return directory, result


def _bytes(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob('*') if path.is_file()}


def test_two_models_one_work_and_another_work_keep_all_results(report_case):
    cfg, catalog = report_case
    first, first_result = _publish(cfg, catalog)
    first_bytes = _bytes(first)
    changed = with_overrides(cfg, {'model.classifier.C': 17.0})
    second, second_result = _publish(changed, catalog)
    other, _ = _publish(changed, catalog, 'unknown/two')
    assert first != second != other
    assert _bytes(first) == first_bytes
    assert evidence.verify_structured_prediction(changed, 'unknown/one') == second_result
    assert evidence.verify_structured_prediction(changed, 'unknown/one', result_id=first.name) == first_result
    assert load_strict(first / 'bundle.json')['config_id'] != load_strict(second / 'bundle.json')['config_id']
    assert load_strict(first / 'config.json')['model']['classifier']['C'] == 1.0
    assert load_strict(second / 'config.json')['model']['classifier']['C'] == 17.0
    with pytest.raises(evidence.SectionEvidenceError, match='stale'):
        evidence.verify_prediction(cfg, 'unknown/one')
    assert build.run_prediction(changed, 'unknown/one', result_id=first.name) == first / 'index.html'
    assert _bytes(first) == first_bytes


def test_explicit_history_uses_saved_config_after_inputs_and_models_are_gone(report_case, monkeypatch):
    cfg, catalog = report_case
    first, result = _publish(cfg, catalog)
    before = _bytes(first)
    shutil.rmtree(catalog)
    shutil.rmtree(pathlib.Path(cfg.get_path('paths.data')))
    changed = with_overrides(cfg, {'model.classifier.C': 99.0,
                                   'deployment.expected_bundle_token': 'f' * 32})
    monkeypatch.setattr(evidence, '_code_tree_sha256', lambda: 'd' * 64)
    assert evidence.verify_structured_prediction(changed, result_id=first.name) == result
    assert build.run_prediction(changed, result_id=first.name) == first / 'index.html'
    assert _bytes(first) == before


def test_changed_target_content_keeps_original_historical_result(report_case):
    cfg, catalog = report_case
    model = _model(cfg)
    first, _ = _publish(cfg, catalog, model=model)
    (catalog / 'unknown' / 'one' / '0.txt').write_text('Другой синтетический текст.', encoding='utf-8')
    second, _ = _publish(cfg, catalog, model=model)
    assert first != second
    old_identity = load_strict(first / 'prediction.evidence.json')['identity']
    new_identity = load_strict(second / 'prediction.evidence.json')['identity']
    assert old_identity['target_sha256'] != new_identity['target_sha256']
    assert evidence.verify_prediction(cfg, 'unknown/one', result_id=first.name)


def test_second_payload_failure_preserves_complete_prior_publication(report_case, monkeypatch):
    cfg, catalog = report_case
    model = _model(cfg)
    first, result = _publish(cfg, catalog, model=model)
    original = evidence._atomic_write_text
    calls = []
    def fail_second(path, body):
        calls.append(path)
        if len(calls) == 2:
            raise OSError('injected second-payload failure')
        original(path, body)
    monkeypatch.setattr(evidence, '_atomic_write_text', fail_second)
    with pytest.raises(OSError, match='second-payload failure'):
        _publish(cfg, catalog, model=model)
    assert len(calls) == 2
    assert evidence.prediction_directory(cfg, 'unknown/one') == first
    assert evidence.verify_structured_prediction(cfg, 'unknown/one') == result
    assert build.run_prediction(cfg, 'unknown/one') == first / 'index.html'


def test_actual_pointer_replace_failure_keeps_prior_html_and_evidence(report_case, monkeypatch):
    cfg, catalog = report_case
    model = _model(cfg)
    first, _ = _publish(cfg, catalog, model=model)
    prior = _bytes(first)
    original = _snapshot.os.replace
    def fail_pointer(source, target):
        if pathlib.Path(target).name == _snapshot.CURRENT_POINTER:
            raise OSError('injected pointer replace failure')
        return original(source, target)
    monkeypatch.setattr(_snapshot.os, 'replace', fail_pointer)
    changed = with_overrides(cfg, {'evaluation.top_k_candidates': 1})
    with pytest.raises(OSError, match='pointer replace failure'):
        _publish(changed, catalog, model=model)
    assert evidence.prediction_directory(cfg, 'unknown/one') == first
    assert _bytes(first) == prior
    assert build.run_prediction(cfg, 'unknown/one') == first / 'index.html'


@pytest.mark.parametrize('boundary', ['before', 'after'])
def test_process_crash_switches_report_files_as_one_generation(tmp_path, boundary):
    docs = tmp_path / 'reports'
    docs.mkdir()
    first = evidence._publish_section(docs, section='prediction',
        files={'prediction.json': '{"fixture":"old"}', 'prediction.txt': 'old', 'index.html': 'old'},
        identity={'fixture': 'old'})
    code = '''
import os, pathlib, sys
from stylo.report import evidence
from stylo.pipeline import _snapshot
original = _snapshot._publish_current_pointer
def crash(store, token):
    if sys.argv[2] == 'after': original(store, token)
    os._exit(73)
_snapshot._publish_current_pointer = crash
evidence._publish_section(pathlib.Path(sys.argv[1]), section='prediction',
    files={'prediction.json':'{"fixture":"new"}', 'prediction.txt':'new', 'index.html':'new'},
    identity={'fixture':'new'})
'''
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
    crashed = subprocess.run([sys.executable, '-c', code, str(docs), boundary],
                             capture_output=True, text=True, env=env, timeout=30)
    assert crashed.returncode == 73
    identity, bodies = evidence._verify_section(docs, section='prediction',
                                              expected_files={'prediction.json', 'prediction.txt', 'index.html'})
    expected = 'old' if boundary == 'before' else 'new'
    assert identity['fixture'] == expected
    assert bodies['prediction.txt'] == bodies['index.html'] == expected
    assert (first / 'index.html').read_text() == 'old'


@pytest.mark.parametrize('file', ['index.html', 'config.json', 'bundle.json', 'prediction.json'])
def test_whole_generation_is_verified_before_returning_html(report_case, file):
    cfg, catalog = report_case
    generation, _ = _publish(cfg, catalog)
    (generation / file).write_text('tampered', encoding='utf-8')
    with pytest.raises(build.ReportEvidenceError, match='inventory/hash mismatch'):
        build.run_prediction(cfg, 'unknown/one', result_id=generation.name)


def test_selector_never_crosses_target_collections_or_trusts_another_model(report_case):
    cfg, catalog = report_case
    first, _ = _publish(cfg, catalog)
    _publish(cfg, catalog, 'unknown/two')
    with pytest.raises(evidence.SectionEvidenceError):
        evidence.verify_prediction(cfg, 'unknown/two', result_id=first.name)
    wrong = with_overrides(cfg, {'deployment.expected_bundle_token': 'f' * 32})
    with pytest.raises(evidence.SectionEvidenceError, match='trusted bundle token'):
        evidence.verify_prediction(wrong, 'unknown/one')
    with pytest.raises(evidence.SectionEvidenceError, match='full generation hash'):
        evidence.verify_prediction(cfg, 'unknown/one', result_id='../one')


def test_legacy_flat_target_stays_readable_and_is_never_migrated_destructively(report_case):
    cfg, catalog = report_case
    current, result = _publish(cfg, catalog)
    envelope = load_strict(current / 'prediction.evidence.json')
    identity = dict(envelope['identity'])
    identity['prediction_schema_version'] = 'stylo.target-prediction.v2'
    identity.pop('presentation_settings')
    old_cfg = with_overrides(cfg, {'paths.docs': str(pathlib.Path(cfg.get_path('paths.docs')) / 'old')})
    collection = evidence.prediction_directory(old_cfg, 'unknown/one', create=True)
    evidence._publish_locked_section(collection, section='prediction',
        files={name: (current / name).read_text() for name in ('prediction.json', 'prediction.txt')},
        identity=identity)
    old_files = {path.name: path.read_bytes() for path in collection.iterdir() if path.is_file()}
    assert evidence.verify_structured_prediction(old_cfg, 'unknown/one') == result
    _publish(old_cfg, catalog)
    assert evidence.verify_structured_prediction(old_cfg, 'unknown/one', result_id='legacy') == result
    view = build.run_prediction(old_cfg, 'unknown/one', result_id='legacy')
    assert view.is_file()
    assert {name: (collection / name).read_bytes() for name in old_files} == old_files


def test_model_token_mismatch_cannot_commit_or_damage_previous_result(report_case):
    cfg, catalog = report_case
    model = _model(cfg)
    original, _ = _publish(cfg, catalog, model=model)
    before = _bytes(original)
    with pytest.raises(evidence.SectionEvidenceError, match='model token'):
        _publish(cfg, catalog, model=('f' * 32, model[1]))
    assert evidence.prediction_directory(cfg, 'unknown/one') == original
    assert _bytes(original) == before
