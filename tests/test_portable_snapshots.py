"""Crash, concurrent publication, and read-only legacy migration contracts."""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import tempfile
import time

import pytest

from stylo import _io
from stylo.pipeline import _snapshot as snapshots

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _stage(target, value):
    stage = pathlib.Path(tempfile.mkdtemp(prefix='.staging-', dir=target.parent))
    for name in ('first.txt', 'second.txt'):
        (stage / name).write_text(value, encoding='utf-8')
    return stage


def _publish(target, value):
    return snapshots.publish_directory_snapshot(_stage(target, value), target)


def _process(code, *args):
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT / 'src')
    return subprocess.Popen([sys.executable, '-c', code, *map(str, args)],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _finish(process):
    out, err = process.communicate(timeout=30)
    assert process.returncode == 0, (out, err)


def test_legacy_migration_retains_flat_bytes_and_pinned_reader(tmp_path):
    target = tmp_path / 'clean'
    target.mkdir()
    (target / 'first.txt').write_text('legacy', encoding='utf-8')
    assert snapshots.resolve_directory_snapshot(target) == target
    first = _publish(target, 'generation-one')
    assert (target / 'first.txt').read_text() == 'legacy'
    assert snapshots.resolve_directory_snapshot(target) == first
    second = _publish(target, 'generation-two')
    assert first != second
    assert (first / 'first.txt').read_text() == 'generation-one'
    assert snapshots.resolve_directory_snapshot(target) == second
    assert (target / 'first.txt').read_text() == 'legacy'


@pytest.mark.parametrize('boundary', ['before', 'after'])
def test_process_crash_at_pointer_boundary_keeps_a_complete_snapshot(tmp_path, boundary):
    target = tmp_path / 'clean'
    old = _publish(target, 'old')
    code = '''
import os, pathlib, sys, tempfile
from stylo.pipeline import _snapshot as s
target, boundary = pathlib.Path(sys.argv[1]), sys.argv[2]
stage = pathlib.Path(tempfile.mkdtemp(prefix='.staging-', dir=target.parent))
for name in ('first.txt', 'second.txt'):
    (stage / name).write_text('new', encoding='utf-8')
original = s._publish_current_pointer
def crash(store, token):
    if boundary == 'after': original(store, token)
    os._exit(71)
s._publish_current_pointer = crash
s.publish_directory_snapshot(stage, target)
'''
    process = _process(code, target, boundary)
    process.communicate(timeout=30)
    assert process.returncode == 71
    current = snapshots.resolve_directory_snapshot(target)
    expected = 'old' if boundary == 'before' else 'new'
    assert (current / 'first.txt').read_text() == expected
    assert (current / 'second.txt').read_text() == expected
    assert (old / 'first.txt').read_text() == 'old'
    # A crashed process releases its publication lock without removing it.
    assert (_publish(target, 'recovered') / 'first.txt').read_text() == 'recovered'


def test_concurrent_writers_and_readers_never_mix_generations(tmp_path):
    target = tmp_path / 'clean'
    pinned = _publish(target, 'initial')
    code = '''
import pathlib, sys, tempfile
from stylo.pipeline import _snapshot as s
target, author = pathlib.Path(sys.argv[1]), sys.argv[2]
for turn in range(8):
    stage = pathlib.Path(tempfile.mkdtemp(prefix='.staging-', dir=target.parent))
    for name in ('first.txt', 'second.txt'):
        (stage / name).write_text(f'{author}-{turn}', encoding='utf-8')
    s.publish_directory_snapshot(stage, target)
'''
    processes = [_process(code, target, name) for name in ('writer-one', 'writer-two')]
    reads = 0
    try:
        while any(process.poll() is None for process in processes):
            current = snapshots.resolve_directory_snapshot(target)
            assert (current / 'first.txt').read_bytes() == (current / 'second.txt').read_bytes()
            reads += 1
            time.sleep(0.001)
        for process in processes:
            _finish(process)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate()
    assert reads > 0
    assert (pinned / 'first.txt').read_text() == 'initial'


@pytest.mark.parametrize('damage', ['json', 'missing-version', 'symlink', 'unreadable'])
def test_broken_current_never_falls_back_to_legacy(tmp_path, monkeypatch, damage):
    target = tmp_path / 'clean'
    target.mkdir()
    (target / 'legacy.txt').write_text('legacy', encoding='utf-8')
    _publish(target, 'new')
    pointer = snapshots.snapshot_store(target) / snapshots.CURRENT_POINTER
    if damage == 'json':
        pointer.write_text('{broken', encoding='utf-8')
    elif damage == 'missing-version':
        pointer.write_text('{"schema_version":"stylo.directory-snapshot-pointer.v1",'
                           '"generation_id":"' + '0' * 64 + '"}', encoding='utf-8')
    elif damage == 'symlink':
        body = pointer.read_bytes()
        pointer.unlink()
        alternate = tmp_path / 'alternate.json'
        alternate.write_bytes(body)
        try:
            pointer.symlink_to(alternate)
        except OSError:
            pytest.skip('platform does not allow this symlink fixture')
    else:
        original = snapshots.read_regular
        def denied(path, **kwargs):
            if path == pointer:
                raise PermissionError('injected unreadable pointer')
            return original(path, **kwargs)
        monkeypatch.setattr(snapshots, 'read_regular', denied)
    with pytest.raises(snapshots.SnapshotPublishError):
        snapshots.resolve_directory_snapshot(target)
    assert (target / 'legacy.txt').read_text() == 'legacy'


def test_snapshot_rejects_tampered_inventory(tmp_path):
    target = tmp_path / 'clean'
    generation = _publish(target, 'original')
    (generation / 'first.txt').write_text('tampered', encoding='utf-8')
    with pytest.raises(snapshots.SnapshotPublishError, match='inventory/hash mismatch'):
        snapshots.resolve_directory_snapshot(target)


def test_duplicate_publication_reuses_generation_without_replacing_reader_files(tmp_path):
    target = tmp_path / 'clean'
    first = _publish(target, 'same')
    opened = (first / 'first.txt').open('rb')
    try:
        assert _publish(target, 'same') == first
        assert opened.read() == b'same'
    finally:
        opened.close()
    assert not list(tmp_path.glob('.staging-*'))


def test_process_lock_serializes_read_modify_write(tmp_path):
    counter = tmp_path / 'counter'
    counter.write_text('0', encoding='utf-8')
    code = '''
import pathlib, sys
from stylo._io import exclusive_file_lock
counter = pathlib.Path(sys.argv[1])
for _ in range(40):
    with exclusive_file_lock(counter.with_suffix('.lock')):
        value = int(counter.read_text())
        counter.write_text(str(value + 1))
'''
    processes = [_process(code, counter) for _ in range(3)]
    try:
        for process in processes:
            _finish(process)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate()
    assert counter.read_text() == '120'


def test_process_lock_is_released_after_forced_exit(tmp_path):
    lock = tmp_path / 'permanent.lock'
    acquired = tmp_path / 'acquired'
    code = '''
import pathlib, sys, time
from stylo._io import exclusive_file_lock
with exclusive_file_lock(sys.argv[1]):
    pathlib.Path(sys.argv[2]).touch()
    time.sleep(30)
'''
    process = _process(code, lock, acquired)
    try:
        deadline = time.monotonic() + 10
        while not acquired.exists():
            assert process.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.01)
        process.kill()
        process.communicate(timeout=10)
        with _io.exclusive_file_lock(lock):
            assert lock.is_file()
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


def test_lock_rejects_symlink(tmp_path):
    original = tmp_path / 'original'
    original.write_text('untouched', encoding='utf-8')
    link = tmp_path / 'lock'
    try:
        link.symlink_to(original)
    except OSError:
        pytest.skip('platform does not allow this symlink fixture')
    with pytest.raises(ValueError, match='symlink'):
        with _io.exclusive_file_lock(link):
            pytest.fail('symlink lock acquired')
    assert original.read_text() == 'untouched'


def test_windows_lock_adapter_retries_contention_and_unlocks_same_byte(tmp_path, monkeypatch):
    from types import SimpleNamespace

    calls = []
    sleeps = []
    def locking(fd, mode, size):
        calls.append((mode, size, os.lseek(fd, 0, os.SEEK_CUR)))
        if len(calls) == 1:
            raise OSError(13, 'synthetic contention')
    monkeypatch.setitem(sys.modules, 'msvcrt', SimpleNamespace(LK_NBLCK=1, LK_UNLCK=2, locking=locking))
    monkeypatch.setattr(_io, 'os', SimpleNamespace(name='nt', fstat=os.fstat, write=os.write,
                                                 lseek=os.lseek, SEEK_SET=os.SEEK_SET))
    monkeypatch.setattr(_io, 'time', SimpleNamespace(sleep=sleeps.append))
    with (tmp_path / 'lock').open('w+b') as handle:
        _io._lock_descriptor(handle.fileno())
        _io._unlock_descriptor(handle.fileno())
    assert calls == [(1, 1, 0), (1, 1, 0), (2, 1, 0)]
    assert sleeps == [0.02]


def test_ordinary_pipeline_imports_without_fcntl():
    process = _process('''
import sys
sys.modules['fcntl'] = None
from stylo.pipeline import clean, split
from stylo import nlp, workdoc
''')
    _finish(process)


def test_clean_split_and_work_loader_use_the_published_generation(tmp_path, monkeypatch):
    from stylo.config import load_config, with_overrides
    from stylo.corpus_tools.validate_corpus import validate
    from stylo.jsonio import load_strict
    from stylo.pipeline import clean, split
    from stylo.workdoc import load_work_balanced_dataset

    cfg = with_overrides(load_config(), {
        'paths.input_raw': str(tmp_path / 'raw'),
        'paths.input_clean': str(tmp_path / 'clean'),
        'paths.data': str(tmp_path / 'data'),
        'chunking.chunk_size': 40, 'chunking.min_words': 20,
        'evaluation.n_jobs': 1,
    })
    monkeypatch.setattr(clean, '_ner_identity', lambda *_args: {'synthetic': True})
    monkeypatch.setattr(clean, 'normalize', lambda text, *_args: text)
    for author in ('alpha', 'beta'):
        path = tmp_path / 'raw' / author / 'work.txt'
        path.parent.mkdir(parents=True)
        path.write_text(' '.join(' '.join(f'{author}{i}_{j}' for j in range(8)) + '.'
                                 for i in range(80)), encoding='utf-8')
    clean.run(cfg)
    clean_root = snapshots.resolve_directory_snapshot(tmp_path / 'clean')
    assert not (tmp_path / 'clean').exists()
    assert validate(clean_root).summary['n_books'] == 2
    split.run(cfg)
    fragments = split.resolve_fragment_snapshot(tmp_path / 'data')
    dataset = load_work_balanced_dataset(fragments.train_root, cfg=cfg)
    assert dataset.authors == ['alpha', 'beta']
    assert len(dataset) >= 10
    receipt = load_strict(clean_root / clean.CLEAN_MANIFEST)
    assert {entry['source'] for entry in receipt['files']} == {'alpha/work.txt', 'beta/work.txt'}
    old_bytes = (clean_root / 'alpha' / 'work.txt').read_bytes()
    (tmp_path / 'raw' / 'alpha' / 'work.txt').write_text('Replacement synthetic source. ' * 80,
                                                     encoding='utf-8')
    clean.run(cfg)
    assert snapshots.resolve_directory_snapshot(tmp_path / 'clean') != clean_root
    assert (clean_root / 'alpha' / 'work.txt').read_bytes() == old_bytes
    # Explicitly pinned provenance stays available after a concurrent cleaner.
    pinned = load_work_balanced_dataset(fragments.train_root, cfg=cfg, input_clean_root=clean_root)
    assert len(pinned) == len(dataset)


def test_nested_lock_for_same_canonical_path_fails_without_deadlock(tmp_path):
    lock = tmp_path / 'same.lock'
    alias = tmp_path / 'unused' / '..' / 'same.lock'
    with _io.exclusive_file_lock(lock):
        with pytest.raises(RuntimeError, match='nested exclusive_file_lock'):
            with _io.exclusive_file_lock(alias):
                pytest.fail('nested OS lock acquired')
    with _io.exclusive_file_lock(lock):
        pass


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='fork-specific process contract')
def test_fork_resets_thread_lock_and_preserves_parent_os_lock(tmp_path):
    code = '''
import os, pathlib, select, sys, threading, time
from stylo import _io
lock = pathlib.Path(sys.argv[1])
entered, release = threading.Event(), threading.Event()
def writer():
    with _io.exclusive_file_lock(lock):
        entered.set()
        release.wait(10)
thread = threading.Thread(target=writer)
thread.start()
assert entered.wait(10)
read_fd, write_fd = os.pipe()
pid = os.fork()
if pid == 0:
    os.close(read_fd)
    assert not _io._ACTIVE_LOCK_FDS
    with _io.exclusive_file_lock(lock):
        os.write(write_fd, b'acquired')
    os._exit(0)
os.close(write_fd)
# Child neither inherits a permanently blocked thread lock nor releases the
# parent's OS lock when closing its inherited descriptor copies.
assert not select.select([read_fd], [], [], 0.1)[0]
release.set()
thread.join(10)
assert not thread.is_alive()
assert select.select([read_fd], [], [], 10)[0]
assert os.read(read_fd, 8) == b'acquired'
assert os.waitpid(pid, 0)[1] == 0
'''
    _finish(_process(code, tmp_path / 'fork.lock'))


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='fork-specific process contract')
def test_inherited_context_does_not_close_a_reused_child_descriptor(tmp_path):
    code = '''
import os, pathlib, sys
from stylo import _io
lock, output = map(pathlib.Path, sys.argv[1:])
with _io.exclusive_file_lock(lock):
    pid = os.fork()
    if pid == 0:
        assert not _io._ACTIVE_LOCK_FDS
        child_fd = os.open(output, os.O_WRONLY | os.O_CREAT, 0o600)
if pid == 0:
    os.write(child_fd, b'child descriptor retained')
    os.close(child_fd)
    with _io.exclusive_file_lock(lock):
        pass
    os._exit(0)
assert os.waitpid(pid, 0)[1] == 0
assert output.read_bytes() == b'child descriptor retained'
'''
    _finish(_process(code, tmp_path / 'fork.lock', tmp_path / 'child-output'))


def test_stage_mutation_before_pointer_commit_cannot_replace_current(tmp_path, monkeypatch):
    target = tmp_path / 'clean'
    old = _publish(target, 'old')
    original = snapshots._fsync_tree
    def mutate(stage):
        original(stage)
        (stage / 'first.txt').write_text('injected mutation', encoding='utf-8')
    monkeypatch.setattr(snapshots, '_fsync_tree', mutate)
    with pytest.raises(snapshots.SnapshotPublishError, match='inventory/hash mismatch'):
        _publish(target, 'new')
    assert snapshots.resolve_directory_snapshot(target) == old
