"""Smoke tests for scripts/reclaim_disk.py.

The tool deletes only the redundant intermediates (base/, covers/) of runs
that are BOTH packaged and older than --keep-days; manifest.json (the
cross-run diversity memory) and package/ are never touched, unpackaged or
recent runs are skipped entirely, and the default mode is a dry-run report.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

import scripts.reclaim_disk as R


def _mk_run(series: Path, name: str, *, packaged: bool, old: bool) -> Path:
    """A run dir with base/ + covers/ intermediates and a manifest; optionally
    a package/ tree and an mtime older than any sane --keep-days window."""
    run = series / name
    (run / "base").mkdir(parents=True)
    (run / "base" / "ad01_scene_s1.png").write_bytes(b"x" * 2048)
    (run / "covers").mkdir()
    (run / "covers" / "cover01_s2.png").write_bytes(b"y" * 1024)
    (run / "manifest.json").write_text("{}")
    if packaged:
        pkg = run / "package" / "Folder" / "public"
        pkg.mkdir(parents=True)
        (pkg / "keeper.png").write_bytes(b"z" * 512)
    if old:
        stale = time.time() - 30 * 86400
        os.utime(run, (stale, stale))
    return run


@pytest.fixture
def series_tree(tmp_path, monkeypatch):
    """Point the module constants at a tmp tree with the three canonical
    cases: old+packaged (reclaimable), old+unpackaged, recent+packaged."""
    series = tmp_path / "output" / "art_series"
    series.mkdir(parents=True)
    monkeypatch.setattr(R, "ROOT", tmp_path)
    monkeypatch.setattr(R, "SERIES_DIR", series)
    return (
        _mk_run(series, "20260101_000000", packaged=True, old=True),
        _mk_run(series, "20260102_000000", packaged=False, old=True),
        _mk_run(series, "20260810_000000", packaged=True, old=False),
    )


def test_dry_run_default_deletes_nothing(series_tree, monkeypatch, capsys):
    """No --apply → report only; every file survives."""
    monkeypatch.setattr(sys, "argv", ["reclaim_disk.py"])
    assert R.main() == 0
    out = capsys.readouterr().out
    assert "would delete" in out
    assert "re-run with --apply" in out
    for run in series_tree:
        assert (run / "base" / "ad01_scene_s1.png").exists()
        assert (run / "covers").is_dir()


def test_apply_reclaims_only_old_packaged(series_tree, monkeypatch, capsys):
    """--apply drops ONLY the old packaged run's intermediates; its
    manifest.json + package/ stay, and the other runs are untouched."""
    old_packaged, old_unpackaged, recent = series_tree
    monkeypatch.setattr(sys, "argv", ["reclaim_disk.py", "--apply"])
    assert R.main() == 0
    assert "deleted" in capsys.readouterr().out
    assert not (old_packaged / "base").exists()
    assert not (old_packaged / "covers").exists()
    assert (old_packaged / "manifest.json").exists()      # diversity memory
    assert (old_packaged / "package" / "Folder" / "public" / "keeper.png").exists()
    # unpackaged (may be mid-pipeline) and recent runs: never touched
    assert (old_unpackaged / "base").is_dir()
    assert (old_unpackaged / "covers").is_dir()
    assert (recent / "base").is_dir()
    assert (recent / "covers").is_dir()


def test_nothing_reclaimable_message(tmp_path, monkeypatch, capsys):
    """Empty series dir → graceful 'nothing reclaimable' report, exit 0."""
    series = tmp_path / "output" / "art_series"
    series.mkdir(parents=True)
    monkeypatch.setattr(R, "ROOT", tmp_path)
    monkeypatch.setattr(R, "SERIES_DIR", series)
    monkeypatch.setattr(sys, "argv", ["reclaim_disk.py"])
    assert R.main() == 0
    assert "nothing reclaimable" in capsys.readouterr().out
