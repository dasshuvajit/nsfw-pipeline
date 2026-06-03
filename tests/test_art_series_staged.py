"""Staged Phase-2 orchestration in scripts/art_series.py.

Tests the two stage functions (`_render_stage_base`, `_render_stage_refine`)
with a fake builder/client: stage 1 records `base_path`+`seed`; stage 2 sets
`path` (the review image) reusing the SAME seed; a refine failure falls back
to the base image so nothing is silently dropped; and the resulting manifest
shape still carries the `path` curation/packaging depend on.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.art_series as A


def _src_png(tmp_path: Path) -> Path:
    """A real on-disk file so shutil.copy in the stage functions works."""
    p = tmp_path / "comfy_out.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    return p


class _FakeClient:
    """Returns one 'output' RenderedImage-like object pointing at a real file."""
    def __init__(self, src: Path):
        self.src = src
        self.calls = 0

    def render_single_with_retry(self, wf, timeout=1800):
        self.calls += 1
        return [SimpleNamespace(type="output", file_path=self.src)]


class _FakeBuilder:
    def __init__(self):
        self.image_stage_calls = []  # (template, image_path, seed)

    def build_external(self, *, external_template, prompt_text, negative_prompt,
                       resolution, seed):
        return {"_t": external_template, "_seed": seed}

    def build_image_stage(self, *, external_template, image_path, seed=None,
                          upscale_by=None, largest_size=None):
        self.image_stage_calls.append((external_template, image_path, seed))
        return {"_t": external_template, "_img": image_path, "_seed": seed}


@pytest.fixture
def rows():
    return [
        {"look": "soft morning", "prompt": "p0", "audit_score": 9},
        {"look": "golden dusk", "prompt": "p1", "audit_score": 8},
    ]


def test_stage_base_records_base_path_and_seed(tmp_path, rows):
    out_dir = tmp_path / "series"
    builder, client = _FakeBuilder(), _FakeClient(_src_png(tmp_path))
    manifest = A._render_stage_base(
        rows, builder=builder, client=client, base_template="templates/chroma/base.json",
        negative="neg", resolution=(896, 1152), base_seed=100, seeds=1,
        dest_dir=out_dir / "base", out_dir=out_dir, prefix="ad")
    assert len(manifest) == 2
    for i, entry in enumerate(manifest):
        assert len(entry["images"]) == 1
        im = entry["images"][0]
        assert im["base_path"].startswith("base/")
        assert "path" not in im            # path is set by the refine stage, not base
        assert im["seed"] == 100
    # files actually written under base/
    assert (out_dir / "base").is_dir()
    assert len(list((out_dir / "base").glob("*.png"))) == 2


def test_stage_refine_sets_path_and_reuses_seed(tmp_path, rows):
    out_dir = tmp_path / "series"
    builder, client = _FakeBuilder(), _FakeClient(_src_png(tmp_path))
    manifest = A._render_stage_base(
        rows, builder=builder, client=client, base_template="templates/chroma/base.json",
        negative="neg", resolution=(896, 1152), base_seed=100, seeds=1,
        dest_dir=out_dir / "base", out_dir=out_dir, prefix="ad")
    A._render_stage_refine(
        manifest, builder=builder, client=client,
        refine_template="templates/chroma/refine.json",
        dest_dir=out_dir / "images", out_dir=out_dir)
    for entry in manifest:
        im = entry["images"][0]
        assert im["path"].startswith("images/")     # review image — what curation reads
        assert im["review_path"] == im["path"]
    # refine reused the SAME seed as the base render, and got an absolute path
    assert len(builder.image_stage_calls) == 2
    for tmpl, image_path, seed in builder.image_stage_calls:
        assert tmpl == "templates/chroma/refine.json"
        assert Path(image_path).is_absolute()
        assert seed == 100


def test_used_niche_tracking(tmp_path, monkeypatch):
    """--auto used-niche cycle file I/O: read empty, append, dedup, and a
    post-reset write (used=[]) starts a fresh single-entry cycle."""
    f = tmp_path / ".used_niches"
    monkeypatch.setattr(A, "USED_NICHES_FILE", f)
    assert A._read_used_niches() == []                  # missing file → empty
    A._record_used_niche([], "fine_art_figure_study")
    assert A._read_used_niches() == ["fine_art_figure_study"]
    A._record_used_niche(["fine_art_figure_study"], "goth_romantic")
    assert A._read_used_niches() == ["fine_art_figure_study", "goth_romantic"]
    # dedup: recording an already-present id is a no-op append
    A._record_used_niche(["fine_art_figure_study", "goth_romantic"], "goth_romantic")
    assert A._read_used_niches() == ["fine_art_figure_study", "goth_romantic"]
    # cycle reset: caller passes used=[] → file becomes just the fresh pick
    A._record_used_niche([], "old_hollywood_glamour")
    assert A._read_used_niches() == ["old_hollywood_glamour"]


def test_refine_failure_falls_back_to_base(tmp_path, rows):
    out_dir = tmp_path / "series"
    builder, client = _FakeBuilder(), _FakeClient(_src_png(tmp_path))
    manifest = A._render_stage_base(
        rows, builder=builder, client=client, base_template="templates/chroma/base.json",
        negative="neg", resolution=(896, 1152), base_seed=100, seeds=1,
        dest_dir=out_dir / "base", out_dir=out_dir, prefix="ad")

    def boom(**kwargs):
        raise RuntimeError("refine exploded")
    builder.build_image_stage = boom  # type: ignore[assignment]

    A._render_stage_refine(
        manifest, builder=builder, client=client,
        refine_template="templates/chroma/refine.json",
        dest_dir=out_dir / "images", out_dir=out_dir)
    for entry in manifest:
        im = entry["images"][0]
        assert im["path"] == im["base_path"]   # fell back to the base image
        assert im["review_path"] == im["base_path"]
