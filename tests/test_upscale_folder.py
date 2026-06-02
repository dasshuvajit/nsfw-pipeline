"""Standalone 4K folder upscaler — upscale_by math + per-image wiring."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import scripts.upscale_folder as U


@pytest.mark.parametrize("long_edge,target,expected", [
    (1440, 3840, 2.67),   # 3840/1440=2.667 -> ceil 2-dp -> 2.67; 1440*2.67=3845 >=3840
    (1920, 3840, 2.0),    # exact
    (1152, 3840, 3.34),   # 3.333 -> 3.34; 1152*3.34=3847 >=3840
    (3840, 3840, 1.0),    # already at target
    (4000, 3840, 1.0),    # already over -> clamp, never downscale
    (0, 3840, 1.0),       # guard
])
def test_compute_upscale_by(long_edge, target, expected):
    assert U.compute_upscale_by(long_edge, target) == expected


def test_upscale_by_always_reaches_target():
    for le in range(800, 3900, 137):
        ub = U.compute_upscale_by(le, 3840)
        assert le * ub >= 3840 - 1e-6, f"{le}*{ub} < 3840"


class _FakeClient:
    def __init__(self, src: Path):
        self.src = src
    def render_single_with_retry(self, wf, timeout=2400):
        return [SimpleNamespace(type="output", file_path=self.src)]


class _FakeBuilder:
    def __init__(self):
        self.calls = []
    def build_image_stage(self, *, external_template, image_path, seed=None,
                          upscale_by=None, largest_size=None):
        self.calls.append({"template": external_template, "image_path": image_path,
                           "seed": seed, "upscale_by": upscale_by})
        return {}


def _png(path: Path, size):
    Image.new("RGB", size, (100, 80, 60)).save(path)


def test_upscale_folder_wiring(tmp_path):
    indir = tmp_path / "keepers"
    indir.mkdir()
    _png(indir / "a.png", (1152, 1440))   # long edge 1440 -> x2.67
    _png(indir / "b.png", (1440, 1152))   # long edge 1440 -> x2.67
    src_out = tmp_path / "comfy.png"
    src_out.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    builder, client = _FakeBuilder(), _FakeClient(src_out)
    out_dir = tmp_path / "keepers_4k"

    results = U.upscale_folder(
        indir, out_dir, builder=builder, client=client,
        template="templates/sdxl/upscale_4k.json", target=3840, seed=7,
        glob="*.png", skip_existing=False)

    assert len(results) == 2
    # outputs written with _4k suffix
    assert (out_dir / "a_4k.png").exists()
    assert (out_dir / "b_4k.png").exists()
    # builder called with absolute path + correct computed upscale_by + seed
    assert len(builder.calls) == 2
    for c in builder.calls:
        assert Path(c["image_path"]).is_absolute()
        assert c["upscale_by"] == 2.67
        assert c["seed"] == 7
        assert c["template"] == "templates/sdxl/upscale_4k.json"


def test_skip_existing(tmp_path):
    indir = tmp_path / "keepers"
    indir.mkdir()
    _png(indir / "a.png", (1152, 1440))
    out_dir = tmp_path / "keepers_4k"
    out_dir.mkdir()
    (out_dir / "a_4k.png").write_bytes(b"existing")
    src_out = tmp_path / "comfy.png"
    src_out.write_bytes(b"x")
    builder, client = _FakeBuilder(), _FakeClient(src_out)

    results = U.upscale_folder(
        indir, out_dir, builder=builder, client=client,
        template="t.json", target=3840, seed=None, glob="*.png",
        skip_existing=True)
    assert results[0]["skipped"] is True
    assert builder.calls == []   # nothing rendered
