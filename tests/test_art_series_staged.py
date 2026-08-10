"""Base-render Phase-2 orchestration in scripts/art_series.py.

Tests `_render_stage_base` with a fake builder/client: stage 1 records
`base_path`+`seed` per image and renders each prompt at its own orientation.
(The staged SDXL refine tests were retired with the refine stage — archived
2026-08, see legacy/.) Also covers the ComfyUI self-heal (`_ensure_comfyui`
+ the in-stage heal/breaker interplay) and the T1/T2 drift auto-reroll
(`_reroll_drifted`).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.art_series as A
from src.render.comfyui_client import ComfyUIError


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
    def build_external(self, *, external_template, prompt_text, negative_prompt,
                       resolution, seed):
        return {"_t": external_template, "_seed": seed}


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
        rows, builder=builder, client=client, base_template="templates/zimage/base.json",
        negative="neg", resolution=(896, 1152), base_seed=100, seeds=1, random_seeds=False,
        dest_dir=out_dir / "base", out_dir=out_dir, prefix="ad")
    assert len(manifest) == 2
    for i, entry in enumerate(manifest):
        assert len(entry["images"]) == 1
        im = entry["images"][0]
        assert im["base_path"].startswith("base/")
        assert "path" not in im            # path is set later in main(), not by base
        # Per-PROMPT seeds (2026-06): seed = base_seed + idx*seeds + k — each
        # prompt gets its OWN initial latent noise (was base_seed+k for all).
        assert im["seed"] == 100 + i
    # files actually written under base/
    assert (out_dir / "base").is_dir()
    assert len(list((out_dir / "base").glob("*.png"))) == 2


def test_stage_base_uses_per_row_orientation(tmp_path):
    """Each prompt renders at ITS OWN orientation (the LLM chose it); base
    resolution comes from rp.base_resolution per row (Phase 1)."""
    out_dir = tmp_path / "series"
    rp = {"base_resolution": {"portrait": [896, 1152], "square": [1024, 1024],
                              "landscape": [1152, 896]}}
    mixed = [
        {"look": "a", "prompt": "p0", "orientation": "landscape", "shot_type": "full_body"},
        {"look": "b", "prompt": "p1", "orientation": "square", "shot_type": "close_up"},
        {"look": "c", "prompt": "p2"},  # no orientation → default_orientation
    ]

    captured = []

    class _CapBuilder(_FakeBuilder):
        def build_external(self, *, external_template, prompt_text, negative_prompt,
                           resolution, seed):
            captured.append(resolution)
            return {}

    builder, client = _CapBuilder(), _FakeClient(_src_png(tmp_path))
    manifest = A._render_stage_base(
        mixed, builder=builder, client=client, base_template="t.json",
        negative="n", resolution=(896, 1152), base_seed=1, seeds=1, random_seeds=False,
        dest_dir=out_dir / "base", out_dir=out_dir, prefix="ad",
        rp=rp, default_orientation="portrait")
    assert captured == [(1152, 896), (1024, 1024), (896, 1152)]  # per-row
    assert manifest[0]["orientation"] == "landscape"
    assert manifest[1]["orientation"] == "square"
    assert manifest[2]["orientation"] == "portrait"   # fallback
    assert manifest[0]["resolution"] == [1152, 896]


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


# ── 5.1: ComfyUI self-heal ──────────────────────────────────────────


def test_ensure_comfyui_noop_when_up(monkeypatch):
    """Server already answering /system_stats → True with NO side effects
    (no pkill, no relaunch)."""
    monkeypatch.setattr(A, "_comfyui_stats_up", lambda url, timeout=3.0: True)

    def _boom(*a, **k):
        raise AssertionError("no subprocess call when the server is up")
    monkeypatch.setattr(A.subprocess, "run", _boom)
    monkeypatch.setattr(A.subprocess, "Popen", _boom)
    assert A._ensure_comfyui("http://x:8188", None) is True


def test_ensure_comfyui_relaunch_sequence(tmp_path, monkeypatch):
    """Down → pkill zombie + detached nohup relaunch (Homebrew python3, MPS
    watermark env, start_new_session) + poll until /system_stats answers.
    All side effects are isolated in this one function (monkeypatched here)."""
    stats = iter([False, False, True])   # initial probe, poll #1, poll #2
    monkeypatch.setattr(A, "_comfyui_stats_up",
                        lambda url, timeout=3.0: next(stats))
    ran, popped = [], []
    monkeypatch.setattr(A.subprocess, "run",
                        lambda cmd, check=False: ran.append(cmd))
    monkeypatch.setattr(A.subprocess, "Popen",
                        lambda cmd, **kw: popped.append((cmd, kw)))
    monkeypatch.setattr(A.time, "sleep", lambda s: None)
    monkeypatch.setattr(A, "_SELFHEAL_LOG", tmp_path / "heal.log")
    assert A._ensure_comfyui("http://x:8188", tmp_path) is True
    assert ran[0][:2] == ["pkill", "-9"]
    cmd, kw = popped[0]
    assert cmd == ["nohup", "python3", "main.py"]
    assert kw["cwd"] == str(tmp_path)
    assert kw["start_new_session"] is True
    assert kw["env"]["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.0"


def test_selfheal_retries_image_and_resets_breaker(tmp_path, monkeypatch):
    """In-stage heal: a connectivity death triggers _ensure_comfyui; a
    successful heal retries the SAME image (same seed) and does NOT count
    toward the 3-strike breaker. Heals cap at _SELFHEAL_MAX_PER_STAGE —
    three healed-or-not conn failures here, yet no breaker abort (only the
    single unhealed strike counted)."""
    out_dir = tmp_path / "series"
    src = _src_png(tmp_path)

    class _FlakyClient:
        """Connectivity error on the FIRST call for each seed, then fine."""
        def __init__(self):
            self.calls, self.seen = 0, set()

        def render_single_with_retry(self, wf, timeout=1800):
            self.calls += 1
            if wf["_seed"] not in self.seen:
                self.seen.add(wf["_seed"])
                raise ComfyUIError("connection refused")
            return [SimpleNamespace(type="output", file_path=src)]

    heals = []
    monkeypatch.setattr(A, "_ensure_comfyui",
                        lambda url, d: heals.append((url, d)) or True)
    rows = [{"look": f"l{i}", "prompt": f"p{i}"} for i in range(3)]
    manifest = A._render_stage_base(
        rows, builder=_FakeBuilder(), client=_FlakyClient(), base_template="t.json",
        negative="n", resolution=(896, 1152), base_seed=1, seeds=1,
        random_seeds=False, dest_dir=out_dir / "base", out_dir=out_dir,
        prefix="ad", selfheal_comfy_dir=tmp_path)
    assert len(heals) == A._SELFHEAL_MAX_PER_STAGE == 2     # per-stage cap
    assert heals[0][1] == tmp_path                          # comfy_dir threaded
    # images 0+1: healed then retried at the SAME per-prompt seed
    assert [im["seed"] for im in manifest[0]["images"]] == [1]
    assert [im["seed"] for im in manifest[1]["images"]] == [2]
    # image 2: heal budget spent → single (non-fatal) strike, image lost
    assert manifest[2]["images"] == []


def test_selfheal_off_preserves_breaker_abort(tmp_path, monkeypatch):
    """selfheal_comfy_dir=None (--no-comfy-selfheal): _ensure_comfyui is NEVER
    called and 3 consecutive connectivity failures abort exactly as before."""
    class _DeadClient:
        def render_single_with_retry(self, wf, timeout=1800):
            raise ComfyUIError("connection refused")

    called = []
    monkeypatch.setattr(A, "_ensure_comfyui",
                        lambda *a: called.append(a) or True)
    rows = [{"look": f"l{i}", "prompt": f"p{i}"} for i in range(4)]
    with pytest.raises(RuntimeError, match="ComfyUI unreachable"):
        A._render_stage_base(
            rows, builder=_FakeBuilder(), client=_DeadClient(),
            base_template="t.json", negative="n", resolution=(896, 1152),
            base_seed=1, seeds=1, random_seeds=False,
            dest_dir=tmp_path / "series" / "base", out_dir=tmp_path / "series",
            prefix="ad", selfheal_comfy_dir=None)
    assert called == []


# ── 5.2: T1/T2 drift auto-reroll ────────────────────────────────────


def _drift_manifest(out_dir: Path, n: int) -> list[dict]:
    """A curated manifest of n keeper images that exist on disk under base/."""
    base = out_dir / "base"
    base.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i in range(n):
        p = base / f"ad{i + 1:02d}_look_s{100 + i}.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
        rel = str(p.relative_to(out_dir))
        manifest.append({
            "index": i, "look": "look", "prompt": f"p{i}",
            "orientation": "portrait", "resolution": [896, 1152],
            "images": [{"base_path": rel, "path": rel, "review_path": rel,
                        "seed": 100 + i, "keeper": True}],
        })
    return manifest


def test_drift_reroll_replaces_drifted_keeper(tmp_path, monkeypatch):
    """A nudity-flagged funnel keeper is re-rendered with a FRESH random seed
    at its own resolution, re-screened, and swapped into the manifest (file
    lands in base/ with an _rr suffix; NSFW LoRA stays clamped OFF)."""
    out_dir = tmp_path / "series"
    manifest = _drift_manifest(out_dir, 1)
    orig_rel = manifest[0]["images"][0]["path"]
    # original screens NUDE; the rerolled _rr candidate screens clean
    monkeypatch.setattr(A, "_nudity_hit",
                        lambda p, dets: "_rr" not in Path(p).name)
    wfs = []

    class _CapBuilder:
        def build_external(self, *, external_template, prompt_text,
                           negative_prompt, resolution, seed):
            wf = {"_seed": seed, "_res": resolution,
                  "lora_nsfw": {"inputs": {"strength_model": 0.8}}}
            wfs.append(wf)
            return wf

    replaced, still = A._reroll_drifted(
        manifest, out_dir=out_dir, builder=_CapBuilder(),
        client=_FakeClient(_src_png(tmp_path)), base_template="t.json",
        negative="n", rp=None, default_orientation="portrait",
        tier="T2_implied")
    assert (replaced, still) == (1, 0)
    im = manifest[0]["images"][0]
    assert "_rr" in Path(im["path"]).name
    assert im["path"] == im["base_path"] == im["review_path"]
    assert im["rerolled_from"] == orig_rel
    assert (out_dir / im["path"]).exists()
    assert im["seed"] == wfs[0]["_seed"] != 100          # fresh random seed
    assert wfs[0]["_res"] == (896, 1152)                 # same resolution
    assert wfs[0]["lora_nsfw"]["inputs"]["strength_model"] == 0.0


def test_drift_reroll_budget_exhausts_to_quarantine(tmp_path, monkeypatch):
    """Replacements that never screen clean: ≤2 attempts per image and ≤4
    per series, then the keepers fall through to the packaging quarantine
    untouched (0 replaced, all still drifted)."""
    out_dir = tmp_path / "series"
    manifest = _drift_manifest(out_dir, 3)
    monkeypatch.setattr(A, "_nudity_hit", lambda p, dets: True)  # never clean
    client = _FakeClient(_src_png(tmp_path))
    replaced, still = A._reroll_drifted(
        manifest, out_dir=out_dir, builder=_FakeBuilder(), client=client,
        base_template="t.json", negative="n", rp=None,
        default_orientation="portrait", tier="T1_suggestive")
    assert (replaced, still) == (0, 3)
    # 2 + 2 + 0 renders — the series budget stops image 3 cold
    assert client.calls == A._DRIFT_REROLL_MAX_PER_SERIES == 4
    for e in manifest:                       # keepers untouched for quarantine
        assert "_rr" not in e["images"][0]["path"]
        assert "rerolled_from" not in e["images"][0]


def test_drift_reroll_never_touches_explicit_tiers(tmp_path, monkeypatch):
    """T3/T4 (nudity IS the product): no screening, no rendering, (0, 0)."""
    out_dir = tmp_path / "series"
    manifest = _drift_manifest(out_dir, 1)

    def _boom(*a, **k):
        raise AssertionError("nudity screen must not run at T3/T4")
    monkeypatch.setattr(A, "_nudity_hit", _boom)

    class _NoRenderClient:
        def render_single_with_retry(self, wf, timeout=1800):
            raise AssertionError("render must not run at T3/T4")

    for tier in sorted(A._EXPLICIT_TIERS):
        assert A._reroll_drifted(
            manifest, out_dir=out_dir, builder=_FakeBuilder(),
            client=_NoRenderClient(), base_template="t.json", negative="n",
            rp=None, default_orientation="portrait", tier=tier) == (0, 0)
