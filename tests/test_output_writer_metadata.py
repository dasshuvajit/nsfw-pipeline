"""PNG metadata embedding — Phase 4b.

The pipeline writes two tEXt chunks on every exported PNG:

* ``parameters`` — AUTOMATIC1111 / Civitai format string. Civitai +
  sd-prompt-reader + A1111 webui all read this.
* ``nsfw_pipeline`` — pipeline-native JSON carrying the structured
  facets, vocab_version, scene/series linkage. Round-trips back into
  this codebase even if the SQLite DB is unavailable.

Both chunks survive the watermarker re-save (the watermarker preserves
existing PNG chunks). These tests cover the building / writing /
reading helpers in isolation; integration with the engine is covered
by the regression smoke fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from src.render.metadata import (
    build_a1111_parameters,
    build_pipeline_metadata,
    read_png_metadata,
    write_png_metadata,
)


# ── A1111 parameters string ────────────────────────────────────────


def test_a1111_full_parameters_string():
    out = build_a1111_parameters(
        prompt_text="a portrait of a woman",
        negative_prompt="blurry, low-res",
        steps=30,
        sampler="DPM++ 2M",
        scheduler="Karras",
        cfg_scale=6.0,
        seed=12345,
        model="gonzalomo_photo_v70",
        size=(832, 1216),
        clip_skip=2,
    )
    # Positive prompt on first line
    lines = out.split("\n")
    assert lines[0] == "a portrait of a woman"
    # Negative prompt on second line with the canonical prefix
    assert lines[1] == "Negative prompt: blurry, low-res"
    # Tail line carries the comma-separated parameters
    tail = lines[2]
    assert "Steps: 30" in tail
    assert "Sampler: DPM++ 2M Karras" in tail
    assert "CFG scale: 6.0" in tail
    assert "Seed: 12345" in tail
    assert "Size: 832x1216" in tail
    assert "Model: gonzalomo_photo_v70" in tail
    assert "Clip skip: 2" in tail


def test_a1111_drops_missing_fields():
    """Partial data is acceptable — A1111 / Civitai readers tolerate it."""
    out = build_a1111_parameters(
        prompt_text="a woman",
        negative_prompt=None,
        steps=20,
        sampler="Euler",
        scheduler=None,
        cfg_scale=None,
        seed=None,
        model=None,
    )
    # No "Negative prompt:" line when negative is None
    assert "Negative prompt:" not in out
    # Tail has only the populated fields
    tail = out.split("\n")[-1]
    assert tail == "Steps: 20, Sampler: Euler"


def test_a1111_no_tail_when_all_params_missing():
    out = build_a1111_parameters(
        prompt_text="a woman",
        negative_prompt=None,
        steps=None,
        sampler=None,
        scheduler=None,
        cfg_scale=None,
        seed=None,
        model=None,
    )
    assert out == "a woman"


# ── Pipeline metadata JSON chunk ───────────────────────────────────


def test_pipeline_metadata_carries_repro_set():
    raw = build_pipeline_metadata(
        vocab_version=1,
        family="sdxl",
        model_id="gonzalomo_photo_v70",
        scene_id="scn_001",
        series_id="ser_001",
        prompt_hash="abc123",
        seed=42,
        sampler="DPM++ 2M",
        scheduler="Karras",
        steps=30,
        cfg=6.0,
        content_level="T2_implied",
        structured_facet={
            "lighting_directive": "LIGHT_REMBRANDT",
            "realism_camera": "CAMERA_SONY_A7RV",
            "ignored_none": None,  # dropped from output
        },
    )
    parsed = json.loads(raw)
    assert parsed["vocab_version"] == 1
    assert parsed["family"] == "sdxl"
    assert parsed["model_id"] == "gonzalomo_photo_v70"
    assert parsed["scene_id"] == "scn_001"
    assert parsed["seed"] == 42
    assert parsed["content_level"] == "T2_implied"
    # Structured facet survives, None values dropped
    assert parsed["structured_facet"] == {
        "lighting_directive": "LIGHT_REMBRANDT",
        "realism_camera": "CAMERA_SONY_A7RV",
    }


def test_pipeline_metadata_omits_structured_facet_when_empty():
    raw = build_pipeline_metadata(
        vocab_version=1,
        family="sdxl", model_id="x",
        scene_id=None, series_id=None, prompt_hash=None,
        seed=None, sampler=None, scheduler=None, steps=None,
        cfg=None, content_level=None,
        structured_facet=None,
    )
    parsed = json.loads(raw)
    assert "structured_facet" not in parsed


def test_pipeline_metadata_extra_field_propagates():
    raw = build_pipeline_metadata(
        vocab_version=2,
        family="flux", model_id="gonzalomo_flux_v30",
        scene_id="scn", series_id=None, prompt_hash=None,
        seed=1, sampler="euler", scheduler="simple",
        steps=4, cfg=1.0, content_level="T3_artnude",
        extra={"trigger_words_used": ["ultra_real_v4"]},
    )
    parsed = json.loads(raw)
    assert parsed["trigger_words_used"] == ["ultra_real_v4"]


# ── PNG round-trip ─────────────────────────────────────────────────


@pytest.fixture
def png_path(tmp_path: Path) -> Path:
    img = Image.new("RGB", (16, 16), color="red")
    p = tmp_path / "test.png"
    img.save(p, "PNG")
    return p


def test_write_then_read_round_trip(png_path: Path):
    write_png_metadata(
        png_path,
        a1111_parameters="prompt\nNegative prompt: bad\nSteps: 30",
        pipeline_metadata='{"vocab_version":1,"family":"sdxl"}',
    )
    chunks = read_png_metadata(png_path)
    assert "parameters" in chunks
    assert chunks["parameters"].startswith("prompt")
    assert "Steps: 30" in chunks["parameters"]
    assert "nsfw_pipeline" in chunks
    parsed = json.loads(chunks["nsfw_pipeline"])
    assert parsed["family"] == "sdxl"


def test_write_only_a1111_chunk(png_path: Path):
    write_png_metadata(png_path, a1111_parameters="hello world")
    chunks = read_png_metadata(png_path)
    assert chunks.get("parameters") == "hello world"
    assert "nsfw_pipeline" not in chunks


def test_write_only_pipeline_chunk(png_path: Path):
    write_png_metadata(png_path, pipeline_metadata='{"k": 1}')
    chunks = read_png_metadata(png_path)
    assert "parameters" not in chunks
    assert chunks.get("nsfw_pipeline") == '{"k": 1}'


def test_write_neither_is_noop(png_path: Path):
    """No args → no-op (no exception raised)."""
    write_png_metadata(png_path)
    chunks = read_png_metadata(png_path)
    # Neither chunk present
    assert "parameters" not in chunks
    assert "nsfw_pipeline" not in chunks


def test_write_preserves_unrelated_chunks(png_path: Path):
    """An existing chunk we don't touch must survive a metadata write."""
    from PIL import PngImagePlugin
    info = PngImagePlugin.PngInfo()
    info.add_text("comment", "previous tag")
    img = Image.new("RGB", (16, 16))
    img.save(png_path, "PNG", pnginfo=info)
    # Now write our chunks
    write_png_metadata(png_path, a1111_parameters="new")
    chunks = read_png_metadata(png_path)
    assert chunks.get("comment") == "previous tag"
    assert chunks.get("parameters") == "new"


def test_write_skips_non_png(tmp_path: Path):
    """JPEG files are silently skipped — A1111 PNG chunks don't apply."""
    p = tmp_path / "test.jpg"
    Image.new("RGB", (16, 16)).save(p, "JPEG", quality=80)
    # No exception, no chunks written
    write_png_metadata(p, a1111_parameters="ignored")
    # read_png_metadata returns empty dict for non-PNG
    assert read_png_metadata(p) == {}


def test_write_to_missing_file_logs_warning(tmp_path: Path, caplog):
    """Trying to write metadata to a missing file logs WARNING but
    doesn't crash."""
    import logging
    p = tmp_path / "does_not_exist.png"
    with caplog.at_level(logging.WARNING):
        write_png_metadata(p, a1111_parameters="x")
    assert any(
        "cannot open" in r.message.lower() for r in caplog.records
    )


def test_overwriting_chunks_replaces_previous_value(png_path: Path):
    write_png_metadata(png_path, a1111_parameters="first")
    write_png_metadata(png_path, a1111_parameters="second")
    chunks = read_png_metadata(png_path)
    assert chunks["parameters"] == "second"


# ── Audit fix B1: engine reads ctx.model_config (not ctx.model) ───


def test_a1111_parameters_capture_real_sampler_cfg_steps():
    """Regression — the engine's _embed_png_metadata path reads
    sampler/scheduler/steps/cfg/clip_skip off ctx.model_config (a
    ModelRegistryEntry). Pre-fix the call read ctx.model which doesn't
    exist; hasattr was False; metadata silently lost the render
    tuning. This test pins the A1111-format string contains the real
    values so the regression doesn't sneak back."""
    out = build_a1111_parameters(
        prompt_text="a portrait",
        negative_prompt="blurry",
        steps=30,
        sampler="dpmpp_sde",
        scheduler="karras",
        cfg_scale=5.0,
        seed=42,
        model="cyberrealistic_pony_v170",
        size=(896, 1152),
        clip_skip=2,
    )
    # Tail line carries every field — none silently None.
    tail = out.split("\n")[-1]
    assert "Steps: 30" in tail
    assert "Sampler: dpmpp_sde karras" in tail
    assert "CFG scale: 5.0" in tail
    assert "Seed: 42" in tail
    assert "Size: 896x1152" in tail
    assert "Model: cyberrealistic_pony_v170" in tail
    assert "Clip skip: 2" in tail


def test_a1111_with_all_nones_leaves_only_prompt():
    """Pre-fix bug shape: every render-tuning field None. This test
    pins that the regressed shape is recognisably impoverished — so
    if we ever revert to ctx.model the loss surface is visible.

    (The fix in engine._embed_png_metadata reads ctx.model_config.*
    so this output should NOT happen on a real render.)"""
    out = build_a1111_parameters(
        prompt_text="a portrait",
        negative_prompt="blurry",
        steps=None, sampler=None, scheduler=None, cfg_scale=None,
        seed=None, model=None, size=None, clip_skip=None,
    )
    # No tail line — only positive + negative.
    assert out.count("\n") == 1
    assert "Steps:" not in out
    assert "Sampler:" not in out
    assert "Model:" not in out
