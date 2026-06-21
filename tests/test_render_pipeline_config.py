"""Render-pipeline config resolution: parse, precedence, base_resolution merge."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.render.render_pipeline import (
    DEFAULTS,
    base_resolution_for,
    resolve_render_pipeline,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_yaml_block_parses():
    cfg = yaml.safe_load((PROJECT_ROOT / "config" / "pipeline.yaml").read_text())
    rp = cfg.get("render_pipeline")
    assert rp is not None, "pipeline.yaml missing render_pipeline block"
    assert rp["base_template"] == "templates/chroma/base.json"
    assert rp["refine_template"] == "templates/chroma/refine.json"
    assert rp["refine_template_t4"] == "templates/chroma/refine_T4.json"  # T4 vagina variant
    assert rp["upscale_template"] == "templates/sdxl/upscale_4k.json"
    assert rp["enable_refine"] is False  # base-only by default; refine is opt-in
    assert rp["target_4k_long_edge"] == 3840
    assert rp["base_resolution"]["portrait"] == [896, 1152]  # reverted to native


def test_defaults_when_no_layers():
    rp = resolve_render_pipeline()
    assert rp["base_template"] == DEFAULTS["base_template"]
    assert rp["enable_refine"] is False  # base-only by default; refine is opt-in
    assert rp["base_resolution"]["portrait"] == [896, 1152]


def test_global_over_defaults():
    rp = resolve_render_pipeline(global_rp={"enable_refine": False,
                                            "target_4k_long_edge": 4096})
    assert rp["enable_refine"] is False
    assert rp["target_4k_long_edge"] == 4096
    # untouched keys keep defaults
    assert rp["base_template"] == DEFAULTS["base_template"]


def test_model_over_global():
    rp = resolve_render_pipeline(
        global_rp={"refine_template": "templates/chroma/refine.json"},
        model_rp={"refine_template": "templates/chroma/refine_alt.json"},
    )
    assert rp["refine_template"] == "templates/chroma/refine_alt.json"


def test_cli_over_model_and_global():
    rp = resolve_render_pipeline(
        global_rp={"base_template": "g.json"},
        model_rp={"base_template": "m.json"},
        cli={"base_template": "cli.json"},
    )
    assert rp["base_template"] == "cli.json"


def test_none_cli_value_does_not_clobber():
    rp = resolve_render_pipeline(
        global_rp={"base_template": "g.json"},
        cli={"base_template": None, "enable_refine": False},
    )
    assert rp["base_template"] == "g.json"   # None ignored
    assert rp["enable_refine"] is False


def test_base_resolution_merges_per_orientation():
    rp = resolve_render_pipeline(
        global_rp={"base_resolution": {"portrait": [1024, 1536]}},
    )
    assert rp["base_resolution"]["portrait"] == [1024, 1536]   # overridden
    assert rp["base_resolution"]["square"] == [1024, 1024]      # inherited default
    assert rp["base_resolution"]["landscape"] == [1152, 896]    # inherited default


def test_base_resolution_for_helper():
    rp = resolve_render_pipeline()
    assert base_resolution_for(rp, "portrait") == (896, 1152)
    assert base_resolution_for(rp, "square") == (1024, 1024)
    assert base_resolution_for(rp, "landscape") == (1152, 896)
    # 2026-06-22 widescreen 16:9 + story 9:16 lanes (off-native ~1MP, MPS-safe)
    assert base_resolution_for(rp, "widescreen") == (1360, 768)
    assert base_resolution_for(rp, "story") == (768, 1360)
    # unknown orientation -> portrait fallback (NOTE: base_resolution_for fails OPEN,
    # so every base_resolution table must carry every orientation or it mis-renders)
    assert base_resolution_for(rp, "weird") == (896, 1152)


def test_all_orientations_are_mps_safe_and_in_every_table():
    """Every orientation must (1) appear in all three base_resolution tables — the
    DEFAULTS, the pipeline.yaml override, and the --hires table — because
    base_resolution_for silently falls back to portrait for a missing key; and
    (2) stay MPS-safe: latent tokens (w/16)*(h/16) well under the ~12288 crash.
    (_HIRES_BASE_RESOLUTION is a local inside main(), so it's checked via source.)"""
    import yaml as _yaml
    import scripts.art_director as AD
    rp = resolve_render_pipeline()
    pyaml = _yaml.safe_load((PROJECT_ROOT / "config" / "pipeline.yaml").read_text())
    py_br = pyaml["render_pipeline"]["base_resolution"]
    import re as _re
    src = (PROJECT_ROOT / "scripts" / "art_series.py").read_text()
    hires_block = src.split("_HIRES_BASE_RESOLUTION = {", 1)[1].split("}", 1)[0]
    for o in AD.ORIENTATIONS_ALLOWED:
        w, h = base_resolution_for(rp, o)
        assert o in py_br, f"{o} missing from pipeline.yaml base_resolution"
        assert (w // 16) * (h // 16) < 12288, f"{o} {w}x{h} exceeds the MPS token ceiling"
        # the --hires bucket must ALSO be present and MPS-safe (it's the bigger one)
        m = _re.search(rf'"{o}":\s*\[(\d+),\s*(\d+)\]', hires_block)
        assert m, f"{o} missing from _HIRES_BASE_RESOLUTION"
        hw, hh = int(m.group(1)), int(m.group(2))
        assert (hw // 16) * (hh // 16) < 12288, f"hires {o} {hw}x{hh} exceeds the ceiling"
