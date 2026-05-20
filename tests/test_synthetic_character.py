"""Round-16 (2026-05-21) — tests for the synthetic-character fallback
in ``PipelineEngine._synthetic_character_for_scene``.

The 2026-05-21 Qwen3.5-9b MLX prep run showed every scene's prompt
build failing with "character is missing base_prompt: ''" because:

  - Scene Pydantic schema declares ``subject_detail`` only via
    ``extra="allow"`` — not required, so the LLM may skip it.
  - SeriesPlan schema does NOT declare ``subject_description`` at all
    (also extra="allow") — the LLM may skip it too.

When both were empty Qwen3.5-9b produced 21 scenes with 0 prompts
persisted. The defensive fallback synthesizes a base_prompt from the
scene's pose + camera + expression so the build always succeeds.
"""

from __future__ import annotations

from src.core.engine import PipelineEngine


def _call(series_plan, scene, style_profile=None):
    return PipelineEngine._synthetic_character_for_scene(
        series_plan=series_plan,
        style_profile=style_profile or {},
        scene=scene,
    )


# ── Primary sources (existing behaviour — make sure we didn't regress) ─


def test_subject_detail_per_scene_wins():
    """When the scene has subject_detail, it overrides everything else."""
    out = _call(
        series_plan={"subject_description": "Series-level woman"},
        scene={
            "subject_detail": "Her bare shoulders catch warm light",
            "pose": "standing",
        },
    )
    assert out["base_prompt"] == "Her bare shoulders catch warm light"


def test_subject_description_series_level_used_when_scene_lacks_detail():
    out = _call(
        series_plan={"subject_description": "Adult woman, mature features"},
        scene={"pose": "standing"},  # no subject_detail
    )
    assert out["base_prompt"] == "Adult woman, mature features"


def test_subject_bias_used_when_subject_description_empty():
    """Niche mode emits subject_bias instead of subject_description."""
    out = _call(
        series_plan={"subject_bias": "Niche-mode subject"},
        scene={"pose": "standing"},
    )
    assert out["base_prompt"] == "Niche-mode subject"


def test_style_keywords_prepended_when_present():
    """Style mode prepends style_keywords to subject."""
    out = _call(
        series_plan={
            "style_keywords": "cinematic, dramatic",
            "subject_description": "adult woman",
        },
        scene={"pose": "standing"},
    )
    assert out["base_prompt"] == "cinematic, dramatic, adult woman"


# ── Round-16 fallback: both primary sources empty ──────────────────


def test_fallback_synthesizes_from_scene_pose_camera_expression():
    """The Qwen3.5-9b MLX failure case — subject_detail empty AND
    subject_description empty. The fallback assembles a viable
    base_prompt from scene fields so PromptBuilder doesn't raise."""
    out = _call(
        series_plan={"subject_description": ""},
        scene={
            "subject_detail": "",
            "pose": "standing confident",
            "camera": "medium shot",
            "expression": "soft contemplative gaze",
        },
    )
    bp = out["base_prompt"]
    assert bp
    # "adult woman" prepended (single-female invariant).
    assert bp.startswith("adult woman")
    # All three scene fields woven in.
    assert "standing confident" in bp
    assert "medium shot" in bp
    assert "soft contemplative gaze" in bp


def test_fallback_with_only_pose():
    """When only pose is present, still produce a valid anchor."""
    out = _call(
        series_plan={"subject_description": ""},
        scene={"pose": "kneeling"},
    )
    assert out["base_prompt"] == "adult woman, kneeling"


def test_fallback_hard_floor_when_no_scene_fields():
    """Defensive — even with empty scene, we never emit base_prompt=''.
    This branch should be unreachable in production (scenes always
    have pose at minimum) but the floor guards against an upstream
    regression."""
    out = _call(
        series_plan={"subject_description": ""},
        scene={},
    )
    assert out["base_prompt"] == "adult woman"


def test_fallback_handles_none_scene_defensively():
    """Engine should never pass scene=None, but guard against it."""
    out = _call(
        series_plan={"subject_description": ""},
        scene=None,
    )
    assert out["base_prompt"] == "adult woman"


# ── Negative-prompt threading ────────────────────────────────────


def test_negative_prompt_threads_from_style_profile_in_all_paths():
    """Whether the primary path or fallback fires, the negative prompt
    must still come from the style_profile's base_negative_prompt."""
    sp = {"base_negative_prompt": "low quality, blurry"}
    # Primary path (subject_detail present)
    out_primary = _call(
        series_plan={}, scene={"subject_detail": "x"}, style_profile=sp,
    )
    assert out_primary["negative_prompt"] == "low quality, blurry"
    # subject_description path
    out_desc = _call(
        series_plan={"subject_description": "y"},
        scene={"pose": "z"},
        style_profile=sp,
    )
    assert out_desc["negative_prompt"] == "low quality, blurry"
    # Fallback path
    out_fb = _call(
        series_plan={"subject_description": ""},
        scene={"pose": "standing"},
        style_profile=sp,
    )
    assert out_fb["negative_prompt"] == "low quality, blurry"


def test_negative_prompt_empty_when_no_style_profile_default():
    """No base_negative_prompt on the profile → negative_prompt is ""."""
    out = _call(
        series_plan={"subject_description": "x"},
        scene={"pose": "y"},
        style_profile={},
    )
    assert out["negative_prompt"] == ""
