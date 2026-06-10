"""Round-17 (2026-05-21) tests for the LLM-helpers module shared
across mode planners.

Covers:
  - ``validate_scene_list`` min_count gating — the 2026-05-21 Qwen3.5-9b
    MLX prep shipped 8 scenes when 25 were asked for; min_count rejects
    under-shipped batches so retry can re-roll.
  - ``widen_compat_intersection`` is in test_aesthetic_anchor_check.py.
"""

from __future__ import annotations

from src.modes._llm_helpers import validate_scene_list


_VALID_SCENE = {
    "variation_axis": "pose",
    "pose": "standing",
    "camera": "medium shot",
    "camera_angle": "eye level",
    "lighting": "soft window light",
    "environment_detail": "ivy-covered wall",
    "mood_note": "contemplative",
}
_REQUIRED = {
    "variation_axis", "pose", "camera", "camera_angle",
    "lighting", "environment_detail", "mood_note",
}


def test_validate_returns_all_valid_scenes_when_above_min_count():
    validator = validate_scene_list(_REQUIRED, min_count=5)
    scenes = [_VALID_SCENE] * 10
    out = validator(scenes)
    assert out is not None
    assert len(out) == 10


def test_validate_returns_none_when_below_min_count():
    """Round-17 — under-shipped batches trigger retry."""
    validator = validate_scene_list(_REQUIRED, min_count=20)
    scenes = [_VALID_SCENE] * 8  # The Qwen3.5-9b failure case.
    out = validator(scenes)
    assert out is None  # Rejected → retry.


def test_validate_drops_invalid_scenes_then_checks_count():
    """Invalid scenes are dropped first; if the SURVIVING count is
    below min_count, reject. Pre-fix: validator counted ALL scenes
    including invalid ones; post-fix only the valid ones count."""
    bad_scene = {"pose": "x"}  # missing 6 required keys.
    validator = validate_scene_list(_REQUIRED, min_count=5)
    # 3 valid + 5 invalid = 3 surviving < min_count 5 → reject.
    scenes = [_VALID_SCENE] * 3 + [bad_scene] * 5
    assert validator(scenes) is None


def test_validate_default_min_count_is_1():
    """Back-compat — the default min_count=1 preserves pre-round-17
    behaviour where a single valid scene survives."""
    validator = validate_scene_list(_REQUIRED)
    assert validator([_VALID_SCENE]) == [_VALID_SCENE]
    assert validator([]) is None  # Zero valid scenes still rejected.


def test_validate_non_list_input_returns_none():
    validator = validate_scene_list(_REQUIRED, min_count=1)
    assert validator({"not": "a list"}) is None
    assert validator(None) is None
    assert validator("a string") is None


def test_validate_zero_valid_scenes_returns_none():
    """All scenes invalid → 0 surviving → reject regardless of
    min_count."""
    validator = validate_scene_list(_REQUIRED, min_count=1)
    assert validator([{"pose": "x"}, {"pose": "y"}]) is None
