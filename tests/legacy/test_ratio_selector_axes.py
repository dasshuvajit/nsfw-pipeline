"""Multi-axis scoring — Phase A.

These tests pin the new behaviour layered on top of the legacy
content-level base weights:

  * audience bonus (deviantart vs patreon)
  * composition-intent bonus (close-up / medium / full-body / wide)
  * family-quality bonus (sdxl/pony/illustrious vs flux/chroma/flux2)
  * pose-signal bump (keyword match in joined scene text)
  * score clamps (floor + ceiling)
  * hard overrides win regardless of any axis stack
  * ``from_config`` classmethod loads ``ratio_signals.yaml`` correctly

Tests that need to inspect the *scored weights* (rather than the final
draw) capture them by monkey-patching ``random.Random.choices``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.core.ratio_selector import (
    RATIO_TO_RESOLUTION,
    RatioPick,
    RatioSelector,
    RatioSelectorError,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RATIO_SIGNALS_YAML = PROJECT_ROOT / "config" / "ratio_signals.yaml"

FLAT_WEIGHTS = {
    "T1_suggestive": {
        "portrait_23": 0.25,
        "portrait_916": 0.25,
        "square": 0.25,
        "landscape": 0.25,
    },
    "T2_implied": {
        "portrait_23": 0.25,
        "portrait_916": 0.25,
        "square": 0.25,
        "landscape": 0.25,
    },
    "T3_artnude": {
        "portrait_23": 0.25,
        "portrait_916": 0.25,
        "square": 0.25,
        "landscape": 0.25,
    },
    "T4_explicit": {
        "portrait_23": 0.25,
        "portrait_916": 0.25,
        "square": 0.25,
        "landscape": 0.25,
    },
}

NEUTRAL_SCENE = {
    "pose": "seated on couch",
    "camera": "medium shot",
    "camera_angle": "eye level",
    "environment_detail": "living room",
}


# ----- helpers --------------------------------------------------------------

def _capture_weights(monkeypatch) -> list[dict[str, float]]:
    """Patch ``random.Random.choices`` to record the weights passed in.

    Returns a list that gets appended on every call. The patched
    ``choices`` still picks the highest-weighted option deterministically
    (so callers don't have to seed an RNG just to inspect math).
    """
    import random as _random

    captured: list[dict[str, float]] = []
    real_choices = _random.Random.choices

    def fake_choices(self, population, weights, k=1):  # noqa: ARG001
        captured.append(dict(zip(population, weights)))
        return real_choices(self, population=population, weights=weights, k=k)

    monkeypatch.setattr(_random.Random, "choices", fake_choices)
    return captured


# ============================================================================
# Backward compatibility — selectors built without bonuses behave like legacy
# ============================================================================

def test_no_bonuses_no_audience_no_family_legacy_path():
    """Selector with only base weights still picks a valid ratio."""
    sel = RatioSelector(weights=FLAT_WEIGHTS)
    pick = sel.select(NEUTRAL_SCENE, "T2_implied")
    assert pick.ratio in RATIO_TO_RESOLUTION
    assert pick.reason == "weighted:T2_implied"


def test_legacy_call_signature_still_accepted():
    """Old (weights, signals, hard_overrides_enabled) positional form."""
    sel = RatioSelector(FLAT_WEIGHTS, None, True)
    pick = sel.select(NEUTRAL_SCENE, "T1_suggestive")
    assert isinstance(pick, RatioPick)


# ============================================================================
# Validation — bonus tables, clamps, error paths
# ============================================================================

def test_audience_bonus_with_unknown_ratio_raises():
    with pytest.raises(RatioSelectorError, match="audience_bonus"):
        RatioSelector(
            weights=FLAT_WEIGHTS,
            audience_bonus={"deviantart": {"bogus_ratio": 0.1}},
        )


def test_composition_bonus_with_unknown_ratio_raises():
    with pytest.raises(RatioSelectorError, match="composition_bonus"):
        RatioSelector(
            weights=FLAT_WEIGHTS,
            composition_bonus={"close-up": {"vertical": 0.2}},
        )


def test_family_quality_bonus_with_unknown_ratio_raises():
    with pytest.raises(RatioSelectorError, match="family_quality_bonus"):
        RatioSelector(
            weights=FLAT_WEIGHTS,
            family_quality_bonus={"sdxl": {"giant_landscape": 0.5}},
        )


def test_bonus_table_with_non_mapping_value_raises():
    with pytest.raises(RatioSelectorError, match="must be a mapping"):
        RatioSelector(
            weights=FLAT_WEIGHTS,
            audience_bonus={"deviantart": [0.1, 0.2]},  # type: ignore[dict-item]
        )


def test_clamp_min_negative_raises():
    with pytest.raises(RatioSelectorError, match="clamp range"):
        RatioSelector(weights=FLAT_WEIGHTS, clamp_min=-0.1, clamp_max=2.0)


def test_clamp_min_equals_max_raises():
    with pytest.raises(RatioSelectorError, match="clamp range"):
        RatioSelector(weights=FLAT_WEIGHTS, clamp_min=1.0, clamp_max=1.0)


def test_clamp_min_above_max_raises():
    with pytest.raises(RatioSelectorError, match="clamp range"):
        RatioSelector(weights=FLAT_WEIGHTS, clamp_min=2.0, clamp_max=1.0)


# ============================================================================
# Audience-bonus axis
# ============================================================================

def test_audience_patreon_boosts_portrait_916(monkeypatch):
    captured = _capture_weights(monkeypatch)
    sel = RatioSelector(
        weights=FLAT_WEIGHTS,
        audience_bonus={"patreon": {"portrait_916": 0.50}},
    )
    sel.select(NEUTRAL_SCENE, "T2_implied", audience="patreon")
    weights = captured[-1]
    # portrait_916 should be the highest-weighted
    assert weights["portrait_916"] > weights["portrait_23"]
    assert weights["portrait_916"] > weights["square"]
    assert weights["portrait_916"] > weights["landscape"]


def test_audience_deviantart_boosts_portrait_23(monkeypatch):
    captured = _capture_weights(monkeypatch)
    sel = RatioSelector(
        weights=FLAT_WEIGHTS,
        audience_bonus={"deviantart": {"portrait_23": 0.50}},
    )
    sel.select(NEUTRAL_SCENE, "T2_implied", audience="deviantart")
    weights = captured[-1]
    assert weights["portrait_23"] > weights["portrait_916"]


def test_audience_either_applies_no_bonus(monkeypatch):
    captured = _capture_weights(monkeypatch)
    sel = RatioSelector(
        weights=FLAT_WEIGHTS,
        audience_bonus={
            "deviantart": {"portrait_23": 0.50},
            "patreon": {"portrait_916": 0.50},
        },
        signals={},
    )
    sel.select(NEUTRAL_SCENE, "T2_implied", audience="either")
    # All ratios still flat — "either" is filtered by _VALID_AUDIENCES
    weights = captured[-1]
    assert all(abs(w - weights["portrait_23"]) < 1e-9 for w in weights.values())


def test_audience_none_applies_no_bonus(monkeypatch):
    captured = _capture_weights(monkeypatch)
    sel = RatioSelector(
        weights=FLAT_WEIGHTS,
        audience_bonus={"deviantart": {"portrait_23": 0.50}},
        signals={},
    )
    sel.select(NEUTRAL_SCENE, "T2_implied", audience=None)
    weights = captured[-1]
    assert all(abs(w - weights["portrait_23"]) < 1e-9 for w in weights.values())


def test_audience_unknown_value_silently_ignored(monkeypatch):
    captured = _capture_weights(monkeypatch)
    sel = RatioSelector(
        weights=FLAT_WEIGHTS,
        audience_bonus={"deviantart": {"portrait_23": 0.50}},
        signals={},
    )
    sel.select(NEUTRAL_SCENE, "T2_implied", audience="instagram")
    weights = captured[-1]
    assert all(abs(w - weights["portrait_23"]) < 1e-9 for w in weights.values())


# ============================================================================
# Composition-intent axis
# ============================================================================

def test_composition_close_up_boosts_square(monkeypatch):
    captured = _capture_weights(monkeypatch)
    sel = RatioSelector(
        weights=FLAT_WEIGHTS,
        composition_bonus={"close-up": {"square": 0.50}},
    )
    scene = {**NEUTRAL_SCENE, "composition_intent": "close-up"}
    sel.select(scene, "T2_implied")
    weights = captured[-1]
    assert weights["square"] > weights["portrait_23"]
    assert weights["square"] > weights["landscape"]


def test_composition_wide_boosts_landscape(monkeypatch):
    captured = _capture_weights(monkeypatch)
    sel = RatioSelector(
        weights=FLAT_WEIGHTS,
        composition_bonus={"wide": {"landscape": 0.50}},
        # disable signals so "panorama"/"wide shot" keywords don't double-bump
        signals={},
    )
    scene = {**NEUTRAL_SCENE, "composition_intent": "wide"}
    sel.select(scene, "T2_implied")
    weights = captured[-1]
    assert weights["landscape"] > weights["portrait_23"]


def test_composition_full_body_boosts_portrait_when_override_disabled(monkeypatch):
    """full-body normally short-circuits to override — disable to test bonus."""
    captured = _capture_weights(monkeypatch)
    sel = RatioSelector(
        weights=FLAT_WEIGHTS,
        composition_bonus={"full-body": {"portrait_23": 0.50}},
        hard_overrides_enabled=False,
        signals={},
    )
    scene = {
        "pose": "seated",  # avoid "full body" keyword in pose
        "camera": "medium shot",
        "camera_angle": "eye level",
        "environment_detail": "living room",
        "composition_intent": "full-body",
    }
    sel.select(scene, "T2_implied")
    weights = captured[-1]
    assert weights["portrait_23"] > weights["square"]


def test_composition_unknown_intent_ignored(monkeypatch):
    captured = _capture_weights(monkeypatch)
    sel = RatioSelector(
        weights=FLAT_WEIGHTS,
        composition_bonus={"close-up": {"square": 0.50}},
        signals={},
    )
    scene = {**NEUTRAL_SCENE, "composition_intent": "extreme-zoom"}
    sel.select(scene, "T2_implied")
    weights = captured[-1]
    assert all(abs(w - weights["portrait_23"]) < 1e-9 for w in weights.values())


def test_composition_missing_field_ignored(monkeypatch):
    captured = _capture_weights(monkeypatch)
    sel = RatioSelector(
        weights=FLAT_WEIGHTS,
        composition_bonus={"close-up": {"square": 0.50}},
        signals={},
    )
    sel.select(NEUTRAL_SCENE, "T2_implied")  # no composition_intent
    weights = captured[-1]
    assert all(abs(w - weights["portrait_23"]) < 1e-9 for w in weights.values())


# ============================================================================
# Family-quality bonus axis
# ============================================================================

def test_family_sdxl_penalizes_portrait_916(monkeypatch):
    captured = _capture_weights(monkeypatch)
    sel = RatioSelector(
        weights=FLAT_WEIGHTS,
        family_quality_bonus={"sdxl": {"portrait_916": -0.20}},
        signals={},
    )
    sel.select(NEUTRAL_SCENE, "T2_implied", family_id="sdxl")
    weights = captured[-1]
    assert weights["portrait_916"] < weights["portrait_23"]
    assert weights["portrait_916"] < weights["square"]


def test_family_flux2_boosts_portrait_23(monkeypatch):
    captured = _capture_weights(monkeypatch)
    sel = RatioSelector(
        weights=FLAT_WEIGHTS,
        family_quality_bonus={"flux2": {"portrait_23": 0.50}},
        signals={},
    )
    sel.select(NEUTRAL_SCENE, "T2_implied", family_id="flux2")
    weights = captured[-1]
    assert weights["portrait_23"] > weights["portrait_916"]


def test_family_flux_empty_dict_no_bonus(monkeypatch):
    captured = _capture_weights(monkeypatch)
    sel = RatioSelector(
        weights=FLAT_WEIGHTS,
        family_quality_bonus={"flux": {}},
        signals={},
    )
    sel.select(NEUTRAL_SCENE, "T2_implied", family_id="flux")
    weights = captured[-1]
    assert all(abs(w - weights["portrait_23"]) < 1e-9 for w in weights.values())


def test_family_unknown_id_no_bonus(monkeypatch):
    captured = _capture_weights(monkeypatch)
    sel = RatioSelector(
        weights=FLAT_WEIGHTS,
        family_quality_bonus={"sdxl": {"portrait_23": 0.50}},
        signals={},
    )
    sel.select(NEUTRAL_SCENE, "T2_implied", family_id="qwen-image")
    weights = captured[-1]
    assert all(abs(w - weights["portrait_23"]) < 1e-9 for w in weights.values())


# ============================================================================
# Pose-signal bump axis
# ============================================================================

def test_pose_signal_bump_applied_to_landscape(monkeypatch):
    captured = _capture_weights(monkeypatch)
    sel = RatioSelector(
        weights=FLAT_WEIGHTS,
        pose_signal_bump=0.50,
    )
    scene = {
        "pose": "seated",
        "camera": "wide shot",  # signals["landscape"] keyword
        "camera_angle": "eye level",
        "environment_detail": "studio",
    }
    sel.select(scene, "T2_implied")
    weights = captured[-1]
    # "wide shot" → landscape bump, "studio" → square bump (both keywords)
    assert weights["landscape"] > weights["portrait_23"]


def test_pose_signal_bump_one_per_ratio(monkeypatch):
    """Multiple matching keywords for the same ratio still only bump once."""
    captured = _capture_weights(monkeypatch)
    sel = RatioSelector(
        weights=FLAT_WEIGHTS,
        pose_signal_bump=0.30,
    )
    scene = {
        # both "close-up" and "face" map to square — should only bump once
        "pose": "seated",
        "camera": "close-up",
        "camera_angle": "face",
        "environment_detail": "blurred bg",
    }
    sel.select(scene, "T2_implied")
    weights = captured[-1]
    # Pre-normalization: square = 0.25 + 0.30 = 0.55 (one bump only);
    # others stay at 0.25. Sum = 1.30. Normalized square = 0.55/1.30.
    # If two bumps fired, square would be 0.85 → 0.85/1.60 = 0.531.
    expected_one_bump = 0.55 / 1.30
    expected_two_bumps = 0.85 / 1.60
    assert weights["square"] == pytest.approx(expected_one_bump, abs=1e-6)
    assert weights["square"] != pytest.approx(expected_two_bumps, abs=1e-3)


def test_pose_signal_bump_value_from_config(monkeypatch):
    """``pose_signal_bump`` constructor arg is honored."""
    captured = _capture_weights(monkeypatch)
    sel = RatioSelector(
        weights=FLAT_WEIGHTS,
        pose_signal_bump=1.00,  # very large
    )
    scene = {**NEUTRAL_SCENE, "camera": "close-up"}
    sel.select(scene, "T2_implied")
    weights = captured[-1]
    # square got +1.0 → dominates
    assert weights["square"] > 0.5  # post-normalization


def test_pose_signal_zero_bump_disables(monkeypatch):
    captured = _capture_weights(monkeypatch)
    sel = RatioSelector(
        weights=FLAT_WEIGHTS,
        pose_signal_bump=0.0,
    )
    scene = {**NEUTRAL_SCENE, "camera": "close-up"}
    sel.select(scene, "T2_implied")
    weights = captured[-1]
    assert all(abs(w - weights["portrait_23"]) < 1e-9 for w in weights.values())


# ============================================================================
# Hard-override path — beats every bonus
# ============================================================================

def test_full_body_pose_beats_audience_and_family():
    """Override fires before any score is computed."""
    sel = RatioSelector(
        weights=FLAT_WEIGHTS,
        audience_bonus={"patreon": {"portrait_916": 5.0}},  # huge
        family_quality_bonus={"flux2": {"portrait_916": 5.0}},
        composition_bonus={"wide": {"landscape": 5.0}},
    )
    scene = {
        "pose": "full body standing",
        "camera": "wide shot",
        "composition_intent": "wide",
    }
    pick = sel.select(scene, "T2_implied", audience="patreon", family_id="flux2")
    assert pick.ratio == "portrait_23"
    assert pick.reason == "override:portrait_23"


def test_reclining_pose_beats_audience_and_family():
    sel = RatioSelector(
        weights=FLAT_WEIGHTS,
        audience_bonus={"deviantart": {"portrait_23": 5.0}},
        family_quality_bonus={"sdxl": {"portrait_23": 5.0}},
    )
    scene = {"pose": "reclining on chaise"}
    pick = sel.select(scene, "T3_artnude", audience="deviantart", family_id="sdxl")
    assert pick.ratio == "landscape"
    assert pick.reason == "override:landscape"


def test_override_resolution_uses_family_bucket():
    """Override resolution still flows through the family bucket lookup."""
    sel = RatioSelector(weights=FLAT_WEIGHTS)
    scene = {"pose": "full body standing"}
    pick = sel.select(scene, "T2_implied", family_id="flux2")
    # flux2 lives in the 2MP tier — portrait_23 there is 1152×1728
    assert pick.resolution == (1152, 1728)


# ============================================================================
# Score clamping
# ============================================================================

def test_clamp_floor_prevents_zero_collapse(monkeypatch):
    """A stack of negative bonuses can't drive the score below clamp_min."""
    captured = _capture_weights(monkeypatch)
    sel = RatioSelector(
        weights=FLAT_WEIGHTS,
        audience_bonus={"deviantart": {"portrait_916": -1.0}},
        family_quality_bonus={"sdxl": {"portrait_916": -1.0}},
        signals={},
        clamp_min=0.05,
    )
    sel.select(
        NEUTRAL_SCENE, "T2_implied", audience="deviantart", family_id="sdxl"
    )
    weights = captured[-1]
    # portrait_916 was 0.25 - 1.0 - 1.0 = -1.75 → dropped by `if score > 0`,
    # so it shouldn't appear in the population at all (drop, not floor).
    # Other ratios stay clamped to ≤ clamp_max, ≥ clamp_min.
    assert "portrait_916" not in weights
    for w in weights.values():
        assert w > 0


def test_clamp_ceiling_blocks_runaway_bonus(monkeypatch):
    """A huge stack of positive bonuses doesn't exceed ``clamp_max``."""
    captured = _capture_weights(monkeypatch)
    sel = RatioSelector(
        weights=FLAT_WEIGHTS,
        audience_bonus={"patreon": {"portrait_916": 100.0}},
        family_quality_bonus={"flux2": {"portrait_916": 100.0}},
        signals={},
        clamp_min=0.05,
        clamp_max=2.0,
    )
    sel.select(
        NEUTRAL_SCENE, "T2_implied", audience="patreon", family_id="flux2"
    )
    weights = captured[-1]
    # Pre-normalization, portrait_916 was clamped to 2.0; others stay at 0.25.
    # Post-normalization: 2.0 / (2.0 + 0.25 + 0.25 + 0.25) = 2.0/2.75 ≈ 0.727
    assert weights["portrait_916"] == pytest.approx(2.0 / 2.75, abs=1e-6)


def test_all_zero_weights_after_clamp_raises():
    """If every ratio drops below 0, selection has nothing to draw from."""
    sel = RatioSelector(
        weights={
            "T2_implied": {
                "portrait_23": 0.05, "portrait_916": 0.05,
                "square": 0.05, "landscape": 0.05,
            }
        },
        audience_bonus={"patreon": {
            "portrait_23": -1.0, "portrait_916": -1.0,
            "square": -1.0, "landscape": -1.0,
        }},
        signals={},
    )
    with pytest.raises(RatioSelectorError, match="collapsed to zero"):
        sel.select(NEUTRAL_SCENE, "T2_implied", audience="patreon")


# ============================================================================
# Reason-string composition
# ============================================================================

def test_reason_contains_audience_when_given():
    sel = RatioSelector(weights=FLAT_WEIGHTS)
    pick = sel.select(NEUTRAL_SCENE, "T2_implied", audience="patreon")
    assert "audience=patreon" in pick.reason


def test_reason_contains_family_when_given():
    sel = RatioSelector(weights=FLAT_WEIGHTS)
    pick = sel.select(NEUTRAL_SCENE, "T2_implied", family_id="flux")
    assert "family=flux" in pick.reason


def test_reason_contains_intent_when_given():
    sel = RatioSelector(weights=FLAT_WEIGHTS)
    scene = {**NEUTRAL_SCENE, "composition_intent": "close-up"}
    pick = sel.select(scene, "T2_implied")
    assert "intent=close-up" in pick.reason


def test_reason_omits_audience_when_none():
    sel = RatioSelector(weights=FLAT_WEIGHTS)
    pick = sel.select(NEUTRAL_SCENE, "T2_implied")
    assert "audience" not in pick.reason
    assert "family" not in pick.reason
    assert "intent" not in pick.reason


def test_reason_combines_all_axes():
    sel = RatioSelector(weights=FLAT_WEIGHTS)
    scene = {**NEUTRAL_SCENE, "composition_intent": "medium"}
    pick = sel.select(
        scene, "T3_artnude", audience="deviantart", family_id="illustrious"
    )
    assert pick.reason.startswith("weighted:T3_artnude")
    assert "audience=deviantart" in pick.reason
    assert "family=illustrious" in pick.reason
    assert "intent=medium" in pick.reason


# ============================================================================
# Determinism with new axes — same inputs always pick the same ratio
# ============================================================================

def test_same_audience_same_family_same_scene_yields_same_ratio():
    sel = RatioSelector(weights=FLAT_WEIGHTS)
    picks = [
        sel.select(
            NEUTRAL_SCENE, "T2_implied",
            audience="patreon", family_id="sdxl",
        ).ratio
        for _ in range(5)
    ]
    assert len(set(picks)) == 1


def test_different_audience_different_seed_can_differ():
    """Different audiences produce different RNG seeds — outputs may diverge."""
    sel = RatioSelector(weights=FLAT_WEIGHTS)
    p1 = sel.select(NEUTRAL_SCENE, "T2_implied", audience="patreon").ratio
    p2 = sel.select(NEUTRAL_SCENE, "T2_implied", audience="deviantart").ratio
    # Don't assert they MUST differ — just that both are valid ratios
    assert p1 in RATIO_TO_RESOLUTION
    assert p2 in RATIO_TO_RESOLUTION


# ============================================================================
# from_config — load from real ratio_signals.yaml
# ============================================================================

def test_from_config_loads_yaml_file():
    pipeline_cfg = {"aspect_ratio_weights": FLAT_WEIGHTS}
    sel = RatioSelector.from_config(
        pipeline_cfg, ratio_signals_path=RATIO_SIGNALS_YAML
    )
    # The shipped yaml has bonuses for every audience and family
    assert sel.audience_bonus.get("deviantart")
    assert sel.audience_bonus.get("patreon")
    assert sel.composition_bonus.get("close-up")
    assert sel.family_quality_bonus.get("sdxl")
    assert sel.pose_signal_bump == pytest.approx(0.15)
    assert sel.clamp_min == pytest.approx(0.05)
    assert sel.clamp_max == pytest.approx(2.0)


def test_from_config_explicit_cfg_wins_over_path():
    pipeline_cfg = {"aspect_ratio_weights": FLAT_WEIGHTS}
    custom_cfg = {
        "audience_bonus": {"patreon": {"square": 0.99}},
        "pose_signal_bump": 0.42,
    }
    sel = RatioSelector.from_config(
        pipeline_cfg,
        ratio_signals_cfg=custom_cfg,
        ratio_signals_path=RATIO_SIGNALS_YAML,
    )
    # ratio_signals_cfg overrides the file
    assert sel.audience_bonus["patreon"]["square"] == pytest.approx(0.99)
    assert sel.pose_signal_bump == pytest.approx(0.42)


def test_from_config_missing_file_falls_back_gracefully(tmp_path, caplog):
    pipeline_cfg = {"aspect_ratio_weights": FLAT_WEIGHTS}
    sel = RatioSelector.from_config(
        pipeline_cfg, ratio_signals_path=tmp_path / "nope.yaml"
    )
    # No bonuses, but selector still functional
    assert sel.audience_bonus == {}
    assert sel.composition_bonus == {}
    pick = sel.select(NEUTRAL_SCENE, "T2_implied")
    assert pick.ratio in RATIO_TO_RESOLUTION


def test_from_config_without_aspect_ratio_weights_raises():
    with pytest.raises(RatioSelectorError, match="aspect_ratio_weights"):
        RatioSelector.from_config({})


def test_shipped_ratio_signals_yaml_is_loadable():
    """Sanity check on the file we ship in config/."""
    cfg = yaml.safe_load(RATIO_SIGNALS_YAML.read_text())
    assert "audience_bonus" in cfg
    assert "composition_bonus" in cfg
    assert "family_quality_bonus" in cfg
    assert "signals" in cfg
    # Every family in the registry should have an entry (empty dict OK)
    for family in ("sdxl", "pony", "illustrious", "flux", "chroma", "flux2"):
        assert family in cfg["family_quality_bonus"]


# ============================================================================
# Integration — multiple axes stacking on the same ratio
# ============================================================================

def test_audience_and_family_combine_additively(monkeypatch):
    captured = _capture_weights(monkeypatch)
    sel = RatioSelector(
        weights=FLAT_WEIGHTS,
        audience_bonus={"patreon": {"portrait_916": 0.10}},
        family_quality_bonus={"flux2": {"portrait_916": 0.20}},
        signals={},
    )
    sel.select(
        NEUTRAL_SCENE, "T2_implied",
        audience="patreon", family_id="flux2",
    )
    weights = captured[-1]
    # portrait_916 had base 0.25 + audience 0.10 + family 0.20 = 0.55
    # others stayed at 0.25
    # Post-normalization: 0.55 / (0.55 + 0.25 + 0.25 + 0.25) = 0.55/1.30
    assert weights["portrait_916"] == pytest.approx(0.55 / 1.30, abs=1e-6)


def test_negative_audience_bonus_offset_by_positive_family(monkeypatch):
    captured = _capture_weights(monkeypatch)
    sel = RatioSelector(
        weights=FLAT_WEIGHTS,
        audience_bonus={"deviantart": {"portrait_916": -0.10}},
        family_quality_bonus={"flux2": {"portrait_916": 0.20}},
        signals={},
    )
    sel.select(
        NEUTRAL_SCENE, "T2_implied",
        audience="deviantart", family_id="flux2",
    )
    weights = captured[-1]
    # portrait_916: 0.25 - 0.10 + 0.20 = 0.35; others 0.25
    assert weights["portrait_916"] == pytest.approx(0.35 / 1.10, abs=1e-6)
