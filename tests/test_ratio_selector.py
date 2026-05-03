"""RatioSelector — deterministic seed + hard overrides + tier weights."""

from __future__ import annotations

import random

import pytest

from src.core.ratio_selector import (
    RATIO_TO_RESOLUTION,
    RatioSelector,
    RatioSelectorError,
    get_resolution,
)


TIER_WEIGHTS = {
    "T1_suggestive": {
        "portrait_23": 0.45,
        "portrait_916": 0.20,
        "square": 0.25,
        "landscape": 0.10,
    },
    "T2_implied": {
        "portrait_23": 0.45,
        "portrait_916": 0.20,
        "square": 0.25,
        "landscape": 0.10,
    },
    "T3_artnude": {
        "portrait_23": 0.55,
        "portrait_916": 0.20,
        "square": 0.20,
        "landscape": 0.05,
    },
    "T4_explicit": {
        "portrait_23": 0.55,
        "portrait_916": 0.20,
        "square": 0.20,
        "landscape": 0.05,
    },
}


@pytest.fixture
def selector():
    return RatioSelector(weights=TIER_WEIGHTS)


# ---- Hard overrides ----------------------------------------------------

def test_full_body_pose_forces_portrait_23(selector):
    scene = {"pose": "full body standing in doorway", "camera": "wide"}
    pick = selector.select(scene, "T2_implied")
    assert pick.ratio == "portrait_23"
    assert pick.reason == "override:portrait_23"
    assert pick.resolution == RATIO_TO_RESOLUTION["portrait_23"]


def test_standing_pose_forces_portrait_23(selector):
    scene = {"pose": "standing with arms raised"}
    pick = selector.select(scene, "T1_suggestive")
    assert pick.ratio == "portrait_23"


def test_reclining_pose_forces_landscape(selector):
    scene = {"pose": "reclining on velvet chaise"}
    pick = selector.select(scene, "T3_artnude")
    assert pick.ratio == "landscape"
    assert pick.reason == "override:landscape"


def test_lying_pose_forces_landscape(selector):
    scene = {"pose": "lying on silk sheets"}
    pick = selector.select(scene, "T2_implied")
    assert pick.ratio == "landscape"


def test_hard_overrides_disabled_by_flag():
    sel = RatioSelector(weights=TIER_WEIGHTS, hard_overrides_enabled=False)
    scene = {"pose": "full body standing"}
    pick = sel.select(scene, "T2_implied")
    # Without overrides, must route through weighted draw
    assert pick.reason.startswith("weighted:")


# ---- Deterministic seeding --------------------------------------------

def test_same_scene_same_level_yields_same_ratio(selector):
    scene = {"pose": "seated on couch", "camera": "medium shot",
             "camera_angle": "eye level", "environment_detail": "living room"}
    picks = [selector.select(scene, "T2_implied").ratio for _ in range(5)]
    # Determinism: same inputs → same output every call
    assert len(set(picks)) == 1


def test_different_levels_can_yield_different_ratios(selector):
    scene = {"pose": "seated on couch",
             "camera": "close-up", "camera_angle": "eye level",
             "environment_detail": "living room"}
    # Same scene, different tier → the RNG seed differs, so results CAN differ
    # (don't assert they must — just that both selections succeed).
    pick1 = selector.select(scene, "T1_suggestive")
    pick2 = selector.select(scene, "T4_explicit")
    assert pick1.ratio in RATIO_TO_RESOLUTION
    assert pick2.ratio in RATIO_TO_RESOLUTION


def test_explicit_rng_overrides_deterministic_seed(selector):
    scene = {"pose": "seated", "camera": "close-up",
             "camera_angle": "eye level", "environment_detail": "studio"}
    # A fixed-seed Random should be reproducible across calls
    r1 = random.Random(42)
    r2 = random.Random(42)
    p1 = selector.select(scene, "T2_implied", rng=r1)
    p2 = selector.select(scene, "T2_implied", rng=r2)
    assert p1.ratio == p2.ratio


# ---- Weighted selection reason ----------------------------------------

def test_weighted_path_reason_includes_tier(selector):
    scene = {"pose": "seated", "camera": "close-up",
             "camera_angle": "eye level", "environment_detail": "studio"}
    pick = selector.select(scene, "T3_artnude")
    assert pick.reason == "weighted:T3_artnude"


# ---- Error handling ---------------------------------------------------

def test_unknown_content_level_raises(selector):
    with pytest.raises(RatioSelectorError):
        selector.select({"pose": "seated"}, "T9_mystery")


def test_empty_weights_raises():
    with pytest.raises(RatioSelectorError):
        RatioSelector(weights={})


def test_unknown_ratio_in_weights_raises():
    with pytest.raises(RatioSelectorError):
        RatioSelector(weights={"T1_suggestive": {"bogus_ratio": 1.0}})


# ---- get_resolution ---------------------------------------------------

def test_get_resolution_maps_known_keys():
    for ratio, (w, h) in RATIO_TO_RESOLUTION.items():
        assert get_resolution(ratio) == (w, h)


def test_get_resolution_rejects_unknown():
    with pytest.raises(RatioSelectorError):
        get_resolution("totally_fake_ratio")


def test_model_resolution_override_wins():
    overrides = {"portrait_23": (832, 1216)}
    out = get_resolution("portrait_23", model_resolutions=overrides)
    assert out == (832, 1216)
    # Ratios not in the override map fall back to defaults
    assert get_resolution("square", model_resolutions=overrides) == RATIO_TO_RESOLUTION["square"]
