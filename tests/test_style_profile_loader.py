"""StyleProfileLoader — all 10 archetypes load; render-tuning absent."""

from __future__ import annotations

import pytest

from src.memory.style_profile_loader import (
    StyleProfile,
    StyleProfileLoader,
    StyleProfileNotFound,
)


EXPECTED_ARCHETYPES = {
    "boudoir_noir",
    "old_hollywood_glamour",
    "golden_hour_natural",
    "cinematic_wet_set",
    "fine_art_figurative",
    "vintage_pinup_kodachrome",
    "editorial_fashion_nude",
    "moody_bw",
    "fantasy_castlecore",
    "neo_noir_neon",
}

VALID_TIERS = {"T1_suggestive", "T2_implied", "T3_artnude", "T4_explicit"}
VALID_FAMILIES = {"sdxl", "pony", "illustrious", "flux", "chroma", "flux2"}

FLUX2_ELIGIBLE = {
    "boudoir_noir",
    "old_hollywood_glamour",
    "golden_hour_natural",
    "cinematic_wet_set",
    "fine_art_figurative",
    "editorial_fashion_nude",
    "moody_bw",
    "fantasy_castlecore",
}
FLUX2_INELIGIBLE = {"vintage_pinup_kodachrome", "neo_noir_neon"}


@pytest.fixture
def loader():
    return StyleProfileLoader()


def test_all_archetypes_load(loader):
    ids = {p.id for p in loader.list_profiles()}
    missing = EXPECTED_ARCHETYPES - ids
    assert not missing, f"missing archetypes: {missing}"


def test_each_archetype_has_required_fields(loader):
    for p in loader.list_profiles():
        if p.id not in EXPECTED_ARCHETYPES:
            continue
        assert p.name, f"{p.id} missing name"
        assert p.description, f"{p.id} missing description"
        assert p.base_style_keywords, f"{p.id} missing base_style_keywords"
        assert p.palette_hint, f"{p.id} missing palette_hint"
        assert p.lighting_hint, f"{p.id} missing lighting_hint"
        assert p.suited_tiers, f"{p.id} missing suited_tiers"
        assert p.suited_families, f"{p.id} missing suited_families"


def test_suited_tiers_are_valid(loader):
    for p in loader.list_profiles():
        if p.id not in EXPECTED_ARCHETYPES:
            continue
        for tier in p.suited_tiers:
            assert tier in VALID_TIERS, (
                f"{p.id} has invalid suited_tier {tier!r}"
            )


def test_suited_families_are_valid(loader):
    for p in loader.list_profiles():
        if p.id not in EXPECTED_ARCHETYPES:
            continue
        for fam in p.suited_families:
            assert fam in VALID_FAMILIES, (
                f"{p.id} has invalid suited_family {fam!r}"
            )


def test_render_tuning_absent(loader):
    """Render tuning must live in model YAMLs, not style profiles."""
    profile = loader.get_profile("golden_hour_natural")
    # The dataclass shouldn't have any render-tuning fields.
    forbidden = {"model_id", "sampler", "scheduler", "steps", "cfg",
                 "clip_skip", "lora_stack"}
    for attr in forbidden:
        assert not hasattr(profile, attr), (
            f"StyleProfile still carries render-tuning field {attr!r}"
        )


def test_unknown_profile_raises(loader):
    with pytest.raises(StyleProfileNotFound):
        loader.get_profile("does_not_exist")


def test_flux2_suited_families_routing(loader):
    for aid in FLUX2_ELIGIBLE:
        profile = loader.get_profile(aid)
        assert "flux2" in profile.suited_families, (
            f"{aid} should include flux2 in suited_families"
        )
    for aid in FLUX2_INELIGIBLE:
        profile = loader.get_profile(aid)
        assert "flux2" not in profile.suited_families, (
            f"{aid} must NOT include flux2 (Klein mis-fits this archetype)"
        )
