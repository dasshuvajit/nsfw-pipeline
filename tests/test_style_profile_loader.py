"""StyleProfileLoader — all 11 archetypes load; render-tuning absent."""

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
    "analog_film_intimate",
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
    "analog_film_intimate",
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


# ── Verifier round-3 BLOCKER regression: aesthetic-compat fields ────


# All 11 profiles now declare full aesthetic-compat lists. Original 5
# came from Phase 3 vocab v6; the next 5 (old_hollywood_glamour,
# vintage_pinup_kodachrome, editorial_fashion_nude, moody_bw,
# fantasy_castlecore) were added in vocab v10 (2026-05-22) to close
# the env+photographer cross-axis incoherence — without compat lists
# these profiles let the LLM pick any env+photog combination, which
# produced clashes like fantasy_castlecore + Newton + ruined palazzo.
# analog_film_intimate joined in vocab v16 (2026-05-29).
PROFILES_WITH_COMPAT = {
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
    "analog_film_intimate",
}


def test_profiles_with_compat_load_palette_lists(loader):
    for pid in PROFILES_WITH_COMPAT:
        p = loader.get_profile(pid)
        assert p.compatible_palettes, (
            f"{pid}: compatible_palettes is empty — YAML declares a "
            "list but the dataclass / loader is silently stripping it. "
            "Phase 3 menu narrowing breaks if this regresses."
        )


def test_profiles_with_compat_load_photographer_lists(loader):
    for pid in PROFILES_WITH_COMPAT:
        p = loader.get_profile(pid)
        assert p.compatible_photographers, (
            f"{pid}: compatible_photographers is empty"
        )


def test_profiles_with_compat_load_art_movement_lists(loader):
    for pid in PROFILES_WITH_COMPAT:
        p = loader.get_profile(pid)
        assert p.compatible_art_movements, (
            f"{pid}: compatible_art_movements is empty"
        )


def test_profiles_with_compat_load_environment_lists(loader):
    """compatible_environments rounds out the Phase 3 quartet (added
    round-3). Missing = the environment-menu narrowing in
    SceneFacetGenerator is silently inert for these profiles."""
    for pid in PROFILES_WITH_COMPAT:
        p = loader.get_profile(pid)
        assert p.compatible_environments, (
            f"{pid}: compatible_environments is empty"
        )


def test_compat_lists_use_known_concept_tags(loader):
    """Every compat-list tag MUST exist in the vocabulary YAML —
    otherwise the LLM gets offered a tag the canonicalizer can't
    translate, and the prompt loses an aesthetic anchor."""
    from src.prompt.vocabulary import VocabularyLoader
    vocab = VocabularyLoader()

    def _tags_in_namespace(top: str, sub: str) -> set[str]:
        return set(vocab.concepts_by_namespace(top, sub).keys())

    valid = {
        "compatible_palettes":
            _tags_in_namespace("aesthetic", "color_palette"),
        "compatible_photographers":
            _tags_in_namespace("aesthetic", "photographer_ref"),
        "compatible_art_movements":
            _tags_in_namespace("aesthetic", "art_movement"),
        "compatible_environments":
            _tags_in_namespace("environment", "setting"),
        # 2026-05-23 — added compatible_art_styles per style profile
        # (Verifier audit C3 — Lindbergh + Newton clash).
        "compatible_art_styles":
            _tags_in_namespace("realism", "art_style"),
    }
    for pid in PROFILES_WITH_COMPAT:
        p = loader.get_profile(pid)
        for field_name, allowed in valid.items():
            actual = getattr(p, field_name)
            unknown = [t for t in actual if t not in allowed]
            assert not unknown, (
                f"{pid}.{field_name} references unknown vocab tag(s) "
                f"{unknown!r} — typo in the YAML or stale tag name."
            )


def test_profiles_with_compat_load_art_style_lists(loader):
    """All 10 profiles declare compatible_art_styles (2026-05-23).
    Missing = per-scene art_style_reference menu unconstrained,
    Lindbergh + Newton clash class re-opens."""
    for pid in PROFILES_WITH_COMPAT:
        p = loader.get_profile(pid)
        assert p.compatible_art_styles, (
            f"{pid}: compatible_art_styles is empty — per-scene "
            f"art_style_reference menu will be unconstrained, "
            f"re-opening the Lindbergh + Newton style clash class."
        )
