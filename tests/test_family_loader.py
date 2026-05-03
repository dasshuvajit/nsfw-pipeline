"""FamilyLoader — ensure every family loads and exposes expected fields."""

from __future__ import annotations

import pytest

from src.memory.family_loader import FamilyLoader, FamilyNotFound


EXPECTED_FAMILIES = {"sdxl", "pony", "illustrious", "flux", "chroma", "flux2"}


@pytest.fixture
def loader():
    return FamilyLoader()


def test_all_five_families_present(loader):
    ids = {f.id for f in loader.list_families()}
    assert EXPECTED_FAMILIES.issubset(ids)


def test_each_family_has_prompt_style(loader):
    for f in loader.list_families():
        if f.id not in EXPECTED_FAMILIES:
            continue
        assert f.prompt_style, f"family {f.id!r} missing prompt_style"


def test_llm_temperature_set_per_family(loader):
    """Phase A overhaul: every family should declare its own temperature."""
    for fam_id in EXPECTED_FAMILIES:
        fam = loader.get_family(fam_id)
        assert fam.llm_temperature is not None, f"family {fam_id!r} missing llm_temperature"
        assert 0.0 < fam.llm_temperature < 2.0


def test_expected_temperature_ordering(loader):
    """Booru families low, prose families high."""
    pony = loader.get_family("pony").llm_temperature
    flux = loader.get_family("flux").llm_temperature
    assert pony < flux  # 0.4 < 0.8


def test_chroma_realism_tail_is_period(loader):
    chroma = loader.get_family("chroma")
    assert chroma.realism_tail_style == "period"


def test_non_chroma_has_no_period_tail(loader):
    # sdxl/pony/illustrious/flux shouldn't opt into the period-tail shape
    for fam_id in ("sdxl", "pony", "illustrious", "flux"):
        fam = loader.get_family(fam_id)
        assert fam.realism_tail_style != "period", (
            f"family {fam_id!r} shouldn't use period tail"
        )


def test_sdxl_avoid_words_strips_quality_tokens(loader):
    """Realism SDXL checkpoints are trained AWAY from these tokens."""
    sdxl = loader.get_family("sdxl")
    avoid_lower = {w.lower() for w in sdxl.avoid_words}
    assert "masterpiece" in avoid_lower
    assert "best quality" in avoid_lower


def test_pony_has_score_prefix(loader):
    pony = loader.get_family("pony")
    # Either 4-tier or 6-tier prefix, but must start with score_9
    assert pony.quality_prefix
    assert "score_9" in pony.quality_prefix[0]


def test_illustrious_has_quality_suffix(loader):
    illus = loader.get_family("illustrious")
    suffix_joined = " ".join(illus.quality_suffix).lower()
    for token in ("masterpiece", "best quality", "very aesthetic", "newest"):
        assert token in suffix_joined


def test_flux_does_not_support_negative(loader):
    flux = loader.get_family("flux")
    # Flux schnell and dev don't use negative prompts meaningfully
    assert flux.supports_negative_prompt is False or flux.negative_prompt == ""


def test_unknown_family_raises(loader):
    with pytest.raises(FamilyNotFound):
        loader.get_family("nonexistent_family")


# ---- FLUX.2 Klein 9B family ------------------------------------------

def test_flux2_family_loads(loader):
    fam = loader.get_family("flux2")
    assert fam.prompt_style == "flux2_prose"
    assert fam.supports_negative_prompt is False
    assert fam.supports_weighting is False
    assert fam.negative_prompt == ""
    assert fam.llm_temperature == 0.8


def test_flux2_avoid_words_include_quality_and_booru_tokens(loader):
    fam = loader.get_family("flux2")
    avoid_lower = {w.lower() for w in fam.avoid_words}
    for token in ("masterpiece", "best quality", "8k", "score_9"):
        assert token in avoid_lower


def test_flux2_guide_shape(loader):
    fam = loader.get_family("flux2")
    assert fam.guide is not None
    assert fam.guide["structure_order"] == [
        "subject", "setting", "details", "lighting", "atmosphere",
    ]
    assert fam.guide["target_words"] == [30, 80]
    assert fam.guide["lighting_is_critical"] is True


def test_existing_families_have_no_guide(loader):
    """`guide:` is opt-in — the five pre-existing families stay None."""
    for fam_id in ("sdxl", "pony", "illustrious", "flux", "chroma"):
        fam = loader.get_family(fam_id)
        assert fam.guide is None, f"family {fam_id!r} should not declare a guide block"
