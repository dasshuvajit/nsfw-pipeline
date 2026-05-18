"""Tests for the 2026-05-18 extension of facet post-validation:
``lighting_directive`` is required at EVERY content tier (T1-T4)
across every family schema that carries it.

Pre-2026-05-18, only the NSFW tags were tier-required (nsfw_anatomy at
T3+, nsfw_act at T4). The realism enum tags (camera/lens/film_stock/
lighting_directive/mood_aesthetic/art_style_reference) were all
Optional and unenforced. Empirically, heretic-tuned LLMs (and any LLM
taking the "easy path") nulled them all, leaving the canonicalizer as
dead weight. Adding lighting_directive to the required list protects
the vocab_version 2 contract with a single-field cost (~1 retry per
facet at worst).
"""

from __future__ import annotations

import pytest

from src.agents.scene_facet_generator import (
    _missing_required_fields,
    _missing_required_nsfw_fields,
    _TIER_REQUIRED_FIELDS,
    _TIER_REQUIRED_NSFW_FIELDS,
)


# ── lighting_directive required at every tier ─────────────────────────


@pytest.mark.parametrize("tier", [
    "T1_suggestive", "T2_implied", "T3_artnude", "T4_explicit",
])
def test_lighting_directive_required_at_every_tier(tier):
    """The realism vocab_version 2 contract: lighting is too important
    to leave to free-text. Every tier requires the enum tag."""
    assert "lighting_directive" in _TIER_REQUIRED_FIELDS[tier]


@pytest.mark.parametrize("tier", [
    "T1_suggestive", "T2_implied", "T3_artnude", "T4_explicit",
])
def test_null_lighting_flagged_at_every_tier(tier):
    """A facet missing lighting_directive triggers the retry-nudge
    regardless of tier (the canonicalizer needs the tag to fire)."""
    facet = {"camera_spec": "85mm", "clothing": "silk", "lighting_directive": None}
    missing = _missing_required_fields(facet, tier, prompt_style="sdxl_keywords")
    assert "lighting_directive" in missing


def test_populated_lighting_satisfies_at_t1():
    facet = {"lighting_directive": "LIGHT_REMBRANDT", "camera_spec": "x"}
    missing = _missing_required_fields(
        facet, "T1_suggestive", prompt_style="sdxl_keywords",
    )
    assert missing == []


def test_empty_string_lighting_is_missing():
    """Empty string should be treated the same as None — the
    canonicalizer can't translate "" into a phrase."""
    facet = {"lighting_directive": "", "camera_spec": "x"}
    missing = _missing_required_fields(
        facet, "T2_implied", prompt_style="sdxl_keywords",
    )
    assert "lighting_directive" in missing


# ── existing NSFW requirements still hold (regression guard) ──────────


def test_t3_still_requires_nsfw_anatomy():
    facet = {"lighting_directive": "LIGHT_SOFT", "nsfw_anatomy": None}
    missing = _missing_required_fields(
        facet, "T3_artnude", prompt_style="flux_natural",
    )
    assert "nsfw_anatomy" in missing
    assert "lighting_directive" not in missing  # lighting IS present


def test_t4_requires_both_nsfw_anatomy_and_act():
    facet = {
        "lighting_directive": "LIGHT_GOLDEN_HOUR",
        "nsfw_anatomy": None,
        "nsfw_act": None,
    }
    missing = _missing_required_fields(
        facet, "T4_explicit", prompt_style="flux_natural",
    )
    assert "nsfw_anatomy" in missing
    assert "nsfw_act" in missing


def test_booru_relaxation_still_satisfies_nsfw_anatomy_via_tags():
    """Booru families can express NSFW in booru_tags (e.g. ``nude,
    breasts``) instead of the structured nsfw_anatomy field — that
    relaxation persists at T3 + T4. lighting_directive has no booru
    equivalent (cinematography vocab, not subject vocab), so it must
    still be populated explicitly."""
    facet = {
        "booru_tags": "1girl, solo, nude, breasts, anatomically_correct",
        "lighting_directive": "LIGHT_REMBRANDT",
        "nsfw_anatomy": None,  # null, but booru_tags carry the nudity
    }
    missing = _missing_required_fields(
        facet, "T3_artnude", prompt_style="pony_danbooru",
    )
    assert missing == []


def test_booru_relaxation_does_not_apply_to_lighting():
    """Even with full booru tags, missing lighting_directive is still
    flagged — lighting concepts are not part of the booru vocabulary
    that the relaxation covers."""
    facet = {
        "booru_tags": "1girl, solo, nude, breasts",
        "lighting_directive": None,
        "nsfw_anatomy": None,
    }
    missing = _missing_required_fields(
        facet, "T3_artnude", prompt_style="pony_danbooru",
    )
    assert "lighting_directive" in missing


# ── back-compat aliases still work ────────────────────────────────────


def test_back_compat_function_name_still_works():
    """The pre-2026-05-18 callers (and any pinned import paths)
    keep working via the alias."""
    facet = {"lighting_directive": None}
    assert _missing_required_nsfw_fields is _missing_required_fields
    assert "lighting_directive" in _missing_required_nsfw_fields(
        facet, "T1_suggestive", prompt_style="sdxl_keywords",
    )


def test_back_compat_constant_alias():
    assert _TIER_REQUIRED_NSFW_FIELDS is _TIER_REQUIRED_FIELDS


# ── nothing required if facet is None (defensive) ─────────────────────


def test_none_facet_returns_empty_list():
    assert _missing_required_fields(None, "T4_explicit") == []


# ── Unknown-tag detection (LLM invents fake enums) (2026-05-18) ──────


def test_unknown_lighting_tag_flagged_as_missing():
    """LLM invents ``LIGHT_ULTRAVIOLET`` (not in vocab). The
    canonicalizer would silently drop it; the validator must catch
    it and route to the retry-nudge instead."""
    facet = {
        "lighting_directive": "LIGHT_ULTRAVIOLET",  # not in vocab
        "mood_aesthetic": "MOOD_SERENE",
        "camera_spec": "x",
    }
    missing = _missing_required_fields(
        facet, "T1_suggestive",
        prompt_style="sdxl_keywords", family_id="sdxl",
    )
    assert "lighting_directive" in missing


def test_unknown_mood_tag_flagged_as_missing():
    """Same as above but for mood — heretic-vision was observed
    emitting ``MOOD_ETHEREAL`` (not in vocab) in the 2026-05-18
    smoke series."""
    facet = {
        "lighting_directive": "LIGHT_GOLDEN_HOUR",
        "mood_aesthetic": "MOOD_ETHEREAL",  # invented
        "camera_spec": "x",
    }
    missing = _missing_required_fields(
        facet, "T1_suggestive",
        prompt_style="sdxl_keywords", family_id="sdxl",
    )
    assert "mood_aesthetic" in missing


def test_valid_tags_pass_unknown_tag_check():
    """A facet with all valid known tags passes cleanly."""
    facet = {
        "lighting_directive": "LIGHT_GOLDEN_HOUR",
        "mood_aesthetic": "MOOD_SERENE",
        "camera_spec": "x",
    }
    missing = _missing_required_fields(
        facet, "T1_suggestive",
        prompt_style="sdxl_keywords", family_id="sdxl",
    )
    assert missing == []


def test_unknown_tag_check_disabled_when_family_id_omitted():
    """Back-compat: callers that don't supply ``family_id`` skip the
    unknown-tag check entirely (and pre-existing tests still pass)."""
    facet = {
        "lighting_directive": "LIGHT_ULTRAVIOLET",  # would be invalid
        "mood_aesthetic": "MOOD_SERENE",
    }
    # No family_id → check skipped → value treated as valid.
    missing = _missing_required_fields(
        facet, "T1_suggestive", prompt_style="sdxl_keywords",
    )
    assert missing == []


# ── mood_aesthetic enforcement (2026-05-18) ───────────────────────────


@pytest.mark.parametrize("tier", [
    "T1_suggestive", "T2_implied", "T3_artnude", "T4_explicit",
])
def test_mood_aesthetic_required_at_every_tier(tier):
    """mood_aesthetic joined lighting_directive as a tier-required
    field — same reasoning, complementary canonicalizer namespace."""
    assert "mood_aesthetic" in _TIER_REQUIRED_FIELDS[tier]


def test_null_mood_flagged_when_lighting_present():
    """Lighting present, mood null → only mood flagged. Each field
    is independently checked."""
    facet = {
        "lighting_directive": "LIGHT_REMBRANDT",
        "mood_aesthetic": None,
        "camera_spec": "x",
    }
    missing = _missing_required_fields(
        facet, "T2_implied", prompt_style="sdxl_keywords",
    )
    assert missing == ["mood_aesthetic"]


def test_both_lighting_and_mood_required_and_flagged():
    facet = {
        "lighting_directive": None,
        "mood_aesthetic": None,
        "camera_spec": "x",
    }
    missing = _missing_required_fields(
        facet, "T1_suggestive", prompt_style="sdxl_keywords",
    )
    assert set(missing) == {"lighting_directive", "mood_aesthetic"}


# ── Schema-awareness: skip fields not in the facet dict (Pony) ────────


def test_field_not_in_schema_is_skipped():
    """Pony's schema omits realism_camera; if a future required-fields
    addition includes it, the post-validator must skip — the field is
    not in the facet dict at all (Pydantic omits undeclared fields).
    Simulate by checking with a synthetic required-list patch."""
    from src.agents import scene_facet_generator as sfg
    # Pony facet — has lighting + mood + booru_tags, no realism_camera.
    pony_facet = {
        "booru_tags": "1girl, solo",
        "source_tag": "source_photograph",
        "lighting_directive": "LIGHT_SOFT_FILL",
        "mood_aesthetic": "MOOD_SERENE",
    }
    # Temporarily extend the required-list to include a field Pony
    # doesn't have.
    original = sfg._TIER_REQUIRED_FIELDS["T2_implied"]
    sfg._TIER_REQUIRED_FIELDS["T2_implied"] = original + ("realism_camera",)
    try:
        missing = _missing_required_fields(
            pony_facet, "T2_implied", prompt_style="pony_danbooru",
        )
        # realism_camera not in pony_facet → schema-aware skip.
        assert "realism_camera" not in missing
        # Lighting + mood present → nothing missing.
        assert missing == []
    finally:
        sfg._TIER_REQUIRED_FIELDS["T2_implied"] = original


def test_null_field_in_schema_still_flagged():
    """The complement: a field that IS in the facet dict (so the
    schema declares it) but is null gets flagged correctly. The
    schema-aware skip only protects fields the schema doesn't have."""
    facet_with_null = {
        "lighting_directive": None,
        "mood_aesthetic": "MOOD_INTIMATE",
        "camera_spec": "x",
    }
    missing = _missing_required_fields(
        facet_with_null, "T1_suggestive", prompt_style="sdxl_keywords",
    )
    assert missing == ["lighting_directive"]
