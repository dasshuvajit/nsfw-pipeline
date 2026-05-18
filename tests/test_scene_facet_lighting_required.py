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
