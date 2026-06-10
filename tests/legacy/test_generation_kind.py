"""Tests for ``src.core.engine._classify_generation_kind``.

The classifier stamps every prompt row with how it was produced —
``'llm_success'`` when the facet has a non-empty family-primary field,
``'fallback_t<n>'`` when the composer took the tier-aware fallback
path, ``'unknown'`` for back-compat / indeterminate cases.

The column lives on ``prompts.generation_kind`` (added 2026-05-24)
and replaces the old "LEFT JOIN against scene_facets to figure out
which path the prompt came from" pattern.
"""

from __future__ import annotations

import pytest

from src.core.engine import _classify_generation_kind


class TestLLMSuccess:
    """Non-empty primary field for the family ⇒ llm_success."""

    @pytest.mark.parametrize("family,field", [
        ("chroma", "scene_prose"),
        ("flux", "scene_prose"),
        ("flux2", "scene_prose"),
        ("pony", "booru_tags"),
        ("illustrious", "booru_tags"),
        ("sdxl", "camera_spec"),
    ])
    def test_each_family_primary_field(self, family, field):
        """Each family's recognised primary field unlocks llm_success."""
        facet = {field: "some non-empty content"}
        assert _classify_generation_kind(
            facet=facet, family_id=family, content_level="T4_explicit",
        ) == "llm_success"


class TestFallback:
    """Empty / missing primary field ⇒ fallback_t<n> per content_level."""

    def test_empty_string_primary(self):
        assert _classify_generation_kind(
            facet={"scene_prose": ""}, family_id="chroma",
            content_level="T4_explicit",
        ) == "fallback_t4"

    def test_whitespace_only_primary(self):
        assert _classify_generation_kind(
            facet={"scene_prose": "  \n\t "}, family_id="chroma",
            content_level="T4_explicit",
        ) == "fallback_t4"

    def test_missing_primary(self):
        assert _classify_generation_kind(
            facet={"art_style_reference": "ART_HELMUT_NEWTON"},
            family_id="chroma", content_level="T3_artnude",
        ) == "fallback_t3"

    def test_empty_facet_dict(self):
        assert _classify_generation_kind(
            facet={}, family_id="chroma", content_level="T4_explicit",
        ) == "fallback_t4"

    def test_none_facet(self):
        assert _classify_generation_kind(
            facet=None, family_id="chroma", content_level="T4_explicit",
        ) == "fallback_t4"

    @pytest.mark.parametrize("tier,expected_suffix", [
        ("T1_suggestive", "fallback_t1"),
        ("T2_implied",    "fallback_t2"),
        ("T3_artnude",    "fallback_t3"),
        ("T4_explicit",   "fallback_t4"),
    ])
    def test_tier_suffix_correct(self, tier, expected_suffix):
        assert _classify_generation_kind(
            facet={}, family_id="chroma", content_level=tier,
        ) == expected_suffix


class TestUnknown:
    """No tier + no facet ⇒ unknown — matches the schema's CHECK
    constraint default."""

    def test_no_tier_no_facet(self):
        assert _classify_generation_kind(
            facet={}, family_id="chroma", content_level=None,
        ) == "unknown"

    def test_no_tier_but_facet_succeeds(self):
        """If the facet has content the result is still llm_success
        regardless of tier (the LLM did its job)."""
        assert _classify_generation_kind(
            facet={"scene_prose": "x"}, family_id="chroma",
            content_level=None,
        ) == "llm_success"


class TestUnknownFamily:
    """An unknown family id can't be classified as llm_success (we
    don't know which field is primary). Falls back to tier-suffix."""

    def test_unknown_family_falls_back(self):
        assert _classify_generation_kind(
            facet={"scene_prose": "anything"},
            family_id="not_a_real_family",
            content_level="T4_explicit",
        ) == "fallback_t4"
