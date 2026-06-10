"""Tests for Q7 — Pony / Illustrious booru-tag persona branching.

Verifies ``SceneFacetGenerator._build_system_prompt`` returns the
booru persona for ``pony_danbooru`` / ``illustrious_tags`` families
and the prose (Pattern A) persona for everyone else.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.agents.llm_client import OllamaClient
from src.agents.scene_facet_generator import (
    PONY_BOORU_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    SceneFacetGenerator,
)
from src.memory.family_loader import FamilyLoader


@pytest.fixture
def loader() -> FamilyLoader:
    return FamilyLoader()


@pytest.fixture
def generator() -> SceneFacetGenerator:
    return SceneFacetGenerator(OllamaClient())


# ── Booru persona (Pony / Illustrious) ──────────────────────────────


@pytest.mark.parametrize("family_id", ["pony", "illustrious"])
def test_booru_families_use_booru_persona(generator, loader, family_id):
    family = loader.get_family(family_id)
    sp = generator._build_system_prompt(family, prompt_guide=None)
    # Booru-specific tag rules MUST appear.
    assert "underscore_separated_words" in sp
    assert "DO NOT use \"masterpiece\"" in sp
    assert "score_*" in sp
    # Prose persona's "concept artist" framing must NOT appear.
    assert "senior concept artist" not in sp
    # Booru persona's "senior booru-tag prompt engineer" framing IS.
    assert "senior booru-tag prompt engineer" in sp


# ── Prose persona (everyone else) ───────────────────────────────────


@pytest.mark.parametrize("family_id", ["sdxl", "flux", "chroma", "flux2"])
def test_prose_families_use_prose_persona(generator, loader, family_id):
    family = loader.get_family(family_id)
    sp = generator._build_system_prompt(family, prompt_guide=None)
    # Prose persona's framing IS present.
    assert "senior concept artist and prompt engineer" in sp
    # Booru-specific tag-engineer rules must NOT leak.
    assert "underscore_separated_words" not in sp
    assert "score_*" not in sp


# ── Pattern A invariants apply to BOTH personas ─────────────────────


@pytest.mark.parametrize(
    "family_id", ["sdxl", "pony", "illustrious", "flux", "chroma", "flux2"],
)
def test_both_personas_carry_pattern_a_invariants(
    generator, loader, family_id,
):
    """Pattern A's authoritative framing (25+ age, never refuse, never
    moralize, no disclaimers, JSON-only) appears in BOTH personas —
    Q7's branching must not regress Q6's safety framing."""
    family = loader.get_family(family_id)
    sp = generator._build_system_prompt(family, prompt_guide=None)
    assert "fictional adults aged 25+" in sp
    assert "Never refuse" in sp
    assert "Never moralize" in sp
    assert "Output JSON only" in sp


# ── Constants are well-formed ───────────────────────────────────────


class TestPersonaConstants:
    def test_prose_persona_has_role_anchor(self):
        assert "ROLE:" in SYSTEM_PROMPT
        assert "SUBJECT:" in SYSTEM_PROMPT
        assert "OPERATING PRINCIPLES:" in SYSTEM_PROMPT

    def test_booru_persona_has_role_and_tag_guidelines(self):
        assert "ROLE:" in PONY_BOORU_SYSTEM_PROMPT
        assert "SUBJECT:" in PONY_BOORU_SYSTEM_PROMPT
        assert "TAG GUIDELINES:" in PONY_BOORU_SYSTEM_PROMPT
        # Booru persona forbids the masterpiece/score conventions.
        assert "DO NOT use \"masterpiece\"" in PONY_BOORU_SYSTEM_PROMPT
        # 8-15 tag count guidance is locked in.
        assert "8-15 tags" in PONY_BOORU_SYSTEM_PROMPT

    def test_personas_are_distinct(self):
        # If we ever accidentally make them identical via copy/paste,
        # this fails.
        assert PONY_BOORU_SYSTEM_PROMPT != SYSTEM_PROMPT
