"""Tests for Q8 — tier-stratified few-shot examples.

Each family in ``config/families.yaml`` declares an ``examples:`` list
with at least 3 entries covering T2 + T4. The SceneFacetGenerator's
system prompt selects the closest-tier example at compose time so the
LLM sees a same-tier exemplar instead of generic boilerplate.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.agents.llm_client import OllamaClient
from src.agents.scene_facet_generator import (
    SceneFacetGenerator,
    _render_few_shot_block,
    _TIER_PREFERENCE,
    _FEW_SHOT_MAX,
)
from src.agents.schemas import FewShotExample
from src.memory.family_loader import FamilyLoader


# ── FewShotExample schema ───────────────────────────────────────────


class TestFewShotExampleSchema:
    def test_valid_example_round_trips(self):
        ex = FewShotExample.model_validate({
            "tier": "T2_implied",
            "scene": {"pose": "standing", "lighting": "soft"},
            "expected_facet": {"camera_spec": "85mm", "clothing": "silk"},
        })
        assert ex.tier == "T2_implied"
        assert ex.scene["pose"] == "standing"
        assert ex.expected_facet["camera_spec"] == "85mm"

    def test_invalid_tier_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="tier"):
            FewShotExample.model_validate({
                "tier": "T5_super_explicit",  # not a valid tier
                "scene": {},
                "expected_facet": {},
            })

    def test_extra_keys_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            FewShotExample.model_validate({
                "tier": "T2_implied",
                "scene": {},
                "expected_facet": {},
                "stray_field": "no",
            })


# ── families.yaml — every family has T2 and T4 ─────────────────────


class TestFamilyExamplesPresence:
    @pytest.fixture(scope="class")
    def loader(self) -> FamilyLoader:
        return FamilyLoader()

    @pytest.mark.parametrize(
        "family_id",
        ["sdxl", "pony", "illustrious", "flux", "chroma", "flux2"],
    )
    def test_family_has_at_least_3_examples(self, loader, family_id):
        family = loader.get_family(family_id)
        assert len(family.examples) >= 3, (
            f"{family_id} has only {len(family.examples)} examples; "
            f"plan §10 requires ≥3 with T2 + T4 mandatory"
        )

    @pytest.mark.parametrize(
        "family_id",
        ["sdxl", "pony", "illustrious", "flux", "chroma", "flux2"],
    )
    def test_family_includes_t2_and_t4(self, loader, family_id):
        family = loader.get_family(family_id)
        tiers = {ex["tier"] for ex in family.examples}
        assert "T2_implied" in tiers, f"{family_id}: missing T2 example"
        assert "T4_explicit" in tiers, f"{family_id}: missing T4 example"

    def test_family_loader_validates_tier_values(self, tmp_path):
        """A typo in a family's example tier fails fast at load."""
        from src.memory.family_loader import FamilyLoader, FamilyLoaderError
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text(
            "families:\n"
            "  sdxl:\n"
            "    prompt_style: sdxl_keywords\n"
            "    examples:\n"
            "      - tier: T9_typo\n"
            "        scene: {}\n"
            "        expected_facet: {}\n"
        )
        with pytest.raises(FamilyLoaderError, match="tier"):
            FamilyLoader(bad_yaml)

    def test_family_loader_validates_required_keys(self, tmp_path):
        """Missing scene/expected_facet keys fail fast."""
        from src.memory.family_loader import FamilyLoader, FamilyLoaderError
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text(
            "families:\n"
            "  sdxl:\n"
            "    prompt_style: sdxl_keywords\n"
            "    examples:\n"
            "      - tier: T2_implied\n"  # missing scene + expected_facet
        )
        with pytest.raises(FamilyLoaderError, match="missing keys"):
            FamilyLoader(bad_yaml)


# ── Tier preference + selection ────────────────────────────────────


class TestTierPreference:
    def test_t4_prefers_t4_then_t3(self):
        examples = [
            {"tier": "T2_implied", "scene": {}, "expected_facet": {"a": 1}},
            {"tier": "T3_artnude", "scene": {}, "expected_facet": {"a": 2}},
            {"tier": "T4_explicit", "scene": {}, "expected_facet": {"a": 3}},
        ]
        block = _render_few_shot_block(examples, "T4_explicit")
        # T4 example should appear first.
        assert "tier=T4_explicit" in block
        # And T4's expected_facet is what we render first.
        idx_t4 = block.find("tier=T4_explicit")
        idx_t3 = block.find("tier=T3_artnude")
        idx_t2 = block.find("tier=T2_implied")
        # T4 first, T3 second.
        assert idx_t4 != -1 and idx_t3 != -1
        assert idx_t4 < idx_t3

    def test_t1_falls_back_to_t2_then_t3(self):
        examples = [
            {"tier": "T2_implied", "scene": {}, "expected_facet": {}},
            {"tier": "T3_artnude", "scene": {}, "expected_facet": {}},
        ]
        block = _render_few_shot_block(examples, "T1_suggestive")
        # No exact T1 match → falls to T2 first per _TIER_PREFERENCE.
        assert "tier=T2_implied" in block
        # Should NOT include T4 even if available (T1 prefs skip T4).

    def test_max_examples_cap(self):
        examples = [
            {"tier": "T2_implied", "scene": {"i": i}, "expected_facet": {}}
            for i in range(10)
        ]
        block = _render_few_shot_block(examples, "T2_implied")
        # Max 2 examples rendered regardless of how many are declared.
        count = block.count("INPUT scene:")
        assert count == _FEW_SHOT_MAX

    def test_empty_examples_returns_empty(self):
        assert _render_few_shot_block([], "T2_implied") == ""

    def test_unknown_content_level_uses_default_pref(self):
        examples = [
            {"tier": "T2_implied", "scene": {}, "expected_facet": {"a": 1}},
        ]
        block = _render_few_shot_block(examples, "T_invalid")
        # Falls back to default preference; T2 is in default pref.
        assert "tier=T2_implied" in block


# ── System prompt integration ──────────────────────────────────────


def _ctx() -> MagicMock:
    return MagicMock()


class TestSystemPromptRendering:
    @pytest.fixture
    def generator(self) -> SceneFacetGenerator:
        return SceneFacetGenerator(OllamaClient())

    @pytest.fixture
    def loader(self) -> FamilyLoader:
        return FamilyLoader()

    def test_t4_facet_gets_t4_example(self, generator, loader):
        family = loader.get_family("sdxl")
        sp = generator._build_system_prompt(
            family, prompt_guide=None, content_level="T4_explicit",
        )
        # The system prompt should include the FEW-SHOT block.
        assert "FEW-SHOT EXAMPLES" in sp
        # The first example rendered should be tier=T4 (closest match).
        first_idx = sp.find("tier=T4_explicit")
        assert first_idx != -1, (
            "T4_explicit example missing from rendered block at T4 tier"
        )

    def test_t2_facet_gets_t2_example(self, generator, loader):
        family = loader.get_family("flux")
        sp = generator._build_system_prompt(
            family, prompt_guide=None, content_level="T2_implied",
        )
        assert "tier=T2_implied" in sp

    def test_no_content_level_still_renders(self, generator, loader):
        """When content_level is empty string (legacy path), the
        renderer falls back to the default preference order — block
        is still rendered."""
        family = loader.get_family("flux")
        sp = generator._build_system_prompt(family, prompt_guide=None)
        # Default pref starts with T2 → T3 → T4 → T1.
        assert "FEW-SHOT EXAMPLES" in sp

    def test_block_includes_input_scene_and_expected_facet_labels(
        self, generator, loader,
    ):
        """The rendered block uses 'INPUT scene:' / 'EXPECTED facet:'
        section labels so the LLM sees the structure clearly."""
        family = loader.get_family("sdxl")
        sp = generator._build_system_prompt(
            family, prompt_guide=None, content_level="T2_implied",
        )
        assert "INPUT scene:" in sp
        assert "EXPECTED facet:" in sp


# ── Selection invariants on real config ─────────────────────────────


class TestRealConfigSelection:
    @pytest.fixture(scope="class")
    def loader(self) -> FamilyLoader:
        return FamilyLoader()

    @pytest.mark.parametrize(
        "family_id",
        ["sdxl", "pony", "illustrious", "flux", "chroma", "flux2"],
    )
    def test_real_t4_selection_returns_t4_when_available(
        self, loader, family_id,
    ):
        family = loader.get_family(family_id)
        block = _render_few_shot_block(family.examples, "T4_explicit")
        assert "tier=T4_explicit" in block

    @pytest.mark.parametrize(
        "family_id",
        ["sdxl", "pony", "illustrious", "flux", "chroma", "flux2"],
    )
    def test_real_t2_selection_returns_t2_when_available(
        self, loader, family_id,
    ):
        family = loader.get_family(family_id)
        block = _render_few_shot_block(family.examples, "T2_implied")
        assert "tier=T2_implied" in block
