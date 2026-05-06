"""Tests for ``ModelRegistryLoader.get_family_only_prompt_guide``.

Family-level prompt prep (`prepare_prompts --families <family>`)
needs a `ModelPromptGuide` built from the family alone — no per-model
overlay. Verifies the helper produces a guide with `model_id=None`,
empty per-model lists, and family-derived prompt_style / quality
prefix-suffix / negative_axes.
"""

from __future__ import annotations

import pytest

from src.memory.model_registry import ModelRegistryLoader


@pytest.fixture(scope="module")
def loader() -> ModelRegistryLoader:
    return ModelRegistryLoader()


@pytest.mark.parametrize(
    "family_id,prompt_style,supports_negative",
    [
        ("sdxl", "sdxl_keywords", True),
        ("pony", "pony_danbooru", True),
        ("illustrious", "illustrious_tags", True),
        ("flux", "flux_natural", False),
        ("chroma", "flux_natural", True),
        ("flux2", "flux2_prose", False),
    ],
)
class TestFamilyOnlyGuide:
    def test_prompt_style_matches_family(
        self, loader, family_id, prompt_style, supports_negative,
    ):
        guide = loader.get_family_only_prompt_guide(family_id)
        assert guide.prompt_style == prompt_style

    def test_model_id_is_none(
        self, loader, family_id, prompt_style, supports_negative,
    ):
        guide = loader.get_family_only_prompt_guide(family_id)
        assert guide.model_id is None

    def test_per_model_overlay_fields_empty(
        self, loader, family_id, prompt_style, supports_negative,
    ):
        """trigger_words / avoid_words / negative_embeddings are
        purely per-model — must be empty for a family-only guide."""
        guide = loader.get_family_only_prompt_guide(family_id)
        assert guide.trigger_words == []
        assert guide.avoid_words == []
        assert guide.negative_embeddings == []
        assert guide.example_prompt is None
        assert guide.llm_hint is None
        assert guide.structure_rules is None
        assert guide.notes is None

    def test_family_supports_negative_passes_through(
        self, loader, family_id, prompt_style, supports_negative,
    ):
        guide = loader.get_family_only_prompt_guide(family_id)
        assert guide.supports_negative_prompt is supports_negative


def test_pony_quality_prefix_inherits_score_chain(loader):
    """Pony's 6-tier score prefix is family-level, must flow through."""
    guide = loader.get_family_only_prompt_guide("pony")
    assert "score_9" in guide.quality_prefix
    assert "score_8_up" in guide.quality_prefix
    # Six-tier per AstraliteHeart's V6 model card.
    assert len(guide.quality_prefix) == 7  # score_9 + 6_up tiers + the explicit BREAK


def test_illustrious_quality_suffix_inherits_aesthetic_chain(loader):
    """Illustrious quality suffix `masterpiece, best quality, …` is
    family-level."""
    guide = loader.get_family_only_prompt_guide("illustrious")
    assert "masterpiece" in guide.quality_suffix
    assert "newest" in guide.quality_suffix


def test_negative_axes_family_only(loader):
    """Family-level `negative_axes` flows through; per-model
    `negative_axes.extend` should NOT — verified by checking sdxl
    family-only guide doesn't include per-model overrides like the
    `score_*` avoid words that some models add via extend."""
    guide = loader.get_family_only_prompt_guide("sdxl")
    assert "anatomy" in guide.negative_axes
    assert "medium" in guide.negative_axes


def test_unknown_family_raises(loader):
    """Wrong family id raises FamilyNotFound."""
    from src.memory.family_loader import FamilyNotFound
    with pytest.raises(FamilyNotFound):
        loader.get_family_only_prompt_guide("not_a_real_family")
