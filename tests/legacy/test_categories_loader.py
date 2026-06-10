"""Regression tests for CategoriesLoader's Phase 3 compat lists.

Verifier round-3 BLOCKER-1 root cause: the frozen ThemeCategory /
StyleCategory / NicheCluster dataclasses didn't declare the four
``compatible_*`` fields, so the loader silently dropped them from the
YAML rows. Round-2 fixed the declarations; this test guards against
regressions.

Without these tests, deleting the field declarations on the dataclass
or the corresponding lines in the loader's `from_dict`-equivalent
silently strips compat lists with zero test failures and the
SeriesPlanner / SceneFacetGenerator menu-narrowing falls back to the
full vocab — exactly the round-3 BLOCKER.
"""

from __future__ import annotations

import pytest

from src.memory.categories_loader import CategoriesLoader
from src.prompt.vocabulary import VocabularyLoader


COMPAT_FIELDS = (
    "compatible_palettes",
    "compatible_photographers",
    "compatible_art_movements",
    "compatible_environments",
)


@pytest.fixture
def loader():
    return CategoriesLoader()


@pytest.fixture
def vocab():
    return VocabularyLoader()


# ── ThemeCategory ───────────────────────────────────────────────────


def test_theme_category_dataclass_declares_compat_fields(loader):
    """Every loaded ThemeCategory has the 4 compat field attributes
    (defaults to empty list when YAML omits)."""
    for t in loader.theme_categories():
        for fld in COMPAT_FIELDS:
            assert hasattr(t, fld), (
                f"ThemeCategory missing field {fld!r} — round-3 "
                "BLOCKER regression."
            )
            assert isinstance(getattr(t, fld), list), (
                f"ThemeCategory.{fld} must be a list"
            )


def test_at_least_some_theme_categories_have_compat_environments(loader):
    """Post-Fix-A, 10 theme rows declare compatible_environments. If
    this count drops to 0, the YAML or loader silently regressed."""
    populated = [
        t for t in loader.theme_categories() if t.compatible_environments
    ]
    assert len(populated) >= 5, (
        f"Only {len(populated)} themes have populated "
        "compatible_environments; round-3 IMPORTANT-2 expects ≥5."
    )


# ── StyleCategory ───────────────────────────────────────────────────


def test_style_category_dataclass_declares_compat_fields(loader):
    for s in loader.style_categories():
        for fld in COMPAT_FIELDS:
            assert hasattr(s, fld), (
                f"StyleCategory missing field {fld!r}"
            )
            assert isinstance(getattr(s, fld), list)


def test_every_style_category_has_compat_environments(loader):
    """Round-3 IMPORTANT-2 added compatible_environments to all 6
    styles. Empty list ⇒ NicheMode-equivalent narrowing breaks for
    StyleMode."""
    styles = loader.style_categories()
    for s in styles:
        assert s.compatible_environments, (
            f"StyleCategory {s.id!r}: compatible_environments empty"
        )


# ── NicheCluster ────────────────────────────────────────────────────


def test_niche_cluster_dataclass_declares_compat_fields(loader):
    for c in loader.niche_clusters():
        for fld in COMPAT_FIELDS:
            assert hasattr(c, fld), (
                f"NicheCluster missing field {fld!r}"
            )
            assert isinstance(getattr(c, fld), list)


def test_every_niche_cluster_has_compat_environments(loader):
    clusters = loader.niche_clusters()
    for c in clusters:
        assert c.compatible_environments, (
            f"NicheCluster {c.id!r}: compatible_environments empty"
        )


# ── Cross-vocab tag-validity check ──────────────────────────────────


def test_theme_compat_tags_resolve_in_vocab(loader, vocab):
    """Every compat-list tag must exist in prompt_vocabulary.yaml.
    Typos / stale tag names silently degrade the menu narrowing
    (LLM gets shown the tag but the canonicalizer drops it)."""
    valid = {
        "compatible_palettes": set(
            vocab.concepts_by_namespace("aesthetic", "color_palette")
        ),
        "compatible_photographers": set(
            vocab.concepts_by_namespace("aesthetic", "photographer_ref")
        ),
        "compatible_art_movements": set(
            vocab.concepts_by_namespace("aesthetic", "art_movement")
        ),
        "compatible_environments": set(
            vocab.concepts_by_namespace("environment", "setting")
        ),
    }
    for t in loader.theme_categories():
        for fld, allowed in valid.items():
            unknown = [v for v in getattr(t, fld) if v not in allowed]
            assert not unknown, (
                f"Theme {t.id}.{fld} references unknown tag(s) "
                f"{unknown!r} — fix the YAML."
            )


def test_style_compat_tags_resolve_in_vocab(loader, vocab):
    env_allowed = set(vocab.concepts_by_namespace("environment", "setting"))
    for s in loader.style_categories():
        unknown = [
            v for v in s.compatible_environments if v not in env_allowed
        ]
        assert not unknown, (
            f"Style {s.id}.compatible_environments references unknown "
            f"tag(s) {unknown!r}"
        )


def test_niche_compat_tags_resolve_in_vocab(loader, vocab):
    env_allowed = set(vocab.concepts_by_namespace("environment", "setting"))
    for c in loader.niche_clusters():
        unknown = [
            v for v in c.compatible_environments if v not in env_allowed
        ]
        assert not unknown, (
            f"Niche {c.id}.compatible_environments references unknown "
            f"tag(s) {unknown!r}"
        )
