"""Tests for ``build_family_context`` and the family-kind invariants.

Companion to ``test_generation_context.py``. Covers the family-kind
construction path added in 2026-05 for the family-level prompt-prep
feature:

  1. ``build_family_context`` populates ``target_kind='family'``,
     ``model_id is None``, ``model_config is None``, ``family``
     resolved from ``config/families.yaml``.
  2. Reading checkpoint-only properties (``supports_ipadapter``,
     ``supports_lora``) on a family-kind ctx raises ``AttributeError``
     with a clear "requires a model-kind GenerationContext" message.
  3. ``model_prompt_guide`` on a family-kind ctx is the family-only
     guide — empty ``trigger_words`` / ``avoid_words`` /
     ``negative_embeddings`` / ``example_prompt`` (none of which exist
     at the family level).
  4. The ``__post_init__`` invariant rejects inconsistent ctx
     constructions (target_kind='family' with model_id set;
     target_kind='model' with model_id=None; unknown target_kind).
  5. ``build_family_context`` raises ``ModelNotFound`` /
     ``FamilyNotFound`` for an unknown family id.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from src.core.content_level import ContentLevelLoader
from src.core.generation_context import (
    GenerationContext,
    build_context,
    build_family_context,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def fresh_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "ctx.db"
    result = subprocess.run(
        [sys.executable, "scripts/init_db.py", "--db-path", str(db_path)],
        capture_output=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr.decode()
    return db_path


@pytest.fixture
def style_profile() -> dict:
    return {"id": "golden_hour_natural", "base_negative_prompt": ""}


@pytest.fixture
def content_rules(fresh_db: Path):
    return ContentLevelLoader(fresh_db).load("T2_implied")


# ── 1. happy path: build_family_context populates the right shape ───


def test_build_family_context_populates_family_kind_fields(
    fresh_db, style_profile, content_rules,
):
    ctx = build_family_context(
        family_id="flux",
        mode="theme",
        content_level="T2_implied",
        execution_mode="manual",
        style_profile=style_profile,
        content_rules=content_rules,
        db_path=fresh_db,
        commercial_mode=False,
    )
    assert ctx.target_kind == "family"
    assert ctx.model_id is None
    assert ctx.model_config is None
    assert ctx.family is not None
    assert ctx.family.id == "flux"
    # The family-only prompt guide is attached.
    assert ctx.model_prompt_guide is not None


def test_build_family_context_works_for_all_six_families(
    fresh_db, style_profile, content_rules,
):
    """Every family declared in config/families.yaml must build."""
    for family_id in ("sdxl", "pony", "illustrious", "flux", "chroma", "flux2"):
        ctx = build_family_context(
            family_id=family_id,
            mode="theme",
            content_level="T2_implied",
            execution_mode="manual",
            style_profile=style_profile,
            content_rules=content_rules,
            db_path=fresh_db,
            commercial_mode=False,
        )
        assert ctx.family.id == family_id, family_id
        assert ctx.target_kind == "family"


# ── 2. checkpoint-only properties raise on family-kind ──────────────


def test_supports_ipadapter_raises_on_family_kind(
    fresh_db, style_profile, content_rules,
):
    ctx = build_family_context(
        family_id="flux",
        mode="theme",
        content_level="T2_implied",
        execution_mode="manual",
        style_profile=style_profile,
        content_rules=content_rules,
        db_path=fresh_db,
        commercial_mode=False,
    )
    with pytest.raises(AttributeError, match="requires a model-kind"):
        _ = ctx.supports_ipadapter


def test_supports_lora_raises_on_family_kind(
    fresh_db, style_profile, content_rules,
):
    ctx = build_family_context(
        family_id="sdxl",
        mode="theme",
        content_level="T2_implied",
        execution_mode="manual",
        style_profile=style_profile,
        content_rules=content_rules,
        db_path=fresh_db,
        commercial_mode=False,
    )
    with pytest.raises(AttributeError, match="requires a model-kind"):
        _ = ctx.supports_lora


def test_workflow_family_falls_back_to_family_id(
    fresh_db, style_profile, content_rules,
):
    """``ctx.workflow_family`` reads ``model_config.family`` for
    model-kind ctxs; for family-kind it falls back to ``family.id``
    (since model_config is None)."""
    ctx = build_family_context(
        family_id="pony",
        mode="theme",
        content_level="T2_implied",
        execution_mode="manual",
        style_profile=style_profile,
        content_rules=content_rules,
        db_path=fresh_db,
        commercial_mode=False,
    )
    assert ctx.workflow_family == "pony"


# ── 3. family-only prompt guide has empty per-model overlay ────────


def test_family_prompt_guide_has_empty_per_model_fields(
    fresh_db, style_profile, content_rules,
):
    ctx = build_family_context(
        family_id="flux",
        mode="theme",
        content_level="T2_implied",
        execution_mode="manual",
        style_profile=style_profile,
        content_rules=content_rules,
        db_path=fresh_db,
        commercial_mode=False,
    )
    guide = ctx.model_prompt_guide
    # Per-model overlay fields must be empty/None — only family-level
    # rules apply when target_kind='family'.
    assert guide.trigger_words == []
    assert guide.avoid_words == []
    assert guide.negative_embeddings == []
    assert guide.example_prompt is None or guide.example_prompt == ""


# ── 4. __post_init__ invariant rejects inconsistent ctxs ───────────


def _minimal_args(fresh_db, style_profile, content_rules) -> dict:
    return dict(
        mode="theme",
        content_level="T2_implied",
        execution_mode="manual",
        style_profile=style_profile,
        content_rules=content_rules,
        db_path=fresh_db,
    )


def test_post_init_rejects_family_kind_with_model_id_set(
    fresh_db, style_profile, content_rules,
):
    """Forging target_kind='family' with model_id set must fail fast.
    Catches a future caller wiring an inconsistent ctx."""
    from src.memory.model_registry import ModelRegistryLoader
    loader = ModelRegistryLoader(fresh_db, commercial_mode=False)
    family = loader.get_family("flux")
    with pytest.raises(ValueError, match="target_kind='family'"):
        GenerationContext(
            **_minimal_args(fresh_db, style_profile, content_rules),
            model_id="gonzalomo_flux_v30",          # forbidden when family-kind
            model_config=None,
            family=family,
            target_kind="family",
        )


def test_post_init_rejects_family_kind_with_model_config_set(
    fresh_db, style_profile, content_rules,
):
    from src.memory.model_registry import ModelRegistryLoader
    loader = ModelRegistryLoader(fresh_db, commercial_mode=False)
    family = loader.get_family("flux")
    model_cfg = loader.get_model("gonzalomo_flux_v30")
    with pytest.raises(ValueError, match="target_kind='family'"):
        GenerationContext(
            **_minimal_args(fresh_db, style_profile, content_rules),
            model_id=None,
            model_config=model_cfg,             # forbidden when family-kind
            family=family,
            target_kind="family",
        )


def test_post_init_rejects_model_kind_with_no_model_id(
    fresh_db, style_profile, content_rules,
):
    """target_kind='model' with model_id=None is the symmetric
    invariant violation."""
    from src.memory.model_registry import ModelRegistryLoader
    loader = ModelRegistryLoader(fresh_db, commercial_mode=False)
    family = loader.get_family("flux")
    with pytest.raises(ValueError, match="target_kind='model'"):
        GenerationContext(
            **_minimal_args(fresh_db, style_profile, content_rules),
            model_id=None,                      # required when model-kind
            model_config=None,
            family=family,
            target_kind="model",
        )


def test_post_init_rejects_unknown_target_kind(
    fresh_db, style_profile, content_rules,
):
    from src.memory.model_registry import ModelRegistryLoader
    loader = ModelRegistryLoader(fresh_db, commercial_mode=False)
    family = loader.get_family("flux")
    with pytest.raises(ValueError, match="target_kind must be"):
        GenerationContext(
            **_minimal_args(fresh_db, style_profile, content_rules),
            model_id=None,
            model_config=None,
            family=family,
            target_kind="tier",                 # not 'model' or 'family'
        )


# ── 5. unknown family id surfaces as a loader error ────────────────


def test_build_family_context_unknown_family_raises(
    fresh_db, style_profile, content_rules,
):
    """An unknown family_id surfaces from the FamilyLoader. Implementations
    use ``KeyError`` or ``FamilyNotFound`` — accept either via base
    ``Exception`` so this test isn't pinned to one specific exception
    class."""
    with pytest.raises(Exception):
        build_family_context(
            family_id="not_a_real_family",
            mode="theme",
            content_level="T2_implied",
            execution_mode="manual",
            style_profile=style_profile,
            content_rules=content_rules,
            db_path=fresh_db,
            commercial_mode=False,
        )


# ── 6. baseline check: build_context still works with default target_kind ─


def test_build_context_returns_model_kind_by_default(
    fresh_db, style_profile, content_rules,
):
    """Sanity probe: the existing ``build_context`` factory is
    unchanged — returns target_kind='model', model_id and model_config
    populated. Guards against an accidental flip in the default."""
    ctx = build_context(
        mode="theme",
        content_level="T2_implied",
        execution_mode="manual",
        style_profile=style_profile,
        content_rules=content_rules,
        db_path=fresh_db,
        model_id="gonzalomo_photo_v70",
        commercial_mode=False,
    )
    assert ctx.target_kind == "model"
    assert ctx.model_id == "gonzalomo_photo_v70"
    assert ctx.model_config is not None
    # supports_ipadapter / supports_lora must NOT raise on model-kind.
    _ = ctx.supports_ipadapter
    _ = ctx.supports_lora
