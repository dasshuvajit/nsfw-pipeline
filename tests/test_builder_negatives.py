"""Negative prompt assembly + avoid_words strip per family."""

from __future__ import annotations

import pytest

from src.memory.family_loader import FamilyLoader
from src.prompt.builder import HARD_BLOCK_NEGATIVE, PromptBuilder


@pytest.fixture
def pb():
    return PromptBuilder()


@pytest.fixture
def family_loader():
    return FamilyLoader()


def test_hard_block_prepended_to_every_negative(pb):
    neg = pb.assemble_negative_prompt(
        model_negative="blurry, low-res",
        style_negative="harsh overhead light",
        character_negative="",
    )
    # Age-ambiguity hard-block must appear first
    assert neg.startswith("child")
    # Caller-provided negatives preserved
    assert "blurry" in neg
    assert "harsh overhead light" in neg
    # HARD_BLOCK tokens all present
    for token in ("child", "teen", "loli", "minor", "schoolgirl"):
        assert token in neg


def test_negative_empty_when_family_disables(pb):
    neg = pb.assemble_negative_prompt(
        model_negative="blurry",
        supports_negative=False,
    )
    assert neg == ""


def test_hard_block_has_all_age_terms():
    for term in ("child", "kid", "young", "minor", "teen", "schoolgirl",
                 "loli", "shota", "underage", "baby", "toddler", "preteen"):
        assert term in HARD_BLOCK_NEGATIVE


def test_sdxl_strips_quality_tokens_from_body(pb, family_loader):
    """SDXL's avoid_words blocks masterpiece/best quality/8k tokens."""
    family = family_loader.get_family("sdxl")
    # Simulate an LLM that drifted into the banned vocabulary.
    # The composer should strip those tokens even though they came
    # through the scene fields.
    character = {"base_prompt": "elara"}
    scene = {
        "variation_axis": "pose",
        "pose": "seated",
        "camera": "medium shot",
        "camera_angle": "eye level",
        "lighting": "warm",
        "environment_detail": "bedroom",
        "mood_note": "relaxed, masterpiece, best quality, 8k",
    }
    style = {"base_style_keywords": "4k, cinematic"}
    out = pb.build_one(character, scene, style, family=family)
    text = out["prompt_text"].lower()
    assert "masterpiece" not in text
    assert "best quality" not in text
    assert "8k" not in text
    assert "4k" not in text
    # But legitimate tokens survive
    assert "cinematic" in text


def test_negative_dedups_overlap_between_layers(pb):
    # Overlap between model_negative and style_negative should be collapsed.
    neg = pb.assemble_negative_prompt(
        model_negative="blurry, low-res",
        style_negative="blurry, washed out",
    )
    # "blurry" appears once only
    assert neg.lower().count("blurry") == 1


# ---- Phase 3a: negative prompt token budget enforcement -----------------


def test_negative_within_budget_passes_through_unchanged(pb, family_loader):
    """When the assembled negative fits the family's max_tokens, no
    trim is applied — the byte-stable output stays byte-stable."""
    family = family_loader.get_family("sdxl")
    neg = pb.assemble_negative_prompt(
        model_negative="blurry, low-res",
        family=family,
    )
    # The default for SDXL's max_tokens is 77, and the assembled
    # negative is well under that. So the family-aware path should
    # produce the same output as the pre-Phase-3 path.
    neg_legacy = pb.assemble_negative_prompt(
        model_negative="blurry, low-res",
    )
    assert neg == neg_legacy


def test_negative_over_budget_is_trimmed(pb, family_loader, caplog):
    """When the negative exceeds max_tokens, fit_to_budget trims it
    and a WARNING is logged so drift is visible."""
    family = family_loader.get_family("sdxl")
    # Build a negative that's 200+ comma-tokens — overflows the
    # 77-token CLIP window.
    long_neg = ", ".join([f"junk{i}" for i in range(200)])
    with caplog.at_level("WARNING"):
        neg = pb.assemble_negative_prompt(
            model_negative=long_neg,
            family=family,
        )
    # Trimmed to fit (or near-fit) the budget.
    from src.prompt.tokenizer import count_tokens
    assert count_tokens(neg, family.tokenizer_id) <= family.max_tokens
    # Warning fired — caller can grep logs to confirm drift.
    assert any(
        "negative prompt trimmed" in r.message for r in caplog.records
    )


def test_negative_budget_trim_keeps_ti_block_whole(pb, family_loader):
    """TI tokens are short and high-priority — they must NOT be trimmed."""
    family = family_loader.get_family("sdxl")
    long_neg = ", ".join([f"junk{i}" for i in range(200)])
    neg = pb.assemble_negative_prompt(
        model_negative=long_neg,
        negative_embeddings=[
            "embedding:Real",
            "embedding:Foo:0.8",
        ],
        family=family,
    )
    # TI block at the front is unchanged
    assert neg.startswith("embedding:Real, embedding:Foo:0.8")


def test_negative_budget_no_family_no_trim(pb):
    """When ``family=None`` (the default), no trim runs — back-compat
    for callers that haven't migrated yet."""
    long_neg = ", ".join([f"junk{i}" for i in range(200)])
    neg = pb.assemble_negative_prompt(
        model_negative=long_neg,
    )
    # Untrimmed: all junk tokens still present (the dedup might collapse
    # exact dups but the synthetic ones above are all unique).
    assert "junk199" in neg


# ---- Phase 3c: Pony structure_intro -------------------------------------


def test_pony_structure_intro_emitted_after_quality_prefix(
    pb, family_loader,
):
    """A realism-Pony override of structure_intro emits the listed
    tokens between the BREAK score chain and the body."""
    from dataclasses import replace
    base = family_loader.get_family("pony")
    family = replace(
        base,
        structure_intro=["source_photo", "photo (medium)", "realistic"],
    )
    character = {"base_prompt": "woman"}
    scene = {
        "variation_axis": "pose",
        "pose": "seated",
        "camera": "medium shot",
        "camera_angle": "eye level",
        "lighting": "soft",
        "environment_detail": "studio",
        "mood_note": "relaxed",
        "booru_tags": "1girl, looking_at_viewer",
    }
    style = {"base_style_keywords": "cinematic"}
    out = pb.build_one(character, scene, style, family=family)
    text = out["prompt_text"]
    # All three tokens present in order
    assert "source_photo" in text
    assert "photo (medium)" in text
    assert "realistic" in text
    # Order: BREAK comes from quality_prefix, intro tokens come AFTER it
    break_idx = text.index("BREAK")
    source_idx = text.index("source_photo")
    assert break_idx < source_idx
    # And intro tokens come BEFORE the body's booru tags
    body_idx = text.index("1girl")
    assert source_idx < body_idx


def test_pony_default_structure_intro_is_empty(pb, family_loader):
    """Without an override, base Pony emits no structure_intro tokens."""
    family = family_loader.get_family("pony")
    assert family.structure_intro == []


def test_pony_structure_intro_via_yaml_override(family_loader, tmp_path):
    """The merge_overrides path threads structure_intro from per-model YAML."""
    from src.core.merge_overrides import merge_prompt_config
    family = family_loader.get_family("pony")
    merged = merge_prompt_config(
        family,
        {"override": {
            "structure_intro": ["source_photo", "photo (medium)", "realistic"],
        }},
    )
    assert merged["structure_intro"] == [
        "source_photo", "photo (medium)", "realistic",
    ]


def test_structure_intro_extend_appends_to_family_default(family_loader):
    """``extend.structure_intro`` appends to the family's default list."""
    from src.core.merge_overrides import merge_prompt_config
    from dataclasses import replace
    base = family_loader.get_family("pony")
    family = replace(base, structure_intro=["base_token"])
    merged = merge_prompt_config(
        family,
        {"extend": {"structure_intro": ["model_token"]}},
    )
    assert merged["structure_intro"] == ["base_token", "model_token"]


# ---- Audit fix B2: no double source_photograph in Pony realism -----------


def test_pony_structure_intro_does_not_double_inject_source_photograph(
    pb, family_loader,
):
    """Regression — Pony realism finetunes that declare
    ``structure_intro: [source_photograph, ...]`` must NOT also receive
    a second ``source_photograph`` from ``_ensure_pony_source_tag``."""
    from dataclasses import replace
    base = family_loader.get_family("pony")
    family = replace(
        base,
        structure_intro=["source_photograph", "photo (medium)", "realistic"],
    )
    character = {"base_prompt": "elara, woman"}
    scene = {
        "variation_axis": "pose",
        "pose": "seated",
        "camera": "medium shot",
        "camera_angle": "eye level",
        "lighting": "soft",
        "environment_detail": "studio",
        "mood_note": "calm",
        "booru_tags": "1girl, looking_at_viewer",
        "source_tag": "source_photograph",
    }
    style = {"base_style_keywords": "cinematic"}
    out = pb.build_one(character, scene, style, family=family)
    assert out["prompt_text"].lower().count("source_photograph") == 1


def test_pony_structure_intro_with_different_source_value_in_facet(
    pb, family_loader,
):
    """When intro pins source_photograph but LLM facet says
    source_anime, intro wins (the model author's pin is authoritative).
    Helper is skipped, so the facet's source_anime is dropped."""
    from dataclasses import replace
    base = family_loader.get_family("pony")
    family = replace(
        base,
        structure_intro=["source_photograph", "realistic"],
    )
    character = {"base_prompt": "elara"}
    scene = {
        "variation_axis": "pose",
        "pose": "seated",
        "camera": "medium",
        "camera_angle": "eye level",
        "lighting": "soft",
        "environment_detail": "studio",
        "mood_note": "calm",
        "booru_tags": "1girl",
        "source_tag": "source_anime",  # facet conflicts with intro
    }
    style = {"base_style_keywords": "cinematic"}
    out = pb.build_one(character, scene, style, family=family)
    text = out["prompt_text"].lower()
    assert "source_photograph" in text
    assert "source_anime" not in text


def test_vanilla_pony_no_intro_still_injects_source_photograph(
    pb, family_loader,
):
    """Backwards-compat: when family has no structure_intro,
    _ensure_pony_source_tag still runs and injects source_photograph."""
    family = family_loader.get_family("pony")  # no intro by default
    assert family.structure_intro == []
    character = {"base_prompt": "elara"}
    scene = {
        "variation_axis": "pose",
        "pose": "seated",
        "camera": "medium",
        "camera_angle": "eye level",
        "lighting": "soft",
        "environment_detail": "studio",
        "mood_note": "calm",
        "booru_tags": "1girl, looking_at_viewer",
        # no explicit source_tag → default falls through
    }
    style = {"base_style_keywords": "cinematic"}
    out = pb.build_one(character, scene, style, family=family)
    assert "source_photograph" in out["prompt_text"].lower()
