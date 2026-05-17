"""Negative-prompt textual-inversion embedding support — Phase E.

ComfyUI accepts ``embedding:Name`` (or ``embedding:Name:0.8``) tokens
in negative prompts; they reference TI files in ``models/embeddings/``.
The assembler must:

  * Hoist them to the FRONT of the assembled negative (so the
    position-weighted encoder gives them their full weight).
  * Treat them as identity-distinct: ``embedding:BadHands``,
    ``embedding:BadHands_v1``, and ``embedding:BadHands:0.8`` must NOT
    be merged by the regular keyword dedup.
  * Skip the entire pathway when ``supports_negative=False``
    (Flux family).
"""

from __future__ import annotations

import pytest

from src.prompt.builder import HARD_BLOCK_NEGATIVE, PromptBuilder


@pytest.fixture
def pb():
    return PromptBuilder()


# ----- TI tokens hoisted to the front --------------------------------------

def test_ti_tokens_hoisted_to_front(pb):
    neg = pb.assemble_negative_prompt(
        model_negative="blurry, low-res",
        negative_embeddings=["embedding:BadDream", "embedding:UnrealisticDream"],
    )
    # Both TI tokens come before any HARD_BLOCK or model_negative content
    assert neg.startswith("embedding:BadDream, embedding:UnrealisticDream")
    # Hard block + model negatives still present after TI block
    assert "child" in neg
    assert "blurry" in neg


def test_ti_tokens_with_weight_preserved(pb):
    neg = pb.assemble_negative_prompt(
        model_negative="bad",
        negative_embeddings=["embedding:BadDream:0.8"],
    )
    assert neg.startswith("embedding:BadDream:0.8,")


def test_no_ti_tokens_legacy_path_unchanged(pb):
    """Without TI tokens, the assembled negative is byte-identical to
    the pre-Phase-E shape (dedup of HARD_BLOCK + 3 caller layers)."""
    neg_legacy = pb.assemble_negative_prompt(
        model_negative="blurry, low-res",
        style_negative="harsh light",
        character_negative="watermark",
    )
    neg_ti_path = pb.assemble_negative_prompt(
        model_negative="blurry, low-res",
        style_negative="harsh light",
        character_negative="watermark",
        negative_embeddings=None,
    )
    assert neg_legacy == neg_ti_path


def test_empty_ti_list_falls_through(pb):
    neg = pb.assemble_negative_prompt(
        model_negative="blurry",
        negative_embeddings=[],
    )
    # No TI block, no extra leading comma — starts at HARD_BLOCK
    assert neg.startswith("child")


# ----- Identity dedup -------------------------------------------------------

def test_distinct_ti_tokens_kept_separately(pb):
    neg = pb.assemble_negative_prompt(
        negative_embeddings=[
            "embedding:BadHands",
            "embedding:BadHands_v1",
        ],
    )
    # Both kept — name suffix _v1 makes them distinct files on disk
    assert "embedding:BadHands," in neg or neg.startswith("embedding:BadHands,")
    assert "embedding:BadHands_v1" in neg


def test_same_ti_token_with_different_weights_kept_separately(pb):
    """``embedding:Foo`` and ``embedding:Foo:0.8`` are different
    instructions to ComfyUI — the unweighted form gets full strength."""
    neg = pb.assemble_negative_prompt(
        negative_embeddings=[
            "embedding:Foo",
            "embedding:Foo:0.8",
        ],
    )
    assert "embedding:Foo," in neg
    assert "embedding:Foo:0.8" in neg


def test_exact_duplicate_ti_tokens_collapsed(pb):
    neg = pb.assemble_negative_prompt(
        negative_embeddings=[
            "embedding:BadDream",
            "embedding:BadDream",  # exact dup
        ],
    )
    assert neg.count("embedding:BadDream") == 1


def test_ti_tokens_not_affected_by_keyword_dedup(pb):
    """Even when the keyword path would otherwise rewrite them, TI
    tokens flow through identity-dedup unchanged."""
    neg = pb.assemble_negative_prompt(
        model_negative="blurry, blurry, blurry",  # would dedup to one
        negative_embeddings=["embedding:BadDream"],
    )
    assert neg.startswith("embedding:BadDream,")
    # Regular dedup still collapses the keyword body
    assert neg.lower().count("blurry") == 1


# ----- Conflict between TI list and keyword segments -----------------------

def test_ti_token_in_keyword_segment_is_stripped(pb):
    """If a TI token snuck into a model_negative string (legacy YAML),
    it gets pulled out so the canonical TI block at the front wins."""
    neg = pb.assemble_negative_prompt(
        model_negative="embedding:BadDream, blurry, low-res",
        negative_embeddings=["embedding:BadDream"],
    )
    # Only ONE copy of the TI token, and it's at the front
    assert neg.count("embedding:BadDream") == 1
    assert neg.startswith("embedding:BadDream,")
    # Other keyword content survives
    assert "blurry" in neg
    assert "low-res" in neg


def test_ti_token_only_in_keyword_segment_still_extracted(pb):
    """Even without ``negative_embeddings=`` arg, a TI token in a
    keyword segment should be hoisted out — but the current contract
    only hoists from the explicit list. So a stray TI in a keyword
    string just gets dropped (so it can't be deduped weirdly).
    Document this so future regressions are intentional."""
    neg = pb.assemble_negative_prompt(
        model_negative="embedding:Loose, blurry",
    )
    # Stray TI token isn't in the dedicated list → dropped from output
    assert "embedding:Loose" not in neg
    assert "blurry" in neg


# ----- supports_negative=False short-circuits ------------------------------

def test_supports_negative_false_drops_ti_too(pb):
    neg = pb.assemble_negative_prompt(
        model_negative="blurry",
        negative_embeddings=["embedding:BadDream"],
        supports_negative=False,
    )
    assert neg == ""


# ----- Hard block still fires -----------------------------------------------

def test_hard_block_still_present_when_ti_used(pb):
    neg = pb.assemble_negative_prompt(
        negative_embeddings=["embedding:BadDream"],
    )
    # Hard block tokens come AFTER the TI block but are still there
    assert "embedding:BadDream" in neg
    for token in ("child", "teen", "loli", "minor"):
        assert token in neg


# ----- Garbage input tolerance ---------------------------------------------

def test_blank_strings_in_ti_list_dropped(pb):
    neg = pb.assemble_negative_prompt(
        negative_embeddings=["", "  ", None, "embedding:Real"],  # type: ignore[list-item]
    )
    assert neg.startswith("embedding:Real,")


def test_non_embedding_strings_in_ti_list_dropped(pb):
    """If a caller passes a regular keyword in the TI list by mistake,
    silently drop it rather than corrupting the output ordering."""
    neg = pb.assemble_negative_prompt(
        negative_embeddings=["blurry", "embedding:Real", "low-res"],
    )
    # Only the real TI token survives in the TI block
    assert neg.startswith("embedding:Real,")
    # Garbage keywords didn't end up at the front of the negative
    assert not neg.startswith("blurry")


def test_case_insensitive_embedding_prefix_match(pb):
    """ComfyUI treats ``embedding:`` and ``Embedding:`` the same; we
    accept any case for the prefix but preserve the original casing."""
    neg = pb.assemble_negative_prompt(
        negative_embeddings=["Embedding:Real", "EMBEDDING:Other"],
    )
    assert neg.startswith("Embedding:Real, EMBEDDING:Other")


# ----- ModelPromptGuide integration ----------------------------------------

def test_model_prompt_guide_carries_negative_embeddings_field():
    """Smoke test: ModelPromptGuide gained the field; default is []."""
    from src.memory.model_registry import ModelPromptGuide

    guide = ModelPromptGuide(
        model_id="test", prompt_style="sdxl_keywords", max_prompt_tokens=77
    )
    assert guide.negative_embeddings == []


def test_merge_overrides_extends_negative_embeddings_list():
    """``prompt.extend.negative_embeddings: [...]`` appends onto the
    family's list (which is empty by default)."""
    from src.core.merge_overrides import merge_prompt_config
    from src.memory.family_loader import FamilyLoader

    family = FamilyLoader().get_family("sdxl")
    merged = merge_prompt_config(
        family,
        {"extend": {"negative_embeddings": [
            "embedding:BadDream", "embedding:UnrealisticDream",
        ]}},
    )
    assert merged["negative_embeddings"] == [
        "embedding:BadDream", "embedding:UnrealisticDream",
    ]


def test_merge_overrides_overrides_negative_embeddings_list():
    from src.core.merge_overrides import merge_prompt_config
    from src.memory.family_loader import FamilyLoader

    family = FamilyLoader().get_family("sdxl")
    merged = merge_prompt_config(
        family,
        {"override": {"negative_embeddings": ["embedding:Pinned"]}},
    )
    assert merged["negative_embeddings"] == ["embedding:Pinned"]


def test_merge_overrides_rejects_unknown_keys():
    """Phase E added one new key; the validator should still reject typos."""
    from src.core.merge_overrides import MergeOverrideError, merge_prompt_config
    from src.memory.family_loader import FamilyLoader

    family = FamilyLoader().get_family("sdxl")
    with pytest.raises(MergeOverrideError, match="unknown keys"):
        merge_prompt_config(
            family, {"extend": {"negative_embeddingz": ["typo!"]}}
        )


# ----- Phase 1: 3-form normalization sugar ---------------------------------


def test_normalize_bare_name_gets_embedding_prefix():
    """``epiCNegative`` (bare) → ``embedding:epiCNegative``."""
    from src.core.merge_overrides import merge_prompt_config
    from src.memory.family_loader import FamilyLoader

    family = FamilyLoader().get_family("sdxl")
    merged = merge_prompt_config(
        family, {"extend": {"negative_embeddings": ["epiCNegative"]}}
    )
    assert merged["negative_embeddings"] == ["embedding:epiCNegative"]


def test_normalize_name_colon_weight_gets_embedding_prefix():
    """``Foo:0.8`` → ``embedding:Foo:0.8``."""
    from src.core.merge_overrides import merge_prompt_config
    from src.memory.family_loader import FamilyLoader

    family = FamilyLoader().get_family("sdxl")
    merged = merge_prompt_config(
        family,
        {"extend": {"negative_embeddings": [
            "CyberRealistic_Neg_SDXL:0.8",
        ]}},
    )
    assert merged["negative_embeddings"] == [
        "embedding:CyberRealistic_Neg_SDXL:0.8",
    ]


def test_normalize_canonical_form_passes_through():
    """``embedding:BadDream`` and ``embedding:Foo:0.7`` go through unchanged."""
    from src.core.merge_overrides import merge_prompt_config
    from src.memory.family_loader import FamilyLoader

    family = FamilyLoader().get_family("sdxl")
    merged = merge_prompt_config(
        family,
        {"extend": {"negative_embeddings": [
            "embedding:BadDream",
            "embedding:Foo:0.7",
        ]}},
    )
    assert merged["negative_embeddings"] == [
        "embedding:BadDream",
        "embedding:Foo:0.7",
    ]


def test_normalize_mixed_forms_in_one_list():
    """All three input shapes can be combined; all emerge in canonical form."""
    from src.core.merge_overrides import merge_prompt_config
    from src.memory.family_loader import FamilyLoader

    family = FamilyLoader().get_family("sdxl")
    merged = merge_prompt_config(
        family,
        {"extend": {"negative_embeddings": [
            "epiCNegative",                    # bare
            "Foo:0.8",                          # name:weight
            "embedding:BadDream",               # canonical
            "embedding:Bar:1.2",                # canonical with weight
        ]}},
    )
    assert merged["negative_embeddings"] == [
        "embedding:epiCNegative",
        "embedding:Foo:0.8",
        "embedding:BadDream",
        "embedding:Bar:1.2",
    ]


def test_normalize_drops_blank_entries():
    """Empty / whitespace / None entries silently skipped."""
    from src.core.merge_overrides import merge_prompt_config
    from src.memory.family_loader import FamilyLoader

    family = FamilyLoader().get_family("sdxl")
    merged = merge_prompt_config(
        family,
        {"extend": {"negative_embeddings": [
            "",
            "   ",
            None,  # type: ignore[list-item]
            "epiCNegative",
        ]}},
    )
    assert merged["negative_embeddings"] == ["embedding:epiCNegative"]


def test_normalize_rejects_malformed_token():
    """Tokens with spaces, special chars, or empty body raise."""
    from src.core.merge_overrides import MergeOverrideError, merge_prompt_config
    from src.memory.family_loader import FamilyLoader

    family = FamilyLoader().get_family("sdxl")
    with pytest.raises(MergeOverrideError, match="not a valid embedding name"):
        merge_prompt_config(
            family,
            {"extend": {"negative_embeddings": ["bad name with spaces"]}},
        )


def test_normalize_rejects_empty_embedding_body():
    """``embedding:`` with nothing after the colon raises."""
    from src.core.merge_overrides import MergeOverrideError, merge_prompt_config
    from src.memory.family_loader import FamilyLoader

    family = FamilyLoader().get_family("sdxl")
    with pytest.raises(MergeOverrideError, match="empty body"):
        merge_prompt_config(
            family, {"extend": {"negative_embeddings": ["embedding:"]}}
        )


def test_normalize_rejects_non_list_input():
    """``negative_embeddings:`` must be a list (not a comma-string)."""
    from src.core.merge_overrides import MergeOverrideError, merge_prompt_config
    from src.memory.family_loader import FamilyLoader

    family = FamilyLoader().get_family("sdxl")
    with pytest.raises(MergeOverrideError, match="must be a list"):
        merge_prompt_config(
            family,
            {"extend": {"negative_embeddings": "epiCNegative, BadDream"}},
        )


def test_normalize_rejects_non_string_entry():
    """List entries must be strings, not ints / dicts / etc."""
    from src.core.merge_overrides import MergeOverrideError, merge_prompt_config
    from src.memory.family_loader import FamilyLoader

    family = FamilyLoader().get_family("sdxl")
    with pytest.raises(MergeOverrideError, match="must be strings"):
        merge_prompt_config(
            family,
            {"extend": {"negative_embeddings": [42]}},  # type: ignore[list-item]
        )


# ----- Phase 1: reject `embedding:` strings inside negative_prompt: --------


def test_reject_embedding_in_negative_prompt_extend():
    """A TI token inside ``prompt.extend.negative_prompt:`` raises a
    clear error pointing the author to ``negative_embeddings:``."""
    from src.core.merge_overrides import MergeOverrideError, merge_prompt_config
    from src.memory.family_loader import FamilyLoader

    family = FamilyLoader().get_family("pony")
    with pytest.raises(MergeOverrideError, match="negative_embeddings"):
        merge_prompt_config(
            family,
            {"extend": {
                "negative_prompt": "embedding:CyberRealistic_Negative_PONY_V2-neg",
            }},
        )


def test_reject_embedding_in_negative_prompt_override():
    """Same rule applies to ``override:`` shape."""
    from src.core.merge_overrides import MergeOverrideError, merge_prompt_config
    from src.memory.family_loader import FamilyLoader

    family = FamilyLoader().get_family("pony")
    with pytest.raises(MergeOverrideError, match="negative_embeddings"):
        merge_prompt_config(
            family,
            {"override": {
                "negative_prompt": "blurry, embedding:BadDream, lowres",
            }},
        )


def test_reject_embedding_anywhere_in_negative_prompt_string():
    """A TI token mixed with other negatives is still rejected."""
    from src.core.merge_overrides import MergeOverrideError, merge_prompt_config
    from src.memory.family_loader import FamilyLoader

    family = FamilyLoader().get_family("sdxl")
    with pytest.raises(MergeOverrideError, match="negative_embeddings"):
        merge_prompt_config(
            family,
            {"extend": {
                "negative_prompt": "low quality, embedding:Foo:0.8, deformed",
            }},
        )


def test_negative_prompt_without_embedding_token_still_accepted():
    """Plain keyword negatives still work — the rejection is targeted."""
    from src.core.merge_overrides import merge_prompt_config
    from src.memory.family_loader import FamilyLoader

    family = FamilyLoader().get_family("sdxl")
    # Should not raise.
    merged = merge_prompt_config(
        family,
        {"extend": {
            "negative_prompt": "low quality, blurry, deformed",
        }},
    )
    assert "blurry" in (merged.get("negative_prompt") or "")


# ----- Phase 5: real-registry per-model verification ----------------------


@pytest.mark.parametrize("model_id", [
    "cyberrealistic_pony_v170",
    "pony_realism_v23_ultra",
])
def test_pony_realism_finetunes_carry_structure_intro(model_id):
    """All four Pony realism finetunes pin the photoreal vocabulary
    via ``prompt.override.structure_intro`` so source_photograph,
    'photo (medium)', and realistic land in the leading window."""
    import yaml
    from pathlib import Path
    from src.core.merge_overrides import merge_prompt_config
    from src.memory.family_loader import FamilyLoader

    yaml_path = (
        Path(__file__).resolve().parent.parent
        / "config" / "models" / f"{model_id}.yaml"
    )
    raw_yaml = yaml.safe_load(yaml_path.read_text())
    family = FamilyLoader().get_family(raw_yaml["family"])
    merged = merge_prompt_config(family, raw_yaml.get("prompt"))
    assert merged.get("structure_intro") == [
        "source_photograph", "photo (medium)", "realistic",
    ], f"{model_id} missing or wrong structure_intro"


@pytest.mark.parametrize("model_id", [
    "cyberrealistic_pony_v170",
])
def test_cyberrealistic_pony_carries_negative_embedding(model_id):
    """Both CyberRealistic Pony YAMLs declare the CyberRealistic
    negative TI in the typed `negative_embeddings:` field
    (Phase 1 migration). Verified via the merged config since
    ModelRegistryEntry only exposes the parent guide's flattened
    `base_negative_prompt`, not the typed `negative_embeddings:` list."""
    import yaml
    from pathlib import Path
    from src.core.merge_overrides import merge_prompt_config
    from src.memory.family_loader import FamilyLoader

    yaml_path = (
        Path(__file__).resolve().parent.parent
        / "config" / "models" / f"{model_id}.yaml"
    )
    raw_yaml = yaml.safe_load(yaml_path.read_text())
    family = FamilyLoader().get_family(raw_yaml["family"])
    merged = merge_prompt_config(family, raw_yaml.get("prompt"))
    assert any(
        "CyberRealistic_Negative_PONY_V2-neg" in emb
        for emb in (merged.get("negative_embeddings") or [])
    ), f"{model_id} missing CyberRealistic_Negative TI in negative_embeddings"
