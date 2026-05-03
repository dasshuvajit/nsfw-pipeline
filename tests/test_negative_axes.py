"""Phase D — 7-axis negative taxonomy.

Covers the pure ``negative_axes`` module, its integration with
``merge_overrides``, and the ``conflict_terms`` lexical filter on
``PromptBuilder.assemble_negative_prompt``.
"""

from __future__ import annotations

import pytest

from src.core.merge_overrides import MergeOverrideError, merge_prompt_config
from src.memory.family_loader import FamilyLoader
from src.prompt.builder import HARD_BLOCK_NEGATIVE, PromptBuilder
from src.prompt.negative_axes import (
    AXIS_KEYS,
    empty_axes,
    filter_conflicts,
    flatten_axes,
    merge_axes,
    normalize_axes,
)


# ----- normalize_axes ------------------------------------------------------

def test_empty_axes_has_all_seven_keys():
    out = empty_axes()
    assert set(out) == set(AXIS_KEYS)
    assert all(out[a] == [] for a in AXIS_KEYS)


def test_normalize_strips_blank_tokens_and_whitespace():
    raw = {"anatomy": ["  bad anatomy  ", "", "extra fingers", None]}
    out = normalize_axes(raw)  # type: ignore[arg-type]
    assert out["anatomy"] == ["bad anatomy", "extra fingers"]


def test_normalize_accepts_string_for_axis():
    """Single-line YAML convenience: ``skin: "shiny, glossy"``."""
    out = normalize_axes({"skin": "shiny, glossy"})
    assert out["skin"] == ["shiny", "glossy"]


def test_normalize_rejects_unknown_axis_key():
    with pytest.raises(ValueError, match="unknown negative_axes keys"):
        normalize_axes({"anatomy": ["bad"], "typoooo": ["x"]})


def test_normalize_rejects_non_mapping():
    with pytest.raises(ValueError, match="must be a mapping"):
        normalize_axes(["not", "a", "mapping"])  # type: ignore[arg-type]


def test_normalize_rejects_non_list_axis_value():
    with pytest.raises(ValueError, match="must be a list"):
        normalize_axes({"anatomy": 123})  # type: ignore[dict-item]


# ----- flatten_axes --------------------------------------------------------

def test_flatten_preserves_canonical_axis_order():
    axes = {
        "censor": ["censored"],
        "anatomy": ["bad anatomy"],
        "watermark": ["watermark"],
        "quality": ["blurry"],
    }
    flat = flatten_axes(axes)
    # Canonical order: anatomy, medium, skin, quality, watermark, safety, censor
    assert flat == "bad anatomy, blurry, watermark, censored"


def test_flatten_dedups_across_axes_case_insensitive():
    """A token that appears in two axes is emitted once (first wins)."""
    axes = {
        "anatomy": ["bad anatomy"],
        "quality": ["Bad Anatomy"],   # different case, dedup'd
    }
    assert flatten_axes(axes) == "bad anatomy"


def test_flatten_empty_returns_empty_string():
    assert flatten_axes(None) == ""
    assert flatten_axes({}) == ""
    assert flatten_axes(empty_axes()) == ""


# ----- merge_axes ----------------------------------------------------------

def test_merge_axes_extend_appends_per_axis():
    base = {"anatomy": ["bad anatomy"], "skin": ["plastic"]}
    out = merge_axes(base, extend={"anatomy": ["extra fingers"], "skin": ["shiny"]})
    assert out["anatomy"] == ["bad anatomy", "extra fingers"]
    assert out["skin"] == ["plastic", "shiny"]


def test_merge_axes_override_replaces_per_axis():
    base = {"anatomy": ["bad anatomy", "extra limbs"], "quality": ["blurry"]}
    out = merge_axes(base, override={"anatomy": ["different"]})
    assert out["anatomy"] == ["different"]
    # Untouched axes stay
    assert out["quality"] == ["blurry"]


def test_merge_axes_override_to_empty_list_clears_axis():
    base = {"medium": ["cartoon"]}
    out = merge_axes(base, override={"medium": []})
    assert out["medium"] == []


def test_merge_axes_unknown_axis_key_in_extend_rejected():
    with pytest.raises(ValueError, match="unknown negative_axes extend keys"):
        merge_axes({}, extend={"nottanaxis": ["x"]})


def test_merge_axes_unknown_axis_key_in_override_rejected():
    with pytest.raises(ValueError, match="unknown negative_axes override keys"):
        merge_axes({}, override={"nottanaxis": ["x"]})


# ----- filter_conflicts ----------------------------------------------------

def test_filter_drops_token_appearing_verbatim_in_positive():
    axes = {"quality": ["blurry", "low-res"]}
    filtered, dropped = filter_conflicts(axes, ["blurry portrait"])
    assert filtered["quality"] == ["low-res"]
    assert dropped == [("quality", "blurry")]


def test_filter_strips_weight_parens_for_match():
    """``(blurry:1.2)`` should match ``blurry`` in the positive prompt."""
    axes = {"quality": ["(blurry:1.2)"]}
    filtered, dropped = filter_conflicts(axes, ["blurry shot"])
    assert filtered["quality"] == []
    assert len(dropped) == 1


def test_filter_no_conflict_keeps_everything():
    axes = {"anatomy": ["bad anatomy"], "watermark": ["watermark"]}
    filtered, dropped = filter_conflicts(axes, ["studio portrait, soft light"])
    assert filtered["anatomy"] == ["bad anatomy"]
    assert filtered["watermark"] == ["watermark"]
    assert dropped == []


def test_filter_empty_conflict_terms_returns_axes_unchanged():
    axes = {"quality": ["blurry"]}
    filtered, dropped = filter_conflicts(axes, None)
    assert filtered["quality"] == ["blurry"]
    assert dropped == []


def test_filter_word_boundary_match():
    """Multi-word negative tokens get caught by per-word lookup."""
    axes = {"anatomy": ["extra limbs"]}
    filtered, dropped = filter_conflicts(axes, ["pose with extra limbs visible"])
    # Both words ("extra", "limbs") are in the conflict set
    assert dropped == [("anatomy", "extra limbs")]


# ----- merge_overrides integration -----------------------------------------

def test_merge_overrides_extends_negative_axes_per_axis():
    family = FamilyLoader().get_family("sdxl")
    merged = merge_prompt_config(
        family,
        {"extend": {"negative_axes": {"skin": ["shiny", "rubbery"]}}},
    )
    skin = merged["negative_axes"]["skin"]
    # Family started with [oversharpened, plastic skin]; extend appends.
    assert "oversharpened" in skin
    assert "plastic skin" in skin
    assert "shiny" in skin
    assert "rubbery" in skin


def test_merge_overrides_overrides_negative_axis_to_empty():
    family = FamilyLoader().get_family("sdxl")
    merged = merge_prompt_config(
        family,
        {"override": {"negative_axes": {"medium": []}}},
    )
    # Family had ["cartoon"] in medium; override empties it.
    assert merged["negative_axes"]["medium"] == []
    # Untouched axes survive
    assert "bad anatomy" in merged["negative_axes"]["anatomy"]


def test_merge_overrides_reflattens_negative_prompt_after_axis_change():
    """The flat string is re-derived from axes when axes are touched."""
    family = FamilyLoader().get_family("sdxl")
    merged = merge_prompt_config(
        family,
        {"override": {"negative_axes": {"medium": []}}},
    )
    # "cartoon" was in family.negative_prompt — should be gone now.
    assert "cartoon" not in merged["negative_prompt"].lower()


def test_merge_overrides_rejects_both_legacy_and_axes_in_same_block():
    family = FamilyLoader().get_family("sdxl")
    with pytest.raises(MergeOverrideError, match="both `negative_prompt`"):
        merge_prompt_config(
            family,
            {
                "extend": {"negative_prompt": "extra"},
                "override": {"negative_axes": {"skin": []}},
            },
        )


def test_merge_overrides_rejects_unknown_axis_key():
    family = FamilyLoader().get_family("sdxl")
    with pytest.raises(ValueError, match="unknown negative_axes extend keys"):
        merge_prompt_config(
            family,
            {"extend": {"negative_axes": {"typoooo": ["x"]}}},
        )


# ----- FamilyConfig integration --------------------------------------------

def test_family_config_carries_axes_and_flat_string():
    fam = FamilyLoader().get_family("sdxl")
    assert isinstance(fam.negative_axes, dict)
    assert set(fam.negative_axes) == set(AXIS_KEYS)
    # Flat string is derived from axes
    assert fam.negative_prompt == flatten_axes(fam.negative_axes)


def test_every_family_has_canonical_axes_present():
    for fam in FamilyLoader().list_families():
        assert set(fam.negative_axes) == set(AXIS_KEYS), (
            f"family {fam.id!r} missing canonical axes"
        )


def test_flux_and_flux2_have_empty_axes():
    """CFG-distilled families have negatives ignored at render time;
    axes are empty by design."""
    for fam_id in ("flux", "flux2"):
        fam = FamilyLoader().get_family(fam_id)
        for axis in AXIS_KEYS:
            assert fam.negative_axes[axis] == []
        assert fam.negative_prompt == ""


def test_sdxl_axes_token_set_matches_legacy_flat_string():
    """The migration must preserve the original token set byte-for-byte
    at the lower-cased token level. This is the ``zero-diff guarantee``
    from the Phase D plan: any future axis migration that drops or
    adds a token gets caught here."""
    fam = FamilyLoader().get_family("sdxl")
    legacy = (
        "blurry, low quality, bad anatomy, distorted face, extra limbs, "
        "deformed hands, extra fingers, missing fingers, oversharpened, "
        "plastic skin, cartoon, unrealistic lighting, noise, artifacts, "
        "watermark, text, logo, low resolution, out of frame, cropped badly, "
        "duplicate, disfigured, mutation, ugly, poorly drawn face, "
        "poorly drawn hands"
    )
    legacy_tokens = {t.strip().lower() for t in legacy.split(",") if t.strip()}
    new_tokens = {t.strip().lower() for t in fam.negative_prompt.split(",") if t.strip()}
    assert legacy_tokens == new_tokens


# ----- assemble_negative_prompt + conflict_terms ---------------------------

@pytest.fixture
def pb():
    return PromptBuilder()


def test_assembler_axes_path_runs_conflict_filter(pb, caplog):
    """When axes + conflict_terms are passed, colliding tokens drop and log."""
    axes = {"quality": ["blurry"], "anatomy": ["bad anatomy"]}
    with caplog.at_level("WARNING"):
        out = pb.assemble_negative_prompt(
            model_negative_axes=axes,
            conflict_terms=["a blurry portrait of a woman"],
        )
    assert "blurry" not in out.lower().split(",")[1:]  # not in model layer
    # Filter logged the drop
    assert any("conflict filter" in r.message.lower() for r in caplog.records)
    # Untouched anatomy axis still flows through
    assert "bad anatomy" in out


def test_assembler_axes_path_no_conflict_keeps_all_tokens(pb):
    axes = {"quality": ["blurry", "lowres"], "watermark": ["watermark"]}
    out = pb.assemble_negative_prompt(
        model_negative_axes=axes,
        conflict_terms=["serene mountain landscape"],
    )
    for tok in ("blurry", "lowres", "watermark"):
        assert tok in out


def test_assembler_axes_short_circuited_by_supports_negative_false(pb):
    axes = {"quality": ["blurry"]}
    out = pb.assemble_negative_prompt(
        model_negative_axes=axes,
        supports_negative=False,
    )
    assert out == ""


def test_assembler_legacy_string_still_wins_when_passed(pb):
    """Backward compat: callers still on the flat-string path are not
    forced onto axes. ``model_negative`` takes precedence over axes."""
    axes = {"quality": ["from_axes"]}
    out = pb.assemble_negative_prompt(
        model_negative="from_string",
        model_negative_axes=axes,
    )
    assert "from_string" in out
    assert "from_axes" not in out


def test_assembler_axes_compose_with_ti_and_hard_block(pb):
    """Axes path coexists with TI hoisting and the HARD_BLOCK gate."""
    axes = {"anatomy": ["bad anatomy"]}
    out = pb.assemble_negative_prompt(
        model_negative_axes=axes,
        negative_embeddings=["embedding:BadDream"],
    )
    assert out.startswith("embedding:BadDream,")
    assert "child" in out  # HARD_BLOCK
    assert "bad anatomy" in out


def test_assembler_conflict_filter_drops_only_matching_tokens(pb):
    """Word-boundary match: positive prompt mentions one specific keyword,
    only that one drops."""
    axes = {
        "anatomy": ["bad anatomy", "extra fingers"],
        "quality": ["blurry"],
    }
    out = pb.assemble_negative_prompt(
        model_negative_axes=axes,
        conflict_terms=["hand close-up showing bad anatomy fingers"],
    )
    # "bad anatomy" matches → dropped. "extra fingers" — both words are
    # in the conflict set ("fingers" matches), so it also drops by the
    # whole-word subset rule.
    assert "bad anatomy" not in out
    # "blurry" doesn't appear in the positive — kept
    assert "blurry" in out
