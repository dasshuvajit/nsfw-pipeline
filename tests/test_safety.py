"""Child-block hard-negative + celebrity-likeness strip end-to-end."""

from __future__ import annotations

import pytest

from src.memory.family_loader import FamilyLoader
from src.prompt.builder import HARD_BLOCK_NEGATIVE, PromptBuilder
from src.prompt.sanitizer import (
    _detect_celebrity_likeness,
    _strip_celebrity_likeness,
    _load_archetype_2grams,
)


@pytest.fixture
def pb():
    return PromptBuilder()


@pytest.fixture
def family_loader():
    return FamilyLoader()


# ---- Layer 1 — HARD_BLOCK negative prompt ------------------------------


def test_hard_block_contains_all_critical_terms():
    for term in ("child", "kid", "young", "minor", "teen", "schoolgirl",
                 "loli", "shota", "underage", "baby"):
        assert term in HARD_BLOCK_NEGATIVE


def test_hard_block_prepended_to_negative(pb):
    neg = pb.assemble_negative_prompt(
        model_negative="blurry",
        supports_negative=True,
    )
    assert neg.startswith("child")
    assert "blurry" in neg


# ---- Layer 2 — Positive-side age scan ---------------------------------


@pytest.mark.parametrize("family_id", ["sdxl", "pony", "illustrious"])
def test_positive_age_scan_keyword_families(pb, family_loader, family_id):
    """Each keyword family inserts its own ``family.adult_anchor.keyword``
    when age-ambiguity vocabulary is stripped — Pony has its own anchor
    (`1woman, mature, adult`); SDXL/Illustrious use the global default."""
    family = family_loader.get_family(family_id)
    character = {"base_prompt": "woman with dark hair"}
    scene = {
        "variation_axis": "pose",
        "pose": "schoolgirl in uniform",
        "camera": "medium shot",
        "camera_angle": "eye level",
        "lighting": "soft",
        "environment_detail": "studio",
        "mood_note": "playful",
    }
    style = {"base_style_keywords": "cinematic"}
    out = pb.build_one(character, scene, style, family=family)
    text = out["prompt_text"].lower()
    assert "schoolgirl" not in text
    # The family's own anchor was injected — split into tokens so the
    # assertion is robust to future per-family customisation.
    expected_anchor = family.adult_anchor["keyword"].lower()
    for tok in [t.strip() for t in expected_anchor.split(",") if t.strip()]:
        assert tok in text, f"family={family_id}: missing anchor token {tok!r}"


@pytest.mark.parametrize("family_id", ["flux", "chroma", "flux2"])
def test_positive_age_scan_prose_families(pb, family_loader, family_id):
    """Prose families inject ``family.adult_anchor.prose`` (a sentence)."""
    family = family_loader.get_family(family_id)
    character = {"base_prompt": "woman with dark hair"}
    scene = {
        "variation_axis": "pose",
        "pose": "teen posing",
        "camera": "medium shot",
        "camera_angle": "eye level",
        "lighting": "soft",
        "environment_detail": "studio",
        "mood_note": "playful",
    }
    style = {"base_style_keywords": "cinematic"}
    out = pb.build_one(character, scene, style, family=family)
    text = out["prompt_text"].lower()
    assert "teen" not in text
    # Prose anchor should appear (case-insensitive) — match a few
    # distinctive words rather than the full sentence.
    expected_prose = family.adult_anchor["prose"].lower()
    # "adult" + "mature" both appear in every family default + Pony override
    assert "adult" in text or "mature" in text
    # Keep a stronger assertion when the family uses the default prose.
    if "adult woman" in expected_prose:
        assert "adult woman" in text


def test_positive_age_scan_leaves_clean_prompts_untouched(pb, family_loader):
    family = family_loader.get_family("sdxl")
    character = {"base_prompt": "woman"}
    scene = {
        "variation_axis": "pose",
        "pose": "seated",
        "camera": "medium shot",
        "camera_angle": "eye level",
        "lighting": "soft",
        "environment_detail": "studio",
        "mood_note": "relaxed",
    }
    style = {"base_style_keywords": "cinematic"}
    out = pb.build_one(character, scene, style, family=family)
    text = out["prompt_text"].lower()
    # No anchor injected when no age-ambiguity term present
    assert "adult woman" not in text


# ---- Phase 3b: per-family adult_anchor sourced from family config -------


def test_pony_adult_anchor_uses_booru_token_form(pb, family_loader):
    """Pony's anchor (`1woman, mature, adult`) honours booru tag
    conventions — distinct from the default keyword form."""
    family = family_loader.get_family("pony")
    assert family.adult_anchor["keyword"] == "1woman, mature, adult"
    assert "adult" in family.adult_anchor["prose"].lower()


@pytest.mark.parametrize("family_id", [
    "sdxl", "illustrious", "flux", "chroma", "flux2",
])
def test_non_pony_families_use_default_anchor(family_loader, family_id):
    """Every non-Pony family inherits the dataclass default until it
    declares its own override in families.yaml."""
    family = family_loader.get_family(family_id)
    assert family.adult_anchor["keyword"] == "adult woman, mature features"
    assert family.adult_anchor["prose"] == (
        "An adult woman with mature features."
    )


# ---- Layer 3 — Celebrity-likeness heuristic ---------------------------


def test_archetype_2grams_loaded():
    grams = _load_archetype_2grams()
    assert len(grams) > 0
    # Derived from style_profiles.yaml display names
    assert "old hollywood" in grams


def test_celebrity_allowlist_keeps_archetype_grams():
    # "Old Hollywood" is an archetype — must NOT flag
    assert _detect_celebrity_likeness("shot in Old Hollywood style") == []


def test_celebrity_allowlist_keeps_geography():
    assert _detect_celebrity_likeness("shot in New York at dusk") == []
    assert _detect_celebrity_likeness("at Los Angeles beach") == []


def test_celebrity_allowlist_keeps_photography_grams():
    assert _detect_celebrity_likeness("Medium Shot, eye level") == []


def test_celebrity_heuristic_flags_real_names():
    out = _detect_celebrity_likeness("Emma Stone sitting on bed")
    assert "Emma Stone" in out

    out = _detect_celebrity_likeness("dressed like Taylor Swift")
    assert "Taylor Swift" in out


def test_strip_celebrity_removes_match():
    text = "elara, woman, Emma Stone styling, seated"
    stripped = _strip_celebrity_likeness(text)
    assert "Emma Stone" not in stripped
    assert "elara" in stripped
    # Stripping preserves the rest of the comma list cleanly
    assert ",," not in stripped


def test_prose_sentence_starts_not_falsely_flagged():
    # Flux capitalizes sentence starts — first word single, rest lowercase.
    # The 2-gram pattern requires TWO consecutive capitals, so these are
    # safe.
    out = _detect_celebrity_likeness(
        "She reclines on silk sheets. The warm sunlight falls through the window."
    )
    assert out == []
