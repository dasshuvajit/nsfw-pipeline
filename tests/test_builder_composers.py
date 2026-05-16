"""Pin exact prompt text per family.

If any composer drifts — an unexpected token order, a missing primary
field, a quality prefix reshape — these assertions fail loudly. That
contract is what lets Phase B build renderers without having to poke
Phase A again.
"""

from __future__ import annotations

import pytest

from src.memory.family_loader import FamilyLoader
from src.prompt.builder import PromptBuilder


# Shared fixtures — the scene + character + style triple is the same
# across families so family-specific primary fields are the only
# visible variable.
CHARACTER = {"base_prompt": "elara, woman with auburn hair"}
STYLE = {"base_style_keywords": "cinematic, editorial grade"}

UNIVERSAL_SCENE = {
    "variation_axis": "pose",
    "pose": "seated on bed",
    "camera": "medium shot",
    "camera_angle": "eye level",
    "lighting": "warm golden hour",
    "environment_detail": "silk sheets",
    "mood_note": "relaxed",
}


@pytest.fixture
def family_loader():
    return FamilyLoader()


@pytest.fixture
def pb():
    return PromptBuilder()


def test_sdxl_keywords_comma_joined_with_camera_spec(pb, family_loader):
    family = family_loader.get_family("sdxl")
    scene = {
        **UNIVERSAL_SCENE,
        "camera_spec": "85mm f/1.8, shallow DoF",
        "clothing": "silk slip",
    }
    out = pb.build_one(CHARACTER, scene, STYLE, family=family)
    text = out["prompt_text"]
    assert "elara" in text
    assert "85mm f/1.8" in text
    assert "silk slip" in text
    assert "cinematic" in text
    # comma-joined — no periods outside of lens f-stop
    stripped = text.replace("f/1.8", "f/X")
    assert "." not in stripped


def test_pony_prepends_quality_prefix_and_injects_source(pb, family_loader):
    family = family_loader.get_family("pony")
    scene = {
        **UNIVERSAL_SCENE,
        "booru_tags": "1girl, solo, sitting, bedroom",
    }
    out = pb.build_one(CHARACTER, scene, STYLE, family=family)
    text = out["prompt_text"]
    assert text.startswith("score_9, score_8_up")
    assert "BREAK" in text
    assert "1girl" in text
    assert "source_photograph" in text


def test_pony_prefers_booru_tags_over_universal(pb, family_loader):
    family = family_loader.get_family("pony")
    # booru_tags present — the universal scene fields should NOT
    # appear as additional comma-separated tokens because the primary-
    # field body is used instead.
    scene = {
        **UNIVERSAL_SCENE,
        "booru_tags": "1girl, solo, looking_at_viewer",
    }
    out = pb.build_one(CHARACTER, scene, STYLE, family=family)
    # "warm golden hour" was a universal-field value — should be absent
    # when booru_tags takes primacy.
    assert "warm golden hour" not in out["prompt_text"]
    assert "1girl" in out["prompt_text"]


def test_illustrious_includes_booru_and_prose_and_quality_suffix(pb, family_loader):
    family = family_loader.get_family("illustrious")
    scene = {
        **UNIVERSAL_SCENE,
        "booru_tags": "1girl, solo, sitting",
        "scene_prose": "an intimate boudoir scene at golden hour",
    }
    out = pb.build_one(CHARACTER, scene, STYLE, family=family)
    text = out["prompt_text"]
    assert "1girl" in text
    assert "intimate boudoir" in text
    assert text.endswith("newest") or "newest," in text  # quality suffix trailing


def test_flux_produces_prose_sentences(pb, family_loader):
    family = family_loader.get_family("flux")
    scene = {
        **UNIVERSAL_SCENE,
        "scene_prose": "She reclines on silk sheets at golden hour, warm light through the window.",
    }
    out = pb.build_one(CHARACTER, scene, STYLE, family=family)
    text = out["prompt_text"]
    # prose has periods between sentences
    assert ". " in text
    # no weighting syntax
    assert "(" not in text and ":" not in text
    # The prose sentence survives intact
    assert "reclines on silk sheets" in text


def test_chroma_appends_period_separated_realism_tail(pb, family_loader):
    family = family_loader.get_family("chroma")
    assert family.realism_tail_style == "period"
    scene = {
        **UNIVERSAL_SCENE,
        "scene_prose": "She reclines on silk sheets at golden hour.",
    }
    out = pb.build_one(CHARACTER, scene, STYLE, family=family)
    text = out["prompt_text"]
    # The period-separated tail is appended after the prose body.
    # Tail uses "photographic" (not "photorealistic") per Civitai
    # community guidance — photorealistic pushes toward photorealistic
    # ART, photographic pushes toward real photos. Updated 2026-05-17.
    assert "f/1.8" in text
    assert "35mm" in text
    assert "photographic" in text
    assert "natural skin texture" in text
    # Regression guard: the old "photorealistic" must be gone.
    assert "photorealistic" not in text
    # Ordering: tail comes after the prose
    assert text.index("reclines") < text.index("f/1.8")


def test_flux2_uses_scene_prose_and_omits_realism_tail(pb, family_loader):
    family = family_loader.get_family("flux2")
    scene = {
        **UNIVERSAL_SCENE,
        "scene_prose": (
            "She reclines on cream silk sheets in a late-afternoon bedroom. "
            "A single warm key light rakes across her collarbone from the left, "
            "casting deep amber shadows. The room feels still and intimate."
        ),
    }
    out = pb.build_one(CHARACTER, scene, STYLE, family=family)
    text = out["prompt_text"]
    # Prose body survives
    assert "reclines on cream silk sheets" in text
    # BFL Klein 9B guide bans the camera-tail dump — unlike Chroma, no
    # `f/1.8. 35mm. photographic.` tail should be appended.
    assert "f/1.8" not in text
    assert "photographic" not in text
    assert "natural skin texture" not in text


def test_flux2_negative_is_empty_regardless_of_inputs(pb, family_loader):
    family = family_loader.get_family("flux2")
    # Even when we try to pass a negative from the style profile, the
    # family declares supports_negative_prompt=false so assembly returns
    # "".
    negative = pb.assemble_negative_prompt(
        model_negative="ugly, blurry",
        style_negative="harsh flash",
        character_negative="extra limbs",
        supports_negative=family.supports_negative_prompt,
    )
    assert negative == ""


def test_flux2_strips_age_ambiguity_and_prepends_adult_anchor(pb, family_loader):
    family = family_loader.get_family("flux2")
    scene = {
        **UNIVERSAL_SCENE,
        "scene_prose": "A schoolgirl stands in warm window light.",
    }
    out = pb.build_one(CHARACTER, scene, STYLE, family=family)
    text = out["prompt_text"]
    assert "schoolgirl" not in text.lower()
    # Flux-style prose anchor, not the keyword variant
    assert "adult woman with mature features" in text.lower()


def test_prompt_hash_stable_across_calls(pb, family_loader):
    family = family_loader.get_family("sdxl")
    scene = {**UNIVERSAL_SCENE, "camera_spec": "85mm f/1.8"}
    a = pb.build_one(CHARACTER, scene, STYLE, family=family)
    b = pb.build_one(CHARACTER, scene, STYLE, family=family)
    assert a["prompt_hash"] == b["prompt_hash"]
    assert a["prompt_text"] == b["prompt_text"]


# ---- Phase 4a: vocabulary canonicalizer wired into build_one --------------


def test_sdxl_canonicalizes_lighting_directive(pb, family_loader):
    """LLM emits ``LIGHT_REMBRANDT``; the SDXL composer translates it
    into the family-shaped phrase from prompt_vocabulary.yaml."""
    family = family_loader.get_family("sdxl")
    scene = {
        **UNIVERSAL_SCENE,
        "camera_spec": "85mm f/1.8",
        "clothing": "silk slip",
        "lighting_directive": "LIGHT_REMBRANDT",
    }
    out = pb.build_one(CHARACTER, scene, STYLE, family=family)
    text = out["prompt_text"].lower()
    # SDXL phrasing for LIGHT_REMBRANDT
    assert "rembrandt lighting" in text
    assert "triangle of light" in text


def test_sdxl_canonicalizes_camera_lens_film_stock(pb, family_loader):
    """SDXL composer threads CAMERA / LENS / FILM phrasings into the
    body. The 77-token CLIP window may trim the middle, so assertions
    accept partial adjacency — just confirm each concept landed."""
    family = family_loader.get_family("sdxl")
    scene = {
        **UNIVERSAL_SCENE,
        "camera_spec": "wide DoF",   # generic — vocab phrases supplement
        "clothing": "linen sheet",
        "realism_camera": "CAMERA_SONY_A7RV",
        "realism_lens": "LENS_85MM_F14",
        "realism_film_stock": "FILM_PORTRA_400",
    }
    out = pb.build_one(CHARACTER, scene, STYLE, family=family)
    text = out["prompt_text"].lower()
    # Camera phrase distinctive prefix
    assert "sony a7r" in text or "ultra detailed sensor" in text
    # Lens — the leading 85mm always survives; the rest may be trimmed
    assert "85mm" in text
    # Film stock distinctive prefix
    assert "portra 400" in text


def test_pony_omits_camera_concepts_silently(pb, family_loader):
    """Pony has no camera/lens/film_stock phrasing in the vocabulary —
    those tags are dropped from the output without warning."""
    family = family_loader.get_family("pony")
    scene = {
        **UNIVERSAL_SCENE,
        "booru_tags": "1girl, looking_at_viewer",
        "realism_camera": "CAMERA_SONY_A7RV",       # Pony omits → drop
        "realism_lens": "LENS_85MM_F14",            # Pony omits → drop
        "lighting_directive": "LIGHT_REMBRANDT",     # Pony has this
        "mood_aesthetic": "MOOD_INTIMATE",           # Pony has this
    }
    out = pb.build_one(CHARACTER, scene, STYLE, family=family)
    text = out["prompt_text"].lower()
    # Lighting lands (mood may be trimmed by 77-token budget when
    # combined with quality_prefix; lighting is the more important
    # signal so we test it specifically)
    assert "rembrandt" in text
    # Camera + lens silently dropped (Pony has no phrasing for them)
    assert "sony" not in text
    assert "a7r" not in text
    assert "85mm" not in text


def test_flux_canonicalizes_into_prose_extras(pb, family_loader):
    """For prose families, vocab phrases land in extra_keywords →
    flowing-prose tail of the output."""
    family = family_loader.get_family("flux")
    scene = {
        **UNIVERSAL_SCENE,
        "scene_prose": "A woman reclines on a velvet chaise.",
        "lighting_directive": "LIGHT_GOLDEN_HOUR",
        "art_style_reference": "ART_FINE_NUDE",
    }
    out = pb.build_one(CHARACTER, scene, STYLE, family=family)
    text = out["prompt_text"]
    # Flux uses prose phrasing
    assert "golden-hour" in text.lower() or "golden hour" in text.lower()
    assert "fine art nude" in text.lower()


def test_flux2_canonicalizes_with_5_anchor_lighting(pb, family_loader):
    """FLUX.2 Klein: lighting_directive translates to BFL-style 5-anchor
    phrasing rich with directional + colour-temp detail."""
    family = family_loader.get_family("flux2")
    scene = {
        **UNIVERSAL_SCENE,
        "scene_prose": "A woman seated in a warmly lit study.",
        "lighting_directive": "LIGHT_REMBRANDT",
    }
    out = pb.build_one(CHARACTER, scene, STYLE, family=family)
    text = out["prompt_text"].lower()
    # FLUX.2 phrasing is more descriptive than SDXL's
    assert "rembrandt" in text
    assert ("camera-left" in text or "camera-right" in text
            or "key light" in text)


def test_nsfw_concept_dropped_below_tier(pb, family_loader):
    """T3-gated NSFW concept silently dropped at T2_implied content_level."""
    family = family_loader.get_family("sdxl")
    scene = {
        **UNIVERSAL_SCENE,
        "camera_spec": "85mm",
        "clothing": "fitted dress",
        "nsfw_anatomy": "NSFW_BREAST_NATURAL",  # T3+ gated
    }
    out = pb.build_one(
        CHARACTER, scene, STYLE,
        family=family, content_level="T2_implied",
    )
    text = out["prompt_text"].lower()
    # Phrase from NSFW_BREAST_NATURAL must NOT appear
    assert "natural breasts" not in text


def test_nsfw_concept_passes_at_t3(pb, family_loader):
    family = family_loader.get_family("sdxl")
    scene = {
        **UNIVERSAL_SCENE,
        "camera_spec": "85mm",
        "clothing": "draped silk",
        "nsfw_anatomy": "NSFW_BREAST_NATURAL",
    }
    out = pb.build_one(
        CHARACTER, scene, STYLE,
        family=family, content_level="T3_artnude",
    )
    text = out["prompt_text"].lower()
    assert "natural breasts" in text


def test_unknown_concept_silently_dropped(pb, family_loader):
    """LLM drift (unknown tag) doesn't crash — concept silently ignored."""
    family = family_loader.get_family("sdxl")
    scene = {
        **UNIVERSAL_SCENE,
        "camera_spec": "85mm",
        "clothing": "linen sheet",
        "lighting_directive": "LIGHT_DOES_NOT_EXIST",  # drift
    }
    # No exception raised; output just doesn't include the bogus tag verbatim
    out = pb.build_one(CHARACTER, scene, STYLE, family=family)
    assert "light_does_not_exist" not in out["prompt_text"].lower()


def test_canonicalizer_preserves_byte_stable_hash_with_no_concepts(pb, family_loader):
    """A scene with no vocab fields produces the same hash twice in a row
    (idempotency)."""
    family = family_loader.get_family("sdxl")
    plain_scene = {**UNIVERSAL_SCENE, "camera_spec": "85mm", "clothing": "silk"}
    out = pb.build_one(CHARACTER, plain_scene, STYLE, family=family)
    out2 = pb.build_one(CHARACTER, plain_scene, STYLE, family=family)
    assert out["prompt_hash"] == out2["prompt_hash"]


def test_canonicalizer_changes_hash_when_concept_added(pb, family_loader):
    family = family_loader.get_family("sdxl")
    base = {**UNIVERSAL_SCENE, "camera_spec": "85mm", "clothing": "silk"}
    enriched = {**base, "lighting_directive": "LIGHT_REMBRANDT"}
    a = pb.build_one(CHARACTER, base, STYLE, family=family)
    b = pb.build_one(CHARACTER, enriched, STYLE, family=family)
    # Enriched prompt has more content → different hash
    assert a["prompt_hash"] != b["prompt_hash"]
