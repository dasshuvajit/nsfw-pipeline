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


def test_illustrious_threads_extra_keywords_into_body(pb, family_loader):
    """Verifier round-2 B3 regression — Illustrious composer dropped
    extra_keywords entirely before the fix, so series-aesthetic
    phrases + scene vocab phrases never reached Illustrious renders.

    Before fix: `assert series_kw in text` would fail.
    After fix: extra_keywords land in body_segments, BEFORE the trailing
    style_keywords + quality suffix.
    """
    family = family_loader.get_family("illustrious")
    scene = {
        **UNIVERSAL_SCENE,
        "booru_tags": "1girl, solo, sitting",
        "scene_prose": "an intimate boudoir scene at golden hour",
    }
    # canonicalize_series_aesthetic emits the family-shaped phrasing
    # for the picked tag; PromptBuilder.build_one threads it via
    # series_plan. Use a concrete vocab tag the loader recognises so
    # the test exercises the real chain end-to-end.
    series_plan = {
        "color_palette": "PALETTE_TEAL_ORANGE_BLOCKBUSTER",
    }
    out = pb.build_one(
        CHARACTER, scene, STYLE, family=family, series_plan=series_plan,
    )
    text = out["prompt_text"]
    # The Illustrious phrasing for teal-orange landed in the body.
    # Look for "teal" + "orange" co-occurring — the canonicalizer's
    # output (booru-shaped underscore tags or short phrase) MUST
    # include the colour anchors.
    lower = text.lower()
    assert "teal" in lower, (
        "Illustrious dropped the series-aesthetic palette anchor — "
        "B3 regressed."
    )
    assert "orange" in lower, (
        "Illustrious dropped the series-aesthetic palette anchor — "
        "B3 regressed."
    )


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


@pytest.mark.skip(
    reason="Dual-write pivot (2026-05-23): series_aesthetic canonicalization "
    "no longer flows into chroma/flux/flux2 final prompts. LLM's "
    "scene_prose weaves the aesthetic itself. Test asserted OLD "
    "behavior where planner's palette anchor 'teal/orange' appeared "
    "via series_aesthetic_for_extras — now dropped for prose families. "
    "Archetype suppression test deprecated; see new tests below."
)
def test_archetype_suppressed_when_planner_provided_anchors(pb, family_loader):
    """Round-21 (2026-05-21) — when ``series_plan`` carries planner-
    chosen aesthetic anchors (color_palette / photographer_ref /
    art_movement) the operator's archetype ``base_style_keywords`` is
    suppressed so it can't contradict the planner's vision.

    The audit on series_799bec97e6d7 found every prompt carrying both
    ``golden_hour_natural``'s "Golden hour, warm rim-light, haze,
    natural outdoor" AND a planner-chosen Helmut-Newton/film-noir/
    neon-noir aesthetic — direct visual contradiction."""
    family = family_loader.get_family("chroma")
    scene = {
        **UNIVERSAL_SCENE,
        "scene_prose": "She reclines on silk sheets in a dim apartment.",
    }
    # Style profile carries the archetype keywords the operator picked.
    archetype = {
        "base_style_keywords": (
            "Golden hour, warm rim-light, haze, natural outdoor, "
            "lifestyle editorial, soft glow"
        ),
    }
    # Planner chose its OWN aesthetic anchors — these override archetype.
    series_plan = {
        "color_palette": "PALETTE_TEAL_ORANGE_BLOCKBUSTER",
        "photographer_ref": "PHOTOG_HELMUT_NEWTON",
        "art_movement": "ART_MOVE_FILM_NOIR_1940S",
    }
    out = pb.build_one(
        CHARACTER, scene, archetype,
        family=family, series_plan=series_plan,
    )
    text = out["prompt_text"]
    # Archetype keywords must NOT appear in the prompt.
    assert "Golden hour" not in text, (
        "archetype `Golden hour` leaked into a planner-overridden series — "
        f"round-21 fix regressed. prompt={text!r}"
    )
    assert "natural outdoor" not in text, (
        f"archetype keywords leaked — got: {text!r}"
    )
    # Planner anchors DID land via series_aesthetic canonicalization.
    lower = text.lower()
    assert "teal" in lower and "orange" in lower, (
        "planner's color_palette anchor missing"
    )


@pytest.mark.skip(
    reason="Dual-write pivot (2026-05-23): archetype style_keywords are "
    "now ALWAYS suppressed for prose families (chroma/flux/flux2). "
    "The LLM's scene_prose handles style. This test asserted the "
    "back-compat path where archetype 'Golden hour' was injected — "
    "now obsolete for chroma. SDXL keyword family still gets archetype."
)
def test_archetype_kept_when_planner_provides_no_anchors(pb, family_loader):
    """Round-21 — the override fires ONLY when the planner provided
    aesthetic anchors. Back-compat series without anchors still get
    the operator's archetype keywords injected."""
    family = family_loader.get_family("chroma")
    scene = {
        **UNIVERSAL_SCENE,
        "scene_prose": "She reclines on silk sheets at golden hour.",
    }
    archetype = {
        "base_style_keywords": "Golden hour, warm rim-light, soft glow",
    }
    # No series_plan → archetype must still be injected.
    out = pb.build_one(CHARACTER, scene, archetype, family=family)
    assert "Golden hour" in out["prompt_text"], (
        "archetype was suppressed even though planner provided no anchors — "
        "round-21 fix over-fired."
    )
    # Empty series_plan → same as None.
    out2 = pb.build_one(
        CHARACTER, scene, archetype,
        family=family, series_plan={},
    )
    assert "Golden hour" in out2["prompt_text"], (
        "archetype was suppressed for an EMPTY series_plan — round-21 "
        "fix over-fired."
    )


def test_chroma_realism_tail_without_lens_keeps_focal_hint(pb, family_loader):
    """Round-22 (2026-05-22) — when the facet has NO realism_lens
    populated, the chroma realism tail keeps all four anchor tokens
    (f/1.8, 35mm, photographic, natural skin texture) so the prompt
    still has a focal-length hint. Backwards-compat for the optional-
    lens path."""
    family = family_loader.get_family("chroma")
    assert family.realism_tail_style == "period"
    scene = {
        **UNIVERSAL_SCENE,
        "scene_prose": "She reclines on silk sheets at golden hour.",
        # No realism_lens — tail keeps focal hint.
    }
    out = pb.build_one(CHARACTER, scene, STYLE, family=family)
    text = out["prompt_text"]
    # All four anchor tokens present in the trailing comma sentence.
    assert "f/1.8" in text
    assert "35mm" in text
    assert "photographic" in text
    assert "natural skin texture" in text
    assert "f/1.8, 35mm" in text, (
        f"expected comma-separated realism tail with focal hint, got: {text!r}"
    )


def test_chroma_fallback_prose_t4_demands_nudity(pb, family_loader, caplog):
    """When the LLM facet fails (scene_prose empty), the chroma fallback
    must honour content_level. Pre-fix the hardcoded "tasteful
    photographic composition" line caused SFW leaks on T4_explicit
    series. The tier-aware fallback explicitly requests nudity at T4.
    """
    family = family_loader.get_family("chroma")
    scene = {**UNIVERSAL_SCENE, "scene_prose": ""}  # facet-failure case
    out = pb.build_one(
        CHARACTER, scene, STYLE, family=family,
        content_level="T4_explicit",
    )
    text = out["prompt_text"].lower()
    # The T4 fallback must reach for explicit nudity vocabulary.
    assert any(token in text for token in ("nude", "explicit", "anatomy")), (
        f"T4 fallback prose missing nudity directive: {out['prompt_text']!r}"
    )
    # And NOT degrade to the old tier-blind line.
    assert "tasteful, photographic composition" not in text, (
        f"T4 fallback still uses the tier-blind boilerplate: "
        f"{out['prompt_text']!r}"
    )


def test_chroma_fallback_prose_t3_art_nude(pb, family_loader):
    """T3 fallback says 'fully nude … art-nude' — softer than T4."""
    family = family_loader.get_family("chroma")
    scene = {**UNIVERSAL_SCENE, "scene_prose": ""}
    out = pb.build_one(
        CHARACTER, scene, STYLE, family=family,
        content_level="T3_artnude",
    )
    text = out["prompt_text"].lower()
    assert "nude" in text
    assert "art-nude" in text or "art nude" in text


def test_chroma_fallback_prose_t1_stays_safe(pb, family_loader):
    """T1 fallback must not introduce nudity vocabulary."""
    family = family_loader.get_family("chroma")
    scene = {**UNIVERSAL_SCENE, "scene_prose": ""}
    out = pb.build_one(
        CHARACTER, scene, STYLE, family=family,
        content_level="T1_suggestive",
    )
    text = out["prompt_text"].lower()
    assert "nude" not in text
    assert "explicit" not in text
    # The standing T1 phrasing is preserved verbatim.
    assert "tasteful, photographic composition" in text


def test_chroma_fallback_warns_on_facet_failure(pb, family_loader, caplog):
    """Operator visibility: every fallback fire logs at WARNING so a
    series isn't silently degraded with sparse prompts."""
    import logging
    family = family_loader.get_family("chroma")
    scene = {**UNIVERSAL_SCENE, "scene_prose": ""}
    with caplog.at_level(logging.WARNING, logger="src.prompt.builder"):
        pb.build_one(
            CHARACTER, scene, STYLE, family=family,
            content_level="T4_explicit",
        )
    assert any(
        "chroma fallback prose fired" in rec.message
        for rec in caplog.records
    ), [rec.message for rec in caplog.records]


def test_chroma_prompt_ends_with_period(pb, family_loader):
    """Verifier NC7 (2026-05-23) — every chroma/flux/flux2 prompt must
    end with a period. fit_to_budget's piece-pack occasionally drops
    the trailing period at separator boundaries, leaving the prompt
    looking unfinished. Defensive period-ensure at the end of
    build_one fixes this."""
    family = family_loader.get_family("chroma")
    scene = {
        **UNIVERSAL_SCENE,
        "scene_prose": "She reclines on velvet sheets in soft window light.",
    }
    out = pb.build_one(CHARACTER, scene, STYLE, family=family)
    text = out["prompt_text"]
    assert text.rstrip().endswith("."), (
        f"chroma prompt does not end with period: ...{text[-50:]!r}"
    )


def test_flux2_prompt_ends_with_period(pb, family_loader):
    """Same NC7 fix applied to flux2_prose."""
    family = family_loader.get_family("flux2")
    scene = {
        **UNIVERSAL_SCENE,
        "scene_prose": "She reclines on velvet sheets in soft window light.",
    }
    out = pb.build_one(CHARACTER, scene, STYLE, family=family)
    text = out["prompt_text"]
    assert text.rstrip().endswith("."), (
        f"flux2 prompt does not end with period: ...{text[-50:]!r}"
    )


@pytest.mark.skip(
    reason="Dual-write pivot (2026-05-23): F4 series-aesthetic 1-sentence "
    "consolidation for chroma is obsolete. Series aesthetic anchors "
    "(palette + photographer + art_movement) are now WOVEN into "
    "scene_prose by the LLM, not appended as a separate sentence by "
    "the composer. This test asserted the old F4 behavior."
)
def test_series_aesthetic_consolidated_one_sentence_chroma(pb, family_loader):
    """Round-22 (2026-05-22) — for prose families (chroma uses
    flux_natural prompt_style), the 3 series-aesthetic anchors (palette
    + photographer + art_movement) consolidate into ONE comma-joined
    sentence in the final prompt. Saves ~60-80 tokens per prompt vs
    the prior 3-separate-sentences form."""
    family = family_loader.get_family("chroma")
    scene = {
        **UNIVERSAL_SCENE,
        "scene_prose": "She reclines on silk sheets in a dim apartment.",
    }
    series_plan = {
        "color_palette": "PALETTE_TEAL_ORANGE_BLOCKBUSTER",
        "photographer_ref": "PHOTOG_HELMUT_NEWTON",
        "art_movement": "ART_MOVE_FILM_NOIR_1940S",
    }
    out = pb.build_one(
        CHARACTER, scene, STYLE,
        family=family, series_plan=series_plan,
    )
    text = out["prompt_text"]
    # All three aesthetic anchors are present in the prompt text.
    lower = text.lower()
    assert "teal" in lower and "orange" in lower, "palette anchor missing"
    assert "helmut newton" in lower, "photographer anchor missing"
    assert "film noir" in lower or "1940s" in lower, "art movement missing"
    # Round-22: the three anchors are joined as ONE sentence (no
    # period between palette + photographer + movement). Pre-fix the
    # three landed as 3 separate sentences with periods between.
    # We can't grep for a single specific phrase order, but we CAN
    # count period-separated chroma realism sentences:
    # the prompt should contain at most ONE period between palette
    # ending and photographer start.
    # Find the palette substring and the photographer substring.
    palette_idx = lower.find("teal")
    photog_idx = lower.find("helmut newton")
    assert palette_idx < photog_idx, "palette should come before photographer"
    # Between palette region and photographer, count periods.
    between = text[palette_idx:photog_idx]
    period_count = between.count(".")
    assert period_count <= 0, (
        f"expected NO period between consolidated palette + photographer "
        f"(round-22 consolidation), got {period_count} period(s). "
        f"between text: {between!r}"
    )


def test_series_aesthetic_three_segments_kept_for_booru_families(pb, family_loader):
    """Round-22 — booru families (pony / illustrious) keep the
    3-separate-segments shape. Their composers expect atomic items
    that the keyword dedup can re-order independently."""
    family = family_loader.get_family("illustrious")
    scene = {
        **UNIVERSAL_SCENE,
        "booru_tags": "1girl, solo, sitting",
        "scene_prose": "an intimate boudoir scene at golden hour",
    }
    series_plan = {
        "color_palette": "PALETTE_TEAL_ORANGE_BLOCKBUSTER",
        "photographer_ref": "PHOTOG_HELMUT_NEWTON",
        "art_movement": "ART_MOVE_FILM_NOIR_1940S",
    }
    out = pb.build_one(
        CHARACTER, scene, STYLE,
        family=family, series_plan=series_plan,
    )
    # We don't strongly assert the order/format for booru families
    # (illustrious composer dedups + rearranges tokens). We only
    # verify that the three anchors all land — and that the consolidation
    # branch did NOT consolidate them.
    lower = out["prompt_text"].lower()
    assert "teal" in lower, "palette anchor missing in illustrious"
    assert "helmut" in lower or "newton" in lower, "photographer missing in illustrious"


@pytest.mark.skip(
    reason="Dual-write pivot (2026-05-23): per-scene realism_lens "
    "canonicalization ('85mm f/1.4 lens, shallow DoF, smooth bokeh') "
    "no longer flows into chroma prompts as a separate sentence. "
    "Lens info is woven into scene_prose by the LLM. The chroma realism "
    "tail still drops f/1.8/35mm when realism_lens populated — that "
    "logic is preserved but the lens canonicalization isn't visible "
    "in the test's mock-prose-only fixture."
)
def test_chroma_realism_tail_strips_focal_when_lens_populated(pb, family_loader):
    """Round-22 — when the facet has a per-scene realism_lens
    canonicalized (e.g. LENS_85MM_F14 → "85mm f/1.4 lens..."), the
    chroma realism tail drops its hardcoded "f/1.8, 35mm" tokens to
    avoid two-focal-length contradiction in the same prompt. Tail
    keeps "photographic, natural skin texture" unconditionally —
    those are family realism anchors, not focal specs."""
    family = family_loader.get_family("chroma")
    scene = {
        **UNIVERSAL_SCENE,
        "scene_prose": "She reclines on silk sheets at golden hour.",
        "realism_lens": "LENS_85MM_F14",  # canonicalizes to 85mm f/1.4
    }
    out = pb.build_one(CHARACTER, scene, STYLE, family=family)
    text = out["prompt_text"]
    # The LLM-picked lens canonicalization is present.
    assert "85mm f/1.4" in text or "85mm" in text, (
        f"expected per-scene lens canonicalization, got: {text!r}"
    )
    # The tail's focal-spec tokens MUST NOT appear (would double-spec).
    assert "f/1.8, 35mm" not in text, (
        f"round-22 fix regressed — tail still injected f/1.8 + 35mm "
        f"despite facet lens being populated. prompt={text!r}"
    )
    # Tail's family realism anchors are STILL present.
    assert "photographic" in text
    assert "natural skin texture" in text


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


@pytest.mark.skip(
    reason="Dual-write pivot (2026-05-23): vocab canonicalization no "
    "longer flows into flux/chroma prompts as separate sentences. "
    "Canonicalize_facet still runs (output goes to DB analytics + "
    "diversity tracker) but is dropped from the final rendered prompt "
    "for prose families. Test asserted the OLD behavior where "
    "vocab_phrases landed in prose_extras."
)
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


@pytest.mark.skip(
    reason="Dual-write pivot (2026-05-23): same as flux test above — "
    "vocab canonicalization dropped from final prompt for prose families. "
    "flux2_prose composer now relies on LLM's scene_prose to weave the "
    "BFL 5-anchor structure."
)
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


# ── Round-21: archetype-override helper unit tests ─────────────────


def test_archetype_overridden_helper_recognises_each_anchor():
    """Round-21 — any one of color_palette / photographer_ref /
    art_movement on the series_plan flips the override."""
    from src.prompt.builder import archetype_overridden_by_planner
    assert archetype_overridden_by_planner({"color_palette": "PALETTE_X"})
    assert archetype_overridden_by_planner({"photographer_ref": "PHOTOG_Y"})
    assert archetype_overridden_by_planner({"art_movement": "ART_MOVE_Z"})


def test_archetype_overridden_helper_false_on_empty_or_none():
    """Round-21 — back-compat: None / empty / no-anchors series_plan
    must NOT flip the override (old series rendered before vocab v6)."""
    from src.prompt.builder import archetype_overridden_by_planner
    assert not archetype_overridden_by_planner(None)
    assert not archetype_overridden_by_planner({})
    assert not archetype_overridden_by_planner({
        "theme": "X", "mood": "Y", "environment": "Z",
    })
    # Empty-string anchor is treated as missing.
    assert not archetype_overridden_by_planner({"color_palette": ""})
    assert not archetype_overridden_by_planner({"color_palette": "   "})
