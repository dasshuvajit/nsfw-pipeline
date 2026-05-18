"""Vocabulary library + canonicalizer — Phase 4a.

The vocabulary file (``config/prompt_vocabulary.yaml``) is the single
source of truth for realism + NSFW phrasing per family. The LLM emits
abstract enum tags from a small enumerated menu; the composer
translates them at compose time. Tests verify:

* Each concept canonicalises to the correct family-shaped phrase.
* NSFW concepts are tier-gated; below-tier concepts silently drop.
* Unknown concepts (LLM drift) silently drop with an INFO log.
* Pony's omission of camera / lens / film_stock / art_style is honoured.
* :func:`llm_vocabulary_block` builds a non-empty system-prompt menu
  for every family.
* :func:`canonicalize_facet` produces stable, order-preserving output.
"""

from __future__ import annotations

import logging

import pytest

from src.prompt.vocabulary import (
    VocabularyLoader,
    canonicalize_facet,
    llm_vocabulary_block,
)


@pytest.fixture
def loader():
    return VocabularyLoader()


# ── version + structure ─────────────────────────────────────────────


def test_vocab_version_field_present(loader):
    assert loader.version >= 1


def test_realism_namespaces_populated(loader):
    """The four realism namespaces all carry concepts."""
    assert loader.concepts_by_namespace("realism", "lighting")
    assert loader.concepts_by_namespace("realism", "camera")
    assert loader.concepts_by_namespace("realism", "lens")
    assert loader.concepts_by_namespace("realism", "film_stock")
    assert loader.concepts_by_namespace("realism", "art_style")
    assert loader.concepts_by_namespace("realism", "mood")


def test_nsfw_namespaces_populated(loader):
    """Phase 4a ships T3 anatomy + posture; T4 acts deferred."""
    assert loader.concepts_by_namespace("nsfw", "anatomy")
    assert loader.concepts_by_namespace("nsfw", "posture")


# ── canonicalize: realism happy path ────────────────────────────────


@pytest.mark.parametrize("family", [
    "sdxl", "pony", "illustrious", "flux", "chroma", "flux2",
])
def test_lighting_rembrandt_canonicalizes_for_every_family(loader, family):
    out = loader.canonicalize("LIGHT_REMBRANDT", family)
    assert out is not None
    # Each family must produce a non-empty phrase that mentions Rembrandt.
    assert "rembrandt" in out.lower()


@pytest.mark.parametrize("family", [
    "sdxl", "illustrious", "flux", "chroma", "flux2",
])
def test_camera_canonicalizes_for_every_camera_family(loader, family):
    """Pony omits camera tags — booru tagging carries it via source_photograph."""
    out = loader.canonicalize("CAMERA_SONY_A7RV", family)
    assert out is not None
    assert "sony" in out.lower() or "a7r" in out.lower()


@pytest.mark.parametrize("family", [
    "sdxl", "illustrious", "flux", "chroma", "flux2",
])
def test_lens_canonicalizes_for_every_camera_family(loader, family):
    """Lens namespace mirrors camera — Pony omits, others provide."""
    out = loader.canonicalize("LENS_85MM_F14", family)
    assert out is not None
    assert "85" in out


def test_pony_omits_camera_concepts(loader):
    """Pony deliberately has no camera/lens/film_stock/art_style phrasing."""
    assert loader.canonicalize("CAMERA_SONY_A7RV", "pony") is None
    assert loader.canonicalize("LENS_85MM_F14", "pony") is None
    assert loader.canonicalize("FILM_PORTRA_400", "pony") is None
    assert loader.canonicalize("ART_FINE_NUDE", "pony") is None


def test_pony_keeps_lighting_and_mood(loader):
    """Lighting + mood canonicalise for Pony — they DO translate to
    booru-compatible phrasings."""
    assert loader.canonicalize("LIGHT_REMBRANDT", "pony") is not None
    assert loader.canonicalize("MOOD_INTIMATE", "pony") is not None


# ── canonicalize: NSFW tier gating ─────────────────────────────────


def test_t3_concept_drops_at_t1_content_level(loader):
    """Below-tier NSFW concepts are silently dropped."""
    out = loader.canonicalize(
        "NSFW_BREAST_NATURAL", "sdxl",
        content_level="T1_suggestive",
    )
    assert out is None


def test_t3_concept_drops_at_t2_content_level(loader):
    out = loader.canonicalize(
        "NSFW_BREAST_NATURAL", "sdxl",
        content_level="T2_implied",
    )
    assert out is None


def test_t3_concept_passes_at_t3_content_level(loader):
    out = loader.canonicalize(
        "NSFW_BREAST_NATURAL", "sdxl",
        content_level="T3_artnude",
    )
    assert out is not None


def test_t3_concept_passes_at_t4_content_level(loader):
    """T4 is above the T3 gate — concept passes through."""
    out = loader.canonicalize(
        "NSFW_BREAST_NATURAL", "sdxl",
        content_level="T4_explicit",
    )
    assert out is not None


def test_t4_concept_drops_at_t3_content_level(loader):
    """An NSFW concept gated at T4 is dropped at T3 (and below)."""
    out = loader.canonicalize(
        "NSFW_INTIMATE", "sdxl",
        content_level="T3_artnude",
    )
    assert out is None


def test_t4_concept_passes_at_t4_content_level(loader):
    out = loader.canonicalize(
        "NSFW_INTIMATE", "sdxl",
        content_level="T4_explicit",
    )
    assert out is not None


def test_nsfw_concept_with_no_content_level_drops(loader):
    """Defence-in-depth — missing active tier is rejected for any
    tier-gated concept."""
    out = loader.canonicalize(
        "NSFW_BREAST_NATURAL", "sdxl",
        content_level=None,
    )
    assert out is None


def test_nsfw_concept_outside_tier_order_drops(loader):
    """Garbage tier strings are rejected."""
    out = loader.canonicalize(
        "NSFW_BREAST_NATURAL", "sdxl",
        content_level="T_garbage",
    )
    assert out is None


# ── unknown concepts / robustness ──────────────────────────────────


def test_unknown_concept_returns_none(loader, caplog):
    with caplog.at_level(logging.INFO):
        out = loader.canonicalize("LIGHT_DOES_NOT_EXIST", "sdxl")
    assert out is None
    assert any("unknown concept" in r.message for r in caplog.records)


def test_empty_concept_returns_none(loader):
    assert loader.canonicalize("", "sdxl") is None
    assert loader.canonicalize(None, "sdxl") is None  # type: ignore[arg-type]


def test_helper_tier_min_lookup(loader):
    assert loader.tier_min_for("NSFW_BREAST_NATURAL") == "T3_artnude"
    assert loader.tier_min_for("NSFW_INTIMATE") == "T4_explicit"
    # Realism concept — no tier gate.
    assert loader.tier_min_for("LIGHT_REMBRANDT") is None


def test_helper_is_nsfw(loader):
    assert loader.is_nsfw_concept("NSFW_BREAST_NATURAL") is True
    assert loader.is_nsfw_concept("LIGHT_REMBRANDT") is False
    assert loader.is_nsfw_concept("does_not_exist") is False


# ── canonicalize_facet end-to-end ─────────────────────────────────


def test_canonicalize_facet_produces_ordered_phrases(loader):
    """A facet with several enum fields produces phrases in
    declaration order (matches ``_FIELD_TO_NAMESPACE``)."""
    facet = {
        "realism_camera":      "CAMERA_SONY_A7RV",
        "realism_lens":        "LENS_85MM_F14",
        "realism_film_stock":  "FILM_PORTRA_400",
        "lighting_directive":  "LIGHT_REMBRANDT",
        "mood_aesthetic":      "MOOD_INTIMATE",
        "art_style_reference": "ART_FINE_NUDE",
    }
    out = canonicalize_facet(
        facet, "sdxl", content_level="T2_implied", loader=loader,
    )
    # Six phrases — one per field — in declaration order
    assert len(out) == 6
    # Camera phrase first, art style last
    assert "Sony" in out[0] or "sony" in out[0]
    assert "fine art" in out[5].lower()


def test_canonicalize_facet_drops_below_tier_nsfw(loader):
    """Below-tier NSFW concepts silently dropped from output."""
    facet = {
        "lighting_directive": "LIGHT_REMBRANDT",
        "nsfw_anatomy":       "NSFW_BREAST_NATURAL",   # T3-gated
    }
    out = canonicalize_facet(
        facet, "sdxl", content_level="T1_suggestive", loader=loader,
    )
    # Only the lighting concept survives
    assert len(out) == 1
    assert "rembrandt" in out[0].lower()


def test_canonicalize_facet_keeps_in_tier_nsfw(loader):
    facet = {
        "lighting_directive": "LIGHT_REMBRANDT",
        "nsfw_anatomy":       "NSFW_BREAST_NATURAL",   # T3-gated
    }
    out = canonicalize_facet(
        facet, "sdxl", content_level="T3_artnude", loader=loader,
    )
    # Both survive
    assert len(out) == 2


def test_canonicalize_facet_skips_pony_omitted_namespaces(loader):
    """Pony has no camera phrasing — the field is silently dropped
    from the output even though the LLM emitted a camera tag."""
    facet = {
        "realism_camera":     "CAMERA_85MM_F14",  # Pony omits cameras
        "lighting_directive": "LIGHT_REMBRANDT",   # Pony has lighting
        "mood_aesthetic":     "MOOD_INTIMATE",     # Pony has mood
    }
    out = canonicalize_facet(facet, "pony", loader=loader)
    # Only lighting + mood survive (camera dropped)
    assert len(out) == 2


def test_canonicalize_facet_empty_input(loader):
    out = canonicalize_facet({}, "sdxl", loader=loader)
    assert out == []


def test_canonicalize_facet_unknown_concepts_dropped(loader):
    facet = {
        "realism_camera":     "CAMERA_INVENTED_BY_LLM",  # drift
        "lighting_directive": "LIGHT_REMBRANDT",
    }
    out = canonicalize_facet(facet, "sdxl", loader=loader)
    # Only the valid one survives
    assert len(out) == 1


# ── llm_vocabulary_block ───────────────────────────────────────────


@pytest.mark.parametrize("family", [
    "sdxl", "pony", "illustrious", "flux", "chroma", "flux2",
])
def test_llm_vocabulary_block_produced_for_every_family(loader, family):
    block = llm_vocabulary_block(family, loader=loader)
    assert block, f"family={family} produced empty vocabulary block"
    assert "REALISM VOCABULARY" in block
    # Each family has at least lighting concepts
    assert "realism.lighting" in block
    # Each family has mood concepts
    assert "realism.mood" in block


def test_llm_vocabulary_block_lists_concept_tags(loader):
    block = llm_vocabulary_block("sdxl", loader=loader)
    # Tags present in the menu
    assert "LIGHT_REMBRANDT" in block
    assert "CAMERA_SONY_A7RV" in block
    assert "LENS_85MM_F14" in block
    assert "MOOD_INTIMATE" in block


def test_llm_vocabulary_block_omits_pony_camera_namespace(loader):
    """Pony's block doesn't list the camera namespace because there's
    no Pony phrasing for any CAMERA_* concept."""
    block = llm_vocabulary_block("pony", loader=loader)
    assert "realism.lighting" in block
    assert "realism.mood" in block
    # No camera / lens / film_stock / art_style — Pony omits these
    assert "realism.camera" not in block
    assert "realism.lens" not in block
    assert "realism.film_stock" not in block
    assert "realism.art_style" not in block


# ── Phase 0 / verifier B1 regression ───────────────────────────────
def test_all_concepts_walks_every_top_level_namespace(tmp_path):
    """``all_concepts_for_family`` must walk every top-level namespace,
    not just the legacy ``realism`` + ``nsfw`` pair.

    Pre-Phase-0 the iteration was hardcoded ``for top in ("realism",
    "nsfw"):`` — which meant any new top-level namespace was INVISIBLE
    to ``llm_vocabulary_block`` (and therefore to the LLM). This test
    proves a custom YAML with a brand-new top-level namespace
    (``environment``) surfaces its concept tags in the menu.

    This is the foundation patch the creative-uplift plan
    (Phase 1+) depends on — adding env / aesthetic / narrative /
    composition namespaces requires that walking those tops Just Works.
    """
    yaml_path = tmp_path / "v6_skel.yaml"
    yaml_path.write_text(
        """
version: 6

realism:
  lighting:
    LIGHT_TEST:
      sdxl: "test lighting sdxl"
      flux: "test lighting flux"

environment:
  setting:
    ENV_TEST_BEDROOM:
      sdxl: "test bedroom sdxl"
      flux: "test bedroom flux"

aesthetic:
  color_palette:
    PALETTE_TEST_NOIR:
      sdxl: "test noir sdxl"
      flux: "test noir flux"

narrative:
  moment:
    NARR_TEST_READING:
      sdxl: "test reading sdxl"
      flux: "test reading flux"

composition:
  principle:
    COMP_TEST_SYMMETRY:
      sdxl: "test symmetry sdxl"
      flux: "test symmetry flux"
""".lstrip()
    )
    custom_loader = VocabularyLoader(yaml_path)
    sdxl_block = llm_vocabulary_block("sdxl", loader=custom_loader)
    flux_block = llm_vocabulary_block("flux", loader=custom_loader)

    # Every new top-level namespace surfaces a sub-namespace line
    # with its tag.
    for block in (sdxl_block, flux_block):
        assert "realism.lighting" in block
        assert "LIGHT_TEST" in block
        assert "environment.setting" in block
        assert "ENV_TEST_BEDROOM" in block
        assert "aesthetic.color_palette" in block
        assert "PALETTE_TEST_NOIR" in block
        assert "narrative.moment" in block
        assert "NARR_TEST_READING" in block
        assert "composition.principle" in block
        assert "COMP_TEST_SYMMETRY" in block


def test_all_concepts_skips_version_top_level_key(tmp_path):
    """The ``version:`` top-level key is a stamp, not a namespace —
    iterating it would produce a ``version.X`` entry in the menu."""
    yaml_path = tmp_path / "version_only.yaml"
    yaml_path.write_text(
        """
version: 6

realism:
  lighting:
    LIGHT_ONLY:
      sdxl: "only lighting"
""".lstrip()
    )
    loader = VocabularyLoader(yaml_path)
    block = llm_vocabulary_block("sdxl", loader=loader)
    assert "version" not in block.lower().split("\n")[0:5][1] if "\n" in block else True
    # More direct: the by_ns dict shouldn't have a ``version.*`` entry
    by_ns = loader.all_concepts_for_family("sdxl")
    assert all(not k.startswith("version") for k in by_ns)


# ── vocab_version 2: broadened realism + T4 act vocabulary ─────────


def test_vocab_version_at_least_2_post_4_bis():
    """Phase 4-bis bumped vocabulary version to 2; subsequent edits
    (vocab_version 3 — explicit anatomy expansion for T4 nudity) keep
    bumping. The version monotonically increases; test pins the
    invariant that we're past Phase 4-bis."""
    assert VocabularyLoader().version >= 2


@pytest.mark.parametrize("concept", [
    "LIGHT_SPLIT", "LIGHT_CANDLELIGHT", "LIGHT_BLUE_HOUR",
])
@pytest.mark.parametrize("family", [
    "sdxl", "pony", "illustrious", "flux", "chroma", "flux2",
])
def test_new_lighting_concepts_canonicalize(loader, concept, family):
    """Phase 4-bis additions for lighting — split, candlelight, blue
    hour — must canonicalise for every family."""
    out = loader.canonicalize(concept, family)
    assert out is not None
    assert len(out) > 5  # non-empty phrase


@pytest.mark.parametrize("concept", [
    "CAMERA_PHASE_ONE_IQ4", "CAMERA_PENTAX_67_FILM",
])
@pytest.mark.parametrize("family", [
    "sdxl", "illustrious", "flux", "chroma", "flux2",
])
def test_new_camera_concepts_canonicalize_for_camera_families(
    loader, concept, family,
):
    """Pony omits cameras (booru tagging carries them implicitly)."""
    out = loader.canonicalize(concept, family)
    assert out is not None


def test_new_camera_concepts_dropped_for_pony(loader):
    assert loader.canonicalize("CAMERA_PHASE_ONE_IQ4", "pony") is None
    assert loader.canonicalize("CAMERA_PENTAX_67_FILM", "pony") is None


@pytest.mark.parametrize("concept", [
    "LENS_100MM_MACRO", "LENS_70_200_F28",
])
@pytest.mark.parametrize("family", [
    "sdxl", "illustrious", "flux", "chroma", "flux2",
])
def test_new_lens_concepts_canonicalize(loader, concept, family):
    assert loader.canonicalize(concept, family) is not None


@pytest.mark.parametrize("concept", [
    "FILM_VELVIA_50", "FILM_KODAK_VISION3",
])
@pytest.mark.parametrize("family", [
    "sdxl", "illustrious", "flux", "chroma", "flux2",
])
def test_new_film_stock_concepts_canonicalize(loader, concept, family):
    assert loader.canonicalize(concept, family) is not None


@pytest.mark.parametrize("concept", [
    "ART_HELMUT_NEWTON", "ART_HERB_RITTS_BW", "ART_IRVING_PENN_MINIMALISM",
])
@pytest.mark.parametrize("family", [
    "sdxl", "illustrious", "flux", "chroma", "flux2",
])
def test_new_art_style_concepts_canonicalize(loader, concept, family):
    assert loader.canonicalize(concept, family) is not None


@pytest.mark.parametrize("concept", [
    "MOOD_SENSUAL", "MOOD_SERENE", "MOOD_MELANCHOLIC",
])
@pytest.mark.parametrize("family", [
    "sdxl", "pony", "illustrious", "flux", "chroma", "flux2",
])
def test_new_mood_concepts_canonicalize_for_every_family(
    loader, concept, family,
):
    """Mood concepts apply to every family (incl. Pony)."""
    assert loader.canonicalize(concept, family) is not None


# ── T4 explicit-act vocabulary (Phase 4-bis + 2026-05-17 solo mode) ───
#
# Pre-2026-05-17: all 5 T4 act tags (NSFW_T4_EMBRACE_NUDE,
# NSFW_T4_KISS_PASSIONATE, NSFW_T4_SOLO_TOUCH,
# NSFW_T4_PARTNERED_INTIMATE, NSFW_T4_AFTERGLOW) were active at
# T4_explicit. The 2026-05-17 single-female-mode change moves the 4
# partnered tags to _SOLO_MODE_BANNED_TAGS — they get dropped with
# ERROR even at T4_explicit. Only NSFW_T4_SOLO_TOUCH remains active.
#
# Future multi-subject mode (deferred per CLAUDE.md) lifts the ban by
# emptying _SOLO_MODE_BANNED_TAGS; until then the partnered tags stay
# hidden from the LLM menu AND silently filtered at canonicalize.


def test_t4_solo_touch_present_at_t4(loader):
    """``NSFW_T4_SOLO_TOUCH`` is the only T4 act tag active under
    solo mode. Canonicalises at T4_explicit for the SDXL family."""
    out = loader.canonicalize(
        "NSFW_T4_SOLO_TOUCH", "sdxl", content_level="T4_explicit",
    )
    assert out is not None
    assert "solo" in out.lower() or "self-touch" in out.lower()


@pytest.mark.parametrize("concept", [
    "NSFW_T4_EMBRACE_NUDE",
    "NSFW_T4_KISS_PASSIONATE",
    "NSFW_T4_PARTNERED_INTIMATE",
    "NSFW_T4_AFTERGLOW",
])
def test_t4_partnered_act_concepts_banned_under_solo_mode(loader, concept):
    """The 4 partnered T4 act concepts are in _SOLO_MODE_BANNED_TAGS
    and must drop with ERROR at every tier — including T4_explicit
    where they used to canonicalise pre-2026-05-17."""
    for tier in ("T1_suggestive", "T2_implied", "T3_artnude", "T4_explicit"):
        out = loader.canonicalize(concept, "sdxl", content_level=tier)
        assert out is None, (
            f"{concept} leaked at {tier} (solo-mode banned)"
        )


@pytest.mark.parametrize("tier", [
    "T1_suggestive", "T2_implied", "T3_artnude",
])
def test_t4_solo_touch_dropped_below_tier(loader, tier):
    """NSFW_T4_SOLO_TOUCH still has tier_min=T4_explicit, so it drops
    at T1/T2/T3 just like the original Phase 4-bis behaviour."""
    out = loader.canonicalize("NSFW_T4_SOLO_TOUCH", "sdxl", content_level=tier)
    assert out is None


@pytest.mark.parametrize("family", [
    "sdxl", "pony", "illustrious", "flux", "chroma", "flux2",
])
def test_t4_solo_touch_canonicalizes_for_every_family(loader, family):
    """Every family has phrasing for NSFW_T4_SOLO_TOUCH — the
    sole-allowed solo-mode T4 act tag must work across all families."""
    out = loader.canonicalize(
        "NSFW_T4_SOLO_TOUCH", family,
        content_level="T4_explicit",
    )
    assert out is not None


def test_t4_act_tier_min_correctly_reported(loader):
    """tier_min_for() reports the YAML-declared T4_explicit gate
    regardless of solo-mode ban (the ban is a separate filter)."""
    assert loader.tier_min_for("NSFW_T4_SOLO_TOUCH") == "T4_explicit"
    # Banned tags still have their YAML tier_min queryable —
    # the YAML schema is preserved so future multi-subject mode lifts
    # the ban cleanly without YAML edits.
    assert loader.tier_min_for("NSFW_T4_EMBRACE_NUDE") == "T4_explicit"


def test_partnered_act_concepts_hidden_from_llm_menu(loader):
    """``all_concepts_for_family`` / ``llm_vocabulary_block`` MUST NOT
    include the solo-mode-banned partnered tags. Venice / Cydonia
    should never see them as selectable options."""
    block = llm_vocabulary_block("flux", loader=loader)
    assert "nsfw.act" in block
    assert "NSFW_T4_SOLO_TOUCH" in block
    for banned in (
        "NSFW_T4_PARTNERED_INTIMATE", "NSFW_T4_EMBRACE_NUDE",
        "NSFW_T4_KISS_PASSIONATE", "NSFW_T4_AFTERGLOW",
    ):
        assert banned not in block, (
            f"Banned partnered tag {banned!r} leaked into LLM menu"
        )


# ── canonicalize_facet under solo-mode ────────────────────────────────


def test_canonicalize_facet_translates_solo_touch_at_t4(loader):
    """A facet with NSFW_T4_SOLO_TOUCH + T4 must canonicalise (the
    sole-allowed solo-mode T4 act tag)."""
    facet = {
        "lighting_directive": "LIGHT_REMBRANDT",
        "nsfw_anatomy":       "NSFW_BREAST_NATURAL",
        "nsfw_act":           "NSFW_T4_SOLO_TOUCH",
    }
    out = canonicalize_facet(
        facet, "flux", content_level="T4_explicit", loader=loader,
    )
    # All 3 concepts canonicalised
    assert len(out) == 3
    # Specifically the solo-touch phrase landed
    assert any(
        "solo" in p.lower() or "self-touch" in p.lower() for p in out
    )


def test_canonicalize_facet_drops_partnered_act_even_at_t4(loader):
    """Partnered nsfw_act tags drop even at T4_explicit due to
    solo-mode ban — ERROR-logged, facet's structured field becomes
    null in effect."""
    facet = {
        "lighting_directive": "LIGHT_REMBRANDT",
        "nsfw_act":           "NSFW_T4_PARTNERED_INTIMATE",
    }
    out = canonicalize_facet(
        facet, "flux", content_level="T4_explicit", loader=loader,
    )
    # Only lighting_directive should remain; partnered tag dropped.
    assert len(out) == 1
    assert "partnered" not in " ".join(out).lower()


@pytest.mark.parametrize("family", [
    "sdxl", "pony", "illustrious", "flux", "chroma", "flux2",
])
def test_canonicalize_facet_solo_touch_works_for_every_family(loader, family):
    """NSFW_T4_SOLO_TOUCH canonicalises for every family at T4."""
    facet = {"nsfw_act": "NSFW_T4_SOLO_TOUCH"}
    out = canonicalize_facet(
        facet, family, content_level="T4_explicit", loader=loader,
    )
    assert len(out) == 1, f"family={family} dropped solo_touch unexpectedly"


# ── vocab_version 3: T4 explicit anatomy expansion ────────────────


@pytest.mark.parametrize("concept", [
    "NSFW_NIPPLES_VISIBLE",
    "NSFW_VULVA_VISIBLE",
    "NSFW_FULL_FRONTAL",
    "NSFW_FULL_NUDE",
    "NSFW_BARE_CHEST",
])
@pytest.mark.parametrize("family", [
    "sdxl", "pony", "illustrious", "flux", "chroma", "flux2",
])
def test_t4_explicit_anatomy_canonicalizes_for_every_family(
    loader, concept, family,
):
    """The 5 new explicit-anatomy concepts canonicalise for every
    family at T4_explicit content_level."""
    out = loader.canonicalize(
        concept, family, content_level="T4_explicit",
    )
    assert out is not None, f"{family}/{concept} produced no phrase"
    assert len(out) > 5


@pytest.mark.parametrize("concept", [
    "NSFW_NIPPLES_VISIBLE",
    "NSFW_VULVA_VISIBLE",
    "NSFW_FULL_FRONTAL",
    "NSFW_FULL_NUDE",
    "NSFW_BARE_CHEST",
])
@pytest.mark.parametrize("tier", [
    "T1_suggestive", "T2_implied", "T3_artnude",
])
def test_t4_explicit_anatomy_dropped_below_t4(loader, concept, tier):
    """Below T4 the explicit-anatomy concepts MUST be dropped — the
    T3 NSFW_BREAST_NATURAL etc. are the T3 vocabulary; these are T4
    only."""
    out = loader.canonicalize(concept, "sdxl", content_level=tier)
    assert out is None, f"{concept} leaked at {tier} (must be T4-only)"


def test_full_frontal_phrase_mentions_anatomy_directly(loader):
    """Sanity — NSFW_FULL_FRONTAL doesn't euphemise. The phrase must
    explicitly reference visible anatomy (breasts, vulva, etc.) so
    the SD encoder gets clear signal."""
    for family in ("sdxl", "flux", "chroma", "flux2"):
        out = loader.canonicalize(
            "NSFW_FULL_FRONTAL", family, content_level="T4_explicit",
        )
        # Each prose family's phrase should mention breasts AND vulva
        # (the anatomy the user explicitly asked us to surface).
        lower = out.lower()
        assert "breast" in lower or "breasts" in lower
        assert "vulva" in lower


def test_art_nude_photography_anchor_present(loader):
    """ART_NUDE_PHOTOGRAPHY is the aesthetic anchor for explicit T4
    work — the user asked for "artistic nude photography" by name."""
    for family in ("sdxl", "illustrious", "flux", "chroma", "flux2"):
        out = loader.canonicalize("ART_NUDE_PHOTOGRAPHY", family)
        assert out is not None
        assert "nude" in out.lower()


def test_t4_directive_pushes_for_full_frontal_default():
    """The rewritten T4 llm_directive in categories.yaml says
    front-facing full nudity is the DEFAULT. Pre-fix, the directive
    said "as composition demands" which let the LLM self-censor."""
    from src.memory.categories_loader import CategoriesLoader
    rules = CategoriesLoader().content_level_rules("T4_explicit")
    directive = rules.llm_directive.lower()
    # Load-bearing phrases — if any of these go missing, the LLM
    # loses the strong directive and reverts to tasteful prose.
    assert "fully nude" in directive
    assert "front-facing" in directive
    assert "vulva" in directive
    assert "nipple" in directive
    assert "default" in directive
    # The directive must explicitly tell the LLM NOT to euphemise.
    assert "do not euphemise" in directive or "do not default to" in directive


# ── Phase C: tier-aware llm_vocabulary_block ───────────────────────


def test_vocab_block_no_content_level_uses_legacy_line(loader):
    """Back-compat: callers that don't pass content_level get the
    pre-Phase-C generic instructional line."""
    block = llm_vocabulary_block("sdxl", loader=loader)
    assert "Pick at most one tag per namespace" in block


def test_vocab_block_t1_forbids_nsfw(loader):
    """At T1_suggestive, the block tells the LLM NOT to pick nsfw_*."""
    block = llm_vocabulary_block(
        "sdxl", content_level="T1_suggestive", loader=loader,
    )
    assert "Do NOT pick any nsfw_*" in block
    # And NOT the generic line
    assert "Pick at most one tag per namespace. Setting any to null" not in block


def test_vocab_block_t2_forbids_nsfw(loader):
    block = llm_vocabulary_block(
        "sdxl", content_level="T2_implied", loader=loader,
    )
    assert "Do NOT pick any nsfw_*" in block


def test_vocab_block_t3_requires_nsfw_anatomy(loader):
    """At T3_artnude, the block REQUIRES nsfw_anatomy and forbids
    nsfw_act."""
    block = llm_vocabulary_block(
        "sdxl", content_level="T3_artnude", loader=loader,
    )
    assert "REQUIRED: pick exactly one `nsfw_anatomy` tag" in block
    assert "Do NOT pick `nsfw_act` tags (T4-only)" in block


def test_vocab_block_t4_requires_both_anatomy_and_act(loader):
    """At T4_explicit, the block REQUIRES both nsfw_anatomy AND
    nsfw_act tags — the load-bearing piece that re-enables T4 NSFW."""
    block = llm_vocabulary_block(
        "sdxl", content_level="T4_explicit", loader=loader,
    )
    assert "REQUIRED:" in block
    assert "nsfw_anatomy" in block
    assert "nsfw_act" in block
    # T3-style "DO NOT pick nsfw_act" must NOT appear at T4
    assert "Do NOT pick `nsfw_act`" not in block


@pytest.mark.parametrize("tier", [
    "T1_suggestive", "T2_implied", "T3_artnude", "T4_explicit",
])
def test_vocab_block_tier_keyword_in_block(loader, tier):
    """Each tier-aware block produces non-empty output that lists
    the realism namespaces (sanity)."""
    block = llm_vocabulary_block(
        "sdxl", content_level=tier, loader=loader,
    )
    assert "realism.lighting" in block
    assert "REALISM VOCABULARY" in block
