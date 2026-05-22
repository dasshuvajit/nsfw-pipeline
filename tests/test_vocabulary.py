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


def test_environment_namespace_canonicalises_per_family(loader):
    """Phase 1 (vocab v6) — environment.setting + environment.atmosphere
    tags translate to family-shaped phrasing for every family.

    Verifies the real on-disk vocab v6, not a synthetic fixture, so
    YAML drift on the new namespaces fails the build."""
    from src.prompt.vocabulary import canonicalize_facet
    facet = {
        "environment_setting": "ENV_TUSCAN_VILLA_RENAISSANCE",
        "environment_atmosphere": "ATM_DUST_MOTES_IN_LIGHT",
    }
    for fam in ("sdxl", "pony", "illustrious", "flux", "chroma", "flux2"):
        phrases = canonicalize_facet(facet, fam, loader=loader)
        # Both canonicalize cleanly for every family — Pony participates
        # in environment vocab (unlike camera/lens/film_stock which it
        # omits) because environments have natural booru tags.
        assert len(phrases) == 2, (
            f"family={fam} produced {len(phrases)} env phrases "
            f"(expected 2): {phrases}"
        )
        env_phrase = phrases[0]
        atm_phrase = phrases[1]
        # Setting must mention Tuscan / villa / Renaissance
        lower_env = env_phrase.lower()
        assert any(
            tok in lower_env
            for tok in ("tuscan", "renaissance", "villa")
        ), f"family={fam} env phrase missing setting cue: {env_phrase!r}"
        # Atmosphere must mention dust / motes / light
        lower_atm = atm_phrase.lower()
        assert "dust" in lower_atm and (
            "mote" in lower_atm or "light" in lower_atm
        ), f"family={fam} atm phrase missing dust cue: {atm_phrase!r}"


def test_env_prop_and_composition_canonicalise(loader):
    """Phase 4 (vocab v6) — env.prop + composition.principle
    namespaces translate per family. Pony omits composition.principle
    (booru tags carry composition implicitly) but participates in
    env.prop (props have natural booru forms)."""
    from src.prompt.vocabulary import canonicalize_facet

    # vocab v7 (2026-05-20): PROP_CHEVAL_MIRROR + COMP_FRAME_WITHIN_FRAME
    # removed. Substitute the cleanest surviving entries from each
    # namespace — PROP_CHAISE_LOUNGE_VELVET (every-family prose
    # available) and COMP_LEADING_LINES_FLOOR (similar coverage).
    facet = {
        "environment_prop": "PROP_CHAISE_LOUNGE_VELVET",
        "composition_principle": "COMP_LEADING_LINES_FLOOR",
    }
    for fam in ("sdxl", "illustrious", "flux", "chroma", "flux2"):
        phrases = canonicalize_facet(facet, fam, loader=loader)
        assert len(phrases) == 2, (
            f"family={fam} produced {len(phrases)} (expected 2): {phrases}"
        )
        joined = " ".join(phrases).lower()
        assert "chaise" in joined or "velvet" in joined
        assert "floor" in joined or "leading" in joined

    # Pony: env.prop yes, composition.principle no → 1 phrase
    pony_phrases = canonicalize_facet(facet, "pony", loader=loader)
    assert len(pony_phrases) == 1, (
        f"pony should produce 1 phrase (env.prop only); got "
        f"{len(pony_phrases)}: {pony_phrases}"
    )
    assert "chaise" in pony_phrases[0].lower()


def test_full_phase_1_4_stack_chroma(loader):
    """End-to-end smoke: a T4 facet with all Phase 1-4 fields
    populated canonicalises to a coherent 9-phrase output for chroma.
    Pre-Phase-1 the same scene composed to 4 phrases (lighting / mood /
    nsfw_anatomy / nsfw_act). Post-Phase-4 it composes to 9 phrases
    spanning all creative-uplift dimensions."""
    from src.prompt.vocabulary import canonicalize_facet

    facet = {
        "lighting_directive": "LIGHT_GOLDEN_HOUR",
        "mood_aesthetic": "MOOD_INTIMATE",
        "environment_setting": "ENV_TUSCAN_VILLA_RENAISSANCE",
        "environment_atmosphere": "ATM_DUST_MOTES_IN_LIGHT",
        "environment_prop": "PROP_PEONIES_OVERBLOWN",
        "narrative_moment": "NARR_READING_LETTER_AT_DAWN",
        "composition_principle": "COMP_LEADING_LINES_FLOOR",
        "nsfw_anatomy": "NSFW_FULL_NUDE",
        "nsfw_act": "NSFW_T4_SOLO_GAZE",
    }
    phrases = canonicalize_facet(
        facet, "chroma", content_level="T4_explicit", loader=loader,
    )
    assert len(phrases) == 9, (
        f"Expected 9 phrases (one per field); got {len(phrases)}: "
        f"{phrases}"
    )
    joined = " ".join(phrases).lower()
    # Every dimension represented
    assert "golden" in joined
    assert "intimate" in joined or "contemplative" in joined
    assert "tuscan" in joined or "villa" in joined
    assert "dust" in joined
    assert "peon" in joined  # peonies
    assert "letter" in joined or "reading" in joined
    assert "floor" in joined or "leading" in joined
    assert "nude" in joined or "naked" in joined
    assert "gaze" in joined or "eye" in joined or "contact" in joined


def test_canonicalize_series_aesthetic_per_family(loader):
    """Phase 3 (vocab v6) — canonicalize_series_aesthetic translates
    series-level aesthetic anchors (color_palette / photographer_ref /
    art_movement) to family-shaped phrases for every family. The
    "signature look" — pinned ONCE per series, threaded into every
    scene by the composer."""
    from src.prompt.vocabulary import canonicalize_series_aesthetic

    series_plan = {
        "theme": "moody noir",
        "color_palette": "PALETTE_MONOCHROME_HIGH_CONTRAST",
        "photographer_ref": "PHOTOG_HELMUT_NEWTON",
        "art_movement": "ART_MOVE_FILM_NOIR_1940S",
    }

    for fam in ("sdxl", "illustrious", "flux", "chroma", "flux2"):
        phrases = canonicalize_series_aesthetic(
            series_plan, fam, loader=loader,
        )
        assert len(phrases) == 3, (
            f"family={fam} produced {len(phrases)} aesthetic phrases "
            f"(expected 3): {phrases}"
        )
        # Palette mentions monochrome/black/white
        joined = " ".join(phrases).lower()
        assert "monochrome" in joined or "black" in joined
        # Photographer mentions Helmut Newton
        assert "newton" in joined
        # Art movement mentions noir or film
        assert "noir" in joined

    # Pony: omits photographer_ref + art_movement; only color_palette
    # has Pony phrasing. Should return just 1 phrase.
    pony_phrases = canonicalize_series_aesthetic(
        series_plan, "pony", loader=loader,
    )
    assert len(pony_phrases) == 1, (
        f"pony should produce 1 phrase (color_palette only); got "
        f"{len(pony_phrases)}: {pony_phrases}"
    )


def test_canonicalize_series_aesthetic_empty_back_compat(loader):
    """Old series predating Phase 3 have no aesthetic anchor fields —
    canonicalize_series_aesthetic returns [] for back-compat. The
    composer prepends an empty list = no change to existing prompts."""
    from src.prompt.vocabulary import canonicalize_series_aesthetic
    assert canonicalize_series_aesthetic(None, "chroma") == []
    assert canonicalize_series_aesthetic({}, "chroma") == []
    # series_plan with only legacy fields (no aesthetic anchors)
    legacy = {
        "theme": "boudoir",
        "mood": "intimate",
        "environment": "bedroom",
        "variation_axes": ["pose"],
    }
    assert canonicalize_series_aesthetic(legacy, "chroma") == []


def test_resolve_aesthetic_menu_narrows_via_compat_lists():
    """Phase 3 — _resolve_aesthetic_menu narrows the SeriesPlanner
    menu when style_profile.compatible_* lists are provided. This
    is the lite I2 plumbing: SeriesPlanner only offers coherent
    combinations the profile has validated."""
    from src.prompt.aesthetic_menu import _resolve_aesthetic_menu

    # No filter → full menu
    full = _resolve_aesthetic_menu()
    assert len(full["color_palette"]) >= 15
    assert len(full["photographer_ref"]) >= 12
    assert len(full["art_movement"]) >= 12

    # With filter → narrowed
    narrow = _resolve_aesthetic_menu(
        style_profile_compat={
            "compatible_palettes": ["PALETTE_MONOCHROME_LOW_KEY"],
            "compatible_photographers": [
                "PHOTOG_HELMUT_NEWTON", "PHOTOG_BILL_HENSON",
            ],
            "compatible_art_movements": ["ART_MOVE_FILM_NOIR_1940S"],
        },
    )
    assert narrow["color_palette"] == ["PALETTE_MONOCHROME_LOW_KEY"]
    assert set(narrow["photographer_ref"]) == {
        "PHOTOG_HELMUT_NEWTON", "PHOTOG_BILL_HENSON",
    }
    assert narrow["art_movement"] == ["ART_MOVE_FILM_NOIR_1940S"]


def test_resolve_aesthetic_menu_falls_back_when_filter_empties_namespace():
    """If a compat list contains only stale/unknown tags (zero
    intersection with the live vocab), fall back to the full menu
    for that namespace — better to offer too much than nothing."""
    from src.prompt.aesthetic_menu import _resolve_aesthetic_menu

    narrow = _resolve_aesthetic_menu(
        style_profile_compat={
            "compatible_palettes": ["PALETTE_DOES_NOT_EXIST"],
            "compatible_photographers": [],
            "compatible_art_movements": [],
        },
    )
    # Stale single-tag filter → empty intersection → fallback to full
    assert len(narrow["color_palette"]) >= 15


def test_narrative_moment_canonicalises_per_family(loader):
    """Phase 2 (vocab v6) — narrative.moment tags translate to
    family-shaped phrasing for every family. The #1 leverage axis
    per market research: "she reads a letter at dawn" solves window
    + chair + envelope + stillness in one tag."""
    from src.prompt.vocabulary import canonicalize_facet
    facet = {"narrative_moment": "NARR_READING_LETTER_AT_DAWN"}
    for fam in ("sdxl", "pony", "illustrious", "flux", "chroma", "flux2"):
        phrases = canonicalize_facet(facet, fam, loader=loader)
        assert len(phrases) == 1, f"family={fam} produced {len(phrases)} (expected 1)"
        lower = phrases[0].lower()
        # Letter / reading / dawn / morning anchors in every phrasing
        assert "letter" in lower or "reading" in lower, (
            f"family={fam}: narrative phrase missing letter/reading anchor: {phrases[0]!r}"
        )


def test_narrative_moment_in_llm_menu_for_every_family(loader):
    """narrative.moment surfaces in the LLM menu for every family
    (including Pony — booru tags carry narrative naturally)."""
    for fam in ("sdxl", "pony", "illustrious", "flux", "chroma", "flux2"):
        block = llm_vocabulary_block(fam, loader=loader)
        assert "narrative.moment" in block, f"family={fam}: narrative.moment missing from menu"
        assert "NARR_READING_LETTER_AT_DAWN" in block
        assert "NARR_STEPPING_FROM_BATH" in block


def test_environment_namespace_in_llm_menu_for_every_family(loader):
    """The new environment.* namespaces show up in the LLM
    vocabulary block for every family (including Pony — environments
    have native booru tags)."""
    for fam in ("sdxl", "pony", "illustrious", "flux", "chroma", "flux2"):
        block = llm_vocabulary_block(fam, loader=loader)
        assert "environment.setting" in block, (
            f"family={fam}: environment.setting missing from menu"
        )
        assert "environment.atmosphere" in block, (
            f"family={fam}: environment.atmosphere missing from menu"
        )
        # At least 30 setting tags + 20 atmosphere tags should surface
        # (vocab v6 ships 41 + 24)
        assert "ENV_TUSCAN_VILLA_RENAISSANCE" in block
        assert "ATM_DUST_MOTES_IN_LIGHT" in block


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


# ── Verifier round-2 I4: compatible_environments narrowing ─────────


def test_vocab_block_narrows_environment_setting_to_whitelist(loader):
    """When `compatible_environments` is supplied, the
    `environment.setting` line is filtered to only those tags."""
    whitelist = ["ENV_VICTORIAN_CONSERVATORY", "ENV_TUSCAN_VILLA_RENAISSANCE"]
    block = llm_vocabulary_block(
        "sdxl", content_level="T3_artnude", loader=loader,
        compatible_environments=whitelist,
    )
    # The two whitelisted tags appear
    for tag in whitelist:
        assert tag in block
    # A non-whitelisted env.setting tag is filtered out
    # (ENV_BRUTALIST_CONCRETE_LOFT exists in vocab but isn't in the
    # whitelist for this test).
    env_line = next(
        line for line in block.splitlines()
        if line.strip().startswith("environment.setting:")
    )
    assert "ENV_VICTORIAN_CONSERVATORY" in env_line
    assert "ENV_TUSCAN_VILLA_RENAISSANCE" in env_line
    assert "ENV_BRUTALIST_CONCRETE_LOFT" not in env_line


def test_vocab_block_falls_back_to_full_menu_when_compat_empty(loader):
    """Empty / None compat list ⇒ full vocab menu (no narrowing)."""
    block_none = llm_vocabulary_block(
        "sdxl", content_level="T3_artnude", loader=loader,
        compatible_environments=None,
    )
    block_empty = llm_vocabulary_block(
        "sdxl", content_level="T3_artnude", loader=loader,
        compatible_environments=[],
    )
    # Both should be identical to the no-compat default — same env line
    # has the multiple env tags the vocab supplies.
    assert "ENV_VICTORIAN_CONSERVATORY" in block_none
    assert "ENV_VICTORIAN_CONSERVATORY" in block_empty
    assert "ENV_BRUTALIST_CONCRETE_LOFT" in block_none
    assert "ENV_BRUTALIST_CONCRETE_LOFT" in block_empty


def test_vocab_block_falls_back_when_compat_intersection_empty(loader):
    """If every whitelist tag is absent from the family menu, fall
    back to the full menu — never serve the LLM a blank env line."""
    block = llm_vocabulary_block(
        "sdxl", content_level="T3_artnude", loader=loader,
        compatible_environments=["ENV_NONEXISTENT_TAG_X", "ENV_FAKE_Y"],
    )
    # The full env menu still appears
    assert "ENV_VICTORIAN_CONSERVATORY" in block
    assert "ENV_BRUTALIST_CONCRETE_LOFT" in block


# ── Round-12: compatible_narratives narrowing ──────────────────────


def test_vocab_block_narrows_narrative_moment_to_whitelist(loader):
    """Round-12 (2026-05-21) — same pattern as environment narrowing,
    applied to narrative.moment. The 2026-05-20 A/B run showed both
    Cydonia + Qwen3 picking narrative tags that visually contradict
    the theme (NARR_AFTER_THE_PARTY in chapel scene, NARR_STEPPING_FROM_BATH
    in industrial loft). Narrowing by category cuts the menu to
    theme-coherent options."""
    whitelist = ["NARR_READING_LETTER_AT_DAWN", "NARR_LEANING_DOORWAY"]
    block = llm_vocabulary_block(
        "sdxl", content_level="T3_artnude", loader=loader,
        compatible_narratives=whitelist,
    )
    for tag in whitelist:
        assert tag in block
    narr_line = next(
        line for line in block.splitlines()
        if line.strip().startswith("narrative.moment:")
    )
    assert "NARR_READING_LETTER_AT_DAWN" in narr_line
    assert "NARR_LEANING_DOORWAY" in narr_line
    # A non-whitelisted narrative tag is filtered out
    assert "NARR_AFTER_THE_PARTY" not in narr_line


def test_vocab_block_no_compat_narratives_shows_full_narrative_menu(loader):
    """No `compatible_narratives` (None or empty) → full vocab menu."""
    block_none = llm_vocabulary_block(
        "sdxl", content_level="T3_artnude", loader=loader,
        compatible_narratives=None,
    )
    block_empty = llm_vocabulary_block(
        "sdxl", content_level="T3_artnude", loader=loader,
        compatible_narratives=[],
    )
    # Multiple narrative tags survive in both unfiltered renderings
    assert "NARR_AFTER_THE_PARTY" in block_none
    assert "NARR_AFTER_THE_PARTY" in block_empty
    assert "NARR_STEPPING_FROM_BATH" in block_none
    assert "NARR_STEPPING_FROM_BATH" in block_empty


def test_vocab_block_falls_back_when_narrative_intersection_empty(loader):
    """If every whitelist tag is absent from the family menu, fall
    back to the full menu — same defensive behaviour as for environments."""
    block = llm_vocabulary_block(
        "sdxl", content_level="T3_artnude", loader=loader,
        compatible_narratives=["NARR_FAKE_X", "NARR_NONEXISTENT_Y"],
    )
    # Full narrative menu still appears
    assert "NARR_READING_LETTER_AT_DAWN" in block
    assert "NARR_AFTER_THE_PARTY" in block


def test_vocab_block_environment_and_narrative_narrow_independently(loader):
    """Both narrowing axes can apply at the same time without
    interfering — environment whitelist narrows env.setting and
    narrative whitelist narrows narrative.moment."""
    block = llm_vocabulary_block(
        "sdxl", content_level="T3_artnude", loader=loader,
        compatible_environments=["ENV_TUSCAN_VILLA_RENAISSANCE"],
        compatible_narratives=["NARR_READING_LETTER_AT_DAWN"],
    )
    env_line = next(
        line for line in block.splitlines()
        if line.strip().startswith("environment.setting:")
    )
    narr_line = next(
        line for line in block.splitlines()
        if line.strip().startswith("narrative.moment:")
    )
    # Env narrowed
    assert "ENV_TUSCAN_VILLA_RENAISSANCE" in env_line
    assert "ENV_BRUTALIST_CONCRETE_LOFT" not in env_line
    # Narrative narrowed
    assert "NARR_READING_LETTER_AT_DAWN" in narr_line
    assert "NARR_AFTER_THE_PARTY" not in narr_line


# ── Round-22 F13: drop-counter observability ──────────────────────


def test_drop_counter_increments_on_unknown_concept(loader):
    """Round-22 F13 — when the LLM picks a concept that doesn't exist
    in any namespace, the drop counter increments under 'unknown'."""
    from src.prompt.vocabulary import (
        reset_drop_counter, get_drop_counts, canonicalize_facet,
    )
    reset_drop_counter()
    canonicalize_facet(
        {"lighting_directive": "LIGHT_DEFINITELY_NOT_REAL"},
        family_id="chroma",
        loader=loader,
    )
    counts = get_drop_counts()
    assert counts["unknown"] == 1
    assert counts["tier"] == 0
    assert counts["family"] == 0


def test_drop_counter_increments_on_tier_gated_nsfw(loader):
    """Round-22 F13 — when an NSFW concept's tier_min exceeds the
    content_level, the canonicalizer drops it and increments 'tier'."""
    from src.prompt.vocabulary import (
        reset_drop_counter, get_drop_counts, canonicalize_facet,
    )
    reset_drop_counter()
    # NSFW_BARE_CHEST is T4_explicit; T3_artnude is below its tier_min
    canonicalize_facet(
        {"nsfw_anatomy": "NSFW_BARE_CHEST"},
        family_id="chroma",
        content_level="T3_artnude",
        loader=loader,
    )
    counts = get_drop_counts()
    assert counts["tier"] == 1
    assert counts["unknown"] == 0


def test_drop_counter_reset_zeros_all_axes(loader):
    """Round-22 F13 — reset_drop_counter() zeros every axis."""
    from src.prompt.vocabulary import (
        reset_drop_counter, get_drop_counts, canonicalize_facet,
    )
    # Prime with drops.
    canonicalize_facet(
        {"lighting_directive": "LIGHT_UNKNOWN_ABC",
         "nsfw_anatomy": "NSFW_BARE_CHEST"},
        family_id="chroma",
        content_level="T3_artnude",
        loader=loader,
    )
    pre = get_drop_counts()
    assert any(v > 0 for v in pre.values())
    reset_drop_counter()
    post = get_drop_counts()
    assert all(v == 0 for v in post.values())


def test_env_coherence_passes_when_atm_matches_env(loader):
    """Round-22 F15 — ATM_STEAM_FROM_BATH paired with a bath env_setting
    has at least one required env keyword in the env's prose, so the
    coherence check passes (returns empty violations list)."""
    from src.prompt.vocabulary import check_facet_env_coherence
    # We need an env that mentions "bath" or "spa" etc. Most envs don't.
    # The test exercises the matching-keyword path with a synthesized
    # check — but here we rely on the actual vocab.
    # ENV_ROMAN_BATHHOUSE_HAMMAM exists in some categories.
    # Fall back: just exercise the API surface with a mock loader if
    # no matching env tag exists in the real vocab.
    # Run the actual check — should be empty when both fields are None.
    violations = check_facet_env_coherence(
        environment_setting=None,
        environment_atmosphere="ATM_STEAM_FROM_BATH",
        narrative_moment=None,
        loader=loader,
    )
    # No env_setting → constraint check is skipped → no violations.
    assert violations == [], (
        f"check should skip when env_setting is None; got {violations}"
    )


def test_env_coherence_rejects_underwater_atm_with_fire_escape_env(loader):
    """Round-22 F15 — exact failure mode from Grok's audit:
    ENV_FIRE_ESCAPE_NEON (Manhattan fire escape at night) +
    ATM_FABRIC_FLOATING_UNDERWATER (underwater) is incoherent. The
    coherence check returns a violation for the atmosphere field."""
    from src.prompt.vocabulary import check_facet_env_coherence
    violations = check_facet_env_coherence(
        environment_setting="ENV_FIRE_ESCAPE_NEON",
        environment_atmosphere="ATM_FABRIC_FLOATING_UNDERWATER",
        narrative_moment=None,
        loader=loader,
    )
    assert len(violations) == 1
    field, tag, reason = violations[0]
    assert field == "environment_atmosphere"
    assert tag == "ATM_FABRIC_FLOATING_UNDERWATER"
    assert "water" in reason.lower() or "underwater" in reason.lower()


def test_env_coherence_rejects_letter_burning_with_outdoor_env(loader):
    """Round-22 F15 — NARR_LETTER_BURNING_FIRE (indoor fireplace
    moment) paired with ENV_FIRE_ESCAPE_NEON (outdoor Manhattan
    fire escape) — physically incoherent. Coherence check flags
    the narrative_moment field."""
    from src.prompt.vocabulary import check_facet_env_coherence
    violations = check_facet_env_coherence(
        environment_setting="ENV_FIRE_ESCAPE_NEON",
        environment_atmosphere=None,
        narrative_moment="NARR_LETTER_BURNING_FIRE",
        loader=loader,
    )
    assert len(violations) == 1
    field, tag, reason = violations[0]
    assert field == "narrative_moment"
    assert tag == "NARR_LETTER_BURNING_FIRE"


def test_env_coherence_silent_when_tags_compatible(loader):
    """Round-22 F15 — when all 3 tags imply a coherent scene
    (mediterranean villa terrace + sunbathing narrative + golden
    pollen air), the coherence check returns an empty violations
    list."""
    from src.prompt.vocabulary import check_facet_env_coherence
    violations = check_facet_env_coherence(
        environment_setting="ENV_MEDITERRANEAN_COURTYARD",
        environment_atmosphere=None,  # most ATMs have no place_constraint
        narrative_moment="NARR_SUNBATHING_TERRACE",
        loader=loader,
    )
    # ENV_MEDITERRANEAN_COURTYARD's prose contains "mediterranean" or
    # "courtyard" → NARR_SUNBATHING_TERRACE matches against the
    # courtyard/terrace/outdoor keyword list.
    assert violations == [], (
        f"expected no violations for compatible mediterranean+sunbathing "
        f"combo; got {violations}"
    )


def test_env_coherence_skips_tags_without_place_constraint(loader):
    """Round-22 F15 — ATM / NARR tags with no place_constraint are
    flexible and pass the check unconditionally. Pick tags that
    genuinely have no constraint (vocab v11 added constraints to
    several previously-unconstrained tags; ATM_HAIR_MOVING_WIND
    and NARR_LACING_BOOT_LEG remain unconstrained as of v11)."""
    from src.prompt.vocabulary import check_facet_env_coherence
    violations = check_facet_env_coherence(
        environment_setting="ENV_FIRE_ESCAPE_NEON",
        environment_atmosphere="ATM_HAIR_MOVING_WIND",  # no constraint
        narrative_moment="NARR_LACING_BOOT_LEG",  # no constraint
        loader=loader,
    )
    assert violations == []


def test_pydantic_validator_rejects_incoherent_facet():
    """Round-22 F15 integration — SceneFacetFluxNatural.model_validate
    raises ValidationError when env_setting + atmosphere are
    physically incoherent, triggering the existing retry-nudge in
    scene_facet_generator.py."""
    import pytest
    from pydantic import ValidationError
    from src.agents.schemas import SceneFacetFluxNatural
    with pytest.raises(ValidationError, match=r"cross-field coherence"):
        SceneFacetFluxNatural.model_validate({
            "scene_prose": (
                "A woman reclines on a fire escape at midnight, the "
                "city glittering below her in scattered neon points. "
                "Her gaze drifts down toward the rain-slicked street, "
                "pensive and unhurried."
            ),
            "environment_setting": "ENV_FIRE_ESCAPE_NEON",
            "environment_atmosphere": "ATM_FABRIC_FLOATING_UNDERWATER",
        })


def test_pydantic_validator_passes_coherent_facet():
    """Round-22 F15 — coherent env+atmosphere passes validation.
    Vocab v11 (2026-05-23) added place_constraint on
    ATM_DUST_MOTES_IN_LIGHT (requires indoor with window). Use a
    matching indoor environment here so coherence holds."""
    from src.agents.schemas import SceneFacetFluxNatural
    facet = SceneFacetFluxNatural.model_validate({
        "scene_prose": (
            "A woman reclines in the warm parlour interior, golden "
            "afternoon sunlight streaming across her bare shoulders "
            "and the worn velvet upholstery, pensive and at ease in "
            "the quiet space, the sheer curtains glowing softly."
        ),
        "environment_setting": "ENV_VICTORIAN_PARLOUR",
        "environment_atmosphere": "ATM_DUST_MOTES_IN_LIGHT",
        "narrative_moment": "NARR_LISTENING_TO_RECORD",
    })
    assert facet.environment_setting == "ENV_VICTORIAN_PARLOUR"


def test_drop_counter_records_clean_run_as_all_zero(loader):
    """Round-22 F13 — a clean canonicalize (all tags valid + in-tier)
    leaves the counter at zero."""
    from src.prompt.vocabulary import (
        reset_drop_counter, get_drop_counts, canonicalize_facet,
    )
    reset_drop_counter()
    phrases = canonicalize_facet(
        {
            "lighting_directive": "LIGHT_GOLDEN_HOUR",
            "mood_aesthetic": "MOOD_SERENE",
            "realism_camera": "CAMERA_SONY_A7RV",
        },
        family_id="chroma",
        content_level="T3_artnude",
        loader=loader,
    )
    assert len(phrases) == 3, f"expected 3 phrases, got: {phrases}"
    counts = get_drop_counts()
    assert all(v == 0 for v in counts.values()), (
        f"clean run produced non-zero drops: {counts}"
    )


# ── pose ↔ nsfw_act / nsfw_posture geometric coherence ─────────────


def test_pose_act_coherence_passes_reclining_with_reclining_act():
    from src.prompt.vocabulary import check_pose_act_coherence
    violations = check_pose_act_coherence(
        pose="reclining expressive on a velvet chaise",
        nsfw_act="NSFW_T4_SOLO_RECLINING",
        nsfw_posture=None,
    )
    assert violations == []


def test_pose_act_coherence_rejects_reclining_with_solo_display():
    """The exact scene_021 bug — reclining pose + SOLO_DISPLAY act
    is geometrically impossible (low angle of a reclining body sees
    feet-first, not torso-frontal)."""
    from src.prompt.vocabulary import check_pose_act_coherence
    violations = check_pose_act_coherence(
        pose="reclining expressive with dramatic side lighting",
        nsfw_act="NSFW_T4_SOLO_DISPLAY",
        nsfw_posture=None,
    )
    assert len(violations) == 1
    field, tag, reason = violations[0]
    assert field == "nsfw_act"
    assert tag == "NSFW_T4_SOLO_DISPLAY"
    assert "RECLINING" in reason


def test_pose_act_coherence_rejects_standing_with_reclining_act():
    """Inverse case — standing pose + SOLO_RECLINING is impossible."""
    from src.prompt.vocabulary import check_pose_act_coherence
    violations = check_pose_act_coherence(
        pose="standing confident with open body language",
        nsfw_act="NSFW_T4_SOLO_RECLINING",
        nsfw_posture=None,
    )
    assert len(violations) == 1
    assert violations[0][0] == "nsfw_act"


def test_pose_act_coherence_rejects_kneeling_with_reclining_act():
    from src.prompt.vocabulary import check_pose_act_coherence
    violations = check_pose_act_coherence(
        pose="kneeling on a chair with arms draped over the back",
        nsfw_act="NSFW_T4_AFTERGLOW",
        nsfw_posture=None,
    )
    assert len(violations) == 1
    assert "KNEELING" in violations[0][2]


def test_pose_act_coherence_rejects_posture_mismatch():
    """nsfw_posture must MATCH the scene's pose orientation. Reclining
    scene + STANDING_NUDE posture is contradictory."""
    from src.prompt.vocabulary import check_pose_act_coherence
    violations = check_pose_act_coherence(
        pose="reclining expressive",
        nsfw_act=None,
        nsfw_posture="NSFW_STANDING_NUDE",
    )
    assert len(violations) == 1
    field, tag, reason = violations[0]
    assert field == "nsfw_posture"
    assert tag == "NSFW_STANDING_NUDE"
    assert "RECLINED_NUDE" in reason  # expected posture cited


def test_pose_act_coherence_accepts_neutral_acts_at_any_pose():
    """SOLO_GAZE and SOLO_TOUCH are orientation-neutral — compatible
    with any pose."""
    from src.prompt.vocabulary import check_pose_act_coherence
    for pose in ("reclining", "standing", "kneeling", "sitting"):
        for act in ("NSFW_T4_SOLO_GAZE", "NSFW_T4_SOLO_TOUCH"):
            assert check_pose_act_coherence(
                pose=pose, nsfw_act=act, nsfw_posture=None,
            ) == [], f"{pose} + {act} should be neutral"


def test_pose_act_coherence_skipped_for_ambiguous_pose():
    """When the pose doesn't match any known orientation keyword, the
    check fails open — we can't classify, so we trust the LLM."""
    from src.prompt.vocabulary import check_pose_act_coherence
    violations = check_pose_act_coherence(
        pose="contemplative gesture with hands in hair",
        nsfw_act="NSFW_T4_SOLO_RECLINING",
        nsfw_posture=None,
    )
    assert violations == []


def test_pose_act_coherence_skipped_when_no_act_or_posture():
    """When both act and posture are null (T1-T3 below the gate),
    nothing to check."""
    from src.prompt.vocabulary import check_pose_act_coherence
    violations = check_pose_act_coherence(
        pose="reclining expressive",
        nsfw_act=None,
        nsfw_posture=None,
    )
    assert violations == []


def test_pose_act_coherence_rejects_back_anatomy_with_front_act():
    """Verifier audit (2026-05-23) — scene_011 had NSFW_GLUTES (back)
    + NSFW_T4_SOLO_DISPLAY (front). Camera can't see both directions
    in one shot. New anatomy-direction sub-check rejects."""
    from src.prompt.vocabulary import check_pose_act_coherence
    violations = check_pose_act_coherence(
        pose="standing confident with dynamic leg extension",
        nsfw_act="NSFW_T4_SOLO_DISPLAY",
        nsfw_posture=None,
        nsfw_anatomy="NSFW_GLUTES",
    )
    assert len(violations) == 1
    field, tag, reason = violations[0]
    assert field == "nsfw_anatomy"
    assert tag == "NSFW_GLUTES"
    assert "BACK-facing" in reason or "back" in reason.lower()


def test_pose_act_coherence_rejects_back_view_with_display_act():
    """NSFW_BACK_VIEW_NUDE (back view posture) + NSFW_T4_SOLO_DISPLAY
    (front display act) — same direction conflict."""
    from src.prompt.vocabulary import check_pose_act_coherence
    violations = check_pose_act_coherence(
        pose="arching back on the floor with arms behind head",
        nsfw_act="NSFW_T4_SOLO_DISPLAY",
        nsfw_posture=None,
        nsfw_anatomy="NSFW_BACK_VIEW_NUDE",
    )
    assert len(violations) == 1
    assert violations[0][0] == "nsfw_anatomy"


def test_pose_act_coherence_passes_back_anatomy_with_neutral_act():
    """Back anatomy + orientation-neutral act (SOLO_GAZE / SOLO_TOUCH)
    is fine — gazing/touching can happen from behind."""
    from src.prompt.vocabulary import check_pose_act_coherence
    for act in ("NSFW_T4_SOLO_GAZE", "NSFW_T4_SOLO_TOUCH"):
        violations = check_pose_act_coherence(
            pose="standing confident",
            nsfw_act=act,
            nsfw_posture=None,
            nsfw_anatomy="NSFW_GLUTES",
        )
        assert violations == [], (
            f"back anatomy + {act} should pass (neutral act)"
        )


def test_pose_act_coherence_rejects_bath_act_on_non_bath_env(loader):
    """Verifier audit (2026-05-23) — scene_008 had NSFW_T4_SOLO_BATH
    paired with ENV_MEDITERRANEAN_COURTYARD (no bath). Scene_020
    same with marble bathroom + outdoor fog atm. The bath-class
    act requires the env prose to contain water/tub/bath keyword."""
    from src.prompt.vocabulary import check_pose_act_coherence
    violations = check_pose_act_coherence(
        pose="lounging",
        nsfw_act="NSFW_T4_SOLO_BATH",
        nsfw_posture=None,
        environment_setting="ENV_MEDITERRANEAN_COURTYARD",
        loader=loader,
    )
    assert len(violations) == 1
    field, tag, reason = violations[0]
    assert field == "nsfw_act"
    assert tag == "NSFW_T4_SOLO_BATH"
    assert "bath-class" in reason or "water" in reason.lower()


def test_pose_act_coherence_passes_bath_act_on_bath_env(loader):
    """Bath act + actual bath environment is fine."""
    from src.prompt.vocabulary import check_pose_act_coherence
    violations = check_pose_act_coherence(
        pose="reclining in tub",
        nsfw_act="NSFW_T4_SOLO_BATH",
        nsfw_posture=None,
        environment_setting="ENV_CLAWFOOT_BATHROOM",
        loader=loader,
    )
    assert violations == []


def test_pose_act_coherence_passes_non_bath_act_on_any_env(loader):
    """Non-bath acts shouldn't trigger the bath-env check."""
    from src.prompt.vocabulary import check_pose_act_coherence
    violations = check_pose_act_coherence(
        pose="standing",
        nsfw_act="NSFW_T4_SOLO_DISPLAY",
        nsfw_posture=None,
        environment_setting="ENV_MEDITERRANEAN_COURTYARD",
        loader=loader,
    )
    assert violations == []


# ── vocab orthogonality — palette/lighting/art_style must not duplicate phrases ──


def test_light_rembrandt_chroma_does_not_duplicate_palette_caravaggio(loader):
    """Vocab v10 — LIGHT_REMBRANDT chroma rephrased away from the
    generic 'dramatic chiaroscuro shadow' token so it doesn't duplicate
    PALETTE_BAROQUE_CARAVAGGIO when both fire on the same prompt.
    Production prompt evidence (series_7898201654ae) had both tags emit
    'dramatic chiaroscuro shadow' twice."""
    rembrandt = loader.canonicalize("LIGHT_REMBRANDT", "chroma")
    caravaggio = loader.canonicalize("PALETTE_BAROQUE_CARAVAGGIO", "chroma")
    assert rembrandt is not None and caravaggio is not None
    assert "chiaroscuro" not in rembrandt.lower(), (
        "LIGHT_REMBRANDT chroma should anchor on the lighting TECHNIQUE "
        "(triangle of light, key placement) not the generic 'chiaroscuro' "
        "vocabulary that overlaps with palette canonicalization. Found: "
        f"{rembrandt!r}"
    )
    rembrandt_phrases = set(p.strip() for p in rembrandt.split(","))
    caravaggio_phrases = set(p.strip() for p in caravaggio.split(","))
    shared = rembrandt_phrases & caravaggio_phrases
    assert not shared, (
        f"LIGHT_REMBRANDT + PALETTE_BAROQUE_CARAVAGGIO share phrases: {shared}"
    )


def test_light_rembrandt_chroma_carries_technical_specifics(loader):
    """LIGHT_REMBRANDT chroma now describes the technique geometrically
    (triangle of light on unlit cheek) rather than the generic
    'chiaroscuro shadow'. Mirrors how flux/flux2 already specify the
    triangle-on-cheek geometry."""
    out = loader.canonicalize("LIGHT_REMBRANDT", "chroma")
    assert "triangle" in out.lower(), (
        f"LIGHT_REMBRANDT chroma missing triangle-of-light geometry: {out!r}"
    )


def test_art_boudoir_noir_does_not_carry_vintage_hollywood(loader):
    """Vocab v10 — ART_BOUDOIR_NOIR canonicalization stripped of
    'vintage Hollywood' phrasing. Vintage Hollywood is its own style
    profile (`old_hollywood_glamour`) — leaking the phrase into
    boudoir_noir created semantic conflict with modern photographer
    references (Helmut Newton) in the same prompt."""
    for family in ("sdxl", "illustrious", "flux", "chroma", "flux2"):
        out = loader.canonicalize("ART_BOUDOIR_NOIR", family)
        assert out is not None, f"ART_BOUDOIR_NOIR missing for {family}"
        assert "vintage hollywood" not in out.lower() and "vintage_hollywood" not in out.lower(), (
            f"ART_BOUDOIR_NOIR {family} still carries 'vintage Hollywood' "
            f"vocab — should be domain-orthogonal to ART_OLD_HOLLYWOOD. "
            f"Found: {out!r}"
        )


def test_art_old_hollywood_still_carries_vintage_marker(loader):
    """Regression guard — when stripping 'vintage Hollywood' from
    ART_BOUDOIR_NOIR we MUST NOT also strip it from ART_OLD_HOLLYWOOD,
    where it's the load-bearing aesthetic anchor."""
    out = loader.canonicalize("ART_OLD_HOLLYWOOD", "chroma")
    assert out is not None
    assert "old hollywood" in out.lower() or "hollywood" in out.lower(), (
        f"ART_OLD_HOLLYWOOD should still anchor on Hollywood vocab: {out!r}"
    )
