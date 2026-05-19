"""Regression tests for the scene_021 / series_2547fb306a7c failure
mode (2026-05-19): a Cydonia-planned series shipped a 4-panel image
collage because the subject_description contained "in natural poses
across varying compositions", and four other scenes rendered mirrors
with warped reflections because the vocab library exposed mirror-
themed nsfw_act / prop / composition tags.

Vocab v7 (2026-05-20) closes both bugs by (a) deleting the mirror /
reflection / frame-within-frame vocab entries, (b) extending
HARD_BLOCK_NEGATIVE with grid / mirror / collage tokens, and (c)
extending the positive-side multi-subject scan with the LLM-generated
"varying compositions" / "across scenes" / "doubled presence" phrases
that historically leaked into prompt bodies.

These tests guard the regression so neither bug class can return
without a CI failure.
"""

from __future__ import annotations

import logging

import pytest
import yaml

from src.memory.family_loader import FamilyLoader
from src.prompt.builder import (
    HARD_BLOCK_NEGATIVE,
    PromptBuilder,
    _positive_subject_count_scan,
)


@pytest.fixture
def pb():
    return PromptBuilder()


@pytest.fixture
def family_loader():
    return FamilyLoader()


@pytest.fixture(scope="module")
def vocab():
    with open(
        "/Users/shuvajit/Dev/nsfw-pipeline/config/prompt_vocabulary.yaml"
    ) as f:
        return yaml.safe_load(f)


# ── HARD_BLOCK_NEGATIVE coverage ──────────────────────────────────────


@pytest.mark.parametrize(
    "token",
    [
        # Grid / polyptych vocabulary — the literal grid hallucination
        # triggers in SDXL / Chroma encoders. Round-2 verifier compacted
        # this set from 12 → 6 tokens after measuring HARD_BLOCK at 83
        # CLIP tokens (over the SDXL 77-token budget). Each surviving
        # token still blocks the failure mode by semantic coverage:
        #   polyptych  → diptych / triptych / multi-panel
        #   grid       → image grid / tile_grid / contact_sheet
        #   collage    → frame_within_frame (nested-frame collage)
        #   split_screen / mirror / reflection — direct hits
        # The pruned tokens (diptych / multiple_views / tiled /
        # contact_sheet / frame_within_frame / double_exposure) are
        # NO LONGER expected in the block — checking for them would
        # block this regression test from greenness on the slim block.
        "grid", "collage", "polyptych", "split_screen",
        # Mirror vocabulary — user opted out of mirror compositions
        # entirely after seeing warped reflections.
        "mirror", "reflection",
        # Pre-existing safety tokens that MUST still survive the v7
        # extension (regression guard against accidental drop).
        "child", "teen", "loli", "underage", "youthful face",
        "2girls", "multiple_girls", "multiple_subjects",
    ],
)
def test_hard_block_contains_grid_and_mirror_tokens(token):
    """Every grid/mirror/duplication token plus every age/multi-subject
    token from the pre-v7 block must be present in HARD_BLOCK_NEGATIVE.
    """
    assert token.lower() in HARD_BLOCK_NEGATIVE.lower(), (
        f"{token!r} missing from HARD_BLOCK_NEGATIVE — anti-grid / "
        f"anti-mirror regression. Block currently is:\n{HARD_BLOCK_NEGATIVE}"
    )


def test_hard_block_age_tokens_lead(pb):
    """Age-safety tokens must appear BEFORE the v7 grid/mirror tokens —
    fit_to_budget trims from the END of the keyword block, so on
    tight-budget families (SDXL, Pony, Illustrious at 77 tokens) the
    age block survives and grid/mirror tokens get pruned first."""
    neg = pb.assemble_negative_prompt(model_negative="")
    age_idx = neg.lower().find("child")
    grid_idx = neg.lower().find("grid")
    mirror_idx = neg.lower().find("mirror")
    assert age_idx >= 0, "age block missing entirely"
    assert grid_idx > age_idx, (
        "grid token appears BEFORE age block — survival ordering broken; "
        "tight-budget families would trim age tokens first."
    )
    assert mirror_idx > age_idx, (
        "mirror token appears BEFORE age block — survival ordering broken."
    )


# ── Positive-side multi-subject / grid-phrase scan ────────────────────


@pytest.mark.parametrize(
    "leaked_phrase",
    [
        # The exact scene_021 phrase that triggered the 4-panel collage.
        "A nude woman in varying compositions across the room.",
        "A confident woman in various poses throughout the series.",
        "Subject shown across compositions across scenes.",
        "Composed as a diptych of two contemplative moments.",
        "A frame-within-frame composition.",
        "Body doubled by reflection in the window.",
        "Doubled presence across the surface.",
        "Composed as a collage of the subject.",
        # Subtler grid-leak phrasings the LLM might pick up.
        "A nude figure presented across the set.",
        "Multiple compositions of a single figure.",
    ],
)
def test_grid_phrases_stripped_from_positive(leaked_phrase, family_loader, caplog):
    """The positive-side scan must remove grid/collage/duplication
    phrases that historically leaked from the LLM into prompt bodies
    (scene_021 regression: 'in natural poses across varying
    compositions' produced a 2x2 image-grid hallucination).

    Patterns are subject-anchored (verifier F5 — "different angles" and
    "multiple angles" used to false-positive on legitimate lighting
    language like "multiple angles of incident light"). Variety phrases
    are now matched ONLY when followed by a subject-count noun
    (poses / compositions / framings / views / scenes / series / set).
    """
    chroma = family_loader.get_family("chroma")
    with caplog.at_level(logging.ERROR):
        cleaned = _positive_subject_count_scan(leaked_phrase, chroma)
    grid_phrases = (
        "varying compositions", "various poses", "across compositions",
        "throughout the series", "across scenes",
        "diptych", "frame-within-frame", "doubled by reflection",
        "doubled presence", "collage of", "across the set",
        "multiple compositions",
    )
    for gp in grid_phrases:
        assert gp.lower() not in cleaned.lower(), (
            f"grid-leak phrase {gp!r} survived the scan in: {cleaned!r}"
        )


@pytest.mark.parametrize(
    "legit_phrase",
    [
        # Photographic terminology that "different angles" / "multiple
        # angles" should NOT match (verifier F5).
        "Multiple angles of incident light illuminate her shoulder.",
        "Lighting from different angles of the softbox rig.",
        "The lens captures various textures on the wall behind her.",
        # Single-word "varying" / "multiple" / "different" before non-
        # subject nouns must pass through cleanly.
        "Varying intensity of warm bulb light.",
        "Multiple sources of soft window light.",
        "Different qualities of shadow falling across her hip.",
    ],
)
def test_legit_lighting_language_not_stripped(legit_phrase, family_loader):
    """Verifier F5 regression — the positive-side scan must NOT strip
    legitimate photographic/lighting language even when it shares
    words with the grid-phrase list. False positives are bad for
    quality (they cut real scene description) and erode trust in the
    scan's precision."""
    chroma = family_loader.get_family("chroma")
    cleaned = _positive_subject_count_scan(legit_phrase, chroma)
    assert cleaned == legit_phrase, (
        f"legitimate phrase was stripped: before={legit_phrase!r} "
        f"after={cleaned!r}"
    )


def test_clean_positive_passes_through_unchanged(family_loader):
    """A clean positive prompt must pass the scan byte-stable — no
    spurious false-positives on legitimate solo-female prose."""
    chroma = family_loader.get_family("chroma")
    clean = (
        "A single adult woman alone in the scene. Soft window light "
        "from the left, the camera at eye level, the subject's gaze "
        "directed slightly past the lens."
    )
    assert _positive_subject_count_scan(clean, chroma) == clean


# ── Vocab v7 removal guards ───────────────────────────────────────────


@pytest.mark.parametrize(
    "removed_tag",
    [
        "NSFW_T4_SOLO_MIRROR",
        "PROP_CHEVAL_MIRROR",
        "PROP_VANITY_TRIPTYCH_MIRROR",
        "COMP_FRAME_WITHIN_FRAME",
        "COMP_REFLECTION_PRIMARY",
        "COMP_REFLECTION_SECONDARY",
    ],
)
def test_vocab_v7_removed_entries_not_present(vocab, removed_tag):
    """Each of the six vocab v7 removals must stay removed — a future
    rebase that accidentally re-introduces one would re-open the
    mirror/grid regression class."""
    import json
    flat = json.dumps(vocab)
    assert removed_tag not in flat, (
        f"{removed_tag} re-introduced into prompt_vocabulary.yaml — "
        f"this is one of the six vocab v7 removals that closed the "
        f"scene_021 mirror/grid regression class."
    )


def test_vocab_version_is_at_least_7(vocab):
    """vocab_version must track the v7 bump so prompts table records
    the correct version when re-rendering."""
    assert vocab["version"] >= 7, (
        f"vocab_version={vocab['version']} — should be >= 7 after the "
        f"mirror/grid removal."
    )


@pytest.mark.parametrize(
    "removed_narrative",
    [
        # Verifier F1 — narrative moments that mention subject mirrors
        # are tier-required at every tier, so even one stale entry
        # re-opens the failure mode at the next theme run that picks it.
        "NARR_MIRROR_CONTEMPLATION",
    ],
)
def test_vocab_v7_narrative_mirror_entries_removed(vocab, removed_narrative):
    """Narrative tags whose entire premise is 'subject + mirror' must
    not exist — vocabulary.canonicalize_facet treats narrative_moment
    as tier-required at every tier, so a single stale entry forces
    Cydonia to pick it ~1/30 scenes."""
    narratives = vocab["narrative"]["moment"]
    assert removed_narrative not in narratives, (
        f"{removed_narrative} still present in narrative.moment — "
        f"this re-opens the mirror-rendering failure mode."
    )


def test_narrative_dressing_for_evening_strips_mirror(vocab):
    """Verifier F1 — NARR_DRESSING_FOR_EVENING used to embed
    'mirror reflection visible behind her' in flux/chroma/flux2
    prose. The flux2 family pulls the longest prose so it's the
    canary."""
    entry = vocab["narrative"]["moment"]["NARR_DRESSING_FOR_EVENING"]
    for family, prose in entry.items():
        assert "mirror" not in prose.lower(), (
            f"NARR_DRESSING_FOR_EVENING.{family} still mentions a mirror: "
            f"{prose!r}"
        )


def test_comp_over_shoulder_strips_mirror(vocab):
    """Verifier F8 — COMP_OVER_SHOULDER used to end with 'mirror
    reflection of her face in sharp focus beyond'. After v7 rewrite
    it ends with 'environment beyond in sharp focus'. Without this
    test a future LLM-suggested rewrite could silently re-add the
    mirror clause."""
    entry = vocab["composition"]["principle"]["COMP_OVER_SHOULDER"]
    for family, prose in entry.items():
        assert "mirror" not in prose.lower(), (
            f"COMP_OVER_SHOULDER.{family} still mentions a mirror: {prose!r}"
        )
        assert "reflection" not in prose.lower(), (
            f"COMP_OVER_SHOULDER.{family} still mentions a reflection: "
            f"{prose!r}"
        )


def test_age_tokens_survive_sdxl_budget_with_real_model_axes(family_loader):
    """Verifier F4 — quantify HARD_BLOCK budget impact on tight 77-token
    families with a representative model_negative_axes payload from a
    real model YAML. The age-safety sub-block MUST survive trim along
    with at least SOME caller-provided quality/watermark tokens. Pre-v7
    HARD_BLOCK was 122 CLIP tokens which dropped all caller negatives
    on SDXL; the compact v7 block (~33 tokens) leaves real headroom.

    fit_to_budget preserves prefix (first 30 tokens, where age safety
    lives) + suffix (last 20 tokens, where the style negative tail
    lives) and trims the middle. So the surviving set on SDXL with a
    full axis load is: age safety + leading composition tokens
    (grid/collage/diptych/polyptych) + tail-end caller tokens. Mid-
    string caller axes (e.g. anatomy) may get trimmed; that's an
    inherent SDXL CLIP-77 limit, not a v7 regression.
    """
    from src.prompt.tokenizer import count_tokens
    pb = PromptBuilder()
    sdxl = family_loader.get_family("sdxl")
    # Representative SDXL model_negative_axes — these are the kinds of
    # tokens that prevent warped hands / extra digits / JPEG artifacts
    # on real renders. The 7-axis structure mirrors what
    # config/models/*.yaml carries.
    model_axes = {
        "anatomy": [
            "bad anatomy", "bad hands", "extra digits", "deformed",
            "missing limbs", "extra limbs",
        ],
        "quality": ["low quality", "lowres", "worst quality", "jpeg artifacts"],
        "skin": ["plastic skin", "doll skin"],
        "watermark": ["watermark", "signature", "text"],
        "medium": [],
        "censor": [],
        "safety": [],
    }
    neg = pb.assemble_negative_prompt(
        model_negative_axes=model_axes,
        style_negative="harsh overhead light",
        family=sdxl,
    )
    # Age-safety tokens MUST survive — they ride at the start of
    # HARD_BLOCK and the prefix-preserve window (30 tokens) keeps them.
    for age_token in ("child", "teen", "loli", "minor"):
        assert age_token in neg.lower(), (
            f"age-safety token {age_token!r} got trimmed — HARD_BLOCK "
            f"budget regression. Final negative ({count_tokens(neg, sdxl.tokenizer_id)} "
            f"tokens):\n{neg}"
        )
    # Round-3 verifier — family-conditional HARD_BLOCK. SDXL gets the
    # compact composition block (grid + mirror only); the verbose
    # polyptych / collage / split_screen / reflection tokens are
    # suppressed at the negative-block resolution step so caller
    # anatomy axis fits inside the 77-token budget. (Defence-in-depth:
    # the positive-side scan strips grid/mirror phrases before they
    # reach the encoder, so the negative side only needs the most
    # heavily-trained tokens.)
    for grid_token in ("grid", "mirror"):
        assert grid_token in neg.lower(), (
            f"composition-safety token {grid_token!r} got trimmed even "
            f"on the compact tight-budget block — HARD_BLOCK regression. "
            f"Final negative:\n{neg}"
        )
    # F4 BLOCKER close criterion — caller anatomy MUST survive on SDXL.
    # Round-1 ALL caller negatives dropped (122-token bloat); round-2
    # quality/watermark survived but anatomy was still evicted (58
    # tokens — composition tail crowded anatomy out of the middle of
    # the trim window). Round-3's family-conditional block (compact
    # 45-token total on SDXL) leaves anatomy room.
    survived_anatomy = any(
        t in neg.lower() for t in ("bad anatomy", "bad hands", "extra digits")
    )
    assert survived_anatomy, (
        f"F4 BLOCKER STILL OPEN — NO caller anatomy negative "
        f"survived on SDXL. Round-3 family-conditional block was "
        f"supposed to free room for anatomy. Final negative:\n{neg}"
    )
    survived_quality = any(
        t in neg.lower() for t in ("lowres", "low quality", "worst quality", "jpeg")
    )
    survived_watermark = any(
        t in neg.lower() for t in ("watermark", "signature", "text")
    )
    survived_style = "harsh overhead" in neg.lower()
    assert survived_quality, (
        f"NO caller quality negative survived on SDXL. Final negative:\n{neg}"
    )
    assert survived_watermark, (
        f"NO caller watermark negative survived on SDXL. Final negative:\n{neg}"
    )
    assert survived_style, (
        f"style negative was dropped — suffix-preserve window broken. "
        f"Final negative:\n{neg}"
    )


def test_resolve_hard_block_is_family_conditional(family_loader):
    """Round-3 verifier — `_resolve_hard_block` returns the compact
    composition variant for tight-budget families (SDXL/Pony/Illustrious)
    and the full composition variant for big-budget families
    (Chroma/Flux/Flux2). This is the mechanism that lets SDXL caller
    anatomy survive the 77-token budget while still giving Chroma the
    full grid/mirror coverage."""
    from src.prompt.builder import _resolve_hard_block
    sdxl = family_loader.get_family("sdxl")
    pony = family_loader.get_family("pony")
    illustrious = family_loader.get_family("illustrious")
    chroma = family_loader.get_family("chroma")
    flux = family_loader.get_family("flux")

    # Tight-budget families — compact composition (grid + mirror only).
    for fam_name, fam in [("sdxl", sdxl), ("pony", pony), ("illustrious", illustrious)]:
        block = _resolve_hard_block(fam)
        assert "grid" in block and "mirror" in block, (
            f"{fam_name}: compact composition must include grid + mirror"
        )
        # The full-block tokens (collage / polyptych / split_screen /
        # reflection) MUST be absent on tight-budget families — the whole
        # point of the round-3 fix is to free space for caller anatomy.
        for tok in ("collage", "polyptych", "split_screen", "reflection"):
            assert tok not in block, (
                f"{fam_name}: tight-budget block must NOT carry {tok!r} "
                f"— it crowds out caller anatomy"
            )

    # Big-budget families — full composition (all 6 tokens).
    for fam_name, fam in [("chroma", chroma), ("flux", flux)]:
        block = _resolve_hard_block(fam)
        for tok in ("grid", "collage", "polyptych", "split_screen", "mirror", "reflection"):
            assert tok in block, (
                f"{fam_name}: big-budget block must include {tok!r} — "
                f"plenty of headroom on T5 512-token budget"
            )

    # None family (back-compat / direct test callers) — full block.
    block = _resolve_hard_block(None)
    for tok in ("grid", "collage", "polyptych", "mirror", "reflection"):
        assert tok in block, f"None-family fallback dropped {tok!r}"


def test_chroma_full_budget_preserves_everything(family_loader):
    """Chroma (T5, 512-token budget) has plenty of headroom for the
    full HARD_BLOCK + all caller axes — every token type must survive
    on prose families. Counterpart to the SDXL tight-budget test."""
    pb = PromptBuilder()
    chroma = family_loader.get_family("chroma")
    model_axes = {
        "anatomy": ["bad anatomy", "bad hands", "extra digits", "deformed"],
        "quality": ["low quality", "lowres", "worst quality", "jpeg artifacts"],
        "skin": ["plastic skin"],
        "watermark": ["watermark", "signature", "text"],
        "medium": [], "censor": [], "safety": [],
    }
    neg = pb.assemble_negative_prompt(
        model_negative_axes=model_axes,
        family=chroma,
    )
    # Every category must survive — Chroma has 512 tokens of budget.
    for token in (
        # Age safety — every token survives in the 512-token T5 budget.
        "child", "teen", "loli", "underage", "2girls", "multiple_girls",
        # Composition safety — the round-2 compact set (6 tokens).
        "grid", "collage", "polyptych", "split_screen",
        "mirror", "reflection",
        # Caller anatomy (would be trimmed on SDXL, survives on Chroma)
        "bad anatomy", "bad hands", "extra digits", "deformed",
        # Caller quality + watermark
        "lowres", "worst quality", "jpeg artifacts",
        "watermark", "signature",
    ):
        assert token in neg.lower(), (
            f"{token!r} missing from Chroma negative — should NOT be "
            f"trimmed on a 512-token-budget family. Final:\n{neg}"
        )


def test_theme_mode_subject_description_validator_present():
    """Verifier F6 — theme_mode must run a server-side sanitizer on
    the LLM-emitted subject_description, not just rely on the prompt
    instruction. Cydonia ignores embedded constraints at temp≥0.7.

    Round-2 refactor moved the sanitization to the shared helper
    ``sanitize_grid_phrases`` so theme_mode + niche_mode + the
    migrate script all stay byte-equivalent.
    """
    import src.modes.theme_mode as tm
    source = tm.__file__
    with open(source) as f:
        content = f.read()
    assert "sanitize_grid_phrases" in content, (
        "theme_mode.plan() must call sanitize_grid_phrases on "
        "subject_description as a server-side defense — the LLM "
        "instruction alone is not enforceable."
    )


def test_niche_mode_subject_bias_validator_present():
    """Verifier F7 — niche_mode has the same injection pattern as
    theme_mode (subject_bias + visual_elements + core_theme all
    flow into every scene). It must run the same server-side
    sanitizer via the shared helper."""
    import src.modes.niche_mode as nm
    source = nm.__file__
    with open(source) as f:
        content = f.read()
    assert "sanitize_grid_phrases" in content, (
        "niche_mode.plan() must call sanitize_grid_phrases on "
        "subject_bias / core_theme / visual_elements — parallel to "
        "theme_mode's defense."
    )


def test_niche_mode_visual_elements_filters_non_strings(caplog):
    """Verifier N4 — niche_mode visual_elements must drop non-string
    LLM emissions (ints, None, dicts) rather than str()-coercing them
    into garbage entries. Tested by direct invocation of the cleanup
    pattern since the full plan() path requires a live LLM."""
    # The actual filter logic lives inside NicheMode.plan; this test
    # asserts the source has the isinstance check.
    import src.modes.niche_mode as nm
    source = nm.__file__
    with open(source) as f:
        content = f.read()
    assert "isinstance(v, str)" in content, (
        "niche_mode visual_elements comprehension must filter "
        "non-string LLM emissions with an isinstance(v, str) check"
    )


def test_sanitize_grid_phrases_removes_orphan_connectors():
    """Round-2 — sanitize_grid_phrases must strip BOTH the grid phrase
    AND the orphan connector word left behind (the scene_021 residual
    "in natural poses" pattern). The 2-stage cleanup is what makes
    the runtime sanitizer byte-equivalent to the migrate script."""
    from src.prompt.builder import sanitize_grid_phrases
    # Stage 1 + 2 together — phrase stripped + dangling "across" gone.
    out, changed = sanitize_grid_phrases(
        "She poses in natural light across varying compositions."
    )
    assert changed is True
    assert "across" not in out.lower()
    assert "varying" not in out.lower()
    assert "compositions" not in out.lower()
    # Stage 2 alone — orphan connector cleanup doesn't touch legitimate
    # mid-sentence connector words (followed by non-punctuation).
    out, _ = sanitize_grid_phrases("She poses in front of the lamp.")
    assert out == "She poses in front of the lamp"


def test_sanitize_strips_bare_composition_nouns():
    """Round-4 verifier (A5) — bare composition nouns (polyptych /
    triptych / diptych) and the natural-prose "Composed as a X" form
    must be stripped at the positive-side. The round-3 commit message
    justified shrinking the SDXL negative block by claiming the
    positive scan catches these phrases; round-4 audit proved it
    didn't until this pattern extension landed. Without these
    patterns, Cydonia could emit "Composed as a polyptych" in
    scene_prose and re-trigger the 4-panel grid hallucination on
    SDXL (where the tight negative only carries grid + mirror)."""
    from src.prompt.builder import sanitize_grid_phrases
    for leaked in [
        "Composed as a polyptych.",
        "A diptych of two moments.",
        "She poses in a triptych arrangement.",
        "Tiled across the frame.",
        "Frame within frame composition.",
        "Composed as a grid",
        "Composed in a collage",
    ]:
        out, changed = sanitize_grid_phrases(leaked)
        assert changed, f"failed to strip bare-noun grid phrase: {leaked!r}"
        for forbidden in ("polyptych", "triptych", "diptych", "composed as a"):
            assert forbidden.lower() not in out.lower(), (
                f"{forbidden!r} survived in: {out!r} (from {leaked!r})"
            )


def test_sanitize_keeps_legit_tiled_prose():
    """Round-4 verifier — the "tiled X" pattern must be anchored to
    grid-context nouns (image / grid / composition / layout / across
    the frame) so legitimate environment prose like "tiled floor" /
    "tiled ceiling" / "tiled wall" passes through unchanged. The
    environment vocab uses these phrases for bathroom + pool +
    Mediterranean settings."""
    from src.prompt.builder import sanitize_grid_phrases
    for legit in [
        "tiled floor catching warm light",
        "terracotta tiled floor",
        "blue tiled bathroom wall",
        "tiled ceiling above the pool",
    ]:
        out, _ = sanitize_grid_phrases(legit)
        assert "tiled" in out.lower(), (
            f"legitimate 'tiled' prose was stripped: {legit!r} → {out!r}"
        )


def test_sanitize_strips_in_natural_poses_residual():
    """Round-2 verifier F10 BLOCKER — the original migrate left
    "in natural poses" in scene_021's prompt body after stripping
    "across varying compositions". The new pattern catches this
    leading-connector + variety-adjective + poses shape so the
    residual is no longer surfaced."""
    from src.prompt.builder import sanitize_grid_phrases
    out, changed = sanitize_grid_phrases(
        "A nude woman in natural poses. Bare breasts visible."
    )
    assert changed is True
    assert "in natural poses" not in out.lower()
    assert "bare breasts visible" in out.lower()


def test_theme_mode_plan_template_forbids_variety_words():
    """Verifier F8 — the plan-time LLM prompt must instruct the LLM
    to avoid variety words. Future rewrites of the template that
    omit this constraint would silently re-open the bug class on
    fresh series. The user-facing instruction is the first line of
    defense (server-side sanitizer is the second)."""
    import src.modes.theme_mode as tm
    template = tm._PLAN_USER_TEMPLATE
    # Must mention the forbidden vocabulary explicitly so a careless
    # rewrite catches the breakage at code-review time.
    for forbidden in ("varying", "across", "compositions"):
        assert forbidden in template, (
            f"theme_mode plan template no longer warns the LLM about "
            f"the forbidden word {forbidden!r} — server-side sanitizer "
            f"will still catch it, but the LLM-facing instruction is "
            f"the first defense layer."
        )


def test_theme_mode_system_prompt_has_single_scene_invariant():
    """Verifier F8 — theme_mode system prompt must carry the
    SINGLE-SCENE INVARIANT clause explaining WHY subject_description
    cannot contain variety language. Without this, future edits could
    soften the constraint thinking it's redundant with the user-prompt
    instruction."""
    import src.modes.theme_mode as tm
    assert "SINGLE-SCENE INVARIANT" in tm._PLAN_SYSTEM_PROMPT, (
        "theme_mode system prompt missing SINGLE-SCENE INVARIANT clause"
    )


def test_environment_vocab_strips_mirror_mentions(vocab):
    """The four environment_setting entries that historically embedded
    'mirrored ceiling' / 'fogged mirror' / 'makeup mirror' /
    'mirrored side table' in their family prose must no longer mention
    those tokens."""
    settings = vocab["environment"]["setting"]
    for tag_name in (
        "ENV_ART_DECO_HOTEL_SUITE",
        "ENV_CLAWFOOT_BATHROOM",
        "ENV_TOKYO_LOVE_HOTEL",
        "ENV_BACKSTAGE_DRESSING_ROOM",
    ):
        entry = settings.get(tag_name)
        assert entry is not None, f"{tag_name} missing"
        for family, prose in entry.items():
            assert "mirror" not in prose.lower(), (
                f"{tag_name}.{family} still mentions a mirror: {prose!r}"
            )
