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


# ── Mirror sentence-drop (2026-05-23 verifier audit) ──────────────────


@pytest.mark.parametrize(
    "lossy_prose, must_not_contain",
    [
        # series_79ae3b962c8d scene_000 — bare-noun strip left
        # "ornate gilded, her bare body reflected" dangling.
        (
            "She stands confidently nude before an ornate gilded mirror, "
            "her bare body reflected back at her. Sunlight streams "
            "through floor-to-ceiling windows.",
            ["before an ornate gilded,", "reflected back at her"],
        ),
        # series_79ae3b962c8d scene_006 — bare-noun strip left
        # "kneels before her, dramatic side lighting".
        (
            "A confident adult woman kneels before her mirror, dramatic "
            "side lighting casting sharp shadows across her mature features.",
            ["kneels before her,", "kneels before her mirror"],
        ),
        # scene_007 — "in a hand mirror" → "in a hand" dangling.
        (
            "She gazes at herself in a hand mirror with a content smile, "
            "her natural beauty illuminated by the golden hour glow.",
            ["in a hand,", "in a hand mirror"],
        ),
        # scene_012 — "smiles gently at in" dangling.
        (
            "She smiles gently at the mirror, one hand resting on her "
            "full hip, the other playfully tucking a loose strand.",
            ["smiles gently at,", "smiles gently at the mirror"],
        ),
        # scene_015 — "stands before, naked and unapologetic".
        (
            "She stands before the mirror, naked and unapologetic, her "
            "muscular frame cast in harsh window light.",
            ["stands before,", "stands before the mirror"],
        ),
        # scene_026 — multi-sentence + reflection-without-mirror tail.
        (
            "In the soft glow from a nearby window, she examines herself "
            "in a handheld mirror, capturing an intimate moment of "
            "self-appreciation. Her pose reveals natural curves, one arm "
            "draped across her bare chest. The other hand holds the "
            "mirror at eye level, reflecting her confident smile back at "
            "her. The bathroom's warm lighting casts dramatic shadows.",
            [
                "in a handheld,",
                "examines herself in",
                "holds the at eye level",
                "reflecting her confident smile back at her",
            ],
        ),
    ],
)
def test_mirror_sentence_drop_removes_dangling_syntax(lossy_prose, must_not_contain):
    """Verifier audit (series_79ae3b962c8d, 2026-05-23) — the bare-noun
    mirror strip left 10+ prompts with visible dangling syntax. The
    sentence-drop approach removes the WHOLE sentence containing
    mirror/reflection language so no orphan prepositions / articles
    survive."""
    from src.prompt.builder import sanitize_grid_phrases
    cleaned, changed = sanitize_grid_phrases(lossy_prose)
    assert changed, f"sanitizer did not change input: {lossy_prose!r}"
    for bad in must_not_contain:
        assert bad.lower() not in cleaned.lower(), (
            f"dangling-syntax artifact {bad!r} survived in cleaned: {cleaned!r}"
        )
    # Sanity — output should remain valid sentence prose (no orphan
    # comma-comma sequences, no leading commas).
    assert ", ," not in cleaned
    assert not cleaned.startswith(",")


def test_mirror_sentence_drop_preserves_non_mirror_content():
    """Sentence-drop should keep all non-mirror sentences intact."""
    from src.prompt.builder import sanitize_grid_phrases
    prose = (
        "She stands by the window, golden light tracing her shoulder. "
        "She examines herself in a handheld mirror. "
        "Her hair catches the afternoon sun."
    )
    cleaned, changed = sanitize_grid_phrases(prose)
    assert changed
    assert "stands by the window" in cleaned
    assert "afternoon sun" in cleaned
    assert "mirror" not in cleaned.lower()


def test_mirror_sentence_drop_noop_when_clean():
    """Mirror sentence drop must NOT touch prose with no mirror /
    reflection language (regression guard for false positives).
    Note: sanitize_grid_phrases strips trailing punctuation; compare
    on content-only basis (strip trailing ` ,.` from both)."""
    from src.prompt.builder import sanitize_grid_phrases
    prose = (
        "She kneels by the fireplace, soft amber light tracing the "
        "curve of her bare shoulder. Her hands rest on her hips."
    )
    cleaned, _ = sanitize_grid_phrases(prose)
    # Content-equivalent (modulo trailing period strip).
    assert cleaned.rstrip(" ,.") == prose.rstrip(" ,.")
    assert "kneels by the fireplace" in cleaned
    assert "Her hands rest on her hips" in cleaned


def test_mirror_sentence_drop_preserves_ambient_reflections():
    """Verifier carve-out — `reflected light`, `rippling reflections`
    in WATER / on FLOOR / on CEILING etc. are ambient photographic
    terms, not subject-mirror. These should NOT trigger the sentence
    drop."""
    from src.prompt.builder import sanitize_grid_phrases
    prose = (
        "Warm reflected light fills the room from the polished floor. "
        "Rippling reflections of the pool play across the ceiling."
    )
    cleaned, _ = sanitize_grid_phrases(prose)
    # Both ambient-reflection sentences must survive intact.
    assert "Warm reflected light" in cleaned, (
        f"ambient reflection sentence dropped: {cleaned!r}"
    )
    assert "Rippling reflections" in cleaned, (
        f"ambient reflection sentence dropped: {cleaned!r}"
    )


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
        # Skip metadata keys (place_constraint is a dict, not a prose string).
        if not isinstance(prose, str):
            continue
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


def test_theme_mode_plan_template_carries_lighting_lock():
    """2026-05-29 — the planner template must carry the LIGHTING & SETTING
    LOCK so the style profile's lighting_hint constrains the theme / mood /
    environment. Prior to this the planner only saw base_style_keywords and
    set a dark 'Midnight in the Velvet Parlour' theme for a soft-faded
    profile, which the downstream facet lighting lock couldn't override."""
    import src.modes.theme_mode as tm
    template = tm._PLAN_USER_TEMPLATE
    assert "LIGHTING & SETTING LOCK" in template
    # The three profile-sourced fields must be wired as placeholders.
    for placeholder in ("{lighting_hint}", "{palette_hint}",
                        "{preferred_environments}"):
        assert placeholder in template, (
            f"planner template missing {placeholder} — the profile's "
            f"lighting intent won't reach the planner LLM."
        )
    # The directive must tell the LLM the lock beats the category default.
    assert "OUTWEIGHS" in template


def test_humanize_env_tags():
    """ENV_* tags become a readable hint; empty list → permissive
    fallback (never blocks the planner)."""
    from src.modes.theme_mode import _humanize_env_tags
    assert _humanize_env_tags(
        ["ENV_MORNING_BEDROOM", "ENV_OLD_HOLLYWOOD_BOUDOIR"]
    ) == "morning bedroom, old hollywood boudoir"
    assert "any setting" in _humanize_env_tags([])
    assert "any setting" in _humanize_env_tags(None)


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


@pytest.mark.parametrize(
    "subject_mirror_phrase",
    [
        # Round-5 verifier (F1 BLOCKER) — these are the actual phrases
        # the LLM emitted into scene_prose for 20 of 25 scenes in the
        # user's series_2547fb306a7c. Each must be stripped at compose
        # time so the encoder never sees subject-mirror language.
        "She gazes softly at her reflection with lips slightly parted.",
        "stands confidently in front of a floor mirror, her reflection capturing her polished look",
        "A nude woman reclines beside a floor-length mirror, her silhouette dramatically lit",
        "background features classic vanity mirrors with warm vanity lights",
        "She is studying her own reflection in the antique mirror.",
        "considering her reflection in a tarnished antique mirror",
        "She gazes into the mirror with intimate self-regard.",
        # COMP_REFLECTION_PRIMARY canonical leftover
        "Composition with primary subject visible only as a reflection, real subject out of frame.",
        # COMP_REFLECTION_SECONDARY canonical leftover
        "Subject in frame and additionally reflected in a mirror, doubled presence.",
        # Mirrored environment leftover (after vocab v7 mirror prose strip)
        "Tokyo love-hotel room, mirrored ceiling above",
    ],
)
def test_subject_mirror_prose_stripped(subject_mirror_phrase):
    """F1 BLOCKER — every subject-mirror prose pattern that the LLM
    historically wrote into scene_prose must be stripped. No `mirror`
    or `her reflection` survives in the output."""
    from src.prompt.builder import sanitize_grid_phrases
    out, changed = sanitize_grid_phrases(subject_mirror_phrase)
    assert changed, f"FAILED to strip subject-mirror phrase: {subject_mirror_phrase!r}"
    out_lower = out.lower()
    for forbidden in (
        "mirror", "her reflection", "mirror reflection", "polyptych",
        "visible only as a reflection",
    ):
        # Special case: `mirrorless` is allowed (camera body); whole-word
        # match would not strip it but our regex strips bare mirror.
        assert forbidden not in out_lower or forbidden == "mirror" and "mirrorless" in out_lower, (
            f"forbidden subject-mirror token {forbidden!r} survived in: {out!r}"
        )


@pytest.mark.parametrize(
    "ambient",
    [
        # F1 carve-outs — atmospheric reflections / mirrorless cameras
        # must pass through unchanged. The user explicitly opted out of
        # SUBJECT-mirror compositions but accepts ambient/atmospheric
        # ones (water, light, sculpture reflections).
        "rippling reflections playing across the tiled ceiling",
        "terracotta tiled floor catching warm reflected light",
        "shot on a Sony A7R V full-frame mirrorless body",
        "Captured on a Canon EOS R5 full-frame mirrorless body",
        "metallic sculpture nearby catches the light, adding geometric reflections",
        "wet cobblestone reflecting streetlight",
    ],
)
def test_ambient_reflections_preserved(ambient):
    """F1 carve-out — ambient/atmospheric reflections + mirrorless
    camera-body terminology must pass through unchanged. Stripping
    these would degrade legitimate scene language and break per-family
    realism_camera canonicalization."""
    from src.prompt.builder import sanitize_grid_phrases
    out, _ = sanitize_grid_phrases(ambient)
    # Specific tokens that must survive
    if "mirrorless" in ambient.lower():
        assert "mirrorless" in out.lower(), (
            f"mirrorless camera-body terminology stripped: {ambient!r} → {out!r}"
        )
    if "rippling" in ambient.lower():
        assert "rippling" in out.lower(), (
            f"water reflection prose stripped: {ambient!r} → {out!r}"
        )
    if "reflected light" in ambient.lower():
        assert "reflected light" in out.lower(), (
            f"reflected light atmospheric prose stripped: {ambient!r} → {out!r}"
        )


def test_facet_generator_sanitizes_freetext_fields():
    """F2 — SceneFacetGenerator's `_sanitize_facet_freetext` helper
    must strip mirror/grid language from every free-text field
    declared in _FACET_SANITIZABLE_FIELDS. This is the runtime defense
    against the LLM writing grid/mirror prose into scene_prose,
    booru_tags, camera_spec, clothing."""
    from src.agents.scene_facet_generator import (
        _FACET_SANITIZABLE_FIELDS,
        _sanitize_facet_freetext,
    )
    # All four free-text fields covered by the helper.
    assert set(_FACET_SANITIZABLE_FIELDS) == {
        "scene_prose", "booru_tags", "camera_spec", "clothing",
    }
    dirty = {
        "scene_prose": "She gazes at her reflection in the mirror.",
        "booru_tags": "1girl, looking_at_mirror, mirror, reflection",
        "camera_spec": "85mm f/1.4 mirrorless body",  # mirrorless should survive
        "clothing": "silk gown beside a floor-length mirror",
        "lighting_directive": "LIGHT_REMBRANDT",  # not in sanitizable set
    }
    cleaned = _sanitize_facet_freetext(dirty, scene_id="test_001", family_id="chroma")
    # scene_prose: mirror + her reflection stripped
    assert "mirror" not in cleaned["scene_prose"].lower()
    assert "her reflection" not in cleaned["scene_prose"].lower()
    # booru_tags: mirror stripped
    assert "mirror" not in cleaned["booru_tags"].lower() or "mirrorless" in cleaned["booru_tags"].lower()
    # camera_spec: mirrorless camera terminology preserved (whole-word boundary)
    assert "mirrorless" in cleaned["camera_spec"].lower()
    # clothing: mirror stripped
    assert "mirror" not in cleaned["clothing"].lower()
    # lighting_directive: untouched (not a free-text field)
    assert cleaned["lighting_directive"] == "LIGHT_REMBRANDT"


def test_theme_mode_hard_fails_on_empty_sanitized_subject():
    """F5 — when sanitize_grid_phrases reduces subject_description to
    empty (LLM emitted ONLY grid/mirror phrases), theme_mode must
    raise rather than silently downgrade to no-subject. This surfaces
    LLM drift as an operator-visible error rather than a quietly
    generic render."""
    import src.modes.theme_mode as tm
    source = tm.__file__
    with open(source) as f:
        content = f.read()
    # Explicit raise path with a recognizable message must exist.
    assert "subject_description sanitized to" in content, (
        "theme_mode.plan() must raise ThemeModeError when sanitized "
        "subject_description is empty — silent downgrade hides LLM drift"
    )
    assert "ThemeModeError" in content
    assert "if not cleaned_sd:" in content


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
            # Skip metadata keys (place_constraint is dict-shaped).
            if not isinstance(prose, str):
                continue
            assert "mirror" not in prose.lower(), (
                f"{tag_name}.{family} still mentions a mirror: {prose!r}"
            )
