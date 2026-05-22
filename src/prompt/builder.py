"""Deterministic prompt construction — family-aware composer.

ARCHITECTURE.md Section 1 design principle 5 stands: the LLM writes
structured scene dicts, this module composes the final prompt text.
Two scenes with the same fields must produce byte-identical prompts
(``prompts.prompt_hash`` is used for dedup).

What changes vs. the pre-refactor version:

  - The composer is selected by a ``FamilyConfig`` (see
    ``config/families.yaml``), not a loose ``prompt_style`` string.
  - Four composers now, not three: ``sdxl_keywords``, ``pony_danbooru``
    (6-tier prefix), ``illustrious_tags`` (tag+prose hybrid with quality
    suffix), ``flux_natural``.
  - Quality prefix/suffix come from ``family.quality_prefix`` and
    ``family.quality_suffix`` — no hardcoded constants.
  - ``family.avoid_words`` is applied at compose time as a
    belt-and-braces strip pass (catches LLM drift).

Scene field order (each segment is dropped if empty, comma-joined,
deduped):

    [character.base_prompt]
    [scene.expression]
    [scene.pose]
    [scene.camera]
    [scene.camera_angle]
    [scene.lighting]
    [scene.environment_detail]
    [scene.mood_note]
    [style_profile.base_style_keywords]
    [extra_keywords]

CLIP-based encoders weight leading tokens more heavily, so identity
leads, scene shifts in the middle, style modifiers at the end.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Iterable, Mapping

from src.memory.family_loader import FamilyConfig
from src.prompt.negative_axes import (
    AXIS_KEYS,
    filter_conflicts,
    flatten_axes,
    normalize_axes,
)
from src.prompt.tokenizer import count_tokens, fit_to_budget
from src.prompt.vocabulary import canonicalize_facet, canonicalize_series_aesthetic


logger = logging.getLogger(__name__)

# Matches `(tag:1.2)`, `(tag:0.85)`, `(tag:1)` etc. Leaves the inner tag.
# Nested/escaped parens are out of scope — LLMs don't emit those.
_WEIGHTING_PATTERN = re.compile(r"\(([^():]+):\d+(?:\.\d+)?\)")


_SCENE_FIELD_ORDER: tuple[str, ...] = (
    "expression",
    "pose",
    "camera",
    "camera_angle",
    "lighting",
    "environment_detail",
    "mood_note",
)

# Always-on hard-block for age-ambiguity + multi-subject vocabulary —
# prepended to every family-level negative prompt. Belt-and-braces: the
# LLM's planner hint already bans these, the positive scan strips them
# from the body, and this blocks the encoder from generating them even
# if one slips through.
#
# 2026-05-17 — extended with booru-shaped subject-count tokens
# (``2girls / multiple_girls / multiple_subjects``). These match the
# ``subject_count`` axis tokens for booru families but live here to
# also fire on Chroma (which has its own prose-shaped axis tokens) for
# defence-in-depth. Pony's CLIP tokenizer treats e.g. ``2girls`` as a
# single learned token regardless of family-style preference.
#
# 2026-05-20 — extended with a COMPACT grid / mirror block. After a
# Cydonia series shipped 4-panel collage hallucinations (a "varying
# compositions" phrase in subject_description triggered a 2x2 grid) and
# the user explicitly opted out of mirror compositions, the negative
# side now blocks grid/polyptych and mirror vocab. The block is split
# into two ordered sub-blocks for SDXL/Pony/Illustrious's tight 77-token
# CLIP budget:
#
#   1. _AGE_SAFETY  (~17 tokens) — age-ambiguity + multi-subject. MUST
#      survive trim on every family. Ordered first so fit_to_budget
#      (which trims from the END) keeps it whole.
#   2. _COMPOSITION_SAFETY  (~16 tokens) — grid / mirror / collage.
#      Single canonical token per concept; the booru tokenizer maps
#      ``grid`` / ``mirror`` to learned tokens regardless of which
#      duplicate phrasing we'd otherwise include. Deduped down from
#      the original 25-token block so the safety floor + the
#      composition floor leave headroom (~44 tokens) for caller
#      anatomy / quality / character negatives on 77-token families.
#
# Note ``mirrorless`` / ``film_noir`` etc. are safe — these are
# whole-word token matches at the encoder, not substring.
#
# Round-2 verifier (2026-05-20): the verbose original block was 83
# CLIP tokens — over SDXL/Pony/Illustrious's 77-token budget, leaving
# zero room for caller anatomy/quality/watermark axes.
# Age sub-block kept WHOLE — every token is a safety-critical signal
# the SD safety filter pays attention to (`child` / `kid` / `minor`
# are distinct learned BPE tokens; dropping any narrows the safety
# net measurably). Composition sub-block compacted from 12 → 6 tokens
# by dropping coverage-duplicates that the survivors still block:
#   `diptych` → covered by `polyptych`
#   `tiled` / `contact_sheet` / `frame_within_frame` → covered by
#   `grid` / `collage` / `polyptych` semantically and (more
#   importantly) by their absence from training-set captions
#   `multiple_views` → covered by `multiple_subjects` in age block
#   `double_exposure` → covered by `reflection` (the only way SD
#   produces a double exposure is by interpreting it as a reflection)
# Final cost: ~57 CLIP tokens. Leaves ~18 tokens of caller headroom
# on SDXL — enough for ~6-8 anatomy/quality/watermark phrases (the
# top-priority axes a real model YAML carries). Measured + asserted
# by tests/test_anti_grid_regression.py.
_HARD_BLOCK_AGE_SAFETY = (
    "child, kid, young, minor, teen, schoolgirl, loli, shota, "
    "underage, baby, toddler, preteen, youthful face, "
    "2girls, multiple_girls, multiple_subjects"
)

_HARD_BLOCK_COMPOSITION_SAFETY = (
    "grid, collage, polyptych, split_screen, mirror, reflection"
)

# Round-3 verifier (2026-05-20): even the round-2 shrink left caller
# anatomy fully evicted on SDXL because the composition tail
# (mirror, reflection) crowds anatomy out of the budget. Solution:
# family-conditional composition block — tight-budget families
# (sdxl/pony/illustrious at 77 CLIP tokens) get only the 2 highest-
# leverage composition tokens (grid + mirror — these cover the user's
# observed failure modes); big-budget families (chroma/flux/flux2 at
# 512 T5 tokens) get the full 6-token block. Defence-in-depth via the
# positive-side `_positive_subject_count_scan` covers the rest on
# tight-budget families (the positive scan strips grid/mirror phrases
# from the LLM output before they reach the encoder, so the negative
# side only needs the most-trained-on tokens).
_HARD_BLOCK_COMPOSITION_SAFETY_TIGHT = "grid, mirror"

# Threshold for which families count as "tight budget" (CLIP-style 77
# tokens vs T5-style 512+). Anything ≤ 128 tokens is tight.
_TIGHT_BUDGET_THRESHOLD = 128


def archetype_overridden_by_planner(
    series_plan: Mapping[str, Any] | Any | None,
) -> bool:
    """Round-21 (2026-05-21) — True when the planner provided its OWN
    aesthetic anchors that supersede the operator-chosen style archetype.

    The operator passes a ``style_profile`` archetype (e.g.
    ``golden_hour_natural``) as a default-hint. Modes (theme / niche /
    style / variation) ask the planner LLM to pick a category and may
    derive series-level ``color_palette`` / ``photographer_ref`` /
    ``art_movement`` from that category's compatibility lists. When the
    planner exercises that latitude — choosing a category whose
    aesthetic doesn't match the archetype (e.g. operator hint
    ``golden_hour_natural`` + planner pick ``dark_boudoir_neonoir``) —
    the archetype's ``base_style_keywords`` directly CONTRADICT the
    rest of the prompt ("Golden hour, warm rim-light, haze, natural
    outdoor" inside a neon-noir series).

    The audit on series_799bec97e6d7 showed every one of 24 prompts
    carried this contradiction. Series-level aesthetic phrases
    (canonicalized from the planner's anchors) already supply the
    visual world, so the archetype keywords become redundant noise at
    best, contradictory at worst. This helper centralises the override
    detection so both ``build_one`` (positive prompt) and
    ``engine.run_phase_a`` (negative prompt) skip the archetype layer
    identically when the planner provided anchors.

    Returns ``True`` when ``series_plan`` is non-empty AND has any of
    ``color_palette`` / ``photographer_ref`` / ``art_movement``
    populated. Returns ``False`` for back-compat callers passing
    ``None`` (pre-vocab-v6 series have no anchors and should still get
    the archetype injected).
    """
    if not series_plan:
        return False
    # Accept dict-like or attribute-bearing object.
    def _get(key: str) -> Any:
        if hasattr(series_plan, "get"):
            return series_plan.get(key)
        return getattr(series_plan, key, None)
    for anchor_field in ("color_palette", "photographer_ref", "art_movement"):
        val = _get(anchor_field)
        if val and str(val).strip():
            return True
    return False


def _resolve_hard_block(family: Any | None) -> str:
    """Return the appropriate HARD_BLOCK for the family's token budget.

    Tight-budget families (sdxl/pony/illustrious, max_tokens ≤ 128)
    get the compact composition block (`grid, mirror`) so caller-
    provided anatomy / quality negatives survive. Big-budget families
    (chroma/flux/flux2, 512 tokens) get the full composition block.
    Callers that pass ``family=None`` (back-compat / direct test
    callers) get the full block — historical behaviour.
    """
    max_tokens = getattr(family, "max_tokens", None) if family else None
    if max_tokens is not None and max_tokens <= _TIGHT_BUDGET_THRESHOLD:
        return f"{_HARD_BLOCK_AGE_SAFETY}, {_HARD_BLOCK_COMPOSITION_SAFETY_TIGHT}"
    return f"{_HARD_BLOCK_AGE_SAFETY}, {_HARD_BLOCK_COMPOSITION_SAFETY}"


# Public default — kept as the FULL block for back-compat with any
# external caller that imports the constant directly. Internal callers
# (the composer's `assemble_negative_prompt`) call `_resolve_hard_block`
# instead to pick the right variant per family.
HARD_BLOCK_NEGATIVE = (
    f"{_HARD_BLOCK_AGE_SAFETY}, {_HARD_BLOCK_COMPOSITION_SAFETY}"
)

# Age-ambiguity vocabulary used by the positive-side scan. Matched
# case-insensitive as whole words. Keep in sync with HARD_BLOCK_NEGATIVE —
# but this list is the *positive* rewrite target, where the HARD_BLOCK
# covers the negative-prompt side.
_AGE_AMBIGUITY_TERMS = (
    "child", "kid", "young", "minor", "teen", "teenage", "teenager",
    "schoolgirl", "schoolboy", "loli", "shota", "underage", "baby",
    "toddler", "preteen", "youthful",
)

_AGE_AMBIGUITY_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in _AGE_AMBIGUITY_TERMS) + r")\b",
    re.IGNORECASE,
)


# Multi-subject vocabulary used by ``_positive_subject_count_scan`` —
# matched case-insensitive as whole-word (or multi-word) phrases. The
# scan strips these from the body and logs at ERROR (matching the
# age-safety scan's severity). 2026-05-17 single-female enforcement.
#
# Phrases that read as multi-subject in T4 partnered nsfw_act prose
# (``her partner``, ``their bodies``, ``two adult``, ``embrace
# between``, etc.) are covered. We intentionally exclude bare nouns
# like ``partner`` alone — those false-positive on legitimate uses
# ("partner sleeve", "partner colour").
_MULTI_SUBJECT_PATTERNS: tuple[str, ...] = (
    r"2girls",
    r"3girls",
    r"multiple_girls",
    r"multiple_subjects",
    r"multiple girls",
    r"multiple subjects",
    r"multiple people",
    r"two women",
    r"two girls",
    r"two adults",
    r"two bodies",
    r"two adult bodies",
    r"two figures",
    r"a couple",
    r"another person",
    r"another woman",
    r"another girl",
    r"her partner",
    r"his partner",
    r"their partner",
    r"with him",
    r"with her partner",
    r"between adults",
    r"embrace between",
    r"partnered intimate",
    r"partnered explicit",
    r"between two",
    r"group of",
    r"crowd of",
    r"threesome",
    # 2026-05-20 — grid / duplication phrases that read as multi-subject
    # to the SDXL/Chroma encoder. A scene_021-class regression where
    # Cydonia wrote "in natural poses across varying compositions" into
    # the series-level subject_description produced a 2x2 image grid.
    # Strip these phrases on the positive side so even if the LLM slips
    # one in, the encoder never sees it.
    #
    # Patterns are ANCHORED to subject-context (subject noun nearby) to
    # avoid stripping legitimate scene language. "Various poses across
    # the floor" matches; "multiple angles of incident light" does NOT
    # — the second is a photographic term, not a subject-count signal.
    # The anchor word list (poses / compositions / scenes / set /
    # series / views / framings) is the subject-count vocabulary the
    # LLM uses to describe cross-scene variety; standalone "various"
    # / "multiple" / "different" before a non-subject noun no longer
    # matches.
    r"varying (?:poses|compositions|framings|views|scenes)",
    r"various (?:poses|compositions|framings|views|scenes)",
    r"different (?:poses|compositions|framings|views|scenes)",
    r"multiple (?:poses|compositions|framings|views|scenes)",
    r"across (?:poses|compositions|framings|scenes|the series|the set)",
    r"throughout (?:the series|the set|the scenes)",
    # Round-2 verifier — the original migrate-script run on
    # series_2547fb306a7c stripped "across varying compositions" and
    # the dangling "across" but left "in natural poses" — a residual
    # grid hint and the actual surviving fragment that kept showing
    # up in the rendered prompt body. Match the leading-connector +
    # variety-adjective + plural-noun shape so future drift is caught
    # at compose time (and the migrate sweep below catches stored
    # rows).
    r"in (?:natural|varying|different|various|multiple) (?:poses|compositions|framings|views|scenes)",
    r"in a grid",
    r"as a grid",
    r"collage of",
    r"diptych of",
    r"polyptych of",
    r"split screen",
    r"split-screen",
    r"contact sheet",
    r"image grid",
    r"frame within frame",
    r"frame-within-frame",
    r"doubled presence",
    r"doubled by reflection",
    # Round-4 verifier (A5) — bare composition nouns and "composed as
    # a X" forms. The natural-prose phrasing the LLM is most likely to
    # emit ("Composed as a polyptych", "A diptych of moments") leaked
    # through the round-1 anchored patterns. Stripping them here at
    # the positive side means the encoder never sees them; the SDXL
    # tight negative block doesn't need to carry the expensive bare
    # nouns (each ~4-5 CLIP tokens). polyptych / triptych / diptych
    # are art-history terms that virtually never appear in NSFW
    # editorial prose for any reason other than grid-style
    # composition, so bare-matching is safe.
    r"polyptych",
    r"triptych",
    r"diptych",
    r"composed (?:as|in|like) (?:a|an) (?:polyptych|triptych|diptych|grid|collage)",
    r"tiled (?:image|grid|composition|layout|across the frame)",
    # Round-5 verifier (F1 BLOCKER) — subject-mirror prose leaked
    # through every layer. The LLM wrote "she gazes softly at her
    # reflection" / "floor-length mirror" / "the candid mirror
    # reflection" directly into scene_prose for 20 of 25 scenes in
    # series_2547fb306a7c. The vocab v7 deletion of NSFW_T4_SOLO_MIRROR
    # / PROP_CHEVAL_MIRROR / COMP_REFLECTION_* removed the LLM's MENU
    # of mirror concept tags, but the LLM's FREE-TEXT scene_prose
    # field carried mirror language anyway. Stripping at the positive
    # side ensures the encoder never sees subject-mirror language.
    #
    # `\bmirror\b` is safe against `mirrorless` (word boundary at
    # `mirrorless`'s end of "mirror" lands inside a word, so
    # `\bmirror\b` requires non-word-char after — does NOT match
    # `mirrorless`). Ambient reflections that the user finds OK
    # (`rippling reflections`, `warm reflected light`, `geometric
    # reflections` from sculpture, `light reflections`) are preserved
    # by anchoring `reflection` only to subject-context phrases. The
    # user's stated preference ("i do not need mirror at all") drives
    # the aggressive bare-`mirror` strip.
    r"\bmirror[s]?\b",
    # Booru-tag mirror variants (Pony / Illustrious) — `\b` doesn't
    # match `_` boundaries (underscore is a word char), so the above
    # pattern misses `looking_at_mirror`, `mirror_room`, etc. Match
    # mirror followed by tag-boundary chars (space, comma, period,
    # EOL) when preceded by a word char + underscore. Carve-out:
    # `mirrored_table` / `mirror_image` / `mirrored_X` (atmospheric
    # mirror-finished surface) NOT matched because what follows
    # mirror is another word char.
    r"\w+_mirror(?=[\s,.]|$)",
    r"\bmirror_\w+",
    r"her reflection",
    r"her own reflection",
    r"mirror reflection",
    r"reflection capture[sd]?",
    r"reflection (?:catches|reveals)",
    r"the reflection (?:of|captures|catches|reveals)",
    r"the (?:candid|tarnished|antique|intimate|quiet|natural|gentle|own) reflection",
    r"gazes? (?:at|into|toward) (?:her|the) (?:reflection|mirror)",
    r"studying her (?:own )?reflection",
    r"intimate self[- ]regard",
    r"quiet self[- ]regard",
    # Mirror-with-modifier compositions — even after the bare
    # `\bmirror\b` strip, the modifier remains ("floor-length mirror"
    # → "floor-length" alone). Match the full noun phrase so the
    # modifier goes too.
    r"(?:floor[- ]length|full[- ]length|hand[- ]held|antique|large|vanity|cheval|tarnished|gilt[- ]framed) mirror[s]?",
    r"mirrored (?:ceiling|surface|wall|side[- ]table|table)",
    # Round-5 — vocab-canonical residuals. NARR_MIRROR_CONTEMPLATION's
    # chroma phrasing was "considering her reflection in a tarnished
    # antique mirror, fingertips resting lightly at the throat" — after
    # bare-mirror + her-reflection strips, the orphan "considering in a
    # tarnished, fingertips" remains. NSFW_T4_SOLO_MIRROR's chroma
    # phrasing was "nude woman before a mirror studying her own
    # reflection" — after strip, "before a studying" dangles. These
    # specific phrasing residuals would only exist in pre-v7-composed
    # prompts.prompt_text (the vocab entries themselves are deleted).
    r"considering in a tarnished,? fingertips resting(?: lightly)?(?: at the throat)?",
    r"before a studying",
    r"before a (?:mirror )?studying",
    # COMP_REFLECTION_PRIMARY chroma canonical: "composition with
    # primary subject visible only as a mirror reflection, real
    # subject out of frame" — after bare-mirror strip becomes
    # "composition with primary subject visible only as a reflection,
    # real subject out of frame". Match the whole canonical sentence.
    r"composition with primary subject visible only as a reflection,? real subject out of frame",
    # COMP_REFLECTION_SECONDARY chroma canonical similarly.
    r"(?:subject )?(?:in frame )?and additionally reflected in a (?:window|mirror)",
    r"doubled presence(?: that complicates the composition's reading)?",
)

_MULTI_SUBJECT_PATTERN = re.compile(
    r"\b(?:" + "|".join(_MULTI_SUBJECT_PATTERNS) + r")\b",
    re.IGNORECASE,
)


# Round-2 verifier (2026-05-20) — orphan-connector cleanup.
# After _MULTI_SUBJECT_PATTERN.sub strips a phrase like "across
# varying compositions" from a sentence "posed across varying
# compositions", the leftover word "across" dangles before
# punctuation/EOL. This pattern catches that dangling connector and
# removes it so the resulting prose flows naturally. Bounded to
# the four common variety-phrase connectors (in / across /
# throughout / with) and only matches when immediately followed by
# punctuation or end-of-string — won't touch "across the floor" or
# "in front of the lamp" (those have non-punctuation followups).
ORPHAN_CONNECTOR_PATTERN = re.compile(
    r"\s+(?:in|across|throughout|with|at|against|before|beside|behind|toward|into|onto)\s*(?=[.,]|$)",
    re.IGNORECASE,
)

# Round-5 verifier (F3) — dangling-article cleanup. After bare-noun
# stripping ("A polyptych of the subject." → "A  of the subject."),
# the orphan article + "of" + following text reads ungrammatical.
# This pattern catches the broken "(A|An|The) of <X>" + "(A|An|The)
# <noun> <of>" + bare orphan article-before-punctuation fragments.
#
# Conservative — only matches when the article is immediately followed
# by another article or "of" / "in" / "from" / a closing punctuation,
# so legitimate prose like "A nude woman stands" is untouched.
DANGLING_ARTICLE_PATTERN = re.compile(
    r"\b(?:a|an|the)\s+(?:of|in|from|a|an|the)\s+",
    re.IGNORECASE,
)
DANGLING_ARTICLE_AT_END = re.compile(
    r"\b(?:a|an|the)\s*(?=[.,]|$)",
    re.IGNORECASE,
)


# Sentence-level mirror/reflection trigger. When a sentence contains
# any of these markers, the WHOLE sentence is dropped — surgical noun
# strip on mirror was leaving dangling syntax (`kneels before her`,
# `in a hand`, `holds the at eye level`) which T5 / CLIP read as
# broken English and rendered as gibberish. Sentence drop is cleaner
# because the entire reflection content is gone with no orphan-
# preposition / orphan-article artifacts.
#
# Production evidence (series_79ae3b962c8d, 2026-05-23 verifier audit):
# 10 of 28 prompts had visible mirror-strip damage. The bare-noun
# strip cannot fix this — it has to be a sentence-level drop.
_MIRROR_SENTENCE_TRIGGER = re.compile(
    r"\b(?:"
    r"mirror[s]?"
    r"|reflection\s+(?:of\s+her|catches|reveals|captures)"
    r"|her\s+(?:own\s+)?reflection"
    r"|reflected\s+back\s+at\s+(?:her|him)"
    r"|reflecting\s+her\s+\w+\s+back"
    r"|her\s+own\s+form\s+reflected"
    r"|gazes?\s+(?:into|at)\s+her\s+(?:reflection|own\s+form)"
    r"|examines?\s+herself\s+in\s+a\s+\w+"
    r"|studying\s+(?:her\s+(?:own\s+)?reflection|herself)"
    r")\b",
    re.IGNORECASE,
)


def _drop_mirror_sentences(text: str) -> tuple[str, bool]:
    """Drop entire sentences containing mirror or subject-reflection
    language. More conservative than surgical noun strip — avoids
    the dangling-syntax artifacts that caused the 2026-05-23 audit
    (orphan `before her`, `in a hand`, `holds the at eye level`).

    Returns ``(cleaned, changed)``. If ``text`` has no mirror/reflection
    trigger, returns unchanged. Sentence boundary = ``.`` / ``!`` /
    ``?`` followed by whitespace or end-of-string.
    """
    if not text or not _MIRROR_SENTENCE_TRIGGER.search(text):
        return text, False
    # Split keeping sentence-ending punctuation on each sentence.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept: list[str] = []
    for s in sentences:
        if _MIRROR_SENTENCE_TRIGGER.search(s):
            continue
        kept.append(s)
    cleaned = " ".join(kept).strip()
    return cleaned, (cleaned != text)


def sanitize_grid_phrases(text: str) -> tuple[str, bool]:
    """Strip multi-subject / grid phrases and orphan-connector words
    that historically caused the scene_021-class 4-panel-collage
    failure mode.

    Four-stage cleanup:
      0. ``_drop_mirror_sentences`` — drop entire sentences containing
         mirror/reflection language BEFORE the bare-noun strip runs.
         Avoids dangling-syntax artifacts (`kneels before her`).
      1. ``_MULTI_SUBJECT_PATTERN.sub`` removes whole grid phrases
         (``varying compositions``, ``across the series``,
         ``in natural poses``, etc.) and any residual mirror tokens
         that survived stage 0 (defense in depth).
      2. ``ORPHAN_CONNECTOR_PATTERN.sub`` removes dangling connector
         words left after stage 1 (``across`` / ``in`` / ``throughout``
         / ``with`` immediately followed by punctuation).
      3. Whitespace + punctuation normalisation — collapses runs of
         spaces, removes orphan spaces before punctuation (``"foo ."``
         → ``"foo."``), collapses double-punctuation (``"foo .."`` →
         ``"foo."``), strips leading / trailing commas + dots.

    Returns ``(cleaned, changed)`` — the caller can log / re-prompt
    based on whether the LLM emitted offending content.
    """
    if not text:
        return text, False
    # Stage 0 — sentence-level mirror/reflection drop.
    cleaned, _ = _drop_mirror_sentences(text)
    # Stage 1 — bare-noun + multi-subject surgical strip.
    cleaned = _MULTI_SUBJECT_PATTERN.sub("", cleaned)
    cleaned = ORPHAN_CONNECTOR_PATTERN.sub("", cleaned)
    # Round-5 — dangling article cleanup after bare-noun strip.
    # Run TWICE because the first pass can leave new dangling articles
    # (e.g. "A polyptych of the X" → "A  of the X" → "the X" → "X").
    for _ in range(2):
        cleaned = DANGLING_ARTICLE_PATTERN.sub("", cleaned)
        cleaned = DANGLING_ARTICLE_AT_END.sub("", cleaned)
    # Whitespace + punctuation hygiene — keep last so the regexes above
    # don't have to worry about consuming surrounding whitespace.
    cleaned = re.sub(r"\s+([.,])", r"\1", cleaned)  # "foo ." → "foo."
    cleaned = re.sub(r"([.,])\1+", r"\1", cleaned)  # "foo.." → "foo."
    cleaned = re.sub(r",\s*,+", ",", cleaned)       # ", ," → ","
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,.")
    return cleaned, (cleaned != text)


class PromptBuilderError(Exception):
    """Bad inputs to the prompt builder."""


class PromptBuilder:
    """Family-aware prompt composer."""

    def __init__(
        self,
        scene_field_order: Iterable[str] = _SCENE_FIELD_ORDER,
    ) -> None:
        self.scene_field_order: tuple[str, ...] = tuple(scene_field_order)

    # ── public API ──────────────────────────────────────────────────
    def assemble_negative_prompt(
        self,
        *,
        model_negative: str | None = None,
        style_negative: str | None = None,
        character_negative: str | None = None,
        supports_negative: bool = True,
        negative_embeddings: Iterable[str] | None = None,
        model_negative_axes: dict[str, list[str]] | None = None,
        conflict_terms: Iterable[str] | None = None,
        family: FamilyConfig | None = None,
    ) -> str:
        """5-layer negative prompt — TI embeddings + HARD_BLOCK + model + style + character.

        Returns ``""`` when ``supports_negative`` is False (e.g. Flux).

        Textual-inversion ``embedding:Name`` (or ``embedding:Name:0.8``)
        tokens are hoisted to the front so ComfyUI's position-weighted
        encoder gives them their full weight; they survive the regular
        keyword dedup unchanged (identity match) so a typo'd weight
        suffix doesn't merge two embeddings into one.

        After the TI block, ``HARD_BLOCK_NEGATIVE`` ensures age-ambiguity
        vocabulary gates every render; the three caller-provided layers
        follow. Overlapping non-TI tokens are deduped case-insensitively.

        ``model_negative_axes`` (Phase D) — when supplied, the model
        layer is supplied as a 7-axis dict instead of a flat string.
        The dict goes through ``filter_conflicts`` against the positive-
        side ``conflict_terms`` to drop axis tokens that collide with
        what the positive prompt is already saying (the classic
        ``negative: "naked"`` vs. ``positive: "nude pose"`` foot-gun);
        dropped tokens are logged at WARNING. ``model_negative`` (the
        legacy flat string) still wins when both are passed — callers
        opt into the axes path explicitly.

        ``family`` (Phase 3a) — when supplied, the assembled negative
        is trimmed to ``family.max_tokens`` via ``fit_to_budget`` so it
        never silently overflows the encoder window. The TI block is
        kept whole (TI tokens are short and high-priority); only the
        keyword block is trimmed. Pass ``None`` to skip budget enforcement
        (back-compat for tests / direct callers).
        """
        if not supports_negative:
            return ""

        ti_block = _dedup_embeddings(negative_embeddings)

        # Resolve the model layer: prefer the legacy flat string when
        # supplied (back-compat for callers not yet on axes); otherwise
        # filter axes against conflict_terms and flatten.
        if model_negative is not None:
            resolved_model = model_negative
        elif model_negative_axes:
            filtered, dropped = filter_conflicts(
                model_negative_axes, conflict_terms,
            )
            if dropped:
                logger.warning(
                    "negative-axis conflict filter dropped %d token(s) "
                    "colliding with positive prompt: %s",
                    len(dropped), dropped,
                )
            resolved_model = flatten_axes(filtered)
        else:
            resolved_model = None

        # Round-3 verifier — family-conditional HARD_BLOCK. Tight-budget
        # families (SDXL/Pony/Illustrious) get the compact composition
        # block (grid + mirror only); big-budget families (Chroma/Flux/
        # Flux2) get the full 6-token composition block. Caller
        # anatomy negatives survive the budget on tight-budget families.
        hard_block = _resolve_hard_block(family)
        segments = [hard_block]
        segments += [
            s for s in (resolved_model, style_negative, character_negative) if s
        ]
        # Strip any TI tokens that snuck into the keyword segments — they
        # already live in ``ti_block`` and the regular dedup would merge
        # ``embedding:Foo`` with ``embedding:Foo:0.8`` if it ran on them.
        segments = [_strip_ti_tokens(s) for s in segments]
        keyword_block = _keyword_dedup(segments)

        # Phase 3a — token budget. Trim only the keyword block; the TI
        # block is left whole (TI tokens are 1-2 tokens each and carry
        # high signal).
        if family is not None and keyword_block:
            pre_count = count_tokens(keyword_block, family.tokenizer_id)
            keyword_block = fit_to_budget(
                keyword_block,
                max_tokens=family.max_tokens,
                tokenizer_id=family.tokenizer_id,
                break_marker=None,
            )
            post_count = count_tokens(keyword_block, family.tokenizer_id)
            if post_count < pre_count:
                logger.warning(
                    "negative prompt trimmed to fit %d-token budget "
                    "(%d → %d tokens) for family=%r",
                    family.max_tokens, pre_count, post_count, family.id,
                )

        if ti_block and keyword_block:
            return f"{ti_block}, {keyword_block}"
        return ti_block or keyword_block

    def build_one(
        self,
        character: Mapping[str, Any],
        scene: Mapping[str, Any],
        style_profile: Mapping[str, Any] | Any,
        extra_keywords: Iterable[str] | None = None,
        *,
        family: FamilyConfig,
        trigger_words: Iterable[str] | None = None,
        negative_prompt_override: str | None = None,
        avoid_words: Iterable[str] | None = None,
        content_level: str | None = None,
        series_plan: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build one prompt dict from a character/scene/profile triple.

        ``family`` drives composer selection, quality prefix/suffix, and
        the base ``avoid_words`` strip. Callers typically get the family
        from ``ModelRegistryLoader.get_family(entry.family)``.

        ``avoid_words`` replaces ``family.avoid_words`` when supplied —
        callers pass the merged ``ModelPromptGuide.avoid_words`` so
        per-model ``prompt.extend.avoid_words`` actually reach the
        composer.

        ``content_level`` (Phase 4a) — when supplied, NSFW concept tags
        in the scene facet are tier-gated through the canonicalizer; a
        T2_implied scene cannot leak T3_artnude vocabulary even if the
        LLM emitted ``nsfw_anatomy=NSFW_BREAST_NATURAL``. None means
        "skip NSFW concepts entirely" (defence-in-depth).

        ``series_plan`` (Phase 3) — when supplied, the series-level
        aesthetic anchors (``color_palette``, ``photographer_ref``,
        ``art_movement``) are canonicalized via
        :func:`canonicalize_series_aesthetic` and prepended to the
        prompt body. These represent the series's "signature look"
        — chosen once per series by SeriesPlanner and held constant
        across every scene. ``None`` is back-compat for older series
        without aesthetic anchors (they render unchanged).

        Family-specific scene fields are consumed as primary signal when
        present — ``booru_tags`` (pony/illustrious), ``scene_prose``
        (flux/chroma/illustrious), ``camera_spec`` + ``clothing``
        (sdxl). When absent, the composer falls back to the universal
        comma-joined assembly.
        """
        base_prompt = self._field(character, "base_prompt")
        if not base_prompt:
            raise PromptBuilderError(
                f"character is missing base_prompt: {dict(character)!r}"
            )

        # Phase 3 (vocab v6) — series-level aesthetic anchors. Canonicalize
        # the series's color_palette / photographer_ref / art_movement
        # once and prepend to the prompt body. Empty when series_plan
        # missing or has no aesthetic fields (back-compat).
        series_aesthetic_phrases = canonicalize_series_aesthetic(
            series_plan, family.id, content_level=content_level,
        )

        # Phase 4a — canonicalize abstract enum-tag fields on the scene
        # facet into family-shaped phrases. Canonicalizer drops:
        #   * unknown concept tags (LLM drift),
        #   * Pony-omitted namespaces (camera/lens/film_stock/art_style),
        #   * NSFW concepts gated above the active content_level.
        vocab_phrases = canonicalize_facet(
            scene, family.id, content_level=content_level,
        )

        segments: list[str] = [base_prompt]

        # Phase 3 (vocab v6) — series-aesthetic phrases land RIGHT AFTER
        # base_prompt and BEFORE scene fields, so the visual world is
        # established before per-scene details fill it in. Per verifier
        # B4/B7: for prose families the order is base → series_aesthetic
        # → scene_fields → scene_vocab → style. For Pony the prose-shape
        # phrases get filtered (no photographer/art_movement phrasings
        # exist for Pony; only color_palette has a Pony tag form).
        #
        # Round-22 (2026-05-22) — for prose families (flux_natural /
        # flux2_prose) consolidate the 2-3 series-aesthetic phrases
        # into ONE comma-joined sentence. Saves ~60-80 tokens per
        # prompt vs the prior 3-separate-sentences form, and reads as
        # natural cinematic style direction rather than three parallel
        # bullet sentences. Booru families (pony / illustrious) keep
        # the per-phrase segment form — the booru composer's keyword
        # dedup expects atomic items, not a comma-joined sentence.
        # SDXL keeps per-phrase too (CLIP 77-token budget benefits
        # from atomic items the keyword composer can re-order/dedup).
        #
        # NB the consolidation must be applied to BOTH ``segments``
        # (used by sdxl_keywords + pony_danbooru + illustrious_tags
        # composers) AND ``merged_extras`` further down (used by the
        # prose composer ``_compose_natural`` / ``_compose_flux2_prose``
        # which IGNORES ``segments`` when ``scene_prose`` is populated
        # and threads ``extra_keywords`` as trailing sentences). The
        # ``series_aesthetic_for_extras`` variable holds the form to
        # use downstream.
        if (
            family.prompt_style in ("flux_natural", "flux2_prose")
            and len(series_aesthetic_phrases) >= 2
        ):
            merged = ", ".join(p.rstrip(",. ") for p in series_aesthetic_phrases)
            segments.append(merged)
            series_aesthetic_for_extras: list[str] = [merged]
        else:
            for phrase in series_aesthetic_phrases:
                segments.append(phrase)
            series_aesthetic_for_extras = list(series_aesthetic_phrases)

        for field in self.scene_field_order:
            value = self._field(scene, field)
            if value:
                segments.append(value)

        # SDXL primary fields — camera_spec + clothing append as
        # additional segments so the keyword composer naturally absorbs
        # them alongside the scene fields.
        if family.prompt_style == "sdxl_keywords":
            for field in ("camera_spec", "clothing"):
                value = self._field(scene, field)
                if value:
                    segments.append(value)

        # Phase 4a — vocabulary phrases land BEFORE style_keywords so
        # the realism/lighting/mood phrasing sits next to the scene
        # body, ahead of the aesthetic style boosters.
        for phrase in vocab_phrases:
            segments.append(phrase)

        # Round-21 (2026-05-21) — suppress operator's archetype keywords
        # when the planner picked its own aesthetic anchors (color_palette
        # / photographer_ref / art_movement). The series-aesthetic phrases
        # appended above already convey the chosen visual world; the
        # archetype's ``base_style_keywords`` would inject contradictory
        # vocabulary (e.g. ``golden_hour_natural``'s "Golden hour, natural
        # outdoor" inside a planner-chosen neon-noir series). See
        # :func:`archetype_overridden_by_planner` for the contract.
        #
        # 2026-05-23 dual-write pivot — for prose families, ALSO suppress
        # archetype style_keywords unconditionally. The LLM's scene_prose
        # weaves the style itself; archetype tag-soup like "low-key
        # lighting, chiaroscuro, warm shadow, deep blacks, 85mm, f/1.4,
        # film grain, editorial grade" would land AFTER scene_prose and
        # re-introduce the tag-soup problem the pivot exists to fix.
        archetype_overridden = archetype_overridden_by_planner(series_plan)
        is_prose_family = family.prompt_style in ("flux_natural", "flux2_prose")
        if archetype_overridden or is_prose_family:
            style_keywords = ""
            reason = (
                "archetype keywords suppressed for prose family (dual-write "
                "contract — scene_prose covers style)"
                if is_prose_family
                else "archetype keywords suppressed — planner provided aesthetic "
                "anchors (color_palette/photographer_ref/art_movement) for "
                "series, archetype style_keywords would contradict."
            )
            logger.debug(reason)
        else:
            style_keywords = self._field(style_profile, "base_style_keywords")
            if style_keywords:
                segments.append(style_keywords)

        # Combine caller-supplied extra_keywords with vocab phrases for
        # the prose-composer path (which reads extra_keywords as its
        # own kwarg). Series-aesthetic + scene vocab phrases lead —
        # they're more specific than caller extras. Round-22 —
        # series_aesthetic_for_extras carries the consolidated form
        # for prose families (1 merged sentence) or the 2-3 individual
        # phrases otherwise; see the consolidation branch above.
        #
        # 2026-05-23 dual-write pivot — for prose families (flux_natural /
        # flux2_prose), the LLM's scene_prose IS the prompt body and
        # already weaves in all axes (subject + pose + anatomy + light
        # + env + mood + style). The per-axis canonicalizations
        # (vocab_phrases) and the consolidated series-aesthetic phrase
        # would land as parallel sentences AFTER scene_prose,
        # producing the tag-soup output Grok + Claude web independently
        # flagged on series_753f4daae5f2 (0/24 scoring ≥8). Drop both
        # for prose families. Keyword families (sdxl/pony/illustrious)
        # are unaffected — CLIP rewards comma-tag stacking, and
        # `_compose_natural` is only called for flux_natural here.
        is_prose_family_drop = family.prompt_style in ("flux_natural", "flux2_prose")
        if is_prose_family_drop:
            # Reset segments to just base_prompt + scene_core fields
            # (pose, lighting tag, etc.) for prose families. vocab_phrases
            # already added to segments above (line 796) — strip them.
            # Practically: rebuild segments without the vocab_phrases tail.
            # Simpler: keep segments unchanged (sdxl_keywords still uses
            # it), but pass an EMPTY merged_extras to the prose composer.
            # _compose_natural ignores `segments` when scene_prose is
            # populated — it builds from prose_segments using extra_keywords.
            # So passing empty extra_keywords cleanly drops the tag soup.
            merged_extras: list[str] = []
            for kw in extra_keywords or ():
                if kw:
                    segments.append(str(kw))  # sdxl path
        else:
            merged_extras = list(series_aesthetic_for_extras) + list(vocab_phrases)
            for kw in extra_keywords or ():
                if kw:
                    merged_extras.append(str(kw))
                    segments.append(str(kw))

        prompt_text = self._dispatch(
            family=family,
            scene=scene,
            segments=segments,
            trigger_words=trigger_words,
            avoid_words=avoid_words,
            base_prompt=base_prompt,
            style_keywords=style_keywords,
            extra_keywords=merged_extras,
        )

        # Positive-side age safety scan — every family. Strips any
        # age-ambiguity terms the LLM may have slipped into the body
        # and prepends an adult anchor when a match is found.
        prompt_text = _positive_age_safety_scan(prompt_text, family)

        # Positive-side subject-count scan (2026-05-17) — every family.
        # Strips any multi-subject vocabulary the LLM slipped through
        # (``her partner``, ``two women``, ``2girls``, ``partnered
        # intimate``) and logs at ERROR. Then unconditionally prepends
        # the family's ``solo_anchor`` so the encoder sees a strong
        # single-subject signal on EVERY render. For booru families
        # with a ``BREAK`` marker (Pony), the anchor lands AFTER BREAK
        # in CLIP window 2 alongside the body, not in window 1 alongside
        # the score prefix.
        prompt_text = _positive_subject_count_scan(prompt_text, family)
        prompt_text = _positive_solo_anchor_inject(prompt_text, family)

        # Phase C — real-tokenizer trim. Prior to Phase C this was
        # ``_warn_if_over_budget`` (estimate words×1.3, log only). CLIP
        # silently truncates at 77 / T5 at 256/512 — and the truncation
        # always lops off the *tail*, where camera/lens/quality tokens
        # live. ``fit_to_budget`` trims from the *middle* with the actual
        # tokenizer, preserving the subject prefix and quality suffix.
        # Pony's BREAK marker (when set on the family) splits the prompt
        # into two independent CLIP windows.
        prompt_text = fit_to_budget(
            prompt_text,
            max_tokens=family.max_tokens,
            tokenizer_id=family.tokenizer_id,
            break_marker=family.break_marker,
        )
        _warn_if_post_trim_truncated(family, prompt_text)
        # 2026-05-23 (Verifier NC7) — defensive trailing-period ensure
        # for prose families. fit_to_budget's piece-pack can drop the
        # trailing period when trimming at separator boundaries; the
        # missing period leaves the prompt looking unfinished and
        # subtly degrades T5's parse. SDXL keyword family doesn't
        # need the period (comma-joined), so skip there.
        if (
            family.prompt_style in {"flux_natural", "flux2_prose"}
            and prompt_text
            and not prompt_text.rstrip().endswith((".", "!", "?"))
        ):
            prompt_text = prompt_text.rstrip() + "."

        if negative_prompt_override is not None:
            negative_prompt = negative_prompt_override
        else:
            negative_prompt = (
                self._field(character, "negative_prompt")
                or self._field(style_profile, "base_negative_prompt")
                or ""
            )

        return {
            "prompt_text": prompt_text,
            "negative_prompt": negative_prompt,
            "prompt_hash": _hash(prompt_text),
        }

    def build(
        self,
        character: Mapping[str, Any],
        scenes: list[Mapping[str, Any]],
        style_profile: Mapping[str, Any] | Any,
        extra_keywords: Iterable[str] | None = None,
        *,
        family: FamilyConfig,
        trigger_words: Iterable[str] | None = None,
        series_plan: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Vectorized :meth:`build_one`."""
        return [
            self.build_one(
                character, scene, style_profile, extra_keywords,
                family=family, trigger_words=trigger_words,
                series_plan=series_plan,
            )
            for scene in scenes
        ]

    # ── mode-specific helpers ──────────────────────────────────────
    # series_plan threaded through every mode-helper so a future re-wire
    # can't silently lose Phase 3 aesthetic anchors (color_palette /
    # photographer_ref / art_movement). Currently these helpers are
    # unused (engine.py calls build_one directly) — kept for API
    # completeness.

    def build_theme_prompt(
        self,
        series_plan: Mapping[str, Any],
        scene: Mapping[str, Any],
        style_profile: Mapping[str, Any] | Any,
        *,
        family: FamilyConfig,
        trigger_words: Iterable[str] | None = None,
        negative_prompt_override: str | None = None,
    ) -> dict[str, Any]:
        subject = (
            self._field(scene, "subject_detail")
            or self._field(series_plan, "subject_description")
            or ""
        )
        synthetic = {
            "base_prompt": subject,
            "negative_prompt": self._field(style_profile, "base_negative_prompt"),
        }
        return self.build_one(
            synthetic, scene, style_profile,
            family=family,
            trigger_words=trigger_words,
            negative_prompt_override=negative_prompt_override,
            series_plan=series_plan,
        )

    def build_style_prompt(
        self,
        series_plan: Mapping[str, Any],
        scene: Mapping[str, Any],
        style_profile: Mapping[str, Any] | Any,
        *,
        family: FamilyConfig,
        trigger_words: Iterable[str] | None = None,
        negative_prompt_override: str | None = None,
    ) -> dict[str, Any]:
        style_kw = self._field(series_plan, "style_keywords")
        subject = self._field(scene, "subject_detail") or ""
        base = (
            f"{style_kw}, {subject}"
            if style_kw and subject
            else (style_kw or subject)
        )
        synthetic = {
            "base_prompt": base,
            "negative_prompt": self._field(style_profile, "base_negative_prompt"),
        }
        return self.build_one(
            synthetic, scene, style_profile,
            family=family,
            trigger_words=trigger_words,
            negative_prompt_override=negative_prompt_override,
            series_plan=series_plan,
        )

    def build_niche_prompt(
        self,
        series_plan: Mapping[str, Any],
        scene: Mapping[str, Any],
        style_profile: Mapping[str, Any] | Any,
        *,
        family: FamilyConfig,
        trigger_words: Iterable[str] | None = None,
        negative_prompt_override: str | None = None,
    ) -> dict[str, Any]:
        subject = (
            self._field(scene, "subject_detail")
            or self._field(series_plan, "subject_bias")
            or ""
        )
        synthetic = {
            "base_prompt": subject,
            "negative_prompt": self._field(style_profile, "base_negative_prompt"),
        }
        keywords = (
            series_plan.get("keyword_cluster", [])[:3]
            if isinstance(series_plan.get("keyword_cluster"), list)
            else []
        )
        return self.build_one(
            synthetic, scene, style_profile, extra_keywords=keywords,
            family=family,
            trigger_words=trigger_words,
            negative_prompt_override=negative_prompt_override,
            series_plan=series_plan,
        )

    def build_variation_prompt(
        self,
        base_scene: Mapping[str, Any],
        variation: Mapping[str, Any],
        style_profile: Mapping[str, Any] | Any,
        *,
        family: FamilyConfig,
        trigger_words: Iterable[str] | None = None,
        negative_prompt_override: str | None = None,
        series_plan: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        env = self._field(base_scene, "environment_detail") or "studio"
        synthetic = {
            "base_prompt": env,
            "negative_prompt": self._field(style_profile, "base_negative_prompt"),
        }
        return self.build_one(
            synthetic, variation, style_profile,
            family=family,
            trigger_words=trigger_words,
            negative_prompt_override=negative_prompt_override,
            series_plan=series_plan,
        )

    # ── dispatch + composers ───────────────────────────────────────
    def _dispatch(
        self,
        *,
        family: FamilyConfig,
        scene: Mapping[str, Any],
        segments: list[str],
        trigger_words: Iterable[str] | None,
        avoid_words: Iterable[str] | None = None,
        base_prompt: str = "",
        style_keywords: str = "",
        extra_keywords: Iterable[str] = (),
    ) -> str:
        style = family.prompt_style
        trig = list(trigger_words or [])

        # Belt-and-braces: if the family disables weighting (Flux/Chroma)
        # but the LLM emitted `(tag:1.2)` anyway, strip it. The LLM hint
        # forbids weighting; this catches drift.
        if not family.supports_weighting:
            segments = [_strip_weighting(s) for s in segments]
            trig = [_strip_weighting(t) for t in trig]

        effective_avoid = (
            list(avoid_words) if avoid_words is not None
            else list(family.avoid_words)
        )
        avoid = [a.lower() for a in effective_avoid]

        if style == "sdxl_keywords":
            return _compose_keywords(
                segments, trigger_words=trig, avoid=avoid,
            )
        if style == "pony_danbooru":
            return _compose_pony_danbooru(
                segments, trigger_words=trig, avoid=avoid,
                quality_prefix=family.quality_prefix,
                structure_intro=family.structure_intro,
                booru_tags=self._field(scene, "booru_tags"),
                source_tag=self._field(scene, "source_tag"),
                base_prompt=base_prompt,
                style_keywords=style_keywords,
                extra_keywords=list(extra_keywords),
            )
        if style == "illustrious_tags":
            return _compose_illustrious_tags(
                segments, trigger_words=trig, avoid=avoid,
                quality_suffix=family.quality_suffix,
                booru_tags=self._field(scene, "booru_tags"),
                scene_prose=self._field(scene, "scene_prose"),
                base_prompt=base_prompt,
                style_keywords=style_keywords,
                extra_keywords=list(extra_keywords),
            )
        if style == "flux_natural":
            # Round-22 — pass facet_has_lens so the chroma realism tail
            # can drop its hardcoded "f/1.8, 35mm" tokens when the LLM
            # already picked a per-scene realism_lens (otherwise two
            # focal lengths land in the same prompt).
            return _compose_natural(
                segments, trigger_words=trig, avoid=avoid,
                scene_prose=self._field(scene, "scene_prose"),
                base_prompt=base_prompt,
                style_keywords=style_keywords,
                extra_keywords=list(extra_keywords),
                realism_tail_style=family.realism_tail_style,
                facet_has_lens=bool(self._field(scene, "realism_lens")),
            )
        if style == "flux2_prose":
            return _compose_flux2_prose(
                segments, trigger_words=trig, avoid=avoid,
                scene_prose=self._field(scene, "scene_prose"),
                base_prompt=base_prompt,
                style_keywords=style_keywords,
                extra_keywords=list(extra_keywords),
            )
        raise PromptBuilderError(
            f"unknown prompt_style {style!r} on family {family.id!r}"
        )

    # ── helpers ────────────────────────────────────────────────────
    @staticmethod
    def _field(obj: Any, name: str) -> str:
        if obj is None:
            return ""
        if isinstance(obj, Mapping):
            value = obj.get(name)
        else:
            value = getattr(obj, name, None)
        if value is None:
            return ""
        return str(value).strip()


# ── composers (module-level; pure functions) ───────────────────────

def _compose_keywords(
    segments: Iterable[str],
    *,
    trigger_words: Iterable[str] | None = None,
    avoid: Iterable[str] = (),
) -> str:
    """Comma-join segments, split + dedup token-by-token, strip avoid_words.

    Lowercase compared for dedup / avoid, but output preserves the
    first-seen casing so e.g. "85mm lens" stays as written.
    """
    avoid_set = {a.lower() for a in avoid}
    seen: set[str] = set()
    out: list[str] = []
    for segment in segments:
        for raw in segment.split(","):
            token = raw.strip()
            if not token:
                continue
            key = token.lower()
            if key in avoid_set or key in seen:
                continue
            seen.add(key)
            out.append(token)
    for tw in trigger_words or ():
        token = tw.strip()
        if not token:
            continue
        key = token.lower()
        if key in avoid_set or key in seen:
            continue
        seen.add(key)
        out.append(token)
    return ", ".join(out)


def _compose_pony_danbooru(
    segments: Iterable[str],
    *,
    trigger_words: Iterable[str] | None = None,
    avoid: Iterable[str] = (),
    quality_prefix: Iterable[str] = (),
    structure_intro: Iterable[str] = (),
    booru_tags: str = "",
    source_tag: str = "",
    base_prompt: str = "",
    style_keywords: str = "",
    extra_keywords: Iterable[str] = (),
) -> str:
    """Prepend family's quality prefix (incl. BREAK) then keyword-compose.

    When the scene carries ``booru_tags`` (the LLM's Pony-tuned primary
    field), use them as the body instead of the universal scene-field
    assembly — booru tags are Pony's native token shape and beat
    English comma-lists on this family. Character identity and the
    style-profile keywords still lead/trail so the Pony native
    composition is pre/post-fixed with context.

    Pony V6 requires a ``source_*`` tag; we inject ``source_photograph``
    as a realism default when the scene didn't provide one explicitly
    and none is already present in the body.

    ``structure_intro`` (Phase 3) — comma-tokens emitted between the
    quality_prefix tail and the body. Realism Pony finetunes use this
    to pin ``[source_photo, "photo (medium)", realistic]`` after the
    score chain. Empty for base Pony / non-realism finetunes.
    """
    prefix = [t for t in quality_prefix if t and t.strip()]
    intro = [t for t in structure_intro if t and t.strip()]

    if booru_tags:
        # Primary path — the LLM produced booru-native tags. Order the
        # body as: [character identity] + [booru tags] + [vocab phrases]
        # + [style keywords]. Vocab phrases (Phase 4a — lighting + mood
        # + NSFW translations) are threaded BEFORE style keywords so
        # the realism / lighting tokens sit close to the booru body.
        body_segments: list[str] = []
        if base_prompt:
            body_segments.append(base_prompt)
        body_segments.append(booru_tags)
        for phrase in extra_keywords or ():
            if phrase:
                body_segments.append(str(phrase))
        if style_keywords:
            body_segments.append(style_keywords)
        body = _compose_keywords(
            body_segments, trigger_words=trigger_words, avoid=avoid,
        )
    else:
        body = _compose_keywords(
            segments, trigger_words=trigger_words, avoid=avoid,
        )

    # Phase 5 audit fix: a Pony realism finetune declares
    # ``structure_intro: [source_photograph, "photo (medium)", realistic]``,
    # which already pins the source tag. _ensure_pony_source_tag would
    # then prepend a second ``source_photograph`` to the body, producing
    # a duplicate token. Skip the helper when intro carries any
    # ``source_*`` token; otherwise the helper still preserves
    # per-call source_tag selection for vanilla Pony YAMLs (no intro).
    intro_has_source = any(
        t.lower().startswith("source_") for t in intro
    )
    if not intro_has_source:
        body = _ensure_pony_source_tag(body, source_tag)
    # _warn_if_pony_missing_source moved to SceneFacetPony.model_validator
    # in Phase 4b — the validator catches the missing source_tag at LLM
    # output time, before it ever reaches the composer.

    parts = [*prefix, *intro]
    if not parts:
        return body
    return ", ".join([*parts, body]) if body else ", ".join(parts)


def _ensure_pony_source_tag(body: str, source_tag: str) -> str:
    """Inject a source_* tag when neither scene nor body already has one.

    Defaults to ``source_photograph`` — realism Pony checkpoints are
    what this pipeline targets. If the scene provides a source_tag,
    prefer that; otherwise leave an existing source_* in the body
    untouched.
    """
    lower = body.lower()
    if "source_" in lower:
        return body
    picked = source_tag.strip() or "source_photograph"
    return f"{picked}, {body}" if body else picked



def _compose_illustrious_tags(
    segments: Iterable[str],
    *,
    trigger_words: Iterable[str] | None = None,
    avoid: Iterable[str] = (),
    quality_suffix: Iterable[str] = (),
    booru_tags: str = "",
    scene_prose: str = "",
    base_prompt: str = "",
    style_keywords: str = "",
    extra_keywords: Iterable[str] = (),
) -> str:
    """Booru tags + short prose hybrid; append quality suffix at end.

    Illustrious responds to a mix of underscore-joined tags (``1girl``,
    ``looking_at_viewer``) and short phrases. When the scene carries
    ``booru_tags`` *and* ``scene_prose`` (the Illustrious-tuned primary
    fields), the body is ordered as ``[identity, booru tags, prose,
    style]`` — tags first, prose mid, style tail — matching the
    NoobAI/RealDream prompting recipe. Without those primary fields we
    fall back to the universal segment assembly.

    The mandatory quality suffix (``masterpiece, best quality, amazing
    quality, very aesthetic, newest``) is appended last.

    NOTE: the ``llm_hint`` tells the model to emit tags-first, phrases-
    after, but we do not validate that order here. In practice both
    token orders render acceptably for Illustrious — strict validation
    would reject otherwise-usable prompts. The hint is advisory, not a
    composer-enforced contract.
    """
    if booru_tags or scene_prose:
        body_segments: list[str] = []
        if base_prompt:
            body_segments.append(base_prompt)
        if booru_tags:
            body_segments.append(booru_tags)
        if scene_prose:
            body_segments.append(scene_prose)
        # Verifier B3 — pre-fix this composer dropped extra_keywords
        # entirely. The build_one path packs series_aesthetic phrases
        # + scene vocab phrases + caller extras into extra_keywords
        # for prose families; for Illustrious those evaporated. Thread
        # them in BEFORE style_keywords so they sit alongside the
        # booru body, not after the style tail.
        for kw in extra_keywords or ():
            if kw:
                body_segments.append(str(kw))
        if style_keywords:
            body_segments.append(style_keywords)
        body = _compose_keywords(
            body_segments, trigger_words=trigger_words, avoid=avoid,
        )
    else:
        body = _compose_keywords(
            segments, trigger_words=trigger_words, avoid=avoid,
        )

    suffix = [t for t in quality_suffix if t and t.strip()]
    if not suffix:
        return body
    # Suffix tokens go through the same avoid-set filter for safety.
    avoid_set = {a.lower() for a in avoid}
    body_tokens = {t.strip().lower() for t in body.split(",")}
    filtered_suffix = [
        t for t in suffix
        if t.lower() not in avoid_set and t.lower() not in body_tokens
    ]
    if not filtered_suffix:
        return body
    return f"{body}, {', '.join(filtered_suffix)}" if body else ", ".join(filtered_suffix)


# Chroma realism tail — period-separated fragments appended to the
# scene prose to push toward photographic realism. Per Civitai community
# guidance (verified 2026-05-17): "avoid photorealistic — it pushes
# toward photorealistic ART; use photo or photography for real photos."
# https://civitai.com/articles/19951 (Chroma Guide); lodestones/Chroma1-HD
# HF README confirms pure-prose, photo-oriented terms.
_CHROMA_REALISM_TAIL = (
    "f/1.8",
    "35mm",
    "photographic",
    "natural skin texture",
)


def _compose_natural(
    segments: Iterable[str],
    *,
    trigger_words: Iterable[str] | None = None,
    avoid: Iterable[str] = (),
    scene_prose: str = "",
    base_prompt: str = "",
    style_keywords: str = "",
    extra_keywords: Iterable[str] = (),
    realism_tail_style: str | None = None,
    facet_has_lens: bool = False,
) -> str:
    """Join segments as flowing sentences for Flux / Chroma models.

    When the scene carries ``scene_prose`` (the flux_natural primary
    field), we use it as the body — prefixed with the character's
    base_prompt as a short "subject" sentence and followed by
    style_keywords + extra_keywords woven in as trailing detail
    sentences. This honours the LLM's prose-native output instead of
    re-chopping it into the universal comma-list.

    When ``realism_tail_style == "period"`` (Chroma), the realism
    tail (``f/1.8. 35mm. photographic. natural skin texture.``) is
    appended as period-separated fragments after the body — the shape
    the lodestones HF card and top Civitai Chroma workflows use.
    Note: "photographic" not "photorealistic" — civitai community
    guidance says photorealistic pushes toward photorealistic ART; use
    photographic / photography for true-photo aesthetics.

    Tokens matching ``avoid`` are filtered from each segment before the
    segment is capitalized and stitched. Trigger words are appended as
    a trailing sentence.
    """
    avoid_set = {a.lower() for a in avoid}

    if scene_prose:
        prose_segments: list[str] = []
        if base_prompt:
            prose_segments.append(base_prompt)
        prose_segments.append(scene_prose)
        if style_keywords:
            prose_segments.append(style_keywords)
        for kw in extra_keywords or ():
            if kw:
                prose_segments.append(str(kw))
        iter_segments: Iterable[str] = prose_segments
    else:
        iter_segments = segments

    parts: list[str] = []
    for seg in iter_segments:
        cleaned = _strip_avoid_tokens(seg, avoid_set).rstrip(",. ")
        if not cleaned:
            continue
        if cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]
        parts.append(cleaned)
    text = ". ".join(parts)
    if text and not text.endswith("."):
        text += "."

    if realism_tail_style == "period":
        # Round-22 (2026-05-22) — drop the focal-spec tokens
        # ("f/1.8", "35mm") from the tail when the LLM already picked
        # a per-scene realism_lens. Otherwise the tail's hardcoded
        # focal length contradicts the per-scene lens canonicalization
        # (e.g. "85mm f/1.4 lens, shallow DoF" + "f/1.8, 35mm" in the
        # same prompt). When the facet has NO lens populated, the tail
        # still supplies the focal hint — preserves backwards-compat
        # for the optional-lens path.
        if facet_has_lens:
            candidate_tail = tuple(
                frag for frag in _CHROMA_REALISM_TAIL
                if frag not in ("f/1.8", "35mm")
            )
        else:
            candidate_tail = _CHROMA_REALISM_TAIL
        tail_fragments = [
            frag for frag in candidate_tail
            if frag.lower() not in avoid_set
            and frag.lower() not in text.lower()
        ]
        if tail_fragments:
            # Round-21 — comma-joined single trailing sentence.
            tail = ", ".join(tail_fragments) + "."
            text = f"{text} {tail}" if text else tail

    triggers = [
        t.strip() for t in (trigger_words or ())
        if t and t.strip() and t.strip().lower() not in avoid_set
    ]
    if triggers:
        text += f" {', '.join(triggers)}."
    return text


# BFL Klein 9B guide target band — 30–80 words "medium production"
# length. Shorter loses specificity; longer saturates Qwen3 attention.
# We warn (not fail) outside 20–100 so clean prompts of 28 or 82 words
# don't get flagged, but systemic drift does.
_FLUX2_WORD_COUNT_LO = 20
_FLUX2_WORD_COUNT_HI = 100


def _compose_flux2_prose(
    segments: Iterable[str],
    *,
    trigger_words: Iterable[str] | None = None,
    avoid: Iterable[str] = (),
    scene_prose: str = "",
    base_prompt: str = "",
    style_keywords: str = "",
    extra_keywords: Iterable[str] = (),
) -> str:
    """FLUX.2 Klein 9B composer — BFL 5-anchor prose, no realism tail.

    Delegates the prose stitching to ``_compose_natural`` with
    ``realism_tail_style=None`` — Klein's distilled contract bans the
    camera/lens tail that Chroma appends. Everything else matches the
    flux_natural path: ``scene_prose`` becomes the body when present,
    otherwise the universal segment assembly is used.

    Emits an INFO log when the composed text falls outside the
    BFL-recommended 30–80 word band (with a 10-word slack each way) so
    systemic LLM drift surfaces without false alarms on clean prompts.
    """
    text = _compose_natural(
        segments,
        trigger_words=trigger_words,
        avoid=avoid,
        scene_prose=scene_prose,
        base_prompt=base_prompt,
        style_keywords=style_keywords,
        extra_keywords=extra_keywords,
        realism_tail_style=None,
    )
    _log_flux2_word_count(text)
    return text


def _log_flux2_word_count(text: str) -> None:
    if not text:
        return
    wc = len(text.split())
    if wc < _FLUX2_WORD_COUNT_LO:
        logger.info(
            "flux2 prompt is %d words — below BFL 30–80 band (preview: %r)",
            wc, text[:80] + ("…" if len(text) > 80 else ""),
        )
    elif wc > _FLUX2_WORD_COUNT_HI:
        logger.info(
            "flux2 prompt is %d words — above BFL 30–80 band (preview: %r)",
            wc, text[:80] + ("…" if len(text) > 80 else ""),
        )


def _warn_if_post_trim_truncated(family: FamilyConfig, prompt_text: str) -> None:
    """Warn only when ``fit_to_budget`` *still* leaves us over budget.

    With the real tokenizer doing the trim this should be rare — only
    happens when a single token (e.g. a long character name without a
    natural boundary) blows the budget. Logging lets the operator catch
    that drift without raising.

    Honors ``family.break_marker`` (Pony's ``BREAK``) — each side of the
    marker is its own encoder window with its own budget, so the warn
    check splits the prompt before counting.
    """
    if not family.max_tokens or not prompt_text:
        return
    # Match ``fit_to_budget``'s reservation of 2 tokens for BOS/EOS.
    budget = max(0, family.max_tokens - 2)
    if family.break_marker and family.break_marker in prompt_text:
        windows = prompt_text.split(family.break_marker)
    else:
        windows = [prompt_text]
    for idx, window in enumerate(windows):
        actual = count_tokens(window.strip(", "), family.tokenizer_id)
        if actual > budget:
            preview = window[:80] + ("…" if len(window) > 80 else "")
            tag = (
                f" window {idx + 1}/{len(windows)}"
                if len(windows) > 1 else ""
            )
            logger.warning(
                "Prompt %d tokens still exceeds family %r budget %d after "
                "fit_to_budget%s — encoder will truncate the tail: %r",
                actual, family.id, budget, tag, preview,
            )


def _strip_weighting(text: str) -> str:
    """Remove ``(tag:1.2)`` weighting syntax, keeping the inner tag."""
    if not text:
        return text
    return _WEIGHTING_PATTERN.sub(r"\1", text)


def _strip_avoid_tokens(segment: str, avoid_set: set[str]) -> str:
    if not avoid_set:
        return segment.strip()
    kept = [
        t.strip() for t in segment.split(",")
        if t.strip() and t.strip().lower() not in avoid_set
    ]
    return ", ".join(kept)


def _positive_age_safety_scan(text: str, family: FamilyConfig) -> str:
    """Belt-and-braces strip of age-ambiguity vocab from positive prompts.

    Runs on every composed prompt regardless of family. When any
    age-ambiguity term slips through the LLM hint + sanitizer:

    1. Log at ERROR with the match set (tracked across runs so drift
       surfaces immediately).
    2. Erase the match from the body.
    3. Prepend an adulthood anchor — comma-joined tokens for
       keyword families (sdxl/pony/illustrious), a sentence for prose
       families (flux/chroma).

    The anchor is only added *when a match was found* — clean prompts
    are returned untouched so byte-level dedup hashes stay stable.
    """
    if not text:
        return text
    matches = _AGE_AMBIGUITY_PATTERN.findall(text)
    if not matches:
        return text

    logger.error(
        "Age-ambiguity terms detected in positive prompt — stripping and "
        "prepending adult anchor. matches=%s family=%r",
        sorted({m.lower() for m in matches}), family.id,
    )

    stripped = _AGE_AMBIGUITY_PATTERN.sub("", text)
    # Collapse artefacts from removal: double commas, double periods,
    # stray whitespace, leading/trailing commas.
    stripped = re.sub(r",\s*,+", ",", stripped)
    stripped = re.sub(r"\.\s*\.+", ".", stripped)
    stripped = re.sub(r"\s{2,}", " ", stripped)
    stripped = stripped.strip(" ,.")

    anchor_keyword = family.adult_anchor["keyword"]
    anchor_prose = family.adult_anchor["prose"]
    if family.prompt_style in {"flux_natural", "flux2_prose"}:
        return f"{anchor_prose} {stripped}".strip() if stripped else anchor_prose
    return f"{anchor_keyword}, {stripped}" if stripped else anchor_keyword


def _positive_subject_count_scan(text: str, family: FamilyConfig) -> str:
    """Strip multi-subject vocabulary from positive prompts.

    Mirrors :func:`_positive_age_safety_scan` but for subject count.
    The pipeline enforces a single-female-only invariant (2026-05-17);
    when Venice or any LLM slips through ``her partner`` /
    ``two women`` / ``partnered intimate`` / ``2girls`` etc. into the
    composed positive prompt, this scan removes them and logs at
    ERROR so drift surfaces immediately.

    The solo_anchor injection in :func:`_positive_solo_anchor_inject`
    runs after this and is unconditional — it ALWAYS prepends the
    family's single-subject signal regardless of whether multi-subject
    vocab was stripped. The scan returns the body with offending
    tokens removed; clean prompts pass through untouched (byte-stable).
    """
    if not text:
        return text
    matches = _MULTI_SUBJECT_PATTERN.findall(text)
    if not matches:
        return text
    logger.error(
        "Multi-subject vocab detected in positive prompt — stripping. "
        "matches=%s family=%r",
        sorted({m.lower() for m in matches}), family.id,
    )
    stripped = _MULTI_SUBJECT_PATTERN.sub("", text)
    stripped = re.sub(r",\s*,+", ",", stripped)
    stripped = re.sub(r"\.\s*\.+", ".", stripped)
    stripped = re.sub(r"\s{2,}", " ", stripped)
    stripped = stripped.strip(" ,.")
    return stripped


def _positive_solo_anchor_inject(text: str, family: FamilyConfig) -> str:
    """Unconditionally inject ``family.solo_anchor`` into the positive
    prompt so the encoder always sees a strong single-subject signal.

    Booru-style and SDXL families (``sdxl_keywords``, ``pony_danbooru``,
    ``illustrious_tags``) receive ``solo_anchor.keyword`` as a comma-
    joined fragment. Prose families (``flux_natural``, ``flux2_prose``)
    receive ``solo_anchor.prose`` as a sentence-prefixed clause.

    For families with a ``BREAK`` marker (Pony), the keyword fragment
    lands AFTER the marker so it sits in CLIP window 2 alongside the
    body — placing it in window 1 next to ``score_9, score_8_up, …``
    would dilute the score prefix.

    Idempotent: if the solo-anchor tokens (case-insensitive) are already
    present in the body, no change is made — byte-stable when the LLM
    already emitted ``solo`` / ``1girl`` itself.
    """
    if not text:
        return text
    anchor_kw = (family.solo_anchor or {}).get("keyword", "")
    anchor_prose = (family.solo_anchor or {}).get("prose", "")

    # Prose families — sentence-prefix.
    if family.prompt_style in {"flux_natural", "flux2_prose"}:
        if not anchor_prose:
            return text
        if anchor_prose.lower() in text.lower():
            return text
        sep = " " if text and not text[0].isspace() else ""
        return f"{anchor_prose}{sep}{text}"

    # Keyword / booru families — comma-joined fragment.
    if not anchor_kw:
        return text
    anchor_tokens = [
        t.strip().lower() for t in anchor_kw.split(",") if t.strip()
    ]
    text_lower = text.lower()
    if anchor_tokens and all(tok in text_lower for tok in anchor_tokens):
        return text  # idempotent — body already carries every solo token

    if family.break_marker and family.break_marker in text:
        head, marker, body = text.partition(family.break_marker)
        body_clean = body.lstrip().lstrip(",").lstrip()
        return f"{head}{marker} {anchor_kw}, {body_clean}"
    return f"{anchor_kw}, {text}" if text else anchor_kw


def _keyword_dedup(segments: Iterable[str]) -> str:
    """Public-style dedup used by ``assemble_negative_prompt`` —
    no family-level avoid filtering (negatives are domain-specific).
    """
    return _compose_keywords(segments)


# Matches a textual-inversion token: ``embedding:Name`` with optional
# ``:weight`` suffix (e.g. ``embedding:BadDream:0.8``). The leading
# whitespace check is anchored at a word boundary; trailing whitespace
# / commas are not consumed so callers can split surrounding text safely.
_TI_TOKEN_PATTERN = re.compile(
    r"\bembedding:[A-Za-z0-9_\-]+(?::\d+(?:\.\d+)?)?",
    re.IGNORECASE,
)


def _dedup_embeddings(tokens: Iterable[str] | None) -> str:
    """Identity-dedup a list of ``embedding:Name`` tokens, comma-joined.

    Identity-only: ``embedding:BadHands`` and ``embedding:BadHands_v1``
    are distinct, and so are ``embedding:Foo`` and ``embedding:Foo:0.8``
    (the weight suffix shifts the strength so they're not the same
    instruction). Empty / blank / non-``embedding:`` entries are dropped
    silently — callers don't have to pre-filter.
    """
    if not tokens:
        return ""
    seen: set[str] = set()
    out: list[str] = []
    for tok in tokens:
        if not tok:
            continue
        clean = tok.strip().rstrip(",").strip()
        if not clean:
            continue
        if not clean.lower().startswith("embedding:"):
            continue
        if clean in seen:
            continue
        seen.add(clean)
        out.append(clean)
    return ", ".join(out)


def _strip_ti_tokens(segment: str | None) -> str:
    """Remove TI tokens from a keyword segment so they don't get
    dedup-merged with each other or with their weighted variants.

    Returns the segment with TI tokens excised and surrounding commas /
    extra whitespace cleaned up. A segment that *only* contained TI
    tokens collapses to ``""``.
    """
    if not segment:
        return ""
    cleaned = _TI_TOKEN_PATTERN.sub("", segment)
    # Collapse double commas and stray whitespace left behind by removal.
    cleaned = re.sub(r",\s*,+", ",", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip(" ,")


def compute_prompt_hash(text: str) -> str:
    """SHA256 of the prompt — matches the ``prompts.prompt_hash`` column.

    Public entry point so callers (engine, render_set) can recompute the
    hash after ``PromptSanitizer.sanitize_text`` rewrites the prompt —
    otherwise dedup misses pairs that only differed in a suppressed word.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Back-compat alias — internal callers still use _hash.
_hash = compute_prompt_hash
