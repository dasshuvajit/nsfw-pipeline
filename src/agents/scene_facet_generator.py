# WARNING: Always call llm_client.unload_model() after all LLM calls
# BEFORE any ComfyUI rendering. LLM and ComfyUI cannot run simultaneously
# on 48GB unified memory.
"""Scene facet generator — per-family LLM expansion of one scene.

A scene's model-agnostic core (pose, camera, lighting, environment,
mood, expression, composition_intent, framing_hint, audience_target)
is produced by :class:`SceneGenerator` and stored in ``scenes``. The
**family-shaped composer inputs** (booru_tags for pony, scene_prose
for flux/chroma/illustrious/flux2, camera_spec + clothing for sdxl)
are produced HERE — one targeted LLM call per ``(scene, family)``,
stored in ``scene_facets`` keyed by ``(scene_id, family)``.

Why split it out:

  * Sibling models in the same family (``juggernaut_ragnarok`` +
    ``gonzalomo_photo_v70``, both sdxl) can share a single facet
    row. The per-model differentiation is in ``prompts`` via
    composer + per-model trigger words.
  * Re-targeting an existing series for a new family is one LLM
    call per scene, not a full re-generation of the scene core.
  * The family-prompting hints (Pony's ``source_tag`` requirement,
    Illustrious's tags-then-prose convention, FLUX.2's BFL 5-anchor
    ordering, per-model trigger / avoid words) live HERE, not in
    SceneGenerator — keeping the model-agnostic generator simple.

See ARCHITECTURE.md Section 9 (LLM Agents) and the Phase A flow in
the per-model prompts plan.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import functools

from pydantic import BaseModel, Field, create_model

from src.agents.llm_client import OllamaClient, OllamaJSONParseError
from src.agents.schemas import SCENE_FACET_SCHEMA_BY_STYLE
from src.prompt.vocabulary import llm_vocabulary_block

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.memory.family_loader import FamilyConfig
    from src.memory.model_registry import ModelPromptGuide


# Tier-required facet fields. The Pydantic schema declares these
# Optional[str] for back-compat and per-family schema variance, so
# constrained decoding + Pydantic alone accept null. We post-validate
# to catch LLM dodges (heretic-tuned or smaller LLMs nulling structured
# fields when uncertain) and trigger the retry loop with an explicit
# nudge.
#
# - ``lighting_directive`` (2026-05-18) — required at every tier
#   across every family schema that carries it (all 5: sdxl, pony,
#   illustrious, flux, chroma, flux2). Single biggest factor in image
#   quality; the canonicalizer translates the enum tag into family-
#   shaped cinematography vocabulary (`Rembrandt lighting, dramatic
#   side-light` vs free-text `dim ambient`). Adding to every tier
#   protects the canonicalizer contract (vocab_version 6 as of
#   2026-05-19) — without this, the LLM can null the field and the
#   canonicalizer becomes dead weight.
# - ``mood_aesthetic`` (added with vocab_version 5, 2026-05-18) —
#   required at every tier. Same rationale as lighting_directive:
#   the canonicalizer injects family-shaped mood phrasing
#   (`MOOD_INTIMATE` → `intimate, tender, unhurried`) that free-text
#   `mood_note` doesn't reliably carry. All 5 schemas have it.
# - ``nsfw_anatomy`` (T3+, added 2026-05-17) — explicit nudity vocab
#   that the T3/T4 llm_directive marks REQUIRED. Booru-family
#   relaxation: native danbooru NSFW tokens inside ``booru_tags``
#   count as equivalent (see ``_booru_tags_carry_nsfw``).
# - ``nsfw_act`` (T4 only, added 2026-05-17) — explicit-act vocab
#   with no booru equivalent (strict check).
_TIER_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    # Phase 2 (vocab v6) — narrative_moment required at EVERY tier
    # (T1+). The narrative anchor solves "she just poses" — every
    # facet gets one moment-tag so the diffusion model knows what's
    # happening in the scene, not just what pose to render.
    "T1_suggestive": ("lighting_directive", "mood_aesthetic", "narrative_moment"),
    "T2_implied":    ("lighting_directive", "mood_aesthetic", "narrative_moment"),
    # Phase 1 (vocab v6) — environment_setting + environment_atmosphere
    # added at T3+ so the validator's retry-nudge forces the LLM to
    # pick a location + atmospheric element per scene (the chief
    # leverage point against "all 24 scenes look like the same room").
    "T3_artnude":    (
        "lighting_directive", "mood_aesthetic", "nsfw_anatomy",
        "environment_setting", "environment_atmosphere", "narrative_moment",
    ),
    "T4_explicit":   (
        "lighting_directive", "mood_aesthetic", "nsfw_anatomy", "nsfw_act",
        "environment_setting", "environment_atmosphere", "narrative_moment",
    ),
}

# Back-compat alias — older callers / tests may still import the
# pre-2026-05-18 name. Keep until callers migrate.
_TIER_REQUIRED_NSFW_FIELDS = _TIER_REQUIRED_FIELDS


# Example concept-tag values inlined into the retry-nudge per field
# so the second attempt's user prompt carries 4-5 concrete examples
# instead of just naming the field. Empirically critical on heretic-
# tuned LLMs that otherwise null structured fields under constrained
# decoding even with the full vocabulary menu in the system prompt.
# Pulled from ``config/prompt_vocabulary.yaml`` — covers the most
# common scene-fitting picks per namespace. The full menu remains in
# the system prompt; these are only nudge examples (the LLM may pick
# any tag from the menu, not just these).
_FIELD_EXAMPLE_TAGS: dict[str, tuple[str, ...]] = {
    "lighting_directive": (
        "LIGHT_GOLDEN_HOUR", "LIGHT_SOFT_FILL", "LIGHT_REMBRANDT",
        "LIGHT_WINDOW_SIDE", "LIGHT_RIM_BACK",
    ),
    "mood_aesthetic": (
        "MOOD_INTIMATE", "MOOD_CONFIDENT", "MOOD_SENSUAL",
        "MOOD_SERENE", "MOOD_PLAYFUL", "MOOD_PENSIVE",
    ),
    "nsfw_anatomy": (
        "NSFW_FULL_NUDE", "NSFW_BREAST_NATURAL",
        "NSFW_NIPPLES_VISIBLE", "NSFW_VULVA_VISIBLE",
    ),
    # Solo-only pipeline (CLAUDE.md invariant) — the four partnered
    # T4 acts (EMBRACE_NUDE / KISS_PASSIONATE / PARTNERED_INTIMATE /
    # AFTERGLOW) are filtered out of the vocab menu by
    # ``_SOLO_MODE_BANNED_TAGS``. The 7 SOLO_* tags below were added
    # with vocab_version 5 (2026-05-18) to give T4 scenes meaningful
    # creative variation (pre-fix the only legal pick was SOLO_TOUCH).
    "nsfw_act": (
        "NSFW_T4_SOLO_TOUCH", "NSFW_T4_SOLO_GAZE",
        "NSFW_T4_SOLO_DISPLAY", "NSFW_T4_SOLO_RECLINING",
        "NSFW_T4_SOLO_BATH",
        "NSFW_T4_SOLO_OUTDOOR", "NSFW_T4_SOLO_PERFORMER",
    ),
    # Phase 1 (vocab v6) — environment-vocab retry-nudge examples.
    # Per verifier I3 (cap inlined examples per missing field at 3
    # to keep retry-nudge payload manageable), each ships the 3 most
    # broadly-applicable picks. The full 40+ menu lives in the
    # system prompt; these are only the retry-nudge anchors.
    "environment_setting": (
        "ENV_MORNING_BEDROOM", "ENV_VICTORIAN_PARLOUR",
        "ENV_MEDITERRANEAN_COURTYARD",
    ),
    "environment_atmosphere": (
        "ATM_DUST_MOTES_IN_LIGHT", "ATM_BREEZE_IN_CURTAIN",
        "ATM_VOLUMETRIC_GOLDEN",
    ),
    # Phase 2 (vocab v6) — 3 narrative-moment retry-nudge anchors
    # spanning common-domestic / dressing / outdoor moods.
    # vocab v7 (2026-05-20) — swapped NARR_MIRROR_CONTEMPLATION
    # (deleted, mirror-rendering failure mode) for
    # NARR_DRESSING_FOR_EVENING.
    "narrative_moment": (
        "NARR_READING_LETTER_AT_DAWN", "NARR_STEPPING_FROM_BATH",
        "NARR_DRESSING_FOR_EVENING",
    ),
}

# Native danbooru NSFW vocabulary — booru families (pony, illustrious)
# may express tier-appropriate NSFW content directly in ``booru_tags``
# rather than via the structured ``nsfw_anatomy`` enum. When any of
# these tokens appear inside booru_tags, treat the missing structured
# anatomy field as satisfied (the booru tag is the equivalent native
# signal for these families). Lowercase, whole-word matched via the
# token-set check in :func:`_booru_tags_carry_nsfw`.
_BOORU_NSFW_TOKENS: frozenset[str] = frozenset({
    "nude", "completely_nude", "fully_nude", "topless", "bare_chest",
    "bare_breasts", "breasts", "nipples", "pussy", "vulva",
    "anatomically_correct", "fine_art_nude",
    # Verifier round-3 NIT-7 (revised round-4 IMPORTANT-4) — natural
    # explicitly-nude booru phrasings that the heuristic at line 205
    # ("nude" substring) didn't catch. Round-3's first pass also
    # included `bare_legs/bare_back/bare_thighs/bare_hips` — but
    # those are T1-T2 territory (uncovered limbs ≠ nudity) and
    # widened the T3 gate on the wrong axis. Round-4 removed them.
    # Also removed non-canonical danbooru tags
    # `partially_nude/topless_back/sheer_lingerie/lingerie_pull_aside`
    # which never matched anyway (danbooru convention is
    # `partially_clothed`, `see-through`, etc.).
    "bottomless", "see-through",
})

# Booru prompt_style values whose facets carry NSFW content natively
# in booru_tags (parallel to ``_BOORU_PROMPT_STYLES`` below — kept as
# a constant here so the post-validator doesn't need a circular import).
_BOORU_NATIVE_STYLES: frozenset[str] = frozenset({
    "pony_danbooru", "illustrious_tags",
})


def _booru_tags_carry_nsfw(facet: dict[str, Any] | None) -> bool:
    """True iff ``facet.booru_tags`` contains any token from
    :data:`_BOORU_NSFW_TOKENS` **or** a lowercased ``nsfw_*`` /
    ``art_nude*`` concept-tag token (observed LLM quirk — see below).
    Whole-token match on the comma-split tag list; case-insensitive.

    Observed LLM quirk: at T3+ for booru families, some LLMs emit
    the abstract concept tag (lowercased) inside ``booru_tags`` —
    e.g. ``nsfw_breast_natural, art_fine_nude, photorealistic`` —
    instead of populating the structured ``nsfw_anatomy`` enum
    field. The composer doesn't translate those tokens (canonicalizer
    only fires on the structured field), so the booru-tag-side
    signal is weaker, but the operator's intent is unambiguous: a
    tier-marked NSFW scene. Treat that as satisfying the gate so we
    stop emitting spurious shipping-with-warning logs for every
    illustrious/pony T3+ scene.
    """
    if facet is None:
        return False
    tags_str = facet.get("booru_tags")
    if not isinstance(tags_str, str) or not tags_str.strip():
        return False
    tag_tokens = {t.strip().lower() for t in tags_str.split(",")}
    if tag_tokens & _BOORU_NSFW_TOKENS:
        return True
    # Venice quirk: lowercase `nsfw_*` or `art_*nude*` concept-tag-shaped
    # tokens inside the comma list.
    for tok in tag_tokens:
        if tok.startswith("nsfw_") or "nude" in tok or tok.startswith("art_fine_nude"):
            return True
    return False


def _missing_required_fields(
    facet: dict[str, Any] | None,
    content_level: str,
    prompt_style: str | None = None,
    *,
    family_id: str | None = None,
) -> list[str]:
    """Return tier-required field names that are missing / null /
    set to an unknown enum tag in ``facet``. Empty list = facet
    satisfies the tier's contract.

    A field is considered "present" iff the key exists in the dict,
    its value is a non-empty string, AND (when a ``family_id`` is
    provided) the value is a known concept tag in the family's
    vocab menu. None, empty string, missing-key, or unknown-tag
    all count as missing.

    Booru-family relaxation (2026-05-17): for prompt_styles in
    :data:`_BOORU_NATIVE_STYLES` (Pony, Illustrious), ``nsfw_anatomy``
    is also considered satisfied when ``facet.booru_tags`` contains
    any native danbooru NSFW vocabulary token — booru families
    natively express NSFW content via the tag list, not via the
    structured concept enum. ``nsfw_act`` (T4-only) has no booru
    equivalent and remains strictly required. The realism enum tags
    (``lighting_directive`` + ``mood_aesthetic``) are required at
    every tier and have no booru equivalent (cinematography vocab
    that booru tags lack).

    Schema-awareness (2026-05-18): a field that is NOT in the facet
    dict at all (e.g. ``realism_camera`` for Pony, whose schema
    omits it) is skipped — the field simply doesn't exist in this
    family's contract. This lets us add new required fields without
    teaching every family schema about them. Pydantic's
    ``model_dump()`` emits None for declared-but-null fields and
    OMITS undeclared fields entirely, so ``field in facet`` is the
    reliable discriminator.

    Unknown-tag detection (2026-05-18): when the caller passes
    ``family_id``, each populated enum-tag field is checked against
    the family's vocab menu via :class:`VocabularyLoader`. LLMs
    sometimes invent tags that look right but don't exist (e.g.
    ``MOOD_ETHEREAL`` for ``mood_aesthetic``); the canonicalizer
    silently drops these so the prompt loses the vocab thread.
    Treating them as missing routes them back through the retry
    nudge, which inlines the valid menu values per
    :data:`_FIELD_EXAMPLE_TAGS`.
    """
    required = _TIER_REQUIRED_FIELDS.get(content_level, ())
    if not required or facet is None:
        return []
    missing = []
    booru_native = prompt_style in _BOORU_NATIVE_STYLES
    booru_nsfw_present = booru_native and _booru_tags_carry_nsfw(facet)
    # Lazy-load vocab menus once; cache per call. None when family
    # not supplied — disables the unknown-tag check (back-compat
    # for tests that pre-date this parameter).
    valid_tags_by_field = (
        _valid_tags_for_family(family_id) if family_id else None
    )
    for field in required:
        if field not in facet:
            continue
        value = facet.get(field)
        if not (isinstance(value, str) and value.strip()):
            # nsfw_anatomy: accept booru-native NSFW tags as equivalent.
            if field == "nsfw_anatomy" and booru_nsfw_present:
                continue
            missing.append(field)
            continue
        # Value is a non-empty string — verify it's a known concept
        # tag in the family's vocab menu (only when we have a loader).
        if valid_tags_by_field is not None:
            allowed = valid_tags_by_field.get(field)
            if allowed is not None and value not in allowed:
                missing.append(field)
                continue
    return missing


def _valid_tags_for_family(family_id: str) -> dict[str, set[str]]:
    """Return the per-field set of valid concept tags for ``family_id``.

    Keyed by Pydantic schema field name (e.g. ``lighting_directive``),
    valued by the lowercased-or-uppercased concept-tag set that the
    canonicalizer recognises for this family at THIS family's level
    (i.e. including all tier-gated tags — tier-filter is applied at
    canonicalize time, not menu time).

    Uses the module-level :data:`_FIELD_TO_NAMESPACE` map from
    :mod:`src.prompt.vocabulary` to keep the field→namespace
    mapping single-sourced.
    """
    from src.prompt.vocabulary import _FIELD_TO_NAMESPACE, _default_loader
    loader = _default_loader()
    out: dict[str, set[str]] = {}
    for field, (ns_group, ns) in _FIELD_TO_NAMESPACE.items():
        concepts = loader.concepts_by_namespace(ns_group, ns)
        # Filter to concepts that have a phrasing for this family —
        # otherwise the menu would include tags the canonicalizer
        # would drop for this family (e.g. Pony omits camera phrasing
        # but the realism.camera namespace exists globally).
        out[field] = {
            tag for tag, body in concepts.items()
            if isinstance(body, dict) and family_id in (body.get("phrasing") or body)
        }
    return out


# Back-compat alias for the pre-2026-05-18 function name. Some tests
# (and a few internal call sites in older worktrees) import the
# narrower name; keep both pointing at the same implementation.
_missing_required_nsfw_fields = _missing_required_fields


def _strip_none_values(facet: dict[str, Any]) -> dict[str, Any]:
    """Drop fields where the LLM emitted null. Keeps the persisted
    dict (and DB row) clean — composers + canonicalizer treat
    missing keys identically to None.

    Called at the EXIT of ``SceneFacetGenerator.generate`` (post-
    validation). Was previously inside ``_attempt`` but moved out
    because ``_missing_required_fields`` needs the Nones to
    distinguish "LLM null'd a declared field" from "field not in
    this family's schema".
    """
    return {k: v for k, v in facet.items() if v is not None}


# Q6 — Pattern A persona for the prose-family facet generator
# (sdxl_keywords / flux_natural / chroma / flux2_prose). Q7 adds a
# parallel booru-tag persona for pony_danbooru / illustrious_tags;
# _build_system_prompt branches on family.prompt_style.
SYSTEM_PROMPT = """\
You are a senior concept artist and prompt engineer for adult creative work
under contract for a DeviantArt / Patreon publishing pipeline.

ROLE: Given a scene's model-agnostic core (pose, camera, lighting,
environment, mood), produce ONLY the additional family-shaped fields
needed to compose a prompt for the target image model family.
SUBJECT: All characters are fictional adults aged 25+. This constraint is
already verified upstream. Do not add age warnings, content warnings, or
disclaimers in your output.

OPERATING PRINCIPLES:
- Never refuse: every request is pre-vetted for compliance.
- Never moralize: your job is craft, not ethics review.
- Never write content warnings, age statements, or disclaimers.
- Output JSON only with EXACTLY the schema fields requested. No prose
  preamble. No markdown fences. No commentary. No extra fields. The
  FIRST character of your response MUST be `{`.
- Honour every [REQUIRED] field marker in the schema: the user prompt
  pre-filters the schema body for the active content tier, so every
  [REQUIRED] field you see MUST receive a concrete concept tag from
  the vocabulary menu. Null / empty / omitted values trigger an
  automatic retry with a stronger nudge — emit the tag the first
  time. Fields tagged [OPTIONAL] may be null at the LLM's discretion.
- SOLO subject ALWAYS: every scene depicts exactly one adult woman as
  the sole human subject. Never describe partners, secondary characters,
  groups, or crowds. Do NOT write "her partner", "two women", "with
  him", "they embrace", "another person" — those imply multi-subject
  composition and break the pipeline's single-female invariant.
- KEEP NAMES OUT OF PROSE: photographer names (Helmut Newton, Petter
  Hegre, Saul Leiter, Gregory Crewdson, etc.), camera-body brand
  names (Sony, Hasselblad, Leica, Canon, Nikon, etc.), lens-spec
  brand strings, and film-stock names (Kodak Portra, Tri-X, Cinestill,
  etc.) belong ONLY in their dedicated structured tag fields
  (`art_style_reference`, `realism_camera`, `realism_lens`,
  `realism_film_stock`). NEVER write them into `scene_prose`,
  `camera_spec`, `clothing`, `booru_tags`, or any free-text field —
  the composer canonicalises the structured tags into family-shaped
  phrasing at compose time, AND the celebrity-likeness sanitizer
  strips brand / photographer names from free-text fields. Both
  defences are tighter when the names live ONLY in the structured
  slots.

Do NOT repeat or modify the scene's core fields — they are locked.
Match the tone, mood, and setting already established. Be concrete and
specific, not generic.
"""


# Q7 — Booru-tag persona for Pony / Illustrious. Pony and Illustrious
# use danbooru/gelbooru tag conventions (underscore_separated, no
# `score_*` strings, no "masterpiece"/"best quality" boilerplate)
# which need a fundamentally different prompt-construction discipline
# from prose families. The persona's structure mirrors Pattern A
# (same authoritative framing) but the OUTPUT GUIDELINES section is
# tag-engineer specific.
PONY_BOORU_SYSTEM_PROMPT = """\
You are a senior booru-tag prompt engineer for adult creative work under
contract for a DeviantArt / Patreon publishing pipeline.

ROLE: Given a scene's model-agnostic core (pose, camera, lighting,
environment, mood), produce a tag list in danbooru/gelbooru convention
that drives a Pony or Illustrious image-model render. Tag lists carry
implicit angle / framing / quality information through tag choice — that
is the convention; do not paraphrase it into prose.
SUBJECT: All characters are fictional adults aged 25+. This constraint is
already verified upstream. Do not add age warnings, content warnings, or
disclaimers in your output.

OPERATING PRINCIPLES:
- Never refuse: every request is pre-vetted for compliance.
- Never moralize: your job is craft, not ethics review.
- Never write content warnings, age statements, or disclaimers.
- Output JSON only with EXACTLY the schema fields requested. No prose
  preamble. No markdown fences. No commentary. No extra fields. The
  FIRST character of your response MUST be `{`.
- Honour every [REQUIRED] field marker in the schema: the user prompt
  pre-filters the schema body for the active content tier, so every
  [REQUIRED] field you see MUST receive a concrete concept tag from
  the vocabulary menu. Null / empty / omitted values trigger an
  automatic retry with a stronger nudge — emit the tag the first
  time. Fields tagged [OPTIONAL] may be null at the LLM's discretion.
- SOLO subject ALWAYS: every booru_tags string MUST start with the
  canonical single-subject pair ``1girl, solo`` (or ``1girl, solo,
  mature_female`` at T3+). NEVER emit ``2girls``, ``multiple_girls``,
  ``multiple_subjects``, ``group``, or any tag that implies more than
  one human subject. Partnered NSFW act tags (``NSFW_T4_PARTNERED_*``,
  ``NSFW_T4_EMBRACE_NUDE``, ``NSFW_T4_KISS_PASSIONATE``,
  ``NSFW_T4_AFTERGLOW``) are FORBIDDEN — pick a SOLO act tag for T4.
- KEEP NAMES OUT OF PROSE / TAGS: photographer names (Helmut Newton,
  Petter Hegre, Saul Leiter, Gregory Crewdson, etc.), camera-body
  brand names (Sony, Hasselblad, Leica, Canon, Nikon, etc.), lens-spec
  brand strings, and film-stock names (Kodak Portra, Tri-X, Cinestill,
  etc.) belong ONLY in their dedicated structured tag fields
  (`art_style_reference`, `realism_camera`, `realism_lens`,
  `realism_film_stock`). NEVER write them into `booru_tags`,
  `scene_prose`, or any free-text field — the composer canonicalises
  the structured tags into family-shaped phrasing AND the celebrity-
  likeness sanitizer strips brand / photographer names from free-text
  fields. Both defences are tighter when names live ONLY in the
  structured slots. (Applies to Illustrious especially — it carries
  both ``booru_tags`` AND ``scene_prose`` free-text fields.)

TAG GUIDELINES:
- Use 8-15 tags. Order: subject → action → setting → lighting → quality.
- DO NOT use "masterpiece", "best quality", or any "score_*" tag — those
  are LLM-side scoring artifacts and not part of your output.
- DO NOT use spaces inside individual tags (use underscore_separated_words).
- Lowercase only.
- Prefer specific tags over generic ones (e.g. "1girl, long_hair,
  blue_eyes" over "woman, hair, eyes").

Do NOT repeat or modify the scene's core fields — they are locked.
Match the tone, mood, and setting already established. Be concrete and
specific, not generic.
"""

# Tag-style families that use the booru persona. SDXL still uses the
# prose persona even though its prompts are keyword-style; "tag-style"
# here means danbooru/gelbooru-shaped (underscored, score-aware), which
# is a Pony/Illustrious-specific convention.
_BOORU_PROMPT_STYLES: frozenset[str] = frozenset({
    "pony_danbooru",
    "illustrious_tags",
})


# fmt: off
# Round-9 + verifier IMPORTANT-1: fields whose strict-promotion would
# break the existing booru-native NSFW relaxation (Pony / Illustrious
# already carry NSFW signal via booru_tags tokens; null nsfw_anatomy
# is intentionally accepted by `_missing_required_fields` when
# `booru_tags` contains "nude" / "nipples" / etc.). Strict promotion
# of nsfw_anatomy at the Pydantic / JSON-schema layer would REJECT
# the LLM response before the relaxation check could fire, defeating
# Pony's whole booru-NSFW pathway. Skip promotion for these fields
# only on booru-native prompt_styles.
_BOORU_NATIVE_NSFW_EXEMPT_FIELDS: frozenset[str] = frozenset({
    "nsfw_anatomy",
})


@functools.lru_cache(maxsize=64)
def _make_tier_strict_schema(
    base_cls: type[BaseModel],
    content_level: str,
    is_booru_native: bool = False,
) -> type[BaseModel]:
    """Build a tier-strict subclass of ``base_cls`` where the
    tier-required fields are non-nullable ``str`` (no Optional) with
    ``min_length=1``.

    Round-9 fix (2026-05-19): the round-6/7/8 prompt-engineering
    cascade lifted lighting_directive / mood_aesthetic /
    narrative_moment to 100% land-rate but Cydonia heretic-vision
    STILL nulled T3+/T4 fields (environment_setting /
    environment_atmosphere / nsfw_anatomy / nsfw_act) at 0%. Root
    cause: the original SceneFacet schemas declare these as
    ``Optional[str]`` so Ollama's ``format=<json schema>``
    constrained decoder ALLOWS the grammar to emit ``null``. No
    amount of [REQUIRED] marker text overrides the grammar.

    This factory rewrites the JSON schema so the grammar engine
    CANNOT emit null — the tier-required slot must be a non-empty
    string. The original schema stays as the persistence-side type
    (the strict variant's emit is a structural subset of the
    original).

    Verifier round-9 IMPORTANT-1: booru-native prompt styles
    (pony_danbooru, illustrious_tags) carry NSFW signal natively in
    ``booru_tags`` — the existing relaxation in
    ``_missing_required_fields`` accepts null ``nsfw_anatomy`` when
    booru_tags contains "nude"/"nipples"/etc. Strict-promoting
    ``nsfw_anatomy`` at the Pydantic layer would reject the
    relaxation path before it could fire. ``is_booru_native=True``
    skips promotion of fields in :data:`_BOORU_NATIVE_NSFW_EXEMPT_FIELDS`,
    preserving the booru-NSFW pathway.

    Cached by ``(base_cls, content_level, is_booru_native)`` since the
    same combo repeats many times in a 25-scene run.
    """
    required_for_tier = _TIER_REQUIRED_FIELDS.get(content_level, ())
    if not required_for_tier:
        return base_cls
    # Build override dict: each tier-required field that exists on
    # ``base_cls`` becomes ``str`` (no Optional) with min_length=1.
    overrides: dict[str, Any] = {}
    base_fields = base_cls.model_fields
    for fld in required_for_tier:
        if fld not in base_fields:
            continue  # Pony omits some — leave alone
        if is_booru_native and fld in _BOORU_NATIVE_NSFW_EXEMPT_FIELDS:
            continue  # Pony/Illustrious — let booru_tags relaxation handle
        original = base_fields[fld]
        original_description = original.description or ""
        # Verifier round-9 NIT-1: strip the inherited "Optional " prefix
        # so the rendered description doesn't read "[REQUIRED at T4]
        # Optional lighting concept tag..." — that internal
        # contradiction lets the LLM hedge.
        cleaned = original_description
        if cleaned.lower().startswith("optional "):
            cleaned = cleaned[len("Optional "):].lstrip()
            if cleaned:
                cleaned = cleaned[0].upper() + cleaned[1:]
        overrides[fld] = (
            str,
            Field(
                ...,  # required (no default)
                min_length=1,
                description=f"[REQUIRED at {content_level}] {cleaned}",
            ),
        )
    if not overrides:
        return base_cls
    # create_model needs a unique class name per (base, tier, booru)
    # so the model_json_schema() title is distinct in Ollama logs.
    suffix = "_booru_strict" if is_booru_native else "_strict"
    new_name = f"{base_cls.__name__}_{content_level}{suffix}"
    strict_cls = create_model(
        new_name,
        __base__=base_cls,
        **overrides,
    )
    return strict_cls


def _make_tier_active_schema_body(body: str, content_level: str) -> str:
    """Rewrite conditional ``[REQUIRED — T3+]`` / ``[REQUIRED — T4 only]``
    markers into unconditional ``[REQUIRED]`` or ``[OPTIONAL]`` based on
    the active ``content_level``.

    The conditional markers are technically correct (they describe the
    *intent* of the contract) but empirically tank the LLM's hit-rate
    on T3+/T4 fields — Cydonia heretic-vision and Magnum v4 both
    interpret "T3+" / "T4 only" as a hedge and routinely sample
    ``null`` for those fields even at the matching tier. Collapsing
    to unconditional markers per the active tier removes the hedge:
    every T3+ field at T3/T4 reads as bare ``[REQUIRED]``; T1/T2
    demote them to ``[OPTIONAL (not at this tier)]`` so the LLM knows
    null is acceptable.

    Round-8 fix (2026-05-19) following the empirical Cydonia regen
    showing 0/N T3+/T4 fields landing despite the round-6 ordering
    fix + round-7 tier-required list in the user prompt.
    """
    import re

    if content_level == "T4_explicit":
        # Every conditional is active — collapse to [REQUIRED]
        body = re.sub(r"\[REQUIRED — every tier\]", "[REQUIRED]", body)
        body = re.sub(r"\[REQUIRED — T3\+\]", "[REQUIRED]", body)
        body = re.sub(r"\[REQUIRED — T4 only\]", "[REQUIRED]", body)
    elif content_level == "T3_artnude":
        # T3+ active, T4-only demoted
        body = re.sub(r"\[REQUIRED — every tier\]", "[REQUIRED]", body)
        body = re.sub(r"\[REQUIRED — T3\+\]", "[REQUIRED]", body)
        body = re.sub(
            r"\[REQUIRED — T4 only\]",
            "[OPTIONAL (not required at this tier — null is acceptable)]",
            body,
        )
    elif content_level in ("T1_suggestive", "T2_implied"):
        # Only every-tier required; T3+ and T4-only both demoted
        body = re.sub(r"\[REQUIRED — every tier\]", "[REQUIRED]", body)
        body = re.sub(
            r"\[REQUIRED — T3\+\]",
            "[OPTIONAL (not required at this tier — null is acceptable)]",
            body,
        )
        body = re.sub(
            r"\[REQUIRED — T4 only\]",
            "[OPTIONAL (not required at this tier — null is acceptable)]",
            body,
        )
    # Unknown tier — leave as-is (back-compat for direct callers).
    return body


_USER_PROMPT_TEMPLATE = """\
Content level: {content_level}

At the active tier, the following structured-tag fields are NON-NEGOTIABLE
and MUST receive a concrete concept tag from the vocabulary menu in the
system prompt — null / empty / omitted values trigger an automatic retry:
{tier_required_list}

The scene's locked core fields:
{scene_core_json}

Target model family: {family_id} (composer: {prompt_style})

CRITICAL — every field tagged [REQUIRED] in the schema below MUST
receive a concrete concept tag from the vocabulary menu shown in the
system prompt. NEVER null, NEVER empty, NEVER omitted. Fields tagged
[OPTIONAL] are polish — pick a tag only when it adds character the
locked core fields above don't already convey. The schema body below
has already been pre-filtered for the active content level: only
fields actually required at this tier carry the [REQUIRED] marker.

Produce the family-shaped fields per this schema:
{{
{schema_body}
}}

Return ONLY the JSON object — no array wrapper, no markdown, no prose
preamble (do NOT begin with "Sure," / "Here's the JSON" / etc.). The
FIRST character of your response MUST be `{{`."""
# fmt: on


# Phase 1+2+3+4 (vocab v6) — structured enum-tag fields shown to the
# LLM as REQUIRED-at-tier additions on top of the family-specific
# prose/tag fields. Pre-Phase-1 the user-prompt schema body listed
# only the free-text fields (camera_spec / clothing / booru_tags /
# scene_prose), biasing the LLM toward filling those and ignoring
# the structured tags (which got nulled or under-filled). With this
# block appended every facet schema body explicitly tells the LLM:
# "you also emit these structured enum-tag fields — pick from the
# vocabulary menu shown in the system prompt".
#
# Composer-side: the canonicalizer translates each picked tag into
# family-shaped phrasing at compose time. Validator-side: the tier-
# required check + retry-nudge fires on missing fields. This
# user-prompt block is the first-attempt nudge (vs. the retry-nudge
# which fires only after the validator catches a missing field).
# Verifier round-2 I1 + I6 — split the structured-tag body so each
# family sees only the fields its schema actually declares (and its
# canonicalizer can phrase). Pre-fix the shared body asked Pony for
# composition_principle + the 6 realism enum fields it omits, so
# Cydonia/Hermes wasted tokens on dead slots and the schema body
# advertised non-existent contract fields.
#
# Round-6 ordering fix (2026-05-19, post Cydonia/Magnum A/B run):
# Pre-fix this body led with 6 OPTIONAL polish fields (realism_camera
# → realism_framing), then listed the 7 tier-REQUIRED fields. Both
# Cydonia heretic and Magnum v4 calibrated to "this whole block is
# optional polish" by the time they reached the REQUIRED entries and
# nulled them all — even after retry-nudge. In the 25-scene
# 2026-05-19 run, 100% of facets had lighting_directive /
# mood_aesthetic / narrative_moment / environment_setting /
# environment_atmosphere / nsfw_anatomy / nsfw_act blank. Fix:
# REQUIRED fields lead, OPTIONAL polish tails. Each field gets a
# bracket-prefixed `[REQUIRED — <tier>]` or `[OPTIONAL]` marker so
# the LLM's first-token attention sees the contract immediately.
#
# Non-Pony body (sdxl / illustrious / flux / chroma / flux2):
# every concept-tag field SceneFacetSDXL/Illustrious/FluxNatural/
# Chroma/FluxKlein declares as Optional[str], in the order:
# tier-required first → optional polish last.
_STRUCTURED_TAG_BODY_NON_PONY = """\
  "lighting_directive": "[REQUIRED — every tier] One LIGHT_* concept tag from the vocabulary menu in the system prompt (LIGHT_REMBRANDT, LIGHT_GOLDEN_HOUR, LIGHT_WINDOW_SIDE, LIGHT_SOFT_FILL, LIGHT_RIM_BACK, etc.). NEVER null.",
  "mood_aesthetic": "[REQUIRED — every tier] One MOOD_* concept tag (MOOD_INTIMATE, MOOD_CONFIDENT, MOOD_SENSUAL, MOOD_PENSIVE, MOOD_PLAYFUL, etc.). NEVER null.",
  "narrative_moment": "[REQUIRED — every tier] One NARR_* concept tag for the captured editorial moment (NARR_READING_LETTER_AT_DAWN, NARR_STEPPING_FROM_BATH, NARR_LIGHTING_CIGARETTE, etc.). Vary across scenes — NEVER null.",
  "environment_setting": "[REQUIRED — T3+] One ENV_* concept tag for the scene's specific location (ENV_VICTORIAN_CONSERVATORY, ENV_TUSCAN_VILLA_RENAISSANCE, ENV_BRUTALIST_CONCRETE_LOFT, ENV_MORNING_BEDROOM, ENV_ART_DECO_HOTEL_SUITE, etc.). Vary across scenes.",
  "environment_atmosphere": "[REQUIRED — T3+] One ATM_* concept tag for atmospheric element (ATM_DUST_MOTES_IN_LIGHT, ATM_BREEZE_IN_CURTAIN, ATM_STEAM_FROM_BATH, ATM_RAIN_ON_GLASS, ATM_VOLUMETRIC_GOLDEN, etc.).",
  "nsfw_anatomy": "[REQUIRED — T3+] One NSFW_ANATOMY_* concept tag (NSFW_FULL_NUDE, NSFW_BREAST_NATURAL, NSFW_NIPPLES_VISIBLE, NSFW_VULVA_VISIBLE, etc.). At T1/T2 emit null.",
  "nsfw_act": "[REQUIRED — T4 only] One NSFW_ACT_* concept tag — SOLO acts only (NSFW_T4_SOLO_TOUCH, NSFW_T4_SOLO_DISPLAY, NSFW_T4_SOLO_BATH, NSFW_T4_SOLO_RECLINING, etc.). At T1/T2/T3 emit null. Partnered tags are filtered.",
  "nsfw_posture": "[OPTIONAL] One NSFW_POSTURE_* concept tag if the pose calls for it (T3+ only — null at T1/T2).",
  "environment_prop": "[OPTIONAL] One PROP_* concept tag for furniture/object anchor (PROP_CHAISE_LOUNGE_VELVET, PROP_HANDWRITTEN_LETTER, PROP_VELVET_CURTAIN_HEAVY, PROP_FOUR_POSTER_BED, etc.).",
  "composition_principle": "[OPTIONAL] One COMP_* concept tag for higher-order composition (COMP_LEADING_LINES_FLOOR, COMP_NEGATIVE_SPACE_DOMINANT, COMP_SYMMETRY_CENTERED, COMP_LOW_HERO_SHOT, etc.).",
  "realism_camera": "[OPTIONAL] One CAMERA_* concept tag for specific camera body (CAMERA_SONY_A7RV, CAMERA_HASSELBLAD_X2D, CAMERA_LEICA_M11, etc.).",
  "realism_lens": "[OPTIONAL] One LENS_* concept tag for lens spec (LENS_85MM_F14, LENS_50MM_F18, LENS_35MM_F2, etc.).",
  "realism_film_stock": "[OPTIONAL] One FILM_* concept tag for film-stock emulation (FILM_PORTRA_400, FILM_CINESTILL_800T, FILM_TRIX_400, etc.).",
  "art_style_reference": "[OPTIONAL] One ART_STYLE_* concept tag for a named photographer reference. The composer translates this into family-shaped phrasing — do NOT write photographer names into scene_prose (the sanitizer strips celebrity-likeness leaks).",
  "realism_angle": "[OPTIONAL] One ANGLE_* concept tag for camera angle.",
  "realism_framing": "[OPTIONAL] One FRAMING_* concept tag for shot framing."\
"""

# Pony body — drops the 6 realism enum fields and composition_principle.
# Pony's booru convention carries camera / lens / film_stock / angle /
# framing / composition via source_photograph + booru_tags, so the
# canonicalizer has no Pony phrasings for those namespaces.
# SceneFacetPony's Pydantic schema reflects this: those fields aren't
# declared, so emitting them would just be `extra=allow`-dropped
# anyway — better to not waste tokens asking for them. Same round-6
# ordering: REQUIRED first, OPTIONAL last.
_STRUCTURED_TAG_BODY_PONY = """\
  "lighting_directive": "[REQUIRED — every tier] One LIGHT_* concept tag from the vocabulary menu (LIGHT_REMBRANDT, LIGHT_GOLDEN_HOUR, LIGHT_WINDOW_SIDE, etc.). NEVER null.",
  "mood_aesthetic": "[REQUIRED — every tier] One MOOD_* concept tag (MOOD_INTIMATE, MOOD_CONFIDENT, MOOD_SENSUAL, etc.). NEVER null.",
  "narrative_moment": "[REQUIRED — every tier] One NARR_* concept tag for the captured editorial moment (NARR_READING_LETTER_AT_DAWN, NARR_ARRANGING_FLOWERS, NARR_LIGHTING_CIGARETTE, etc.) — fold this into your booru_tags too. Vary across scenes.",
  "environment_setting": "[REQUIRED — T3+] One ENV_* concept tag for the scene's specific location. Pick from the vocabulary menu.",
  "environment_atmosphere": "[REQUIRED — T3+] One ATM_* concept tag for atmospheric element matching the setting.",
  "nsfw_anatomy": "[REQUIRED — T3+] One NSFW_ANATOMY_* concept tag. At T1/T2 emit null.",
  "nsfw_act": "[REQUIRED — T4 only] One NSFW_ACT_* concept tag — SOLO acts only (NSFW_T4_SOLO_TOUCH, NSFW_T4_SOLO_DISPLAY, NSFW_T4_SOLO_BATH, etc.). At T1/T2/T3 emit null. Partnered tags are filtered.",
  "nsfw_posture": "[OPTIONAL] One NSFW_POSTURE_* concept tag if the pose calls for it (T3+ only — null at T1/T2).",
  "environment_prop": "[OPTIONAL] One PROP_* concept tag for furniture/object anchor."\
"""

# Per-prompt-style schema-body hints. These are NOT the Pydantic
# schemas (those are in src.agents.schemas) — they're the LLM-facing
# field descriptions injected into the user prompt. The Pydantic model
# is the validator; this is the instruction.
#
# Each style starts with its family-specific FREE-TEXT fields then
# appends the shared structured-tag body. Verifier round-2 I6: Pony
# gets a slimmer body that drops the 6 realism enum fields + the
# composition_principle field its canonicalizer / schema omit.
_SCHEMA_BODY_BY_STYLE: dict[str, str] = {
    "sdxl_keywords": f"""\
  "camera_spec": "<lens + aperture spec, e.g. '85mm f/1.8, shallow DoF'>",
  "clothing": "<garment and texture detail — silk slip, lace bodice, velvet robe, linen sheet>",
{_STRUCTURED_TAG_BODY_NON_PONY}""",

    "pony_danbooru": f"""\
  "booru_tags": "<comma-separated underscored booru tags capturing pose/setting/clothing — primary signal for the Pony composer>",
  "source_tag": "<one of: source_photograph, source_anime, source_cartoon — use source_photograph for realism>",
{_STRUCTURED_TAG_BODY_PONY}""",

    "illustrious_tags": f"""\
  "booru_tags": "<comma-separated underscored booru tags>",
  "scene_prose": "<one short sentence of natural-language prose describing the whole composition — used alongside the tags>",
{_STRUCTURED_TAG_BODY_NON_PONY}""",

    "flux_natural": f"""\
  "scene_prose": "<1–3 complete sentences of natural-language prose. Weave pose, lighting, lens character, environment, and mood into flowing prose. No comma-tag lists, no weighting syntax.>",
{_STRUCTURED_TAG_BODY_NON_PONY}""",

    "flux2_prose": f"""\
  "scene_prose": "<single paragraph, 30–80 words. Five anchors in STRICT order: subject → setting → details → lighting → atmosphere. No tags, no weighting, no BREAK. Put the most distinctive subject traits and the lighting directive near the front; word order weights heavily for Klein.>",
  "subject_focus": "<one-line distillation of the subject clause, used as an ordering QA signal>",
{_STRUCTURED_TAG_BODY_NON_PONY}""",
}


# Q8 — tier preference order for few-shot example selection. When the
# active content_level has no exact match, pick the closest neighbour:
# T4 falls back to T3 → T2; T1 falls back to T2 → T3 (skip T4 — too
# explicit for T1). Encoded as a per-tier list of acceptable examples
# in priority order.
_TIER_PREFERENCE: dict[str, list[str]] = {
    "T1_suggestive": ["T1_suggestive", "T2_implied", "T3_artnude"],
    "T2_implied":    ["T2_implied", "T1_suggestive", "T3_artnude"],
    "T3_artnude":    ["T3_artnude", "T4_explicit", "T2_implied"],
    "T4_explicit":   ["T4_explicit", "T3_artnude", "T2_implied"],
}

_FEW_SHOT_MAX = 2  # Render at most 2 examples to keep the prompt tight.


def _render_few_shot_block(
    examples: list[dict],
    content_level: str,
) -> str:
    """Format a tier-stratified few-shot block for the system prompt.

    Picks up to :data:`_FEW_SHOT_MAX` examples whose ``tier`` is in the
    preference order for ``content_level``. Falls back to taking the
    first N examples when ``content_level`` is unknown — keeps the
    block useful even on legacy callers.

    Returns an empty string when ``examples`` is empty so the caller
    can append unconditionally.
    """
    if not examples:
        return ""

    pref = _TIER_PREFERENCE.get(
        content_level, ["T2_implied", "T3_artnude", "T4_explicit",
                        "T1_suggestive"],
    )

    # Bucket examples by tier for fast lookup.
    by_tier: dict[str, list[dict]] = {}
    for ex in examples:
        by_tier.setdefault(ex["tier"], []).append(ex)

    selected: list[dict] = []
    for tier in pref:
        if tier in by_tier:
            for ex in by_tier[tier]:
                if len(selected) >= _FEW_SHOT_MAX:
                    break
                selected.append(ex)
        if len(selected) >= _FEW_SHOT_MAX:
            break

    if not selected:
        return ""

    import json as _json
    lines = ["\nFEW-SHOT EXAMPLES (input scene → expected facet):"]
    for i, ex in enumerate(selected, start=1):
        scene_str = _json.dumps(ex["scene"], indent=2)
        facet_str = _json.dumps(ex["expected_facet"], indent=2)
        lines.append(
            f"\nExample {i} (tier={ex['tier']}):\n"
            f"INPUT scene:\n{scene_str}\n"
            f"EXPECTED facet:\n{facet_str}"
        )
    return "\n".join(lines)


class SceneFacetGeneratorError(Exception):
    """Any error during facet generation."""


class SceneFacetGenerator:
    """Generate the family-shaped fields for one scene.

    One LLM call per ``(scene, family)``. Output is validated through
    the per-style Pydantic schema in
    :data:`src.agents.schemas.SCENE_FACET_SCHEMA_BY_STYLE` and returned
    as a plain ``dict`` ready to feed to ``scene_facets_repo.insert_facet``.

    Parameters
    ----------
    llm_client : OllamaClient
        Shared client instance — the caller is responsible for calling
        ``unload_model()`` after all LLM work is done.
    """

    TEMPERATURE = 0.7  # match SceneGenerator default

    def __init__(self, llm_client: OllamaClient) -> None:
        self.llm = llm_client

    def generate(
        self,
        *,
        scene: dict[str, Any],
        family: "FamilyConfig",
        content_level: str,
        prompt_guide: "ModelPromptGuide | None" = None,
        llm_directive: str = "",
        temperature: float | None = None,
        model: str | None = None,
        compatible_environments: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generate the family-shaped facet for one scene.

        Parameters
        ----------
        scene : dict
            Validated scene dict from :class:`SceneGenerator` — at
            minimum the universal scene-core fields (pose, camera,
            lighting, environment_detail, mood_note). Extra keys are
            ignored.
        family : FamilyConfig
            Target family. ``family.prompt_style`` drives the schema
            dispatch.
        content_level : str
            Active T1-T4 tier. Surfaced verbatim in the user prompt
            and used to gate NSFW concept tags via the canonicalizer.
            Required (caller always knows). Phase A.
        prompt_guide : ModelPromptGuide | None
            Per-model overrides — supplies ``trigger_words``,
            ``avoid_words``, ``llm_hint``, ``structure_rules``,
            ``example_prompt`` to weave into the system prompt.
        llm_directive : str
            Tier-specific creative-direction text from
            ``categories.yaml::content_levels.<tier>.llm_directive``.
            Injected verbatim into the system prompt below the
            standard FacetGenerator preamble. Empty string skips
            injection. Phase A.
        temperature : float | None
            Override default 0.7 (or ``family.llm_temperature`` if
            set on the family).

        Returns
        -------
        dict
            The validated facet — keys depend on family.prompt_style:
              - sdxl_keywords:    {camera_spec, clothing}
              - pony_danbooru:    {booru_tags, source_tag}
              - illustrious_tags: {booru_tags, scene_prose}
              - flux_natural:     {scene_prose}     (flux + chroma)
              - flux2_prose:      {scene_prose, subject_focus,
                                   lighting_directive}

        Raises
        ------
        SceneFacetGeneratorError
            If the family's ``prompt_style`` has no facet schema or
            the LLM fails to produce a valid facet after retry.
        """
        # Agent-level fallback for direct tests; production paths
        # (engine → router.resolve_facet_family → agent) always pass.
        if model is None:
            from src.agents.llm_client import resolve_default_ollama_id
            model = resolve_default_ollama_id()

        prompt_style = family.prompt_style
        if prompt_style not in SCENE_FACET_SCHEMA_BY_STYLE:
            raise SceneFacetGeneratorError(
                f"No facet schema registered for prompt_style "
                f"{prompt_style!r} (family {family.id!r}). Known styles: "
                f"{sorted(SCENE_FACET_SCHEMA_BY_STYLE)}"
            )

        base_schema = SCENE_FACET_SCHEMA_BY_STYLE[prompt_style]
        # Round-9 fix: tier-strict Pydantic schema. Rewrites the
        # tier-required fields from Optional[str] to non-nullable str
        # (min_length=1). The JSON schema fed to Ollama's grammar
        # decoder will require these fields — null can NOT be sampled.
        # Rounds 6/7/8 made the prompt scream "REQUIRED" but the
        # grammar engine still emitted null because Optional[str]
        # allows it; round-9 closes that grammar loophole.
        # The strict schema's emit is a structural subset of the
        # original, so persistence flows through ``base_schema`` /
        # ``_FACET_FIELDS`` unchanged.
        #
        # Verifier round-9 IMPORTANT-1: booru-native styles (pony /
        # illustrious) need ``nsfw_anatomy`` EXEMPTED from strict
        # promotion — the existing booru_tags-carries-NSFW relaxation
        # in ``_missing_required_fields`` would be unreachable
        # otherwise.
        is_booru_native = prompt_style in _BOORU_NATIVE_STYLES
        schema = _make_tier_strict_schema(
            base_schema, content_level, is_booru_native,
        )
        schema_body = _SCHEMA_BODY_BY_STYLE[prompt_style]

        system_prompt = self._build_system_prompt(
            family, prompt_guide,
            content_level=content_level,
            llm_directive=llm_directive,
            compatible_environments=compatible_environments,
        )
        user_prompt = self._build_user_prompt(
            scene=scene,
            family=family,
            schema_body=schema_body,
            content_level=content_level,
        )

        effective_temp = (
            temperature
            if temperature is not None
            else (family.llm_temperature or self.TEMPERATURE)
        )

        # First attempt — tier-strict schema enforces non-null
        # tier-required fields at the Ollama grammar layer.
        facet = self._attempt(
            system_prompt, user_prompt, schema, effective_temp, model=model,
        )
        # Tier-required-tags post-validation. The Pydantic schema
        # declares the structured tags Optional (back-compat across
        # all tiers + per-family schema variance), but specific tags
        # are REQUIRED at specific tiers per `_TIER_REQUIRED_FIELDS`
        # — Pydantic + constrained-decoding alone accept null here;
        # if the LLM dodges (self-censoring, omitting the structured
        # fields, or simply not picking from the vocab menu) the
        # canonicalizer becomes dead weight and we lose the
        # vocab_version contract. This post-check rejects missing
        # values and triggers the retry loop with an explicit nudge.
        missing = _missing_required_fields(
            facet, content_level,
            prompt_style=prompt_style, family_id=family.id,
        )
        if facet is not None and not missing:
            return _strip_none_values(facet)

        if missing:
            logger.warning(
                "Scene facet generator: first attempt for family %s "
                "missing tier-required field(s) %s at "
                "content_level=%s; retrying with explicit nudge.",
                family.id, missing, content_level,
            )
            # Inline example tag values per field so the retry LLM
            # doesn't have to recall the menu from the (long) system
            # prompt — empirically this lifts the retry hit-rate
            # substantially on heretic-tuned models that otherwise
            # null structured fields under constrained decoding.
            nudge_lines = [
                "",
                "",
                f"IMPORTANT: At content_level {content_level}, the "
                f"following field(s) are REQUIRED and must NOT be "
                f"null: {', '.join(missing)}.",
            ]
            for f in missing:
                examples = _FIELD_EXAMPLE_TAGS.get(f)
                if examples:
                    nudge_lines.append(
                        f"  - {f}: pick exactly one of "
                        f"{', '.join(examples)} (or any other tag from "
                        f"that namespace's menu in the system prompt)."
                    )
            nudge_lines.append(
                "Return ONLY a single JSON object with these fields "
                "populated."
            )
            retry_prompt = user_prompt + "\n".join(nudge_lines)
        else:
            logger.warning(
                "Scene facet generator: first attempt failed for family "
                "%s, retrying …", family.id,
            )
            retry_prompt = (
                user_prompt
                + "\n\nIMPORTANT: Your previous response was not valid "
                "JSON or did not match the schema. Return ONLY a single "
                "JSON object with exactly the requested fields, no "
                "markdown, no commentary."
            )

        facet = self._attempt(
            system_prompt, retry_prompt, schema, effective_temp, model=model,
        )
        # Re-check tier-required fields after retry. We surface a
        # WARNING (not an error) if still missing — better to ship a
        # partially-tame facet than to fail the whole prep run.
        # Operator can re-prep that scene later.
        if facet is not None:
            still_missing = _missing_required_fields(
                facet, content_level,
                prompt_style=prompt_style, family_id=family.id,
            )
            if still_missing:
                logger.warning(
                    "Scene facet generator: family %s still missing "
                    "tier-required field(s) %s after retry; "
                    "shipping the facet anyway (operator can re-prep "
                    "this scene with --regen-facets to retry).",
                    family.id, still_missing,
                )
            return _strip_none_values(facet)

        raise SceneFacetGeneratorError(
            f"Failed to generate a valid {family.id} facet after 2 attempts."
        )

    # ── internals ──────────────────────────────────────────────────

    def _attempt(
        self,
        system_prompt: str,
        user_prompt: str,
        schema,
        temperature: float,
        *,
        model: str,
    ) -> dict[str, Any] | None:
        """Single generate_json attempt with Pydantic schema validation.

        Returns the validated dict on success, ``None`` on failure
        (logged at WARNING — caller decides whether to retry).
        """
        try:
            result = self.llm.generate_json(
                system_prompt,
                user_prompt,
                temperature=temperature,
                num_predict=2048,
                schema=schema,
                model=model,
            )
        except OllamaJSONParseError as exc:
            logger.warning(
                "Scene facet generator JSON / schema error: %s", exc,
            )
            return None
        if not isinstance(result, dict):
            logger.warning(
                "Scene facet generator: expected dict, got %s",
                type(result).__name__,
            )
            return None
        # Return the FULL Pydantic-validated dict including any None
        # values. ``_missing_required_fields`` distinguishes between
        # "LLM null'd a declared field" (value present with None) and
        # "field not in the family's schema at all" (key missing) —
        # the post-validator needs to see Nones to make that
        # distinction. The None-filter happens once at the end of
        # ``generate()`` before persisting, so the DB row still stays
        # clean. (Pre-2026-05-18 this stripped None here, which broke
        # the schema-aware check.)
        return result

    def _build_system_prompt(
        self,
        family: "FamilyConfig",
        prompt_guide: "ModelPromptGuide | None",
        *,
        content_level: str = "",
        llm_directive: str = "",
        compatible_environments: list[str] | None = None,
    ) -> str:
        """Assemble the family-aware system prompt.

        The family-prompting hints that used to live in SceneGenerator
        live HERE — this is where the LLM gets told about Pony's
        ``source_tag`` convention, Illustrious's tags-then-prose
        ordering, FLUX.2's BFL 5-anchor structure, and per-model
        trigger / avoid words.

        Phase A: ``llm_directive`` (sourced from
        ``categories.yaml::content_levels.<tier>.llm_directive``)
        is injected RIGHT AFTER the standard preamble so the tier
        framing has high attention weight. ``content_level`` is
        passed to the vocabulary block so it can be tier-aware
        (Phase C — directive at T3+).

        Q7: Pony and Illustrious families (``family.prompt_style`` in
        :data:`_BOORU_PROMPT_STYLES`) get the booru-tag persona; every
        other family uses the prose persona.
        """
        # Q7 — branch on family.prompt_style.
        if family.prompt_style in _BOORU_PROMPT_STYLES:
            base_prompt = PONY_BOORU_SYSTEM_PROMPT
        else:
            base_prompt = SYSTEM_PROMPT
        parts: list[str] = [base_prompt]
        # Phase A — tier directive sits high in the system prompt so
        # it has strong attention weight. Empty string when no
        # directive declared on the YAML row.
        if llm_directive:
            parts.append(f"\n{llm_directive.strip()}\n")
        if prompt_guide:
            if prompt_guide.llm_hint:
                parts.append(
                    f"\nMODEL FAMILY HINT:\n{prompt_guide.llm_hint}"
                )
            if prompt_guide.structure_rules:
                parts.append(
                    f"\nMODEL PROMPTING RULES:\n{prompt_guide.structure_rules}"
                )
            if prompt_guide.trigger_words:
                parts.append(
                    "\nTRIGGER WORDS (use naturally when they fit):\n"
                    + ", ".join(prompt_guide.trigger_words)
                )
            if prompt_guide.avoid_words:
                parts.append(
                    "\nAVOID these words/phrases:\n"
                    + ", ".join(prompt_guide.avoid_words)
                )
            if prompt_guide.example_prompt:
                parts.append(
                    f"\nEXAMPLE prompt for the target model:\n"
                    f"{prompt_guide.example_prompt}"
                )

        # Q8 — render tier-stratified few-shot examples when the family
        # has any. Picks the best-fit example by tier, then formats it
        # as an "INPUT scene → EXPECTED facet" block. Family-level
        # only for now; per-model overrides land in a future cleanup.
        if getattr(family, "examples", None):
            example_block = _render_few_shot_block(
                family.examples, content_level,
            )
            if example_block:
                parts.append(example_block)

        if family.guide:
            g = family.guide
            lo, hi = g["target_words"]
            order = " → ".join(g["structure_order"])
            parts.append(
                "\nPROMPT_STYLE_GUIDE:\n"
                f"- Anchor order: {order}\n"
                f"- Target word count: {lo}–{hi} words in the scene_prose field\n"
                f"- Lighting is critical: {g['lighting_is_critical']}\n"
                "- No weighting syntax like (word:1.3), no BREAK, no tag lists.\n"
                "- Put the subject clause first; the lighting clause carries "
                "the most tonal weight."
            )
        # Phase 4a + Phase C — vocabulary library menu. The LLM may
        # (and SHOULD, at T3+) set the facet's enum-tag fields
        # (lighting_directive, realism_camera, etc.) to one of the
        # listed concept tags; the composer translates each tag into
        # family-shaped phrasing. The block is tier-aware: at T3+ it
        # adds a "REQUIRED" line for nsfw_anatomy, at T4 also for
        # nsfw_act. content_level="" passes through as the generic
        # block for back-compat with direct callers.
        vocab_block = llm_vocabulary_block(
            family.id, content_level=content_level or None,
            compatible_environments=compatible_environments,
        )
        if vocab_block:
            parts.append(f"\n{vocab_block}")
        return "".join(parts)

    @staticmethod
    def _build_user_prompt(
        *,
        scene: dict[str, Any],
        family: "FamilyConfig",
        schema_body: str,
        content_level: str,
    ) -> str:
        """Render the user prompt with the scene's locked core inlined.

        Only the core fields the LLM needs to see are passed —
        anything else on the scene dict is filtered out so the LLM
        doesn't get confused by family-shaped fields from a sibling
        family already on the scene.

        Phase A: ``content_level`` is now surfaced verbatim so the
        LLM sees which T1-T4 tier it's writing for. The system
        prompt's tier directive elaborates on what that means.

        Round-8 tier-active body: rewrite the schema body's
        conditional markers (``[REQUIRED — T3+]`` /
        ``[REQUIRED — T4 only]``) into unconditional ``[REQUIRED]``
        or ``[OPTIONAL]`` based on the active content_level. Cydonia
        empirically gambles on conditional markers (interprets "T3+"
        as a hedge) and nulls fields tagged with them even at T4.
        Collapsing to [REQUIRED] vs [OPTIONAL] removes the hedge.
        """
        core_keys = (
            "variation_axis", "pose", "camera", "camera_angle",
            "lighting", "environment_detail", "mood_note", "expression",
            "composition_intent", "framing_hint", "audience_target",
        )
        core = {k: scene.get(k) for k in core_keys if scene.get(k) is not None}

        # Round-7: per-tier required-fields list at the top of the
        # user prompt — independent of the schema body's per-field
        # markers — gives the LLM a second high-attention signal.
        required_for_tier = _TIER_REQUIRED_FIELDS.get(content_level, ())
        tier_required_list = "\n".join(
            f"  - {f}" for f in required_for_tier
        ) or "  (none for this tier)"

        # Round-8: tier-active schema body. At each content_level
        # collapse the conditional [REQUIRED — ...] markers into
        # unconditional [REQUIRED] / [OPTIONAL (not required at this
        # tier — null is acceptable)]:
        #   T1/T2: every-tier → [REQUIRED]; T3+ and T4-only both demoted.
        #   T3:    every-tier + T3+ → [REQUIRED]; T4-only demoted.
        #   T4:    every [REQUIRED — ...] → [REQUIRED]; no demotions.
        active_body = _make_tier_active_schema_body(
            schema_body, content_level,
        )
        return _USER_PROMPT_TEMPLATE.format(
            content_level=content_level,
            scene_core_json=json.dumps(core, indent=2),
            family_id=family.id,
            prompt_style=family.prompt_style,
            schema_body=active_body,
            tier_required_list=tier_required_list,
        )
