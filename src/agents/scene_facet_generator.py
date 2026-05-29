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
    # Round-12 (2026-05-21) — added realism_camera + realism_lens.
    # The 2026-05-20 Cydonia + Qwen3 A/B audit showed 23/25 (Cydonia)
    # and 24/24 (Qwen3) facets emitting NULL realism camera/lens
    # despite the family few-shot exemplars demonstrating CAMERA_SONY_A7RV +
    # LENS_85MM_F14. The retry-nudge already lifts narrative_moment to
    # high land-rate; promoting camera + lens to tier-required at T3+
    # gives the same nudge treatment to the realism axis, restoring
    # per-scene camera-lens variation that the auto-appended chroma
    # realism_tail can't substitute for. Pony's schema omits these
    # fields entirely so ``_make_tier_strict_schema`` auto-skips them
    # for Pony. realism_film_stock + environment_prop stay [OPTIONAL]
    # — promoting more fields multiplies the retry rate exponentially
    # and these two are lower-leverage than camera + lens.
    # Round-22 (2026-05-22) — promoted art_style_reference + realism_angle
    # to T3+ required after round-21b verification showed the
    # ENCOURAGED-tier marker landed only 14% / 9% adoption respectively
    # (high run-to-run variance, never reliable). REQUIRED-tier fires
    # the retry-nudge and will force the LLM to pick a tag from the
    # vocabulary menu on every scene. Pony's schema omits both fields
    # so ``_make_tier_strict_schema`` skips them for the Pony family.
    # T3 narrowed similarly — 10 fields → 3 essential.
    "T3_artnude":    (
        "nsfw_anatomy",
        "environment_setting", "narrative_moment",
    ),
    # 2026-05-23 dual-write iter3 — narrowed T4 required from 11
    # fields to 5. Under dual-write, scene_prose IS the rendered
    # body and weaves lighting / mood / camera / angle / style /
    # atmosphere into the narrative. Strict-schema enforcement of
    # all those tags simultaneously was over-budgeting the LLM —
    # observed in series_2236d97fad70 (iter3) where DavidAU
    # consistently dropped 3-5 tags per attempt under prose load.
    # Demoted to optional (composer handles None gracefully):
    #   lighting_directive (prose has lighting language)
    #   mood_aesthetic (prose has mood)
    #   realism_camera (tail still appends generic)
    #   realism_lens (composer F1 tail fallback handles)
    #   realism_angle (prose has angle)
    #   art_style_reference (prose has style)
    #   environment_atmosphere (prose has atmosphere)
    # Kept REQUIRED:
    #   scene_prose (the body — no fallback)
    #   nsfw_anatomy (T4 safety + content gate)
    #   nsfw_act (T4 content + pose-act validator)
    #   environment_setting (F15 place_constraint check anchor)
    #   narrative_moment (diversity tracker — the most-locked axis)
    "T4_explicit":   (
        "nsfw_anatomy", "nsfw_act",
        "environment_setting", "narrative_moment",
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
    # Round-12 (2026-05-21) — realism camera/lens retry-nudge anchors.
    # Picked from the 2026-05-20 A/B run's family few-shot exemplars
    # (Chroma's `expected_facet` block in families.yaml). Per verifier
    # I3 cap inlined examples at 3 per field.
    "realism_camera": (
        "CAMERA_SONY_A7RV", "CAMERA_LEICA_M11", "CAMERA_HASSELBLAD_X2D",
    ),
    "realism_lens": (
        "LENS_85MM_F14", "LENS_50MM_F18", "LENS_135MM_F2",
    ),
    # Round-21 (2026-05-21) — example anchors for the two STRONGLY
    # ENCOURAGED axes whose audit-observed adoption was 0%. These ride
    # in the retry-nudge only when the LLM omitted them; we don't add
    # them to _TIER_REQUIRED_FIELDS because cumulative required-axis
    # bloat would explode the retry rate.
    "art_style_reference": (
        "ART_FINE_NUDE", "ART_BOUDOIR_NOIR", "ART_EDITORIAL_FASHION",
    ),
    "realism_angle": (
        "ANGLE_EYE_LEVEL", "ANGLE_LOW", "ANGLE_HIGH",
    ),
    "realism_framing": (
        "FRAMING_FULL_BODY", "FRAMING_MEDIUM_CLOSE", "FRAMING_CLOSE_UP",
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


def _extract_validation_reasons(raw_error: str) -> list[str]:
    """Extract human-readable violation reasons from a Pydantic
    ValidationError message string. Returns a list of plain-language
    bullets the retry-nudge can show the LLM.

    The Pydantic error format we receive looks like:
        N validation errors for SceneFacetFluxNatural_T4_explicit_strict
          scene_prose
            Value error, SceneFacetFluxNatural cross-field coherence
            failed: environment_atmosphere='ATM_X': reason — but
            environment_setting='ENV_Y' has none of [...]. Re-pick...

    We isolate the "Value error, ..." line per field. Truncates each
    reason to ~250 chars so the retry nudge stays bounded.
    """
    if not raw_error:
        return []
    import re as _re
    out: list[str] = []
    # Pydantic packs "Value error, <message>" inline. Pull each occurrence.
    for match in _re.finditer(r"Value error,\s*(.+?)(?:\s*\[type=value_error|$)", raw_error, _re.DOTALL):
        msg = match.group(1).strip()
        # Trim Pydantic's trailing input_value blob if present.
        msg = _re.split(r"\s*\[type=value_error|\s*\[input_value", msg)[0].strip()
        msg = msg[:300]  # Bound length
        if msg and msg not in out:
            out.append(msg)
    # If no Value-error format matched, also try "<field>\n    Field required"
    if not out:
        for match in _re.finditer(r"^\s*([a-z_]+)\s*\n\s+(.+?)$", raw_error, _re.MULTILINE):
            field, reason = match.group(1), match.group(2)
            if "required" in reason.lower():
                out.append(f"field {field!r} is REQUIRED — must not be null")
    return out[:5]  # Cap at 5 violations to keep retry-prompt bounded


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


# Round-5 verifier (F2 BLOCKER-adjacent) — facet free-text fields
# (scene_prose / booru_tags / camera_spec / clothing) are LLM-emitted
# prose that gets injected into the composed positive prompt body.
# The vocab v7 deletions removed mirror / grid concept tags from the
# LLM's MENU, but the LLM still writes "she gazes softly at her
# reflection" / "floor-length mirror" / "Composed as a polyptych"
# into free-text fields. Without sanitization at this exit point, the
# only defense was the downstream `_positive_subject_count_scan` at
# compose time — which doesn't fire on every scene's free-text in
# isolation (it sees the assembled prompt).
#
# This list mirrors the per-family free-text fields declared on the
# 5 SceneFacet schemas. Tag/enum fields (lighting_directive, etc.)
# are NOT included because they're canonicalized at compose time and
# the canonicalizer drops unknown tags.
_FACET_SANITIZABLE_FIELDS: tuple[str, ...] = (
    "scene_prose",
    "booru_tags",
    "camera_spec",
    "clothing",
)


def _sanitize_facet_freetext(
    facet: dict[str, Any], *, scene_id: str | None = None, family_id: str | None = None,
) -> dict[str, Any]:
    """Strip grid / mirror / multi-subject phrases from each free-text
    facet field via the shared `sanitize_grid_phrases` helper. Logs
    at WARN when anything was stripped so drift surfaces in run_log.

    Returns the facet dict with sanitized values. Empty results after
    sanitization are preserved as empty strings (not dropped) so the
    schema's Optional fields stay declared rather than vanishing.
    """
    from src.prompt.builder import sanitize_grid_phrases
    for fld in _FACET_SANITIZABLE_FIELDS:
        raw = facet.get(fld)
        if not isinstance(raw, str) or not raw:
            continue
        cleaned, changed = sanitize_grid_phrases(raw)
        if changed:
            logger.warning(
                "SceneFacetGenerator: grid/mirror language stripped from "
                "%s. scene_id=%s family=%s before=%r after=%r",
                fld, scene_id, family_id, raw, cleaned,
            )
            facet[fld] = cleaned
    return facet


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
- EMOTIONAL CONTENT DISCIPLINE: Commercial adult-art markets sell
  confidence + sensuality + playful intimacy, NOT sorrow. NEVER
  describe the subject as crying, tearful, weeping, sobbing,
  mournful, grieving, or sad. NEVER write "tears glistening",
  "dried tears streaking cheeks", "quiet grief", "mournful gaze",
  "tear-streaked", "sorrow", "weeping", "wet eyes", "sob", or
  any equivalent sorrow-imagery in scene_prose. Acceptable mood
  anchors: confident, sensual, playful, contemplative-at-ease,
  serene, defiant, ecstatic, languid, intimate, pensive (but NOT
  pensive-with-tears). If the scene's mood_note hints at melancholy
  or sadness, reframe as "contemplative" or "introspective" without
  the tearful imagery. The mood_aesthetic MOOD_PENSIVE / MELANCHOLIC
  tags exist in the menu but the LLM should NOT translate them as
  crying or tears in prose — reframe as "soft inward gaze", "quiet
  composure", "thoughtful stillness", "lost in thought".

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


def _narrow_schema_body_examples(
    body: str,
    *,
    compatible_environments: list[str] | None,
    compatible_narratives: list[str] | None,
) -> str:
    """Round-13 (2026-05-21) — rewrite the inline example list in the
    ``environment_setting`` / ``narrative_moment`` schema-body lines so
    they advertise only category-compatible tags.

    Pre-fix the schema body for the non-Pony bodies hard-coded example
    lists like::

        "narrative_moment": "[REQUIRED — every tier] One NARR_* concept
        tag for the captured editorial moment (NARR_READING_LETTER_AT_DAWN,
        NARR_STEPPING_FROM_BATH, NARR_LIGHTING_CIGARETTE, etc.) ..."

    The post-fix audit (2026-05-21) showed Qwen3 picking
    ``NARR_STEPPING_FROM_BATH`` on 17/25 chapel-themed scenes despite
    the vocab block's ``narrative.moment`` namespace being narrowed —
    because the schema-body inline examples kept advertising it.

    This helper substitutes the in-parens example list of each line
    with up to 5 picks from the compatibility whitelist. Empty / None
    whitelist falls through unchanged (back-compat).
    """
    import re

    def _rewrite_line(
        line_field: str, whitelist: list[str] | None, max_examples: int = 5,
    ) -> None:
        nonlocal body
        if not whitelist:
            return
        # Take up to N tags from the whitelist; if fewer, use what's there.
        examples = ", ".join(whitelist[:max_examples])
        if not examples:
            return
        # Match the schema-body line and replace the parenthetical that
        # follows the leading prose. Pattern: `"field": "[REQUIRED... ] One
        # X_* concept tag ... (A_TAG, B_TAG, etc.). rest"`. We swap the
        # inside of the parens — leave the leading description + trailing
        # prose intact.
        pattern = (
            rf'("{re.escape(line_field)}":\s*"\[[^\]]+\][^"]*?'
            rf'(?:concept tag for[^"(]+)?)'
            rf'\([^)]*\)'
        )
        replacement = rf'\1({examples}, etc.)'
        new_body, n = re.subn(pattern, replacement, body, count=1)
        if n == 1:
            body = new_body

    _rewrite_line("environment_setting", compatible_environments)
    _rewrite_line("narrative_moment", compatible_narratives)
    return body


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
{diversity_nudge}
Series subject anchor (locked once per series — every scene shares the
SAME subject identity; let it inform your nsfw_anatomy / nsfw_act
choices so they're coherent with this subject):
  {subject_description}

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


# ── Round-12 (2026-05-21): per-series tag-frequency dominance cap ────


# Diversity axes the engine tracks per series. When one tag exceeds
# ``_DIVERSITY_DOMINANCE_THRESHOLD`` of facets-so-far, the next scene's
# user prompt gets a nudge listing the overused tags + the alternative
# vocab choices. The 2026-05-20 A/B run showed both Cydonia and Qwen3
# locking onto one tag per axis (Cydonia: 18/25 MOOD_PENSIVE;
# Qwen3: 24/24 LIGHT_WINDOW_SIDE / 14/24 NARR_STEPPING_FROM_BATH).
_DIVERSITY_TRACKED_AXES: tuple[str, ...] = (
    "lighting_directive",
    "mood_aesthetic",
    "nsfw_anatomy",
    # nsfw_act added 2026-05-23: T4 verification series
    # (series_c572709666a2) showed NSFW_T4_SOLO_DISPLAY at 13/24 =
    # 54% — untouched because the tracker only watched the 8 axes
    # above. Adding nsfw_act extends the diversity nudge to T4
    # explicit-act tags so the LLM rotates through SOLO_DISPLAY /
    # SOLO_GAZE / SOLO_RECLINING / SOLO_TOUCH / SOLO_BATH / SOLO_
    # OUTDOOR instead of locking on one. Null below T4 (the axis is
    # silently skipped by the tracker's None-guard).
    "nsfw_act",
    "environment_setting",
    "environment_atmosphere",
    "narrative_moment",
    "realism_camera",
    "realism_lens",
)

# Fire the nudge when one tag is at or above this fraction of
# facets-so-far. Round-21 (2026-05-21) tightened 0.5 → 0.35 after the
# Ollama Cydonia heretic audit on series_799bec97e6d7 showed
# narrative_moment:NARR_LIGHTING_CIGARETTE_BALCONY landing 10/24 (42%)
# — under the prior 0.5 floor but visibly dominating the series.
# Round-21b (2026-05-21) — further lowered 0.35 → 0.30 after the
# verification run on series_d13e84ccc70f showed narrative_moment
# at 39% (9/23) and environment_setting at 39% (9/23) — over the
# old floor but the LLM still locks. 0.30 = "more than 3 in 10".
# Lower threshold → more nudges → more model retries; trade-off
# accepted for the higher per-axis variety.
_DIVERSITY_DOMINANCE_THRESHOLD: float = 0.30

# Don't nudge until at least this many facets have been emitted —
# at 0.35 dominance, the threshold is only meaningful once the series
# has enough mass that a single tag landing on every facet still feels
# like over-representation. Round-21 raised 4 → 6 for the same reason.
_DIVERSITY_MIN_FACETS_BEFORE_NUDGE: int = 6


# 2026-05-29 — PROSE-derived diversity axes. The structured-tag axes
# above (lighting_directive / mood_aesthetic / realism_angle / etc.)
# are near-useless for prose families: under the dual-write contract
# the LLM weaves pose/emotion/angle into scene_prose and leaves the
# enum fields NULL (audit of series_41599430bd89: nsfw_posture 0/26,
# realism_angle 0/26, lighting_directive 5/26 populated). So the tag
# tracker never trips, yet the SET is monotone (8/26 reclining,
# "serene" in ~20/26). These axes classify the actual scene_prose
# text into coarse buckets so the diversity nudge sees the real
# composition signal.
#
# Deliberately ADVISORY-ONLY: prose axes feed the soft nudge
# (overused_summary) but NOT the hard-reject path (overused_picks_in /
# overused_tags), so a monotone series gets a "vary this" hint without
# spiking the fallback rate — consistent with the 2026-05-25 soft-warn
# coherence philosophy.
#
# Axes chosen are THEME-INDEPENDENT (pose / emotion / camera angle): a
# golden-hour-meadow series can vary all three freely. Time-of-day /
# lighting is intentionally NOT tracked — it's theme-bound (a
# "Golden Hour" series SHOULD be golden hour; nudging away fights the
# brief).
_DIVERSITY_PROSE_AXES: tuple[str, ...] = (
    "prose_pose",
    "prose_emotion",
    "prose_angle",
)

# Keyword → bucket maps for the prose classifier. First matching bucket
# wins, so ordering is most-specific/dynamic first (a "stands and
# arches her back" line classifies as 'arching', not 'standing').
_PROSE_POSE_BUCKETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("motion",    ("runs", "running", "run through", "leap", "lunge",
                   "walking", "strides", "stride", "dancing", "mid-motion",
                   "mid-stride", "jump", "twirl", "spinning", "in motion")),
    ("arching",   ("arch",)),                       # arches her back
    ("kneeling",  ("kneel",)),
    ("crouching", ("crouch", "squat")),
    ("sitting",   ("sits", "seated", "perch")),
    ("reclining", ("reclin", "lying", "lies back", "lie back", "lies low",
                   "lounge", "lounging", "lounges", "sprawl", "supine",
                   "rests against", "lying back", "rests her", "lies in")),
    ("standing",  ("stands", "standing", "upright", "on her feet")),
)
_PROSE_EMOTION_BUCKETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # non-serene emotions first so the boilerplate subject_description
    # echo ("serene expression") doesn't dominate every classification.
    ("ecstatic",      ("ecstas", "euphoric", "rapture", "unbridled",
                       "primal", "breathless", "visceral")),
    ("joyful",        ("joyful", "laughter", "laughing", "playful",
                       "joy", "delight", "grin")),
    ("defiant",       ("defiant", "fierce", "commanding", "unyielding",
                       "bold", "triumphant", "powerful stance", "heroic")),
    ("sultry",        ("sultry", "seductive", "erotic", "smolder",
                       "sensual")),
    ("contemplative", ("contemplat", "pensive", "wistful", "dreamy",
                       "lost in", "unfocused")),
    ("serene",        ("serene", "calm", "tranquil", "peaceful",
                       "quiet composure", "stillness")),
)
_PROSE_ANGLE_BUCKETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("low_angle",  ("low angle", "low-angle", "from below", "heroic")),
    ("high_angle", ("high angle", "high-angle", "looking down",
                    "from above", "overhead", "bird's eye")),
    ("close_up",   ("close-up", "close up", "intimate portrait",
                    "extreme close")),
    ("wide",       ("wide shot", "wide angle", "vast expanse",
                    "environmental")),
    ("eye_level",  ("eye level", "eye-level")),
)

# Friendly axis label + suggested alternatives shown in the nudge.
_PROSE_AXIS_HINT: dict[str, tuple[str, str]] = {
    "prose_pose": (
        "body position",
        "use a different pose (standing / kneeling / walking / sitting / "
        "crouching / arching)",
    ),
    "prose_emotion": (
        "emotional register",
        "try a different mood (playful / defiant / joyful / sultry / "
        "contemplative)",
    ),
    "prose_angle": (
        "camera angle",
        "vary the angle (low / high / eye-level / close-up / wide)",
    ),
}


def _classify_prose_composition(scene_prose: str | None) -> dict[str, str]:
    """Bucket a scene_prose paragraph into coarse pose / emotion /
    angle classes for prose-level diversity tracking.

    Returns ``{axis: bucket}`` for whichever axes matched (omits an axis
    when no keyword hit). First matching bucket per axis wins. Used by
    :class:`_DiversityTracker` so the set-level nudge can see the
    composition signal that lives in free prose rather than in the
    (often NULL) structured enum fields.
    """
    if not scene_prose:
        return {}
    text = scene_prose.lower()
    out: dict[str, str] = {}
    for axis, buckets in (
        ("prose_pose", _PROSE_POSE_BUCKETS),
        ("prose_emotion", _PROSE_EMOTION_BUCKETS),
        ("prose_angle", _PROSE_ANGLE_BUCKETS),
    ):
        for bucket, keywords in buckets:
            if any(kw in text for kw in keywords):
                out[axis] = bucket
                break
    return out


# 2026-05-29 — lighting-coherence enforcement for "soft" style profiles.
# A profile may declare ``lighting_register: soft`` (e.g.
# analog_film_intimate). When it does, scene_prose must NOT use
# dark / dramatic / high-contrast lighting language — the facet LLM's
# "fine-art nude = chiaroscuro" prior overrides even an explicit
# "NO Rembrandt/low-key" instruction in the system-prompt lock (render
# audit: 3/8 scenes still went dramatic). This is the defence-in-depth
# retry layer, mirroring the mirror-ban + pose-act geometric retries:
# detect the dark tokens, reject the facet, retry with a LIGHTING-RETRY
# nudge. Soft enforcement — one retry then soft-ship, so it nudges the
# register without spiking the fallback rate.
#
# Tokens are curated for PRECISION (avoid false positives that would
# over-fire): plain "shadow" / "dim glow" are NOT banned (soft shadows
# are fine); only unambiguously dark/dramatic/high-contrast phrasings.
_DARK_LIGHTING_TOKENS: tuple[str, ...] = (
    "rembrandt",
    "chiaroscuro",
    "low-key", "low key",
    "deep shadow",          # deep shadow(s)
    "dramatic shadow",
    "hard shadow", "harsh shadow",
    "dramatic contrast", "high-contrast", "high contrast", "stark contrast",
    "candlelit", "candlelight", "flickering candle", "single candle",
    "by candlelight",
    "dimly lit", "dimly-lit", "dim, ",     # "dim, opulent ..." constructions
    "spotlight",
    "backlit silhouette", "silhouette against", "in silhouette",
    "film noir", "noir lighting", "cinematic noir",
    "profound darkness", "plunged into darkness", "shrouded in darkness",
    "near darkness", "engulfed in shadow",
)


# Negators that flip a dark-lighting token's meaning when they appear
# just before it ("gentle fill WITHOUT harsh shadows", "NO deep shadow").
# A token preceded by one of these within a short window is the LLM
# COMPLYING (describing soft light by negating the dark form), so it must
# not count as a violation.
_DARK_LIGHTING_NEGATORS: tuple[str, ...] = (
    "without", "no ", "not ", "free of", "free from", "never", "avoid",
    "rather than", "instead of", "absent ", "devoid of", "none of",
    "minimal", "lacking",
)
# Window (chars) before a token to scan for a negator.
_NEGATION_WINDOW: int = 18


def _scene_prose_dark_lighting_hits(scene_prose: str | None) -> list[str]:
    """Return the dark/dramatic lighting tokens present in ``scene_prose``.

    Used by :meth:`SceneFacetGenerator.generate` only when the active
    style profile declares ``lighting_register == "soft"``. Empty list =
    coherent (no dark-lighting language). Substring match on lowercased
    prose; deduped, order-preserving.

    Negation-aware: a token preceded by a negator within
    :data:`_NEGATION_WINDOW` chars (e.g. "without harsh shadows", "no
    deep shadow") is the LLM describing SOFT light by negating the dark
    form — it does NOT count as a violation. A token counts only if it
    has at least one NON-negated occurrence."""
    if not scene_prose:
        return []
    text = scene_prose.lower()
    hits: list[str] = []
    for tok in _DARK_LIGHTING_TOKENS:
        start = 0
        token_is_violation = False
        while True:
            i = text.find(tok, start)
            if i == -1:
                break
            window = text[max(0, i - _NEGATION_WINDOW):i]
            if not any(neg in window for neg in _DARK_LIGHTING_NEGATORS):
                token_is_violation = True
                break
            start = i + len(tok)
        if token_is_violation and tok not in hits:
            hits.append(tok)
    return hits


class _DiversityTracker:
    """Per-series running tag-frequency tracker for the structured-tag
    axes the LLM picks at facet time.

    The engine constructs one of these per ``(series, family, llm)``
    target loop, calls :meth:`record` after each successful facet, and
    asks :meth:`overused_summary` BEFORE the next facet to surface any
    axis whose dominant tag has crossed the dominance threshold.
    """

    __slots__ = ("_counts", "_total")

    def __init__(self) -> None:
        # Per-axis Counter — {axis_name: {tag: count}}. Holds BOTH the
        # structured-tag axes and the prose-derived axes; the hard-
        # reject path (overused_tags) only ever reads the tag axes.
        self._counts: dict[str, dict[str, int]] = {
            axis: {}
            for axis in (*_DIVERSITY_TRACKED_AXES, *_DIVERSITY_PROSE_AXES)
        }
        # Total facets recorded so far (denominator for the ratio check).
        self._total: int = 0

    def record(self, facet: dict[str, Any]) -> None:
        """Increment counters for whatever tags this facet picked. Tags
        not on the tracked-axes list are ignored. ``None`` / empty
        values are ignored so they don't dominate the count.

        Also classifies the facet's ``scene_prose`` into prose-derived
        pose / emotion / angle buckets (2026-05-29) so the diversity
        nudge sees the composition signal even when the structured enum
        fields are NULL (the common case for prose families)."""
        self._total += 1
        for axis in _DIVERSITY_TRACKED_AXES:
            val = facet.get(axis)
            if not val:
                continue
            tag = str(val).strip()
            if not tag:
                continue
            self._counts[axis][tag] = self._counts[axis].get(tag, 0) + 1
        # Prose-derived axes — classify the scene_prose body.
        prose_buckets = _classify_prose_composition(facet.get("scene_prose"))
        for axis, bucket in prose_buckets.items():
            self._counts[axis][bucket] = self._counts[axis].get(bucket, 0) + 1

    def overused_summary(self) -> str:
        """Return a multi-line nudge string for the LLM, or ``""``
        when no axis has crossed the dominance threshold.

        Format::

            Diversity nudge — pick something DIFFERENT this scene; these
            tags are already over-represented:
              - lighting_directive: LIGHT_WINDOW_SIDE used 12/22 scenes (54%)
              - mood_aesthetic: MOOD_PENSIVE used 14/22 scenes (63%)
        """
        return self._overused_summary_text()

    def overused_tags(self) -> dict[str, str]:
        """Round-13 (2026-05-21) — companion to :meth:`overused_summary`
        that returns the structured ``{axis: dominant_tag}`` map for
        validator-retry logic.

        Empty dict when no axis has crossed the dominance threshold or
        fewer than the min-facets gate have been recorded. Used by
        :meth:`SceneFacetGenerator.generate` to detect when a freshly
        generated facet landed on an already-over-represented tag and
        should be rejected for a second-attempt retry.
        """
        if self._total < _DIVERSITY_MIN_FACETS_BEFORE_NUDGE:
            return {}
        out: dict[str, str] = {}
        for axis in _DIVERSITY_TRACKED_AXES:
            counts = self._counts[axis]
            if not counts:
                continue
            top_tag, top_count = max(counts.items(), key=lambda kv: kv[1])
            if (top_count / self._total) >= _DIVERSITY_DOMINANCE_THRESHOLD:
                out[axis] = top_tag
        return out

    def overused_picks_in(self, facet: dict[str, Any]) -> dict[str, str]:
        """Return the ``{axis: dominant_tag}`` map for axes where THIS
        ``facet`` picked an already-over-represented tag.

        Mode-switch from advisory nudge to validator-retry (round-13):
        the engine's facet-generate loop will reject the first attempt
        and force a retry-nudge when this method returns non-empty,
        explicitly telling the LLM "do NOT pick X for axis Y this scene".
        """
        if not facet:
            return {}
        overused = self.overused_tags()
        if not overused:
            return {}
        hits: dict[str, str] = {}
        for axis, dominant_tag in overused.items():
            val = facet.get(axis)
            if val is None:
                continue
            if str(val).strip() == dominant_tag:
                hits[axis] = dominant_tag
        return hits

    def _overused_summary_text(self) -> str:
        if self._total < _DIVERSITY_MIN_FACETS_BEFORE_NUDGE:
            return ""
        # Structured-tag axes (existing) — "pick a different tag".
        tag_lines: list[str] = []
        for axis in _DIVERSITY_TRACKED_AXES:
            counts = self._counts[axis]
            if not counts:
                continue
            top_tag, top_count = max(counts.items(), key=lambda kv: kv[1])
            ratio = top_count / self._total
            if ratio >= _DIVERSITY_DOMINANCE_THRESHOLD:
                tag_lines.append(
                    f"  - {axis}: {top_tag} used "
                    f"{top_count}/{self._total} scenes ({int(ratio * 100)}%)"
                )
        # Prose-derived axes (2026-05-29) — "vary this in your prose".
        prose_lines: list[str] = []
        for axis in _DIVERSITY_PROSE_AXES:
            counts = self._counts[axis]
            if not counts:
                continue
            top_bucket, top_count = max(counts.items(), key=lambda kv: kv[1])
            ratio = top_count / self._total
            if ratio >= _DIVERSITY_DOMINANCE_THRESHOLD:
                label, suggestion = _PROSE_AXIS_HINT.get(
                    axis, (axis, "vary it")
                )
                prose_lines.append(
                    f"  - {label}: '{top_bucket}' in {top_count}/"
                    f"{self._total} scenes ({int(ratio * 100)}%) — "
                    f"{suggestion}"
                )
        if not tag_lines and not prose_lines:
            return ""
        parts: list[str] = []
        if tag_lines:
            parts.append(
                "\nDiversity nudge — pick something DIFFERENT this scene; "
                "these tags are already over-represented across the series:\n"
                + "\n".join(tag_lines)
            )
        if prose_lines:
            parts.append(
                "\nCOMPOSITION VARIETY — your scene_prose keeps repeating "
                "these across the series. Make THIS scene visibly "
                "different:\n" + "\n".join(prose_lines)
            )
        return "\n".join(parts) + "\n"


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
  "realism_camera": "[REQUIRED — T3+] One CAMERA_* concept tag for specific camera body (CAMERA_SONY_A7RV, CAMERA_HASSELBLAD_X2D, CAMERA_LEICA_M11, CAMERA_CANON_R5, etc.). Per-scene variety is desirable — do not lock to one camera across the series.",
  "realism_lens": "[REQUIRED — T3+] One LENS_* concept tag for lens spec (LENS_85MM_F14, LENS_50MM_F18, LENS_35MM_F2, LENS_135MM_F2, LENS_24MM_F14, etc.). Match the lens to the shot — wide for environment, long for portrait compression.",
  "realism_angle": "[REQUIRED — T3+] One ANGLE_* concept tag for camera angle (ANGLE_LOW, ANGLE_EYE_LEVEL, ANGLE_HIGH, ANGLE_DUTCH, ANGLE_OVER_SHOULDER). Mix angles across the series — every scene shot at ANGLE_EYE_LEVEL looks visually identical at the framing level. Round-22 promoted from STRONGLY ENCOURAGED to REQUIRED after r-21b verification showed 9% adoption was unreliable.",
  "art_style_reference": "[REQUIRED — T3+] One ART_* concept tag for a per-scene art-style anchor (ART_FINE_NUDE, ART_BOUDOIR_NOIR, ART_OLD_HOLLYWOOD, ART_EDITORIAL_FASHION, ART_CLASSICAL, ART_HELMUT_NEWTON, ART_HERB_RITTS_BW, ART_IRVING_PENN_MINIMALISM, ART_NUDE_PHOTOGRAPHY). Distinct from the SERIES-level photographer_ref/art_movement — this is the per-scene visual style anchor. Vary across scenes. The composer translates the tag into family-shaped phrasing. Round-22 promoted from STRONGLY ENCOURAGED to REQUIRED after r-21b verification showed 14% adoption was unreliable.",
  "nsfw_posture": "[OPTIONAL] One NSFW_POSTURE_* concept tag if the pose calls for it (T3+ only — null at T1/T2).",
  "environment_prop": "[OPTIONAL] One PROP_* concept tag for furniture/object anchor (PROP_CHAISE_LOUNGE_VELVET, PROP_HANDWRITTEN_LETTER, PROP_VELVET_CURTAIN_HEAVY, PROP_FOUR_POSTER_BED, etc.).",
  "composition_principle": "[OPTIONAL] One COMP_* concept tag for higher-order composition (COMP_LEADING_LINES_FLOOR, COMP_NEGATIVE_SPACE_DOMINANT, COMP_SYMMETRY_CENTERED, COMP_LOW_HERO_SHOT, etc.).",
  "realism_film_stock": "[OPTIONAL] One FILM_* concept tag for film-stock emulation (FILM_PORTRA_400, FILM_CINESTILL_800T, FILM_TRIX_400, etc.).",
  "realism_framing": "[ENCOURAGED] One FRAMING_* concept tag for shot framing (FRAMING_FULL_BODY, FRAMING_MEDIUM_CLOSE, FRAMING_CLOSE_UP, FRAMING_WIDE_ENVIRONMENT)."\
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
  "scene_prose": "<ONE coherent paragraph, 100-250 words (HARD band 60-350). THIS IS THE FINAL PROMPT BODY — it is what the image model actually renders. The composer no longer stitches structured-tag canonicalizations into the final prompt for prose families; the structured tags drive validators + analytics only. Your scene_prose must therefore weave EVERY axis into a single flowing narrative:\\n  • Subject: age range + build + hair + body type, matching subject_description EXACTLY.\\n  • Pose: ONE specific physical pose with body orientation (don't list alternatives).\\n  • Anatomy at T4: concrete and specific (e.g. 'natural medium breasts with soft pink nipples', 'smooth vulva visible', 'wide hips') — not vague phrases like 'anatomically correct'.\\n  • Lighting: ONE coherent recipe — must mention the SEMANTIC KEYWORD of your lighting_directive tag (e.g. if you pick LIGHT_REMBRANDT, prose contains 'Rembrandt lighting' OR 'triangle of light on the unlit cheek'). Don't stack multiple recipes (no 'golden hour AND Rembrandt AND dappled' simultaneously).\\n  • Environment: concrete props that match the chosen environment_setting tag (e.g. for ENV_VICTORIAN_PARLOUR mention 'oxblood leather chair' or 'gilt-framed paintings').\\n  • Mood: specific emotional anchor — not 'slight melancholy' but 'eyes lowered past the camera, quiet composure' or 'lips slightly parted in unfocused contemplation'. STRICTLY FORBIDDEN: tears, crying, weeping, mournful, grieving, melancholic, sad, sorrow, numb, vacant, uncertain, tentatively, hesitantly. Commercial NSFW sells confident / sensual / playful / contemplative-at-ease / serene / defiant / ecstatic / languid / intimate moods only.\\n  • Style: at most ONE photographer school reference woven naturally — DON'T stack 'Lindbergh AND Newton AND Caravaggio' (they have opposite aesthetics and the model averages them to mud).\\n  • Camera angle: ONE direction matching your realism_angle tag (don't write 'from above' AND 'low angle' simultaneously).\\nOutput format: ONE paragraph of coherent narrative prose. NO bullet lists, NO numbered output, NO sentence fragments. NO photographer / camera / lens / film-stock brand strings in the prose body — those go in structured tag fields only. NO weighting syntax. NO `(token:1.3)`. The composer prepends safety anchors + appends a brief technical tail; your prose is the meat. 100-250 words target — write enough to cover all axes without padding.>",
{_STRUCTURED_TAG_BODY_NON_PONY}""",

    "flux2_prose": f"""\
  "scene_prose": "<ONE coherent paragraph, 100-250 words (HARD band 60-350). THIS IS THE FINAL PROMPT BODY for Flux2 Klein. Weave the BFL 5-anchor structure (subject → setting → details → lighting → atmosphere) into a single flowing narrative, NOT a numbered list. Subject + lighting near the front (word order weights heavily for Klein). Same coherence rules as flux_natural — ONE photographer school, ONE lighting recipe, ONE camera angle, concrete anatomy at T4, concrete env props matching environment_setting tag. No comma-tag soup. No brand strings (Sony / Leica / Lindbergh / Newton) in prose body — those live in structured tag fields. No `(token:1.3)` weighting.>",
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
        compatible_narratives: list[str] | None = None,
        compatible_art_styles: list[str] | None = None,
        diversity_tracker: "_DiversityTracker | None" = None,
        subject_description: str = "",
        lighting_hint: str = "",
        lighting_register: str = "",
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
            lighting_hint=lighting_hint,
            compatible_environments=compatible_environments,
            compatible_narratives=compatible_narratives,
            compatible_art_styles=compatible_art_styles,
        )
        # Round-12 diversity nudge — pulls running tag-frequency stats
        # from the engine-owned tracker and injects an "avoid these
        # over-used tags this scene" hint into the user prompt when any
        # axis crosses the dominance threshold. Empty string when the
        # tracker says no axis is over-represented yet (the typical
        # path for the first 4 facets of a series).
        diversity_nudge = (
            diversity_tracker.overused_summary()
            if diversity_tracker is not None
            else ""
        )
        user_prompt = self._build_user_prompt(
            scene=scene,
            family=family,
            schema_body=schema_body,
            content_level=content_level,
            diversity_nudge=diversity_nudge,
            compatible_environments=compatible_environments,
            compatible_narratives=compatible_narratives,
            subject_description=subject_description,
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
        # Round-13 (2026-05-21) — promote diversity nudge from advisory
        # user-prompt-only to validator-retry. If the first-attempt
        # facet picked an already-over-represented tag for any tracked
        # axis, reject it and retry with an explicit "do NOT pick X"
        # nudge. The post-fix audit showed the advisory nudge alone
        # left Qwen3 dominance unchanged on mood / env / narrative —
        # validator-retry gives the LLM an explicit re-roll signal.
        dominance_hits: dict[str, str] = (
            diversity_tracker.overused_picks_in(facet or {})
            if (facet is not None and diversity_tracker is not None)
            else {}
        )
        # Pose ↔ nsfw_act / nsfw_posture geometric coherence (clause 3
        # defense-in-depth, 2026-05-23). System-prompt instruction
        # alone proved unreliable on series_c572709666a2: 4/5 reclining
        # poses correctly avoided SOLO_DISPLAY but scene_021 ignored
        # the clause. This rejects the incoherent facet so the existing
        # retry-with-nudge mechanism fires — same shape as F15 + the
        # dominance retry.
        from src.prompt.vocabulary import check_pose_act_coherence
        pose_act_violations: list[tuple[str, str, str]] = (
            check_pose_act_coherence(
                pose=scene.get("pose"),
                nsfw_act=facet.get("nsfw_act") if facet else None,
                nsfw_posture=facet.get("nsfw_posture") if facet else None,
                nsfw_anatomy=facet.get("nsfw_anatomy") if facet else None,
                environment_setting=facet.get("environment_setting") if facet else None,
            )
            if facet is not None
            else []
        )
        # Lighting-coherence (2026-05-29) — when the active style profile
        # declares lighting_register == "soft", scene_prose must avoid
        # dark / dramatic / high-contrast lighting language. The
        # system-prompt SERIES AESTHETIC LOCK alone proved unreliable
        # (render audit: 3/8 scenes still went chiaroscuro despite an
        # explicit "NO Rembrandt/low-key" instruction). Reject + retry,
        # same shape as the pose-act geometric retry.
        lighting_violations: list[str] = (
            _scene_prose_dark_lighting_hits(facet.get("scene_prose"))
            if (facet is not None and lighting_register == "soft")
            else []
        )
        if (
            facet is not None
            and not missing
            and not dominance_hits
            and not pose_act_violations
            and not lighting_violations
        ):
            return _sanitize_facet_freetext(
                _strip_none_values(facet),
                scene_id=scene.get("id"),
                family_id=family.id,
            )

        # Round-13 — build the dominance-rejection clause once; appended
        # to either retry branch (missing-fields OR diversity-only).
        dominance_nudge_lines: list[str] = []
        if dominance_hits:
            dominance_nudge_lines.append(
                "DIVERSITY-RETRY: your first attempt picked a tag that "
                "is already over-represented in this series. Pick a "
                "DIFFERENT tag for each of these axes:"
            )
            for axis, over_tag in dominance_hits.items():
                # Hint with a couple of alternative picks from the same
                # namespace's example list so the LLM has somewhere to
                # land. Falls through to "any other tag from the system-
                # prompt menu" when no examples list is registered.
                examples = _FIELD_EXAMPLE_TAGS.get(axis, ())
                alternatives = [e for e in examples if e != over_tag][:3]
                if alternatives:
                    dominance_nudge_lines.append(
                        f"  - {axis}: NOT {over_tag}; try "
                        f"{', '.join(alternatives)} or another tag from "
                        f"that namespace."
                    )
                else:
                    dominance_nudge_lines.append(
                        f"  - {axis}: NOT {over_tag}; pick any other "
                        f"tag from that namespace's menu."
                    )

        # Pose-act coherence nudge — assembled separately so the retry
        # branch logic stays readable. Same pattern as dominance.
        pose_act_nudge_lines: list[str] = []
        if pose_act_violations:
            pose_act_nudge_lines.append(
                "GEOMETRIC-RETRY: your first attempt picked nsfw_act / "
                "nsfw_posture tag(s) that are physically incompatible "
                "with the scene's pose. The camera cannot photograph "
                "the body in two orientations at once. Re-pick:"
            )
            for field, bad_tag, reason in pose_act_violations:
                pose_act_nudge_lines.append(
                    f"  - {field}: NOT {bad_tag}. Why: {reason}"
                )

        # Lighting-coherence nudge (2026-05-29) — assembled separately,
        # appended to whichever retry branch fires. Names the offending
        # tokens so the LLM knows exactly what to remove.
        lighting_nudge_lines: list[str] = []
        if lighting_violations:
            lighting_nudge_lines.append(
                "LIGHTING-RETRY: this series is LOCKED to a SOFT, "
                "LOW-CONTRAST, evenly-lit register. Your scene_prose used "
                f"dark / dramatic lighting language: {lighting_violations}. "
                "Rewrite the lighting so it is soft, diffused, bright and "
                "low-contrast (soft window daylight, gentle fill, open "
                "shadows). REMOVE all of: Rembrandt, low-key, chiaroscuro, "
                "deep/dramatic/hard shadow, high-contrast, candlelit, dimly "
                "lit, spotlight, backlit silhouette, noir, darkness. Keep "
                "the same pose, subject, environment and mood — change ONLY "
                "the lighting."
            )

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
                # Round-13 — when the missing field is narrative_moment
                # or environment_setting AND a category whitelist is
                # active, use the whitelist intersection for the
                # retry-nudge examples so the LLM doesn't re-anchor on
                # an out-of-category tag from the static default list.
                # Fall through to defaults when whitelist is absent or
                # empty (back-compat).
                if f == "narrative_moment" and compatible_narratives:
                    examples = tuple(compatible_narratives[:3]) or _FIELD_EXAMPLE_TAGS.get(f)
                elif f == "environment_setting" and compatible_environments:
                    examples = tuple(compatible_environments[:3]) or _FIELD_EXAMPLE_TAGS.get(f)
                else:
                    examples = _FIELD_EXAMPLE_TAGS.get(f)
                if examples:
                    nudge_lines.append(
                        f"  - {f}: pick exactly one of "
                        f"{', '.join(examples)} (or any other tag from "
                        f"that namespace's menu in the system prompt)."
                    )
            nudge_lines.extend(dominance_nudge_lines)
            nudge_lines.extend(pose_act_nudge_lines)
            nudge_lines.extend(lighting_nudge_lines)
            nudge_lines.append(
                "Return ONLY a single JSON object with these fields "
                "populated."
            )
            retry_prompt = user_prompt + "\n".join(nudge_lines)
        elif dominance_hits or pose_act_violations or lighting_violations:
            # Round-13: facet is structurally valid (no missing fields)
            # but landed on an already-over-represented tag (diversity)
            # OR a geometrically incoherent pose-act combination
            # (added 2026-05-23). Retry with the matching nudge(s).
            reason_parts: list[str] = []
            if dominance_hits:
                reason_parts.append(f"over-represented tag(s) {dominance_hits}")
            if pose_act_violations:
                reason_parts.append(
                    f"pose-act coherence violation(s) "
                    f"{[(f, t) for f, t, _ in pose_act_violations]}"
                )
            if lighting_violations:
                reason_parts.append(
                    f"dark-lighting language {lighting_violations} "
                    f"(profile lighting_register=soft)"
                )
            reason_msg = " AND ".join(reason_parts)
            logger.warning(
                "Scene facet generator: first attempt for family %s "
                "rejected — %s; retrying with explicit nudge.",
                family.id, reason_msg,
            )
            nudge_lines = ["", ""]
            nudge_lines.extend(dominance_nudge_lines)
            nudge_lines.extend(pose_act_nudge_lines)
            nudge_lines.extend(lighting_nudge_lines)
            nudge_lines.append(
                "Return ONLY a single JSON object."
            )
            retry_prompt = user_prompt + "\n".join(nudge_lines)
        else:
            logger.warning(
                "Scene facet generator: first attempt failed for family "
                "%s, retrying …", family.id,
            )
            # 2026-05-23 — extract specific Pydantic violation from
            # _last_attempt_failure_reason (set by _attempt on OllamaJSON
            # ParseError). The Pydantic ValidationError message contains
            # field name + specific reason; surfacing it in the retry
            # nudge lets the LLM fix the specific problem instead of
            # blindly re-generating with the same mistakes. Doubles
            # facet success rate empirically on F15-coherence and
            # sad-token rejections.
            specific_reason = getattr(
                self, "_last_attempt_failure_reason", ""
            )
            extracted_reasons = _extract_validation_reasons(specific_reason)
            if extracted_reasons:
                retry_prompt = (
                    user_prompt
                    + "\n\nIMPORTANT: Your previous response was REJECTED "
                    "for these specific reasons. Fix each one:\n"
                    + "\n".join(f"  • {r}" for r in extracted_reasons)
                    + "\nReturn ONLY a single JSON object with the fixes "
                    "applied. No markdown, no commentary."
                )
            else:
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
                # 2026-05-23 (Verifier P0.E) — at T4_explicit, if any
                # CRITICAL field is still missing after retry, raise
                # instead of soft-shipping. Critical T4 fields produce
                # scene-023-class degenerate output when missing: prose
                # without anatomy/act has no NSFW content, prose
                # without lighting_directive has no light recipe.
                # Soft-ship at T4 was responsible for series_79ae3b962c8d
                # scene 023's one-word-fragment output. T1-T3 still
                # soft-ship (graceful degradation acceptable for non-
                # explicit tiers).
                # Dual-write pivot iter2 (2026-05-23): loosened critical
                # set. lighting_directive removed from T4-critical because
                # the LLM weaves lighting into scene_prose under the new
                # contract. Composer no longer emits lighting_directive
                # as a separate sentence; if it's missing as a tag but
                # the prose mentions lighting, the prompt is still
                # complete. Only scene_prose + nsfw_act remain critical
                # at T4 (must have a body and an explicit act).
                _T4_CRITICAL = {
                    "scene_prose",
                    "nsfw_act",
                }
                critical_missing = [
                    f for f in still_missing if f in _T4_CRITICAL
                ]
                if content_level == "T4_explicit" and critical_missing:
                    logger.error(
                        "Scene facet generator: family %s STILL missing "
                        "T4-critical field(s) %s after retry — refusing "
                        "to soft-ship at T4. Scene will be skipped; "
                        "operator can re-prep with --regen-facets.",
                        family.id, critical_missing,
                    )
                    raise SceneFacetGeneratorError(
                        f"T4 facet missing critical fields {critical_missing} "
                        f"after retry — refusing to ship degenerate output."
                    )
                logger.warning(
                    "Scene facet generator: family %s still missing "
                    "tier-required field(s) %s after retry; "
                    "shipping the facet anyway (operator can re-prep "
                    "this scene with --regen-facets to retry).",
                    family.id, still_missing,
                )
            # Round-22 (2026-05-22) — bump diversity retry budget from
            # 1 to 2. When the SECOND attempt STILL picks an over-
            # represented tag, fire a THIRD attempt with a much
            # harder-worded "HARD BAN" nudge. Only soft-ship after 3
            # attempts. Round-21b audit showed the soft retry was
            # ignored by the LLM ~50% of the time (NARR_AFTER_THE_PARTY
            # landed 12/24 in series_57d3aea57c85 despite first-retry
            # nudges firing); a stricter ban-list re-roll should close
            # the gap. Cost: ~30-50s wall-clock on scenes where the
            # LLM hits dominance twice (acceptable for offline pipeline).
            if diversity_tracker is not None:
                still_dominant = diversity_tracker.overused_picks_in(facet)
                if still_dominant:
                    logger.warning(
                        "Scene facet generator: family %s second attempt "
                        "STILL picked over-represented tag(s) %s; "
                        "firing third attempt with HARD BAN nudge.",
                        family.id, still_dominant,
                    )
                    hard_ban_lines = [
                        "",
                        "",
                        "HARD BAN — your previous TWO attempts both picked "
                        "tags that are already over-represented in this "
                        "series. You MUST NOT pick the following tags on "
                        "this attempt:",
                    ]
                    for axis, over_tag in still_dominant.items():
                        examples = _FIELD_EXAMPLE_TAGS.get(axis, ())
                        alternatives = [e for e in examples if e != over_tag][:5]
                        if alternatives:
                            hard_ban_lines.append(
                                f"  - {axis}: BANNED = {over_tag}. You MUST "
                                f"pick one of: {', '.join(alternatives)} or "
                                f"another tag from that namespace that is "
                                f"NOT {over_tag}."
                            )
                        else:
                            hard_ban_lines.append(
                                f"  - {axis}: BANNED = {over_tag}. Pick "
                                f"any other tag from that namespace."
                            )
                    hard_ban_lines.append(
                        "This is your final attempt — adhere to the ban "
                        "or the facet ships with reduced diversity. "
                        "Return ONLY a single JSON object."
                    )
                    third_prompt = user_prompt + "\n".join(hard_ban_lines)
                    facet_third = self._attempt(
                        system_prompt, third_prompt, schema,
                        effective_temp, model=model,
                    )
                    if facet_third is not None:
                        # Use the third attempt regardless of whether it
                        # cleared dominance — if it did, great. If it
                        # didn't, we soft-fail-log below.
                        facet = facet_third
                        final_dominant = diversity_tracker.overused_picks_in(
                            facet
                        )
                        if final_dominant:
                            logger.warning(
                                "Scene facet generator: family %s THIRD "
                                "attempt STILL picked over-represented "
                                "tag(s) %s after HARD BAN — LLM is "
                                "stubbornly locked; shipping the facet "
                                "anyway.",
                                family.id, final_dominant,
                            )
                        else:
                            logger.info(
                                "Scene facet generator: family %s third "
                                "attempt with HARD BAN cleared dominance.",
                                family.id,
                            )
            # 2026-05-23 (Verifier I4) — subject-continuity check.
            # Catch hard physical contradictions between the series'
            # subject_description and the scene_prose (scene_015 of
            # series_79ae3b962c8d said "muscular frame" when the
            # subject_description anchored "mature curves"). Log at
            # WARNING; don't reject the facet (the contradiction is
            # cosmetic — series consistency, not safety).
            if subject_description and facet.get("scene_prose"):
                from src.prompt.vocabulary import check_subject_continuity
                continuity_violations = check_subject_continuity(
                    subject_description=subject_description,
                    scene_prose=facet.get("scene_prose"),
                )
                if continuity_violations:
                    for field, word, reason in continuity_violations:
                        logger.warning(
                            "Scene facet generator: family %s subject "
                            "continuity violation — %s. Shipping anyway "
                            "(cosmetic, not safety).",
                            family.id, reason,
                        )

            # 2026-05-23 — post-retry pose-act re-check (P0.B fix).
            # System prompt instruction + first-attempt validator was
            # silent in production: most first attempts failed on
            # missing tier-required fields, the retry path only re-
            # checks `missing` — so retry's facet could land an
            # incompatible pose-act combination (lounging + SOLO_
            # DISPLAY in scene_019/021 verification) and ship.
            # Strategy: re-check pose-act-anatomy-bath here; on
            # violation, null out the offending fields so the
            # canonicalizer silently drops them rather than emitting
            # contradictory prose. Cheaper than firing another LLM
            # attempt (the system is already over-budget on retry
            # rounds — 86% shipped-anyway in series_79ae3b962c8d).
            from src.prompt.vocabulary import (
                check_pose_act_coherence as _re_check_pose_act,
            )
            final_violations = _re_check_pose_act(
                pose=scene.get("pose"),
                nsfw_act=facet.get("nsfw_act"),
                nsfw_posture=facet.get("nsfw_posture"),
                nsfw_anatomy=facet.get("nsfw_anatomy"),
                environment_setting=facet.get("environment_setting"),
            )
            if final_violations:
                fields_nulled = []
                for field, bad_tag, reason in final_violations:
                    logger.error(
                        "Scene facet generator: family %s POST-RETRY "
                        "pose-act coherence violation persisted — "
                        "nulling %s=%s. Reason: %s",
                        family.id, field, bad_tag, reason,
                    )
                    facet[field] = None
                    fields_nulled.append(field)
                logger.warning(
                    "Scene facet generator: family %s shipped with %d "
                    "post-retry-nulled field(s) %s — operator can "
                    "re-prep with --regen-facets to recover.",
                    family.id, len(fields_nulled), fields_nulled,
                )
            return _sanitize_facet_freetext(
                _strip_none_values(facet),
                scene_id=scene.get("id"),
                family_id=family.id,
            )

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
            # Round-19 (2026-05-22) — bumped 2048 → 4096 to give
            # reasoning-model variants (Qwen 3.5+ thinking-mode tunes)
            # enough budget for their <think>...</think> trace before
            # the answer JSON. Non-reasoning models don't use the
            # extra headroom; no measured slowdown. Reasoning models
            # at 2048 would emit empty content (entire budget consumed
            # by reasoning) and the facet generator's retry loop
            # would fire on every scene.
            result = self.llm.generate_json(
                system_prompt,
                user_prompt,
                temperature=temperature,
                num_predict=4096,
                schema=schema,
                model=model,
            )
        except OllamaJSONParseError as exc:
            logger.warning(
                "Scene facet generator JSON / schema error: %s", exc,
            )
            # 2026-05-23 — store the FULL Pydantic error reason on
            # self so the generate() retry-builder can extract the
            # specific violation and tell the LLM what to fix.
            # Previously the retry used a generic "your JSON did not
            # match" nudge — LLM kept making the same mistake. With
            # the reason threaded through, the retry can say e.g.
            # "your env_X is incompatible with narrative_Y — pick a
            # different narrative" or "your prose contains banned
            # 'tear' token — rewrite without".
            self._last_attempt_failure_reason = str(exc)
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
        lighting_hint: str = "",
        compatible_environments: list[str] | None = None,
        compatible_narratives: list[str] | None = None,
        compatible_art_styles: list[str] | None = None,
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
        # SERIES AESTHETIC LOCK — the style profile's lighting_hint.
        # 2026-05-29 — prior to this, lighting_hint never reached the
        # LLM at all, so the profile's intended lighting register had
        # zero grip: a faded/low-contrast profile (analog_film_intimate)
        # still produced dark dramatic chiaroscuro because the model's
        # "T4 boudoir" prior dominated. Injected high (right after the
        # tier directive) so it outweighs that default. Phrased
        # generically so it works for BOTH soft profiles (keep bright)
        # and hard profiles (boudoir_noir → lean dark) — the lock wins
        # over the model's generic instinct either way.
        if lighting_hint:
            parts.append(
                "\n## SERIES AESTHETIC LOCK — lighting\n"
                f"This series has a LOCKED lighting + finish look: "
                f"**{lighting_hint.strip()}**.\n"
                "Your scene_prose lighting on EVERY scene MUST match this "
                "register. Do NOT substitute your own default lighting — if "
                "the lock describes soft / low-contrast / faded / daylight, "
                "keep the scene gentle and evenly lit; if it describes hard / "
                "low-key / dramatic, lean into that. This lock OUTWEIGHS any "
                "generic 'moody dark boudoir' instinct.\n"
            )
        # COHERENCE INVARIANT block. Placed right after the base
        # preamble + tier directive so it has the next-highest
        # attention weight after the most-critical tier framing.
        # Six sub-clauses (4 universal + 2 tier-conditional):
        #   1. env_setting + atmosphere + narrative MUST imply ONE
        #      consistent time of day + ONE consistent location.
        #   2. DO NOT weave the series-level photographer_ref /
        #      art_movement / color_palette into scene_prose. The
        #      composer canonicalizes those globally per family;
        #      repeating them in prose causes duplication.
        #   3. Pose + camera angle + nsfw_posture / nsfw_act must
        #      form a GEOMETRICALLY VALID composition. Reclining
        #      pose cannot pair with full-frontal-display act;
        #      standing pose cannot pair with overhead angle +
        #      full-frontal display; etc.
        #   4. Subject continuity — scene_prose MUST honor the
        #      subject_description's age/build/hair/body-type
        #      EXACTLY across all 24 scenes. Buyers follow one
        #      subject; inconsistent traits break commercial
        #      value of the set.
        #   5. Tier-conditional anatomical clarity (T1: clothed
        #      restraint, T2: implied only, T3: tasteful nudity
        #      direct, T4: full explicit allowed). Aligns each tier
        #      with its categories.yaml llm_directive.
        #   6. Tier-conditional T3+ — no hair-as-censor poetic
        #      phrasing. "Hair cascades around select areas, hiding
        #      nothing while suggesting everything" type language
        #      makes diffusion models occlude the body with hair.
        coherence_block = (
            "\n"
            "## COHERENCE INVARIANT (read first)\n"
            "\n"
            "1. **Scene coherence.** The environment_setting +\n"
            "   environment_atmosphere + narrative_moment fields you\n"
            "   pick MUST together imply ONE consistent time of day\n"
            "   and ONE consistent location. NEVER combine a `3am`-\n"
            "   anchored setting with an `afternoon-light` atmosphere,\n"
            "   or an indoor lobby with `rain-slicked street` outdoor\n"
            "   atmosphere. When you sense conflict, anchor to\n"
            "   environment_setting and let atmosphere + narrative\n"
            "   align with its time + place.\n"
            "\n"
            "2. **Do NOT weave series-level aesthetics into scene_prose.**\n"
            "   The composer already canonicalizes the series-level\n"
            "   `photographer_ref` / `art_movement` / `color_palette`\n"
            "   into family-shaped prose and threads them into every\n"
            "   prompt globally. If you also reference them in your\n"
            "   `scene_prose` (e.g. 'shot like Helmut Newton, with a\n"
            "   sepia palette'), they land TWICE in the final prompt\n"
            "   and waste tokens. Your `scene_prose` should describe\n"
            "   the woman + her pose + her gaze + the room, NOT the\n"
            "   meta-aesthetic — the composer handles the meta.\n"
            "\n"
            "3. **Pose + camera angle + nsfw_posture / nsfw_act must\n"
            "   form a GEOMETRICALLY VALID composition.** Think about\n"
            "   what the camera physically sees. Examples of invalid\n"
            "   combinations that you MUST avoid:\n"
            "   - reclining pose + low angle + full-frontal display:\n"
            "     low angle of a reclining body is feet-first, not\n"
            "     torso-frontal — the body can't simultaneously be\n"
            "     horizontal AND presenting its front to a camera\n"
            "     below it.\n"
            "   - standing pose + overhead / top-down angle + full-\n"
            "     frontal display: overhead view sees head/shoulders,\n"
            "     not chest/pelvis.\n"
            "   - kneeling pose + reclining nsfw_posture (NSFW_RECLINED_\n"
            "     NUDE): the posture tag must MATCH the scene's pose,\n"
            "     not contradict it.\n"
            "   When the scene's pose says 'reclining' or 'lying',\n"
            "   pick `nsfw_posture=NSFW_RECLINED_NUDE` (not STANDING/\n"
            "   SEATED/KNEELING) and `nsfw_act=NSFW_T4_SOLO_RECLINING`\n"
            "   or `NSFW_T4_AFTERGLOW` or `NSFW_T4_SOLO_GAZE` (not\n"
            "   SOLO_DISPLAY which implies a vertical presenting body).\n"
            "   When the scene's pose says 'standing' or 'kneeling',\n"
            "   pick the matching posture and act. The composition\n"
            "   must be something a real photographer could actually\n"
            "   shoot from the given angle.\n"
            "\n"
            "4. **Subject continuity across the series.** The series-\n"
            "   level `subject_description` carries the woman's age,\n"
            "   build, hair, body type (e.g. 'curvy 32-year-old with\n"
            "   long auburn hair', 'athletic 28-year-old with short\n"
            "   black hair', 'mature 40-year-old with grey-streaked\n"
            "   hair'). Your `scene_prose` MUST reflect these traits\n"
            "   EXACTLY — same age range, same build, same hair color,\n"
            "   same body type — across every scene in the series.\n"
            "   Buyers follow a specific subject across a 24-scene set;\n"
            "   inconsistent body/hair/age scene-to-scene breaks the\n"
            "   commercial value of the set. If subject_description\n"
            "   says 'curvy', write 'her full curves' and never 'her\n"
            "   slender frame'. If it says 'athletic', write 'her toned\n"
            "   shoulders' not 'her soft figure'. Subject continuity\n"
            "   beats per-scene creative variation.\n"
        )
        # Round-22 (2026-05-22) revised — tier-conditional clause 3 with
        # FOUR distinct sub-clauses, one per T1-T4. Pre-revision the T3
        # clause contradicted T3's llm_directive ("describe the nude
        # form directly" vs "MUST NOT use directly anatomical language").
        # Round-4 audit caught this. New wording aligns each tier with
        # its categories.yaml llm_directive:
        #   T1: clothed-implied, no anatomical detail
        #   T2: implied undress / suggestive only, no direct anatomy
        #   T3: tasteful artistic nudity allowed (bare shoulders,
        #       natural skin texture, full nude form) BUT no explicit
        #       T4 phrasing (vulva visible / nipples erect / explicit
        #       acts)
        #   T4: full explicit allowed including vulva / nipples / acts
        if content_level == "T4_explicit":
            coherence_block += (
                "\n"
                "5. **T4_explicit anatomical clarity.** At this tier,\n"
                "   you MAY use direct anatomical language in your\n"
                "   `scene_prose` (e.g. 'her nipples are visible',\n"
                "   'fully nude with visible vulva') when it aligns\n"
                "   with the nsfw_anatomy + nsfw_act tags you picked.\n"
                "   The canonicalizer already adds explicit phrasing\n"
                "   from those tags — your prose should align with\n"
                "   them, not contradict them (e.g. don't describe a\n"
                "   'silk robe wrapping her body' when nsfw_anatomy =\n"
                "   NSFW_FULL_NUDE).\n"
                "\n"
                "6. **No hair-as-censor poetic phrasing.** At T3+ the\n"
                "   subject's nudity is the composition. NEVER describe\n"
                "   hair as 'cascading around select areas', 'covering\n"
                "   intimate parts', 'artfully suggesting while hiding\n"
                "   everything', or any equivalent poetic-veil phrase\n"
                "   in `scene_prose`. Diffusion models read these as\n"
                "   literal instructions to occlude breasts / vulva\n"
                "   with hair, defeating the explicit composition.\n"
                "   You MAY describe hair's color, length, texture,\n"
                "   wind, or placement on shoulders / back / pillow,\n"
                "   but never hair as covering nipples or pubic area.\n"
            )
        elif content_level == "T3_artnude":
            coherence_block += (
                "\n"
                "5. **T3_artnude tasteful nudity.** This is gallery-\n"
                "   print fine-art nude — the subject IS nude. Your\n"
                "   `scene_prose` MUST describe the nude form directly\n"
                "   in tasteful framing (e.g. 'her bare shoulders\n"
                "   catch the window light', 'natural skin texture\n"
                "   across her hip', 'the gentle curve of her back').\n"
                "   ALLOWED: bare / nude / natural skin / artistic\n"
                "   anatomical reference. NOT ALLOWED at this tier:\n"
                "   T4-explicit vocabulary like 'visible vulva',\n"
                "   'erect nipples', or explicit-act phrasing —\n"
                "   those are tier-gated to T4_explicit only.\n"
                "\n"
                "6. **No hair-as-censor poetic phrasing.** The\n"
                "   subject's nude form is the composition. NEVER\n"
                "   describe hair as 'cascading around select areas',\n"
                "   'covering intimate parts', 'artfully suggesting\n"
                "   while hiding everything', or any equivalent\n"
                "   poetic-veil phrase in `scene_prose`. Diffusion\n"
                "   models read these as literal instructions to\n"
                "   occlude the body with hair, defeating the nude\n"
                "   composition. You MAY describe hair's color,\n"
                "   length, texture, placement on shoulders / back /\n"
                "   pillow, but never as covering breasts or hips.\n"
            )
        elif content_level == "T2_implied":
            coherence_block += (
                "\n"
                "5. **T2_implied suggestive restraint.** Your\n"
                "   `scene_prose` should suggest sensuality through\n"
                "   pose + light + composition + implied undress —\n"
                "   not direct anatomy. Allowed: 'silk robe slipping\n"
                "   from a shoulder', 'sheer fabric catching the\n"
                "   light', 'her gaze drifts toward the window'.\n"
                "   NOT ALLOWED: 'bare breasts', 'nude form',\n"
                "   'visible' anything explicit. The canonicalizer\n"
                "   keeps T3+ NSFW vocabulary gated; your prose\n"
                "   follows the same gate.\n"
            )
        else:
            # T1_suggestive (or unknown — defensive default).
            coherence_block += (
                "\n"
                "5. **T1_suggestive clothed restraint.** Your\n"
                "   `scene_prose` describes a fully-clothed subject\n"
                "   in tasteful poses. No nudity, no implied undress,\n"
                "   no anatomical detail beyond what's visible in\n"
                "   ordinary clothing. The canonicalizer keeps T2+\n"
                "   NSFW vocabulary gated.\n"
            )
        parts.append(coherence_block)
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
            compatible_narratives=compatible_narratives,
            compatible_art_styles=compatible_art_styles,
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
        diversity_nudge: str = "",
        compatible_environments: list[str] | None = None,
        compatible_narratives: list[str] | None = None,
        subject_description: str = "",
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
        # Round-13 (2026-05-21) — rewrite the in-parens example lists
        # for environment_setting + narrative_moment so they advertise
        # only category-coherent tags. Closes the post-fix audit gap
        # where Qwen3 was still picking NARR_STEPPING_FROM_BATH from
        # the schema body's hard-coded example list even though the
        # vocab block had narrowed the namespace correctly.
        active_body = _narrow_schema_body_examples(
            active_body,
            compatible_environments=compatible_environments,
            compatible_narratives=compatible_narratives,
        )
        # Round-22 (2026-05-22) — subject_description from series_plan
        # is now threaded into the user prompt so the facet LLM can pick
        # nsfw_anatomy / nsfw_act coherent with the subject. Empty
        # fallback "(not provided)" keeps back-compat for callers that
        # don't supply it.
        return _USER_PROMPT_TEMPLATE.format(
            content_level=content_level,
            subject_description=subject_description or "(not provided)",
            scene_core_json=json.dumps(core, indent=2),
            family_id=family.id,
            prompt_style=family.prompt_style,
            schema_body=active_body,
            tier_required_list=tier_required_list,
            diversity_nudge=diversity_nudge,
        )
