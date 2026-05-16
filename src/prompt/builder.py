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
from src.prompt.vocabulary import canonicalize_facet


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

# Always-on hard-block for age-ambiguity vocabulary — prepended to every
# family-level negative prompt. Belt-and-braces: the LLM's planner hint
# already bans these, the positive scan strips them from the body, and
# this blocks the encoder from generating them even if one slips through.
HARD_BLOCK_NEGATIVE = (
    "child, kid, young, minor, teen, schoolgirl, loli, shota, "
    "underage, baby, toddler, preteen, youthful face"
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

        segments = [HARD_BLOCK_NEGATIVE]
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

        # Phase 4a — canonicalize abstract enum-tag fields on the scene
        # facet into family-shaped phrases. Canonicalizer drops:
        #   * unknown concept tags (LLM drift),
        #   * Pony-omitted namespaces (camera/lens/film_stock/art_style),
        #   * NSFW concepts gated above the active content_level.
        vocab_phrases = canonicalize_facet(
            scene, family.id, content_level=content_level,
        )

        segments: list[str] = [base_prompt]
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

        style_keywords = self._field(style_profile, "base_style_keywords")
        if style_keywords:
            segments.append(style_keywords)

        # Combine caller-supplied extra_keywords with vocab phrases for
        # the prose-composer path (which reads extra_keywords as its
        # own kwarg). Vocab phrases lead — they're more specific.
        merged_extras: list[str] = list(vocab_phrases)
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
    ) -> list[dict[str, Any]]:
        """Vectorized :meth:`build_one`."""
        return [
            self.build_one(
                character, scene, style_profile, extra_keywords,
                family=family, trigger_words=trigger_words,
            )
            for scene in scenes
        ]

    # ── mode-specific helpers ──────────────────────────────────────
    def build_character_prompt(
        self,
        character: Mapping[str, Any],
        scene: Mapping[str, Any],
        style_profile: Mapping[str, Any] | Any,
        *,
        family: FamilyConfig,
        trigger_words: Iterable[str] | None = None,
        negative_prompt_override: str | None = None,
    ) -> dict[str, Any]:
        return self.build_one(
            character, scene, style_profile,
            family=family,
            trigger_words=trigger_words,
            negative_prompt_override=negative_prompt_override,
        )

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
            )
        if style == "flux_natural":
            return _compose_natural(
                segments, trigger_words=trig, avoid=avoid,
                scene_prose=self._field(scene, "scene_prose"),
                base_prompt=base_prompt,
                style_keywords=style_keywords,
                extra_keywords=list(extra_keywords),
                realism_tail_style=family.realism_tail_style,
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
        tail_fragments = [
            frag for frag in _CHROMA_REALISM_TAIL
            if frag.lower() not in avoid_set
            and frag.lower() not in text.lower()
        ]
        if tail_fragments:
            tail = ". ".join(tail_fragments) + "."
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
