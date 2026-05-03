"""Pydantic schemas — typed contracts for LLM JSON outputs.

The series planner, scene generator, and scene-facet generator each
ask the LLM to return JSON with a known shape. This module replaces
the old hand-rolled dict validation with Pydantic v2 models so:

  * Validation errors carry exact field paths
    ("scene[3].pose: missing") rather than a fuzzy
    "missing fields {...}" set.
  * Phase A optional fields (composition_intent, framing_hint,
    audience_target) are typed once and reused everywhere.
  * Family-shaped fields live in dedicated **SceneFacet* schemas**
    (one per ``family.prompt_style``) — not co-mingled with the
    model-agnostic ``Scene``. The facet generator picks the right
    schema via ``SCENE_FACET_SCHEMA_BY_STYLE``.

All models use ``extra="allow"`` so the LLM can return additional
fields without breaking the schema, and ``str_strip_whitespace=True``
so leading/trailing whitespace is normalised before length checks.

Returning typed instances vs. plain dicts: the agents (SeriesPlanner,
SceneGenerator, SceneFacetGenerator) keep returning ``dict`` /
``list[dict]`` for backward compatibility with downstream consumers
(PromptBuilder, RatioSelector, DB writers). Pydantic is used only for
the validation step.
"""

from __future__ import annotations

import logging
import re

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    field_validator,
    model_validator,
)


logger = logging.getLogger(__name__)


# ── Series plan ─────────────────────────────────────────────────────


class SeriesPlan(BaseModel):
    """LLM-generated series plan (output of :class:`SeriesPlanner`).

    Fields match ``REQUIRED_FIELDS`` in :mod:`src.agents.series_planner`:
    ``theme``, ``mood``, ``environment``, ``variation_axes``. Additional
    fields the LLM might emit (e.g. ``style_notes``) are accepted and
    forwarded via ``extra="allow"``.
    """

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    theme: str = Field(min_length=1)
    mood: str = Field(min_length=1)
    environment: str = Field(min_length=1)
    variation_axes: list[str] = Field(min_length=1)

    @field_validator("variation_axes")
    @classmethod
    def _filter_blank_axes(cls, v: list[str]) -> list[str]:
        cleaned = [s.strip() for s in v if isinstance(s, str) and s.strip()]
        if not cleaned:
            raise ValueError(
                "variation_axes must contain at least one non-empty string"
            )
        return cleaned


# ── Scene (model-agnostic core) ─────────────────────────────────────


class Scene(BaseModel):
    """One LLM-generated scene description — model-agnostic core.

    Required fields are the universal scene anchors (pose, camera,
    lighting, environment, mood). Optional fields cover Phase A
    aspect-ratio intent (composition_intent, framing_hint,
    audience_target), which feed the multi-axis ratio scorer.

    **Family-shaped fields no longer live here.** Per-family
    composer hints (booru_tags, scene_prose, camera_spec, clothing,
    source_tag) are produced by :class:`SceneFacetGenerator` and stored
    in the ``scene_facets`` table keyed by ``(scene_id, family)``.
    Sibling models in the same family share a single facet row.
    """

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    # ----- Required universal fields ---------------------------------------
    variation_axis: str = Field(min_length=1)
    pose: str = Field(min_length=1)
    camera: str = Field(min_length=1)
    camera_angle: str = Field(min_length=1)
    lighting: str = Field(min_length=1)
    environment_detail: str = Field(min_length=1)
    mood_note: str = Field(min_length=1)

    # ----- Optional: expression (sometimes asked for, sometimes not) ------
    expression: str | None = None

    # ----- Phase A aspect-ratio intent (all optional) ----------------------
    # Validated only loosely — RatioSelector silently ignores values
    # outside the known enum, so we don't lock the schema down here.
    composition_intent: str | None = None
    framing_hint: str | None = None
    audience_target: str | None = None


class SceneList(RootModel[list[Scene]]):
    """Validated list of scenes — wraps :class:`SceneGenerator` output.

    Pydantic's RootModel makes ``model_validate_json`` work on a bare
    JSON array without needing an outer wrapper object. The ``.root``
    attribute is the underlying list; iteration / indexing / length
    proxy through to it.
    """

    def __iter__(self):
        return iter(self.root)

    def __len__(self) -> int:
        return len(self.root)

    def __getitem__(self, idx: int) -> Scene:
        return self.root[idx]


# ── Scene facets (per-family LLM expansion) ─────────────────────────


# ── Phase 4a: structured enum-tag fields shared across SceneFacet schemas ──
#
# Every prose / SDXL / Illustrious facet may emit these abstract
# concept tags from ``config/prompt_vocabulary.yaml``. The composer
# canonicalises the tags into family-specific phrasing at compose time
# via :func:`src.prompt.vocabulary.canonicalize_facet`. All tags are
# Optional[str]: when the LLM sets them, they enrich the prompt; when
# omitted, the composer simply skips them. Pony deliberately omits
# camera / lens / film_stock / art_style — booru tagging covers those
# implicitly via ``source_photograph + booru_tags``.

_REALISM_ENUM_FIELDS = {
    "realism_camera": (
        "Optional camera-body concept tag from prompt_vocabulary.yaml's "
        "realism.camera namespace. Examples: CAMERA_SONY_A7RV, "
        "CAMERA_FUJI_XT5, CAMERA_HASSELBLAD_X2D, CAMERA_LEICA_M11. "
        "Composer translates the tag into family-shaped phrasing."
    ),
    "realism_lens": (
        "Optional lens concept tag from realism.lens namespace. "
        "Examples: LENS_85MM_F14, LENS_50MM_F18, LENS_135MM_F2."
    ),
    "realism_film_stock": (
        "Optional film-stock concept tag from realism.film_stock "
        "namespace. Examples: FILM_PORTRA_400, FILM_FUJI_PROVIA, "
        "FILM_CINESTILL_800T, FILM_TRIX_400, FILM_DIGITAL_RAW."
    ),
    "art_style_reference": (
        "Optional art-style concept tag from realism.art_style "
        "namespace. Examples: ART_FINE_NUDE, ART_BOUDOIR_NOIR, "
        "ART_OLD_HOLLYWOOD, ART_EDITORIAL_FASHION, ART_CLASSICAL."
    ),
}

_LIGHTING_MOOD_ENUM_FIELDS = {
    "lighting_directive": (
        "Optional lighting concept tag from realism.lighting namespace. "
        "Examples: LIGHT_REMBRANDT, LIGHT_GOLDEN_HOUR, LIGHT_LOW_KEY_NOIR, "
        "LIGHT_SOFT_FILL, LIGHT_RIM_BACK, LIGHT_OVERCAST, LIGHT_NEON_NIGHT, "
        "LIGHT_WINDOW_SIDE, LIGHT_BUTTERFLY. The composer translates the "
        "tag into family-shaped phrasing — for Flux2 specifically this is "
        "more critical than for keyword families since lighting drives "
        "much of Klein's output character."
    ),
    "mood_aesthetic": (
        "Optional mood concept tag from realism.mood namespace. "
        "Examples: MOOD_INTIMATE, MOOD_CONFIDENT, MOOD_PLAYFUL, "
        "MOOD_PENSIVE, MOOD_DEFIANT."
    ),
}

_NSFW_ENUM_FIELDS = {
    "nsfw_anatomy": (
        "Optional NSFW anatomy concept tag from nsfw.anatomy namespace. "
        "Tier-gated: T3_artnude+ for natural-anatomy concepts. The "
        "canonicalizer drops the tag below tier_min. Examples: "
        "NSFW_BREAST_NATURAL, NSFW_HIPS_THIGHS, NSFW_GLUTES."
    ),
    "nsfw_posture": (
        "Optional NSFW posture concept tag from nsfw.posture namespace. "
        "Tier-gated. Examples: NSFW_RECLINED_NUDE, NSFW_KNEELING_NUDE "
        "(both T3_artnude); NSFW_INTIMATE (T4_explicit only)."
    ),
    "nsfw_act": (
        "Optional NSFW explicit-act concept tag from nsfw.act namespace. "
        "Tier-gated to T4_explicit ONLY — canonicalizer drops below T4. "
        "Examples: NSFW_T4_EMBRACE_NUDE, NSFW_T4_KISS_PASSIONATE, "
        "NSFW_T4_SOLO_TOUCH, NSFW_T4_PARTNERED_INTIMATE, "
        "NSFW_T4_AFTERGLOW. Phase 4-bis."
    ),
}


class SceneFacetSDXL(BaseModel):
    """Per-scene SDXL composer inputs (camera/lens spec + garment).

    Used by ``_compose_keywords``. Two SDXL siblings (e.g.
    ``lustify_v7`` + ``juggernaut_ragnarok``) share this facet row.
    """

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    camera_spec: str = Field(
        min_length=1,
        description="Lens + aperture spec, e.g. '85mm f/1.8, shallow DoF'.",
    )
    clothing: str = Field(
        min_length=1,
        description="Garment + texture detail (silk slip, lace bodice, "
        "velvet robe, linen sheet).",
    )

    # Phase 4a — abstract concept tags translated by the canonicalizer.
    realism_camera: str | None = Field(default=None, description=_REALISM_ENUM_FIELDS["realism_camera"])
    realism_lens: str | None = Field(default=None, description=_REALISM_ENUM_FIELDS["realism_lens"])
    realism_film_stock: str | None = Field(default=None, description=_REALISM_ENUM_FIELDS["realism_film_stock"])
    art_style_reference: str | None = Field(default=None, description=_REALISM_ENUM_FIELDS["art_style_reference"])
    lighting_directive: str | None = Field(default=None, description=_LIGHTING_MOOD_ENUM_FIELDS["lighting_directive"])
    mood_aesthetic: str | None = Field(default=None, description=_LIGHTING_MOOD_ENUM_FIELDS["mood_aesthetic"])
    nsfw_anatomy: str | None = Field(default=None, description=_NSFW_ENUM_FIELDS["nsfw_anatomy"])
    nsfw_posture: str | None = Field(default=None, description=_NSFW_ENUM_FIELDS["nsfw_posture"])
    nsfw_act: str | None = Field(default=None, description=_NSFW_ENUM_FIELDS["nsfw_act"])

    @model_validator(mode="after")
    def _reject_avoid_words(self) -> "SceneFacetSDXL":
        """Reject facets containing tokens the SDXL family explicitly
        avoids — those are composer-side concerns and degrade SDXL
        realism finetune output."""
        avoid = ("masterpiece", "best quality", "8k", "4k", "absurdres")
        for field_name in ("camera_spec", "clothing"):
            value = (getattr(self, field_name, "") or "").lower()
            for tok in avoid:
                if tok in value:
                    raise ValueError(
                        f"SceneFacetSDXL.{field_name} contains avoid-word "
                        f"{tok!r} — drop quality boilerplate from facet "
                        f"output; the composer handles quality framing."
                    )
        return self


class SceneFacetPony(BaseModel):
    """Per-scene Pony composer inputs (booru tags + source_tag).

    Used by ``_compose_pony_danbooru``. Pony deliberately omits
    camera / lens / film_stock / art_style enum fields — booru tagging
    carries those implicitly via ``source_photograph + booru_tags``.
    """

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    booru_tags: str = Field(
        min_length=1,
        description="Comma-separated underscored booru tags capturing "
        "pose, setting, clothing — primary signal for the Pony composer.",
    )
    source_tag: str | None = Field(
        default=None,
        description="One of source_photograph / source_anime / "
        "source_cartoon. Use source_photograph for realism. Optional — "
        "the composer falls back to family default.",
    )

    # Phase 4a — only lighting + mood + NSFW translate cleanly to booru.
    lighting_directive: str | None = Field(default=None, description=_LIGHTING_MOOD_ENUM_FIELDS["lighting_directive"])
    mood_aesthetic: str | None = Field(default=None, description=_LIGHTING_MOOD_ENUM_FIELDS["mood_aesthetic"])
    nsfw_anatomy: str | None = Field(default=None, description=_NSFW_ENUM_FIELDS["nsfw_anatomy"])
    nsfw_posture: str | None = Field(default=None, description=_NSFW_ENUM_FIELDS["nsfw_posture"])
    nsfw_act: str | None = Field(default=None, description=_NSFW_ENUM_FIELDS["nsfw_act"])

    @model_validator(mode="after")
    def _check_pony_invariants(self) -> "SceneFacetPony":
        """Reject score_* / source_pony tokens (composer prepends the
        score chain + source tag); warn when source_tag is missing
        (absorbed from ``_warn_if_pony_missing_source`` in builder.py)."""
        body = (self.booru_tags or "").lower()
        # Composer prepends `score_9, score_8_up, …, score_4_up, BREAK`.
        if "score_9" in body or "score_8_up" in body:
            raise ValueError(
                "SceneFacetPony.booru_tags contains score_* tokens — "
                "the composer prepends the 6-tier score prefix "
                "automatically. Drop score_* from facet output."
            )
        if "source_pony" in body:
            raise ValueError(
                "SceneFacetPony.booru_tags contains source_pony — for "
                "realism Pony finetunes use source_photograph (or set "
                "the source_tag field explicitly)."
            )
        # Source-tag warning (moved from builder.py:626 in Phase 4b —
        # the validator is the natural home for "did the LLM provide
        # what we asked for?" checks).
        if not (self.source_tag and self.source_tag.strip()):
            if "source_" not in body:
                logger.warning(
                    "SceneFacetPony missing source_* tag — for realism "
                    "Pony finetunes prefer source_photograph; the "
                    "composer falls back to source_photograph. Body: %r",
                    self.booru_tags[:80] + (
                        "…" if len(self.booru_tags) > 80 else ""
                    ),
                )
        return self


class SceneFacetIllustrious(BaseModel):
    """Per-scene Illustrious composer inputs (tags first + short prose).

    Used by ``_compose_illustrious_tags`` (hybrid: tags then prose).
    """

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    booru_tags: str = Field(
        min_length=1,
        description="Comma-separated underscored booru tags.",
    )
    scene_prose: str = Field(
        min_length=1,
        description="One short sentence of natural-language prose "
        "describing the whole composition — used alongside the tags.",
    )

    # Phase 4a — full enum-field set (Illustrious accepts hybrid).
    realism_camera: str | None = Field(default=None, description=_REALISM_ENUM_FIELDS["realism_camera"])
    realism_lens: str | None = Field(default=None, description=_REALISM_ENUM_FIELDS["realism_lens"])
    realism_film_stock: str | None = Field(default=None, description=_REALISM_ENUM_FIELDS["realism_film_stock"])
    art_style_reference: str | None = Field(default=None, description=_REALISM_ENUM_FIELDS["art_style_reference"])
    lighting_directive: str | None = Field(default=None, description=_LIGHTING_MOOD_ENUM_FIELDS["lighting_directive"])
    mood_aesthetic: str | None = Field(default=None, description=_LIGHTING_MOOD_ENUM_FIELDS["mood_aesthetic"])
    nsfw_anatomy: str | None = Field(default=None, description=_NSFW_ENUM_FIELDS["nsfw_anatomy"])
    nsfw_posture: str | None = Field(default=None, description=_NSFW_ENUM_FIELDS["nsfw_posture"])
    nsfw_act: str | None = Field(default=None, description=_NSFW_ENUM_FIELDS["nsfw_act"])

    @model_validator(mode="after")
    def _reject_appended_quality_suffix(self) -> "SceneFacetIllustrious":
        """Reject facets containing ``masterpiece`` / ``best quality`` /
        ``amazing quality`` / ``very aesthetic`` — the Illustrious
        composer appends these as the family's quality_suffix and
        duplicates degrade output."""
        suffix = (
            "masterpiece", "best quality", "amazing quality",
            "very aesthetic",
        )
        for field_name in ("booru_tags", "scene_prose"):
            value = (getattr(self, field_name, "") or "").lower()
            for tok in suffix:
                if tok in value:
                    raise ValueError(
                        f"SceneFacetIllustrious.{field_name} contains "
                        f"quality-suffix token {tok!r} — the composer "
                        f"appends these automatically. Drop from facet "
                        f"output."
                    )
        return self


class SceneFacetFluxNatural(BaseModel):
    """Per-scene prose-composer inputs (flux + chroma share this).

    Used by ``_compose_natural``. The ``flux_natural`` prompt_style
    covers both the flux and chroma families.
    """

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    scene_prose: str = Field(
        min_length=1,
        description="1–3 complete sentences of natural-language prose. "
        "Weave pose, lighting, lens character, environment, and mood "
        "into flowing prose. No comma-tag lists, no weighting syntax.",
    )

    # Phase 4a — full enum-field set.
    realism_camera: str | None = Field(default=None, description=_REALISM_ENUM_FIELDS["realism_camera"])
    realism_lens: str | None = Field(default=None, description=_REALISM_ENUM_FIELDS["realism_lens"])
    realism_film_stock: str | None = Field(default=None, description=_REALISM_ENUM_FIELDS["realism_film_stock"])
    art_style_reference: str | None = Field(default=None, description=_REALISM_ENUM_FIELDS["art_style_reference"])
    lighting_directive: str | None = Field(default=None, description=_LIGHTING_MOOD_ENUM_FIELDS["lighting_directive"])
    mood_aesthetic: str | None = Field(default=None, description=_LIGHTING_MOOD_ENUM_FIELDS["mood_aesthetic"])
    nsfw_anatomy: str | None = Field(default=None, description=_NSFW_ENUM_FIELDS["nsfw_anatomy"])
    nsfw_posture: str | None = Field(default=None, description=_NSFW_ENUM_FIELDS["nsfw_posture"])
    nsfw_act: str | None = Field(default=None, description=_NSFW_ENUM_FIELDS["nsfw_act"])

    @model_validator(mode="after")
    def _check_prose_shape(self) -> "SceneFacetFluxNatural":
        """Prose families want flowing sentences, not tag soup. Reject:
        * ``(word:1.3)`` weighting syntax (Flux/Chroma ignore it),
        * tag-soup heuristic — more than 4 commas per sentence,
        * underscored multi-word tags (``long_hair`` etc. — booru style)."""
        prose = self.scene_prose or ""
        if re.search(r"\([^():]+:\d+(?:\.\d+)?\)", prose):
            raise ValueError(
                "SceneFacetFluxNatural.scene_prose contains weighting "
                "syntax `(word:1.3)` — Flux / Chroma encoders ignore "
                "this and it pollutes the prose."
            )
        # Heuristic: very tag-soupy prose has many commas per sentence.
        sentences = [s for s in re.split(r"[.!?]", prose) if s.strip()]
        for sentence in sentences:
            if sentence.count(",") > 4:
                raise ValueError(
                    "SceneFacetFluxNatural.scene_prose has a sentence "
                    "with >4 commas — looks like a tag list. Use flowing "
                    f"prose instead. Sentence: {sentence.strip()!r}"
                )
        # Underscored multi-word tags are a booru convention; Flux /
        # Chroma encoders see them as a single weird token.
        if re.search(r"\b\w+_\w+\b", prose):
            raise ValueError(
                "SceneFacetFluxNatural.scene_prose contains "
                "underscored multi-word tokens (booru style) — Flux / "
                "Chroma encoders prefer space-separated natural words."
            )
        return self


class SceneFacetFlux2(BaseModel):
    """Per-scene FLUX.2 Klein composer inputs (BFL 5-anchor prose).

    Used by ``_compose_flux2_prose``. As of Phase 4a, ``subject_focus``
    is kept as a non-persistent QA signal; ``lighting_directive`` is
    promoted to a structured enum-tag field that the canonicalizer
    translates (still kept in the schema as Optional[str]).
    """

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    scene_prose: str = Field(
        min_length=1,
        description="Single paragraph, 30–80 words. Five anchors in "
        "STRICT order: subject → setting → details → lighting → "
        "atmosphere. No tags, no weighting, no BREAK. The most "
        "distinctive subject traits and the lighting directive go "
        "near the front; word order weights heavily for Klein.",
    )
    subject_focus: str | None = Field(
        default=None,
        description="One-line distillation of the subject clause, "
        "used as an ordering QA signal. Not persisted.",
    )

    # Phase 4a — full enum-field set; lighting_directive is now a
    # structured concept tag (LIGHT_*) translated by the canonicalizer
    # — promoted from QA-only to persistent and family-shared.
    realism_camera: str | None = Field(default=None, description=_REALISM_ENUM_FIELDS["realism_camera"])
    realism_lens: str | None = Field(default=None, description=_REALISM_ENUM_FIELDS["realism_lens"])
    realism_film_stock: str | None = Field(default=None, description=_REALISM_ENUM_FIELDS["realism_film_stock"])
    art_style_reference: str | None = Field(default=None, description=_REALISM_ENUM_FIELDS["art_style_reference"])
    lighting_directive: str | None = Field(default=None, description=_LIGHTING_MOOD_ENUM_FIELDS["lighting_directive"])
    mood_aesthetic: str | None = Field(default=None, description=_LIGHTING_MOOD_ENUM_FIELDS["mood_aesthetic"])
    nsfw_anatomy: str | None = Field(default=None, description=_NSFW_ENUM_FIELDS["nsfw_anatomy"])
    nsfw_posture: str | None = Field(default=None, description=_NSFW_ENUM_FIELDS["nsfw_posture"])
    nsfw_act: str | None = Field(default=None, description=_NSFW_ENUM_FIELDS["nsfw_act"])

    @model_validator(mode="after")
    def _check_word_count_band(self) -> "SceneFacetFlux2":
        """BFL Klein 9B targets 30–80 words; with 5-word slack the
        accept band is 25–95. Outside that, fail hard so a retry is
        triggered. Inside 25–95 but outside 30–80, log a warning so
        operators can monitor drift without blocking renders."""
        prose = self.scene_prose or ""
        words = len(prose.split())
        if not (25 <= words <= 95):
            raise ValueError(
                f"SceneFacetFlux2.scene_prose has {words} words; BFL "
                f"Klein 9B requires 25–95 (target 30–80). Tighten the "
                f"prose."
            )
        if not (30 <= words <= 80):
            logger.warning(
                "SceneFacetFlux2.scene_prose word count %d is outside "
                "the 30–80 BFL target band (still inside 25–95 slack). "
                "Consider tightening.", words,
            )
        return self


# Dispatcher: ``family.prompt_style`` → facet schema.
# The 5 prompt_style values exhaust the family enum; "flux_natural"
# is shared by flux + chroma (both use _compose_natural).
SCENE_FACET_SCHEMA_BY_STYLE: dict[str, type[BaseModel]] = {
    "sdxl_keywords":    SceneFacetSDXL,
    "pony_danbooru":    SceneFacetPony,
    "illustrious_tags": SceneFacetIllustrious,
    "flux_natural":     SceneFacetFluxNatural,
    "flux2_prose":      SceneFacetFlux2,
}
