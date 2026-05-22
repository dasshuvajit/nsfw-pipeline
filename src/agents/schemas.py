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
from typing import Annotated, Any

from annotated_types import MinLen
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

    Phase 3 (vocab v6, 2026-05-19) — added three series-level
    aesthetic-anchor fields the SeriesPlanner picks ONCE per series
    and the composer threads into every scene via
    ``canonicalize_series_aesthetic``: ``color_palette``,
    ``photographer_ref``, ``art_movement``. **They MUST be on this
    schema** (not just via extra="allow") because Ollama 0.5+
    constrains the LLM output against the Pydantic JSON schema —
    fields absent from the schema are not grammar-required and a
    constrained LLM (Cydonia / Qwen) takes the path of least
    resistance and skips them. Verifier B2.

    Round-12 (2026-05-20) — ``color_palette`` was promoted to
    required ``str`` (min_length 3, ``PALETTE_*`` pattern) after the
    Cydonia + Qwen3 A/B run showed both LLMs nulling all three
    anchors despite explicit prompt instructions. Optional[str]
    let Ollama's constrained decoder sample ``null`` legally; the
    required-str + pattern combo closes that grammar loophole.
    ``photographer_ref`` + ``art_movement`` stay Optional because
    Pony legitimately omits them (booru tagging has no equivalent
    vocabulary) but the loose-tag-shape repair in
    ``src.modes._llm_helpers.repair_aesthetic_anchor_keys`` handles
    trailing-space / trailing-colon / comma-joined LLM quirks on
    all three.
    """

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    theme: str = Field(min_length=1)
    mood: str = Field(min_length=1)
    environment: str = Field(min_length=1)
    variation_axes: list[str] = Field(min_length=1)

    # Round-17 (2026-05-21) — subject_description promoted to required.
    # Pre-fix the field was accepted only via extra="allow", so LLMs
    # that skipped it (Qwen3.5-9b MLX in particular) silently produced
    # plans with empty subject_description, which then collapsed
    # base_prompt to "" and broke every scene's prompt build. Making
    # it Pydantic-required forces Ollama's constrained decoder + LM
    # Studio's response_format:json_schema to emit a non-empty value.
    # The round-16 fallback in ``engine._synthetic_character_for_scene``
    # stays in place as defence-in-depth for legacy plans + edge cases
    # the schema can't catch (purely whitespace, sanitization
    # producing empty result, etc.).
    #
    # Length bounds: min_length 8 rejects single-word stubs; the
    # SeriesPlanner user-template asks for "≤ 18 words", which is
    # roughly 100-200 chars of prose — max_length 300 catches runaway
    # output without rejecting genuinely-rich descriptions.
    subject_description: str = Field(
        ...,
        min_length=8,
        max_length=300,
        description=(
            "≤ 18 WORDS describing ONLY the subject (appearance, "
            "attire / state of undress, age signal). At T3/T4 name "
            "nudity per the tier directive. NEVER use cross-scene "
            "variety language (varying, various, across, throughout, "
            "multiple, different angles, compositions, poses) — "
            "those phrases get injected verbatim into every scene's "
            "prompt and the diffusion model reads them as a polyptych "
            "instruction. Required — must be a substantive single-"
            "scene subject anchor."
        ),
    )

    # Phase 3 — series-level aesthetic anchors. ``color_palette`` is
    # required-str (no Optional) so Ollama's constrained decoder
    # cannot emit null for it. ``photographer_ref`` + ``art_movement``
    # stay Optional[str] — Pony legitimately omits both because
    # booru tagging carries no photographer / art-movement signal.
    color_palette: str = Field(
        ...,
        min_length=3,
        pattern=r"^PALETTE_[A-Z0-9_]+$",
        description=(
            "Series-level color-palette concept tag from "
            "aesthetic.color_palette namespace (e.g. "
            "PALETTE_BAROQUE_CARAVAGGIO, PALETTE_TEAL_ORANGE_BLOCKBUSTER, "
            "PALETTE_MONOCHROME_HIGH_CONTRAST, PALETTE_WES_ANDERSON_PASTEL, "
            "PALETTE_DEAKINS_AMBER_TUNGSTEN, PALETTE_TUSCAN_EARTH). "
            "REQUIRED — pick exactly one PALETTE_* tag from the "
            "narrowed menu in the planner user prompt. NEVER null, "
            "NEVER empty. Pinned ONCE per series; the canonicalizer "
            "threads the phrasing into every scene's composed prompt."
        ),
    )
    photographer_ref: str | None = Field(
        default=None,
        description=(
            "Series-level photographer-signature concept tag from "
            "aesthetic.photographer_ref namespace (e.g. "
            "PHOTOG_HELMUT_NEWTON, PHOTOG_PETTER_HEGRE, "
            "PHOTOG_PETER_LINDBERGH, PHOTOG_GREGORY_CREWDSON, "
            "PHOTOG_ROBERT_MAPPLETHORPE, PHOTOG_SLIM_AARONS). "
            "Series's photographer-signature aesthetic, held constant "
            "across every scene. Pony omits this namespace."
        ),
    )
    art_movement: str | None = Field(
        default=None,
        description=(
            "Optional series-level art-movement concept tag from "
            "aesthetic.art_movement namespace (e.g. "
            "ART_MOVE_FILM_NOIR_1940S, ART_MOVE_BAROQUE_CARAVAGGIO, "
            "ART_MOVE_DUTCH_GOLDEN_VERMEER, ART_MOVE_WES_ANDERSON, "
            "ART_MOVE_HOPPER, ART_MOVE_ART_NOUVEAU_KLIMT). Pony omits."
        ),
    )

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


class SceneList(RootModel[Annotated[list[Scene], MinLen(1)]]):
    """Validated list of scenes — wraps :class:`SceneGenerator` output.

    Pydantic's RootModel makes ``model_validate_json`` work on a bare
    JSON array without needing an outer wrapper object. The ``.root``
    attribute is the underlying list; iteration / indexing / length
    proxy through to it.

    The ``MinLen(1)`` annotation forbids an empty list at both the
    grammar level (Ollama's ``format: <schema>`` emits ``minItems: 1``)
    and the Pydantic post-validation level. Without it, a loose-aligned
    LLM (Venice, Magnum) takes the path of least resistance and emits
    ``[]`` — grammatically valid but semantically a no-op. The downstream
    ``validate_scene_list`` filter drops empty lists too, so an LLM that
    slips past the grammar still hits a wall — defence in depth.
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
    # Q10 (vocab v4) — composition gap-fill.
    "realism_angle": (
        "Optional camera-angle concept tag from realism.angle namespace. "
        "Examples: ANGLE_LOW, ANGLE_EYE_LEVEL, ANGLE_HIGH, ANGLE_DUTCH, "
        "ANGLE_OVER_SHOULDER. Pony omits — booru tagging carries angle "
        "implicitly via tags like `low_angle`, `from_below`, `from_above`."
    ),
    "realism_framing": (
        "Optional shot-size / framing concept tag from realism.framing "
        "namespace. Examples: FRAMING_EXTREME_CLOSE_UP, FRAMING_CLOSE_UP, "
        "FRAMING_MEDIUM_CLOSE, FRAMING_MEDIUM, FRAMING_MEDIUM_WIDE, "
        "FRAMING_WIDE, FRAMING_EXTREME_WIDE. Pony omits — booru tagging "
        "carries framing via `close-up`, `cowboy_shot`, `full_body`, etc."
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

# Phase 1 (vocab_version 6) — environment vocabulary.
# Per-scene location + atmospheric pick from the new `environment.*`
# top-level namespace. The canonicalizer translates the abstract tag
# into family-shaped phrasing at compose time, so the LLM picks one
# concept name (e.g. ENV_VICTORIAN_CONSERVATORY) and the composer
# emits the right prose for chroma vs. the right booru tag list for
# pony. Categories whitelist compatible_environments per theme so
# the LLM picks from a coherent pool, not the full 40-tag library.
# All families participate (no Pony-omission like the
# camera/lens/film_stock/art_style realism namespaces — environments
# DO have natural booru representations).
# Phase 2 (vocab_version 6) — narrative_moment vocabulary.
# THE single highest-leverage axis per market research. "She reads
# a letter at dawn" forces window light + chair + envelope + stillness
# — 5 axes resolved from one tag. Tier-AGNOSTIC: the moment itself
# isn't tier-restricted; tier-appropriate anatomy/explicitness comes
# from the nsfw_* tags. Required at EVERY tier (T1+) — every facet
# row gets one narrative anchor so no scene is just "she poses".
_NARRATIVE_ENUM_FIELDS = {
    "narrative_moment": (
        "Required narrative-anchor concept tag from narrative.moment "
        "namespace. Describes what's happening in the scene — what "
        "moment is being captured — rather than just naming a pose. "
        "Examples: NARR_READING_LETTER_AT_DAWN, "
        "NARR_JUST_WOKEN_TANGLED_SHEETS, NARR_DRESSING_FOR_EVENING, "
        "NARR_STEPPING_FROM_BATH, NARR_LIGHTING_CIGARETTE_BALCONY, "
        "NARR_POURING_WINE_ALONE, "
        "NARR_LACING_CORSET_BACK, NARR_AFTER_THE_PARTY, "
        "NARR_LISTENING_TO_RECORD, NARR_DRYING_HAIR_TOWEL_WINDOW, "
        "NARR_BAREFOOT_AT_3AM_KITCHEN, NARR_SUNBATHING_TERRACE, "
        "NARR_RAIN_AT_WINDOW. Pick the moment that fits the chosen "
        "environment_setting + pose + lighting — the moment-tag is "
        "the visual story-anchor every fine-art editorial frame needs."
    ),
}

_ENVIRONMENT_ENUM_FIELDS = {
    "environment_setting": (
        "Optional location concept tag from environment.setting "
        "namespace. Picks the scene's specific setting. Examples: "
        "ENV_VICTORIAN_CONSERVATORY, ENV_TUSCAN_VILLA_RENAISSANCE, "
        "ENV_ART_DECO_HOTEL_SUITE, ENV_BRUTALIST_CONCRETE_LOFT, "
        "ENV_MORNING_BEDROOM, ENV_CLAWFOOT_BATHROOM, "
        "ENV_MEDITERRANEAN_COURTYARD, ENV_FOG_PINE_FOREST, "
        "ENV_DESERT_DUNE, ENV_ROOFTOP_CITY_NIGHT, "
        "ENV_TOKYO_LOVE_HOTEL, ENV_INDOOR_POOL_NIGHT. Required at "
        "T3+ and SHOULD vary across scenes in a series — picking "
        "the same ENV_* for every scene defeats the diversity push. "
        "Categories whitelist compatible_environments to keep the "
        "menu coherent with the theme (intersection with style_profile "
        "and niche compat lists when present)."
    ),
    "environment_atmosphere": (
        "Optional atmospheric-element concept tag from "
        "environment.atmosphere namespace. Adds particulate / weather / "
        "light / wind / optical detail beyond the lighting_directive. "
        "Examples: ATM_DUST_MOTES_IN_LIGHT, ATM_INCENSE_SMOKE_CURL, "
        "ATM_RAIN_ON_GLASS, ATM_FOG_ROLLING_IN, ATM_STEAM_FROM_BATH, "
        "ATM_CREPUSCULAR_RAYS, ATM_BREEZE_IN_CURTAIN, "
        "ATM_CINESTILL_HALATION, ATM_BOKEH_BUBBLES. Required at T3+. "
        "Pick an atmosphere that matches the chosen environment_setting "
        "and lighting_directive."
    ),
    # Phase 4 (vocab v6) — environment.prop. Optional polish layer.
    # Per verifier B6, single-tag (not comma-joined multi-tag) to
    # avoid splitter complexity. LLM picks at most one prop per
    # scene when it adds value.
    "environment_prop": (
        "Optional prop / set-dressing concept tag from environment.prop "
        "namespace. Adds named furniture / objects to anchor the scene. "
        "Examples: PROP_CHAISE_LOUNGE_VELVET, PROP_CHESTERFIELD_LEATHER, "
        "PROP_FOUR_POSTER_BED, PROP_SILK_DRAPE, PROP_TAPERED_CANDLE_CLUSTER, "
        "PROP_HANDWRITTEN_LETTER, PROP_OIL_PAINTING_NUDE, "
        "PROP_PEONIES_OVERBLOWN, PROP_WINE_BOTTLE_RED. Pick a prop "
        "that fits the scene's environment_setting + narrative_moment."
    ),
}

# Phase 4 (vocab v6) — composition.principle. Higher-order
# compositional rules beyond angle + framing. Pony OMITS.
_COMPOSITION_ENUM_FIELDS = {
    "composition_principle": (
        "Optional composition-principle concept tag from "
        "composition.principle namespace. Beyond angle/framing, picks "
        "higher-order compositional rules. Examples: "
        "COMP_LEADING_LINES_FLOOR, COMP_NEGATIVE_SPACE_DOMINANT, "
        "COMP_FOREGROUND_MIDGROUND_BACKGROUND, COMP_SYMMETRY_CENTERED, "
        "COMP_LOW_HERO_SHOT, COMP_SILHOUETTE_BACKLIT, "
        "COMP_GOLDEN_RATIO_SPIRAL, COMP_TRIANGULAR, COMP_RULE_OF_ODDS. "
        "Pony omits — booru tags carry composition implicitly via "
        "positional tags."
    ),
}


class SceneFacetSDXL(BaseModel):
    """Per-scene SDXL composer inputs (camera/lens spec + garment).

    Used by ``_compose_keywords``. Two SDXL siblings (e.g.
    ``juggernaut_ragnarok`` + ``gonzalomo_photo_v70``) share this facet row.
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
    # Q10 (vocab v4) — composition vocab gap-fill.
    realism_angle: str | None = Field(default=None, description=_REALISM_ENUM_FIELDS["realism_angle"])
    realism_framing: str | None = Field(default=None, description=_REALISM_ENUM_FIELDS["realism_framing"])
    # Phase 1 (vocab v6) — environment vocab gap-fill (creative uplift).
    environment_setting: str | None = Field(default=None, description=_ENVIRONMENT_ENUM_FIELDS["environment_setting"])
    environment_atmosphere: str | None = Field(default=None, description=_ENVIRONMENT_ENUM_FIELDS["environment_atmosphere"])
    # Phase 2 (vocab v6) — narrative_moment (the #1 leverage axis).
    narrative_moment: str | None = Field(default=None, description=_NARRATIVE_ENUM_FIELDS["narrative_moment"])
    # Phase 4 (vocab v6) — environment.prop + composition.principle polish.
    environment_prop: str | None = Field(default=None, description=_ENVIRONMENT_ENUM_FIELDS["environment_prop"])
    composition_principle: str | None = Field(default=None, description=_COMPOSITION_ENUM_FIELDS["composition_principle"])

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
    # Phase 1 (vocab v6) — environment vocab; Pony participates because
    # environments DO have natural booru-tag representations (unlike
    # camera/lens/film_stock which Pony's booru convention omits).
    environment_setting: str | None = Field(default=None, description=_ENVIRONMENT_ENUM_FIELDS["environment_setting"])
    environment_atmosphere: str | None = Field(default=None, description=_ENVIRONMENT_ENUM_FIELDS["environment_atmosphere"])
    # Phase 2 (vocab v6) — narrative_moment. Pony participates: booru
    # tags carry narrative moments natively (looking_at_mirror,
    # arranging_flowers, lighting_cigarette, etc.).
    narrative_moment: str | None = Field(default=None, description=_NARRATIVE_ENUM_FIELDS["narrative_moment"])
    # Phase 4 (vocab v6) — Pony participates in environment.prop
    # (props have natural booru tags) but OMITS composition.principle
    # (booru tags carry composition implicitly via from_above /
    # from_behind / etc.).
    environment_prop: str | None = Field(default=None, description=_ENVIRONMENT_ENUM_FIELDS["environment_prop"])

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
    # Q10 (vocab v4) — composition vocab gap-fill.
    realism_angle: str | None = Field(default=None, description=_REALISM_ENUM_FIELDS["realism_angle"])
    realism_framing: str | None = Field(default=None, description=_REALISM_ENUM_FIELDS["realism_framing"])
    # Phase 1 (vocab v6) — environment vocab gap-fill.
    environment_setting: str | None = Field(default=None, description=_ENVIRONMENT_ENUM_FIELDS["environment_setting"])
    environment_atmosphere: str | None = Field(default=None, description=_ENVIRONMENT_ENUM_FIELDS["environment_atmosphere"])
    # Phase 2 (vocab v6) — narrative_moment (the #1 leverage axis).
    narrative_moment: str | None = Field(default=None, description=_NARRATIVE_ENUM_FIELDS["narrative_moment"])
    # Phase 4 (vocab v6) — environment.prop + composition.principle polish.
    environment_prop: str | None = Field(default=None, description=_ENVIRONMENT_ENUM_FIELDS["environment_prop"])
    composition_principle: str | None = Field(default=None, description=_COMPOSITION_ENUM_FIELDS["composition_principle"])

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
        description="2–4 complete sentences, 30–80 words total. Weave "
        "pose, lighting, lens character, environment, and mood into "
        "flowing prose. Describe the woman + her pose + her gaze + "
        "the room in vivid concrete detail. Do NOT weave series-level "
        "photographer / art-movement / palette references into the "
        "prose (composer canonicalizes those globally). No comma-tag "
        "lists, no weighting syntax.",
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
    # Q10 (vocab v4) — composition vocab gap-fill.
    realism_angle: str | None = Field(default=None, description=_REALISM_ENUM_FIELDS["realism_angle"])
    realism_framing: str | None = Field(default=None, description=_REALISM_ENUM_FIELDS["realism_framing"])
    # Phase 1 (vocab v6) — environment vocab gap-fill.
    environment_setting: str | None = Field(default=None, description=_ENVIRONMENT_ENUM_FIELDS["environment_setting"])
    environment_atmosphere: str | None = Field(default=None, description=_ENVIRONMENT_ENUM_FIELDS["environment_atmosphere"])
    # Phase 2 (vocab v6) — narrative_moment (the #1 leverage axis).
    narrative_moment: str | None = Field(default=None, description=_NARRATIVE_ENUM_FIELDS["narrative_moment"])
    # Phase 4 (vocab v6) — environment.prop + composition.principle polish.
    environment_prop: str | None = Field(default=None, description=_ENVIRONMENT_ENUM_FIELDS["environment_prop"])
    composition_principle: str | None = Field(default=None, description=_COMPOSITION_ENUM_FIELDS["composition_principle"])

    @model_validator(mode="after")
    def _check_prose_shape(self) -> "SceneFacetFluxNatural":
        """Prose families want flowing sentences, not tag soup. Reject:
        * ``(word:1.3)`` weighting syntax (Flux/Chroma ignore it),
        * tag-soup heuristic — more than 4 commas per sentence,
        * underscored multi-word tags (``long_hair`` etc. — booru style).

        Round-22 (2026-05-22) — soft warn on word-count band 40–90
        (target) vs 20–140 (hard reject). Pre-round-22 the prose was
        capped at "1–3 sentences" which the facet LLM saturated when
        trying to compress pose + lighting + lens + env + mood + 11+
        tags into 30 words — then nulled structured fields under
        attention pressure. Looser band lets the LLM write properly
        descriptive prose without the saturation failure mode.
        """
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
        # 2026-05-23 dual-write pivot — band calibration iter 2.
        # Iteration 2 (series_ffd7509f0af3, 0/5 facets at 60-floor)
        # showed DavidAU 12B's actual range is 45-90 words despite
        # system-prompt directive for 100-250. Adopt LLM's empirical
        # mode: 40-350 hard, 100-250 target (warn only). Composer-
        # drop still delivers the architectural win — eliminating
        # 12-canon-sentence tag soup — even at 50-word LLM prose,
        # because the parallel attribute sentences stop firing entirely
        # for prose families. Final prompt = solo anchor (~5w) +
        # adult anchor (~5w) + scene_prose (~50-90w) + camera tail
        # (~10w) + realism tail (~5w) ≈ 75-115 words — MUCH cleaner
        # than the old 250-word tag-soup outputs that scored 4.88/10.
        words = len(prose.split())
        if not (40 <= words <= 350):
            raise ValueError(
                f"SceneFacetFluxNatural.scene_prose has {words} words; "
                f"prose-family hard band is 40-350 (target 100-250). "
                f"At dual-write contract, scene_prose IS the prompt body "
                f"and must cover subject + pose + anatomy + light + env "
                f"+ mood + style in ONE coherent paragraph."
            )
        if not (100 <= words <= 250):
            logger.warning(
                "SceneFacetFluxNatural.scene_prose word count %d is "
                "outside the 100-250 target band (still inside 40-350 "
                "slack). Tighten or expand for better axis coverage.",
                words,
            )

        # 2026-05-23 — max ONE photographer reference per prompt.
        # Verifier audit (Grok + Claude web): Lindbergh + Newton +
        # Caravaggio simultaneously = mud. Photographer names belong
        # in art_style_reference structured tag, not in prose body —
        # but the LLM sometimes leaks them. Cap at 1 mention.
        _PHOTOGRAPHER_NAMES = (
            "Helmut Newton", "Peter Lindbergh", "Herb Ritts",
            "Robert Mapplethorpe", "Petter Hegre", "Slim Aarons",
            "Annie Leibovitz", "Mario Testino", "Richard Avedon",
            "Sarah Moon", "Paolo Roversi", "Bill Henson",
            "Petra Collins", "Gregory Crewdson", "Nan Goldin",
            "Irving Penn",
        )
        prose_lower = prose.lower()
        photog_hits = [n for n in _PHOTOGRAPHER_NAMES if n.lower() in prose_lower]
        if len(photog_hits) > 1:
            raise ValueError(
                f"SceneFacetFluxNatural.scene_prose contains {len(photog_hits)} "
                f"photographer references {photog_hits} — max 1 allowed. "
                f"Photographer schools have opposite aesthetics; stacking "
                f"them produces averaged-out mud. Pick ONE photographer "
                f"reference and remove the others from scene_prose."
            )

        # Round-22 F15 (2026-05-22) — env / atmosphere / narrative
        # cross-field coherence check. Some ATM / NARR tags declare a
        # ``place_constraint`` requiring specific env keywords (e.g.
        # ATM_FABRIC_FLOATING_UNDERWATER requires the
        # environment_setting prose to mention water / pool / bath).
        # If the LLM picked an incompatible combination (env: fire
        # escape, atm: underwater fabric), reject the facet here so
        # the existing retry-with-nudge mechanism in scene_facet_
        # generator.py fires. Grok's audit on series_bf84fa88de1c
        # caught exactly this pattern.
        from src.prompt.vocabulary import check_facet_env_coherence
        violations = check_facet_env_coherence(
            environment_setting=self.environment_setting,
            environment_atmosphere=self.environment_atmosphere,
            narrative_moment=self.narrative_moment,
        )
        if violations:
            joined = "; ".join(
                f"{field}={tag!r}: {reason}"
                for field, tag, reason in violations
            )
            raise ValueError(
                f"SceneFacetFluxNatural cross-field coherence failed: "
                f"{joined}. Re-pick atm / narrative or change "
                f"environment_setting so they imply ONE physically "
                f"plausible scene."
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
    # Q10 (vocab v4) — composition vocab gap-fill.
    realism_angle: str | None = Field(default=None, description=_REALISM_ENUM_FIELDS["realism_angle"])
    realism_framing: str | None = Field(default=None, description=_REALISM_ENUM_FIELDS["realism_framing"])
    # Phase 1 (vocab v6) — environment vocab gap-fill.
    environment_setting: str | None = Field(default=None, description=_ENVIRONMENT_ENUM_FIELDS["environment_setting"])
    environment_atmosphere: str | None = Field(default=None, description=_ENVIRONMENT_ENUM_FIELDS["environment_atmosphere"])
    # Phase 2 (vocab v6) — narrative_moment (the #1 leverage axis).
    narrative_moment: str | None = Field(default=None, description=_NARRATIVE_ENUM_FIELDS["narrative_moment"])
    # Phase 4 (vocab v6) — environment.prop + composition.principle polish.
    environment_prop: str | None = Field(default=None, description=_ENVIRONMENT_ENUM_FIELDS["environment_prop"])
    composition_principle: str | None = Field(default=None, description=_COMPOSITION_ENUM_FIELDS["composition_principle"])

    @model_validator(mode="after")
    def _check_word_count_band(self) -> "SceneFacetFlux2":
        """2026-05-23 dual-write pivot iter2 — band adjusted to LLM's
        actual empirical range. 40-350 hard, 100-250 target (warn)."""
        prose = self.scene_prose or ""
        words = len(prose.split())
        if not (40 <= words <= 350):
            raise ValueError(
                f"SceneFacetFlux2.scene_prose has {words} words; dual-"
                f"write pivot band is 40-350 (target 100-250). "
                f"scene_prose IS the prompt body; weave subject + pose "
                f"+ anatomy + light + env + mood + style into ONE "
                f"coherent paragraph."
            )
        if not (100 <= words <= 250):
            logger.warning(
                "SceneFacetFlux2.scene_prose word count %d is outside "
                "the 100-250 target band (still inside 40-350 slack). "
                "Consider tightening or expanding.", words,
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


# ── Metadata (output of MetadataGenerator) ──────────────────────────


class MetadataSchema(BaseModel):
    """LLM-generated post metadata for a finished image set.

    Bound to ``MetadataGenerator.REQUIRED_FIELDS = {title, description,
    tags}``. Constrained decoding via Ollama ``format:`` enforces the
    shape at the wire; Pydantic post-validation catches per-field
    constraints (length bands + tag count + non-empty strings).

    The bands are deliberately wide so the LLM has room — the goal is
    to reject obvious failures (single-word title, empty description,
    no tags) without rejecting reasonable creative output.
    """

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    title: str = Field(min_length=4, max_length=120)
    description: str = Field(min_length=20, max_length=600)
    tags: list[str] = Field(min_length=3, max_length=25)

    @field_validator("tags")
    @classmethod
    def _filter_blank_tags(cls, v: list[str]) -> list[str]:
        cleaned = [t.strip() for t in v if isinstance(t, str) and t.strip()]
        if len(cleaned) < 3:
            raise ValueError(
                "tags must contain at least 3 non-empty strings after "
                "stripping"
            )
        return cleaned


# CharacterSchema (LLM-generated character identity) deleted 2026-05-20
# with the character-mode + character_creator.py removal.

# ── Few-shot example (Q8) ───────────────────────────────────────────


class FewShotExample(BaseModel):
    """Tier-stratified few-shot example for the SceneFacetGenerator.

    Each family in ``config/families.yaml`` declares an ``examples:``
    list with at least 3 entries covering both T2 and T4 — the
    SceneFacetGenerator's system prompt selects the example whose
    ``tier`` matches the active ``content_level`` so the LLM sees a
    same-tier exemplar instead of the legacy single-prompt boilerplate.

    Per-model YAMLs may extend or override (``prompt.extend.examples``
    appends; ``prompt.override.examples`` replaces) — same merge
    semantics as ``trigger_words`` / ``avoid_words``.

    Fields:
      * ``tier`` — content-level this example targets.
      * ``scene`` — the model-agnostic scene core that drives the
        facet (pose, camera, lighting, environment, mood). Free-shape
        dict so families can illustrate the fields they care about.
      * ``expected_facet`` — the canonical output for that scene; the
        renderer formats it so the LLM sees "input → expected output".
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    tier: str = Field(
        description="One of T1_suggestive / T2_implied / T3_artnude / "
        "T4_explicit.",
    )
    scene: dict[str, Any] = Field(
        description="Input scene-core dict the example illustrates.",
    )
    expected_facet: dict[str, Any] = Field(
        description="Canonical facet output for the example scene.",
    )

    @field_validator("tier")
    @classmethod
    def _validate_tier(cls, v: str) -> str:
        valid = {"T1_suggestive", "T2_implied", "T3_artnude", "T4_explicit"}
        if v not in valid:
            raise ValueError(
                f"tier must be one of {sorted(valid)}; got {v!r}"
            )
        return v

