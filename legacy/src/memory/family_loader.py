"""Family config loader — reads ``config/families.yaml``.

A "family" is a model-architecture archetype (sdxl, pony, illustrious,
flux, chroma). It declares how the PromptBuilder composes text for the
models that belong to it: which composer to run, whether negatives /
weighting are meaningful, what quality prefix/suffix to inject, and
what the LLM system prompt hints should say.

Per-model YAMLs (``config/models/*.yaml``) reference a family by id
and may extend/override individual fields (see ``merge_overrides``).

Load-once: the YAML is immutable at runtime. The module-level singleton
avoids re-parsing the file on every PromptBuilder call.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.prompt.negative_axes import (
    AXIS_KEYS,
    empty_axes,
    flatten_axes,
    normalize_axes,
)

logger = logging.getLogger(__name__)

# Family fields that feed into LLM system prompts. A family missing any
# of these still loads (defaults are empty strings) but the
# ``scene_generator`` / ``series_planner`` system prompts silently lose
# that guidance — quality degrades without a visible error. We warn at
# load time so the gap is traceable without raising.
_LLM_GUIDANCE_FIELDS = ("llm_hint", "structure_rules", "example_prompt")


CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "config" / "families.yaml"
)


class FamilyLoaderError(Exception):
    """Base for family loader errors."""


class FamilyNotFound(FamilyLoaderError):
    """Lookup by id returned no row."""


@dataclass(frozen=True)
class FamilyConfig:
    """One family row from ``config/families.yaml``.

    Mirrors §5.1 of the refactor plan. ``quality_prefix`` and
    ``quality_suffix`` are kept as lists (not strings) so the composer
    can decide how to join them (e.g. Pony prepends a ``BREAK`` token
    mid-list, Illustrious comma-joins the suffix).
    """

    id: str
    prompt_style: str
    quality_prefix: list[str] = field(default_factory=list)
    quality_suffix: list[str] = field(default_factory=list)
    # Phase D — 7-axis negative taxonomy (anatomy/medium/skin/quality/
    # watermark/safety/censor). The flattened ``negative_prompt`` string
    # is derived from this; consumers that still want the raw string can
    # call ``family.negative_prompt`` (a plain attribute, materialized at
    # load time so it stays a frozen dataclass field).
    negative_axes: dict[str, list[str]] = field(default_factory=empty_axes)
    negative_prompt: str = ""
    supports_negative_prompt: bool = True
    supports_weighting: bool = True
    clip_skip: int = 1
    max_tokens: int = 77
    # Phase C — tokenizer backend used by ``fit_to_budget``. Valid
    # values: ``"clip"`` (open_clip SimpleTokenizer for CLIP-based
    # encoders), ``"t5"`` (transformers T5 v1.1 — used by FLUX/Chroma
    # T5-XXL/Schnell), ``"heuristic"`` (word-count fallback for
    # tokenizers we don't ship — currently FLUX.2 / Qwen3-8B).
    tokenizer_id: str = "clip"
    # Optional encoder-window split marker. When set (Pony's
    # ``"BREAK"``), ``fit_to_budget`` treats each side of the marker
    # as an independent budget — Pony's BREAK splits the prompt into
    # two CLIP windows of 75 tokens each.
    break_marker: str | None = None
    structure_rules: str = ""
    avoid_words: list[str] = field(default_factory=list)
    llm_hint: str = ""
    example_prompt: str = ""
    # Optional per-family LLM temperature override. When set, callers
    # should pass this to SeriesPlanner / SceneGenerator / MetadataGenerator
    # instead of the agent-class default (0.7 / 0.7 / 0.6). Booru-tag
    # families (pony) tend to want lower temps (~0.5) for predictable
    # tag structure; prose families (flux) can go higher (~0.8) for
    # varied phrasing. Leave unset to keep the agent default.
    llm_temperature: float | None = None
    # How the realism tail (camera/lens/photoreal tokens) is joined.
    # ``None`` / ``"comma"`` → comma-joined, stitched into the keyword
    # list (SDXL, Pony, Illustrious, Flux).
    # ``"period"`` → period-separated fragments appended to the prose
    # body (Chroma — the lodestones HF card and top Civitai workflows
    # consistently use this shape).
    realism_tail_style: str | None = None
    # Optional per-family prompting guide the LLM system prompt can
    # inject verbatim. FLUX.2 Klein is the only family that uses this
    # so far — BFL prescribes a specific 5-anchor ordering + word-count
    # band. Shape:
    #   {
    #     "structure_order": ["subject", "setting", "details",
    #                         "lighting", "atmosphere"],
    #     "target_words": [30, 80],
    #     "lighting_is_critical": True,
    #   }
    # Absent for all other families; scene_generator.py only injects
    # the extra system-prompt block when this is set.
    guide: dict | None = None
    # Phase 3 — per-family adult-anchor strings. The
    # ``_positive_age_safety_scan`` helper inserts these when it strips
    # age-ambiguity vocabulary from a positive prompt. ``keyword`` is
    # the comma-token form for SDXL/Pony/Illustrious composers; ``prose``
    # is the natural-language sentence for prose families
    # (flux/chroma/flux2). Pony realism finetunes typically override
    # ``keyword`` to ``"1girl, mature_female, adult"`` since `1girl` +
    # `mature_female` is the canonical Danbooru adult-female pair
    # (``1woman`` is not a real Danbooru tag).
    adult_anchor: dict[str, str] = field(default_factory=lambda: {
        "keyword": "adult woman, mature features",
        "prose": "An adult woman with mature features.",
    })
    # Single-female enforcement (2026-05-17). Unlike ``adult_anchor``
    # (which only fires when the age-safety scan strips age-ambiguity
    # vocabulary), ``solo_anchor`` is injected by
    # ``_positive_solo_anchor_scan`` on EVERY composed prompt so the
    # subject-count signal is always present in the positive prompt.
    # ``keyword`` is the booru/SDXL form (booru families use the
    # canonical ``1girl, solo, mature_female`` pair; SDXL realism uses
    # natural keywords); ``prose`` is the natural-language sentence for
    # prose families. Pony-style families' anchor is injected *after*
    # the ``BREAK`` marker so it lands in CLIP window 2 alongside the
    # body, not in window 1 alongside the 6-tier score prefix.
    solo_anchor: dict[str, str] = field(default_factory=lambda: {
        # Keep keyword form MINIMAL — CLIP 77-token budgets are tight
        # and the solo signal only needs ~1 token to land. Booru
        # families override to the canonical `1girl, solo` pair.
        "keyword": "solo",
        "prose": "A single adult woman alone in the scene.",
    })
    # Phase 3 — comma-tokens emitted between ``quality_prefix`` and the
    # composed body. Pony realism finetunes use this to prepend
    # ``[source_photo, "photo (medium)", realistic]`` after the 6-tier
    # score prefix so the photoreal vocabulary is in the LLM-stable
    # leading window. Empty list means "no intro" (default for every
    # family).
    structure_intro: list[str] = field(default_factory=list)
    # Q8 — tier-stratified few-shot examples. Each entry has
    # ``tier`` (T1-T4), ``scene`` (input scene-core dict), and
    # ``expected_facet`` (canonical output). The SceneFacetGenerator
    # picks the closest-tier example at compose time. When empty,
    # falls back to the legacy ``example_prompt`` string.
    examples: list[dict] = field(default_factory=list)
    # 2026-05-20 — per-family default render model + default external
    # template. Used by render_prompts.py when --models / --templates
    # is omitted. ``default_template`` is a path under
    # ``config/comfyui_workflows/`` (e.g. ``templates/chroma/X.json``);
    # validated at load time. ``None`` means "no default; explicit
    # flag required" — render raises a clear error in that case.
    default_render_model: str | None = None
    default_template: str | None = None

    @classmethod
    def from_dict(cls, fam_id: str, d: dict) -> "FamilyConfig":
        required = {"prompt_style"}
        missing = required - d.keys()
        if missing:
            raise FamilyLoaderError(
                f"family {fam_id!r}: missing required keys {sorted(missing)}"
            )
        raw_temp = d.get("llm_temperature")
        llm_temperature = float(raw_temp) if raw_temp is not None else None
        raw_tail = d.get("realism_tail_style")
        realism_tail_style = (
            str(raw_tail).strip().lower() if raw_tail is not None else None
        )
        if realism_tail_style is not None and realism_tail_style not in {
            "comma", "period"
        }:
            raise FamilyLoaderError(
                f"family {fam_id!r}: realism_tail_style must be 'comma', "
                f"'period', or omitted (got {realism_tail_style!r})"
            )
        raw_guide = d.get("guide")
        guide = _parse_guide(fam_id, raw_guide) if raw_guide is not None else None

        # Phase D — every family declares ``negative_axes:`` (the 7-axis
        # taxonomy: anatomy/medium/skin/quality/watermark/safety/censor).
        # Missing → empty axes (no negative prompt for that family).
        raw_axes = d.get("negative_axes")
        if raw_axes is not None:
            try:
                axes = normalize_axes(raw_axes)
            except ValueError as exc:
                raise FamilyLoaderError(
                    f"family {fam_id!r}: {exc}"
                ) from exc
        else:
            axes = empty_axes()
        flat_negative = flatten_axes(axes)

        # Phase 3 — adult_anchor: optional per-family override of the
        # global keyword/prose strings. Missing = use module defaults
        # baked into the dataclass field.
        anchor = _parse_adult_anchor(fam_id, d.get("adult_anchor"))
        # 2026-05-17 — solo_anchor: per-family override for the
        # ``_positive_solo_anchor_inject`` injection text. Booru
        # families override to ``1girl, solo`` (the canonical
        # subject-count pair); other families fall back to the
        # single-token default ``solo`` to fit CLIP 77-budget.
        solo_anchor_data = _parse_solo_anchor(fam_id, d.get("solo_anchor"))
        # Phase 3 — structure_intro: optional list of comma-tokens
        # emitted after quality_prefix.
        intro = list(d.get("structure_intro") or [])
        if not all(isinstance(x, str) for x in intro):
            raise FamilyLoaderError(
                f"family {fam_id!r}: structure_intro must be a list[str]"
            )
        # Q8 — parse tier-stratified examples. Each entry must have
        # tier / scene / expected_facet keys. Validate the tier value
        # at load time so a typo in YAML fails fast.
        raw_examples = d.get("examples") or []
        if not isinstance(raw_examples, list):
            raise FamilyLoaderError(
                f"family {fam_id!r}: examples must be a list (got "
                f"{type(raw_examples).__name__})"
            )
        examples: list[dict] = []
        valid_tiers = {
            "T1_suggestive", "T2_implied", "T3_artnude", "T4_explicit",
        }
        for i, ex in enumerate(raw_examples):
            if not isinstance(ex, dict):
                raise FamilyLoaderError(
                    f"family {fam_id!r}: examples[{i}] must be a dict"
                )
            missing = {"tier", "scene", "expected_facet"} - ex.keys()
            if missing:
                raise FamilyLoaderError(
                    f"family {fam_id!r}: examples[{i}] missing keys "
                    f"{sorted(missing)}"
                )
            if ex["tier"] not in valid_tiers:
                raise FamilyLoaderError(
                    f"family {fam_id!r}: examples[{i}].tier must be one "
                    f"of {sorted(valid_tiers)}; got {ex['tier']!r}"
                )
            examples.append({
                "tier": ex["tier"],
                "scene": dict(ex["scene"] or {}),
                "expected_facet": dict(ex["expected_facet"] or {}),
            })

        return cls(
            id=fam_id,
            prompt_style=d["prompt_style"],
            quality_prefix=list(d.get("quality_prefix") or []),
            quality_suffix=list(d.get("quality_suffix") or []),
            negative_axes=axes,
            negative_prompt=flat_negative,
            supports_negative_prompt=bool(d.get("supports_negative_prompt", True)),
            supports_weighting=bool(d.get("supports_weighting", True)),
            clip_skip=int(d.get("clip_skip", 1)),
            max_tokens=int(d.get("max_tokens", 77)),
            tokenizer_id=str(d.get("tokenizer_id") or "clip").strip().lower(),
            break_marker=(
                str(d["break_marker"]).strip()
                if d.get("break_marker")
                else None
            ),
            structure_rules=(d.get("structure_rules") or "").strip(),
            avoid_words=list(d.get("avoid_words") or []),
            llm_hint=(d.get("llm_hint") or "").strip(),
            example_prompt=(d.get("example_prompt") or "").strip(),
            llm_temperature=llm_temperature,
            realism_tail_style=realism_tail_style,
            guide=guide,
            adult_anchor=anchor,
            solo_anchor=solo_anchor_data,
            structure_intro=intro,
            examples=examples,
            default_render_model=(
                str(d["default_render_model"])
                if d.get("default_render_model") else None
            ),
            default_template=(
                str(d["default_template"])
                if d.get("default_template") else None
            ),
        )


class FamilyLoader:
    """Read-only accessor for ``config/families.yaml``.

    Holds the full dict in memory; lookups are O(1). The loader is
    stateless apart from the one-shot YAML read, so instantiating it
    repeatedly is cheap (the parse is cached module-wide).
    """

    def __init__(self, config_path: Path | str | None = None) -> None:
        self.config_path = Path(config_path) if config_path else CONFIG_PATH
        self._families = _load_families(self.config_path)

    def get_family(self, family_id: str) -> FamilyConfig:
        try:
            return self._families[family_id]
        except KeyError:
            raise FamilyNotFound(
                f"Family {family_id!r} not found in {self.config_path}. "
                f"Known families: {sorted(self._families)}"
            ) from None

    def list_families(self) -> list[FamilyConfig]:
        return [self._families[k] for k in sorted(self._families)]

    def has_family(self, family_id: str) -> bool:
        return family_id in self._families


_DEFAULT_ADULT_ANCHOR = {
    "keyword": "adult woman, mature features",
    "prose": "An adult woman with mature features.",
}


# Single-female enforcement (2026-05-17). Booru families override to
# ``1girl, solo``; everything else falls back to the single-token
# ``solo`` form (fits inside CLIP 77-token budget cleanly).
_DEFAULT_SOLO_ANCHOR = {
    "keyword": "solo",
    "prose": "A single adult woman alone in the scene.",
}


def _parse_solo_anchor(fam_id: str, raw: object) -> dict[str, str]:
    """Parse the optional ``solo_anchor:`` block.

    Per-family override of the ``_positive_solo_anchor_inject``
    insertion text. Both ``keyword`` and ``prose`` keys are required
    when the block is present so a typo can't silently drop one form.
    Missing block falls back to :data:`_DEFAULT_SOLO_ANCHOR`
    (``"solo"`` keyword + a generic prose sentence) — adequate for
    most families; Pony / Illustrious override to the canonical
    booru ``1girl, solo`` pair.
    """
    if raw is None:
        return dict(_DEFAULT_SOLO_ANCHOR)
    if not isinstance(raw, dict):
        raise FamilyLoaderError(
            f"family {fam_id!r}: solo_anchor must be a mapping "
            f"(got {type(raw).__name__})"
        )
    missing = {"keyword", "prose"} - raw.keys()
    if missing:
        raise FamilyLoaderError(
            f"family {fam_id!r}: solo_anchor missing keys {sorted(missing)}"
        )
    keyword = raw["keyword"]
    prose = raw["prose"]
    if not isinstance(keyword, str) or not keyword.strip():
        raise FamilyLoaderError(
            f"family {fam_id!r}: solo_anchor.keyword must be a non-empty string"
        )
    if not isinstance(prose, str) or not prose.strip():
        raise FamilyLoaderError(
            f"family {fam_id!r}: solo_anchor.prose must be a non-empty string"
        )
    return {"keyword": keyword.strip(), "prose": prose.strip()}


def _parse_adult_anchor(fam_id: str, raw: object) -> dict[str, str]:
    """Parse the optional ``adult_anchor:`` block.

    Per-family override of the ``_positive_age_safety_scan`` insertion
    text. Both ``keyword`` and ``prose`` keys are required when the
    block is present so a typo can't silently drop one form.
    """
    if raw is None:
        return dict(_DEFAULT_ADULT_ANCHOR)
    if not isinstance(raw, dict):
        raise FamilyLoaderError(
            f"family {fam_id!r}: adult_anchor must be a mapping "
            f"(got {type(raw).__name__})"
        )
    missing = {"keyword", "prose"} - raw.keys()
    if missing:
        raise FamilyLoaderError(
            f"family {fam_id!r}: adult_anchor missing keys {sorted(missing)}"
        )
    keyword = raw["keyword"]
    prose = raw["prose"]
    if not isinstance(keyword, str) or not keyword.strip():
        raise FamilyLoaderError(
            f"family {fam_id!r}: adult_anchor.keyword must be a non-empty string"
        )
    if not isinstance(prose, str) or not prose.strip():
        raise FamilyLoaderError(
            f"family {fam_id!r}: adult_anchor.prose must be a non-empty string"
        )
    return {"keyword": keyword.strip(), "prose": prose.strip()}


def _parse_guide(fam_id: str, raw: object) -> dict:
    """Validate the optional ``guide:`` block.

    Only the shape is validated; the dict is passed verbatim to
    scene_generator. All three keys are required when the block is
    present so a typo can't silently drop guidance.
    """
    if not isinstance(raw, dict):
        raise FamilyLoaderError(
            f"family {fam_id!r}: guide must be a mapping (got {type(raw).__name__})"
        )
    missing = {"structure_order", "target_words", "lighting_is_critical"} - raw.keys()
    if missing:
        raise FamilyLoaderError(
            f"family {fam_id!r}: guide missing keys {sorted(missing)}"
        )
    structure = raw["structure_order"]
    if not (isinstance(structure, list) and all(isinstance(x, str) for x in structure)):
        raise FamilyLoaderError(
            f"family {fam_id!r}: guide.structure_order must be a list[str]"
        )
    tw = raw["target_words"]
    if not (isinstance(tw, list) and len(tw) == 2
            and all(isinstance(x, int) for x in tw) and tw[0] < tw[1]):
        raise FamilyLoaderError(
            f"family {fam_id!r}: guide.target_words must be [min, max] ints with min<max"
        )
    if not isinstance(raw["lighting_is_critical"], bool):
        raise FamilyLoaderError(
            f"family {fam_id!r}: guide.lighting_is_critical must be bool"
        )
    return {
        "structure_order": list(structure),
        "target_words": [int(tw[0]), int(tw[1])],
        "lighting_is_critical": bool(raw["lighting_is_critical"]),
    }


@functools.lru_cache(maxsize=4)
def _load_families(config_path: Path) -> dict[str, FamilyConfig]:
    if not config_path.exists():
        raise FamilyLoaderError(f"families.yaml not found at {config_path}")
    with open(config_path) as f:
        data = yaml.safe_load(f) or {}
    families_raw = data.get("families")
    if not isinstance(families_raw, dict) or not families_raw:
        raise FamilyLoaderError(
            f"{config_path}: expected top-level `families:` mapping"
        )
    families = {
        fid: FamilyConfig.from_dict(fid, fdict)
        for fid, fdict in families_raw.items()
    }
    for fam in families.values():
        empty = [f for f in _LLM_GUIDANCE_FIELDS if not getattr(fam, f)]
        if empty:
            logger.warning(
                "family %r: empty LLM guidance field(s) %s — scene/plan "
                "system prompts will omit that guidance",
                fam.id, empty,
            )
    return families
