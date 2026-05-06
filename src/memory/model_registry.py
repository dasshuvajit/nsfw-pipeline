"""Model registry — YAML-backed lookup over ``config/models/*.yaml``.

Replaces the old ``model_registry`` + ``model_prompt_guides`` SQLite
tables. YAML is the single source of truth for model config; the
registry read path loads all files at startup, caches dataclasses, and
serves two consumers:

  1. **Style profile dereference** — ``StyleProfileLoader`` provides
     ``model_id``; this class resolves that to the checkpoint filename,
     family, and default sampler/cfg the WorkflowBuilder needs.
  2. **--model CLI override** — ``render_set.py`` / ``compare_models.py``
     look up the CLI-provided id and use its registry defaults.

``get_prompt_guide(model_id)`` returns a *merged* ``ModelPromptGuide``
— family defaults + per-model ``prompt.extend`` / ``prompt.override``
applied by ``src.core.merge_overrides``. Callers never see the raw
family or per-model dicts; they get the effective config.
"""

from __future__ import annotations

import functools
import glob
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.core.merge_overrides import merge_prompt_config
from src.memory.family_loader import FamilyConfig, FamilyLoader

logger = logging.getLogger(__name__)


class ModelRegistryError(Exception):
    """Base for model registry errors."""


class ModelNotFound(ModelRegistryError):
    """Lookup by id returned no row, or the row is marked ``active=False``."""


MODELS_DIR = (
    Path(__file__).resolve().parent.parent.parent / "config" / "models"
)


@dataclass(frozen=True)
class ModelRegistryEntry:
    """One per-model YAML, typed.

    ``family`` is the archetype id (see ``config/families.yaml``).
    Callers who need the full ``FamilyConfig`` should ask the loader
    via ``get_family(entry.family)`` — we keep only the string here so
    per-model rows stay decoupled from family load order.
    """

    id: str
    display_name: str
    filename: str
    architecture: str
    family: str
    default_sampler: str
    default_scheduler: str
    default_steps: int
    default_cfg: float
    default_clip_skip: int | None
    supports_ipadapter: bool
    supports_lora: bool
    resolution_portrait: tuple[int, int] | None
    resolution_square: tuple[int, int] | None
    resolution_landscape: tuple[int, int] | None
    vae_filename: str | None
    text_encoder: str | None
    notes: str | None
    active: bool
    # License metadata — only used by the commercial_mode gate. Default
    # to commercial_use=True for legacy models so pre-existing YAMLs that
    # don't declare a license stay unchanged.
    license: str | None
    commercial_use: bool
    # Enabled LoRAs, merged into StyleProfileForWorkflow.lora_stack when
    # the profile provides none. Frozen tuple of {"name", "strength"}
    # dicts — keep to the two fields the WorkflowBuilder reads.
    lora_stack: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_dict(cls, d: dict[str, Any], source: Path) -> "ModelRegistryEntry":
        required = {
            "id", "display_name", "filename", "architecture", "family",
            "default_sampler", "default_scheduler", "default_steps", "default_cfg",
        }
        missing = required - d.keys()
        if missing:
            raise ModelRegistryError(
                f"{source}: missing required keys {sorted(missing)}"
            )
        raw_loras = d.get("lora_stack") or []
        enabled_loras = tuple(
            {"name": str(e["name"]), "strength": float(e["strength"])}
            for e in raw_loras
            if e.get("enabled") is True
        )
        if len(enabled_loras) > 2:
            raise ModelRegistryError(
                f"{source}: lora_stack has {len(enabled_loras)} enabled "
                f"entries; max 2 per render (CLAUDE.md invariant)."
            )
        return cls(
            id=str(d["id"]),
            display_name=str(d["display_name"]),
            filename=str(d["filename"]),
            architecture=str(d["architecture"]),
            family=str(d["family"]),
            default_sampler=str(d["default_sampler"]),
            default_scheduler=str(d["default_scheduler"]),
            default_steps=int(d["default_steps"]),
            default_cfg=float(d["default_cfg"]),
            default_clip_skip=(
                None if d.get("default_clip_skip") is None
                else int(d["default_clip_skip"])
            ),
            supports_ipadapter=bool(d.get("supports_ipadapter", False)),
            supports_lora=bool(d.get("supports_lora", False)),
            resolution_portrait=_parse_resolution(d.get("resolution_portrait")),
            resolution_square=_parse_resolution(d.get("resolution_square")),
            resolution_landscape=_parse_resolution(d.get("resolution_landscape")),
            vae_filename=(d.get("vae_filename") or None),
            text_encoder=(d.get("text_encoder") or None),
            notes=(d.get("notes") or None),
            active=bool(d.get("active", True)),
            license=(d.get("license") or None),
            commercial_use=bool(d.get("commercial_use", True)),
            lora_stack=enabled_loras,
        )


@dataclass(frozen=True)
class ModelPromptGuide:
    """Effective prompt config for a model — family + per-model merged.

    The merge collapses family defaults with the per-model ``prompt:``
    block (``extend:`` appends, ``override:`` replaces) so callers can
    treat this as the single source of truth.

    ``model_id`` is ``None`` for family-level guides (built via
    :meth:`ModelRegistryLoader.get_family_only_prompt_guide`) — those
    omit per-model overlay fields entirely.
    """

    model_id: str | None
    prompt_style: str
    max_prompt_tokens: int | None
    trigger_words: list[str] = field(default_factory=list)
    avoid_words: list[str] = field(default_factory=list)
    structure_rules: str | None = None
    base_negative_prompt: str | None = None
    example_prompt: str | None = None
    quality_prefix: list[str] = field(default_factory=list)
    quality_suffix: list[str] = field(default_factory=list)
    llm_hint: str | None = None
    supports_negative_prompt: bool = True
    supports_weighting: bool = True
    notes: str | None = None
    # Textual-inversion negative embeddings — list of ComfyUI-syntax
    # tokens like "embedding:BadDream" or "embedding:UnrealisticDream:0.8".
    # Hoisted to the front of the assembled negative by
    # ``PromptBuilder.assemble_negative_prompt``. Sourced from
    # ``prompt.extend.negative_embeddings`` in per-model YAML; merged via
    # ``merge_prompt_config``.
    negative_embeddings: list[str] = field(default_factory=list)
    # Phase D — 7-axis negative taxonomy (anatomy/medium/skin/quality/
    # watermark/safety/censor). Sourced from family defaults + per-model
    # ``prompt.extend.negative_axes`` / ``prompt.override.negative_axes``.
    # The flattened ``base_negative_prompt`` is derived from these axes;
    # callers that want axis-level control can read this dict directly.
    negative_axes: dict[str, list[str]] = field(default_factory=dict)


class ModelRegistryLoader:
    """YAML-backed accessor for ``config/models/*.yaml``.

    One instance parses every file under ``models_dir`` on construction;
    subsequent calls are in-memory lookups. Safe to instantiate more
    than once — the underlying parse is memoized per path.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,  # accepted for API compat; unused
        *,
        models_dir: str | Path | None = None,
        families_path: str | Path | None = None,
        commercial_mode: bool = False,
    ) -> None:
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self._family_loader = FamilyLoader(families_path)
        cached_models, cached_raw = _load_models(self.models_dir)
        # Copy per-instance — _apply_commercial_gate mutates these, and
        # _load_models is LRU-cached so sharing would poison the cache
        # across loaders (e.g. a test using commercial_mode=True would
        # permanently hide gated models from the default loader).
        self._models = dict(cached_models)
        self._raw = dict(cached_raw)
        self._validate_family_refs()
        self._commercial_mode = bool(commercial_mode)
        self._apply_commercial_gate()

    def _apply_commercial_gate(self) -> None:
        """Drop non-commercial models from the registry when the pipeline
        is flagged as commercial.

        Why a hard drop rather than a soft warning: the pipeline ships
        output to DA/Patreon/Fanvue, and a non-commercial license (FLUX
        NCL, CreativeML OpenRAIL-NC, etc.) can be a legally actionable
        violation if the resulting render is sold. Silently rendering
        anyway would be a liability foot-gun. Callers in commercial_mode
        will get `ModelNotFound` for any gated model and must explicitly
        flip commercial_mode off or switch to a commercial-safe model.
        """
        if not self._commercial_mode:
            return
        dropped = []
        for mid, entry in list(self._models.items()):
            if not entry.commercial_use:
                logger.error(
                    "commercial_mode=true: refusing to register model %r "
                    "(license=%s, commercial_use=False)",
                    mid,
                    entry.license or "unknown",
                )
                del self._models[mid]
                self._raw.pop(mid, None)
                dropped.append(mid)
        if dropped:
            logger.warning(
                "commercial_mode=true dropped %d non-commercial model(s): %s",
                len(dropped), sorted(dropped),
            )

    def _validate_family_refs(self) -> None:
        for entry in self._models.values():
            if not self._family_loader.has_family(entry.family):
                raise ModelRegistryError(
                    f"Model {entry.id!r} references unknown family "
                    f"{entry.family!r}. Valid: "
                    f"{[f.id for f in self._family_loader.list_families()]}"
                )

    # ── Registry lookups ────────────────────────────────────────────
    def get_model(
        self, model_id: str, *, require_active: bool = True
    ) -> ModelRegistryEntry:
        entry = self._models.get(model_id)
        if entry is None:
            raise ModelNotFound(
                f"Model {model_id!r} not found in {self.models_dir}. "
                f"Run `python scripts/list_models.py` to see valid ids."
            )
        if require_active and not entry.active:
            raise ModelNotFound(
                f"Model {model_id!r} is marked inactive. Run "
                f"`python scripts/list_models.py --all` to see all models."
            )
        return entry

    def list_models(
        self,
        *,
        family: str | None = None,
        include_inactive: bool = False,
    ) -> list[ModelRegistryEntry]:
        out = []
        for mid in sorted(self._models):
            e = self._models[mid]
            if family and e.family != family:
                continue
            if not include_inactive and not e.active:
                continue
            out.append(e)
        return out

    # ── Family + prompt guide ───────────────────────────────────────
    def get_family(self, family_id: str) -> FamilyConfig:
        return self._family_loader.get_family(family_id)

    def get_prompt_guide(self, model_id: str) -> ModelPromptGuide | None:
        """Return the merged effective prompt config for a model.

        Never raises on missing data — a model with no ``prompt:`` block
        falls through to the pure family defaults.
        """
        entry = self._models.get(model_id)
        if entry is None:
            return None

        raw = self._raw[model_id]
        family = self._family_loader.get_family(entry.family)
        merged = merge_prompt_config(family, raw.get("prompt"))

        neg = merged.get("negative_prompt") or None
        example = merged.get("example_prompt") or None
        structure = merged.get("structure_rules") or None
        llm_hint = merged.get("llm_hint") or None

        return ModelPromptGuide(
            model_id=model_id,
            prompt_style=merged["prompt_style"],
            max_prompt_tokens=(
                None if merged.get("max_tokens") is None
                else int(merged["max_tokens"])
            ),
            trigger_words=list(merged.get("trigger_words") or []),
            avoid_words=list(merged.get("avoid_words") or []),
            structure_rules=structure,
            base_negative_prompt=neg,
            example_prompt=example,
            quality_prefix=list(merged.get("quality_prefix") or []),
            quality_suffix=list(merged.get("quality_suffix") or []),
            llm_hint=llm_hint,
            supports_negative_prompt=bool(merged.get("supports_negative_prompt", True)),
            supports_weighting=bool(merged.get("supports_weighting", True)),
            notes=entry.notes,
            negative_embeddings=list(merged.get("negative_embeddings") or []),
            negative_axes={
                axis: list(tokens)
                for axis, tokens in (merged.get("negative_axes") or {}).items()
            },
        )

    def get_family_only_prompt_guide(
        self, family_id: str,
    ) -> ModelPromptGuide:
        """Return a family-level prompt guide — no per-model overlay.

        Used by ``GenerationContext.build_family_context`` for family-mode
        prompt preparation (`prepare_prompts --families <family_id>`).
        Built by passing ``raw_prompt=None`` to
        :func:`src.core.merge_overrides.merge_prompt_config`, which
        returns a clean family-only dict with no per-model
        ``trigger_words`` / ``avoid_words`` / ``negative_embeddings``
        / ``example_prompt`` / ``llm_hint`` / ``structure_rules`` overlay.

        The resulting :class:`ModelPromptGuide` carries:
          * ``model_id=None`` — no per-model id; the family is the
            target.
          * ``prompt_style`` and ``max_prompt_tokens`` — from family.
          * ``quality_prefix`` / ``quality_suffix`` — from family
            (no per-model append).
          * ``negative_axes`` — family-level only (no per-model
            anatomy/medium/skin axes appended).
          * ``trigger_words`` / ``avoid_words`` /
            ``negative_embeddings`` — empty lists (purely per-model
            constructs).
          * ``example_prompt`` / ``llm_hint`` / ``structure_rules`` —
            None (purely per-model overrides; the family's own
            ``llm_hint``/``structure_rules`` flow through directly
            via the ``family`` arg downstream).

        Raises ``FamilyNotFound`` if ``family_id`` is not in
        ``config/families.yaml``.
        """
        family = self._family_loader.get_family(family_id)
        merged = merge_prompt_config(family, None)

        # Per the docstring: per-model overlay fields are deliberately
        # blanked even if merge_prompt_config would have populated them
        # from family defaults (none today, but defensive).
        return ModelPromptGuide(
            model_id=None,
            prompt_style=merged["prompt_style"],
            max_prompt_tokens=(
                None if merged.get("max_tokens") is None
                else int(merged["max_tokens"])
            ),
            trigger_words=[],
            avoid_words=[],
            structure_rules=None,
            base_negative_prompt=(
                merged.get("negative_prompt") or None
            ),
            example_prompt=None,
            quality_prefix=list(merged.get("quality_prefix") or []),
            quality_suffix=list(merged.get("quality_suffix") or []),
            llm_hint=None,
            supports_negative_prompt=bool(
                merged.get("supports_negative_prompt", True)
            ),
            supports_weighting=bool(
                merged.get("supports_weighting", True)
            ),
            notes=None,
            negative_embeddings=[],
            negative_axes={
                axis: list(tokens)
                for axis, tokens in (
                    merged.get("negative_axes") or {}
                ).items()
            },
        )


# ── helpers ────────────────────────────────────────────────────────
def _parse_resolution(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ModelRegistryError(f"expected [w, h] pair, got {value!r}")


@functools.lru_cache(maxsize=4)
def _load_models(
    models_dir: Path,
) -> tuple[dict[str, ModelRegistryEntry], dict[str, dict[str, Any]]]:
    if not models_dir.is_dir():
        raise ModelRegistryError(f"models dir not found at {models_dir}")
    paths = sorted(glob.glob(str(models_dir / "*.yaml")))
    if not paths:
        raise ModelRegistryError(f"no *.yaml files under {models_dir}")

    entries: dict[str, ModelRegistryEntry] = {}
    raws: dict[str, dict[str, Any]] = {}
    for p in paths:
        path = Path(p)
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        if "registry" in d or "prompt_guide" in d:
            raise ModelRegistryError(
                f"{path}: legacy `registry:`/`prompt_guide:` wrapper "
                f"detected; flatten to the new schema."
            )
        entry = ModelRegistryEntry.from_dict(d, path)
        if entry.id in entries:
            raise ModelRegistryError(
                f"duplicate model id {entry.id!r} in {path} "
                f"(also in {entries[entry.id].filename})"
            )
        entries[entry.id] = entry
        raws[entry.id] = d
    return entries, raws
