"""Scene-aware aspect ratio selection — ARCHITECTURE.md Section 6 +
multi-axis scoring (Phase A, 2026-04).

The pipeline picks an aspect ratio per scene (not per series, not random)
by combining additive layers in this order:

1. **Per-content-level base weights** from ``pipeline.yaml`` →
   ``aspect_ratio_weights.{T1_suggestive|T2_implied|T3_artnude|T4_explicit}``.
   T1/T2 skew more square-ish for discovery posts; T3/T4 skew more
   portrait for explicit framing.

2. **Audience bonus** — DeviantArt's grid favors classic 2:3 portraits;
   Patreon's mobile feed rewards 9:16 vertical. Pulled from
   ``ratio_signals.yaml::audience_bonus`` keyed on the optional
   ``audience`` arg to ``select()``.

3. **Composition-intent bonus** — the scene_generator emits an optional
   ``composition_intent`` field (close-up / medium / full-body / wide);
   each value biases specific ratios.

4. **Pose signal bump** — keyword-based +0.15 (configurable in
   ``ratio_signals.yaml::pose_signal_bump``) when a scene-text keyword
   matches a ratio's ``signals`` list. One bump per ratio per scene.

5. **Family-quality bonus** — SDXL/Pony/Illustrious dislike 9:16 (UNet
   off-distribution); Flux/Chroma are flat; Flux.2 slightly favors
   portrait. Keyed on the optional ``family_id`` arg.

The summed score per ratio is clamped to ``[clamp.min, clamp.max]``
(default ``[0.05, 2.0]``), zero-or-negative scores are dropped, weights
are normalized, and a deterministic per-scene RNG draws one ratio.

**Hard overrides** survive unchanged: ``pose: "full body"|"standing"``
forces ``portrait_23``; ``pose: "reclining"|"lying"`` forces
``landscape``. They short-circuit the score entirely.

### Backward compatibility

Every existing call site keeps working — every new bonus parameter is
optional. Constructing ``RatioSelector(weights=...)`` with only the
content-level weights and calling ``select(scene, content_level)``
behaves identically to pre-Phase-A (modulo the score clamp, which is
a no-op when all bonuses are absent because the legacy weights are all
in ``[0, 1]``).

Three small deviations from the Section 6 snippet, all noted inline:

A. ``select()`` accepts a ``GenerationContext``-like object **or** a
   plain string ``content_level``, because the first caller
   (``scripts/render_set.py``) doesn't yet build a real ctx.

B. The snippet's ``_check_override`` reads ``scene.get('camera','').lower()
   == 'square'`` to detect the conflict, but ``scene['camera']`` in the
   actual schema is a description ("close-up", "medium shot"), not a
   ratio name. We instead trigger ``portrait_23`` whenever the pose says
   "full body" or "standing".

C. We seed the ``random`` module with the prompt-relevant scene text by
   default so the same scene + content level always picks the same
   ratio. The snippet uses unseeded ``random``; that's fine for live
   runs but useless for any reproducible test. Pass
   ``rng=random.Random()`` to restore the spec's behaviour.
"""

from __future__ import annotations

import hashlib
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.core.aspect_ratio_buckets import get_family_resolution

logger = logging.getLogger(__name__)


# ----- public constants -----------------------------------------------------

RATIO_TO_RESOLUTION: dict[str, tuple[int, int]] = {
    "portrait_23":  (768, 1152),
    "portrait_916": (768, 1366),
    "square":       (1024, 1024),
    "landscape":    (1152, 768),
}

# Default scene→ratio signal map. Loaded from ``config/ratio_signals.yaml``
# when present (preferred) or used inline as a fallback so the selector
# works without the yaml file.
DEFAULT_RATIO_SIGNALS: dict[str, list[str]] = {
    "portrait_23": [
        "full body",
        "standing",
        "kneeling",
        "three-quarter",
        "medium shot",
        "upper body",
    ],
    "square": [
        "close-up",
        "face",
        "portrait",
        "macro",
        "studio",
    ],
    "landscape": [
        "reclining",
        "lying",
        "wide shot",
        "panorama",
        "establishing",
    ],
}

# Default bump per matched signal — overridable via
# ``ratio_signals.yaml::pose_signal_bump``.
_DEFAULT_POSE_SIGNAL_BUMP = 0.15

# Default score clamps — overridable via ``ratio_signals.yaml::clamp``.
_DEFAULT_CLAMP_MIN = 0.05
_DEFAULT_CLAMP_MAX = 2.0


# ----- exceptions -----------------------------------------------------------


class RatioSelectorError(Exception):
    """Bad config or unknown ratio key."""


# ----- public API -----------------------------------------------------------


def model_resolution_overrides(model: Any) -> dict[str, tuple[int, int]]:
    """Build a ratio→resolution dict from a ``ModelRegistryEntry``.

    Reads the ``resolution_portrait``, ``resolution_square``, and
    ``resolution_landscape`` fields (added in Step 1 of the model
    profile system). Returns an empty dict if none are set.
    """
    overrides: dict[str, tuple[int, int]] = {}
    if getattr(model, "resolution_portrait", None):
        overrides["portrait_23"] = tuple(model.resolution_portrait)
        overrides["portrait_916"] = tuple(model.resolution_portrait)
    if getattr(model, "resolution_square", None):
        overrides["square"] = tuple(model.resolution_square)
    if getattr(model, "resolution_landscape", None):
        overrides["landscape"] = tuple(model.resolution_landscape)
    return overrides


def get_resolution(
    ratio: str,
    model_checkpoint: str | None = None,
    *,
    model_resolutions: dict[str, tuple[int, int]] | None = None,
    family_id: str | None = None,
) -> tuple[int, int]:
    """Resolve a ratio key to a concrete ``(width, height)`` tuple.

    Resolution precedence (first match wins):
      1. ``model_resolutions`` — from per-model YAML resolution fields
      2. ``family_id`` family-tier bucket — 1MP/1.5MP/2MP per family
      3. ``RATIO_TO_RESOLUTION`` — base SDXL 1MP defaults
    """
    if ratio not in RATIO_TO_RESOLUTION:
        raise RatioSelectorError(
            f"Unknown ratio key {ratio!r}. Known: {list(RATIO_TO_RESOLUTION)}"
        )
    if model_resolutions and ratio in model_resolutions:
        return model_resolutions[ratio]
    if family_id:
        family_res = get_family_resolution(family_id, ratio)
        if family_res is not None:
            return family_res
    return RATIO_TO_RESOLUTION[ratio]


@dataclass
class RatioPick:
    """A selection result with the bookkeeping callers actually want."""

    ratio: str
    resolution: tuple[int, int]
    reason: str  # "override:full_body", "weighted:T2_implied", etc.


# Composition-intent values the scene_generator can emit. Anything else
# is treated as "no bonus" rather than raising — the LLM is best-effort.
_VALID_COMPOSITION_INTENTS = {"close-up", "medium", "full-body", "wide"}

# Audience values the CLI / scene_generator can pass in. Anything else
# (including "either") returns no bonus, which is the desired default.
_VALID_AUDIENCES = {"deviantart", "patreon"}


class RatioSelector:
    """Picks aspect ratios per scene with multi-axis additive scoring."""

    def __init__(
        self,
        weights: Mapping[str, Mapping[str, float]],
        signals: Mapping[str, list[str]] | None = None,
        hard_overrides_enabled: bool = True,
        *,
        audience_bonus: Mapping[str, Mapping[str, float]] | None = None,
        composition_bonus: Mapping[str, Mapping[str, float]] | None = None,
        family_quality_bonus: Mapping[str, Mapping[str, float]] | None = None,
        pose_signal_bump: float = _DEFAULT_POSE_SIGNAL_BUMP,
        clamp_min: float = _DEFAULT_CLAMP_MIN,
        clamp_max: float = _DEFAULT_CLAMP_MAX,
    ) -> None:
        """Construct from a parsed ``aspect_ratio_weights`` block.

        Args:
            weights: ``{content_level: {ratio_key: weight}}`` — exactly the
                shape of ``pipeline.yaml → aspect_ratio_weights``.
            signals: Optional override of ``DEFAULT_RATIO_SIGNALS``. Pass
                ``{}`` to disable the pose signal bump entirely.
            hard_overrides_enabled: Set to ``False`` in tests that want to
                exercise the weighted-draw path on full-body / reclining
                scenes.
            audience_bonus: ``{audience: {ratio: bonus}}`` — added per
                ratio when ``select(audience=...)`` is given. Default empty.
            composition_bonus: ``{intent: {ratio: bonus}}`` — added per
                ratio based on ``scene["composition_intent"]``.
            family_quality_bonus: ``{family_id: {ratio: bonus}}`` — added
                per ratio when ``select(family_id=...)`` is given.
            pose_signal_bump: Score added when a scene-text keyword
                matches a ratio's ``signals`` list. Default 0.15.
            clamp_min, clamp_max: Score clamps applied after all bonuses
                sum, before the weighted draw.
        """
        if not weights:
            raise RatioSelectorError(
                "RatioSelector requires non-empty aspect_ratio_weights "
                "(see pipeline.yaml)."
            )
        # Defensive copy so callers can't mutate our internals.
        self.weights: dict[str, dict[str, float]] = {
            level: dict(buckets) for level, buckets in weights.items()
        }
        for level, buckets in self.weights.items():
            for ratio in buckets:
                if ratio not in RATIO_TO_RESOLUTION:
                    raise RatioSelectorError(
                        f"aspect_ratio_weights['{level}'] has unknown ratio "
                        f"{ratio!r}. Known: {list(RATIO_TO_RESOLUTION)}"
                    )
        # Use ``is not None`` so an explicit ``signals={}`` disables the
        # pose-signal bump entirely (per the docstring) instead of falling
        # back to defaults via the falsy-`or` shortcut.
        effective_signals = signals if signals is not None else DEFAULT_RATIO_SIGNALS
        self.signals: dict[str, list[str]] = (
            {k: list(v) for k, v in effective_signals.items()}
        )
        self.hard_overrides_enabled = hard_overrides_enabled

        self.audience_bonus: dict[str, dict[str, float]] = _validate_bonus_table(
            audience_bonus, "audience_bonus"
        )
        self.composition_bonus: dict[str, dict[str, float]] = _validate_bonus_table(
            composition_bonus, "composition_bonus"
        )
        self.family_quality_bonus: dict[str, dict[str, float]] = _validate_bonus_table(
            family_quality_bonus, "family_quality_bonus"
        )
        self.pose_signal_bump = float(pose_signal_bump)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        if self.clamp_min < 0 or self.clamp_max <= self.clamp_min:
            raise RatioSelectorError(
                f"Invalid clamp range [{self.clamp_min}, {self.clamp_max}]: "
                f"min must be ≥ 0 and < max."
            )

    @classmethod
    def from_config(
        cls,
        pipeline_cfg: Mapping[str, Any],
        ratio_signals_cfg: Mapping[str, Any] | None = None,
        *,
        ratio_signals_path: Path | str | None = None,
        hard_overrides_enabled: bool = True,
    ) -> "RatioSelector":
        """Build a selector from parsed YAML configs.

        ``pipeline_cfg`` is the loaded ``pipeline.yaml`` (we read its
        ``aspect_ratio_weights`` block). ``ratio_signals_cfg`` is the
        loaded ``ratio_signals.yaml`` — if omitted and
        ``ratio_signals_path`` is given, the file is loaded; if both are
        omitted, the inline ``DEFAULT_RATIO_SIGNALS`` is used and all
        the new bonus tables are empty (legacy behavior).
        """
        weights = pipeline_cfg.get("aspect_ratio_weights")
        if not weights:
            raise RatioSelectorError(
                "pipeline_cfg is missing 'aspect_ratio_weights' (see "
                "pipeline.yaml)."
            )

        if ratio_signals_cfg is None and ratio_signals_path is not None:
            path = Path(ratio_signals_path).expanduser()
            if path.exists():
                ratio_signals_cfg = yaml.safe_load(path.read_text()) or {}
            else:
                logger.warning(
                    "ratio_signals_path %s not found — falling back to "
                    "inline DEFAULT_RATIO_SIGNALS with empty bonus tables.",
                    path,
                )
                ratio_signals_cfg = {}

        cfg = ratio_signals_cfg or {}
        return cls(
            weights=weights,
            signals=cfg.get("signals"),
            hard_overrides_enabled=hard_overrides_enabled,
            audience_bonus=cfg.get("audience_bonus"),
            composition_bonus=cfg.get("composition_bonus"),
            family_quality_bonus=cfg.get("family_quality_bonus"),
            pose_signal_bump=cfg.get("pose_signal_bump", _DEFAULT_POSE_SIGNAL_BUMP),
            clamp_min=(cfg.get("clamp") or {}).get("min", _DEFAULT_CLAMP_MIN),
            clamp_max=(cfg.get("clamp") or {}).get("max", _DEFAULT_CLAMP_MAX),
        )

    # ----- selection --------------------------------------------------------

    def select(
        self,
        scene: Mapping[str, Any],
        content_level: str,
        *,
        model_checkpoint: str | None = None,
        model_resolutions: dict[str, tuple[int, int]] | None = None,
        family_id: str | None = None,
        audience: str | None = None,
        rng: random.Random | None = None,
    ) -> RatioPick:
        """Pick a ratio + resolution for one scene.

        ``content_level`` is one of the 4 tier strings (``T1_suggestive``,
        ``T2_implied``, ``T3_artnude``, ``T4_explicit``). Pass an explicit
        ``rng`` for reproducibility; otherwise we deterministically seed a
        new ``random.Random`` from the scene text + content level +
        audience + family so back-to-back runs against the same scene
        yield the same ratio.

        ``family_id`` (e.g. ``"sdxl"``, ``"flux2"``) drives the
        family-quality bonus AND, via ``get_resolution``, the per-family
        megapixel bucket lookup. ``audience`` ("deviantart" / "patreon")
        drives the audience bonus. Both default to None — same scoring
        as pre-Phase-A.

        ``scene["composition_intent"]`` (optional) drives the
        composition bonus. Valid values: close-up / medium / full-body /
        wide. Anything else is silently ignored.
        """
        if content_level not in self.weights:
            raise RatioSelectorError(
                f"No aspect_ratio_weights for content_level={content_level!r}. "
                f"Known: {list(self.weights)}"
            )

        if self.hard_overrides_enabled:
            forced = self._check_override(scene)
            if forced:
                return RatioPick(
                    ratio=forced,
                    resolution=get_resolution(
                        forced, model_checkpoint,
                        model_resolutions=model_resolutions,
                        family_id=family_id,
                    ),
                    reason=f"override:{forced}",
                )

        scene_text = self._scene_text(scene)
        composition_intent = (scene.get("composition_intent") or "").lower().strip()

        # Start from the content-level base weights.
        scored: dict[str, float] = dict(self.weights[content_level])

        # 1. Pose signal bump — keyword match on scene text.
        for ratio, keywords in self.signals.items():
            for kw in keywords:
                if kw and kw.lower() in scene_text:
                    scored[ratio] = scored.get(ratio, 0.0) + self.pose_signal_bump
                    break  # one bump per ratio bucket per scene

        # 2. Audience bonus.
        if audience and audience.lower() in _VALID_AUDIENCES:
            for ratio, bonus in self.audience_bonus.get(audience.lower(), {}).items():
                scored[ratio] = scored.get(ratio, 0.0) + bonus

        # 3. Composition-intent bonus.
        if composition_intent in _VALID_COMPOSITION_INTENTS:
            for ratio, bonus in self.composition_bonus.get(composition_intent, {}).items():
                scored[ratio] = scored.get(ratio, 0.0) + bonus

        # 4. Family-quality bonus.
        if family_id:
            for ratio, bonus in self.family_quality_bonus.get(family_id, {}).items():
                scored[ratio] = scored.get(ratio, 0.0) + bonus

        # 5. Clamp + drop ≤0. Floor prevents zero-collapse from stacked
        # negatives; cap prevents any single axis from dominating.
        scored = {
            ratio: max(self.clamp_min, min(self.clamp_max, score))
            for ratio, score in scored.items()
            if score > 0
        }
        if not scored:
            raise RatioSelectorError(
                f"All ratio weights collapsed to zero for content_level="
                f"{content_level!r} and scene={dict(scene)!r}"
            )

        # Normalize so the weights sum to 1.0; useful for debugging logs
        # but not strictly required by random.choices().
        total = sum(scored.values())
        normalized = {k: v / total for k, v in scored.items()}

        chosen_rng = rng or self._rng_from_scene(
            scene_text, content_level, audience, family_id
        )
        ratio = chosen_rng.choices(
            population=list(normalized.keys()),
            weights=list(normalized.values()),
            k=1,
        )[0]

        reason_parts = [f"weighted:{content_level}"]
        if audience and audience.lower() in _VALID_AUDIENCES:
            reason_parts.append(f"audience={audience.lower()}")
        if family_id:
            reason_parts.append(f"family={family_id}")
        if composition_intent in _VALID_COMPOSITION_INTENTS:
            reason_parts.append(f"intent={composition_intent}")
        return RatioPick(
            ratio=ratio,
            resolution=get_resolution(
                ratio, model_checkpoint,
                model_resolutions=model_resolutions,
                family_id=family_id,
            ),
            reason="|".join(reason_parts),
        )

    # ----- helpers ----------------------------------------------------------

    @staticmethod
    def _scene_text(scene: Mapping[str, Any]) -> str:
        parts = [
            scene.get("pose"),
            scene.get("camera"),
            scene.get("camera_angle"),
            scene.get("environment_detail"),
        ]
        return " ".join(p for p in parts if p).lower()

    @staticmethod
    def _check_override(scene: Mapping[str, Any]) -> str | None:
        """Hard overrides — see Section 6 + module docstring deviation B."""
        pose = (scene.get("pose") or "").lower()
        if "full body" in pose or "standing" in pose:
            return "portrait_23"
        if "reclining" in pose or "lying" in pose:
            return "landscape"
        return None

    @staticmethod
    def _rng_from_scene(
        scene_text: str,
        content_level: str,
        audience: str | None,
        family_id: str | None,
    ) -> random.Random:
        """Deterministic per-scene RNG seed.

        Same scene text + same content level + same audience + same
        family always yields the same draw. Useful for reproducible
        tests and for replaying a series after a config tweak. Live
        runs that explicitly pass ``rng=random.Random()`` get the
        snippet's original behaviour.
        """
        seed_str = (
            f"{content_level}::{audience or ''}::{family_id or ''}::{scene_text}"
        )
        digest = hashlib.sha1(seed_str.encode()).hexdigest()
        return random.Random(int(digest[:16], 16))


# ----- internal helpers -----------------------------------------------------


def _validate_bonus_table(
    table: Mapping[str, Mapping[str, float]] | None,
    table_name: str,
) -> dict[str, dict[str, float]]:
    """Defensive copy + ratio-key validation for one bonus table."""
    if not table:
        return {}
    out: dict[str, dict[str, float]] = {}
    for key, ratios in table.items():
        if not isinstance(ratios, Mapping):
            raise RatioSelectorError(
                f"{table_name}['{key}'] must be a mapping of ratio→bonus, "
                f"got {type(ratios).__name__}"
            )
        out[key] = {}
        for ratio, bonus in ratios.items():
            if ratio not in RATIO_TO_RESOLUTION:
                raise RatioSelectorError(
                    f"{table_name}['{key}'] has unknown ratio {ratio!r}. "
                    f"Known: {list(RATIO_TO_RESOLUTION)}"
                )
            out[key][ratio] = float(bonus)
    return out
