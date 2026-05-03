"""Mode 5 — Variation-based pipeline.

ARCHITECTURE.md Section 11:
  - DETERMINISTIC scene generation — minimal/no LLM needed for scenes
  - Source scene is a high-scoring scene from past runs
  - Level-specific pose and expression templates
  - Axis weights: pose 40%, camera 30%, lighting 20%, expression 10%
  - IPAdapter OFF unless the base scene came from character mode

The variation mode:
  1. ``plan(ctx)`` — selects a high-scoring scene from past runs
     (filtered by content_level, avg quality_score > 0.7). If no good
     past scene exists, the LLM generates one strong base scene as
     fallback.
  2. ``generate_scenes(series_plan, ctx)`` — DETERMINISTIC, no LLM.
     Uses ``VARIATION_TEMPLATES`` with level-specific pose and expression
     options, generating variations by iterating through all combinations
     weighted by ``AXIS_WEIGHTS``.
"""

from __future__ import annotations

import logging
import random
import sqlite3
from itertools import product
from pathlib import Path
from typing import Any, TYPE_CHECKING

from src.agents.llm_client import OllamaClient, OllamaJSONParseError
from src.core.generation_context import GenerationContext
from src.modes.base_mode import BaseMode

if TYPE_CHECKING:
    from src.agents.llm_router import LLMRouter

logger = logging.getLogger(__name__)

# Per ARCHITECTURE.md Section 11 — tier-specific templates.
# T1/T2 share the "restrained" vocabulary; T3/T4 share the
# "expressive" vocabulary. Per-tier divergence can be revisited
# once we observe engagement.
_POSES_RESTRAINED = [
    "standing relaxed", "sitting elegantly", "looking over shoulder",
    "hand on hip", "leaning gently", "profile pose",
]
_POSES_EXPRESSIVE = [
    "standing confident", "reclining expressive", "arched back",
    "kneeling", "dynamic pose", "lounging",
]
_EXPRESSIONS_RESTRAINED = ["soft smile", "calm gaze", "dreamy", "gentle"]
_EXPRESSIONS_EXPRESSIVE = [
    "intense gaze", "confident", "bold expression", "provocative smile",
]

VARIATION_TEMPLATES: dict[str, Any] = {
    "pose": {
        "T1_suggestive": _POSES_RESTRAINED,
        "T2_implied":    _POSES_RESTRAINED,
        "T3_artnude":    _POSES_EXPRESSIVE,
        "T4_explicit":   _POSES_EXPRESSIVE,
    },
    "camera": [
        "close-up", "medium shot", "three-quarter",
        "full body", "upper body", "side profile",
    ],
    "lighting": [
        "soft diffused", "dramatic side light", "backlit",
        "natural window", "warm golden hour", "cool blue",
    ],
    "expression": {
        "T1_suggestive": _EXPRESSIONS_RESTRAINED,
        "T2_implied":    _EXPRESSIONS_RESTRAINED,
        "T3_artnude":    _EXPRESSIONS_EXPRESSIVE,
        "T4_explicit":   _EXPRESSIONS_EXPRESSIVE,
    },
}

AXIS_WEIGHTS: dict[str, float] = {
    "pose": 0.40,
    "camera": 0.30,
    "lighting": 0.20,
    "expression": 0.10,
}

_FALLBACK_SYSTEM_PROMPT = """\
You are a professional scene director for an adult photography studio.
You create a single strong, well-composed base scene description.

Your output is ALWAYS a single JSON object with NO extra text, NO markdown fences, NO commentary.
"""

_FALLBACK_USER_TEMPLATE = """\
Generate a single strong base scene for an image set.

Content level: {content_level}
Allowed pose types: {allowed_pose_types}
Style keywords: {style_keywords}

Generate a JSON object with exactly these fields:
{{
  "pose": "<specific pose from the allowed types>",
  "camera": "<shot type>",
  "camera_angle": "<angle>",
  "lighting": "<specific lighting>",
  "environment_detail": "<specific environment>",
  "mood_note": "<emotional tone>",
  "expression": "<facial expression>"
}}

This scene should be visually strong and work well as a base for variations.

Return ONLY the JSON object."""


class VariationModeError(Exception):
    """Error in variation mode."""


class VariationMode(BaseMode):
    """Mode 5: deterministic variations of high-scoring past scenes."""

    def __init__(
        self,
        llm_client: OllamaClient,
        router: "LLMRouter | None" = None,
    ) -> None:
        super().__init__(llm_client, router)

    @property
    def name(self) -> str:
        return "variation"

    def plan(
        self,
        ctx: GenerationContext,
        *,
        cli_llm_override: str | None = None,
    ) -> dict[str, Any]:
        """Select a high-scoring base scene or generate one via LLM fallback."""
        base_scene = self._select_base_scene(ctx.db_path, ctx.content_level)

        if base_scene is not None:
            source = "db"
            logger.info(
                "VariationMode: using past scene %r (avg score %.3f)",
                base_scene.get("id", "?"), base_scene.get("avg_score", 0),
            )
        else:
            source = "llm_fallback"
            logger.info("VariationMode: no qualifying past scene, using LLM fallback")
            base_scene = self._fallback_base_scene(
                ctx, cli_llm_override=cli_llm_override,
            )

        # Check if the base scene came from character mode
        character_id = base_scene.get("character_id")
        source_mode = base_scene.get("source_mode", "")

        plan = {
            "theme": f"Variations on: {base_scene.get('environment_detail', 'studio scene')}",
            "mood": base_scene.get("mood_note", "varied"),
            "environment": base_scene.get("environment_detail", "studio"),
            "variation_axes": list(AXIS_WEIGHTS.keys()),
            "base_scene": base_scene,
            "source": source,
            "source_mode": source_mode,
            "character_id": character_id,
        }

        logger.info(
            "VariationMode: plan — source=%s, base_env=%r, character=%s",
            source, base_scene.get("environment_detail", ""), character_id,
        )
        return plan

    def generate_scenes(
        self,
        series_plan: dict[str, Any],
        ctx: GenerationContext,
        *,
        cli_llm_override: str | None = None,
    ) -> list[dict[str, Any]]:
        """DETERMINISTIC scene generation — no LLM needed.

        Generates variations by sampling from VARIATION_TEMPLATES with
        level-specific options, weighted by AXIS_WEIGHTS.

        The RNG is seeded from the base scene's id + environment so
        re-running variation mode on the same base scene produces the
        same variations. That matches the determinism guarantee the
        rest of the pipeline (RatioSelector scene-text seeding) relies
        on — two identical inputs, two identical runs.
        """
        base = series_plan["base_scene"]
        level = ctx.content_level

        # Seed a local RNG from stable base-scene fields. Falls back to
        # the theme + level if the base scene has neither id nor env —
        # we never want to touch the module-global `random` singleton.
        seed_source = "|".join(str(v) for v in (
            base.get("id"),
            base.get("environment_detail"),
            series_plan.get("theme"),
            level,
        ) if v)
        rng = random.Random(seed_source or "variation_default")

        # Get tier-specific options; fall back to T2_implied for unknown tiers
        poses = VARIATION_TEMPLATES["pose"].get(
            level, VARIATION_TEMPLATES["pose"]["T2_implied"]
        )
        cameras = VARIATION_TEMPLATES["camera"]
        lightings = VARIATION_TEMPLATES["lighting"]
        expressions = VARIATION_TEMPLATES["expression"].get(
            level, VARIATION_TEMPLATES["expression"]["T2_implied"]
        )

        # Generate variations by weighted axis sampling
        scenes: list[dict[str, Any]] = []
        target_count = 25

        # Build weighted axis list for sampling which axis to vary
        axes = list(AXIS_WEIGHTS.keys())
        weights = [AXIS_WEIGHTS[a] for a in axes]

        for i in range(target_count):
            # Pick which axis to vary most for this scene
            primary_axis = rng.choices(axes, weights=weights, k=1)[0]

            # Start from base scene values, then vary the primary axis
            scene: dict[str, Any] = {
                "variation_axis": primary_axis,
                "pose": base.get("pose", "standing relaxed"),
                "camera": base.get("camera", "medium shot"),
                "camera_angle": base.get("camera_angle", "eye level"),
                "lighting": base.get("lighting", "soft diffused"),
                "environment_detail": base.get("environment_detail", "studio"),
                "mood_note": base.get("mood_note", "neutral"),
                "expression": base.get("expression", "soft smile"),
            }

            # Always vary the primary axis
            if primary_axis == "pose":
                scene["pose"] = rng.choice(poses)
            elif primary_axis == "camera":
                scene["camera"] = rng.choice(cameras)
            elif primary_axis == "lighting":
                scene["lighting"] = rng.choice(lightings)
            elif primary_axis == "expression":
                scene["expression"] = rng.choice(expressions)

            # Also vary 1-2 secondary axes for diversity
            secondary = [a for a in axes if a != primary_axis]
            num_secondary = rng.choice([1, 2])
            for sec_axis in rng.sample(secondary, min(num_secondary, len(secondary))):
                if sec_axis == "pose":
                    scene["pose"] = rng.choice(poses)
                elif sec_axis == "camera":
                    scene["camera"] = rng.choice(cameras)
                elif sec_axis == "lighting":
                    scene["lighting"] = rng.choice(lightings)
                elif sec_axis == "expression":
                    scene["expression"] = rng.choice(expressions)

            scenes.append(scene)

        logger.info("VariationMode: %d deterministic scenes generated", len(scenes))
        return scenes

    def use_ipadapter(self, ctx: GenerationContext) -> bool:
        """IPAdapter only if the base scene came from character mode."""
        # This is checked by the engine after plan() sets the plan
        return False  # engine checks plan["source_mode"] directly

    # ---- internal helpers ------------------------------------------------

    @staticmethod
    def _select_base_scene(
        db_path: Path, content_level: str,
    ) -> dict[str, Any] | None:
        """Find a high-scoring past scene (avg quality > 0.7)."""
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                """
                SELECT
                    sc.*,
                    s.mode as source_mode,
                    s.character_id,
                    AVG(i.quality_score) as avg_score
                FROM scenes sc
                JOIN series s ON sc.series_id = s.id
                JOIN prompts p ON p.scene_id = sc.id
                JOIN images i ON i.prompt_id = p.id
                WHERE sc.content_level = ?
                  AND s.status IN ('complete', 'dry_run')
                  AND i.quality_score IS NOT NULL
                GROUP BY sc.id
                HAVING AVG(i.quality_score) > 0.7
                ORDER BY avg_score DESC
                LIMIT 1
                """,
                (content_level,),
            ).fetchone()

            if row:
                return dict(row)
            return None
        finally:
            conn.close()

    def _fallback_base_scene(
        self,
        ctx: GenerationContext,
        *,
        cli_llm_override: str | None = None,
    ) -> dict[str, Any]:
        """LLM generates one strong base scene when no past scene qualifies."""
        import json

        allowed_poses = ctx.content_rules.allowed_pose_types
        user_prompt = _FALLBACK_USER_TEMPLATE.format(
            content_level=ctx.content_level,
            allowed_pose_types=json.dumps(allowed_poses),
            style_keywords=ctx.style_profile["base_style_keywords"],
        )

        fallback_sys = ctx.augment_system_prompt(_FALLBACK_SYSTEM_PROMPT)
        fallback_temp = ctx.family.llm_temperature if ctx.family.llm_temperature is not None else 0.6
        # VariationMode is rare-path (only fires when no past scene
        # qualifies); we resolve via the scene_generator role since the
        # fallback's purpose is to fabricate a "scene-like" base.
        fallback_model = self._resolve_role_model(
            "scene_generator", cli_llm_override=cli_llm_override,
        )
        result = self._attempt_fallback(
            user_prompt, fallback_sys, fallback_temp,
            model=fallback_model,
        )
        if result is not None:
            return result

        logger.warning("VariationMode: fallback first attempt failed, retrying")
        retry = user_prompt + (
            "\n\nIMPORTANT: Return ONLY a raw JSON object, no markdown."
        )
        result = self._attempt_fallback(retry, fallback_sys, fallback_temp)
        if result is not None:
            return result

        # Ultimate fallback: use template defaults
        logger.warning("VariationMode: LLM fallback failed, using static defaults")
        level = ctx.content_level
        return {
            "pose": VARIATION_TEMPLATES["pose"][level][0],
            "camera": VARIATION_TEMPLATES["camera"][0],
            "camera_angle": "eye level",
            "lighting": VARIATION_TEMPLATES["lighting"][0],
            "environment_detail": "professional photo studio with neutral backdrop",
            "mood_note": "calm and focused",
            "expression": VARIATION_TEMPLATES["expression"][level][0],
            "source_mode": "fallback",
        }

    def _attempt_fallback(
        self,
        user_prompt: str,
        system_prompt: str = _FALLBACK_SYSTEM_PROMPT,
        temperature: float = 0.6,
        *,
        model: str | None = None,
    ) -> dict[str, Any] | None:
        try:
            result = self.llm.generate_json(
                system_prompt, user_prompt,
                temperature=temperature, num_predict=1024,
                model=model,
            )
        except OllamaJSONParseError as exc:
            logger.warning("VariationMode fallback parse error: %s", exc)
            return None

        if not isinstance(result, dict):
            return None

        required = {"pose", "camera", "lighting", "environment_detail", "mood_note"}
        if required - result.keys():
            return None

        result["source_mode"] = "llm_fallback"
        return result
