"""Mode 3 — Style-based pipeline.

ARCHITECTURE.md Section 9:
  - IPAdapter OFF
  - Category from ``style_categories`` table (weighted random)
  - Content level influence is lighter (style keywords dominate prompt)
  - LLM generates *subjects* (not scenes) — each subject is a different
    person unified by the style

The style mode:
  1. ``plan(ctx)`` — selects a style category via weighted random,
     then calls the LLM to generate a full style profile
     (lighting_profile, color_profile, camera_bias, composition_rules,
     environment_bias, style_keywords).
  2. ``generate_scenes(series_plan, ctx)`` — LLM generates subjects
     (different people) unified by the style. Each "scene" is really a
     subject + environment + pose, all visually cohesive.
"""

from __future__ import annotations

import json
import logging
import random
import sqlite3
from pathlib import Path
from typing import Any

from typing import TYPE_CHECKING

from src.agents.llm_client import OllamaClient
from src.agents.schemas import SceneList, SeriesPlan
from src.core.generation_context import GenerationContext
from src.memory.categories_loader import CategoriesLoader
from src.modes._llm_helpers import run_llm_with_retry, validate_scene_list
from src.modes.base_mode import BaseMode

if TYPE_CHECKING:
    from src.agents.llm_router import LLMRouter

logger = logging.getLogger(__name__)


_PLAN_SYSTEM_PROMPT = """\
You are a visual style designer for a professional adult photography studio.
You design cohesive visual styles that unify diverse subjects in an image set.

Your output is ALWAYS a single JSON object with NO extra text, NO markdown fences, NO commentary.

You must respect the content tier directive provided — never exceed
it, never SOFTEN it. The style's mood / camera-bias / environment-bias
must be compatible with the tier directive at all times.
"""

_PLAN_USER_TEMPLATE = """\
Design a cohesive visual style for an image set based on this style category.

Style category: {category_name}
  Description: {category_description}
  Lighting bias: {lighting_bias}
  Color bias: {color_bias}

Base style profile: {style_name}
  Keywords: {style_keywords}

══════════ CONTENT TIER ({content_level}) — READ CAREFULLY ══════════
{llm_directive}
═════════════════════════════════════════════════════════════════════

Previous styles to AVOID repeating (last 5 series in style mode):
{previous_styles}

Generate a JSON object with exactly these fields:
{{
  "theme": "<the style concept as a theme name — e.g. 'Film Noir Glamour' not just 'cinematic'>",
  "mood": "<emotional tone the style creates>",
  "environment": "<primary environment bias — specific>",
  "variation_axes": ["<axis1>", "<axis2>", "<axis3>"],
  "lighting_profile": "<specific lighting setup description>",
  "color_profile": "<color palette and grading description>",
  "camera_bias": "<preferred camera angles and framing>",
  "composition_rules": "<rules for visual composition in this style>",
  "environment_bias": "<what kinds of environments work with this style>",
  "style_keywords": "<comma-separated keywords that define this visual style>"
}}

The style must be SPECIFIC and visual (not vague like "beautiful lighting").
Style keywords should be concrete terms a diffusion model understands.

Return ONLY the JSON object."""

_SCENE_SYSTEM_PROMPT = """\
You are a professional casting and scene director for an adult photography studio.
You create diverse subjects (different people) unified by a specific visual style.

Your output is ALWAYS a JSON array of scene objects with NO extra text, NO markdown fences, NO commentary.

You must respect the content tier directive and allowed pose types
provided — never exceed them, never SOFTEN them. At T3_artnude and
T4_explicit, every subject's pose / camera / mood_note / subject_detail
must be compatible with the directive's default state of undress.
Each subject should be a different person — vary appearance, but keep
the visual style consistent.
"""

_SCENE_USER_TEMPLATE = """\
Generate {scene_count} unique subjects for this style-focused image set.
Each entry represents a DIFFERENT person, unified by the style.

Style plan:
  Theme: {theme}
  Mood: {mood}
  Environment: {environment}
  Lighting profile: {lighting_profile}
  Color profile: {color_profile}
  Camera bias: {camera_bias}
  Style keywords: {style_keywords_plan}
  Variation axes: {variation_axes}

══════════ CONTENT TIER ({content_level}) — READ CAREFULLY ══════════
{llm_directive}
═════════════════════════════════════════════════════════════════════

Allowed pose types: {allowed_pose_types}

Each scene must be a JSON object with exactly these fields:
{{
  "variation_axis": "<which variation axis this subject explores>",
  "pose": "<specific pose — MUST be one of the allowed pose types>",
  "camera": "<shot type consistent with the camera bias>",
  "camera_angle": "<angle consistent with the style>",
  "lighting": "<specific lighting — must feel like the lighting profile>",
  "environment_detail": "<specific environment consistent with the style>",
  "mood_note": "<emotional note — must match the style's mood>",
  "expression": "<facial expression fitting the style>",
  "subject_detail": "<appearance description of THIS specific person — vary across subjects>"
}}

Rules:
- Each subject should look like a DIFFERENT person (vary age, hair, features)
- The STYLE (lighting, color, composition) must stay consistent across all subjects
- Vary poses and camera angles within the style's constraints

Return ONLY the JSON array of {scene_count} scene objects."""


class StyleModeError(Exception):
    """Error in style mode planning."""


class StyleMode(BaseMode):
    """Mode 3: style-based image sets — diverse subjects, unified style."""

    PLAN_TEMPERATURE = 0.7
    SCENE_TEMPERATURE = 0.7

    def __init__(
        self,
        llm_client: OllamaClient,
        router: "LLMRouter | None" = None,
    ) -> None:
        super().__init__(llm_client, router)

    @property
    def name(self) -> str:
        return "style"

    def plan(
        self,
        ctx: GenerationContext,
        *,
        cli_llm_override: str | None = None,
    ) -> dict[str, Any]:
        """Select style category -> LLM generates a full style profile."""
        category = self._select_category(ctx.db_path)
        logger.info("StyleMode: selected category %r (%s)", category["id"], category["name"])

        previous_styles = self._load_recent_styles(ctx.db_path)

        tier_directive = (
            getattr(ctx.content_rules, "llm_directive", "")
            or f"(No directive declared for {ctx.content_level}.)"
        )
        user_prompt = _PLAN_USER_TEMPLATE.format(
            category_name=category["name"],
            category_description=category.get("description", ""),
            lighting_bias=category.get("lighting_bias", ""),
            color_bias=category.get("color_bias", ""),
            style_name=ctx.style_profile["name"],
            style_keywords=ctx.style_profile["base_style_keywords"],
            content_level=ctx.content_level,
            llm_directive=tier_directive,
            previous_styles="\n".join(f"  - {s}" for s in previous_styles) or "  (none)",
        )

        plan_sys = ctx.augment_system_prompt(_PLAN_SYSTEM_PROMPT)
        planner_model = self._resolve_role_model(
            "series_planner", cli_llm_override=cli_llm_override,
        )
        plan = self._generate_plan(
            user_prompt, plan_sys, ctx.family.llm_temperature,
            model=planner_model,
        )

        plan["category_id"] = category["id"]
        plan["category_name"] = category["name"]

        logger.info(
            "StyleMode: plan — theme=%r, lighting=%r, color=%r",
            plan["theme"],
            plan.get("lighting_profile", ""),
            plan.get("color_profile", ""),
        )
        return plan

    def generate_scenes(
        self,
        series_plan: dict[str, Any],
        ctx: GenerationContext,
        *,
        cli_llm_override: str | None = None,
    ) -> list[dict[str, Any]]:
        """Generate subjects (not scenes) — each is a different person."""
        allowed_poses = ctx.content_rules.allowed_pose_types

        tier_directive = (
            getattr(ctx.content_rules, "llm_directive", "")
            or f"(No directive declared for {ctx.content_level}.)"
        )
        user_prompt = _SCENE_USER_TEMPLATE.format(
            scene_count=25,
            theme=series_plan["theme"],
            mood=series_plan["mood"],
            environment=series_plan["environment"],
            lighting_profile=series_plan.get("lighting_profile", ""),
            color_profile=series_plan.get("color_profile", ""),
            camera_bias=series_plan.get("camera_bias", ""),
            style_keywords_plan=series_plan.get("style_keywords", ""),
            variation_axes=json.dumps(series_plan.get("variation_axes", [])),
            content_level=ctx.content_level,
            llm_directive=tier_directive,
            allowed_pose_types=json.dumps(allowed_poses),
        )

        scene_sys = ctx.augment_system_prompt(_SCENE_SYSTEM_PROMPT)
        scene_gen_model = self._resolve_role_model(
            "scene_generator", cli_llm_override=cli_llm_override,
        )
        scenes = self._generate_scenes(
            user_prompt, scene_sys, ctx.family.llm_temperature,
            model=scene_gen_model,
        )
        logger.info("StyleMode: %d scenes generated", len(scenes))
        return scenes

    # ---- internal helpers ------------------------------------------------

    _VAGUE_STYLES: frozenset[str] = frozenset(
        {"beautiful lighting", "nice style", "good photography", "aesthetic"}
    )
    _PLAN_REQUIRED: frozenset[str] = frozenset(
        {"theme", "mood", "environment", "variation_axes"}
    )
    _SCENE_REQUIRED: frozenset[str] = frozenset(
        {"variation_axis", "pose", "camera", "camera_angle",
         "lighting", "environment_detail", "mood_note"}
    )

    def _validate_plan(self, result: Any) -> dict[str, Any] | None:
        if not isinstance(result, dict):
            return None
        if self._PLAN_REQUIRED - result.keys():
            return None
        theme = (result.get("theme") or "").lower().strip()
        if theme in self._VAGUE_STYLES:
            logger.warning("StyleMode: rejected vague style %r", result["theme"])
            return None
        return result

    def _generate_plan(
        self,
        user_prompt: str,
        system_prompt: str = _PLAN_SYSTEM_PROMPT,
        temperature: float | None = None,
        *,
        model: str | None = None,
    ) -> dict[str, Any]:
        return run_llm_with_retry(
            self.llm,
            system=system_prompt,
            user=user_prompt,
            validator=self._validate_plan,
            temperature=temperature if temperature is not None else self.PLAN_TEMPERATURE,
            num_predict=2048,
            mode_name="StyleMode plan",
            error_factory=StyleModeError,
            model=model,
            schema=SeriesPlan,
        )

    def _generate_scenes(
        self,
        user_prompt: str,
        system_prompt: str = _SCENE_SYSTEM_PROMPT,
        temperature: float | None = None,
        *,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        return run_llm_with_retry(
            self.llm,
            system=system_prompt,
            user=user_prompt,
            validator=validate_scene_list(set(self._SCENE_REQUIRED)),
            temperature=temperature if temperature is not None else self.SCENE_TEMPERATURE,
            num_predict=8192,
            mode_name="StyleMode scenes",
            error_factory=StyleModeError,
            model=model,
            schema=SceneList,
        )

    @staticmethod
    def _select_category(db_path: Path) -> dict[str, Any]:
        """Weighted random selection from YAML style categories.

        ``db_path`` is retained for signature compatibility but unused —
        style categories live in ``config/categories.yaml``.
        """
        categories = CategoriesLoader().style_categories()
        if not categories:
            raise StyleModeError("No style categories in config/categories.yaml")
        weights = [c.weight for c in categories]
        chosen = random.choices(categories, weights=weights, k=1)[0]
        return {
            "id": chosen.id,
            "name": chosen.name,
            "weight": chosen.weight,
            "description": chosen.description,
            "lighting_bias": chosen.lighting_bias,
            "color_bias": chosen.color_bias,
        }

    @staticmethod
    def _load_recent_styles(db_path: Path) -> list[str]:
        conn = sqlite3.connect(str(db_path))
        try:
            rows = conn.execute(
                """
                SELECT theme FROM series
                WHERE mode = 'style' AND status != 'aborted'
                ORDER BY created_at DESC
                LIMIT 5
                """,
            ).fetchall()
            return [r[0] for r in rows if r[0]]
        finally:
            conn.close()
