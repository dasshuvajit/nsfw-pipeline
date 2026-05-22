"""Mode 3 — Style-based pipeline.

ARCHITECTURE.md Section 9:
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
from src.modes._llm_helpers import (
    repair_colon_suffix_aesthetic_keys,
    run_llm_with_retry,
    validate_scene_list,
    validate_aesthetic_anchors_in_vocab,
    warn_if_missing_aesthetic_anchors,
    widen_compat_intersection,
)
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

SOLO SUBJECT INVARIANT: every style is designed for a single adult
female subject. Never design styles that depend on multi-subject
compositions ("dual portrait", "couples editorial", "group dynamic
shoot"). The pipeline produces single-female content only.
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

Series aesthetic menu (Phase 3, vocab v6) — pick coherent combo:
  color_palette options: {color_palette_options}
  photographer_ref options: {photographer_ref_options}
  art_movement options: {art_movement_options}

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
  "style_keywords": "<comma-separated keywords that define this visual style>",
  "color_palette": "<one PALETTE_* tag from the menu — series-level cinematic grade>",
  "photographer_ref": "<one PHOTOG_* tag from the menu — photographer-signature>",
  "art_movement": "<one ART_MOVE_* tag from the menu, OR null>"
}}

The style must be SPECIFIC and visual (not vague like "beautiful lighting").
Style keywords should be concrete terms a diffusion model understands.

═══ AESTHETIC COHERENCE RULE ═══════════════════════════════════════
The (color_palette, photographer_ref, art_movement) triple MUST
form a coherent visual world. See examples in the planner system prompt.
NEVER mix incompatible worlds.

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

SOLO SUBJECT INVARIANT: each scene depicts exactly ONE adult woman
as the sole human subject. Different scenes feature different women,
but within any single scene the composition is always single-female.
NEVER write "her partner", "with him", "they", "two women",
"couple", "another person" in any scene's pose / mood / subject_detail.
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
        # Phase 3 (vocab v6) — aesthetic-anchor menu narrowed by
        # style_profile + style-category compatibility lists.
        from src.prompt.aesthetic_menu import _resolve_aesthetic_menu
        style_profile_compat = {
            "compatible_palettes": ctx.style_profile.get(
                "compatible_palettes", []
            ),
            "compatible_photographers": ctx.style_profile.get(
                "compatible_photographers", []
            ),
            "compatible_art_movements": ctx.style_profile.get(
                "compatible_art_movements", []
            ),
        }
        for key in (
            "compatible_palettes",
            "compatible_photographers",
            "compatible_art_movements",
        ):
            cat_list = category.get(key, [])
            if cat_list:
                prof_list = style_profile_compat.get(key, []) or []
                if prof_list:
                    style_profile_compat[key] = [
                        t for t in prof_list if t in cat_list
                    ]
                else:
                    style_profile_compat[key] = list(cat_list)
        aesthetic_menu = _resolve_aesthetic_menu(
            style_profile_compat=style_profile_compat,
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
            color_palette_options=", ".join(aesthetic_menu["color_palette"]),
            photographer_ref_options=", ".join(aesthetic_menu["photographer_ref"]),
            art_movement_options=", ".join(aesthetic_menu["art_movement"]),
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
        # Verifier round-2 I4 — propagate the style category's environment
        # whitelist (intersected with the style_profile's when both
        # populated) so SceneFacetGenerator narrows the LLM's
        # environment.setting menu in engine.py. Empty list / missing
        # falls through as the full vocab menu.
        # Round-15 (2026-05-21) — widen narrow intersections (< 3
        # items) to the category list. See widen_compat_intersection
        # docstring for the LM Studio Cydonia hallucination case
        # this fix was driven by.
        plan["compatible_environments"] = widen_compat_intersection(
            category.get("compatible_environments"),
            ctx.style_profile.get("compatible_environments"),
        )

        # Round-12 (2026-05-21) — narrative menu narrowing.
        plan["compatible_narratives"] = list(
            category.get("compatible_narratives", []) or []
        )

        # 2026-05-23 — art_style narrowing per style_profile.
        plan["compatible_art_styles"] = list(
            ctx.style_profile.get("compatible_art_styles", []) or []
        )

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
        scene_count = 25
        user_prompt = _SCENE_USER_TEMPLATE.format(
            scene_count=scene_count,
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
            model=scene_gen_model, scene_count=scene_count,
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
        # Round-5 defensive: salvage colon-suffix aesthetic keys first.
        repair_colon_suffix_aesthetic_keys(result)
        # Round-22 F14 — null any anchor not in vocab (planner
        # hallucinated past the regex prefix check).
        validate_aesthetic_anchors_in_vocab(result, mode_name="StyleMode")
        # Verifier round-4 IMPORTANT-5 — soft Phase 3 anchor check.
        warn_if_missing_aesthetic_anchors(result, mode_name="StyleMode")
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
            num_predict=8192,  # round-19: reasoning models eat 4k+ before emitting JSON
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
        scene_count: int = 25,
    ) -> list[dict[str, Any]]:
        # Round-17 — see ThemeMode._generate_scenes for context. 70%
        # ship-rate floor catches under-shipping LLMs.
        min_count = max(1, int(scene_count * 0.7))
        return run_llm_with_retry(
            self.llm,
            system=system_prompt,
            user=user_prompt,
            validator=validate_scene_list(
                set(self._SCENE_REQUIRED), min_count=min_count,
            ),
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
            # Phase 3 (vocab v6) — verifier B1 pipe fix.
            "compatible_palettes": list(chosen.compatible_palettes),
            "compatible_photographers": list(chosen.compatible_photographers),
            "compatible_art_movements": list(chosen.compatible_art_movements),
            "compatible_environments": list(chosen.compatible_environments),
            "compatible_narratives": list(chosen.compatible_narratives),
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
