"""Mode 1 — Character-based pipeline.

ARCHITECTURE.md Section 7:
  - IPAdapter ON (when model supports it)
  - Cross-level scene dedup ON
  - Character selected by LRU rotation

The character mode:
  1. ``plan(ctx)`` — picks the LRU character (or uses ``ctx.character``
     if pre-set), then calls ``SeriesPlanner`` with the character's
     identity and the content-level rules.
  2. ``generate_scenes(series_plan, ctx)`` — calls ``SceneGenerator``
     with the plan, content level, and allowed pose types.

IPAdapter usage is determined by ``ctx.model_config.supports_ipadapter``
AND the character having a ``reference_image_path`` — both must be true.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from typing import TYPE_CHECKING

from src.agents.llm_client import OllamaClient
from src.agents.scene_generator import SceneGenerator
from src.agents.series_planner import SeriesPlanner
from src.core.generation_context import GenerationContext
from src.memory.character_manager import CharacterManager
from src.modes.base_mode import BaseMode

if TYPE_CHECKING:
    from src.agents.llm_router import LLMRouter

logger = logging.getLogger(__name__)


class CharacterMode(BaseMode):
    """Mode 1: character-based image sets with IPAdapter identity lock."""

    def __init__(
        self,
        llm_client: OllamaClient,
        router: "LLMRouter | None" = None,
    ) -> None:
        super().__init__(llm_client, router)
        self._planner = SeriesPlanner(llm_client)
        self._scene_gen = SceneGenerator(llm_client)

    @property
    def name(self) -> str:
        return "character"

    def plan(
        self,
        ctx: GenerationContext,
        *,
        cli_llm_override: str | None = None,
    ) -> dict[str, Any]:
        """Select character → generate series plan.

        If ``ctx.character`` is already set (e.g. by ``--character`` CLI
        flag), uses that. Otherwise picks the least-recently-used active
        character via ``CharacterManager``.

        Sets ``ctx.character`` and ``ctx.character_id`` as a side effect
        so downstream code (scene generation, prompt building, IPAdapter
        staging) can read them.
        """
        # Resolve character
        if ctx.character is None:
            cm = CharacterManager(ctx.db_path)
            character = cm.get_least_recently_used()
            ctx.character = character
            ctx.character_id = character["id"]
            logger.info("CharacterMode: LRU selected character %r", character["id"])
        else:
            logger.info("CharacterMode: using pre-set character %r", ctx.character_id)

        character = ctx.character

        # Load recent themes to avoid repetition
        previous_themes = self._load_recent_themes(ctx.db_path, ctx.character_id)

        # Resolve which LLM the SeriesPlanner should run against —
        # router-resolved when override is None, override-pinned otherwise.
        planner_model = self._resolve_role_model(
            "series_planner", cli_llm_override=cli_llm_override,
        )

        # Build the series plan
        plan = self._planner.plan(
            character_name=ctx.character_id,
            base_prompt=character["base_prompt"],
            vibe=character.get("vibe", ""),
            style_name=ctx.style_profile["name"],
            style_keywords=ctx.style_profile["base_style_keywords"],
            content_level=ctx.content_level,
            content_rules={
                "allowed_pose_types": ctx.content_rules.raw_allowed_pose_types,
                "scene_constraints": ctx.content_rules.raw_scene_constraints,
            },
            previous_themes=previous_themes,
            prompt_guide=ctx.model_prompt_guide,
            temperature=ctx.family.llm_temperature,
            model=planner_model,
        )

        # Attach character metadata to the plan for downstream consumers
        plan["character_id"] = ctx.character_id
        plan["character_name"] = ctx.character_id

        logger.info(
            "CharacterMode: plan generated — theme=%r, mood=%r, %d axes",
            plan["theme"],
            plan["mood"],
            len(plan.get("variation_axes", [])),
        )
        return plan

    def generate_scenes(
        self,
        series_plan: dict[str, Any],
        ctx: GenerationContext,
        *,
        cli_llm_override: str | None = None,
    ) -> list[dict[str, Any]]:
        """Generate scenes from the series plan."""
        # SceneGenerator now produces only the model-agnostic scene
        # core. Family-shaped fields (booru_tags / scene_prose /
        # camera_spec / clothing / source_tag) are produced by
        # SceneFacetGenerator per (scene, family) — see Phase A run
        # plan. ``ctx.family.llm_temperature`` is passed through so
        # families with strong temp preferences (Pony 0.5, Flux.2 ~0.8)
        # still get them.
        scene_gen_model = self._resolve_role_model(
            "scene_generator", cli_llm_override=cli_llm_override,
        )
        scenes = self._scene_gen.generate(
            series_plan=series_plan,
            content_level=ctx.content_level,
            allowed_pose_types=ctx.content_rules.allowed_pose_types,
            scene_count=25,
            temperature=ctx.family.llm_temperature,
            model=scene_gen_model,
        )
        logger.info("CharacterMode: %d scenes generated", len(scenes))
        return scenes

    def use_ipadapter(self, ctx: GenerationContext) -> bool:
        """Whether this run should use IPAdapter.

        True only when:
          1. The model supports IPAdapter
          2. The character has a reference image path set
        """
        if not ctx.supports_ipadapter:
            logger.info(
                "CharacterMode: IPAdapter OFF — model %r does not support it",
                ctx.model_id,
            )
            return False
        if not ctx.character:
            return False
        ref = ctx.character.get("reference_image_path")
        if not ref:
            logger.info(
                "CharacterMode: IPAdapter OFF — character %r has no reference_image_path",
                ctx.character_id,
            )
            return False
        return True

    def get_reference_image_path(self, ctx: GenerationContext) -> str | None:
        """Return the character's reference image path, or None."""
        if ctx.character:
            return ctx.character.get("reference_image_path")
        return None

    @staticmethod
    def _load_recent_themes(db_path: Path, character_id: str) -> list[str]:
        """Load the last 5 series themes for this character."""
        conn = sqlite3.connect(str(db_path))
        try:
            rows = conn.execute(
                """
                SELECT theme FROM series
                WHERE character_id = ? AND status != 'aborted'
                ORDER BY created_at DESC
                LIMIT 5
                """,
                (character_id,),
            ).fetchall()
            return [r[0] for r in rows if r[0]]
        finally:
            conn.close()
