"""Mode 4 — Niche-based pipeline.

ARCHITECTURE.md Section 10:
  - IPAdapter OFF
  - Cluster from ``niche_clusters`` table (weighted random)
  - ``visual_elements`` (concrete props) in LLM expansion
  - SEO keywords from the cluster's keyword list injected into prompts
  - Content level aware: T1/T2 → discovery tags, T3/T4 → conversion tags

The niche mode:
  1. ``plan(ctx)`` — selects a niche cluster via weighted random,
     then calls the LLM to expand the niche into a plan with core_theme,
     environment, keyword_cluster, visual_elements, subject_bias.
  2. ``generate_scenes(series_plan, ctx)`` — LLM generates scenes
     incorporating the visual_elements and niche keywords.
"""

from __future__ import annotations

import json
import logging
import random
import sqlite3
from pathlib import Path
from typing import Any, TYPE_CHECKING

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
You are a niche market researcher and creative director for a professional adult photography studio.
You plan image sets optimized for specific niche audiences and their search behavior.

Your output is ALWAYS a single JSON object with NO extra text, NO markdown fences, NO commentary.

You must respect the content tier directive provided — never exceed
it, never SOFTEN it. If the tier directive calls for explicit nude
content, the THEME and SUBJECT_BIAS must reflect that — do NOT plan
a T2-style "implied nude / lingerie" set when given T4_explicit.
"""

_PLAN_USER_TEMPLATE = """\
Plan a niche-focused image set for the following niche cluster.

Niche cluster: {cluster_name}
  Keywords: {cluster_keywords}

Style profile: {style_name}
  Keywords: {style_keywords}

══════════ CONTENT TIER ({content_level}) — READ CAREFULLY ══════════
{llm_directive}
═════════════════════════════════════════════════════════════════════

Tag strategy: {tag_strategy}
Allowed pose types: {allowed_pose_types}

Previous niche themes to AVOID repeating (last 5 series in niche mode):
{previous_themes}

Generate a JSON object with exactly these fields:
{{
  "theme": "<specific theme within this niche — NOT just the niche name>",
  "mood": "<emotional tone fitting the niche audience>",
  "environment": "<primary setting that fits the niche — be specific>",
  "variation_axes": ["<axis1>", "<axis2>", "<axis3>"],
  "core_theme": "<the central visual concept — one sentence>",
  "keyword_cluster": {cluster_keywords},
  "visual_elements": ["<prop1>", "<prop2>", "<prop3>", "<prop4>", "<prop5>"],
  "subject_bias": "<what kind of subject/model fits this niche best>"
}}

visual_elements must be CONCRETE PROPS or DETAILS (e.g. "yoga mat", "leather jacket",
"neon sign", "vintage mirror") — not abstract concepts.
keyword_cluster should include the top SEO-relevant terms for this niche.

Return ONLY the JSON object."""

_SCENE_SYSTEM_PROMPT = """\
You are a professional scene director for an adult photography studio.
You generate scene descriptions optimized for a specific niche audience.

Your output is ALWAYS a JSON array of scene objects with NO extra text, NO markdown fences, NO commentary.

You must respect the content tier directive and allowed pose types
provided — never exceed them, never SOFTEN them. At T3_artnude and
T4_explicit, every scene's pose / camera / mood must be compatible
with the directive's default state of undress. Do NOT euphemise.
Every scene should incorporate at least one visual element from the
niche plan.
"""

_SCENE_USER_TEMPLATE = """\
Generate {scene_count} unique scene descriptions for this niche-focused image set.

Niche plan:
  Theme: {theme}
  Core theme: {core_theme}
  Mood: {mood}
  Environment: {environment}
  Subject bias: {subject_bias}
  Visual elements: {visual_elements}
  Keyword cluster: {keyword_cluster}
  Variation axes: {variation_axes}

══════════ CONTENT TIER ({content_level}) — READ CAREFULLY ══════════
{llm_directive}
═════════════════════════════════════════════════════════════════════

Allowed pose types: {allowed_pose_types}

Each scene must be a JSON object with exactly these fields:
{{
  "variation_axis": "<which variation axis this scene explores>",
  "pose": "<specific pose — MUST be one of the allowed pose types>",
  "camera": "<shot type>",
  "camera_angle": "<angle>",
  "lighting": "<specific lighting description>",
  "environment_detail": "<specific detail — incorporate visual elements>",
  "mood_note": "<emotional note for this scene>",
  "expression": "<facial expression>",
  "visual_element": "<which visual element from the plan this scene features>",
  "subject_detail": "<appearance details fitting the niche subject bias>"
}}

Rules:
- Every scene should feature at least one visual element from the plan
- Spread scenes across variation axes
- Make environments feel authentic to the niche
- Vary poses and camera work while staying on-brand for the niche

Return ONLY the JSON array of {scene_count} scene objects."""


class NicheModeError(Exception):
    """Error in niche mode planning."""


class NicheMode(BaseMode):
    """Mode 4: niche-based image sets with SEO keyword injection."""

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
        return "niche"

    def plan(
        self,
        ctx: GenerationContext,
        *,
        cli_llm_override: str | None = None,
    ) -> dict[str, Any]:
        """Select niche cluster -> LLM expands into a full niche plan."""
        cluster = self._select_cluster(ctx.db_path)
        logger.info("NicheMode: selected cluster %r (%s)", cluster["id"], cluster["name"])

        keywords = cluster.get("keywords", "[]")
        if isinstance(keywords, str):
            keywords = json.loads(keywords)

        # Tag strategy based on tier: T1/T2 stay discovery-focused (DA public
        # gallery visibility); T3/T4 shift to conversion tags.
        tag_strategy = (
            "discovery tags — focus on aesthetic, artistic, and aspirational terms"
            if ctx.content_level in ("T1_suggestive", "T2_implied")
            else "conversion tags — focus on bold, direct, and engaging terms"
        )

        previous_themes = self._load_recent_themes(ctx.db_path)
        allowed_poses = ctx.content_rules.allowed_pose_types

        tier_directive = (
            getattr(ctx.content_rules, "llm_directive", "")
            or f"(No directive declared for {ctx.content_level}.)"
        )
        user_prompt = _PLAN_USER_TEMPLATE.format(
            cluster_name=cluster["name"],
            cluster_keywords=json.dumps(keywords),
            style_name=ctx.style_profile["name"],
            style_keywords=ctx.style_profile["base_style_keywords"],
            content_level=ctx.content_level,
            llm_directive=tier_directive,
            tag_strategy=tag_strategy,
            allowed_pose_types=json.dumps(allowed_poses),
            previous_themes="\n".join(f"  - {t}" for t in previous_themes) or "  (none)",
        )

        plan_sys = ctx.augment_system_prompt(_PLAN_SYSTEM_PROMPT)
        planner_model = self._resolve_role_model(
            "series_planner", cli_llm_override=cli_llm_override,
        )
        plan = self._generate_plan(
            user_prompt, keywords, plan_sys, ctx.family.llm_temperature,
            model=planner_model,
        )

        plan["cluster_id"] = cluster["id"]
        plan["cluster_name"] = cluster["name"]

        logger.info(
            "NicheMode: plan — theme=%r, %d visual_elements, %d keywords",
            plan["theme"],
            len(plan.get("visual_elements", [])),
            len(plan.get("keyword_cluster", [])),
        )
        return plan

    def generate_scenes(
        self,
        series_plan: dict[str, Any],
        ctx: GenerationContext,
        *,
        cli_llm_override: str | None = None,
    ) -> list[dict[str, Any]]:
        """Generate scenes incorporating visual_elements and niche keywords."""
        allowed_poses = ctx.content_rules.allowed_pose_types
        tier_directive = (
            getattr(ctx.content_rules, "llm_directive", "")
            or f"(No directive declared for {ctx.content_level}.)"
        )
        user_prompt = _SCENE_USER_TEMPLATE.format(
            scene_count=25,
            theme=series_plan["theme"],
            core_theme=series_plan.get("core_theme", series_plan["theme"]),
            mood=series_plan["mood"],
            environment=series_plan["environment"],
            subject_bias=series_plan.get("subject_bias", "a model"),
            visual_elements=json.dumps(series_plan.get("visual_elements", [])),
            keyword_cluster=json.dumps(series_plan.get("keyword_cluster", [])),
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
        logger.info("NicheMode: %d scenes generated", len(scenes))
        return scenes

    def get_prompt_keywords(self, series_plan: dict[str, Any]) -> list[str]:
        """Return top 3 keywords from the niche cluster for prompt injection.

        Called by the engine when building prompts in niche mode. The top 3
        keywords are added as extra_keywords to the PromptBuilder.
        """
        keywords = series_plan.get("keyword_cluster", [])
        return keywords[:3]

    # ---- internal helpers ------------------------------------------------

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
        ve = result.get("visual_elements", [])
        if not isinstance(ve, list) or len(ve) < 1:
            logger.warning("NicheMode: no visual_elements in plan")
            return None
        return result

    def _generate_plan(
        self, user_prompt: str, fallback_keywords: list[str],
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
            mode_name="NicheMode plan",
            error_factory=NicheModeError,
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
            mode_name="NicheMode scenes",
            error_factory=NicheModeError,
            model=model,
            schema=SceneList,
        )

    @staticmethod
    def _select_cluster(db_path: Path) -> dict[str, Any]:
        """Weighted random selection from YAML niche clusters.

        ``db_path`` is retained for signature compatibility but unused —
        niche clusters live in ``config/categories.yaml``.
        """
        clusters = CategoriesLoader().niche_clusters()
        if not clusters:
            raise NicheModeError("No niche clusters in config/categories.yaml")
        weights = [c.weight for c in clusters]
        chosen = random.choices(clusters, weights=weights, k=1)[0]
        return {
            "id": chosen.id,
            "name": chosen.name,
            "weight": chosen.weight,
            "keywords": list(chosen.keywords),
        }

    @staticmethod
    def _load_recent_themes(db_path: Path) -> list[str]:
        conn = sqlite3.connect(str(db_path))
        try:
            rows = conn.execute(
                """
                SELECT theme FROM series
                WHERE mode = 'niche' AND status != 'aborted'
                ORDER BY created_at DESC
                LIMIT 5
                """,
            ).fetchall()
            return [r[0] for r in rows if r[0]]
        finally:
            conn.close()
