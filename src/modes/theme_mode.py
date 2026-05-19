"""Mode 2 — Theme-based pipeline.

ARCHITECTURE.md Section 8:
  - Category from ``theme_categories`` table (weighted random)
  - Subject comes from the LLM theme plan, not the character registry

The theme mode:
  1. ``plan(ctx)`` — selects a theme category via weighted random from
     the DB, then calls the LLM to generate a specific theme within that
     category. Validates the theme is not too vague and not repeated in
     recent memory.
  2. ``generate_scenes(series_plan, ctx)`` — calls ``SceneGenerator``
     with the plan. The LLM invents subjects for each scene.
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
from src.modes._llm_helpers import (
    repair_colon_suffix_aesthetic_keys,
    run_llm_with_retry,
    validate_scene_list,
    warn_if_missing_aesthetic_anchors,
)
from src.modes.base_mode import BaseMode

if TYPE_CHECKING:
    from src.agents.llm_router import LLMRouter

logger = logging.getLogger(__name__)


_PLAN_SYSTEM_PROMPT = """\
You are a creative director for a professional adult photography studio.
You plan cohesive image sets around a specific theme category.

Your output is ALWAYS a single JSON object with NO extra text, NO markdown fences, NO commentary.

You must respect the content tier directive and allowed pose types
provided — never exceed them, never SOFTEN them. If the tier directive
calls for explicit nude content, the THEME and SUBJECT_DESCRIPTION must
reflect that explicitly (e.g. "nude fine-art studio session", not
"lingerie golden-hour"). Tiered fields cascade: a tame theme + tame
subject produces tame scenes downstream, no matter how strong the
tier directive becomes later. Plan at the tier you're given.

SOLO SUBJECT INVARIANT: every theme depicts exactly one adult woman
as the sole human subject. NEVER plan multi-subject themes —
no "couple at sunset", no "two friends", no "group portrait", no
"her partner". The subject_description must read as a single woman
in every case.

SINGLE-SCENE INVARIANT (subject_description): the subject_description
is INJECTED VERBATIM INTO EVERY SCENE's prompt body. It must describe
ONLY the person — appearance, attire / state of undress, age signal.
It must NEVER reference cross-scene variety, composition, framing, or
direction. NEVER write "in varying poses", "across varying
compositions", "in various poses across the series", "from different
angles", "in multiple compositions", "across scenes", "throughout the
series". Those words make the image model render a 4-panel collage
grid because each scene's encoder reads them as a polyptych
instruction. Keep subject_description ≤ 18 words, NO sentence-final
"in/across/throughout" clauses, NO adverbs of variety.
"""

_PLAN_USER_TEMPLATE = """\
Plan a cohesive themed image set for the following theme category.

Theme category: {category_name}
  Description: {category_description}

Style profile: {style_name}
  Keywords: {style_keywords}

══════════ CONTENT TIER ({content_level}) — READ CAREFULLY ══════════
{llm_directive}
═════════════════════════════════════════════════════════════════════

Allowed pose types: {allowed_pose_types}

Previous themes to AVOID repeating (last 5 series in this category):
{previous_themes}

Series aesthetic menu (Phase 3, vocab v6) — pick coherent combo:
  color_palette options: {color_palette_options}
  photographer_ref options: {photographer_ref_options}
  art_movement options: {art_movement_options}

Generate a JSON object with exactly these fields:
{{
  "theme": "<specific evocative theme within this category — NOT just the category name; must reflect the tier above>",
  "mood": "<emotional tone that fits the theme AND the tier>",
  "environment": "<primary setting/location — be specific: 'rain-soaked Tokyo alley at night' not 'outdoor'>",
  "variation_axes": ["<axis1>", "<axis2>", "<axis3>", "<axis4>"],
  "subject_description": "<≤ 18 WORDS describing ONLY the subject (appearance, attire / state of undress, age signal). At T3/T4, name nudity per the tier directive (do NOT default to lingerie if T4 says fully nude). NEVER include the words 'varying', 'various', 'across', 'throughout', 'multiple', 'different angles', 'compositions', 'poses' — those phrases get injected verbatim into every scene's prompt and the image model reads them as a polyptych instruction, producing a 4-panel grid. Example GOOD: 'A confident adult woman, fully nude, mature features, natural skin'. Example BAD: 'A nude woman in varying poses across different compositions'>",
  "color_palette": "<one PALETTE_* tag from the menu above — the series's colour grade>",
  "photographer_ref": "<one PHOTOG_* tag from the menu above — the photographer-signature for the series>",
  "art_movement": "<one ART_MOVE_* tag from the menu, OR null if no movement reference fits>"
}}

The theme must be SPECIFIC (not vague like "beauty" or "nature").
The subject_description must MATCH the tier directive's default state.
Variation_axes should be 3-5 dimensions the scenes will vary across.

═══ AESTHETIC COHERENCE RULE (Phase 3, vocab v6) ═══════════════════
The (color_palette, photographer_ref, art_movement) triple MUST
form a coherent visual world. Known-coherent pairings:
  - PHOTOG_HELMUT_NEWTON + PALETTE_MONOCHROME_HIGH_CONTRAST +
    ART_MOVE_FILM_NOIR_1940S  (provocative urban noir)
  - PHOTOG_PETTER_HEGRE + PALETTE_TUSCAN_EARTH +
    ART_MOVE_DUTCH_GOLDEN_VERMEER  (naturalistic villa nudes)
  - PHOTOG_SLIM_AARONS + PALETTE_LUBEZKI_NATURAL_GOLDEN +
    (omit art_movement)  (golden-age jet-set glamour)
  - PHOTOG_PETRA_COLLINS + PALETTE_WES_ANDERSON_PASTEL +
    ART_MOVE_WES_ANDERSON  (dreamy pastel symmetry)
  - PHOTOG_BILL_HENSON + PALETTE_BAROQUE_CARAVAGGIO +
    ART_MOVE_BAROQUE_CARAVAGGIO  (twilight chiaroscuro)
NEVER mix incompatible worlds (e.g. PHOTOG_HELMUT_NEWTON +
PALETTE_WES_ANDERSON_PASTEL — they belong to different aesthetic
universes and the result will read as confused).

Return ONLY the JSON object."""

_SCENE_SYSTEM_PROMPT = """\
You are a professional scene director for an adult photography studio.
You generate diverse, specific scene descriptions for themed image sets.

Your output is ALWAYS a JSON array of scene objects with NO extra text, NO markdown fences, NO commentary.

You must respect the content tier directive and allowed pose types
provided — never exceed them, never SOFTEN them. At T3_artnude and
T4_explicit, every scene's pose / camera / mood_note must be compatible
with the tier directive's default state of undress. Do NOT euphemise
(no "draped" / "veiled" / "shadow hides her form" at T4 — those belong
to T2). Every scene must be visually distinct from the others.

SOLO SUBJECT INVARIANT: every scene depicts exactly one adult woman
as the sole human subject. Pose / camera / mood_note / expression
must describe a single subject. NEVER write "her partner", "with
him", "they embrace", "two women", "another person", "couple",
"group" — those break the single-female composition. Pronouns are
always singular.
"""

_SCENE_USER_TEMPLATE = """\
Generate {scene_count} unique scene descriptions for this themed image set.

Series plan:
  Theme: {theme}
  Mood: {mood}
  Environment: {environment}
  Subject: {subject_description}
  Variation axes: {variation_axes}

══════════ CONTENT TIER ({content_level}) — READ CAREFULLY ══════════
{llm_directive}
═════════════════════════════════════════════════════════════════════

Allowed pose types: {allowed_pose_types}

Each scene must be a JSON object with exactly these fields:
{{
  "variation_axis": "<which variation axis this scene explores>",
  "pose": "<specific pose — MUST be one of the allowed pose types or a close variant>",
  "camera": "<shot type: close-up, medium shot, three-quarter, full body, upper body, side profile>",
  "camera_angle": "<angle: eye level, low angle, high angle, slightly above, over the shoulder>",
  "lighting": "<specific lighting description>",
  "environment_detail": "<specific detail within the environment>",
  "mood_note": "<emotional note for this scene>",
  "expression": "<facial expression>",
  "subject_detail": "<specific appearance detail for the subject in THIS scene>"
}}

Rules:
- Every scene must use a DIFFERENT combination of pose + camera + lighting
- Spread scenes evenly across all variation axes
- The subject should feel like the same person across scenes (consistent general appearance)
- Environment details should vary but belong to the same location

Return ONLY the JSON array of {scene_count} scene objects."""


class ThemeModeError(Exception):
    """Error in theme mode planning."""


class ThemeMode(BaseMode):
    """Mode 2: theme-based image sets without character identity lock."""

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
        return "theme"

    def plan(
        self,
        ctx: GenerationContext,
        *,
        cli_llm_override: str | None = None,
    ) -> dict[str, Any]:
        """Select theme category -> LLM generates specific theme plan."""
        # Pick a weighted-random theme category
        category = self._select_category(ctx.db_path)
        logger.info("ThemeMode: selected category %r (%s)", category["id"], category["name"])

        # Load recent themes in this category to avoid repetition
        previous_themes = self._load_recent_themes(ctx.db_path, category["id"])

        allowed_poses = ctx.content_rules.allowed_pose_types
        # Tier directive sourced from categories.yaml::content_levels.<tier>
        # so the planner LLM sees the SAME rich guidance the scene-facet
        # generator gets. Without this, the planner only saw a bare
        # "Content level: T4_explicit" string and produced tame
        # themes/subject_descriptions (e.g. "delicate silk lingerie"
        # for a T4_explicit run). Fixed 2026-05-17.
        tier_directive = (
            getattr(ctx.content_rules, "llm_directive", "")
            or f"(No directive declared for {ctx.content_level}.)"
        )
        # Phase 3 (vocab v6) — load the aesthetic-anchor menu narrowed
        # by style_profile + theme compatibility lists. Defaults to full
        # menu when lists are absent (back-compat for pre-Phase-3
        # profiles/themes).
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
        # Intersect with theme.compatible_* lists when category declares
        # them (categories.yaml Phase 4 work; lite version: read whatever
        # the category dict carries).
        for key in (
            "compatible_palettes",
            "compatible_photographers",
            "compatible_art_movements",
        ):
            theme_list = category.get(key, [])
            if theme_list:
                profile_list = style_profile_compat.get(key, []) or []
                if profile_list:
                    # Intersection of profile + theme
                    style_profile_compat[key] = [
                        t for t in profile_list if t in theme_list
                    ]
                else:
                    style_profile_compat[key] = list(theme_list)
        aesthetic_menu = _resolve_aesthetic_menu(
            style_profile_compat=style_profile_compat,
        )

        user_prompt = _PLAN_USER_TEMPLATE.format(
            category_name=category["name"],
            category_description=category.get("description", ""),
            style_name=ctx.style_profile["name"],
            style_keywords=ctx.style_profile["base_style_keywords"],
            content_level=ctx.content_level,
            llm_directive=tier_directive,
            allowed_pose_types=json.dumps(allowed_poses),
            previous_themes="\n".join(f"  - {t}" for t in previous_themes) or "  (none)",
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

        # Attach category metadata
        plan["category_id"] = category["id"]
        plan["category_name"] = category["name"]
        plan.setdefault("subject_description", "")

        # vocab v7 — server-side validator for the SINGLE-SCENE
        # INVARIANT. The plan prompt forbids cross-scene variety
        # language in subject_description, but Cydonia at temp≥0.7
        # routinely ignores embedded constraints. Sanitize: strip
        # offending phrases AND any orphan connector words left
        # behind via the shared ``sanitize_grid_phrases`` helper —
        # same two-stage cleanup the migrate script uses, so runtime
        # + migration are byte-equivalent.
        from src.prompt.builder import sanitize_grid_phrases
        raw_sd = plan.get("subject_description", "") or ""
        if raw_sd:
            cleaned_sd, changed = sanitize_grid_phrases(raw_sd)
            if changed:
                logger.warning(
                    "ThemeMode: SINGLE-SCENE INVARIANT — LLM emitted "
                    "cross-scene variety language in subject_description. "
                    "Sanitized. before=%r after=%r llm=%s",
                    raw_sd, cleaned_sd, planner_model,
                )
                # Round-5 verifier (F5) — hard-fail when sanitization
                # leaves an empty subject_description. The LLM emitted
                # ONLY grid/mirror phrases; downstream composer would
                # render a generic prompt with no subject anchor.
                # Raise so the operator notices + re-prompts.
                if not cleaned_sd:
                    raise ThemeModeError(
                        f"ThemeMode: subject_description sanitized to "
                        f"empty — LLM ({planner_model}) emitted only "
                        f"forbidden grid/mirror phrases. Original: "
                        f"{raw_sd!r}. Re-run plan (the LLM should "
                        f"produce different output on retry) or "
                        f"switch --llm to a different registry id."
                    )
                plan["subject_description"] = cleaned_sd
        # Verifier round-2 I4 — propagate the theme category's environment
        # whitelist (intersected with the style_profile's when both
        # populated) so SceneFacetGenerator narrows the LLM's
        # environment.setting menu in engine.py. Empty list / missing
        # falls through as the full vocab menu.
        cat_envs = category.get("compatible_environments", []) or []
        sp_envs = ctx.style_profile.get(
            "compatible_environments", []
        ) or []
        if cat_envs and sp_envs:
            plan["compatible_environments"] = [
                t for t in sp_envs if t in cat_envs
            ] or list(cat_envs)
        else:
            plan["compatible_environments"] = list(cat_envs or sp_envs)

        logger.info(
            "ThemeMode: plan — theme=%r, mood=%r, subject=%r",
            plan["theme"], plan["mood"], plan.get("subject_description", ""),
        )
        return plan

    def generate_scenes(
        self,
        series_plan: dict[str, Any],
        ctx: GenerationContext,
        *,
        cli_llm_override: str | None = None,
    ) -> list[dict[str, Any]]:
        """Generate scenes — LLM creates subjects per scene (no character lock)."""
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
            subject_description=series_plan.get("subject_description", "a model"),
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
        logger.info("ThemeMode: %d scenes generated", len(scenes))
        return scenes

    # ---- internal helpers ------------------------------------------------

    _VAGUE_THEMES: frozenset[str] = frozenset(
        {"beauty", "nature", "elegance", "photography", "art", "model"}
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
        if theme in self._VAGUE_THEMES:
            logger.warning("ThemeMode: rejected vague theme %r", result["theme"])
            return None
        # Round-5 defensive: salvage `color_palette:` (colon-suffix)
        # quirk before warning. Cydonia heretic-vision was observed
        # producing `"color_palette:": "PALETTE_X"` alongside the
        # schema-required `"color_palette": null`. Repair-then-warn
        # so the warning only fires on real misses.
        repair_colon_suffix_aesthetic_keys(result)
        # Verifier round-4 IMPORTANT-5 — soft warn when aesthetic anchors
        # are missing. Doesn't reject (Pony legitimately omits two of
        # three), but surfaces silent Phase 3 degradation in run_log.
        warn_if_missing_aesthetic_anchors(result, mode_name="ThemeMode")
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
            mode_name="ThemeMode plan",
            error_factory=ThemeModeError,
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
            mode_name="ThemeMode scenes",
            error_factory=ThemeModeError,
            model=model,
            schema=SceneList,
        )

    @staticmethod
    def _select_category(db_path: Path) -> dict[str, Any]:
        """Weighted random selection from YAML theme categories.

        ``db_path`` is retained for signature compatibility but unused —
        theme categories live in ``config/categories.yaml``.
        """
        from src.memory.categories_loader import CategoriesLoader

        categories = CategoriesLoader().theme_categories()
        if not categories:
            raise ThemeModeError("No theme categories in config/categories.yaml")
        weights = [c.weight for c in categories]
        chosen = random.choices(categories, weights=weights, k=1)[0]
        return {
            "id": chosen.id,
            "name": chosen.name,
            "weight": chosen.weight,
            "description": chosen.description,
            # Phase 3 (vocab v6) — propagate compat lists into the
            # plan-time dict so the SeriesPlanner aesthetic-menu
            # intersection logic actually receives them. Verifier B1
            # — pre-fix these were dropped at the dataclass→dict
            # boundary and the YAML lists were inert.
            "compatible_palettes": list(chosen.compatible_palettes),
            "compatible_photographers": list(chosen.compatible_photographers),
            "compatible_art_movements": list(chosen.compatible_art_movements),
            "compatible_environments": list(chosen.compatible_environments),
        }

    @staticmethod
    def _load_recent_themes(db_path: Path, category_id: str) -> list[str]:
        """Load last 5 themes for series in theme mode."""
        conn = sqlite3.connect(str(db_path))
        try:
            rows = conn.execute(
                """
                SELECT theme FROM series
                WHERE mode = 'theme' AND status != 'aborted'
                ORDER BY created_at DESC
                LIMIT 5
                """,
            ).fetchall()
            return [r[0] for r in rows if r[0]]
        finally:
            conn.close()
