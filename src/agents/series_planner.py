# WARNING: Always call llm_client.unload_model() after all LLM calls
# BEFORE any ComfyUI rendering. LLM and ComfyUI cannot run simultaneously
# on 48GB unified memory.
"""Series planner — LLM-generated thematic plan for an image set.

Takes a character (or theme), a style profile, a content level, and
the content-level rules, then asks the LLM to produce a cohesive
series plan with:

  * **theme** — overarching visual/narrative theme for the set
  * **mood** — emotional tone (must stay within content_rules mood_range)
  * **environment** — primary setting
  * **variation_axes** — list of axes the scene generator should vary across

The plan is stored as JSON in ``series.llm_series_plan`` and fed into
the scene generator as context.

See ARCHITECTURE.md Section 7 (Mode 1) and Section 13 (LLM roles).
"""

from __future__ import annotations

import json
import logging

from src.agents.llm_client import OllamaClient, OllamaJSONParseError
from src.agents.schemas import SeriesPlan

logger = logging.getLogger(__name__)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.memory.model_registry import ModelPromptGuide

# Kept for backward compatibility with callers that introspect the
# expected field set; actual validation now flows through SeriesPlan.
REQUIRED_FIELDS = {"theme", "mood", "environment", "variation_axes"}

SYSTEM_PROMPT = """\
You are a creative director for a professional adult photography studio.
You plan cohesive image sets that will be sold on platforms like DeviantArt and Patreon.

Your output is ALWAYS a single JSON object with NO extra text, NO markdown fences, NO commentary.

You must respect the content level and allowed pose types provided — never exceed them.
"""

# fmt: off
_USER_PROMPT_TEMPLATE = """\
Plan a cohesive image set for the following character and style.

Character:
  Name: {character_name}
  Base prompt: {base_prompt}
  Vibe: {vibe}

Style profile: {style_name}
  Keywords: {style_keywords}

Content level: {content_level}
Allowed pose types: {allowed_pose_types}
Mood range: {mood_range}
Environment constraint: {environment_constraint}

Previous themes to AVOID repeating (last 5 series for this character):
{previous_themes}

Generate a JSON object with exactly these fields:
{{
  "theme": "<overarching visual/narrative theme for the set — be specific and evocative>",
  "mood": "<emotional tone — must be from the allowed mood range>",
  "environment": "<primary setting/location — be specific: 'sunlit Parisian loft' not just 'indoor'>",
  "variation_axes": ["<axis1>", "<axis2>", "<axis3>", "<axis4>"]
}}

The variation_axes should be 3-5 specific dimensions the scenes will vary across.
Good axes: "pose intensity", "camera distance", "lighting warmth", "outfit detail", "expression range"
Bad axes: "random", "misc", "other"

Return ONLY the JSON object."""
# fmt: on


def _build_system_prompt(prompt_guide: "ModelPromptGuide | None" = None) -> str:
    """Build a model-aware system prompt for series planning.

    Pulls the family-level LLM hint + per-model trigger/avoid lists
    from the merged ``ModelPromptGuide``. No per-prompt_style branches.
    """
    base = SYSTEM_PROMPT
    if not prompt_guide:
        return base
    parts: list[str] = []
    if prompt_guide.llm_hint:
        parts.append(f"\nMODEL FAMILY HINT:\n{prompt_guide.llm_hint}")
    if prompt_guide.trigger_words:
        parts.append(
            "\nThe target model responds well to these TRIGGER WORDS:\n"
            + ", ".join(prompt_guide.trigger_words)
        )
    if prompt_guide.avoid_words:
        parts.append(
            "\nAVOID these words/phrases in themes and descriptions:\n"
            + ", ".join(prompt_guide.avoid_words)
        )
    return base + "".join(parts) if parts else base


class SeriesPlannerError(Exception):
    """Any error during series planning."""


class SeriesPlanner:
    """Generate a series plan via the LLM.

    Parameters
    ----------
    llm_client : OllamaClient
        Shared client instance — the caller is responsible for calling
        ``unload_model()`` after all LLM work is done.
    """

    TEMPERATURE = 0.7  # per ARCHITECTURE.md Section 13 LLM Roles table

    def __init__(self, llm_client: OllamaClient) -> None:
        self.llm = llm_client

    def plan(
        self,
        *,
        character_name: str,
        base_prompt: str,
        vibe: str = "",
        style_name: str,
        style_keywords: str,
        content_level: str,
        content_rules: dict,
        previous_themes: list[str] | None = None,
        prompt_guide: "ModelPromptGuide | None" = None,
        temperature: float | None = None,
        model: str | None = None,
    ) -> dict:
        """Generate a series plan and return it as a validated dict.

        Parameters
        ----------
        character_name : str
            Character id (e.g. ``char_001``).
        base_prompt : str
            Character's immutable base prompt from the DB.
        vibe : str
            Character vibe from identity.json (e.g. ``"soft elegant"``).
        style_name : str
            Style profile name (e.g. ``"golden_hour_natural"``).
        style_keywords : str
            Comma-separated style keywords from the profile.
        content_level : str
            One of the 4 tiers: ``T1_suggestive``, ``T2_implied``,
            ``T3_artnude``, ``T4_explicit``.
        content_rules : dict
            Row from ``content_level_rules`` with keys:
            ``allowed_pose_types`` (JSON string), ``scene_constraints``
            (JSON string).
        previous_themes : list[str] | None
            Up to 5 recent themes for this character, so the LLM avoids
            repetition.

        Returns
        -------
        dict
            Validated plan with keys: theme, mood, environment, variation_axes.

        Raises
        ------
        SeriesPlannerError
            If the LLM fails to produce valid JSON after one retry.
        """
        # Agent-level fallback so direct tests don't have to fabricate a
        # model string. Production paths (engine → mode → agent) always
        # pass a router-resolved tag.
        if model is None:
            from src.agents.llm_client import resolve_default_ollama_id
            model = resolve_default_ollama_id()

        allowed_poses = content_rules.get("allowed_pose_types", "[]")
        if isinstance(allowed_poses, str):
            allowed_poses = json.loads(allowed_poses)

        constraints = content_rules.get("scene_constraints", "{}")
        if isinstance(constraints, str):
            constraints = json.loads(constraints)

        mood_range = constraints.get("mood_range", [])
        environment_constraint = constraints.get("environment", "any")

        prev_str = "\n".join(f"  - {t}" for t in (previous_themes or [])) or "  (none — this is the first series)"

        user_prompt = _USER_PROMPT_TEMPLATE.format(
            character_name=character_name,
            base_prompt=base_prompt,
            vibe=vibe or "(not specified)",
            style_name=style_name,
            style_keywords=style_keywords,
            content_level=content_level,
            allowed_pose_types=json.dumps(allowed_poses),
            mood_range=json.dumps(mood_range),
            environment_constraint=environment_constraint,
            previous_themes=prev_str,
        )

        system_prompt = _build_system_prompt(prompt_guide)
        effective_temp = temperature if temperature is not None else self.TEMPERATURE

        # First attempt
        plan = self._attempt(user_prompt, system_prompt, effective_temp, model=model)
        if plan is not None:
            return plan

        # Retry once with a nudge
        logger.warning("Series plan: first attempt failed JSON validation, retrying …")
        retry_prompt = (
            user_prompt
            + "\n\nIMPORTANT: Your previous response was not valid JSON. "
            "Return ONLY a raw JSON object, no markdown, no commentary."
        )
        plan = self._attempt(retry_prompt, system_prompt, effective_temp, model=model)
        if plan is not None:
            return plan

        raise SeriesPlannerError(
            "Failed to generate a valid series plan after 2 attempts. "
            "Check Ollama logs and ensure the model supports structured output."
        )

    def _attempt(
        self,
        user_prompt: str,
        system_prompt: str = SYSTEM_PROMPT,
        temperature: float | None = None,
        *,
        model: str,
    ) -> dict | None:
        """Single generate_json attempt. Returns validated dict or None."""
        try:
            result = self.llm.generate_json(
                system_prompt,
                user_prompt,
                temperature=temperature if temperature is not None else self.TEMPERATURE,
                num_predict=2048,
                schema=SeriesPlan,
                model=model,
            )
        except OllamaJSONParseError as exc:
            logger.warning("Series plan JSON/schema error: %s", exc)
            return None

        if not isinstance(result, dict):
            logger.warning("Series plan: expected dict, got %s", type(result).__name__)
            return None

        logger.info(
            "Series plan generated: theme=%r, mood=%r, %d variation axes",
            result["theme"],
            result["mood"],
            len(result["variation_axes"]),
        )
        return result
