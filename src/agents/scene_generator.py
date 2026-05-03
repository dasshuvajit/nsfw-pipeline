# WARNING: Always call llm_client.unload_model() after all LLM calls
# BEFORE any ComfyUI rendering. LLM and ComfyUI cannot run simultaneously
# on 48GB unified memory.
"""Scene generator — LLM-produced model-agnostic scene cores.

Takes the series plan (theme / mood / environment / variation_axes),
the content level, and the allowed pose types, then asks the LLM
to produce 20–30 unique scene cores. Each scene is a dict with:

  * **variation_axis** — which axis from the plan this scene varies
  * **pose** — must be from the allowed pose types
  * **camera** — shot type (close-up, medium, full body, etc.)
  * **camera_angle** — angle (eye level, low angle, high angle, etc.)
  * **lighting** — lighting description
  * **environment_detail** — specific detail within the plan's environment
  * **mood_note** — emotional note for this specific scene
  * **expression** — optional facial-expression note
  * **composition_intent** — Phase A: close-up / medium / full-body /
    wide. Drives the multi-axis aspect-ratio scorer.
  * **framing_hint** — Phase A: centered / rule-of-thirds /
    leading-lines (advisory).
  * **audience_target** — Phase A: deviantart / patreon / either.

What this module does NOT produce:

  * **Family-shaped composer inputs** (booru_tags, scene_prose,
    camera_spec, clothing, source_tag). Those live in the
    ``scene_facets`` table and are produced by
    :class:`SceneFacetGenerator` per (scene, family) — sibling models
    in the same family share. This split lets the same scene concept
    feed multiple model families without re-querying the LLM for the
    model-agnostic core.

Scenes are validated via the Pydantic ``Scene`` model in
``src.agents.schemas``. See ARCHITECTURE.md Section 3 (Pipeline Engine
Phase A) and Section 9 (LLM Agents).
"""

from __future__ import annotations

import json
import logging

from pydantic import ValidationError

from src.agents.llm_client import OllamaClient, OllamaJSONParseError
from src.agents.schemas import Scene, SceneList

logger = logging.getLogger(__name__)


# Universal scene-core fields. Kept as an importable constant for the
# few callers (tests, supervisor preview) that want a "did the LLM
# return at least these?" check at a glance — Pydantic's ``Scene``
# model is the authoritative validator.
REQUIRED_SCENE_FIELDS = {
    "variation_axis",
    "pose",
    "camera",
    "camera_angle",
    "lighting",
    "environment_detail",
    "mood_note",
}

SYSTEM_PROMPT = """\
You are a professional scene director for an adult photography studio.
You generate diverse, specific scene descriptions for image sets.

Your output is ALWAYS a JSON array of scene objects with NO extra text, NO markdown fences, NO commentary.

You must respect the content level and allowed pose types provided — never exceed them.
Every scene must be visually distinct from the others — vary pose, camera, lighting, and environment details.

Produce only the model-agnostic scene core (pose, camera, lighting, environment, mood, etc.). Do NOT include family-specific fields like booru_tags, scene_prose, camera_spec, clothing, or source_tag — those are added in a separate step per target model family.
"""

# The ONE schema body. There is no per-family branching here anymore —
# the scene core is identical regardless of which models will later be
# rendered. Family-shaped fields are produced by SceneFacetGenerator.
_SCHEMA_BODY = """\
  "variation_axis": "<which variation axis this scene explores>",
  "pose": "<specific pose — MUST be one of the allowed pose types or a close variant>",
  "camera": "<shot type: close-up, medium shot, three-quarter, full body, upper body, side profile>",
  "camera_angle": "<angle: eye level, low angle, high angle, slightly above, over the shoulder>",
  "lighting": "<specific lighting: soft diffused, dramatic side light, backlit, natural window, warm golden hour, cool blue>",
  "environment_detail": "<specific detail within the environment — vary this across scenes>",
  "mood_note": "<emotional note for this specific scene — vary this across scenes>",
  "expression": "<facial expression cue — optional, but useful for character mode>",
  "composition_intent": "<one of: close-up, medium, full-body, wide — drives aspect-ratio selection. Pick the framing that best fits this scene's pose and storytelling intent.>",
  "framing_hint": "<one of: centered, rule-of-thirds, leading-lines — advisory composition cue (not a hard constraint).>",
  "audience_target": "<one of: deviantart, patreon, either — leave as 'either' unless this specific scene is shaped for one platform's feed.>\""""


# fmt: off
_USER_PROMPT_TEMPLATE = """\
Generate {scene_count} unique scene descriptions for this image set.

Series plan:
  Theme: {theme}
  Mood: {mood}
  Environment: {environment}
  Variation axes: {variation_axes}

Content level: {content_level}
Allowed pose types: {allowed_pose_types}

Each scene must be a JSON object with exactly these fields:
{{
{schema_body}
}}

Rules:
- Every scene must use a DIFFERENT combination of pose + camera + lighting
- Spread scenes evenly across all variation axes
- Be specific and concrete, not generic
- Environment details should feel like they belong in the same location but highlight different elements
- Do NOT include any family-specific fields (booru_tags, scene_prose, camera_spec, clothing, source_tag) — those are added later

Return ONLY the JSON array of {scene_count} scene objects."""
# fmt: on


class SceneGeneratorError(Exception):
    """Any error during scene generation."""


class SceneGenerator:
    """Generate model-agnostic scene cores via the LLM.

    Family-shaped composer inputs (booru_tags, scene_prose, etc.)
    are NOT produced here — see :class:`SceneFacetGenerator`.

    Parameters
    ----------
    llm_client : OllamaClient
        Shared client instance — the caller is responsible for calling
        ``unload_model()`` after all LLM work is done.
    """

    TEMPERATURE = 0.7  # per ARCHITECTURE.md Section 9 LLM Agents table

    def __init__(self, llm_client: OllamaClient) -> None:
        self.llm = llm_client

    def generate(
        self,
        *,
        series_plan: dict,
        content_level: str,
        allowed_pose_types: list[str],
        scene_count: int = 25,
        temperature: float | None = None,
        model: str | None = None,
    ) -> list[dict]:
        """Generate model-agnostic scene cores from a series plan.

        Parameters
        ----------
        series_plan : dict
            Output of ``SeriesPlanner.plan()`` — must have ``theme``,
            ``mood``, ``environment``, ``variation_axes``.
        content_level : str
            One of the 4 tiers: ``T1_suggestive``, ``T2_implied``,
            ``T3_artnude``, ``T4_explicit``.
        allowed_pose_types : list[str]
            From ``content_level_rules.allowed_pose_types``.
        scene_count : int
            Target number of scenes (default 25, range 20–30 per spec).
        temperature : float | None
            Override the default 0.7. None → use ``self.TEMPERATURE``.

        Returns
        -------
        list[dict]
            Validated scene dicts. May be fewer than ``scene_count`` if
            some scenes failed validation.

        Raises
        ------
        SceneGeneratorError
            If the LLM fails to produce any valid scenes after retry.
        """
        # Agent-level fallback for direct tests; production paths
        # (engine → mode → agent) always pass a router-resolved tag.
        if model is None:
            from src.agents.llm_client import resolve_default_ollama_id
            model = resolve_default_ollama_id()

        scene_count = max(20, min(30, scene_count))

        user_prompt = _USER_PROMPT_TEMPLATE.format(
            scene_count=scene_count,
            theme=series_plan["theme"],
            mood=series_plan["mood"],
            environment=series_plan["environment"],
            variation_axes=json.dumps(series_plan["variation_axes"]),
            content_level=content_level,
            allowed_pose_types=json.dumps(allowed_pose_types),
            schema_body=_SCHEMA_BODY,
        )

        effective_temp = temperature if temperature is not None else self.TEMPERATURE

        # First attempt
        scenes = self._attempt(
            user_prompt, scene_count, SYSTEM_PROMPT, effective_temp, model=model,
        )
        if scenes:
            return scenes

        # Retry with a nudge
        logger.warning("Scene generator: first attempt failed, retrying …")
        retry_prompt = (
            user_prompt
            + "\n\nIMPORTANT: Your previous response was not valid JSON. "
            "Return ONLY a raw JSON array of scene objects, no markdown, no commentary."
        )
        scenes = self._attempt(
            retry_prompt, scene_count, SYSTEM_PROMPT, effective_temp, model=model,
        )
        if scenes:
            return scenes

        raise SceneGeneratorError(
            "Failed to generate any valid scenes after 2 attempts. "
            "Check Ollama logs and ensure the model supports structured output."
        )

    def _attempt(
        self,
        user_prompt: str,
        target_count: int,
        system_prompt: str = SYSTEM_PROMPT,
        temperature: float | None = None,
        *,
        model: str,
    ) -> list[dict] | None:
        """Single generate_json attempt. Returns validated list or None."""
        try:
            result = self.llm.generate_json(
                system_prompt,
                user_prompt,
                temperature=temperature if temperature is not None else self.TEMPERATURE,
                num_predict=8192,
                schema=SceneList,
                model=model,
            )
        except OllamaJSONParseError as exc:
            logger.warning("Scene generator JSON parse error: %s", exc)
            return None

        if not isinstance(result, list):
            logger.warning("Scene generator: expected list, got %s", type(result).__name__)
            return None

        valid = []
        for i, scene in enumerate(result):
            if not isinstance(scene, dict):
                logger.warning("Scene %d: not a dict, skipping", i)
                continue
            try:
                validated = Scene.model_validate(scene)
            except ValidationError as exc:
                logger.warning(
                    "Scene %d: schema validation failed, skipping. Errors: %s",
                    i, exc.errors(include_url=False),
                )
                continue
            valid.append(validated.model_dump(mode="python"))

        if not valid:
            logger.warning("Scene generator: zero valid scenes out of %d returned", len(result))
            return None

        if len(valid) < target_count:
            logger.info(
                "Scene generator: %d/%d valid scenes (target %d)",
                len(valid), len(result), target_count,
            )

        logger.info(
            "Scene generator: %d valid scenes generated, axes covered: %s",
            len(valid),
            sorted(set(s["variation_axis"] for s in valid)),
        )
        return valid
