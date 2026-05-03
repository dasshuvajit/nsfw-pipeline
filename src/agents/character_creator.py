# WARNING: Always call llm_client.unload_model() after all LLM calls
# BEFORE any ComfyUI rendering. LLM and ComfyUI cannot run simultaneously
# on 48GB unified memory.
"""Character creator — LLM-assisted character identity generation.

Generates a character identity JSON suitable for ``bootstrap_character.py``.
Output fields: gender, age_appearance, face, hair, body_type,
distinguishing_features, vibe.

This is for **future automated character creation** (Phase 2+). In manual
mode (Phase 0–1) the user writes identity.json by hand — see
ARCHITECTURE.md Section 12: "Do NOT use LLM initially — you need to
understand what makes a good description before automating it."

See ARCHITECTURE.md Section 12 and Section 13 (LLM Roles: Character Creator,
temperature 0.8, NOT content-level aware).
"""

from __future__ import annotations

import logging

from src.agents.llm_client import OllamaClient, OllamaJSONParseError
from src.agents.schemas import CharacterSchema

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = {
    "gender",
    "age_appearance",
    "face",
    "hair",
    "body_type",
    "distinguishing_features",
    "vibe",
}

SYSTEM_PROMPT = """\
You are an expert character designer for a professional adult photography studio.
You create detailed, consistent character identities that can be rendered across many images.

Your output is ALWAYS a single JSON object with NO extra text, NO markdown fences, NO commentary.

Focus on PHYSICAL APPEARANCE ONLY — no backstory, personality, or narrative.
Be specific and visual: "oval face, sharp jawline, almond eyes, defined cheekbones" not "pretty face".
Every feature should be rendereable by a text-to-image model (Stable Diffusion).
"""

# fmt: off
_USER_PROMPT_TEMPLATE = """\
Create a unique character identity for an adult photography model.

Guidelines:
  Target vibe: {target_vibe}
  {gender_hint}
  {additional_notes}

Existing characters to AVOID duplicating:
{existing_characters}

Generate a JSON object with exactly these fields:
{{
  "gender": "<female|male>",
  "age_appearance": "<apparent age range, e.g. 'late 20s', 'early 30s'>",
  "face": "<detailed face description: shape, eyes, jawline, cheekbones, lips — be specific>",
  "hair": "<hair description: length, color, texture, style>",
  "body_type": "<body type: slim athletic, curvy, petite, etc.>",
  "distinguishing_features": "<1-2 unique visual features: freckles, dimples, beauty mark, etc.>",
  "vibe": "<2-3 word overall aesthetic/mood: 'soft elegant', 'bold confident', 'warm natural'>"
}}

Rules:
- Be SPECIFIC and VISUAL — every descriptor should translate to a Stable Diffusion prompt
- The face description is the most important field — it defines identity consistency
- Avoid generic descriptors like "beautiful" or "attractive"
- Distinguishing features help IPAdapter lock identity across renders

Return ONLY the JSON object."""
# fmt: on


class CharacterCreatorError(Exception):
    """Any error during character identity generation."""


class CharacterCreator:
    """Generate character identities via the LLM.

    Parameters
    ----------
    llm_client : OllamaClient
        Shared client instance — the caller is responsible for calling
        ``unload_model()`` after all LLM work is done.
    """

    TEMPERATURE = 0.8  # per ARCHITECTURE.md Section 13 — NOT content-level aware

    def __init__(self, llm_client: OllamaClient) -> None:
        self.llm = llm_client

    def create(
        self,
        *,
        target_vibe: str = "soft elegant",
        gender: str | None = None,
        additional_notes: str = "",
        existing_characters: list[dict] | None = None,
        model: str | None = None,
    ) -> dict:
        """Generate a character identity.

        Parameters
        ----------
        target_vibe : str
            Desired overall aesthetic (e.g. ``"soft elegant"``,
            ``"bold confident"``).
        gender : str | None
            If specified, constrains gender. Otherwise the LLM chooses.
        additional_notes : str
            Free-form guidance (e.g. ``"redhead"`` or ``"east asian features"``).
        existing_characters : list[dict] | None
            Summaries of existing characters to avoid duplication.

        Returns
        -------
        dict
            Validated character identity with all REQUIRED_FIELDS.

        Raises
        ------
        CharacterCreatorError
            If the LLM fails to produce valid JSON after one retry.
        """
        gender_hint = f"Preferred gender: {gender}" if gender else "Gender: your choice (female or male)"
        notes = f"Additional guidance: {additional_notes}" if additional_notes else ""

        existing_str = "  (none yet)" if not existing_characters else "\n".join(
            f"  - {c.get('name', '?')}: {c.get('vibe', '?')}, {c.get('hair', '?')}"
            for c in (existing_characters or [])
        )

        user_prompt = _USER_PROMPT_TEMPLATE.format(
            target_vibe=target_vibe,
            gender_hint=gender_hint,
            additional_notes=notes,
            existing_characters=existing_str,
        )

        # First attempt
        identity = self._attempt(user_prompt, model=model)
        if identity is not None:
            return identity

        # Retry once
        logger.warning("Character creator: first attempt failed, retrying …")
        retry_prompt = (
            user_prompt
            + "\n\nIMPORTANT: Your previous response was not valid JSON. "
            "Return ONLY a raw JSON object, no markdown, no commentary."
        )
        identity = self._attempt(retry_prompt, model=model)
        if identity is not None:
            return identity

        raise CharacterCreatorError(
            "Failed to generate a valid character identity after 2 attempts. "
            "Check Ollama logs and ensure the model supports structured output."
        )

    def _attempt(
        self, user_prompt: str, *, model: str | None = None,
    ) -> dict | None:
        """Single generate_json attempt. Returns validated dict or None."""
        try:
            result = self.llm.generate_json(
                SYSTEM_PROMPT,
                user_prompt,
                temperature=self.TEMPERATURE,
                num_predict=2048,
                schema=CharacterSchema,
                model=model,
            )
        except OllamaJSONParseError as exc:
            logger.warning("Character creator JSON parse error: %s", exc)
            return None

        if not isinstance(result, dict):
            logger.warning("Character creator: expected dict, got %s", type(result).__name__)
            return None

        missing = REQUIRED_FIELDS - result.keys()
        if missing:
            logger.warning("Character creator: missing required fields: %s", missing)
            return None

        logger.info(
            "Character identity generated: gender=%s, vibe=%r",
            result["gender"],
            result["vibe"],
        )
        return result
