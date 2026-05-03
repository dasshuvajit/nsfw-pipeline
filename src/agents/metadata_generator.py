# WARNING: Always call llm_client.unload_model() after all LLM calls
# BEFORE any ComfyUI rendering. LLM and ComfyUI cannot run simultaneously
# on 48GB unified memory.
"""Metadata generator — LLM-produced title, description, and tags for a set.

Takes series context, the image set, and the content level, then generates
platform-ready metadata for DeviantArt / Patreon:

  * **title** — max 80 characters, evocative, content-level appropriate
  * **description** — 2–3 sentences, content-level aware
  * **tags** — 15–25 tags, tier-aware:
    * ``T1_suggestive`` — lingerie / swimwear tags, fashion-editorial framing
    * ``T2_implied`` — implied-nude tags, beauty/aesthetic emphasis
    * ``T3_artnude`` — fine-art nude tags, painterly / figurative
    * ``T4_explicit`` — bolder, direct; NEVER surfaced to DA public gallery

See ARCHITECTURE.md Section 17 (Metadata & Packaging) and Section 13
(LLM Roles: Metadata Generator, temperature 0.6, content-tier aware).
"""

from __future__ import annotations

import logging

from src.agents.llm_client import OllamaClient, OllamaJSONParseError

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = {"title", "description", "tags"}

SYSTEM_PROMPT = """\
You are a metadata specialist for a professional adult content studio.
You write titles, descriptions, and tags for image sets sold on DeviantArt and Patreon.

Your output is ALWAYS a single JSON object with NO extra text, NO markdown fences, NO commentary.

You must respect the content tier (4-tier system):
- "T1_suggestive": lingerie / swimwear / revealing outfit, NO nudity. Tags
  stay fashion-editorial, platform-safe for public discovery.
- "T2_implied": implied nude, no visible nipples/genitalia. Beauty-focused,
  aesthetic, elegant — DA public-gallery acceptable with care.
- "T3_artnude": full artistic nude, fine-art framing. Painterly, classical,
  figurative vocabulary. DA Premium Gallery / Fanvue only.
- "T4_explicit": graphic / explicit. Bolder, direct tags; keep off any
  public discovery surface — Fanvue-gated only.
"""

# fmt: off
_USER_PROMPT_TEMPLATE = """\
Generate metadata for this image set.

Series:
  Theme: {theme}
  Mood: {mood}
  Environment: {environment}
  Character: {character_name}

Content level: {content_level}
Image count: {image_count}
Style keywords: {style_keywords}

Generate a JSON object with exactly these fields:
{{
  "title": "<max 80 characters, evocative, content-level appropriate>",
  "description": "<2-3 sentences describing the set — sell the mood and aesthetic>",
  "tags": ["<tag1>", "<tag2>", ... ] (15-25 tags)
}}

Tag guidelines for {content_level} level:
{tag_guidelines}

Rules:
- Title MUST be under 80 characters
- Description should make someone want to view/buy the set
- Tags should include: character descriptors, mood, style, environment, AND platform discovery terms
- Include tags like "digital art", "ai art", "photography", "portrait" for discoverability
- Do NOT include platform names or pricing in tags

Return ONLY the JSON object."""
# fmt: on

TAG_GUIDELINES = {
    "T1_suggestive": (
        "- Emphasize: fashion, editorial, lingerie, swimwear, portrait, boudoir-lite\n"
        "- Include mood words: glamorous, confident, elegant, cinematic\n"
        "- Avoid: nude / topless / explicit language — there is NO nudity here\n"
        "- Think: Vogue editorial, fashion campaign, lifestyle photography"
    ),
    "T2_implied": (
        "- Emphasize: beauty, elegance, artistic, aesthetic, portrait, graceful\n"
        "- Include mood words: dreamy, soft, serene, warm, luminous\n"
        "- Avoid: explicit or overtly sexual tags — this is implied, not shown\n"
        "- Think: fine art photography, editorial, beauty campaign"
    ),
    "T3_artnude": (
        "- Emphasize: fine art nude, figurative, painterly, classical, sculptural\n"
        "- Include mood words: contemplative, intimate, reverent, luminous\n"
        "- Frame nudity as art-historical — body as subject, not spectacle\n"
        "- Think: gallery print, Weston / Newton / Ritts heritage, studio nude"
    ),
    "T4_explicit": (
        "- Bolder vocabulary: bold, intense, expressive, intimate, sensual\n"
        "- Include mood words: passionate, daring, provocative, raw\n"
        "- Can reference body/pose intensity more directly\n"
        "- Think: premium adult art, exclusive content, daring photography"
    ),
}


class MetadataGeneratorError(Exception):
    """Any error during metadata generation."""


class MetadataGenerator:
    """Generate set metadata via the LLM.

    Parameters
    ----------
    llm_client : OllamaClient
        Shared client instance — the caller is responsible for calling
        ``unload_model()`` after all LLM work is done.
    """

    TEMPERATURE = 0.6  # per ARCHITECTURE.md Section 13 LLM Roles table

    def __init__(self, llm_client: OllamaClient) -> None:
        self.llm = llm_client

    def generate(
        self,
        *,
        theme: str,
        mood: str,
        environment: str,
        character_name: str = "",
        content_level: str,
        image_count: int,
        style_keywords: str = "",
        temperature: float | None = None,
        model: str | None = None,
    ) -> dict:
        """Generate metadata for an image set.

        Parameters
        ----------
        theme : str
            Series theme.
        mood : str
            Series mood.
        environment : str
            Series environment.
        character_name : str
            Character id or name (empty for non-character modes).
        content_level : str
            One of the 4 tiers: ``T1_suggestive``, ``T2_implied``,
            ``T3_artnude``, ``T4_explicit``.
        image_count : int
            Number of images in the set.
        style_keywords : str
            Comma-separated style keywords from the profile.

        Returns
        -------
        dict
            Validated metadata with keys: title, description, tags.

        Raises
        ------
        MetadataGeneratorError
            If the LLM fails to produce valid metadata after one retry.
        """
        guidelines = TAG_GUIDELINES.get(content_level, TAG_GUIDELINES["T2_implied"])

        user_prompt = _USER_PROMPT_TEMPLATE.format(
            theme=theme,
            mood=mood,
            environment=environment,
            character_name=character_name or "(no specific character)",
            content_level=content_level,
            image_count=image_count,
            style_keywords=style_keywords or "(none)",
            tag_guidelines=guidelines,
        )

        effective_temp = temperature if temperature is not None else self.TEMPERATURE

        # First attempt
        meta = self._attempt(user_prompt, effective_temp, model=model)
        if meta is not None:
            return meta

        # Retry once
        logger.warning("Metadata generator: first attempt failed, retrying …")
        retry_prompt = (
            user_prompt
            + "\n\nIMPORTANT: Your previous response was not valid JSON. "
            "Return ONLY a raw JSON object, no markdown, no commentary."
        )
        meta = self._attempt(retry_prompt, effective_temp, model=model)
        if meta is not None:
            return meta

        raise MetadataGeneratorError(
            "Failed to generate valid metadata after 2 attempts. "
            "Check Ollama logs and ensure the model supports structured output."
        )

    def _attempt(
        self, user_prompt: str, temperature: float | None = None,
        *, model: str | None = None,
    ) -> dict | None:
        """Single generate_json attempt. Returns validated dict or None."""
        try:
            result = self.llm.generate_json(
                SYSTEM_PROMPT,
                user_prompt,
                temperature=temperature if temperature is not None else self.TEMPERATURE,
                num_predict=2048,
                model=model,
            )
        except OllamaJSONParseError as exc:
            logger.warning("Metadata generator JSON parse error: %s", exc)
            return None

        if not isinstance(result, dict):
            logger.warning("Metadata generator: expected dict, got %s", type(result).__name__)
            return None

        missing = REQUIRED_FIELDS - result.keys()
        if missing:
            logger.warning("Metadata generator: missing required fields: %s", missing)
            return None

        # Validate title length
        title = result["title"]
        if len(title) > 80:
            logger.warning("Metadata: title too long (%d chars), truncating to 80", len(title))
            result["title"] = title[:77] + "…"

        # Validate tags
        tags = result["tags"]
        if not isinstance(tags, list):
            logger.warning("Metadata: tags is not a list")
            return None
        if len(tags) < 10:
            logger.warning("Metadata: only %d tags (want 15–25), accepting anyway", len(tags))
        elif len(tags) > 30:
            logger.info("Metadata: %d tags, trimming to 25", len(tags))
            result["tags"] = tags[:25]

        logger.info(
            "Metadata generated: title=%r (%d chars), %d tags",
            result["title"],
            len(result["title"]),
            len(result["tags"]),
        )
        return result
