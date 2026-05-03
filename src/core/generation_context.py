"""Generation context — the bag of state passed to every pipeline component.

``GenerationContext`` is built once at the top of
``PipelineEngine.run_cycle()`` and threaded through every method call
that needs to know "what are we rendering?". It resolves the model via
``ModelRegistryLoader`` (YAML-backed since the 2026-04 refactor) and
attaches the ``FamilyConfig`` so prompt composers, capability checks,
and LLM hint generation all read from a single authoritative source.

See ARCHITECTURE.md §4 (Generation Context).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.content_level import ContentLevelRules
from src.memory.family_loader import FamilyConfig
from src.memory.model_registry import (
    ModelPromptGuide,
    ModelRegistryEntry,
    ModelRegistryLoader,
)


@dataclass
class GenerationContext:
    """Immutable-ish context for one pipeline cycle."""

    mode: str                           # 'character'|'theme'|'style'|'niche'|'variation'
    content_level: str                  # 'T1_suggestive'|'T2_implied'|'T3_artnude'|'T4_explicit'
    execution_mode: str                 # 'manual'|'supervised'|'automated'

    style_profile: dict[str, Any]
    content_rules: ContentLevelRules

    model_id: str
    model_config: ModelRegistryEntry
    family: FamilyConfig                # dereferenced from model_config.family

    model_prompt_guide: ModelPromptGuide | None = None

    character: dict[str, Any] | None = None
    character_id: str | None = None

    db_path: Path | None = None

    # ── shortcuts ──────────────────────────────────────────────────
    @property
    def workflow_family(self) -> str:
        """Deprecated alias — prefer ``ctx.family.id`` or ``ctx.model_config.family``."""
        return self.model_config.family

    @property
    def supports_ipadapter(self) -> bool:
        return self.model_config.supports_ipadapter

    @property
    def supports_lora(self) -> bool:
        return self.model_config.supports_lora

    @property
    def prompt_style(self) -> str:
        return self.family.prompt_style

    @property
    def supports_negative_prompt(self) -> bool:
        # Per-model override takes precedence over family default.
        if self.model_prompt_guide is not None:
            return self.model_prompt_guide.supports_negative_prompt
        return self.family.supports_negative_prompt

    def augment_system_prompt(self, base_prompt: str) -> str:
        """Append model-awareness rules to an LLM system prompt.

        Pulls structure_rules, trigger_words, avoid_words, example_prompt,
        and the family's ``llm_hint`` — so the LLM gets one consistent
        brief regardless of model family.
        """
        parts: list[str] = []
        guide = self.model_prompt_guide
        family = self.family

        if family.llm_hint:
            parts.append(f"\nMODEL FAMILY HINT:\n{family.llm_hint}")

        structure = (
            (guide.structure_rules if guide else None)
            or family.structure_rules
            or None
        )
        if structure:
            parts.append(f"\nMODEL PROMPTING RULES:\n{structure}")

        if guide and guide.trigger_words:
            parts.append(
                "\nTRIGGER WORDS (use naturally when they fit):\n"
                + ", ".join(guide.trigger_words)
            )

        avoid = list(family.avoid_words)
        if guide:
            for w in guide.avoid_words:
                if w not in avoid:
                    avoid.append(w)
        if avoid:
            parts.append(
                "\nAVOID these words/phrases:\n" + ", ".join(avoid)
            )

        example = (
            (guide.example_prompt if guide else None)
            or family.example_prompt
            or None
        )
        if example:
            parts.append(f"\nEXAMPLE prompt for the target model:\n{example}")

        return base_prompt + "".join(parts) if parts else base_prompt


def build_context(
    *,
    mode: str,
    content_level: str,
    execution_mode: str,
    style_profile: dict[str, Any],
    content_rules: ContentLevelRules,
    db_path: Path,
    model_id: str,
    model_override: str | None = None,
    commercial_mode: bool = False,
) -> GenerationContext:
    """Construct a ``GenerationContext``, resolving the model + family.

    ``model_id`` is the baseline pick — from the character row, or from
    ``pipeline.default_model_id`` for non-character modes. ``model_override``
    (CLI ``--model``) wins when set. Post-2026-04 the style profile no
    longer carries a model_id; aesthetic intent and render tuning are
    decoupled.

    ``commercial_mode`` is threaded through from
    ``pipeline.yaml::compliance.commercial_mode``. When true, the model
    registry drops any entry with ``commercial_use: false``; resolving
    such an id raises ``ModelNotFound``.
    """
    loader = ModelRegistryLoader(db_path, commercial_mode=commercial_mode)

    effective_model_id = model_override or model_id
    model_config = loader.get_model(effective_model_id)
    family = loader.get_family(model_config.family)
    prompt_guide = loader.get_prompt_guide(effective_model_id)

    return GenerationContext(
        mode=mode,
        content_level=content_level,
        execution_mode=execution_mode,
        style_profile=style_profile,
        content_rules=content_rules,
        model_id=effective_model_id,
        model_config=model_config,
        family=family,
        model_prompt_guide=prompt_guide,
        db_path=db_path,
    )
