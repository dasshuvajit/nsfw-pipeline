"""Generation context — the bag of state passed to every pipeline component.

``GenerationContext`` is built once at the top of
``PipelineEngine.run_cycle()`` and threaded through every method call
that needs to know "what are we rendering?". It resolves the model via
``ModelRegistryLoader`` (YAML-backed since the 2026-04 refactor) and
attaches the ``FamilyConfig`` so prompt composers, capability checks,
and LLM hint generation all read from a single authoritative source.

Two construction paths since the family-level prompt-prep feature
(2026-05):

* :func:`build_context` — model-kind ctx (``target_kind='model'``).
  Used by every existing call path; full per-model overlay applies.
* :func:`build_family_context` — family-kind ctx (``target_kind='family'``).
  Used by ``run_phase_a``'s per-target loop when preparing prompts
  with ``--families <family>``. ``model_id`` and ``model_config`` are
  ``None``; ``model_prompt_guide`` is the family-only one (no
  per-model trigger / avoid / negative_embeddings overlay).

See ARCHITECTURE.md §4 (Generation Context).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from src.core.content_level import ContentLevelRules
from src.memory.family_loader import FamilyConfig
from src.memory.model_registry import (
    ModelPromptGuide,
    ModelRegistryEntry,
    ModelRegistryLoader,
)


@dataclass
class GenerationContext:
    """Immutable-ish context for one pipeline cycle.

    ``target_kind`` discriminates between two preparation modes:

    * ``'model'`` — per-model prompt prep (existing default). All
      fields populated; ``model_id``, ``model_config`` non-None.
    * ``'family'`` — family-level prompt prep. ``model_id`` and
      ``model_config`` are None; ``family`` is the active
      ``FamilyConfig``; ``model_prompt_guide`` is the family-only
      guide (no per-model overlay).

    The ``__post_init__`` invariant guards consistency: a
    ``target_kind='family'`` ctx with ``model_id`` set, or a
    ``target_kind='model'`` ctx with ``model_id=None``, fails fast.
    """

    mode: str                           # 'character'|'theme'|'style'|'niche'|'variation'
    content_level: str                  # 'T1_suggestive'|'T2_implied'|'T3_artnude'|'T4_explicit'
    execution_mode: str                 # 'manual'|'supervised'|'automated'

    style_profile: dict[str, Any]
    content_rules: ContentLevelRules

    model_id: str | None
    model_config: ModelRegistryEntry | None
    family: FamilyConfig                # always present — derived from model or supplied directly

    model_prompt_guide: ModelPromptGuide | None = None

    character: dict[str, Any] | None = None
    character_id: str | None = None

    db_path: Path | None = None

    # ── target discriminator (added 2026-05 for family-level prep) ──
    target_kind: Literal["model", "family"] = "model"

    def __post_init__(self) -> None:
        """Enforce target_kind ↔ model_id/model_config consistency.

        Per the family-level plan (D3): a `target_kind='family'` ctx
        must have `model_id is None` AND `model_config is None`; a
        `target_kind='model'` ctx must have both populated. Catches
        a future caller accidentally creating an inconsistent ctx
        that would flip behaviour mid-pipeline.
        """
        if self.target_kind == "family":
            if self.model_id is not None or self.model_config is not None:
                raise ValueError(
                    f"GenerationContext invariant: target_kind='family' "
                    f"requires model_id=None and model_config=None; got "
                    f"model_id={self.model_id!r}, "
                    f"model_config={'<set>' if self.model_config else None}"
                )
            if self.family is None:
                raise ValueError(
                    "GenerationContext invariant: target_kind='family' "
                    "requires a non-None family"
                )
        elif self.target_kind == "model":
            if self.model_id is None or self.model_config is None:
                raise ValueError(
                    f"GenerationContext invariant: target_kind='model' "
                    f"requires model_id and model_config to be non-None; "
                    f"got model_id={self.model_id!r}, "
                    f"model_config={'<set>' if self.model_config else None}"
                )
        else:
            raise ValueError(
                f"GenerationContext invariant: target_kind must be "
                f"'model' or 'family'; got {self.target_kind!r}"
            )

    # ── shortcuts ──────────────────────────────────────────────────
    @property
    def workflow_family(self) -> str:
        """Deprecated alias — prefer ``ctx.family.id`` or
        ``ctx.model_config.family``. For family-kind ctx, returns
        ``self.family.id`` directly.
        """
        if self.model_config is None:
            return self.family.id
        return self.model_config.family

    @property
    def supports_ipadapter(self) -> bool:
        if self.model_config is None:
            raise AttributeError(
                "supports_ipadapter requires a model-kind GenerationContext; "
                "got family-kind (model_config is None). IPAdapter is a "
                "per-model checkpoint capability and family-level prompt "
                "preparation is checkpoint-agnostic."
            )
        return self.model_config.supports_ipadapter

    @property
    def supports_lora(self) -> bool:
        if self.model_config is None:
            raise AttributeError(
                "supports_lora requires a model-kind GenerationContext; "
                "got family-kind (model_config is None). LoRA support is "
                "a per-model checkpoint capability."
            )
        return self.model_config.supports_lora

    @property
    def prompt_style(self) -> str:
        return self.family.prompt_style

    @property
    def supports_negative_prompt(self) -> bool:
        # Per-model override takes precedence over family default.
        # For family-kind ctx, the family-only guide carries the
        # family's own value (see ModelRegistryLoader.get_family_only_prompt_guide).
        if self.model_prompt_guide is not None:
            return self.model_prompt_guide.supports_negative_prompt
        return self.family.supports_negative_prompt

    def augment_system_prompt(self, base_prompt: str) -> str:
        """Append model-awareness rules to an LLM system prompt.

        Pulls structure_rules, trigger_words, avoid_words, example_prompt,
        and the family's ``llm_hint`` — so the LLM gets one consistent
        brief regardless of model family. For family-kind ctx, per-model
        fields (trigger_words, avoid_words, example_prompt) are empty
        so only family-level guidance flows through.
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
    """Construct a model-kind ``GenerationContext``.

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
        target_kind="model",
    )


def build_family_context(
    *,
    family_id: str,
    mode: str,
    content_level: str,
    execution_mode: str,
    style_profile: dict[str, Any],
    content_rules: ContentLevelRules,
    db_path: Path,
    commercial_mode: bool = False,
    character: dict[str, Any] | None = None,
    character_id: str | None = None,
) -> GenerationContext:
    """Construct a family-kind ``GenerationContext`` for family-level
    prompt preparation.

    Used inside ``PipelineEngine.run_phase_a``'s per-target loop when
    the user runs ``prepare_prompts --families <family_id>``. The
    resulting ctx has:

    * ``target_kind='family'``
    * ``model_id=None``
    * ``model_config=None``
    * ``family`` resolved from ``config/families.yaml``
    * ``model_prompt_guide`` built via
      :meth:`ModelRegistryLoader.get_family_only_prompt_guide` — no
      per-model trigger / avoid / negative_embeddings overlay.

    Reading checkpoint-only attributes (``supports_ipadapter``,
    ``supports_lora``) on the returned ctx raises ``AttributeError``
    with a clear message. Phase A's per-target loop never reads those.

    Raises ``FamilyNotFound`` if ``family_id`` is not in
    ``config/families.yaml``.
    """
    loader = ModelRegistryLoader(db_path, commercial_mode=commercial_mode)
    family = loader.get_family(family_id)
    prompt_guide = loader.get_family_only_prompt_guide(family_id)

    return GenerationContext(
        mode=mode,
        content_level=content_level,
        execution_mode=execution_mode,
        style_profile=style_profile,
        content_rules=content_rules,
        model_id=None,
        model_config=None,
        family=family,
        model_prompt_guide=prompt_guide,
        db_path=db_path,
        character=character,
        character_id=character_id,
        target_kind="family",
    )
