"""Abstract base for all pipeline modes.

Each mode implements two LLM-phase methods:

  * ``plan(ctx, *, cli_llm_override=None)`` — produce a series plan
    (theme, mood, environment, variation axes). The returned dict is
    stored in ``series.llm_series_plan``.
  * ``generate_scenes(series_plan, ctx, *, cli_llm_override=None)`` —
    produce a list of scene dicts from the plan. Each scene must have
    the fields that ``PromptBuilder`` expects (see
    ``_SCENE_FIELD_ORDER`` in ``src/prompt/builder.py``).

``cli_llm_override`` flows from ``run_cycle`` / ``run_phase_a``: when
non-None, every role uses that LLM (full single-LLM run for clean A/B
comparison). When None, the mode resolves each role via the
:class:`LLMRouter` it received at construction (override → routing →
default chain).

Mode implementations live in sibling files (``character_mode.py``,
``theme_mode.py``, etc.) and are registered with the engine's
``mode_registry`` dict.

See ARCHITECTURE.md Section 4 (Module Map) and Sections 7–11 (Modes).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

from src.agents.llm_client import OllamaClient
from src.core.generation_context import GenerationContext

if TYPE_CHECKING:
    from src.agents.llm_router import LLMRouter


class BaseMode(ABC):
    """Interface contract for all pipeline modes.

    Subclasses must call ``super().__init__(llm_client, router)`` so the
    base captures both. Router can be ``None`` only in test contexts
    that don't exercise routing — production engine always passes a
    real :class:`LLMRouter`.
    """

    def __init__(
        self,
        llm_client: OllamaClient,
        router: "LLMRouter | None" = None,
    ) -> None:
        self.llm = llm_client
        self._router = router

    @property
    @abstractmethod
    def name(self) -> str:
        """Short mode identifier used in DB ``series.mode`` and logs."""
        ...

    @abstractmethod
    def plan(
        self,
        ctx: GenerationContext,
        *,
        cli_llm_override: str | None = None,
    ) -> dict[str, Any]:
        """Generate a series plan via the LLM.

        Returns a dict with at least: ``theme``, ``mood``, ``environment``,
        ``variation_axes`` (list of strings). May also include mode-specific
        keys (e.g. ``character_id`` for character mode).

        The engine stores this as JSON in ``series.llm_series_plan``.
        """
        ...

    @abstractmethod
    def generate_scenes(
        self,
        series_plan: dict[str, Any],
        ctx: GenerationContext,
        *,
        cli_llm_override: str | None = None,
    ) -> list[dict[str, Any]]:
        """Generate scene dicts from a series plan via the LLM.

        Each scene dict should have:
          ``variation_axis``, ``pose``, ``camera``, ``camera_angle``,
          ``lighting``, ``environment_detail``, ``mood_note``

        The engine passes these to the scene constraint enforcer and then
        to the prompt builder.
        """
        ...

    # ── helper for subclasses ────────────────────────────────────────
    def _resolve_role_model(
        self,
        role: str,
        *,
        cli_llm_override: str | None,
    ) -> str | None:
        """Resolve a role to its Ollama tag via the router.

        Returns ``None`` when the mode was constructed without a router
        (test-only path); callers that pass ``model=None`` to the LLM
        agent fall back to the client's default. Production paths always
        return a string.
        """
        if self._router is None:
            return None
        return self._router.resolve_role(
            role, override=cli_llm_override,
        ).ollama_id
