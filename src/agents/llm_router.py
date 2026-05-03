"""LLM Router — resolution chain for "which LLM does this call use?".

Three-layer resolution model (see plan §3.2):

  1. **CLI ``--llm <id>`` override.** When provided, wins for every
     role on this command run (full pipeline uses one LLM — clean
     A/B test).
  2. **Per-role routing in ``pipeline.yaml::llm.routing``.** Fires
     when ``--llm`` is omitted. Empty by default; user opts in for
     per-family quality optimisation.
  3. **``llm_models.yaml::default_llm``.** Universal fallback when
     no routing entry exists for the resolved role.

The router resolves registry IDs (e.g. ``cydonia_24b_v43``) through
:class:`LLMRegistryLoader` to obtain the Ollama tag (``ollama_id``)
that ``OllamaClient.generate(model=...)`` consumes. Single source of
truth: editing the Ollama tag in ``llm_models.yaml`` propagates to
every routing entry automatically.

Validation timing:

  * ``__init__`` validates that every routing target id resolves to
    an active registry entry. Raises :class:`LLMRegistryError` at
    startup so a typo in ``pipeline.yaml::llm.routing.*`` fails
    fast (before any agent fires).
  * ``--llm`` validation happens at the CLI boundary via
    ``registry.get_llm(id)`` — no router involvement.
  * **No call-time validation.** Once the engine is constructed,
    every resolved id is known-good; downstream calls trust the
    router output.
"""

from __future__ import annotations

import logging
from typing import Any

from src.memory.llm_registry import (
    LLMRegistryError,
    LLMRegistryLoader,
    LLMNotFound,
)

logger = logging.getLogger(__name__)


# Roles the router knows about. Adding a role: extend this set + add
# a routing entry name to pipeline.yaml's recommended block.
KNOWN_ROLES: frozenset[str] = frozenset({
    "series_planner",
    "scene_generator",
    "scene_facet_generator",
    "metadata_generator",
    "character_creator",
})


# Source-of-resolution annotations used by `format_resolution_table` and
# in the router's logging. The strings are user-facing (printed in the
# resolution table), so a rename here is a minor breaking change for
# anyone scraping CLI output.
SOURCE_CLI_OVERRIDE = "--llm override"
SOURCE_ROUTING = "routing"
SOURCE_ROUTING_DEFAULT = "routing.default"  # falls back to scene_facet_generator.default
SOURCE_DEFAULT = "default"


class LLMRouter:
    """Resolves role / family → Ollama tag via the resolution chain.

    Constructed once per engine cycle. ``cli_llm_override`` is passed
    to each ``for_role`` / ``for_facet_family`` call so the same router
    instance can serve both override and non-override resolutions.

    Parameters
    ----------
    registry : LLMRegistryLoader
        The loaded LLM registry (from ``config/llm_models.yaml``).
    routing_config : dict[str, Any]
        The ``llm.routing`` block from ``pipeline.yaml`` (may be empty
        — empty means "use default_llm for everything").
    """

    def __init__(
        self,
        registry: LLMRegistryLoader,
        routing_config: dict[str, Any] | None = None,
    ) -> None:
        self._registry = registry
        self._routing = dict(routing_config or {})
        self._validate_routing()

    # ── public API ────────────────────────────────────────────────────
    def for_role(
        self, role: str, *, override: str | None = None,
    ) -> str:
        """Return the Ollama tag for ``role`` honouring the resolution chain.

        ``override`` is the CLI ``--llm <id>`` value (or None). When
        provided, wins for every role.

        Unknown roles fall through to ``default_llm`` (the registry's
        default) — they don't raise. This matches the "graceful add a
        new role later" pattern: a new role gets the project default
        until someone adds a routing entry for it.
        """
        return self._resolve(role, override=override).ollama_id

    def for_facet_family(
        self, prompt_style: str, *, override: str | None = None,
    ) -> str:
        """Return the Ollama tag for the SceneFacetGenerator's
        per-family routing.

        Resolution order:
          1. ``override`` (CLI --llm) — wins.
          2. ``routing.scene_facet_generator.<prompt_style>`` if set.
          3. ``routing.scene_facet_generator.default`` if set.
          4. ``default_llm`` (registry).

        ``prompt_style`` matches the family's ``prompt_style`` field
        (e.g. ``"flux_natural"``, ``"pony_danbooru"``).
        """
        return self._resolve_facet(
            prompt_style, override=override,
        ).ollama_id

    def resolve_role(
        self, role: str, *, override: str | None = None,
    ):
        """Like :meth:`for_role` but returns the full LLMRegistryEntry.

        Engine consumers need both ``entry.id`` (the registry id stored
        in DB rows) and ``entry.ollama_id`` (the tag passed to
        OllamaClient) — exposing the entry avoids two lookup calls.
        """
        return self._resolve(role, override=override)

    def resolve_facet_family(
        self, prompt_style: str, *, override: str | None = None,
    ):
        """Like :meth:`for_facet_family` but returns the LLMRegistryEntry."""
        return self._resolve_facet(prompt_style, override=override)

    def default(self) -> str:
        """Return the Ollama tag for ``default_llm``."""
        return self._registry.get_default_llm().ollama_id

    def fallback(self) -> str:
        """Return the Ollama tag for the registry default.

        Phase 11 will introduce a separate fallback dimension. For now,
        ``fallback() == default()``.
        """
        return self.default()

    # ── resolution-display logging ───────────────────────────────────
    def format_resolution_table(
        self, cli_llm_override: str | None,
    ) -> str:
        """Render the resolution table for this run (plan §3.2c).

        Logged to stderr at the top of every LLM-using CLI invocation
        so the user sees exactly which LLM each role resolves to —
        and, when ``--llm`` overrides routing, what the routing-block
        value would have been.

        Sample output (CLI override active):

            Resolved LLMs for this run:
              series_planner                    → cydonia_24b_v43   (--llm override; routing was: cydonia_24b_v43)
              scene_facet_generator.flux_natural → cydonia_24b_v43   (--llm override; routing was: magnum_v4_22b)
              ...
        """
        rows = self._build_resolution_rows(cli_llm_override)
        if not rows:
            return ""
        # Pad columns for alignment.
        role_w = max(len(r[0]) for r in rows)
        llm_w = max(len(r[1]) for r in rows)
        lines = ["Resolved LLMs for this run:"]
        for role_label, llm_id, annotation in rows:
            lines.append(
                f"  {role_label.ljust(role_w)} → {llm_id.ljust(llm_w)}  "
                f"({annotation})"
            )
        return "\n".join(lines)

    # ── internals ────────────────────────────────────────────────────
    def _resolve(self, role: str, *, override: str | None):
        if override is not None:
            return self._registry.get_llm(override, require_active=True)
        rt = self._routing.get(role)
        if isinstance(rt, str) and rt:
            return self._registry.get_llm(rt, require_active=True)
        # Unknown role / no routing entry → default_llm.
        return self._registry.get_default_llm()

    def _resolve_facet(self, prompt_style: str, *, override: str | None):
        if override is not None:
            return self._registry.get_llm(override, require_active=True)
        sfg = self._routing.get("scene_facet_generator") or {}
        if not isinstance(sfg, dict):
            return self._registry.get_default_llm()
        per_style = sfg.get(prompt_style)
        if isinstance(per_style, str) and per_style:
            return self._registry.get_llm(per_style, require_active=True)
        per_default = sfg.get("default")
        if isinstance(per_default, str) and per_default:
            return self._registry.get_llm(per_default, require_active=True)
        return self._registry.get_default_llm()

    def _resolve_facet_with_source(
        self, prompt_style: str, *, override: str | None,
    ) -> tuple[str, str, str]:
        """Same as ``_resolve_facet`` but also returns the (id, source).

        Internal helper for ``format_resolution_table`` — exposes which
        layer of the chain produced the value so the table can annotate
        each row.
        """
        if override is not None:
            entry = self._registry.get_llm(override, require_active=True)
            return entry.ollama_id, entry.id, SOURCE_CLI_OVERRIDE
        sfg = self._routing.get("scene_facet_generator") or {}
        if isinstance(sfg, dict):
            per_style = sfg.get(prompt_style)
            if isinstance(per_style, str) and per_style:
                entry = self._registry.get_llm(per_style, require_active=True)
                return entry.ollama_id, entry.id, SOURCE_ROUTING
            per_default = sfg.get("default")
            if isinstance(per_default, str) and per_default:
                entry = self._registry.get_llm(per_default, require_active=True)
                return entry.ollama_id, entry.id, SOURCE_ROUTING_DEFAULT
        entry = self._registry.get_default_llm()
        return entry.ollama_id, entry.id, SOURCE_DEFAULT

    def _resolve_role_with_source(
        self, role: str, *, override: str | None,
    ) -> tuple[str, str, str]:
        if override is not None:
            entry = self._registry.get_llm(override, require_active=True)
            return entry.ollama_id, entry.id, SOURCE_CLI_OVERRIDE
        rt = self._routing.get(role)
        if isinstance(rt, str) and rt:
            entry = self._registry.get_llm(rt, require_active=True)
            return entry.ollama_id, entry.id, SOURCE_ROUTING
        entry = self._registry.get_default_llm()
        return entry.ollama_id, entry.id, SOURCE_DEFAULT

    def _build_resolution_rows(
        self, cli_llm_override: str | None,
    ) -> list[tuple[str, str, str]]:
        """Build a list of (role_label, registry_id, annotation) tuples."""
        rows: list[tuple[str, str, str]] = []

        # Single-role agents.
        for role in (
            "series_planner",
            "scene_generator",
            "metadata_generator",
            "character_creator",
        ):
            _, llm_id, source = self._resolve_role_with_source(
                role, override=cli_llm_override,
            )
            annotation = self._format_annotation(
                role, source, cli_llm_override,
            )
            rows.append((role, llm_id, annotation))

        # SceneFacetGenerator — one row per known prompt_style under
        # scene_facet_generator routing (plus default).
        sfg_styles = ["sdxl_keywords", "pony_danbooru", "illustrious_tags",
                      "flux_natural", "flux2_prose", "chroma_natural"]
        for style in sfg_styles:
            _, llm_id, source = self._resolve_facet_with_source(
                style, override=cli_llm_override,
            )
            annotation = self._format_facet_annotation(
                style, source, cli_llm_override,
            )
            rows.append((f"scene_facet_generator.{style}", llm_id, annotation))

        return rows

    def _format_annotation(
        self, role: str, source: str, cli_override: str | None,
    ) -> str:
        if source == SOURCE_CLI_OVERRIDE:
            # Show what routing WOULD have produced (so the user sees
            # what they're overriding).
            rt = self._routing.get(role)
            if isinstance(rt, str) and rt:
                return f"{source}; routing was: {rt}"
            return source
        return source

    def _format_facet_annotation(
        self, prompt_style: str, source: str, cli_override: str | None,
    ) -> str:
        if source == SOURCE_CLI_OVERRIDE:
            sfg = self._routing.get("scene_facet_generator") or {}
            if isinstance(sfg, dict):
                per_style = sfg.get(prompt_style)
                if isinstance(per_style, str) and per_style:
                    return f"{source}; routing was: {per_style}"
                per_default = sfg.get("default")
                if isinstance(per_default, str) and per_default:
                    return f"{source}; routing was: {per_default} (default)"
            return source
        return source

    # ── startup-time validation ──────────────────────────────────────
    def _validate_routing(self) -> None:
        """Reject routing config that points to missing/inactive entries.

        Catches typos in ``pipeline.yaml::llm.routing`` at engine
        startup so the operator gets a clear error before any LLM call
        fires.
        """
        for role, value in self._routing.items():
            if role == "scene_facet_generator":
                if not isinstance(value, dict):
                    raise LLMRegistryError(
                        f"pipeline.yaml::llm.routing.scene_facet_generator "
                        f"must be a mapping (got {type(value).__name__})"
                    )
                for style, llm_id in value.items():
                    if not isinstance(llm_id, str) or not llm_id:
                        raise LLMRegistryError(
                            f"pipeline.yaml::llm.routing."
                            f"scene_facet_generator.{style} must be a "
                            f"non-empty registry id"
                        )
                    self._validate_target(
                        f"scene_facet_generator.{style}", llm_id,
                    )
                continue
            if not isinstance(value, str) or not value:
                raise LLMRegistryError(
                    f"pipeline.yaml::llm.routing.{role} must be a "
                    f"non-empty registry id (got {value!r})"
                )
            self._validate_target(role, value)

    def _validate_target(self, route_name: str, llm_id: str) -> None:
        try:
            self._registry.get_llm(llm_id, require_active=True)
        except LLMNotFound as exc:
            raise LLMRegistryError(
                f"pipeline.yaml::llm.routing.{route_name} -> {llm_id!r} "
                f"is not a valid active registry id.\n{exc}"
            ) from exc
