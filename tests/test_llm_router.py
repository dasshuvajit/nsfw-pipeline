"""Tests for ``src.agents.llm_router.LLMRouter``.

Covers (per plan §3.2 + Round-4 verifier patches):

  * ``for_role`` resolution chain (override → routing → default).
  * ``for_facet_family`` resolution chain (override → per-style →
    routing.default → default_llm) — explicit per Round-4 patch 2.
  * Unknown role / unknown prompt_style fall through to default_llm.
  * Startup-time validation rejects bad routing config.
  * ``format_resolution_table`` annotations (--llm override / routing /
    routing.default / default) — Round-4 patch 3.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agents.llm_router import (
    LLMRouter,
    SOURCE_CLI_OVERRIDE,
    SOURCE_DEFAULT,
    SOURCE_ROUTING,
    SOURCE_ROUTING_DEFAULT,
)
from src.memory.llm_registry import (
    LLMNotFound,
    LLMRegistryError,
    LLMRegistryLoader,
)


@pytest.fixture
def registry(llm_registry_yaml: Path) -> LLMRegistryLoader:
    return LLMRegistryLoader(llm_registry_yaml)


# ── for_role ─────────────────────────────────────────────────────────
class TestForRole:
    def test_override_wins_when_no_routing(self, registry):
        """CLI --llm override beats default_llm."""
        router = LLMRouter(registry, routing_config={})
        assert router.for_role("series_planner", override="test_llm_b") == \
            "test/llm-b:latest"

    def test_override_wins_over_routing(self, registry):
        """CLI --llm override beats explicit routing entry."""
        router = LLMRouter(
            registry, routing_config={"series_planner": "test_llm_b"},
        )
        assert router.for_role(
            "series_planner", override="test_llm_a",
        ) == "test/llm-a:latest"

    def test_routing_wins_when_no_override(self, registry):
        router = LLMRouter(
            registry, routing_config={"series_planner": "test_llm_b"},
        )
        assert router.for_role("series_planner") == "test/llm-b:latest"

    def test_default_when_no_routing_no_override(self, registry):
        router = LLMRouter(registry, routing_config={})
        # default_llm in fixture is test_llm_a → ollama_id test/llm-a:latest.
        assert router.for_role("series_planner") == "test/llm-a:latest"

    def test_unknown_role_falls_through_to_default(self, registry):
        """Round-4 patch 2: unknown role doesn't raise; falls to default_llm."""
        router = LLMRouter(
            registry, routing_config={"series_planner": "test_llm_b"},
        )
        assert router.for_role("not_a_real_role") == "test/llm-a:latest"

    def test_override_with_unknown_id_raises(self, registry):
        router = LLMRouter(registry, routing_config={})
        with pytest.raises(LLMNotFound, match="not_in_registry"):
            router.for_role("series_planner", override="not_in_registry")


# ── for_facet_family ─────────────────────────────────────────────────
class TestForFacetFamily:
    def test_override_wins(self, registry):
        router = LLMRouter(
            registry,
            routing_config={
                "scene_facet_generator": {"flux_natural": "test_llm_b"}
            },
        )
        assert router.for_facet_family(
            "flux_natural", override="test_llm_a",
        ) == "test/llm-a:latest"

    def test_per_style_routing_wins_when_no_override(self, registry):
        router = LLMRouter(
            registry,
            routing_config={
                "scene_facet_generator": {"flux_natural": "test_llm_b"}
            },
        )
        assert router.for_facet_family("flux_natural") == "test/llm-b:latest"

    def test_routing_default_used_when_no_per_style_entry(self, registry):
        """Round-4 patch 2: family-level default within scene_facet_generator
        block fires when no per-style entry is set."""
        router = LLMRouter(
            registry,
            routing_config={
                "scene_facet_generator": {"default": "test_llm_b"}
            },
        )
        assert router.for_facet_family("flux_natural") == "test/llm-b:latest"

    def test_per_style_overrides_routing_default(self, registry):
        router = LLMRouter(
            registry,
            routing_config={
                "scene_facet_generator": {
                    "default": "test_llm_a",
                    "flux_natural": "test_llm_b",
                }
            },
        )
        # flux_natural has its own entry → uses that.
        assert router.for_facet_family("flux_natural") == "test/llm-b:latest"
        # pony_danbooru has no entry → falls to default.
        assert router.for_facet_family("pony_danbooru") == "test/llm-a:latest"

    def test_no_routing_block_falls_to_default_llm(self, registry):
        router = LLMRouter(registry, routing_config={})
        assert router.for_facet_family("flux_natural") == "test/llm-a:latest"

    def test_unknown_prompt_style_falls_through_to_default(self, registry):
        """Round-4 patch 2: unknown prompt_style → default_llm."""
        router = LLMRouter(
            registry,
            routing_config={
                "scene_facet_generator": {"flux_natural": "test_llm_b"}
            },
        )
        assert router.for_facet_family("not_a_real_style") == \
            "test/llm-a:latest"

    def test_override_with_unknown_id_raises(self, registry):
        router = LLMRouter(registry, routing_config={})
        with pytest.raises(LLMNotFound):
            router.for_facet_family("flux_natural", override="missing")


# ── default + fallback ────────────────────────────────────────────────
class TestDefault:
    def test_default_returns_registry_default(self, registry):
        router = LLMRouter(registry, routing_config={})
        assert router.default() == "test/llm-a:latest"

    def test_fallback_currently_aliases_default(self, registry):
        """Phase 11 will split fallback from default; for now they match."""
        router = LLMRouter(registry, routing_config={})
        assert router.fallback() == router.default()


# ── startup-time validation ──────────────────────────────────────────
class TestValidation:
    def test_routing_role_pointing_to_missing_id_rejected(self, registry):
        with pytest.raises(LLMRegistryError, match="not a valid active"):
            LLMRouter(
                registry,
                routing_config={"series_planner": "not_in_registry"},
            )

    def test_routing_role_pointing_to_inactive_id_rejected(self, registry):
        with pytest.raises(LLMRegistryError, match="not a valid active"):
            LLMRouter(
                registry,
                routing_config={"series_planner": "test_llm_inactive"},
            )

    def test_facet_per_style_pointing_to_missing_rejected(self, registry):
        with pytest.raises(LLMRegistryError, match="not a valid active"):
            LLMRouter(
                registry,
                routing_config={
                    "scene_facet_generator": {"flux_natural": "missing_id"}
                },
            )

    def test_facet_default_pointing_to_inactive_rejected(self, registry):
        with pytest.raises(LLMRegistryError, match="not a valid active"):
            LLMRouter(
                registry,
                routing_config={
                    "scene_facet_generator": {"default": "test_llm_inactive"}
                },
            )

    def test_routing_role_with_non_string_value_rejected(self, registry):
        with pytest.raises(LLMRegistryError, match="non-empty registry id"):
            LLMRouter(
                registry, routing_config={"series_planner": 123},
            )

    def test_facet_block_must_be_mapping(self, registry):
        with pytest.raises(LLMRegistryError, match="must be a mapping"):
            LLMRouter(
                registry,
                routing_config={"scene_facet_generator": "not_a_dict"},
            )

    def test_empty_routing_validates_clean(self, registry):
        # Just doesn't raise.
        LLMRouter(registry, routing_config={})
        LLMRouter(registry, routing_config=None)


# ── format_resolution_table ──────────────────────────────────────────
class TestFormatResolutionTable:
    def test_table_renders_with_default_when_no_routing_no_override(
        self, registry,
    ):
        router = LLMRouter(registry, routing_config={})
        table = router.format_resolution_table(cli_llm_override=None)
        assert "Resolved LLMs for this run:" in table
        assert "series_planner" in table
        assert "test_llm_a" in table
        # Every row is annotated `(default)` when no routing applies.
        assert SOURCE_DEFAULT in table

    def test_table_renders_with_routing_annotation(self, registry):
        router = LLMRouter(
            registry,
            routing_config={
                "series_planner": "test_llm_b",
                "scene_facet_generator": {"flux_natural": "test_llm_b"},
            },
        )
        table = router.format_resolution_table(cli_llm_override=None)
        assert SOURCE_ROUTING in table
        assert "test_llm_b" in table

    def test_table_renders_with_routing_default_annotation(self, registry):
        router = LLMRouter(
            registry,
            routing_config={
                "scene_facet_generator": {"default": "test_llm_b"}
            },
        )
        table = router.format_resolution_table(cli_llm_override=None)
        assert SOURCE_ROUTING_DEFAULT in table

    def test_table_renders_with_cli_override_annotation(self, registry):
        router = LLMRouter(
            registry,
            routing_config={"series_planner": "test_llm_b"},
        )
        table = router.format_resolution_table(cli_llm_override="test_llm_a")
        # Override fires for every row.
        assert SOURCE_CLI_OVERRIDE in table
        # And the routing-was annotation shows what was overridden.
        assert "routing was: test_llm_b" in table

    def test_table_shows_facet_routing_was_for_per_style_override(
        self, registry,
    ):
        router = LLMRouter(
            registry,
            routing_config={
                "scene_facet_generator": {"flux_natural": "test_llm_b"}
            },
        )
        table = router.format_resolution_table(cli_llm_override="test_llm_a")
        # The flux_natural row shows what was overridden.
        assert "scene_facet_generator.flux_natural" in table
        assert "routing was: test_llm_b" in table

    def test_table_shows_facet_default_was_for_routing_default_override(
        self, registry,
    ):
        router = LLMRouter(
            registry,
            routing_config={
                "scene_facet_generator": {"default": "test_llm_b"}
            },
        )
        table = router.format_resolution_table(cli_llm_override="test_llm_a")
        # Each per-style row falls back to "default" path; override
        # annotation should mention the default that was overridden.
        assert "test_llm_b (default)" in table
