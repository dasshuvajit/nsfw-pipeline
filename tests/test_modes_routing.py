"""Tests for per-role LLM routing through mode orchestrators (F1).

Verifies that ``cli_llm_override`` flows through every mode's
``plan(ctx, *, cli_llm_override=...)`` and
``generate_scenes(plan, ctx, *, cli_llm_override=...)`` to the
underlying agents/helpers as the resolved Ollama tag.

Without this plumbing, only ``scene_facet_generator`` routing fired —
``metadata_generator``, ``series_planner``, ``scene_generator``, and
``character_creator`` silently used the registry's ``default_llm``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.agents.llm_client import OllamaClient
from src.agents.llm_router import LLMRouter
from src.memory.llm_registry import LLMRegistryLoader


@pytest.fixture
def router(llm_registry_yaml: Path) -> LLMRouter:
    """Router with `test_llm_b` set for series_planner + scene_generator
    so we can verify each role gets the routed model (not default)."""
    registry = LLMRegistryLoader(llm_registry_yaml)
    return LLMRouter(
        registry,
        routing_config={
            "series_planner": "test_llm_b",
            "scene_generator": "test_llm_b",
        },
    )


@pytest.fixture
def fake_client() -> OllamaClient:
    return OllamaClient()


def _ctx_stub(**overrides) -> MagicMock:
    """Minimal GenerationContext stub for mode tests."""
    ctx = MagicMock()
    ctx.content_level = overrides.get("content_level", "T2_implied")
    ctx.style_profile = overrides.get("style_profile", {
        "name": "test_profile",
        "base_style_keywords": "soft elegant",
    })
    ctx.content_rules = MagicMock()
    ctx.content_rules.allowed_pose_types = ["standing", "seated"]
    ctx.content_rules.raw_allowed_pose_types = '["standing"]'
    ctx.content_rules.raw_scene_constraints = "{}"
    ctx.content_rules.llm_directive = ""
    ctx.family = MagicMock()
    ctx.family.llm_temperature = 0.7
    ctx.character = overrides.get("character")
    ctx.character_id = overrides.get("character_id", "char_001")
    ctx.db_path = overrides.get("db_path", Path("/tmp/test.db"))
    ctx.model_prompt_guide = None
    ctx.augment_system_prompt = lambda s: s
    return ctx


# ── ThemeMode ────────────────────────────────────────────────────────
class TestThemeMode:
    def test_plan_uses_routing_when_no_override(
        self, fake_client, router, monkeypatch,
    ):
        from src.modes import theme_mode
        from src.modes.theme_mode import ThemeMode

        # Stub out _select_category and _load_recent_themes to avoid DB.
        monkeypatch.setattr(
            ThemeMode, "_select_category",
            staticmethod(lambda db_path: {"id": "x", "name": "x", "description": ""}),
        )
        monkeypatch.setattr(
            ThemeMode, "_load_recent_themes",
            staticmethod(lambda db_path, cat_id: []),
        )

        mode = ThemeMode(fake_client, router)
        with patch.object(theme_mode, "run_llm_with_retry") as helper:
            helper.return_value = {
                "theme": "x", "mood": "x", "environment": "x",
                "variation_axes": ["a"],
            }
            mode.plan(_ctx_stub(), cli_llm_override=None)
        # Routing entry → test_llm_b → ollama_id "test/llm-b:latest"
        assert helper.call_args.kwargs["model"] == "test/llm-b:latest"

    def test_plan_override_wins_over_routing(
        self, fake_client, router, monkeypatch,
    ):
        from src.modes import theme_mode
        from src.modes.theme_mode import ThemeMode

        monkeypatch.setattr(
            ThemeMode, "_select_category",
            staticmethod(lambda db_path: {"id": "x", "name": "x", "description": ""}),
        )
        monkeypatch.setattr(
            ThemeMode, "_load_recent_themes",
            staticmethod(lambda db_path, cat_id: []),
        )
        mode = ThemeMode(fake_client, router)
        with patch.object(theme_mode, "run_llm_with_retry") as helper:
            helper.return_value = {
                "theme": "x", "mood": "x", "environment": "x",
                "variation_axes": ["a"],
            }
            mode.plan(_ctx_stub(), cli_llm_override="test_llm_a")
        # Override → test_llm_a → "test/llm-a:latest"
        assert helper.call_args.kwargs["model"] == "test/llm-a:latest"

    def test_generate_scenes_uses_scene_generator_role(
        self, fake_client, router, monkeypatch,
    ):
        from src.modes import theme_mode
        from src.modes.theme_mode import ThemeMode

        mode = ThemeMode(fake_client, router)
        with patch.object(theme_mode, "run_llm_with_retry") as helper:
            helper.return_value = [{
                "variation_axis": "pose", "pose": "standing",
                "camera": "medium", "camera_angle": "eye level",
                "lighting": "soft", "environment_detail": "studio",
                "mood_note": "calm",
            }]
            plan = {"theme": "x", "mood": "x", "environment": "x",
                    "variation_axes": ["pose"], "subject_description": "x"}
            mode.generate_scenes(plan, _ctx_stub(), cli_llm_override=None)
        assert helper.call_args.kwargs["model"] == "test/llm-b:latest"

    def test_no_router_falls_back_to_none(self, fake_client, monkeypatch):
        """Back-compat: mode without router passes model=None
        (agent falls back to client.model)."""
        from src.modes import theme_mode
        from src.modes.theme_mode import ThemeMode

        monkeypatch.setattr(
            ThemeMode, "_select_category",
            staticmethod(lambda db_path: {"id": "x", "name": "x", "description": ""}),
        )
        monkeypatch.setattr(
            ThemeMode, "_load_recent_themes",
            staticmethod(lambda db_path, cat_id: []),
        )

        mode = ThemeMode(fake_client, router=None)
        with patch.object(theme_mode, "run_llm_with_retry") as helper:
            helper.return_value = {
                "theme": "x", "mood": "x", "environment": "x",
                "variation_axes": ["a"],
            }
            mode.plan(_ctx_stub(), cli_llm_override=None)
        assert helper.call_args.kwargs["model"] is None


# ── StyleMode ────────────────────────────────────────────────────────
class TestStyleMode:
    def test_plan_uses_routing(self, fake_client, router, monkeypatch):
        from src.modes import style_mode
        from src.modes.style_mode import StyleMode

        monkeypatch.setattr(
            StyleMode, "_select_category",
            staticmethod(lambda db_path: {
                "id": "x", "name": "x",
                "description": "", "lighting_bias": "", "color_bias": "",
            }),
        )
        monkeypatch.setattr(
            StyleMode, "_load_recent_styles",
            staticmethod(lambda db_path: []),
        )

        mode = StyleMode(fake_client, router)
        with patch.object(style_mode, "run_llm_with_retry") as helper:
            helper.return_value = {
                "theme": "x", "mood": "x", "environment": "x",
                "variation_axes": ["a"],
            }
            mode.plan(_ctx_stub(), cli_llm_override=None)
        assert helper.call_args.kwargs["model"] == "test/llm-b:latest"


# ── NicheMode ────────────────────────────────────────────────────────
class TestNicheMode:
    def test_plan_uses_routing(self, fake_client, router, monkeypatch):
        from src.modes import niche_mode
        from src.modes.niche_mode import NicheMode

        monkeypatch.setattr(
            NicheMode, "_select_cluster",
            staticmethod(lambda db_path: {
                "id": "x", "name": "x", "keywords": "[]",
            }),
        )
        monkeypatch.setattr(
            NicheMode, "_load_recent_themes",
            staticmethod(lambda db_path: []),
        )

        mode = NicheMode(fake_client, router)
        with patch.object(niche_mode, "run_llm_with_retry") as helper:
            helper.return_value = {
                "theme": "x", "mood": "x", "environment": "x",
                "variation_axes": ["a"], "visual_elements": ["x"],
            }
            mode.plan(_ctx_stub(), cli_llm_override=None)
        assert helper.call_args.kwargs["model"] == "test/llm-b:latest"


# ── CharacterMode ────────────────────────────────────────────────────
class TestCharacterMode:
    def test_plan_passes_model_to_planner(
        self, fake_client, router, monkeypatch,
    ):
        from src.modes.character_mode import CharacterMode

        mode = CharacterMode(fake_client, router)
        # Stub _load_recent_themes (needs DB).
        monkeypatch.setattr(
            CharacterMode, "_load_recent_themes",
            staticmethod(lambda db_path, character_id: []),
        )
        with patch.object(mode._planner, "plan") as planner_plan:
            planner_plan.return_value = {
                "theme": "x", "mood": "x", "environment": "x",
                "variation_axes": ["a"],
            }
            ctx = _ctx_stub(character={"id": "char_001", "base_prompt": "x"})
            mode.plan(ctx, cli_llm_override=None)
        assert planner_plan.call_args.kwargs["model"] == "test/llm-b:latest"

    def test_generate_scenes_passes_model_to_scene_gen(
        self, fake_client, router,
    ):
        from src.modes.character_mode import CharacterMode

        mode = CharacterMode(fake_client, router)
        with patch.object(mode._scene_gen, "generate") as scene_gen:
            scene_gen.return_value = []
            plan = {"theme": "x", "mood": "x", "environment": "x"}
            mode.generate_scenes(plan, _ctx_stub(), cli_llm_override=None)
        assert scene_gen.call_args.kwargs["model"] == "test/llm-b:latest"


# ── VariationMode ────────────────────────────────────────────────────
class TestVariationMode:
    def test_fallback_uses_scene_generator_routing(
        self, fake_client, router, monkeypatch,
    ):
        from src.modes.variation_mode import VariationMode

        mode = VariationMode(fake_client, router)
        # Force the fallback path (no past scene found).
        monkeypatch.setattr(
            VariationMode, "_select_base_scene",
            staticmethod(lambda db_path, content_level: None),
        )
        with patch.object(mode, "_attempt_fallback") as attempt:
            attempt.return_value = {
                "pose": "x", "camera": "x", "lighting": "x",
                "environment_detail": "x", "mood_note": "x",
            }
            mode.plan(_ctx_stub(), cli_llm_override=None)
        # scene_generator routing → test/llm-b:latest
        assert attempt.call_args.kwargs["model"] == "test/llm-b:latest"
