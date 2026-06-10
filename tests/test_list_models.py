"""Tests for ``scripts/list_models.py``.

LLM-registry-only since 2026-06-10 — the image-model registry half was
archived with the legacy structured path (``legacy/config/models/``); the
active render path hardcodes its checkpoint in the workflow templates.
Uses subprocess against the real on-disk registry so the smoke covers actual
command-line behaviour.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(*flags: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/list_models.py", *flags],
        capture_output=True,
        cwd=REPO_ROOT,
        text=True,
    )


class TestDefaultOutput:
    def test_default_prints_llm_registry(self):
        result = _run()
        assert result.returncode == 0
        assert "LLM registry" in result.stdout
        assert "cydonia_heretic_24b" in result.stdout
        # The archived image-model section is gone.
        assert "Image models" not in result.stdout
        # Default should NOT print routing table.
        assert "Resolved LLMs by role" not in result.stdout

    def test_includes_default_marker(self):
        result = _run()
        # default column shows 'Y' for the configured default_llm
        # (2026-06-11 — DECKARD Gemma-4 31B dense replaced the 26B-A4B MoE
        # after the W5 blind A/B; Cydonia is still the registered fallback).
        assert "deckard_gemma4_31b_heretic" in result.stdout
        assert "default = 'deckard_gemma4_31b_heretic'" in result.stdout

    def test_llms_only_is_retained_noop(self):
        result = _run("--llms-only")
        assert result.returncode == 0
        assert "LLM registry" in result.stdout


class TestRouting:
    def test_routing_prints_resolution_table(self):
        result = _run("--routing")
        assert result.returncode == 0
        assert "Resolved LLMs by role" in result.stdout
        # Forward table includes Source column header
        assert "Source" in result.stdout
        # Includes every known role
        for role in (
            "series_planner",
            "scene_generator",
            "metadata_generator",
        ):
            assert role in result.stdout
        # Includes facet styles
        assert "scene_facet_generator.flux_natural" in result.stdout

    def test_routing_prints_reverse_mapping(self):
        result = _run("--routing")
        assert result.returncode == 0
        assert "Reverse mapping" in result.stdout
        # The fallback LLM should appear in reverse mapping with annotations
        assert "cydonia_heretic_24b" in result.stdout or \
            "deckard_gemma4_31b_heretic" in result.stdout
        # Default annotation visible somewhere
        assert "default" in result.stdout
