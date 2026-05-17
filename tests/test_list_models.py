"""Tests for ``scripts/list_models.py`` (F2 extension).

Exercises the new flags: ``--llms-only``, ``--models-only``, ``--routing``.
Uses subprocess to invoke main() against the real on-disk registries
so the smoke covers actual command-line behaviour.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(*flags: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/list_models.py", *flags],
        capture_output=True,
        cwd=REPO_ROOT,
        text=True,
    )


class TestDefaultOutput:
    def test_default_prints_both_registries(self):
        result = _run()
        assert result.returncode == 0
        assert "Image models" in result.stdout
        assert "LLM registry" in result.stdout
        assert "cydonia_heretic_24b" in result.stdout
        # Default should NOT print routing table.
        assert "Resolved LLMs by role" not in result.stdout


class TestLlmsOnly:
    def test_skips_image_models(self):
        result = _run("--llms-only")
        assert result.returncode == 0
        assert "LLM registry" in result.stdout
        assert "Image models" not in result.stdout

    def test_includes_default_marker(self):
        result = _run("--llms-only")
        # default column shows 'Y' for the configured default_llm
        assert "cydonia_heretic_24b" in result.stdout
        assert "default = 'cydonia_heretic_24b'" in result.stdout


class TestModelsOnly:
    def test_skips_llm_registry(self):
        result = _run("--models-only")
        assert result.returncode == 0
        assert "Image models" in result.stdout
        assert "LLM registry" not in result.stdout

    def test_mutex_with_llms_only(self):
        result = _run("--models-only", "--llms-only")
        assert result.returncode == 2
        assert "mutually exclusive" in result.stderr


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
            "character_creator",
        ):
            assert role in result.stdout
        # Includes facet styles
        assert "scene_facet_generator.flux_natural" in result.stdout

    def test_routing_prints_reverse_mapping(self):
        result = _run("--routing")
        assert result.returncode == 0
        assert "Reverse mapping" in result.stdout
        # The default LLM should appear in reverse mapping with annotations
        assert "cydonia_heretic_24b" in result.stdout
        # Default annotation visible somewhere
        assert "default" in result.stdout

    def test_routing_with_llms_only(self):
        # --llms-only + --routing: should still print routing
        result = _run("--llms-only", "--routing")
        assert result.returncode == 0
        assert "Resolved LLMs by role" in result.stdout
        assert "Image models" not in result.stdout


class TestFamilyFilter:
    def test_family_filter_works(self):
        result = _run("--family", "sdxl")
        assert result.returncode == 0
        assert "Image models" in result.stdout
        # SDXL models present, non-SDXL absent
        assert "gonzalomo_photo_v70" in result.stdout
        assert "chroma_v10HD" not in result.stdout
