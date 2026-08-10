"""Tests for ``scripts/list_models.py``.

LLM-registry-only since 2026-06-10 — the image-model registry half was
archived with the legacy structured path; the ``--routing`` table was
archived 2026-08 with ``src/agents/llm_router.py`` (see legacy/).
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
        # (2026-06-15 evening — HauhauCS Gemma4 26B-A4B Balanced won the overnight
        # 3-way A/B and is the default; Cydonia is still the registered fallback).
        assert "gemma4_26b_a4b_uncensored_hauhaucs_balanced" in result.stdout
        assert "default = 'gemma4_26b_a4b_uncensored_hauhaucs_balanced'" in result.stdout
