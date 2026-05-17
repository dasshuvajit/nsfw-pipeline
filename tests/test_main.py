"""Tests for ``src/main.py`` — the production scheduler entry (F3).

Covers ``run_safe`` threading + the ``--llm`` flag's CLI-boundary
validation. The full scheduler loop is not exercised here (that
would require a live Ollama + ComfyUI); we patch ``PipelineEngine``
and assert the threading.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MAIN_PATH = REPO_ROOT / "src" / "main.py"


@pytest.fixture
def main_module():
    """Re-import src.main for each test (it's a script-style module
    so we can't share state across cases)."""
    spec = importlib.util.spec_from_file_location("src_main", str(MAIN_PATH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("src_main", None)


class TestRunSafeThreading:
    """``run_safe(cli_llm_override=...)`` threads to engine.run_cycle."""

    def test_run_safe_threads_cli_llm_override(
        self, main_module, monkeypatch, tmp_path,
    ):
        """Pass a value through run_safe → engine constructor →
        engine.run_cycle.cli_llm_override."""
        fake_engine = MagicMock()
        fake_engine.run_cycle.return_value = {
            "status": "complete", "series_id": "ser_x",
            "images_selected": 1, "images_generated": 1,
        }

        captured_engine_kwargs = {}

        def fake_engine_ctor(**kwargs):
            captured_engine_kwargs.update(kwargs)
            return fake_engine

        monkeypatch.setattr(main_module, "PipelineEngine", fake_engine_ctor)
        # Stub the ancillary helpers run_safe touches.
        monkeypatch.setattr(main_module, "_cleanup_zombies", lambda db: None)
        monkeypatch.setattr(
            main_module, "_select_content_level",
            lambda cfg: "T2_implied",
        )

        # Pretend lock file logic is OK.
        monkeypatch.setattr(main_module, "LOCK_FILE", str(tmp_path / "lock"))

        cfg = {
            "execution": {"mode": "automated"},
            "pipeline": {"runs_per_day": 3},
        }
        db_path = tmp_path / "test.db"
        db_path.touch()

        main_module.run_safe(
            cfg, db_path,
            force_mode="character",
            force_level="T4_explicit",
            cli_llm_override="cydonia_heretic_24b",
        )

        # cli_llm_override threaded through to run_cycle.
        _, kwargs = fake_engine.run_cycle.call_args
        assert kwargs["cli_llm_override"] == "cydonia_heretic_24b"
        assert kwargs["content_level"] == "T4_explicit"

    def test_run_safe_with_no_llm_override(
        self, main_module, monkeypatch, tmp_path,
    ):
        fake_engine = MagicMock()
        fake_engine.run_cycle.return_value = {
            "status": "complete", "series_id": "ser_x",
            "images_selected": 1, "images_generated": 1,
        }
        monkeypatch.setattr(
            main_module, "PipelineEngine", lambda **kw: fake_engine,
        )
        monkeypatch.setattr(main_module, "_cleanup_zombies", lambda db: None)
        monkeypatch.setattr(
            main_module, "_select_content_level",
            lambda cfg: "T2_implied",
        )
        monkeypatch.setattr(main_module, "LOCK_FILE", str(tmp_path / "lock"))

        cfg = {"execution": {"mode": "automated"},
               "pipeline": {"runs_per_day": 3}}
        db_path = tmp_path / "test.db"
        db_path.touch()
        main_module.run_safe(cfg, db_path, cli_llm_override=None)

        _, kwargs = fake_engine.run_cycle.call_args
        assert kwargs["cli_llm_override"] is None


class TestArgparseHelp:
    """Smoke: --help shows --llm flag."""

    def test_help_includes_llm(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, str(MAIN_PATH), "--help"],
            capture_output=True, text=True,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0
        assert "--llm" in result.stdout
        assert "config/llm_models.yaml" in result.stdout
