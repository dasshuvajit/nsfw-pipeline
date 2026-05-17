"""Tests for ``scripts/prepare_prompts.py`` — the Phase A CLI.

Covers helpers + the main() exit-code paths. The engine's
``run_phase_a`` is mocked so these tests run without Ollama or
ComfyUI; they verify CLI plumbing (arg parsing, multi-model parsing,
collision-error formatting, summary output, exit codes).
"""

from __future__ import annotations

import importlib.util
import sqlite3
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PREPARE_PATH = REPO_ROOT / "scripts" / "prepare_prompts.py"


@pytest.fixture(scope="module")
def prepare_module():
    """Load scripts/prepare_prompts.py as a module so we can call its
    helpers + main() directly (no subprocess startup cost)."""
    spec = importlib.util.spec_from_file_location(
        "prepare_prompts", str(PREPARE_PATH)
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["prepare_prompts"] = module
    spec.loader.exec_module(module)
    return module


# ── helper unit tests ───────────────────────────────────────────────


class TestParseCsv:
    def test_none_returns_empty(self, prepare_module):
        assert prepare_module._parse_csv(None) == []

    def test_empty_string_returns_empty(self, prepare_module):
        assert prepare_module._parse_csv("") == []

    def test_single_value(self, prepare_module):
        assert prepare_module._parse_csv("gonzalomo_photo_v70") == ["gonzalomo_photo_v70"]

    def test_multi_value(self, prepare_module):
        assert prepare_module._parse_csv("a,b,c") == ["a", "b", "c"]

    def test_strips_whitespace(self, prepare_module):
        assert prepare_module._parse_csv(" a , b , c ") == ["a", "b", "c"]

    def test_dedupes_preserving_order(self, prepare_module):
        """A,B,A → A,B (first occurrence wins)."""
        assert prepare_module._parse_csv("a,b,a") == ["a", "b"]

    def test_drops_empty_tokens(self, prepare_module):
        """Trailing comma or empty middle → no empty entries."""
        assert prepare_module._parse_csv("a,,b,") == ["a", "b"]


class TestResolveModels:
    def test_explicit_models_used(self, prepare_module):
        config = {"pipeline": {"default_model_id": "default_x"}}
        assert prepare_module._resolve_models("a,b", config) == ["a", "b"]

    def test_falls_back_to_default(self, prepare_module):
        config = {"pipeline": {"default_model_id": "default_x"}}
        assert prepare_module._resolve_models(None, config) == ["default_x"]

    def test_empty_string_falls_back_to_default(self, prepare_module):
        config = {"pipeline": {"default_model_id": "default_x"}}
        assert prepare_module._resolve_models("", config) == ["default_x"]

    def test_no_models_no_default_returns_empty(self, prepare_module):
        """Post family-level prompt-prep (2026-05): _resolve_models
        returns [] when no flag and no default. The post-parse
        validator in main() emits the helpful "pass --models or
        --families" error — _resolve_models itself stays silent so
        --families-only invocations don't trip a stale error path."""
        config = {"pipeline": {}}
        assert prepare_module._resolve_models(None, config) == []


class TestResolveFamilies:
    """Family-level prompt-prep (2026-05): symmetric helper to
    _resolve_models. Default is always [] — there's no
    pipeline.default_family_id (out of scope)."""

    def test_none_returns_empty(self, prepare_module):
        assert prepare_module._resolve_families(None) == []

    def test_empty_returns_empty(self, prepare_module):
        assert prepare_module._resolve_families("") == []

    def test_single_family(self, prepare_module):
        assert prepare_module._resolve_families("flux") == ["flux"]

    def test_multi_family(self, prepare_module):
        assert prepare_module._resolve_families("flux,pony,sdxl") == [
            "flux", "pony", "sdxl",
        ]


# ── main() exit-code paths (mocked engine) ─────────────────────────


def _patch_engine_init():
    """Patch PipelineEngine.__init__ so the test doesn't need Ollama
    or a real DB."""
    return patch.object(
        sys.modules.get("prepare_prompts") or object(),
        "PipelineEngine",
        new=MagicMock,
    )


def _argv(*args: str) -> list[str]:
    return ["prepare_prompts.py", *args]


def test_prepare_help_exits_clean():
    """Smoke: --help returns 0 and prints usage."""
    result = subprocess.run(
        [sys.executable, str(PREPARE_PATH), "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "Phase A only" in result.stdout
    assert "--models" in result.stdout
    assert "--regen-facets" in result.stdout
    assert "--regen-prompts" in result.stdout


def test_prepare_unknown_level_rejected_by_argparse():
    """Argparse choices= enforces 4-tier set."""
    result = subprocess.run(
        [sys.executable, str(PREPARE_PATH), "--level", "bogus"],
        capture_output=True, text=True,
    )
    assert result.returncode == 2  # argparse exit code for usage errors
    assert "invalid choice: 'bogus'" in result.stderr


# ── main() success path with mocked engine ──────────────────────────


def _make_canned_result(
    series_id: str = "ser_test",
    models: list[str] | None = None,
    status: str = "complete",
    families: list[str] | None = None,
) -> dict:
    return {
        "series_id": series_id,
        "status": status,
        "scenes_created": 5,
        "facets_created": 5,
        "prompts_created": 10,
        "models_completed": models if models is not None else ["gonzalomo_photo_v70"],
        "families_completed": families or [],
    }


def test_main_success_path_prints_next_command(
    prepare_module, capsys, monkeypatch,
):
    """Happy path: engine returns 'complete' → exit 0, summary printed,
    Next: ... line points at render_prompts."""
    fake_engine = MagicMock()
    fake_engine.db_path = Path("/tmp/_phase4_test.db")
    fake_engine.execution_mode = "manual"
    fake_engine._commercial_mode = False
    fake_engine.mode_selector.select.return_value = "character"
    fake_engine.run_phase_a.return_value = _make_canned_result(
        series_id="ser_xyz", models=["gonzalomo_photo_v70", "chroma_v10HD"],
    )

    fake_config = {
        "pipeline": {
            "default_model_id": "gonzalomo_photo_v70",
            "default_style_profile_id": "golden_hour_natural",
        },
        "execution": {"mode": "manual"},
        "compliance": {"commercial_mode": False},
    }

    monkeypatch.setattr(prepare_module, "_load_config", lambda: fake_config)
    monkeypatch.setattr(
        prepare_module, "PipelineEngine", lambda **kw: fake_engine,
    )
    monkeypatch.setattr(
        prepare_module, "_load_style_profile",
        lambda db, sid: {"id": sid, "base_negative_prompt": ""},
    )
    monkeypatch.setattr(
        prepare_module, "build_context",
        lambda **kw: MagicMock(execution_mode="manual"),
    )
    # Skip the content-rules DB load (no real DB).
    monkeypatch.setattr(
        prepare_module, "ContentLevelLoader",
        lambda db: MagicMock(load=lambda lvl: MagicMock()),
    )
    monkeypatch.setattr(
        sys, "argv",
        _argv("--models", "gonzalomo_photo_v70,chroma_v10HD", "--level", "T2_implied"),
    )

    rc = prepare_module.main()
    assert rc == 0

    captured = capsys.readouterr()
    assert "Phase A complete" in captured.out
    assert "Series:           ser_xyz" in captured.out
    assert "Models completed: gonzalomo_photo_v70, chroma_v10HD" in captured.out
    assert "render_prompts.py" in captured.out
    assert "--series-id ser_xyz" in captured.out
    assert "--models gonzalomo_photo_v70,chroma_v10HD" in captured.out


def test_main_unique_collision_returns_2_with_hint(
    prepare_module, capsys, monkeypatch,
):
    """When run_phase_a raises sqlite3.IntegrityError(UNIQUE...) the
    CLI catches it, prints the --regen-prompts hint, returns 2."""
    fake_engine = MagicMock()
    fake_engine.db_path = Path("/tmp/_phase4_test.db")
    fake_engine.execution_mode = "manual"
    fake_engine._commercial_mode = False
    fake_engine.mode_selector.select.return_value = "character"
    fake_engine.run_phase_a.side_effect = sqlite3.IntegrityError(
        "UNIQUE constraint failed: prompts.scene_id, prompts.target_kind, "
        "prompts.model_id, prompts.llm_id"
    )

    monkeypatch.setattr(
        prepare_module, "_load_config",
        lambda: {
            "pipeline": {
                "default_model_id": "gonzalomo_photo_v70",
                "default_style_profile_id": "golden_hour_natural",
            },
            "execution": {"mode": "manual"},
            "compliance": {"commercial_mode": False},
        },
    )
    monkeypatch.setattr(
        prepare_module, "PipelineEngine", lambda **kw: fake_engine,
    )
    monkeypatch.setattr(
        prepare_module, "_load_style_profile",
        lambda db, sid: {"id": sid, "base_negative_prompt": ""},
    )
    monkeypatch.setattr(
        prepare_module, "build_context",
        lambda **kw: MagicMock(execution_mode="manual"),
    )
    monkeypatch.setattr(
        prepare_module, "ContentLevelLoader",
        lambda db: MagicMock(load=lambda lvl: MagicMock()),
    )
    monkeypatch.setattr(
        sys, "argv",
        _argv("--models", "gonzalomo_photo_v70", "--series-id", "ser_existing"),
    )

    rc = prepare_module.main()
    assert rc == 2

    captured = capsys.readouterr()
    assert "ERROR: prompts" in captured.err
    assert "ser_existing" in captured.err
    assert "--regen-prompts gonzalomo_photo_v70" in captured.err


def test_main_supervisor_aborted_returns_3(
    prepare_module, capsys, monkeypatch,
):
    """When run_phase_a returns status='aborted', exit 3."""
    fake_engine = MagicMock()
    fake_engine.db_path = Path("/tmp/_phase4_test.db")
    fake_engine.execution_mode = "supervised"
    fake_engine._commercial_mode = False
    fake_engine.mode_selector.select.return_value = "character"
    fake_engine.run_phase_a.return_value = _make_canned_result(
        status="aborted", models=[],
    )

    monkeypatch.setattr(
        prepare_module, "_load_config",
        lambda: {
            "pipeline": {
                "default_model_id": "gonzalomo_photo_v70",
                "default_style_profile_id": "golden_hour_natural",
            },
            "execution": {"mode": "supervised"},
            "compliance": {"commercial_mode": False},
        },
    )
    monkeypatch.setattr(
        prepare_module, "PipelineEngine", lambda **kw: fake_engine,
    )
    monkeypatch.setattr(
        prepare_module, "_load_style_profile",
        lambda db, sid: {"id": sid, "base_negative_prompt": ""},
    )
    monkeypatch.setattr(
        prepare_module, "build_context",
        lambda **kw: MagicMock(execution_mode="supervised"),
    )
    monkeypatch.setattr(
        prepare_module, "ContentLevelLoader",
        lambda db: MagicMock(load=lambda lvl: MagicMock()),
    )
    monkeypatch.setattr(sys, "argv", _argv("--level", "T2_implied"))

    rc = prepare_module.main()
    assert rc == 3
    assert "aborted" in capsys.readouterr().err.lower()


def test_main_engine_error_returns_1(
    prepare_module, capsys, monkeypatch,
):
    """An EngineError (e.g. unknown model id) → exit 1 + stderr message."""
    from src.core.engine import EngineError

    monkeypatch.setattr(
        prepare_module, "_load_config",
        lambda: {
            "pipeline": {
                "default_model_id": "gonzalomo_photo_v70",
                "default_style_profile_id": "golden_hour_natural",
            },
            "execution": {"mode": "manual"},
            "compliance": {"commercial_mode": False},
        },
    )

    def explode(**kw):
        raise EngineError("bogus_model is not a valid model id")
    monkeypatch.setattr(prepare_module, "PipelineEngine", explode)
    monkeypatch.setattr(sys, "argv", _argv("--models", "bogus_model"))

    rc = prepare_module.main()
    assert rc == 1
    assert "bogus_model" in capsys.readouterr().err


def test_main_passes_regen_flags_through(
    prepare_module, monkeypatch,
):
    """--regen-facets and --regen-prompts are parsed + threaded into
    run_phase_a."""
    fake_engine = MagicMock()
    fake_engine.db_path = Path("/tmp/_phase4_test.db")
    fake_engine.execution_mode = "manual"
    fake_engine._commercial_mode = False
    fake_engine.mode_selector.select.return_value = "character"
    fake_engine.run_phase_a.return_value = _make_canned_result()

    monkeypatch.setattr(
        prepare_module, "_load_config",
        lambda: {
            "pipeline": {
                "default_model_id": "gonzalomo_photo_v70",
                "default_style_profile_id": "golden_hour_natural",
            },
            "execution": {"mode": "manual"},
            "compliance": {"commercial_mode": False},
        },
    )
    monkeypatch.setattr(
        prepare_module, "PipelineEngine", lambda **kw: fake_engine,
    )
    monkeypatch.setattr(
        prepare_module, "_load_style_profile",
        lambda db, sid: {"id": sid, "base_negative_prompt": ""},
    )
    monkeypatch.setattr(
        prepare_module, "build_context",
        lambda **kw: MagicMock(execution_mode="manual"),
    )
    monkeypatch.setattr(
        prepare_module, "ContentLevelLoader",
        lambda db: MagicMock(load=lambda lvl: MagicMock()),
    )
    monkeypatch.setattr(
        sys, "argv",
        _argv(
            "--models", "gonzalomo_photo_v70",
            "--series-id", "ser_existing",
            "--regen-facets", "sdxl,pony",
            "--regen-prompts", "gonzalomo_photo_v70",
        ),
    )

    rc = prepare_module.main()
    assert rc == 0

    # Verify run_phase_a was called with the parsed regen lists.
    fake_engine.run_phase_a.assert_called_once()
    _, kwargs = fake_engine.run_phase_a.call_args
    assert kwargs["models"] == ["gonzalomo_photo_v70"]
    assert kwargs["series_id_existing"] == "ser_existing"
    assert kwargs["regen_facets"] == ["sdxl", "pony"]
    assert kwargs["regen_prompts"] == ["gonzalomo_photo_v70"]


# ── F3: --llm flag coverage ─────────────────────────────────────────


class TestLlmFlag:
    """Dedicated coverage for the --llm flag (F3).

    Verifies the CLI threads cli_llm_override into engine.run_phase_a
    on the happy path, and exits with helpful error on invalid id.
    """

    def _common_monkeypatch(self, prepare_module, monkeypatch, fake_engine):
        monkeypatch.setattr(
            prepare_module, "_load_config",
            lambda: {
                "pipeline": {
                    "default_model_id": "gonzalomo_photo_v70",
                    "default_style_profile_id": "golden_hour_natural",
                },
                "execution": {"mode": "manual"},
                "compliance": {"commercial_mode": False},
            },
        )
        monkeypatch.setattr(
            prepare_module, "PipelineEngine", lambda **kw: fake_engine,
        )
        monkeypatch.setattr(
            prepare_module, "_load_style_profile",
            lambda db, sid: {"id": sid, "base_negative_prompt": ""},
        )
        monkeypatch.setattr(
            prepare_module, "build_context",
            lambda **kw: MagicMock(execution_mode="manual"),
        )
        monkeypatch.setattr(
            prepare_module, "ContentLevelLoader",
            lambda db: MagicMock(load=lambda lvl: MagicMock()),
        )

    def test_valid_llm_threads_cli_override_into_run_phase_a(
        self, prepare_module, monkeypatch,
    ):
        fake_engine = MagicMock()
        fake_engine.db_path = Path("/tmp/_test.db")
        fake_engine.execution_mode = "manual"
        fake_engine._commercial_mode = False
        fake_engine.mode_selector.select.return_value = "character"
        fake_engine.run_phase_a.return_value = _make_canned_result()
        self._common_monkeypatch(prepare_module, monkeypatch, fake_engine)
        monkeypatch.setattr(
            sys, "argv",
            _argv("--models", "gonzalomo_photo_v70", "--llm", "cydonia_24b_v43"),
        )

        rc = prepare_module.main()
        assert rc == 0
        # run_phase_a should have received cli_llm_override.
        _, kwargs = fake_engine.run_phase_a.call_args
        assert kwargs["cli_llm_override"] == "cydonia_24b_v43"

    def test_invalid_llm_exits_2_with_help(
        self, prepare_module, monkeypatch, capsys,
    ):
        fake_engine = MagicMock()
        self._common_monkeypatch(prepare_module, monkeypatch, fake_engine)
        monkeypatch.setattr(
            sys, "argv",
            _argv("--models", "gonzalomo_photo_v70", "--llm", "not_a_real_llm_id"),
        )

        rc = prepare_module.main()
        assert rc == 2
        captured = capsys.readouterr()
        assert "not_a_real_llm_id" in captured.err
        assert "Available LLMs" in captured.err
        # Engine never invoked when --llm fails validation.
        fake_engine.run_phase_a.assert_not_called()

    def test_no_llm_threads_none_to_run_phase_a(
        self, prepare_module, monkeypatch,
    ):
        """--llm omitted → cli_llm_override=None → routing/default chain."""
        fake_engine = MagicMock()
        fake_engine.db_path = Path("/tmp/_test.db")
        fake_engine.execution_mode = "manual"
        fake_engine._commercial_mode = False
        fake_engine.mode_selector.select.return_value = "character"
        fake_engine.run_phase_a.return_value = _make_canned_result()
        self._common_monkeypatch(prepare_module, monkeypatch, fake_engine)
        monkeypatch.setattr(
            sys, "argv", _argv("--models", "gonzalomo_photo_v70"),
        )

        rc = prepare_module.main()
        assert rc == 0
        _, kwargs = fake_engine.run_phase_a.call_args
        assert kwargs["cli_llm_override"] is None


# ── Family-level prompt prep (2026-05) ──────────────────────────────


class TestFamiliesFlag:
    """Dedicated coverage for the --families flag (Phase 4, family-
    level prompt-prep feature).

    Verifies the CLI threads families= into engine.run_phase_a, plays
    nicely with --models (both can coexist), enforces "at least one
    target" post-parse, and prints the right "Next: --families …
    --render-with-model …" suggestion.
    """

    def _common_monkeypatch(self, prepare_module, monkeypatch, fake_engine):
        monkeypatch.setattr(
            prepare_module, "_load_config",
            lambda: {
                "pipeline": {
                    "default_model_id": "gonzalomo_photo_v70",
                    "default_style_profile_id": "golden_hour_natural",
                },
                "execution": {"mode": "manual"},
                "compliance": {"commercial_mode": False},
            },
        )
        monkeypatch.setattr(
            prepare_module, "PipelineEngine", lambda **kw: fake_engine,
        )
        monkeypatch.setattr(
            prepare_module, "_load_style_profile",
            lambda db, sid: {"id": sid, "base_negative_prompt": ""},
        )
        monkeypatch.setattr(
            prepare_module, "build_context",
            lambda **kw: MagicMock(execution_mode="manual"),
        )
        monkeypatch.setattr(
            prepare_module, "ContentLevelLoader",
            lambda db: MagicMock(load=lambda lvl: MagicMock()),
        )
        # _build_baseline_ctx falls back to first model in family when
        # only --families is supplied; stub the registry lookup so we
        # don't need a real DB.
        from src.memory.model_registry import ModelRegistryLoader
        fake_loader = MagicMock()
        fake_loader.list_models.return_value = [
            MagicMock(id="gonzalomo_flux_v30")
        ]
        monkeypatch.setattr(
            prepare_module, "ModelRegistryLoader",
            lambda *a, **kw: fake_loader, raising=False,
        )
        # ModelRegistryLoader is imported INSIDE _build_baseline_ctx
        # (lazy import). Monkeypatch the actual class so that import
        # picks up the stub.
        monkeypatch.setattr(
            "src.memory.model_registry.ModelRegistryLoader",
            lambda *a, **kw: fake_loader,
        )

    def test_families_only_threads_to_run_phase_a(
        self, prepare_module, monkeypatch,
    ):
        """--families flux (no --models) → run_phase_a sees
        families=['flux'] and models=None."""
        fake_engine = MagicMock()
        fake_engine.db_path = Path("/tmp/_fam_test.db")
        fake_engine.execution_mode = "manual"
        fake_engine._commercial_mode = False
        fake_engine.mode_selector.select.return_value = "theme"
        fake_engine.run_phase_a.return_value = _make_canned_result(
            models=[], families=["flux"],
        )
        self._common_monkeypatch(prepare_module, monkeypatch, fake_engine)
        monkeypatch.setattr(
            sys, "argv", _argv("--families", "flux"),
        )

        rc = prepare_module.main()
        assert rc == 0
        _, kwargs = fake_engine.run_phase_a.call_args
        assert kwargs["families"] == ["flux"]
        # In family-only mode, the implicit pipeline.default_model_id
        # fallback is suppressed.
        assert kwargs["models"] is None

    def test_models_and_families_coexist(
        self, prepare_module, monkeypatch,
    ):
        """--models X --families Y → run_phase_a gets BOTH lists."""
        fake_engine = MagicMock()
        fake_engine.db_path = Path("/tmp/_test.db")
        fake_engine.execution_mode = "manual"
        fake_engine._commercial_mode = False
        fake_engine.mode_selector.select.return_value = "character"
        fake_engine.run_phase_a.return_value = _make_canned_result(
            models=["gonzalomo_photo_v70"], families=["flux"],
        )
        self._common_monkeypatch(prepare_module, monkeypatch, fake_engine)
        monkeypatch.setattr(
            sys, "argv",
            _argv("--models", "gonzalomo_photo_v70", "--families", "flux"),
        )

        rc = prepare_module.main()
        assert rc == 0
        _, kwargs = fake_engine.run_phase_a.call_args
        assert kwargs["models"] == ["gonzalomo_photo_v70"]
        assert kwargs["families"] == ["flux"]

    def test_neither_models_nor_families_with_no_default_exits_2(
        self, prepare_module, monkeypatch, capsys,
    ):
        """Both flags omitted AND pipeline.default_model_id is absent
        → exit 2 with the helpful "pass --models or --families"
        error."""
        fake_engine = MagicMock()
        # Override config to drop default_model_id.
        monkeypatch.setattr(
            prepare_module, "_load_config",
            lambda: {
                "pipeline": {
                    "default_style_profile_id": "golden_hour_natural",
                },
                "execution": {"mode": "manual"},
                "compliance": {"commercial_mode": False},
            },
        )
        monkeypatch.setattr(
            prepare_module, "PipelineEngine", lambda **kw: fake_engine,
        )
        monkeypatch.setattr(
            sys, "argv", _argv(),  # no --models, no --families
        )

        rc = prepare_module.main()
        assert rc == 2
        captured = capsys.readouterr()
        assert "pass --models" in captured.err
        assert "--families" in captured.err
        fake_engine.run_phase_a.assert_not_called()

    def test_regen_family_prompts_threads_through(
        self, prepare_module, monkeypatch,
    ):
        """--regen-family-prompts threads to run_phase_a as a
        separate kwarg from --regen-prompts."""
        fake_engine = MagicMock()
        fake_engine.db_path = Path("/tmp/_test.db")
        fake_engine.execution_mode = "manual"
        fake_engine._commercial_mode = False
        fake_engine.mode_selector.select.return_value = "theme"
        fake_engine.run_phase_a.return_value = _make_canned_result(
            models=[], families=["flux"],
        )
        self._common_monkeypatch(prepare_module, monkeypatch, fake_engine)
        monkeypatch.setattr(
            sys, "argv",
            _argv(
                "--families", "flux",
                "--series-id", "ser_existing",
                "--regen-family-prompts", "flux",
            ),
        )

        rc = prepare_module.main()
        assert rc == 0
        _, kwargs = fake_engine.run_phase_a.call_args
        assert kwargs["regen_family_prompts"] == ["flux"]
        # --regen-prompts not passed → None
        assert kwargs["regen_prompts"] is None

    def test_summary_prints_render_with_model_hint(
        self, prepare_module, monkeypatch, capsys,
    ):
        """When families_completed is non-empty, the 'Next:' summary
        suggests --families F --render-with-model <pick>."""
        fake_engine = MagicMock()
        fake_engine.db_path = Path("/tmp/_test.db")
        fake_engine.execution_mode = "manual"
        fake_engine._commercial_mode = False
        fake_engine.mode_selector.select.return_value = "theme"
        fake_engine.run_phase_a.return_value = _make_canned_result(
            series_id="ser_xyz", models=[], families=["flux"],
        )
        self._common_monkeypatch(prepare_module, monkeypatch, fake_engine)
        monkeypatch.setattr(
            sys, "argv", _argv("--families", "flux"),
        )

        rc = prepare_module.main()
        assert rc == 0
        captured = capsys.readouterr()
        assert "Families completed: flux" in captured.out
        assert "--families flux" in captured.out
        assert "--render-with-model" in captured.out
