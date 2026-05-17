"""Tests for ``scripts/render_prompts.py`` — the Phase B+C CLI.

Covers helpers + main() exit-code paths. The engine's
``run_phase_b`` / ``run_phase_c`` are mocked so these tests run
without ComfyUI; they verify CLI plumbing (XOR validation,
template positional pairing, missing-prompts hint, multi-model
loop, --no-export, exit codes).
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
RENDER_PATH = REPO_ROOT / "scripts" / "render_prompts.py"


@pytest.fixture(scope="module")
def render_module():
    spec = importlib.util.spec_from_file_location(
        "render_prompts", str(RENDER_PATH)
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["render_prompts"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fresh_db(tmp_path: Path) -> Path:
    """Real DB built from init_db.py — needed for prompts-existence check."""
    db_path = tmp_path / "render.db"
    result = subprocess.run(
        [sys.executable, "scripts/init_db.py", "--db-path", str(db_path)],
        capture_output=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr.decode()
    return db_path


def _seed_series_with_prompts(
    db_path: Path,
    *,
    series_id: str = "ser_seed",
    scene_id: str = "ser_seed_sc_000",
    models: list[str] | None = None,
):
    """Seed series + 1 scene + 1 prompt per model."""
    models = models or ["gonzalomo_photo_v70"]
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO series (id, mode, content_level, style_profile_id, "
        "theme, target_count, status) "
        "VALUES (?, 'character', 'T2_implied', 'golden_hour_natural', "
        "'test', 1, 'planned')",
        (series_id,),
    )
    conn.execute(
        "INSERT INTO scenes (id, series_id, variation_axis, content_level) "
        "VALUES (?, ?, 'pose', 'T2_implied')",
        (scene_id, series_id),
    )
    for i, model_id in enumerate(models):
        conn.execute(
            "INSERT INTO prompts (id, series_id, scene_id, target_kind, "
            "model_id, llm_id, prompt_text, negative_prompt, prompt_hash, "
            "content_level, status) VALUES "
            "(?, ?, ?, 'model', ?, ?, ?, ?, ?, 'T2_implied', 'pending')",
            (f"p_{i}", series_id, scene_id, model_id, "cydonia_heretic_24b",
             "test prompt", "neg", f"hash_{i}"),
        )
    conn.commit()
    conn.close()


def _argv(*args: str) -> list[str]:
    return ["render_prompts.py", *args]


# ── helper unit tests ───────────────────────────────────────────────


class TestParseCsv:
    def test_none(self, render_module):
        assert render_module._parse_csv(None) == []

    def test_basic(self, render_module):
        assert render_module._parse_csv("a,b") == ["a", "b"]

    def test_dedupes(self, render_module):
        assert render_module._parse_csv("a,b,a") == ["a", "b"]


class TestResolveTemplates:
    """Positional pairing rules per plan §4.5 / §4.4."""

    def test_no_templates_all_system(self, render_module):
        out = render_module._resolve_templates(None, ["a", "b", "c"])
        assert out == [None, None, None]

    def test_one_template_applied_to_all_models(self, render_module):
        out = render_module._resolve_templates(
            "templates/x.json", ["a", "b"],
        )
        assert out == ["templates/x.json", "templates/x.json"]

    def test_one_system_token_applied_to_all(self, render_module):
        out = render_module._resolve_templates("system", ["a", "b"])
        assert out == [None, None]

    def test_n_templates_paired_positionally(self, render_module):
        out = render_module._resolve_templates(
            "system,templates/chroma/x.json", ["a", "b"],
        )
        assert out == [None, "templates/chroma/x.json"]

    def test_mismatched_lengths_rejected(self, render_module):
        from src.core.engine import EngineError
        with pytest.raises(EngineError, match="must equal"):
            render_module._resolve_templates(
                "a,b,c", ["model1", "model2"],
            )


class TestValidatePromptsExist:
    def test_counts_per_model(self, render_module, fresh_db):
        _seed_series_with_prompts(
            fresh_db, models=["gonzalomo_photo_v70", "chroma_v10HD"],
        )
        counts = render_module._validate_prompts_exist(
            fresh_db,
            series_id="ser_seed", scene_id=None,
            targets=["gonzalomo_photo_v70", "chroma_v10HD", "gonzalomo_flux_v30"],
        )
        assert counts == {
            "gonzalomo_photo_v70": 1,
            "chroma_v10HD": 1,
            "gonzalomo_flux_v30": 0,
        }

    def test_scene_id_filter(self, render_module, fresh_db):
        _seed_series_with_prompts(fresh_db)
        counts = render_module._validate_prompts_exist(
            fresh_db,
            series_id=None, scene_id="ser_seed_sc_000",
            targets=["gonzalomo_photo_v70"],
        )
        assert counts == {"gonzalomo_photo_v70": 1}


class TestResolveSeriesIdForScene:
    def test_lookup_returns_series_id(self, render_module, fresh_db):
        _seed_series_with_prompts(fresh_db)
        assert render_module._resolve_series_id_for_scene(
            fresh_db, "ser_seed_sc_000",
        ) == "ser_seed"

    def test_missing_scene_returns_none(self, render_module, fresh_db):
        assert render_module._resolve_series_id_for_scene(
            fresh_db, "no_such_scene",
        ) is None


# ── main() exit-code paths ──────────────────────────────────────────


def test_render_help_exits_clean():
    result = subprocess.run(
        [sys.executable, str(RENDER_PATH), "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "Phase B + Phase C" in result.stdout
    assert "--series-id" in result.stdout
    assert "--scene-id" in result.stdout
    assert "--templates" in result.stdout


def test_series_id_xor_scene_id_required():
    """Argparse mutually-exclusive group: exactly one of --series-id /
    --scene-id."""
    # Neither given → error.
    result = subprocess.run(
        [sys.executable, str(RENDER_PATH)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "one of the arguments --series-id --scene-id is required" in result.stderr

    # Both given → error.
    result = subprocess.run(
        [sys.executable, str(RENDER_PATH),
         "--series-id", "x", "--scene-id", "y"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "not allowed with argument" in result.stderr


def test_main_missing_prompts_returns_2(
    render_module, fresh_db, capsys, monkeypatch,
):
    """When the validate-prompts pre-check finds 0 for any model,
    abort with rc=2 and a hint pointing at prepare_prompts."""
    # DB is fresh — no prompts at all.
    fake_engine = MagicMock()
    fake_engine.db_path = fresh_db

    monkeypatch.setattr(
        render_module, "_load_config",
        lambda: {
            "pipeline": {"default_model_id": "gonzalomo_photo_v70"},
            "execution": {"mode": "manual"},
            "compliance": {"commercial_mode": False},
        },
    )
    monkeypatch.setattr(
        render_module, "PipelineEngine", lambda **kw: fake_engine,
    )
    monkeypatch.setattr(
        sys, "argv",
        _argv("--series-id", "ser_no_prompts", "--models", "gonzalomo_photo_v70"),
    )

    rc = render_module.main()
    assert rc == 2

    captured = capsys.readouterr()
    assert "ERROR: no model-kind prompts in DB" in captured.err
    assert "gonzalomo_photo_v70" in captured.err
    assert "prepare_prompts.py" in captured.err
    assert "--series-id ser_no_prompts" in captured.err
    assert "--models gonzalomo_photo_v70" in captured.err

    # Engine.run_phase_b should NOT have been called.
    fake_engine.run_phase_b.assert_not_called()


def test_main_scene_id_unknown_returns_1(
    render_module, fresh_db, capsys, monkeypatch,
):
    """A --scene-id that doesn't exist in the DB → rc=1, no run_phase_b."""
    fake_engine = MagicMock()
    fake_engine.db_path = fresh_db

    monkeypatch.setattr(
        render_module, "_load_config",
        lambda: {
            "pipeline": {"default_model_id": "gonzalomo_photo_v70"},
            "execution": {"mode": "manual"},
            "compliance": {"commercial_mode": False},
        },
    )
    monkeypatch.setattr(
        render_module, "PipelineEngine", lambda **kw: fake_engine,
    )
    monkeypatch.setattr(
        sys, "argv",
        _argv("--scene-id", "no_such_scene", "--models", "gonzalomo_photo_v70"),
    )

    rc = render_module.main()
    assert rc == 1
    assert "scene_id 'no_such_scene' not found" in capsys.readouterr().err


def _canned_phase_b_result(model_id: str, count: int = 3):
    """Mock return for run_phase_b: (rendered_images, ctx_state)."""
    rendered = [
        {
            "id": f"img_{i}", "prompt_id": f"p_{i}", "model_id": model_id,
            "file_path": f"/tmp/{model_id}_{i}.png", "width": 832,
            "height": 1216, "seed": i, "content_level": "T2_implied",
            "prompt_text": "test", "aspect_ratio": "portrait_23",
            "quality_score": 0.7,
        }
        for i in range(count)
    ]
    ctx_state = {
        "ctx": MagicMock(
            model_id=model_id, content_level="T2_implied",
        ),
        "series_plan": {"theme": "t"},
        "scenes": [],
        "prompts": [],
        "style_profile": {"id": "golden_hour_natural"},
        "style_profile_id": "golden_hour_natural",
    }
    return rendered, ctx_state


def _canned_phase_c_result(
    model_id: str, series_id: str = "ser_seed", count: int = 3,
):
    return {
        "series_id": series_id,
        "status": "complete",
        "images_rendered": count,
        "images_selected": count,
        "export_dir": f"/tmp/output/{series_id}",
        "elapsed_seconds": 1.0,
        "model_id": model_id,
        "theme": "t",
    }


def test_main_success_path_single_model(
    render_module, fresh_db, capsys, monkeypatch,
):
    _seed_series_with_prompts(fresh_db, models=["gonzalomo_photo_v70"])

    fake_engine = MagicMock()
    fake_engine.db_path = fresh_db
    fake_engine.run_phase_b.return_value = _canned_phase_b_result("gonzalomo_photo_v70")
    fake_engine.run_phase_c.return_value = _canned_phase_c_result("gonzalomo_photo_v70")

    monkeypatch.setattr(
        render_module, "_load_config",
        lambda: {
            "pipeline": {"default_model_id": "gonzalomo_photo_v70"},
            "execution": {"mode": "manual"},
            "compliance": {"commercial_mode": False},
        },
    )
    monkeypatch.setattr(
        render_module, "PipelineEngine", lambda **kw: fake_engine,
    )
    monkeypatch.setattr(
        sys, "argv",
        _argv(
            "--series-id", "ser_seed",
            "--models", "gonzalomo_photo_v70",
            "--llm", "cydonia_heretic_24b",
        ),
    )

    rc = render_module.main()
    assert rc == 0

    captured = capsys.readouterr()
    assert "render_prompts complete" in captured.out
    assert "gonzalomo_photo_v70" in captured.out
    assert "status=complete" in captured.out

    fake_engine.run_phase_b.assert_called_once_with(
        series_id="ser_seed", model_id="gonzalomo_photo_v70",
        scene_ids=None, template_override=None,
        cli_llm_override="cydonia_heretic_24b",
        target_kind="model", render_model_id=None,
    )
    fake_engine.run_phase_c.assert_called_once()


def test_main_multi_model_loops_per_model(
    render_module, fresh_db, capsys, monkeypatch,
):
    _seed_series_with_prompts(
        fresh_db, models=["gonzalomo_photo_v70", "chroma_v10HD"],
    )

    fake_engine = MagicMock()
    fake_engine.db_path = fresh_db
    fake_engine.run_phase_b.side_effect = [
        _canned_phase_b_result("gonzalomo_photo_v70", count=2),
        _canned_phase_b_result("chroma_v10HD", count=2),
    ]
    fake_engine.run_phase_c.side_effect = [
        _canned_phase_c_result("gonzalomo_photo_v70", count=2),
        _canned_phase_c_result("chroma_v10HD", count=2),
    ]

    monkeypatch.setattr(
        render_module, "_load_config",
        lambda: {
            "pipeline": {"default_model_id": "gonzalomo_photo_v70"},
            "execution": {"mode": "manual"},
            "compliance": {"commercial_mode": False},
        },
    )
    monkeypatch.setattr(
        render_module, "PipelineEngine", lambda **kw: fake_engine,
    )
    monkeypatch.setattr(
        sys, "argv",
        _argv(
            "--series-id", "ser_seed",
            "--models", "gonzalomo_photo_v70,chroma_v10HD",
            "--llm", "cydonia_heretic_24b",
        ),
    )

    rc = render_module.main()
    assert rc == 0

    # Two run_phase_b + two run_phase_c calls (one per model).
    assert fake_engine.run_phase_b.call_count == 2
    assert fake_engine.run_phase_c.call_count == 2


def test_main_no_export_skips_phase_c(
    render_module, fresh_db, capsys, monkeypatch,
):
    _seed_series_with_prompts(fresh_db, models=["gonzalomo_photo_v70"])

    fake_engine = MagicMock()
    fake_engine.db_path = fresh_db
    fake_engine.run_phase_b.return_value = _canned_phase_b_result("gonzalomo_photo_v70")

    monkeypatch.setattr(
        render_module, "_load_config",
        lambda: {
            "pipeline": {"default_model_id": "gonzalomo_photo_v70"},
            "execution": {"mode": "manual"},
            "compliance": {"commercial_mode": False},
        },
    )
    monkeypatch.setattr(
        render_module, "PipelineEngine", lambda **kw: fake_engine,
    )
    monkeypatch.setattr(
        sys, "argv",
        _argv(
            "--series-id", "ser_seed",
            "--models", "gonzalomo_photo_v70",
            "--no-export",
            "--llm", "cydonia_heretic_24b",
        ),
    )

    rc = render_module.main()
    assert rc == 0

    fake_engine.run_phase_b.assert_called_once()
    fake_engine.run_phase_c.assert_not_called()
    assert "skipped via --no-export" in capsys.readouterr().out


def test_main_template_pairing_passes_through(
    render_module, fresh_db, capsys, monkeypatch,
):
    _seed_series_with_prompts(
        fresh_db, models=["gonzalomo_photo_v70", "chroma_v10HD"],
    )

    fake_engine = MagicMock()
    fake_engine.db_path = fresh_db
    fake_engine.run_phase_b.side_effect = [
        _canned_phase_b_result("gonzalomo_photo_v70"),
        _canned_phase_b_result("chroma_v10HD"),
    ]
    fake_engine.run_phase_c.side_effect = [
        _canned_phase_c_result("gonzalomo_photo_v70"),
        _canned_phase_c_result("chroma_v10HD"),
    ]

    monkeypatch.setattr(
        render_module, "_load_config",
        lambda: {
            "pipeline": {"default_model_id": "gonzalomo_photo_v70"},
            "execution": {"mode": "manual"},
            "compliance": {"commercial_mode": False},
        },
    )
    monkeypatch.setattr(
        render_module, "PipelineEngine", lambda **kw: fake_engine,
    )
    monkeypatch.setattr(
        sys, "argv",
        _argv(
            "--series-id", "ser_seed",
            "--models", "gonzalomo_photo_v70,chroma_v10HD",
            "--templates", "system,templates/chroma/foo.json",
            "--llm", "cydonia_heretic_24b",
        ),
    )

    rc = render_module.main()
    assert rc == 0

    # Verify template pairing: model[0]→None (system), model[1]→external.
    calls = fake_engine.run_phase_b.call_args_list
    assert calls[0].kwargs["model_id"] == "gonzalomo_photo_v70"
    assert calls[0].kwargs["template_override"] is None
    assert calls[1].kwargs["model_id"] == "chroma_v10HD"
    assert calls[1].kwargs["template_override"] == "templates/chroma/foo.json"


def test_main_partial_failure_returns_1(
    render_module, fresh_db, capsys, monkeypatch,
):
    """If any model fails, exit 1 (even if others succeeded)."""
    _seed_series_with_prompts(
        fresh_db, models=["gonzalomo_photo_v70", "chroma_v10HD"],
    )

    fake_engine = MagicMock()
    fake_engine.db_path = fresh_db
    fake_engine.run_phase_b.side_effect = [
        _canned_phase_b_result("gonzalomo_photo_v70"),
        ([], {"aborted": False}),  # zero rendered for chroma
    ]
    fake_engine.run_phase_c.return_value = _canned_phase_c_result("gonzalomo_photo_v70")

    monkeypatch.setattr(
        render_module, "_load_config",
        lambda: {
            "pipeline": {"default_model_id": "gonzalomo_photo_v70"},
            "execution": {"mode": "manual"},
            "compliance": {"commercial_mode": False},
        },
    )
    monkeypatch.setattr(
        render_module, "PipelineEngine", lambda **kw: fake_engine,
    )
    monkeypatch.setattr(
        sys, "argv",
        _argv(
            "--series-id", "ser_seed",
            "--models", "gonzalomo_photo_v70,chroma_v10HD",
            "--llm", "cydonia_heretic_24b",
        ),
    )

    rc = render_module.main()
    assert rc == 1

    captured = capsys.readouterr()
    assert "gonzalomo_photo_v70" in captured.out and "complete" in captured.out
    assert "chroma_v10HD" in captured.out and "failed" in captured.out


def test_main_scene_id_path_resolves_series_and_filters(
    render_module, fresh_db, monkeypatch,
):
    """--scene-id resolves series via DB lookup + passes scene_ids
    filter into run_phase_b."""
    _seed_series_with_prompts(fresh_db, models=["gonzalomo_photo_v70"])

    fake_engine = MagicMock()
    fake_engine.db_path = fresh_db
    fake_engine.run_phase_b.return_value = _canned_phase_b_result("gonzalomo_photo_v70", count=1)
    fake_engine.run_phase_c.return_value = _canned_phase_c_result("gonzalomo_photo_v70", count=1)

    monkeypatch.setattr(
        render_module, "_load_config",
        lambda: {
            "pipeline": {"default_model_id": "gonzalomo_photo_v70"},
            "execution": {"mode": "manual"},
            "compliance": {"commercial_mode": False},
        },
    )
    monkeypatch.setattr(
        render_module, "PipelineEngine", lambda **kw: fake_engine,
    )
    monkeypatch.setattr(
        sys, "argv",
        _argv(
            "--scene-id", "ser_seed_sc_000",
            "--models", "gonzalomo_photo_v70",
            "--llm", "cydonia_heretic_24b",
        ),
    )

    rc = render_module.main()
    assert rc == 0

    call = fake_engine.run_phase_b.call_args
    assert call.kwargs["series_id"] == "ser_seed"   # resolved from scene
    assert call.kwargs["scene_ids"] == ["ser_seed_sc_000"]


# ── F3: --llm flag coverage ─────────────────────────────────────────


class TestLlmFlag:
    """Dedicated coverage for the render_prompts --llm flag (F3).

    Verifies strict-ambiguity check (plan §3.5b) when DB has prompts
    and --llm is omitted, plus filter behaviour when --llm is specified.
    """

    def _common(self, render_module, fresh_db, monkeypatch, fake_engine):
        monkeypatch.setattr(
            render_module, "_load_config",
            lambda: {
                "pipeline": {"default_model_id": "gonzalomo_photo_v70"},
                "execution": {"mode": "manual"},
                "compliance": {"commercial_mode": False},
            },
        )
        monkeypatch.setattr(
            render_module, "PipelineEngine", lambda **kw: fake_engine,
        )

    def test_strict_ambiguity_omitted_llm_with_seeded_prompts_exits_2(
        self, render_module, fresh_db, capsys, monkeypatch,
    ):
        # Seed prompts with the default cydonia_heretic_24b llm_id (the
        # _seed_series_with_prompts helper does this).
        _seed_series_with_prompts(fresh_db, models=["gonzalomo_photo_v70"])

        fake_engine = MagicMock()
        fake_engine.db_path = fresh_db
        self._common(render_module, fresh_db, monkeypatch, fake_engine)

        # Omit --llm — should hit the strict-ambiguity check.
        monkeypatch.setattr(
            sys, "argv",
            _argv("--series-id", "ser_seed", "--models", "gonzalomo_photo_v70"),
        )
        rc = render_module.main()
        assert rc == 2
        captured = capsys.readouterr()
        assert "has prompts from LLM(s)" in captured.err
        assert "cydonia_heretic_24b" in captured.err
        assert "Specify --llm" in captured.err

    def test_invalid_llm_exits_2(
        self, render_module, fresh_db, capsys, monkeypatch,
    ):
        _seed_series_with_prompts(fresh_db, models=["gonzalomo_photo_v70"])
        fake_engine = MagicMock()
        fake_engine.db_path = fresh_db
        self._common(render_module, fresh_db, monkeypatch, fake_engine)
        monkeypatch.setattr(
            sys, "argv",
            _argv(
                "--series-id", "ser_seed",
                "--models", "gonzalomo_photo_v70",
                "--llm", "not_a_real_llm_id",
            ),
        )
        rc = render_module.main()
        assert rc == 2
        captured = capsys.readouterr()
        assert "not_a_real_llm_id" in captured.err
        assert "Available LLMs" in captured.err

    def test_valid_llm_threads_to_run_phase_b(
        self, render_module, fresh_db, monkeypatch,
    ):
        _seed_series_with_prompts(fresh_db, models=["gonzalomo_photo_v70"])
        fake_engine = MagicMock()
        fake_engine.db_path = fresh_db
        fake_engine.run_phase_b.return_value = _canned_phase_b_result(
            "gonzalomo_photo_v70"
        )
        fake_engine.run_phase_c.return_value = _canned_phase_c_result(
            "gonzalomo_photo_v70"
        )
        self._common(render_module, fresh_db, monkeypatch, fake_engine)

        monkeypatch.setattr(
            sys, "argv",
            _argv(
                "--series-id", "ser_seed",
                "--models", "gonzalomo_photo_v70",
                "--llm", "cydonia_heretic_24b",
            ),
        )
        rc = render_module.main()
        assert rc == 0
        # cli_llm_override threaded to run_phase_b.
        _, kwargs = fake_engine.run_phase_b.call_args
        assert kwargs["cli_llm_override"] == "cydonia_heretic_24b"
