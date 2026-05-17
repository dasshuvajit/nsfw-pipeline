"""Tests for the --families / --render-with-model render path.

Companion to ``test_render_prompts.py``. Covers:

  1. ``--families flux --render-with-model gonzalomo_flux_v30`` selects
     family-kind rows and threads ``target_kind='family'`` +
     ``render_model_id='gonzalomo_flux_v30'`` to ``engine.run_phase_b``.
  2. ``--families flux --render-with-model gonzalomo_photo_v70`` → rc=2 at
     parse time (gonzalomo_photo_v70 is sdxl-family, not flux).
  3. ``--models X --families Y`` → rc=2 (mutual exclusion).
  4. ``--families X`` without ``--render-with-model`` → rc=2.
  5. ``--families X`` with NO matching prompts in DB → rc=2 with
     remediation hint pointing at ``prepare_prompts --families X``.

Engine is mocked — these tests verify CLI plumbing only.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RENDER_PATH = REPO_ROOT / "scripts" / "render_prompts.py"


@pytest.fixture(scope="module")
def render_module():
    spec = importlib.util.spec_from_file_location(
        "render_prompts_fam", str(RENDER_PATH),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["render_prompts_fam"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fresh_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "render_fam.db"
    result = subprocess.run(
        [sys.executable, "scripts/init_db.py", "--db-path", str(db_path)],
        capture_output=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr.decode()
    return db_path


def _seed_family_prompts(
    db_path: Path,
    *,
    series_id: str = "ser_fam",
    scene_id: str = "ser_fam_sc_000",
    families: list[str] | None = None,
    llm_id: str = "cydonia_heretic_24b",
):
    """Seed series + scene + 1 family-kind prompt per family.

    The prompts.model_id column carries the family id when
    target_kind='family' (dual-semantic — see init_db comment).
    """
    families = families or ["flux"]
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
    for i, family_id in enumerate(families):
        conn.execute(
            "INSERT INTO prompts (id, series_id, scene_id, target_kind, "
            "model_id, llm_id, prompt_text, negative_prompt, prompt_hash, "
            "content_level, status) VALUES "
            "(?, ?, ?, 'family', ?, ?, ?, ?, ?, 'T2_implied', 'pending')",
            (f"pf_{i}", series_id, scene_id, family_id, llm_id,
             "test family prompt", "neg", f"famhash_{i}"),
        )
    conn.commit()
    conn.close()


def _argv(*args: str) -> list[str]:
    return ["render_prompts.py", *args]


def _common_monkeypatch(render_module, fake_engine, monkeypatch):
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


def _canned_phase_b_result(model_id: str = "gonzalomo_flux_v30", count: int = 2):
    rendered = [
        {
            "id": f"img_{i}", "prompt_id": f"pf_{i}", "model_id": model_id,
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


def _canned_phase_c_result(model_id: str = "gonzalomo_flux_v30", count: int = 2):
    return {
        "series_id": "ser_fam",
        "status": "complete",
        "images_rendered": count,
        "images_selected": count,
        "export_dir": f"/tmp/output/ser_fam/cydonia_heretic_24b/flux/",
        "elapsed_seconds": 1.0,
        "model_id": model_id,
        "theme": "t",
    }


# ── 1. happy path: --families X --render-with-model Y ───────────────


def test_families_threads_target_kind_and_render_model_to_phase_b(
    render_module, fresh_db, monkeypatch,
):
    """``--families flux --render-with-model gonzalomo_flux_v30`` calls
    ``run_phase_b`` with ``target_kind='family'``,
    ``render_model_id='gonzalomo_flux_v30'`` and uses the family id ('flux')
    as the loop-target ``model_id``."""
    _seed_family_prompts(fresh_db, families=["flux"])

    fake_engine = MagicMock()
    fake_engine.db_path = fresh_db
    fake_engine.run_phase_b.return_value = _canned_phase_b_result()
    fake_engine.run_phase_c.return_value = _canned_phase_c_result()

    _common_monkeypatch(render_module, fake_engine, monkeypatch)
    monkeypatch.setattr(
        sys, "argv",
        _argv(
            "--series-id", "ser_fam",
            "--families", "flux",
            "--render-with-model", "gonzalomo_flux_v30",
            "--llm", "cydonia_heretic_24b",
        ),
    )

    rc = render_module.main()
    assert rc == 0

    fake_engine.run_phase_b.assert_called_once_with(
        series_id="ser_fam",
        model_id="flux",                       # loop-target = family id
        scene_ids=None,
        template_override=None,
        cli_llm_override="cydonia_heretic_24b",
        target_kind="family",
        render_model_id="gonzalomo_flux_v30",     # actual render checkpoint
    )
    # Phase C also threaded with target_kind+target_id.
    pc_kwargs = fake_engine.run_phase_c.call_args.kwargs
    assert pc_kwargs["target_kind"] == "family"
    assert pc_kwargs["target_id"] == "flux"


# ── 1b. --families + --templates (refiner workflow path) ────────────


def test_families_with_external_template_threads_template_override(
    render_module, fresh_db, monkeypatch,
):
    """The canonical refiner-workflow path: ``--families chroma
    --render-with-model gonzalomo_chroma_v30 --templates
    templates/chroma/gonzaLomo_Chroma_Refiner_v11.json`` must work —
    the external template is passed through as ``template_override``
    to ``run_phase_b``. Regression guard against the over-restrictive
    mutex check that originally forbade this combination (fixed
    2026-05-15 alongside the refiner contract extension)."""
    _seed_family_prompts(fresh_db, families=["chroma"])

    fake_engine = MagicMock()
    fake_engine.db_path = fresh_db
    fake_engine.run_phase_b.return_value = _canned_phase_b_result(
        "gonzalomo_chroma_v30"
    )
    fake_engine.run_phase_c.return_value = _canned_phase_c_result(
        "gonzalomo_chroma_v30"
    )

    _common_monkeypatch(render_module, fake_engine, monkeypatch)
    monkeypatch.setattr(
        sys, "argv",
        _argv(
            "--series-id", "ser_fam",
            "--families", "chroma",
            "--render-with-model", "gonzalomo_chroma_v30",
            "--templates", "templates/chroma/gonzaLomo_Chroma_Refiner_v11.json",
            "--llm", "cydonia_heretic_24b",
        ),
    )

    rc = render_module.main()
    assert rc == 0

    # run_phase_b must receive the external template path as
    # template_override (NOT None like in the no-template family case).
    fake_engine.run_phase_b.assert_called_once()
    kwargs = fake_engine.run_phase_b.call_args.kwargs
    assert kwargs["template_override"] == (
        "templates/chroma/gonzaLomo_Chroma_Refiner_v11.json"
    )
    assert kwargs["target_kind"] == "family"
    assert kwargs["render_model_id"] == "gonzalomo_chroma_v30"
    assert kwargs["model_id"] == "chroma"


# ── 2. family mismatch: --render-with-model from wrong family ───────


def test_render_with_model_from_wrong_family_exits_2(
    render_module, fresh_db, capsys, monkeypatch,
):
    """``--families flux --render-with-model gonzalomo_photo_v70`` is rejected
    at parse time — gonzalomo_photo_v70 is sdxl-family, not flux."""
    _seed_family_prompts(fresh_db, families=["flux"])

    fake_engine = MagicMock()
    fake_engine.db_path = fresh_db
    _common_monkeypatch(render_module, fake_engine, monkeypatch)
    monkeypatch.setattr(
        sys, "argv",
        _argv(
            "--series-id", "ser_fam",
            "--families", "flux",
            "--render-with-model", "gonzalomo_photo_v70",
            "--llm", "cydonia_heretic_24b",
        ),
    )

    rc = render_module.main()
    assert rc == 2
    captured = capsys.readouterr()
    assert "gonzalomo_photo_v70" in captured.err
    assert "sdxl" in captured.err
    assert "flux" in captured.err
    fake_engine.run_phase_b.assert_not_called()


# ── 3. mutex: --models AND --families together ──────────────────────


def test_models_and_families_mutex_exits_2(
    render_module, fresh_db, capsys, monkeypatch,
):
    fake_engine = MagicMock()
    fake_engine.db_path = fresh_db
    _common_monkeypatch(render_module, fake_engine, monkeypatch)
    monkeypatch.setattr(
        sys, "argv",
        _argv(
            "--series-id", "ser_fam",
            "--models", "gonzalomo_photo_v70",
            "--families", "flux",
            "--render-with-model", "gonzalomo_flux_v30",
            "--llm", "cydonia_heretic_24b",
        ),
    )

    rc = render_module.main()
    assert rc == 2
    captured = capsys.readouterr()
    assert "mutually exclusive" in captured.err
    fake_engine.run_phase_b.assert_not_called()


# ── 4. --families requires --render-with-model ──────────────────────


def test_families_without_render_with_model_exits_2(
    render_module, fresh_db, capsys, monkeypatch,
):
    fake_engine = MagicMock()
    fake_engine.db_path = fresh_db
    _common_monkeypatch(render_module, fake_engine, monkeypatch)
    monkeypatch.setattr(
        sys, "argv",
        _argv(
            "--series-id", "ser_fam",
            "--families", "flux",
            "--llm", "cydonia_heretic_24b",
        ),
    )

    rc = render_module.main()
    assert rc == 2
    captured = capsys.readouterr()
    assert "--families requires --render-with-model" in captured.err
    fake_engine.run_phase_b.assert_not_called()


# ── 4b. --render-with-model without --families ──────────────────────


def test_render_with_model_without_families_exits_2(
    render_module, fresh_db, capsys, monkeypatch,
):
    """``--render-with-model`` alone (no --families) is meaningless
    since for --models the model_id IS the checkpoint."""
    fake_engine = MagicMock()
    fake_engine.db_path = fresh_db
    _common_monkeypatch(render_module, fake_engine, monkeypatch)
    monkeypatch.setattr(
        sys, "argv",
        _argv(
            "--series-id", "ser_fam",
            "--models", "gonzalomo_photo_v70",
            "--render-with-model", "gonzalomo_flux_v30",
            "--llm", "cydonia_heretic_24b",
        ),
    )

    rc = render_module.main()
    assert rc == 2
    captured = capsys.readouterr()
    assert "--render-with-model is only valid with --families" in captured.err
    fake_engine.run_phase_b.assert_not_called()


# ── 5. empty family-kind series: missing-prompts hint ──────────────


def test_no_family_prompts_emits_families_hint(
    render_module, fresh_db, capsys, monkeypatch,
):
    """When the DB has zero family-kind prompts for the given family,
    the missing-prompts hint must say ``--families X``, not
    ``--models X`` (so the operator runs the right prep command)."""
    # Note: do NOT seed any prompts at all — fresh DB.
    fake_engine = MagicMock()
    fake_engine.db_path = fresh_db
    _common_monkeypatch(render_module, fake_engine, monkeypatch)
    monkeypatch.setattr(
        sys, "argv",
        _argv(
            "--series-id", "ser_fam",
            "--families", "flux",
            "--render-with-model", "gonzalomo_flux_v30",
            "--llm", "cydonia_heretic_24b",
        ),
    )

    rc = render_module.main()
    assert rc == 2
    captured = capsys.readouterr()
    assert "ERROR: no family-kind prompts in DB" in captured.err
    assert "--families flux" in captured.err
    assert "prepare_prompts.py" in captured.err
    fake_engine.run_phase_b.assert_not_called()


# ── 6. cross-kind isolation: model-kind prompts don't satisfy --families ──


def test_model_kind_prompts_dont_satisfy_families_query(
    render_module, fresh_db, capsys, monkeypatch,
):
    """If the series has model-kind prompts (e.g. from a previous
    --models run) but no family-kind prompts, --families still
    surfaces the missing-prompts error rather than silently rendering
    the wrong rows."""
    # Seed MODEL-kind prompts only.
    conn = sqlite3.connect(str(fresh_db))
    conn.execute(
        "INSERT INTO series (id, mode, content_level, style_profile_id, "
        "theme, target_count, status) "
        "VALUES ('ser_fam', 'character', 'T2_implied', "
        "'golden_hour_natural', 'test', 1, 'planned')",
    )
    conn.execute(
        "INSERT INTO scenes (id, series_id, variation_axis, content_level) "
        "VALUES ('ser_fam_sc_000', 'ser_fam', 'pose', 'T2_implied')",
    )
    conn.execute(
        "INSERT INTO prompts (id, series_id, scene_id, target_kind, "
        "model_id, llm_id, prompt_text, negative_prompt, prompt_hash, "
        "content_level, status) VALUES "
        "('pm_0', 'ser_fam', 'ser_fam_sc_000', 'model', 'gonzalomo_flux_v30', "
        "'cydonia_heretic_24b', 'model prompt', 'neg', 'mhash_0', "
        "'T2_implied', 'pending')",
    )
    conn.commit()
    conn.close()

    fake_engine = MagicMock()
    fake_engine.db_path = fresh_db
    _common_monkeypatch(render_module, fake_engine, monkeypatch)
    monkeypatch.setattr(
        sys, "argv",
        _argv(
            "--series-id", "ser_fam",
            "--families", "flux",
            "--render-with-model", "gonzalomo_flux_v30",
            "--llm", "cydonia_heretic_24b",
        ),
    )

    rc = render_module.main()
    assert rc == 2
    captured = capsys.readouterr()
    assert "ERROR: no family-kind prompts in DB" in captured.err
    fake_engine.run_phase_b.assert_not_called()
