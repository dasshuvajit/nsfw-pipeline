"""Tests for ``PipelineEngine._save_dry_run`` — the persistence boundary
between Phase A (LLM) and Phase B (render).

Audit found that ``_save_dry_run`` had no direct unit-test coverage.
Now that ``prompts.model_id`` is mandatory, that gap is risky. These
tests exercise the INSERT path against a real SQLite DB built from
``init_db.py``'s SCHEMA_SQL — so the new ``UNIQUE(scene_id, model_id)``
constraint and ``model_id NOT NULL`` are also exercised.

We mock ``GenerationContext`` minimally (only ``mode``, ``content_level``,
``character_id``, ``model_id`` are read by the persistence path; the
heavyweight ``model_config`` / ``family`` aren't touched).
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def fresh_db(tmp_path: Path) -> Path:
    """Build a real DB from init_db.py so the new constraints are live."""
    db_path = tmp_path / "engine_dry_run.db"
    result = subprocess.run(
        [sys.executable, "scripts/init_db.py", "--db-path", str(db_path)],
        capture_output=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr.decode()
    return db_path


@pytest.fixture
def engine(fresh_db: Path):
    """Build a real PipelineEngine pointed at the fresh DB.

    No model_override → no registry ping at construction time.
    """
    from src.core.engine import PipelineEngine
    minimal_config = {
        "execution": {"mode": "manual"},
        "compliance": {"commercial_mode": False},
        "comfyui": {
            "base_url": "http://127.0.0.1:8188",
            "output_dir": "/tmp/_test_comfy_out",
            "input_dir": "/tmp/_test_comfy_in",
            "render_timeout_seconds": 60,
            "max_retry_per_image": 1,
            "workflow_dir": "config/comfyui_workflows",
        },
        "set_builder": {"min_images": 1, "max_images": 5, "quality_cutoff": 0.0},
        "watermark": {"enabled": False},
        "postprocess": {"upscale_enabled": False, "face_refine_enabled": False},
        "scoring": {"use_hps_v2": False, "use_image_reward": False},
        "pipeline": {"output_dir": str(fresh_db.parent / "out")},
        "aspect_ratio_weights": {
            "T1_suggestive": {"portrait_23": 0.5, "square": 0.5},
            "T2_implied":    {"portrait_23": 0.5, "square": 0.5},
            "T3_artnude":    {"portrait_23": 0.7, "landscape": 0.3},
            "T4_explicit":   {"portrait_23": 0.7, "landscape": 0.3},
        },
        "mode_weights": {
            "character": 1.0, "theme": 0.0, "style": 0.0,
            "niche": 0.0, "variation": 0.0,
        },
    }
    return PipelineEngine(config=minimal_config, db_path=fresh_db, dry_run=True)


def _make_ctx(model_id: str = "lustify_v7") -> MagicMock:
    """A GenerationContext stub that exposes only what _save_dry_run reads.

    ``character_id=None`` so we don't have to seed a `characters` row
    just to satisfy the FK; ``series.character_id`` is nullable.
    """
    ctx = MagicMock()
    ctx.mode = "character"
    ctx.content_level = "T2_implied"
    ctx.character_id = None
    ctx.model_id = model_id
    # target_kind is read by _save_prompts_for_target post-Phase 2
    # rename. The dry-run path is always model-kind.
    ctx.target_kind = "model"
    # family.id is the family-kind fallback for target_id resolution;
    # never reached when target_kind='model' but defensive.
    ctx.family.id = "sdxl"
    return ctx


def _series_plan() -> dict:
    return {
        "theme": "test theme",
        "mood": "calm",
        "environment": "studio",
        "variation_axes": ["pose"],
    }


def _scene(scene_id: str) -> dict:
    return {
        "id": scene_id,
        "variation_axis": "pose",
        "pose": "standing",
        "camera": "85mm",
        "camera_angle": "eye-level",
        "lighting": "soft window light",
        "environment_detail": "studio with grey backdrop",
        "mood_note": "calm",
        "expression": "soft smile",
        "aspect_ratio": "portrait_23",
        "resolution_w": 832,
        "resolution_h": 1216,
    }


def _prompt(
    prompt_id: str, scene_id: str, *, model_id: str | None = None,
) -> dict:
    p = {
        "id": prompt_id,
        "scene_id": scene_id,
        "prompt_text": "elara, standing, 85mm, soft window light",
        "negative_prompt": "blurry, low quality",
        "prompt_hash": f"hash_{prompt_id}",
    }
    if model_id is not None:
        p["model_id"] = model_id
    return p


# ── Happy path ──────────────────────────────────────────────────────


def test_save_dry_run_inserts_series_scenes_prompts(
    engine, fresh_db: Path,
) -> None:
    ctx = _make_ctx(model_id="lustify_v7")
    engine._save_dry_run(
        series_id="ser1",
        ctx=ctx,
        series_plan=_series_plan(),
        scenes=[_scene("sc1"), _scene("sc2")],
        prompts=[_prompt("p1", "sc1"), _prompt("p2", "sc2")],
        style_profile_id="golden_hour_natural",
    )

    conn = sqlite3.connect(str(fresh_db))
    conn.row_factory = sqlite3.Row
    series = conn.execute("SELECT * FROM series WHERE id='ser1'").fetchone()
    assert series["mode"] == "character"
    assert series["content_level"] == "T2_implied"
    assert series["status"] == "dry_run"
    assert json.loads(series["llm_series_plan"])["theme"] == "test theme"

    scene_count = conn.execute(
        "SELECT COUNT(*) FROM scenes WHERE series_id='ser1'"
    ).fetchone()[0]
    assert scene_count == 2

    prompt_rows = conn.execute(
        "SELECT scene_id, model_id, status FROM prompts WHERE series_id='ser1' "
        "ORDER BY id"
    ).fetchall()
    assert len(prompt_rows) == 2
    assert all(r["model_id"] == "lustify_v7" for r in prompt_rows)
    assert all(r["status"] == "pending" for r in prompt_rows)


# ── model_id behavior ───────────────────────────────────────────────


def test_save_dry_run_uses_ctx_model_id_when_prompt_lacks_it(
    engine, fresh_db: Path,
) -> None:
    """Default behavior (single-model run_cycle path): per-prompt
    model_id is omitted; persistence falls back to ctx.model_id."""
    ctx = _make_ctx(model_id="chroma_v10HD")
    engine._save_dry_run(
        series_id="s",
        ctx=ctx,
        series_plan=_series_plan(),
        scenes=[_scene("sc1")],
        prompts=[_prompt("p1", "sc1")],   # no model_id key
        style_profile_id="golden_hour_natural",
    )
    conn = sqlite3.connect(str(fresh_db))
    row = conn.execute("SELECT model_id FROM prompts WHERE id='p1'").fetchone()
    assert row[0] == "chroma_v10HD"


def test_save_dry_run_per_prompt_model_id_overrides_ctx(
    engine, fresh_db: Path,
) -> None:
    """Multi-model run_phase_a path: per-prompt model_id wins over the
    ctx default."""
    ctx = _make_ctx(model_id="lustify_v7")  # ctx says one model …
    engine._save_dry_run(
        series_id="s",
        ctx=ctx,
        series_plan=_series_plan(),
        scenes=[_scene("sc1"), _scene("sc2")],
        prompts=[
            _prompt("p1", "sc1", model_id="lustify_v7"),
            _prompt("p2", "sc2", model_id="chroma_v10HD"),
        ],
        style_profile_id="golden_hour_natural",
    )
    conn = sqlite3.connect(str(fresh_db))
    rows = conn.execute(
        "SELECT id, model_id FROM prompts ORDER BY id"
    ).fetchall()
    assert dict(rows) == {"p1": "lustify_v7", "p2": "chroma_v10HD"}


# ── Multi-model on the same scene ───────────────────────────────────


def test_save_dry_run_two_models_one_scene(engine, fresh_db: Path) -> None:
    """The new (scene_id, model_id) UNIQUE constraint allows multiple
    prompts per scene as long as model_id differs."""
    ctx = _make_ctx(model_id="lustify_v7")
    # Need to seed series + scene first (we re-use ctx across two calls).
    engine._save_dry_run(
        series_id="s",
        ctx=ctx,
        series_plan=_series_plan(),
        scenes=[_scene("sc1")],
        prompts=[_prompt("p1", "sc1", model_id="lustify_v7")],
        style_profile_id="golden_hour_natural",
    )

    # Add a second prompt for the SAME scene under a DIFFERENT model
    # (simulates `prepare_prompts --series-id S --models chroma_v10HD`).
    conn = sqlite3.connect(str(fresh_db))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(
        "INSERT INTO prompts (id, series_id, scene_id, model_id, llm_id, "
        "prompt_text, negative_prompt, prompt_hash, content_level, status) "
        "VALUES ('p2', 's', 'sc1', 'chroma_v10HD', 'cydonia_24b_v43', 't', "
        "'n', 'h2', 'T2_implied', 'pending')"
    )
    conn.commit()
    rows = conn.execute(
        "SELECT model_id FROM prompts WHERE scene_id='sc1' ORDER BY model_id"
    ).fetchall()
    assert [r[0] for r in rows] == ["chroma_v10HD", "lustify_v7"]


def test_save_dry_run_duplicate_scene_model_pair_raises(
    engine, fresh_db: Path,
) -> None:
    """The UNIQUE(scene_id, model_id) constraint blocks the
    accidental-double-insert path."""
    ctx = _make_ctx(model_id="lustify_v7")
    engine._save_dry_run(
        series_id="s",
        ctx=ctx,
        series_plan=_series_plan(),
        scenes=[_scene("sc1")],
        prompts=[_prompt("p1", "sc1", model_id="lustify_v7")],
        style_profile_id="golden_hour_natural",
    )
    # Attempting to insert ANOTHER prompt for (sc1, lustify_v7) → IntegrityError.
    conn = sqlite3.connect(str(fresh_db))
    conn.execute("PRAGMA foreign_keys = ON")
    with pytest.raises(sqlite3.IntegrityError, match=r"(?i)unique"):
        conn.execute(
            "INSERT INTO prompts (id, series_id, scene_id, model_id, llm_id, "
            "prompt_text, negative_prompt, prompt_hash, content_level, status) "
            "VALUES ('p_dup', 's', 'sc1', 'lustify_v7', 'cydonia_24b_v43', "
            "'t', 'n', 'h_dup', 'T2_implied', 'pending')"
        )


# ── Schema constraint enforcement ───────────────────────────────────


def test_save_dry_run_rejects_null_model_id(engine, fresh_db: Path) -> None:
    """model_id NOT NULL — explicit None must crash before the INSERT.

    Post-Phase 2 (family-level prompt-prep feature): the rename to
    ``_save_prompts_for_target`` adds a defensive ``ValueError`` that
    fires before the SQL INSERT when target_id cannot be resolved
    (per-prompt key absent AND ctx.model_id None AND ctx.target_kind
    is 'model'). This is preferable to letting SQLite raise a generic
    NOT NULL — the message names the offending prompt id.
    """
    ctx = _make_ctx(model_id=None)
    ctx.model_id = None  # explicit; _make_ctx default is overridden
    # prompt also has no model_id → target_id resolves to None →
    # ValueError before INSERT.
    with pytest.raises(ValueError, match="cannot resolve target_id"):
        engine._save_dry_run(
            series_id="s",
            ctx=ctx,
            series_plan=_series_plan(),
            scenes=[_scene("sc1")],
            prompts=[_prompt("p1", "sc1")],
            style_profile_id="golden_hour_natural",
        )
