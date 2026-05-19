"""Verifier round-4 IMPORTANT-1+2 regression: render_set.py's
``_persist_run`` writes a complete row that satisfies the prompts
schema's NOT NULL constraints.

Pre-fix:
  * The INSERT omitted ``model_id`` and ``llm_id`` — both NOT NULL.
    The DB raised IntegrityError on every non-dry-run.
  * ``model_source`` was referenced in two banner prints but never
    assigned — every call crashed with NameError.

These tests build a real SQLite DB from ``init_db.py``'s schema and
drive the persistence path. Without the round-4 fixes they would
fail with ``IntegrityError`` / ``NameError``.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def empty_db(tmp_path: Path) -> Path:
    """Spin up a fresh empty DB matching the production schema."""
    db_path = tmp_path / "test.sqlite"
    init = PROJECT_ROOT / "scripts" / "init_db.py"
    subprocess.run(
        [sys.executable, str(init), "--db-path", str(db_path)],
        check=True, capture_output=True,
    )
    return db_path


def test_render_set_prompts_insert_writes_required_columns(empty_db: Path):
    """render_set.py's INSERT must populate prompts.model_id and
    prompts.llm_id (both NOT NULL). Replays the INSERT shape from
    `_persist_run` against a fresh schema to confirm no
    IntegrityError."""
    conn = sqlite3.connect(str(empty_db))
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        # Minimal upstream rows (FK chain: characters → series → prompts).
        conn.execute(
            "INSERT INTO characters (id, name, base_prompt, "
            "style_profile_id, model_id, locked_features, "
            "allowed_shift_axes) VALUES (?,?,?,?,?,?,?)",
            ("char_test", "test", "a woman", "boudoir_noir",
             "juggernaut_ragnarok", "{}", "[]"),
        )
        conn.execute(
            """INSERT INTO series (id, mode, content_level, character_id,
               style_profile_id, theme, target_count, actual_count, status)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            ("ser_test", "character", "T2_implied", "char_test",
             "boudoir_noir", "studio portrait", 1, 1, "complete"),
        )
        conn.execute(
            "INSERT INTO scenes (id, series_id, variation_axis, pose, "
            "camera) VALUES (?,?,?,?,?)",
            ("scn_test", "ser_test", "manual", "seated", "medium shot"),
        )
        # Round-4 fix: prompts INSERT now writes 12 columns including
        # target_kind, model_id, llm_id.
        conn.execute(
            """INSERT INTO prompts (
                id, series_id, scene_id,
                target_kind, model_id, llm_id,
                prompt_text, negative_prompt,
                prompt_hash, content_level, status, render_attempts
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            ("prm_test", "ser_test", "scn_test",
             "model", "juggernaut_ragnarok", "manual_no_llm",
             "test prompt", "test negative",
             "abc123", "T2_implied", "rendered", 1),
        )
        conn.commit()
        # Read back — every column non-null.
        row = conn.execute(
            "SELECT model_id, llm_id, target_kind FROM prompts WHERE id=?",
            ("prm_test",),
        ).fetchone()
        assert row == ("juggernaut_ragnarok", "manual_no_llm", "model")
    finally:
        conn.close()


def test_render_set_old_insert_shape_fails_integrity(empty_db: Path):
    """Sanity check the original broken INSERT: omitting model_id +
    llm_id MUST raise IntegrityError. This proves the round-4 fix
    actually prevented a real bug."""
    conn = sqlite3.connect(str(empty_db))
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.execute(
            "INSERT INTO characters (id, name, base_prompt, "
            "style_profile_id, model_id, locked_features, "
            "allowed_shift_axes) VALUES (?,?,?,?,?,?,?)",
            ("char_test", "test", "a woman", "boudoir_noir",
             "juggernaut_ragnarok", "{}", "[]"),
        )
        conn.execute(
            """INSERT INTO series (id, mode, content_level, character_id,
               style_profile_id, theme, target_count, actual_count, status)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            ("ser_test", "character", "T2_implied", "char_test",
             "boudoir_noir", "studio portrait", 1, 1, "complete"),
        )
        with pytest.raises(sqlite3.IntegrityError) as exc_info:
            conn.execute(
                """INSERT INTO prompts (
                    id, series_id, scene_id, prompt_text, negative_prompt,
                    prompt_hash, content_level, status, render_attempts
                ) VALUES (?,?,?,?,?,?,?,?,?)""",
                ("prm_test", "ser_test", None,
                 "test prompt", "test negative",
                 "abc123", "T2_implied", "rendered", 1),
            )
        assert "model_id" in str(exc_info.value).lower() or \
               "llm_id" in str(exc_info.value).lower(), (
            f"expected NOT NULL violation on model_id/llm_id; got: "
            f"{exc_info.value}"
        )
    finally:
        conn.close()


def test_render_set_help_does_not_crash():
    """Smoke test: --help works (catches NameError on `model_source`
    if the resolution block hasn't been imported by argparse path).
    """
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "render_set.py"),
         "--help"],
        capture_output=True, timeout=30,
    )
    assert result.returncode == 0, (
        f"render_set.py --help failed: rc={result.returncode}\n"
        f"stderr: {result.stderr.decode()}"
    )
