"""Prompt dedup — hash determinism, within-batch match, DB lookup."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.prompt.builder import compute_prompt_hash
from src.prompt.deduplicator import PromptDeduplicator


# ---- Hash determinism -------------------------------------------------

def test_compute_prompt_hash_is_deterministic():
    a = compute_prompt_hash("elara, seated, soft light")
    b = compute_prompt_hash("elara, seated, soft light")
    assert a == b
    assert len(a) == 64  # SHA256 hex


def test_compute_prompt_hash_differs_per_input():
    a = compute_prompt_hash("elara, seated")
    b = compute_prompt_hash("elara, standing")
    assert a != b


# ---- Deduplicator with empty DB --------------------------------------

@pytest.fixture
def fresh_db(tmp_path: Path) -> Path:
    """An on-disk DB with the prompts/scenes/series tables stubbed out."""
    db = tmp_path / "dedup_test.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE series (
            id TEXT PRIMARY KEY,
            character_id TEXT,
            content_level TEXT
        );
        CREATE TABLE prompts (
            id TEXT PRIMARY KEY,
            series_id TEXT,
            prompt_hash TEXT,
            content_level TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE scenes (
            id TEXT PRIMARY KEY,
            series_id TEXT,
            pose TEXT,
            environment_detail TEXT,
            lighting TEXT,
            camera TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.commit()
    conn.close()
    return db


def _stub_ctx(character_id: str | None = None,
              content_level: str = "T2_implied"):
    ctx = MagicMock()
    ctx.character_id = character_id
    ctx.content_level = content_level
    return ctx


# ---- Within-batch dedup ----------------------------------------------

def test_within_batch_exact_hash_collapse(fresh_db):
    dedup = PromptDeduplicator(fresh_db)
    prompts = [
        {"prompt_text": "elara, seated", "prompt_hash": "h1"},
        {"prompt_text": "elara, seated", "prompt_hash": "h1"},  # dup
        {"prompt_text": "elara, standing", "prompt_hash": "h2"},
    ]
    ctx = _stub_ctx()
    out = dedup.deduplicate(prompts, scenes=[], ctx=ctx)
    assert len(out) == 2
    hashes = {p["prompt_hash"] for p in out}
    assert hashes == {"h1", "h2"}


def test_empty_prompts_returns_empty(fresh_db):
    dedup = PromptDeduplicator(fresh_db)
    assert dedup.deduplicate([], scenes=[], ctx=_stub_ctx()) == []


# ---- DB-level dedup --------------------------------------------------

def test_db_hash_match_drops_existing(fresh_db):
    # Seed the DB with an existing prompt for a character
    conn = sqlite3.connect(str(fresh_db))
    conn.execute("INSERT INTO series (id, character_id) VALUES ('s1', 'c1')")
    conn.execute(
        "INSERT INTO prompts (id, series_id, prompt_hash, content_level) "
        "VALUES ('p1', 's1', 'existing_hash', 'T2_implied')"
    )
    conn.commit()
    conn.close()

    dedup = PromptDeduplicator(fresh_db)
    prompts = [
        {"prompt_text": "already stored", "prompt_hash": "existing_hash"},
        {"prompt_text": "brand new", "prompt_hash": "novel_hash"},
    ]
    ctx = _stub_ctx(character_id="c1")
    out = dedup.deduplicate(prompts, scenes=[], ctx=ctx)
    assert len(out) == 1
    assert out[0]["prompt_hash"] == "novel_hash"


def test_db_hash_dedup_by_content_level_when_no_character(fresh_db):
    # Non-character mode — dedup by content_level across recent window
    conn = sqlite3.connect(str(fresh_db))
    conn.execute(
        "INSERT INTO prompts (id, series_id, prompt_hash, content_level) "
        "VALUES ('p1', 'sX', 'existing', 'T2_implied')"
    )
    conn.commit()
    conn.close()

    dedup = PromptDeduplicator(fresh_db)
    prompts = [
        {"prompt_text": "dup", "prompt_hash": "existing"},
        {"prompt_text": "fresh", "prompt_hash": "fresh_hash"},
    ]
    ctx = _stub_ctx(character_id=None, content_level="T2_implied")
    out = dedup.deduplicate(prompts, scenes=[], ctx=ctx)
    assert [p["prompt_hash"] for p in out] == ["fresh_hash"]


# ---- Missing-DB safety net -------------------------------------------

def test_missing_db_file_degrades_to_batch_only(tmp_path: Path):
    ghost = tmp_path / "does_not_exist.db"
    dedup = PromptDeduplicator(ghost)
    prompts = [
        {"prompt_text": "a", "prompt_hash": "x"},
        {"prompt_text": "a", "prompt_hash": "x"},  # batch dup
    ]
    out = dedup.deduplicate(prompts, scenes=[], ctx=_stub_ctx())
    # Only batch dedup runs — no crash on missing DB
    assert len(out) == 1


# ---- Hash recompute fallback -----------------------------------------

def test_missing_hash_is_recomputed(fresh_db):
    """If prompt lacks prompt_hash, deduplicator must recompute."""
    dedup = PromptDeduplicator(fresh_db)
    prompts = [
        {"prompt_text": "elara, seated"},
        {"prompt_text": "elara, seated"},  # same text → same recomputed hash
    ]
    out = dedup.deduplicate(prompts, scenes=[], ctx=_stub_ctx())
    assert len(out) == 1
