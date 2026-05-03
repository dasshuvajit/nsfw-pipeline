"""Character DB access — load, select, mark-used.

Thin wrapper over the `characters` table from ARCHITECTURE.md Section 3.
The three operations the engine actually needs are:

    get_character(id)             — direct lookup by primary key
    get_least_recently_used()     — Mode 1 (character) auto-selection
    update_usage(id, image_count) — bookkeeping after each render

Design notes:

1. **Per-operation connections.** SQLite connections are cheap and the
   character path is not hot. A fresh connection per call keeps things
   thread-safe and removes any "did we forget to close" footguns. The
   alternative (long-lived connection) would force callers to think
   about lifecycle for almost no perf benefit.

2. **Foreign keys ON.** SQLite ships with FK enforcement disabled by
   default. We turn it on per-connection so an INSERT/UPDATE that
   references a missing `style_profiles.id` fails loudly instead of
   silently producing a dangling row. This matters for `update_usage`
   (FK is fine, no risk) and for any future writes.

3. **`update_usage` is safe to call repeatedly.** It only touches
   `total_images_generated` and `last_used_at`, never `base_prompt`,
   so the `protect_character_base_prompt` trigger never fires.

4. **LRU ordering.** SQLite sorts NULL as smaller than any value in
   `ORDER BY ASC`, so a never-used character (NULL `last_used_at`)
   wins over any character that *has* been used. Tiebreaker is
   `created_at ASC` so the oldest-created never-used character is
   picked first — predictable bootstrap order across runs.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


class CharacterError(Exception):
    """Base for character manager errors."""


class CharacterNotFound(CharacterError):
    """Lookup by id returned no row."""


class NoActiveCharacters(CharacterError):
    """get_least_recently_used() found no candidates at all."""


class CharacterManager:
    """DB-backed CRUD for the `characters` table."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser()
        if not self.db_path.exists():
            raise CharacterError(
                f"Database not found at {self.db_path}. "
                f"Run `python scripts/init_db.py` first."
            )

    # ----- public API -------------------------------------------------------

    def get_character(self, character_id: str) -> dict[str, Any]:
        """Return one character row as a dict.

        Raises CharacterNotFound if no row matches.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM characters WHERE id = ?", (character_id,)
            ).fetchone()
        if row is None:
            raise CharacterNotFound(f"No character with id={character_id!r}")
        return dict(row)

    def get_least_recently_used(
        self, *, active_only: bool = True
    ) -> dict[str, Any]:
        """Return the active character with the oldest `last_used_at`.

        Never-used characters (NULL `last_used_at`) sort first, so a
        freshly bootstrapped character is always picked before any
        previously rendered one. Tiebreaker is `created_at ASC`.

        Raises NoActiveCharacters if the table has no candidates.
        """
        sql = "SELECT * FROM characters"
        params: tuple[Any, ...] = ()
        if active_only:
            sql += " WHERE active = 1"
        sql += " ORDER BY last_used_at ASC, created_at ASC LIMIT 1"

        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
        if row is None:
            raise NoActiveCharacters(
                "No active characters in the database. "
                "Run `python scripts/bootstrap_character.py` first."
            )
        return dict(row)

    def update_usage(self, character_id: str, image_count: int) -> None:
        """Increment usage counters after a render.

        Adds ``image_count`` to ``total_images_generated`` and stamps
        ``last_used_at`` to NOW. Raises CharacterNotFound if the id
        doesn't exist (rather than silently no-op'ing — that bug pattern
        is too easy to miss in supervised mode).
        """
        if image_count < 0:
            raise CharacterError(
                f"image_count must be >= 0 (got {image_count})"
            )
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE characters
                SET total_images_generated = total_images_generated + ?,
                    last_used_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (image_count, character_id),
            )
            if cursor.rowcount == 0:
                raise CharacterNotFound(
                    f"update_usage: no character with id={character_id!r}"
                )
            conn.commit()

    # ----- internals --------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        # SQLite ships with FK enforcement OFF — turn it on so writes
        # against a missing style_profiles.id fail loudly.
        conn.execute("PRAGMA foreign_keys = ON")
        return conn
