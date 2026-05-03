"""Generation memory — anti-repetition tracker.

Records themes, scenes, and prompts to the ``generation_memory`` table
so the LLM doesn't repeat itself across runs. For character mode,
scenes are checked across ALL content levels (``character_id``-based)
because subscribers who see both soft and full should never encounter
near-identical scenes at different intensity levels.

See ARCHITECTURE.md Section 13 (Anti-Repetition with Cross-Level Awareness).
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MemoryManager:
    """Read/write access to the ``generation_memory`` table."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser()

    def is_novel(
        self,
        content: str,
        content_type: str,
        *,
        content_level: str | None = None,
        character_id: str | None = None,
    ) -> bool:
        """Check if ``content`` is novel. If yes, record it and return True.

        For scenes with a ``character_id``: checks across ALL content levels
        for that character (cross-level dedup). For other content: checks
        within the same content_level.

        Parameters
        ----------
        content : str
            The text to check (theme name, scene description, prompt text).
        content_type : str
            One of ``'theme'``, ``'scene'``, ``'prompt'``, ``'style'``, ``'niche'``.
        content_level : str | None
            Current content level (for non-character dedup).
        character_id : str | None
            Character id (for cross-level dedup in character mode).
        """
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        query = "SELECT 1 FROM generation_memory WHERE content_hash = ? AND type = ?"
        params: list[Any] = [content_hash, content_type]

        # For scenes: check across levels for same character
        if character_id and content_type == "scene":
            query += " AND character_id = ?"
            params.append(character_id)
        elif content_level:
            query += " AND content_level = ?"
            params.append(content_level)

        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            exists = conn.execute(query, params).fetchone()
            if exists:
                return False

            conn.execute(
                """
                INSERT INTO generation_memory
                    (type, content_hash, content_preview,
                     content_level, character_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    content_type,
                    content_hash,
                    content[:200],
                    content_level,
                    character_id,
                ),
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def record_series(
        self,
        series_plan: dict[str, Any],
        scenes: list[dict[str, Any]],
        prompts: list[dict[str, Any]],
        *,
        content_level: str,
        character_id: str | None = None,
    ) -> dict[str, int]:
        """Record all outputs from a pipeline run.

        Returns a summary of how many items were recorded vs. skipped
        (already in memory).
        """
        recorded = {"themes": 0, "scenes": 0, "prompts": 0}
        skipped = {"themes": 0, "scenes": 0, "prompts": 0}

        # Record theme
        theme = series_plan.get("theme", "")
        if theme:
            if self.is_novel(
                theme, "theme",
                content_level=content_level,
                character_id=character_id,
            ):
                recorded["themes"] += 1
            else:
                skipped["themes"] += 1

        # Record scenes
        for scene in scenes:
            scene_text = " ".join(
                filter(None, [
                    scene.get("environment_detail", ""),
                    scene.get("lighting", ""),
                    scene.get("camera", ""),
                    scene.get("pose", ""),
                    scene.get("mood_note", ""),
                ])
            )
            if scene_text:
                if self.is_novel(
                    scene_text, "scene",
                    content_level=content_level,
                    character_id=character_id,
                ):
                    recorded["scenes"] += 1
                else:
                    skipped["scenes"] += 1

        # Record prompts
        for prompt in prompts:
            text = prompt.get("prompt_text", "")
            if text:
                if self.is_novel(
                    text, "prompt",
                    content_level=content_level,
                    character_id=character_id,
                ):
                    recorded["prompts"] += 1
                else:
                    skipped["prompts"] += 1

        logger.info(
            "Memory: recorded %d themes, %d scenes, %d prompts "
            "(skipped %d/%d/%d already known)",
            recorded["themes"], recorded["scenes"], recorded["prompts"],
            skipped["themes"], skipped["scenes"], skipped["prompts"],
        )
        return recorded
