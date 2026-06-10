"""Generation memory — anti-repetition tracker.

Records themes, scenes, and prompts to the ``generation_memory`` table
so the LLM doesn't repeat itself across runs. Dedup is per-content_level
(no cross-level character tracking after the 2026-05-20 character-mode
deletion).

See ARCHITECTURE.md Section 13 (Anti-Repetition).
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
    ) -> bool:
        """Check if ``content`` is novel. If yes, record it and return True.

        Parameters
        ----------
        content : str
            The text to check (theme name, scene description, prompt text).
        content_type : str
            One of ``'theme'``, ``'scene'``, ``'prompt'``, ``'style'``, ``'niche'``.
        content_level : str | None
            Current content level for per-level dedup.
        """
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        query = "SELECT 1 FROM generation_memory WHERE content_hash = ? AND type = ?"
        params: list[Any] = [content_hash, content_type]
        if content_level:
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
                    (type, content_hash, content_preview, content_level)
                VALUES (?, ?, ?, ?)
                """,
                (
                    content_type,
                    content_hash,
                    content[:200],
                    content_level,
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
    ) -> dict[str, int]:
        """Record all outputs from a pipeline run.

        Returns a summary of how many items were recorded vs. skipped
        (already in memory).
        """
        recorded = {"themes": 0, "scenes": 0, "prompts": 0}
        skipped = {"themes": 0, "scenes": 0, "prompts": 0}

        theme = series_plan.get("theme", "")
        if theme:
            if self.is_novel(theme, "theme", content_level=content_level):
                recorded["themes"] += 1
            else:
                skipped["themes"] += 1

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
                if self.is_novel(scene_text, "scene", content_level=content_level):
                    recorded["scenes"] += 1
                else:
                    skipped["scenes"] += 1

        for prompt in prompts:
            text = prompt.get("prompt_text", "")
            if text:
                if self.is_novel(text, "prompt", content_level=content_level):
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
