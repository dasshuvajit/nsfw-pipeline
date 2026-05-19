"""Prompt deduplicator — within-level prompt-hash dedup.

After the 2026-05-20 cleanup (character mode deleted), this becomes a
single-strategy dedup: prompt-hash match against existing DB rows + the
current batch. Cross-level scene-structural similarity (the
sentence-transformers-based path that only fired for character mode) is
gone.

See ARCHITECTURE.md Section 13 (Anti-Repetition).
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

from src.core.generation_context import GenerationContext
from src.prompt.builder import compute_prompt_hash

logger = logging.getLogger(__name__)

# Threshold from ARCHITECTURE.md (kept for documentation parity).
PROMPT_HASH_THRESHOLD = 0.9


def _prompt_hash(text: str) -> str:
    """Full SHA256 of the prompt text.

    Must match what ``prompts.prompt_hash`` stores — otherwise DB-hash
    dedup silently misses every existing row. ``compute_prompt_hash``
    is the single source of truth for that column.
    """
    return compute_prompt_hash(text)


class PromptDeduplicator:
    """Deduplicate prompts against the DB and within a batch via the
    per-row prompt_hash column. Per-content_level scope."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path).expanduser()

    def deduplicate(
        self,
        prompts: list[dict[str, Any]],
        scenes: list[dict[str, Any]],
        ctx: GenerationContext,
    ) -> list[dict[str, Any]]:
        """Remove duplicate prompts from the batch.

        Checks both:
          1. Within-batch duplicates (prompt hash match)
          2. DB historical duplicates (prompt hash match within content_level)

        Returns the deduplicated list (possibly shorter than input).
        """
        if not prompts:
            return prompts

        existing_hashes = self._load_existing_hashes(ctx)
        seen_hashes: set[str] = set()
        result: list[dict[str, Any]] = []

        for prompt in prompts:
            prompt_text = prompt.get("prompt_text", "")
            h = prompt.get("prompt_hash") or _prompt_hash(prompt_text)

            if h in seen_hashes:
                logger.debug(
                    "Dedup: dropping batch-duplicate prompt (hash=%s)",
                    h[:8],
                )
                continue

            if h in existing_hashes:
                logger.debug(
                    "Dedup: dropping DB-duplicate prompt (hash=%s)", h[:8],
                )
                continue

            seen_hashes.add(h)
            result.append(prompt)

        dropped = len(prompts) - len(result)
        if dropped > 0:
            logger.info(
                "Dedup: dropped %d/%d prompts (%d remaining)",
                dropped, len(prompts), len(result),
            )
        return result

    def _load_existing_hashes(self, ctx: GenerationContext) -> set[str]:
        """Load prompt hashes from the DB for dedup (within content_level,
        last 30 days)."""
        if not self.db_path.exists():
            return set()
        conn = sqlite3.connect(str(self.db_path))
        try:
            rows = conn.execute(
                """
                SELECT DISTINCT prompt_hash FROM prompts
                WHERE content_level = ?
                AND created_at > datetime('now', '-30 days')
                """,
                (ctx.content_level,),
            ).fetchall()
            return {r[0] for r in rows if r[0]}
        finally:
            conn.close()
