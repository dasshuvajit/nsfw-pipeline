"""Prompt deduplicator — cross-level dedup for character mode.

Two dedup strategies:

  1. **Prompt hash** — exact match via SHA256. Threshold 0.9 (we only
     reject truly identical prompts since the hash is full-text).
  2. **Scene structural similarity** — compares environment + lighting +
     camera across ALL content levels for the same character (subscribers
     see both tiers, so they'd notice duplicates). Threshold 0.75.

The semantic similarity uses ``sentence-transformers`` when available.
If the package isn't installed, falls back to a simpler hash-based
approach that still catches exact and near-exact duplicates.

See ARCHITECTURE.md Section 5 (Cross-Level Scene Deduplication) and
Section 13 (Anti-Repetition).
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

from src.core.generation_context import GenerationContext
from src.prompt.builder import compute_prompt_hash

logger = logging.getLogger(__name__)

# Thresholds from ARCHITECTURE.md
PROMPT_HASH_THRESHOLD = 0.9
SCENE_STRUCTURAL_THRESHOLD = 0.75

# Try to import sentence-transformers for semantic similarity
_encoder = None
_HAS_SENTENCE_TRANSFORMERS = False
try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    pass


def _get_encoder():
    """Lazy-load the sentence-transformers encoder."""
    global _encoder
    if _encoder is None and _HAS_SENTENCE_TRANSFORMERS:
        logger.info("Loading sentence-transformers model for scene dedup …")
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _encoder


def _cosine_similarity(a, b) -> float:
    """Cosine similarity between two vectors."""
    import numpy as np
    a = np.array(a)
    b = np.array(b)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def _scene_text(scene: dict[str, Any]) -> str:
    """Build a text representation of a scene for similarity comparison."""
    parts = [
        scene.get("environment_detail", ""),
        scene.get("lighting", ""),
        scene.get("camera", ""),
        scene.get("pose", ""),
    ]
    return " ".join(p for p in parts if p)


def _prompt_hash(text: str) -> str:
    """Full SHA256 of the prompt text.

    Must match what ``prompts.prompt_hash`` stores — otherwise DB-hash
    dedup silently misses every existing row. ``compute_prompt_hash``
    is the single source of truth for that column.
    """
    return compute_prompt_hash(text)


class PromptDeduplicator:
    """Deduplicate prompts and scenes against the DB and within a batch.

    For character mode: checks across ALL content levels for the same
    character, not just within the current level. This prevents subscribers
    who see both soft and full from seeing near-identical scenes.
    """

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
          2. DB historical duplicates (prompt hash and scene similarity)

        Returns the deduplicated list (possibly shorter than input).
        """
        if not prompts:
            return prompts

        # Load existing hashes and scene texts from DB
        existing_hashes = self._load_existing_hashes(ctx)
        existing_scenes = self._load_existing_scenes(ctx)

        seen_hashes: set[str] = set()
        result: list[dict[str, Any]] = []

        for prompt in prompts:
            prompt_text = prompt.get("prompt_text", "")
            # Prefer the hash the engine already computed post-sanitize;
            # fall back to recomputing if a caller forgot to set it.
            h = prompt.get("prompt_hash") or _prompt_hash(prompt_text)

            # Within-batch dedup
            if h in seen_hashes:
                logger.debug("Dedup: dropping batch-duplicate prompt (hash=%s)", h[:8])
                continue

            # DB hash dedup
            if h in existing_hashes:
                logger.debug("Dedup: dropping DB-duplicate prompt (hash=%s)", h[:8])
                continue

            # Scene structural dedup (only for character mode with scenes)
            scene_id = prompt.get("scene_id")
            if scene_id and ctx.character_id and existing_scenes:
                scene = self._find_scene(scenes, scene_id)
                if scene and self._is_structurally_similar(scene, existing_scenes):
                    logger.debug(
                        "Dedup: dropping structurally similar scene %s",
                        scene_id,
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
        """Load prompt hashes from the DB for dedup."""
        if not self.db_path.exists():
            return set()
        conn = sqlite3.connect(str(self.db_path))
        try:
            # For character mode, check across ALL levels for this character
            if ctx.character_id:
                rows = conn.execute(
                    """
                    SELECT DISTINCT p.prompt_hash
                    FROM prompts p
                    JOIN series s ON p.series_id = s.id
                    WHERE s.character_id = ?
                    AND p.created_at > datetime('now', '-30 days')
                    """,
                    (ctx.character_id,),
                ).fetchall()
            else:
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

    def _load_existing_scenes(self, ctx: GenerationContext) -> list[str]:
        """Load recent scene text representations from the DB."""
        if not ctx.character_id or not self.db_path.exists():
            return []
        conn = sqlite3.connect(str(self.db_path))
        try:
            rows = conn.execute(
                """
                SELECT sc.pose, sc.environment_detail, sc.lighting, sc.camera
                FROM scenes sc
                JOIN series s ON sc.series_id = s.id
                WHERE s.character_id = ?
                AND sc.created_at > datetime('now', '-30 days')
                """,
                (ctx.character_id,),
            ).fetchall()
            return [
                " ".join(filter(None, [r[0], r[1], r[2], r[3]]))
                for r in rows
            ]
        finally:
            conn.close()

    def _is_structurally_similar(
        self,
        scene: dict[str, Any],
        existing_texts: list[str],
    ) -> bool:
        """Check if a scene is too similar to any existing scene."""
        new_text = _scene_text(scene)
        if not new_text:
            return False

        encoder = _get_encoder()
        if encoder is not None:
            # Semantic similarity via sentence-transformers
            new_emb = encoder.encode([new_text])[0]
            for old_text in existing_texts:
                if not old_text:
                    continue
                old_emb = encoder.encode([old_text])[0]
                sim = _cosine_similarity(new_emb, old_emb)
                if sim > SCENE_STRUCTURAL_THRESHOLD:
                    return True
            return False
        else:
            # Fallback: simple word-overlap similarity
            new_words = set(new_text.lower().split())
            for old_text in existing_texts:
                if not old_text:
                    continue
                old_words = set(old_text.lower().split())
                if not new_words or not old_words:
                    continue
                overlap = len(new_words & old_words) / max(len(new_words), len(old_words))
                if overlap > SCENE_STRUCTURAL_THRESHOLD:
                    return True
            return False

    @staticmethod
    def _find_scene(scenes: list[dict[str, Any]], scene_id: str) -> dict[str, Any] | None:
        """Find a scene by its id in the list."""
        for s in scenes:
            if s.get("id") == scene_id:
                return s
        return None
