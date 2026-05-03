"""Scene-facets repo — per-(scene, family, LLM) LLM expansion of a scene.

A scene's model-agnostic core (pose, camera, lighting, environment,
mood, expression, etc.) lives in ``scenes``. The family-shaped LLM
output that a composer needs to produce a prompt — booru_tags for
pony, scene_prose for flux/chroma/illustrious/flux2, camera_spec +
clothing for sdxl — lives here, keyed by ``(scene_id, family, llm_id)``.

The triple key (added in the multi-LLM upgrade) lets the same scene be
re-expanded by a different LLM without overwriting the prior LLM's
output — manual A/B comparison ("Cydonia vs Magnum on this scene") is
the primary user workflow.

Sibling models in the same family share a single facet row per LLM
(e.g. ``lustify_v7`` and ``juggernaut_ragnarok`` both map to one sdxl
facet for a given LLM); the per-model differentiation lives in
``prompts`` via the composer + per-model trigger words / negative axes.

Design notes mirror ``character_manager.py``:

1. **Per-operation connections.** Cheap; thread-safe; no lifecycle
   footguns.
2. **Foreign keys ON per connection.** A facet INSERT against a
   missing ``scenes.id`` fails loudly.
3. **Idempotency lives at the caller.** The repo writes; ``engine``
   chooses whether to skip / replace / regen by calling ``has_facet``
   first or wrapping ``insert_facet`` in a try/except for the PRIMARY
   KEY conflict.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


_FACET_FIELDS: tuple[str, ...] = (
    "booru_tags",
    "source_tag",
    "scene_prose",
    "camera_spec",
    "clothing",
    # Phase 4a — abstract concept tags. The composer canonicalises
    # these into family-shaped phrasing at compose time via
    # src/prompt/vocabulary.py::canonicalize_facet.
    "realism_camera",
    "realism_lens",
    "realism_film_stock",
    "art_style_reference",
    "lighting_directive",
    "mood_aesthetic",
    "nsfw_anatomy",
    "nsfw_posture",
    "nsfw_act",
)


class SceneFacetError(Exception):
    """Base for scene_facets repo errors."""


class SceneFacetNotFound(SceneFacetError):
    """Lookup returned no row for the requested (scene_id, family, llm_id)."""


class SceneFacetExists(SceneFacetError):
    """Insert hit the (scene_id, family, llm_id) PRIMARY KEY — caller
    must explicitly delete first if it wants to overwrite."""


def _connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def has_facet(
    db_path: str | Path, scene_id: str, family: str, llm_id: str,
) -> bool:
    """Cheap existence check used by the engine to decide whether to
    call the LLM. Returns True only if a row exists for the exact
    ``(scene_id, family, llm_id)`` triple."""
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT 1 FROM scene_facets "
            "WHERE scene_id = ? AND family = ? AND llm_id = ?",
            (scene_id, family, llm_id),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def get_facet(
    db_path: str | Path, scene_id: str, family: str, llm_id: str,
) -> dict[str, Any]:
    """Return the facet row for one ``(scene, family, llm_id)`` as a dict.

    Raises ``SceneFacetNotFound`` when no row exists. The returned
    dict has keys: ``booru_tags``, ``source_tag``, ``scene_prose``,
    ``camera_spec``, ``clothing`` (each may be None depending on the
    family). ``scene_id``, ``family``, ``llm_id``, ``created_at`` are
    also present.
    """
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM scene_facets "
            "WHERE scene_id = ? AND family = ? AND llm_id = ?",
            (scene_id, family, llm_id),
        ).fetchone()
        if row is None:
            raise SceneFacetNotFound(
                f"no scene_facets row for scene_id={scene_id!r} "
                f"family={family!r} llm_id={llm_id!r}"
            )
        return dict(row)
    finally:
        conn.close()


def get_facets_for_scene(
    db_path: str | Path,
    scene_id: str,
    llm_id: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Return facet rows for one scene, keyed by family.

    When ``llm_id`` is provided, only that LLM's facets are returned
    (this is the normal lookup path during render). When ``llm_id`` is
    ``None``, every LLM's facets across every family are returned —
    useful at re-target time to discover which (family, llm_id)
    combinations already exist vs. still need an LLM call.

    With ``llm_id=None`` the result is keyed ``"<family>:<llm_id>"``
    to disambiguate per-(family, llm_id) rows; with a concrete
    ``llm_id`` the result is keyed by family alone (compatible with
    the pre-multi-LLM shape).
    """
    conn = _connect(db_path)
    try:
        if llm_id is None:
            rows = conn.execute(
                "SELECT * FROM scene_facets WHERE scene_id = ?",
                (scene_id,),
            ).fetchall()
            return {
                f"{row['family']}:{row['llm_id']}": dict(row)
                for row in rows
            }
        rows = conn.execute(
            "SELECT * FROM scene_facets "
            "WHERE scene_id = ? AND llm_id = ?",
            (scene_id, llm_id),
        ).fetchall()
        return {row["family"]: dict(row) for row in rows}
    finally:
        conn.close()


def insert_facet(
    db_path: str | Path,
    scene_id: str,
    family: str,
    facet: dict[str, Any],
    llm_id: str,
) -> None:
    """Insert a new facet row.

    ``facet`` is a dict whose keys are a subset of the family-shaped
    fields (booru_tags, source_tag, scene_prose, camera_spec,
    clothing). Missing keys are stored as NULL. Raises
    ``SceneFacetExists`` on PRIMARY KEY conflict — the caller must
    explicitly ``delete_facet`` first to overwrite.
    """
    cols = ["scene_id", "family", "llm_id", *_FACET_FIELDS]
    values = [
        scene_id,
        family,
        llm_id,
        *(facet.get(k) for k in _FACET_FIELDS),
    ]
    placeholders = ", ".join(["?"] * len(cols))
    conn = _connect(db_path)
    try:
        try:
            conn.execute(
                f"INSERT INTO scene_facets ({', '.join(cols)}) "
                f"VALUES ({placeholders})",
                values,
            )
        except sqlite3.IntegrityError as exc:
            if "PRIMARY KEY" in str(exc) or "UNIQUE" in str(exc):
                raise SceneFacetExists(
                    f"scene_facets row already exists for scene_id="
                    f"{scene_id!r} family={family!r} llm_id={llm_id!r}. "
                    f"Caller must delete_facet() first to overwrite."
                ) from exc
            raise
        conn.commit()
    finally:
        conn.close()


def delete_facet(
    db_path: str | Path, scene_id: str, family: str, llm_id: str,
) -> int:
    """Delete one facet row. Returns the number of rows deleted (0 or 1).

    Used by ``prepare_prompts --regen-facets`` to clear before a fresh
    LLM call.
    """
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            "DELETE FROM scene_facets "
            "WHERE scene_id = ? AND family = ? AND llm_id = ?",
            (scene_id, family, llm_id),
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def delete_facets_for_family(
    db_path: str | Path,
    scene_ids: list[str],
    family: str,
    llm_id: str,
) -> int:
    """Bulk-delete one ``(family, llm_id)`` pair's facets across many scenes.

    Used by ``prepare_prompts --regen-facets <family> --llm <id>`` when
    re-rolling every scene in a series for one LLM. Other LLMs' facets
    on the same scenes are untouched.
    """
    if not scene_ids:
        return 0
    placeholders = ", ".join(["?"] * len(scene_ids))
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            f"DELETE FROM scene_facets "
            f"WHERE family = ? AND llm_id = ? "
            f"AND scene_id IN ({placeholders})",
            (family, llm_id, *scene_ids),
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()
