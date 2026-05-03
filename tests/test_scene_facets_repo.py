"""Tests for src/memory/scene_facets_repo.py — the per-(scene, family,
llm_id) facet store.

Builds a real SQLite DB in tmp_path via init_db's schema (so the
``scene_facets.family`` CHECK constraint and ``scenes`` FK are
exercised), then exercises every CRUD path. No mocks.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from src.memory.scene_facets_repo import (
    SceneFacetExists,
    SceneFacetNotFound,
    delete_facet,
    delete_facets_for_family,
    get_facet,
    get_facets_for_scene,
    has_facet,
    insert_facet,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Default llm_id used by every single-LLM test below. Picked once at
# the top so a future rename only touches this constant.
LLM = "test_llm"


@pytest.fixture
def db(tmp_path: Path) -> Path:
    """Init a real DB via scripts/init_db.py and seed one series + scene."""
    db_path = tmp_path / "facets.db"
    result = subprocess.run(
        [sys.executable, "scripts/init_db.py", "--db-path", str(db_path)],
        capture_output=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr.decode()

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute(
        "INSERT INTO series (id, mode, content_level, style_profile_id, theme, "
        "target_count, status) "
        "VALUES ('s1', 'character', 'T2_implied', 'golden_hour_natural', "
        "'test theme', 1, 'planned')"
    )
    conn.execute(
        "INSERT INTO scenes (id, series_id, variation_axis, content_level) "
        "VALUES ('sc1', 's1', 'test', 'T2_implied')"
    )
    conn.execute(
        "INSERT INTO scenes (id, series_id, variation_axis, content_level) "
        "VALUES ('sc2', 's1', 'test', 'T2_implied')"
    )
    conn.commit()
    conn.close()
    return db_path


# ── has_facet / get_facet on missing rows ───────────────────────────


def test_has_facet_returns_false_when_missing(db: Path) -> None:
    assert has_facet(db, "sc1", "sdxl", LLM) is False


def test_get_facet_raises_not_found_when_missing(db: Path) -> None:
    with pytest.raises(SceneFacetNotFound, match="no scene_facets row"):
        get_facet(db, "sc1", "sdxl", LLM)


def test_get_facets_for_scene_empty_when_no_facets(db: Path) -> None:
    assert get_facets_for_scene(db, "sc1") == {}
    assert get_facets_for_scene(db, "sc1", llm_id=LLM) == {}


# ── insert_facet round-trip ─────────────────────────────────────────


def test_insert_then_get_round_trip(db: Path) -> None:
    insert_facet(
        db, "sc1", "sdxl",
        {"camera_spec": "85mm f/1.4", "clothing": "ivory silk dress"},
        LLM,
    )
    facet = get_facet(db, "sc1", "sdxl", LLM)
    assert facet["camera_spec"] == "85mm f/1.4"
    assert facet["clothing"] == "ivory silk dress"
    # Unset fields → NULL → None on read.
    assert facet["booru_tags"] is None
    assert facet["scene_prose"] is None
    assert facet["source_tag"] is None


def test_insert_then_has_facet_returns_true(db: Path) -> None:
    insert_facet(db, "sc1", "pony", {"booru_tags": "long_hair, blue_eyes"}, LLM)
    assert has_facet(db, "sc1", "pony", LLM) is True


def test_insert_partial_facet_stores_only_given_fields(db: Path) -> None:
    insert_facet(db, "sc1", "flux", {"scene_prose": "She sits on a bench."}, LLM)
    facet = get_facet(db, "sc1", "flux", LLM)
    assert facet["scene_prose"] == "She sits on a bench."
    assert facet["camera_spec"] is None
    assert facet["clothing"] is None


def test_insert_includes_metadata_columns_in_returned_dict(db: Path) -> None:
    insert_facet(db, "sc1", "sdxl", {"camera_spec": "x"}, LLM)
    facet = get_facet(db, "sc1", "sdxl", LLM)
    assert facet["scene_id"] == "sc1"
    assert facet["family"] == "sdxl"
    assert facet["llm_id"] == LLM
    assert facet["created_at"] is not None


# ── duplicate insert + delete ───────────────────────────────────────


def test_duplicate_insert_raises_scene_facet_exists(db: Path) -> None:
    insert_facet(db, "sc1", "sdxl", {"camera_spec": "first"}, LLM)
    with pytest.raises(SceneFacetExists, match="already exists"):
        insert_facet(db, "sc1", "sdxl", {"camera_spec": "second"}, LLM)
    # Original survived.
    assert get_facet(db, "sc1", "sdxl", LLM)["camera_spec"] == "first"


def test_delete_facet_removes_row(db: Path) -> None:
    insert_facet(db, "sc1", "sdxl", {"camera_spec": "x"}, LLM)
    n = delete_facet(db, "sc1", "sdxl", LLM)
    assert n == 1
    assert not has_facet(db, "sc1", "sdxl", LLM)


def test_delete_facet_returns_zero_when_missing(db: Path) -> None:
    n = delete_facet(db, "sc1", "sdxl", LLM)
    assert n == 0


def test_delete_facet_leaves_other_families(db: Path) -> None:
    insert_facet(db, "sc1", "sdxl", {"camera_spec": "x"}, LLM)
    insert_facet(db, "sc1", "pony", {"booru_tags": "y"}, LLM)
    delete_facet(db, "sc1", "sdxl", LLM)
    assert not has_facet(db, "sc1", "sdxl", LLM)
    assert has_facet(db, "sc1", "pony", LLM)


# ── per-family bulk delete ──────────────────────────────────────────


def test_delete_facets_for_family_bulk(db: Path) -> None:
    insert_facet(db, "sc1", "sdxl", {"camera_spec": "a"}, LLM)
    insert_facet(db, "sc2", "sdxl", {"camera_spec": "b"}, LLM)
    insert_facet(db, "sc1", "pony", {"booru_tags": "c"}, LLM)
    n = delete_facets_for_family(db, ["sc1", "sc2"], "sdxl", LLM)
    assert n == 2
    assert not has_facet(db, "sc1", "sdxl", LLM)
    assert not has_facet(db, "sc2", "sdxl", LLM)
    assert has_facet(db, "sc1", "pony", LLM)  # different family untouched


def test_delete_facets_for_family_empty_scene_list_is_noop(db: Path) -> None:
    insert_facet(db, "sc1", "sdxl", {"camera_spec": "x"}, LLM)
    n = delete_facets_for_family(db, [], "sdxl", LLM)
    assert n == 0
    assert has_facet(db, "sc1", "sdxl", LLM)


# ── multi-family bookkeeping ────────────────────────────────────────


def test_get_facets_for_scene_returns_all_families(db: Path) -> None:
    insert_facet(db, "sc1", "sdxl", {"camera_spec": "a"}, LLM)
    insert_facet(db, "sc1", "pony", {"booru_tags": "b"}, LLM)
    insert_facet(db, "sc1", "flux", {"scene_prose": "c"}, LLM)
    facets = get_facets_for_scene(db, "sc1", llm_id=LLM)
    assert set(facets.keys()) == {"sdxl", "pony", "flux"}
    assert facets["sdxl"]["camera_spec"] == "a"
    assert facets["pony"]["booru_tags"] == "b"
    assert facets["flux"]["scene_prose"] == "c"


# ── multi-LLM coexistence (the headline feature) ───────────────────


def test_same_scene_family_different_llm_ids_coexist(db: Path) -> None:
    """The triple PK (scene, family, llm_id) lets different LLMs leave
    parallel facet rows on the same scene without overwriting."""
    insert_facet(db, "sc1", "sdxl", {"camera_spec": "cydonia-shot"}, "cydonia")
    insert_facet(db, "sc1", "sdxl", {"camera_spec": "magnum-shot"}, "magnum")
    # Both rows present; lookups disambiguate by llm_id.
    assert has_facet(db, "sc1", "sdxl", "cydonia")
    assert has_facet(db, "sc1", "sdxl", "magnum")
    assert get_facet(db, "sc1", "sdxl", "cydonia")["camera_spec"] == "cydonia-shot"
    assert get_facet(db, "sc1", "sdxl", "magnum")["camera_spec"] == "magnum-shot"


def test_delete_one_llms_facet_leaves_other_llm_untouched(db: Path) -> None:
    insert_facet(db, "sc1", "sdxl", {"camera_spec": "a"}, "cydonia")
    insert_facet(db, "sc1", "sdxl", {"camera_spec": "b"}, "magnum")
    delete_facet(db, "sc1", "sdxl", "cydonia")
    assert not has_facet(db, "sc1", "sdxl", "cydonia")
    assert has_facet(db, "sc1", "sdxl", "magnum")


def test_delete_facets_for_family_filters_by_llm(db: Path) -> None:
    insert_facet(db, "sc1", "sdxl", {"camera_spec": "a"}, "cydonia")
    insert_facet(db, "sc2", "sdxl", {"camera_spec": "b"}, "cydonia")
    insert_facet(db, "sc1", "sdxl", {"camera_spec": "c"}, "magnum")
    n = delete_facets_for_family(db, ["sc1", "sc2"], "sdxl", "cydonia")
    assert n == 2
    assert not has_facet(db, "sc1", "sdxl", "cydonia")
    assert not has_facet(db, "sc2", "sdxl", "cydonia")
    # Magnum's row on sc1 is untouched.
    assert has_facet(db, "sc1", "sdxl", "magnum")


def test_get_facets_for_scene_no_llm_filter_includes_all_llms(db: Path) -> None:
    insert_facet(db, "sc1", "sdxl", {"camera_spec": "a"}, "cydonia")
    insert_facet(db, "sc1", "pony", {"booru_tags": "b"}, "cydonia")
    insert_facet(db, "sc1", "sdxl", {"camera_spec": "c"}, "magnum")
    facets = get_facets_for_scene(db, "sc1")  # llm_id=None
    # Keyed "<family>:<llm_id>" to disambiguate.
    assert set(facets.keys()) == {"sdxl:cydonia", "pony:cydonia", "sdxl:magnum"}
    assert facets["sdxl:cydonia"]["camera_spec"] == "a"
    assert facets["sdxl:magnum"]["camera_spec"] == "c"


def test_get_facets_for_scene_with_llm_filter_excludes_other_llms(db: Path) -> None:
    insert_facet(db, "sc1", "sdxl", {"camera_spec": "a"}, "cydonia")
    insert_facet(db, "sc1", "sdxl", {"camera_spec": "b"}, "magnum")
    cydonia_only = get_facets_for_scene(db, "sc1", llm_id="cydonia")
    assert set(cydonia_only.keys()) == {"sdxl"}
    assert cydonia_only["sdxl"]["camera_spec"] == "a"


# ── schema constraints ──────────────────────────────────────────────


def test_unknown_family_rejected_by_check_constraint(db: Path) -> None:
    with pytest.raises(sqlite3.IntegrityError, match=r"(?i)check"):
        insert_facet(
            db, "sc1", "definitely_not_a_family", {"camera_spec": "x"}, LLM,
        )


def test_orphan_scene_id_rejected_by_fk_constraint(db: Path) -> None:
    with pytest.raises(sqlite3.IntegrityError, match=r"(?i)foreign key"):
        insert_facet(db, "no_such_scene", "sdxl", {"camera_spec": "x"}, LLM)


# ── Phase 4a: structured enum-tag columns ────────────────────────────


def test_phase_4a_enum_columns_round_trip(db: Path) -> None:
    """The 9 persistent enum-tag columns survive insert → get.
    Phase 4a shipped 8; Phase B audit fix added nsfw_act (T4 acts)."""
    insert_facet(
        db, "sc1", "sdxl",
        {
            "camera_spec":         "85mm f/1.4",
            "clothing":            "silk slip",
            # Phase 4a structured enum tags
            "realism_camera":      "CAMERA_SONY_A7RV",
            "realism_lens":        "LENS_85MM_F14",
            "realism_film_stock":  "FILM_PORTRA_400",
            "art_style_reference": "ART_FINE_NUDE",
            "lighting_directive":  "LIGHT_REMBRANDT",
            "mood_aesthetic":      "MOOD_INTIMATE",
            "nsfw_anatomy":        "NSFW_BREAST_NATURAL",
            "nsfw_posture":        "NSFW_RECLINED_NUDE",
            # Phase B (audit fix for Phase 4-bis) — T4 act tag.
            "nsfw_act":            "NSFW_T4_PARTNERED_INTIMATE",
        },
        LLM,
    )
    facet = get_facet(db, "sc1", "sdxl", LLM)
    assert facet["realism_camera"]      == "CAMERA_SONY_A7RV"
    assert facet["realism_lens"]        == "LENS_85MM_F14"
    assert facet["realism_film_stock"]  == "FILM_PORTRA_400"
    assert facet["art_style_reference"] == "ART_FINE_NUDE"
    assert facet["lighting_directive"]  == "LIGHT_REMBRANDT"
    assert facet["mood_aesthetic"]      == "MOOD_INTIMATE"
    assert facet["nsfw_anatomy"]        == "NSFW_BREAST_NATURAL"
    assert facet["nsfw_posture"]        == "NSFW_RECLINED_NUDE"
    assert facet["nsfw_act"]            == "NSFW_T4_PARTNERED_INTIMATE"


def test_phase_4a_enum_columns_optional_default_to_none(db: Path) -> None:
    """A facet that doesn't supply enum tags still inserts cleanly."""
    insert_facet(
        db, "sc1", "sdxl",
        {"camera_spec": "85mm", "clothing": "linen"},
        LLM,
    )
    facet = get_facet(db, "sc1", "sdxl", LLM)
    # New columns default to NULL → None
    assert facet["realism_camera"]      is None
    assert facet["realism_lens"]        is None
    assert facet["realism_film_stock"]  is None
    assert facet["art_style_reference"] is None
    assert facet["lighting_directive"]  is None
    assert facet["mood_aesthetic"]      is None
    assert facet["nsfw_anatomy"]        is None
    assert facet["nsfw_posture"]        is None
    assert facet["nsfw_act"]            is None
