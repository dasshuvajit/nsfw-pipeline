"""Tests for Q10 — composition vocab gap-fill (angle + framing namespaces).

Bumps prompt_vocabulary.yaml to v4 with two new namespaces:
- realism.angle (ANGLE_LOW / EYE_LEVEL / HIGH / DUTCH / OVER_SHOULDER)
- realism.framing (FRAMING_EXTREME_CLOSE_UP through FRAMING_EXTREME_WIDE)

Pony schemas omit both fields (booru tags carry them implicitly), but
the DB column accepts NULLs uniformly so cross-family queries don't
need branching.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from src.memory.scene_facets_repo import _FACET_FIELDS, insert_facet, get_facet
from src.prompt.vocabulary import (
    VocabularyLoader,
    canonicalize_facet,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
LLM = "test_llm"


@pytest.fixture
def db(tmp_path: Path) -> Path:
    db_path = tmp_path / "vocab_v4.db"
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
        "'test', 1, 'planned')"
    )
    conn.execute(
        "INSERT INTO scenes (id, series_id, variation_axis, content_level) "
        "VALUES ('sc1', 's1', 'test', 'T2_implied')"
    )
    conn.commit()
    conn.close()
    return db_path


# ── Vocab structure ────────────────────────────────────────────────


class TestVocabularyV4:
    def test_version_bumped_to_4(self):
        loader = VocabularyLoader()
        assert loader.version == 4

    def test_angle_namespace_exists(self):
        loader = VocabularyLoader()
        concepts = loader.concepts_by_namespace("realism", "angle")
        assert "ANGLE_LOW" in concepts
        assert "ANGLE_EYE_LEVEL" in concepts
        assert "ANGLE_HIGH" in concepts
        assert "ANGLE_DUTCH" in concepts
        assert "ANGLE_OVER_SHOULDER" in concepts

    def test_framing_namespace_exists(self):
        loader = VocabularyLoader()
        concepts = loader.concepts_by_namespace("realism", "framing")
        assert "FRAMING_EXTREME_CLOSE_UP" in concepts
        assert "FRAMING_CLOSE_UP" in concepts
        assert "FRAMING_MEDIUM_CLOSE" in concepts
        assert "FRAMING_MEDIUM" in concepts
        assert "FRAMING_MEDIUM_WIDE" in concepts
        assert "FRAMING_WIDE" in concepts
        assert "FRAMING_EXTREME_WIDE" in concepts

    def test_pony_omits_angle_namespace(self):
        """Pony entries have no `pony:` key in angle namespace."""
        loader = VocabularyLoader()
        # canonicalize for pony returns None — no phrasing for that family
        result = loader.canonicalize("ANGLE_LOW", "pony")
        assert result is None

    def test_pony_omits_framing_namespace(self):
        loader = VocabularyLoader()
        result = loader.canonicalize("FRAMING_CLOSE_UP", "pony")
        assert result is None


# ── Per-family canonicalization ────────────────────────────────────


@pytest.mark.parametrize(
    "family",
    ["sdxl", "illustrious", "flux", "chroma", "flux2"],
)
class TestPerFamilyCanonicalization:
    def test_angle_low_canonicalizes(self, family):
        loader = VocabularyLoader()
        out = loader.canonicalize("ANGLE_LOW", family)
        assert out is not None
        assert "low" in out.lower()

    def test_framing_wide_canonicalizes(self, family):
        loader = VocabularyLoader()
        out = loader.canonicalize("FRAMING_WIDE", family)
        assert out is not None
        # Each family phrasing mentions wide-shot / full-body framing.
        assert "wide" in out.lower() or "full body" in out.lower()


# ── Schema field round-trip ────────────────────────────────────────


class TestSchemaFieldRoundTrip:
    def test_facet_fields_includes_angle_and_framing(self):
        assert "realism_angle" in _FACET_FIELDS
        assert "realism_framing" in _FACET_FIELDS

    def test_insert_with_angle_and_framing(self, db: Path):
        insert_facet(
            db, "sc1", "sdxl",
            {
                "camera_spec": "85mm",
                "clothing": "silk slip",
                "realism_angle": "ANGLE_LOW",
                "realism_framing": "FRAMING_MEDIUM",
            },
            LLM,
        )
        facet = get_facet(db, "sc1", "sdxl", LLM)
        assert facet["realism_angle"] == "ANGLE_LOW"
        assert facet["realism_framing"] == "FRAMING_MEDIUM"

    def test_insert_without_angle_and_framing_works(self, db: Path):
        insert_facet(
            db, "sc1", "flux",
            {"scene_prose": "A simple scene description here."},
            LLM,
        )
        facet = get_facet(db, "sc1", "flux", LLM)
        assert facet["realism_angle"] is None
        assert facet["realism_framing"] is None


# ── Canonicalize_facet integration ─────────────────────────────────


class TestCanonicalizeFacetIntegration:
    def test_canonicalize_facet_maps_angle_field(self):
        """canonicalize_facet should translate realism_angle into a
        family-shaped phrase for the target family."""
        facet = {
            "camera_spec": "85mm",
            "clothing": "silk slip",
            "realism_angle": "ANGLE_LOW",
        }
        out = canonicalize_facet(facet, "sdxl")
        # canonicalize_facet returns a list of phrases (one per
        # canonicalized field). Verify the angle phrase appears.
        all_text = " ".join(str(p) for p in out)
        assert "low" in all_text.lower()

    def test_canonicalize_facet_drops_pony_angle(self):
        """Pony has no phrasing for ANGLE_LOW — canonicalizer returns
        None which the caller drops from the segment list."""
        loader = VocabularyLoader()
        assert loader.canonicalize("ANGLE_LOW", "pony") is None
