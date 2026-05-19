#!/usr/bin/env python3
"""
Initialize the NSFW pipeline SQLite database.

Post-YAML-consolidation schema — 9 runtime-mutable tables only. All
static lookup data (model registry, model prompt guides, style
profiles, theme/style/niche categories, content-level rules) now
lives in ``config/*.yaml`` and is loaded via the dedicated loaders in
``src/memory/``. Cross-references to YAML ids are stored as plain
text columns; validation happens at startup inside the loaders, not
via SQL FKs.

Usage:
    python scripts/init_db.py                       # creates ./nsfw_pipeline.db
    python scripts/init_db.py --db-path custom.db   # custom location
    python scripts/init_db.py --force                # overwrite existing DB

Safe by default: refuses to overwrite an existing DB file unless --force is given.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

import yaml

# -----------------------------------------------------------------------------
# Schema — 10 runtime-mutable tables (characters, series, scenes, prompts,
# scene_facets, images, sets, posts, generation_memory, run_log)
# -----------------------------------------------------------------------------

# Single source of truth for the family list. The `scene_facets.family`
# CHECK constraint below is templated from this file at init time so a
# new family is a one-yaml-edit + DB re-init away (pre-stable, no
# migrations).
FAMILIES_YAML = (
    Path(__file__).resolve().parent.parent / "config" / "families.yaml"
)


def _family_check_clause() -> str:
    """Build the SQL `IN (...)` literal for `scene_facets.family`."""
    with open(FAMILIES_YAML) as f:
        data = yaml.safe_load(f) or {}
    families = sorted((data.get("families") or {}).keys())
    if not families:
        raise RuntimeError(
            f"No families found in {FAMILIES_YAML}. "
            "scene_facets table cannot be created without a family list."
        )
    return ", ".join(f"'{fid}'" for fid in families)


SCHEMA_SQL = """
-- ============================================================
-- CHARACTERS
-- ============================================================
-- style_profile_id is a YAML id (see config/style_profiles.yaml)
-- validated at startup by StyleProfileLoader — no SQL FK.
-- model_id is a YAML id (see config/models/*.yaml) validated by
-- ModelRegistryLoader; NULL falls back to pipeline.default_model_id.
CREATE TABLE characters (
    id TEXT PRIMARY KEY,
    name TEXT,
    style_profile_id TEXT NOT NULL,
    model_id TEXT,
    base_prompt TEXT NOT NULL,               -- IMMUTABLE after creation
    negative_prompt TEXT,
    reference_image_path TEXT,
    locked_features TEXT NOT NULL,           -- JSON
    allowed_shift_axes TEXT NOT NULL,        -- JSON
    outfit_pool TEXT,                        -- JSON
    version INTEGER NOT NULL DEFAULT 1,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    total_images_generated INTEGER DEFAULT 0,
    last_used_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Protect base_prompt from accidental modification
CREATE TRIGGER protect_character_base_prompt
BEFORE UPDATE ON characters
WHEN OLD.base_prompt != NEW.base_prompt
BEGIN
    SELECT RAISE(ABORT, 'base_prompt is immutable after creation. Create a new version instead.');
END;

-- ============================================================
-- SERIES
-- ============================================================
CREATE TABLE series (
    id TEXT PRIMARY KEY,
    mode TEXT NOT NULL CHECK (mode IN (
        'character','theme','style','niche','variation'
    )),
    content_level TEXT NOT NULL DEFAULT 'T2_implied' CHECK (content_level IN (
        'T1_suggestive','T2_implied','T3_artnude','T4_explicit'
    )),
    character_id TEXT REFERENCES characters(id),
    style_profile_id TEXT NOT NULL,          -- YAML id, validated at startup
    theme TEXT NOT NULL,
    mood TEXT,
    environment TEXT,
    variation_axes TEXT,                      -- JSON
    target_count INTEGER NOT NULL DEFAULT 20,
    actual_count INTEGER DEFAULT 0,
    base_seed INTEGER DEFAULT NULL,           -- NULL=random per image, set=reproducible
    status TEXT NOT NULL DEFAULT 'planned' CHECK (status IN (
        'planned','dry_run','rendering','filtering','packaging','complete','failed','partial','aborted'
    )),
    llm_series_plan TEXT,                     -- JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- ============================================================
-- SCENES
-- ============================================================
CREATE TABLE scenes (
    id TEXT PRIMARY KEY,
    series_id TEXT NOT NULL REFERENCES series(id),
    variation_axis TEXT NOT NULL,
    pose TEXT,
    camera TEXT,
    camera_angle TEXT,
    lighting TEXT,
    environment_detail TEXT,
    mood_note TEXT,
    expression TEXT,
    aspect_ratio TEXT,                        -- Selected per-scene by RatioSelector
    resolution_w INTEGER,
    resolution_h INTEGER,
    content_level TEXT NOT NULL DEFAULT 'T2_implied' CHECK (content_level IN (
        'T1_suggestive','T2_implied','T3_artnude','T4_explicit'
    )),
    status TEXT NOT NULL DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- PROMPTS
-- ============================================================
-- llm_id is a YAML id (config/llm_models.yaml) validated at startup
-- by LLMRegistryLoader.
--
-- target_kind discriminates between two preparation modes:
--   * 'model'  — model-level prompt; per-model trigger words /
--                avoid words / negative_embeddings / lora_stack /
--                structure_intro all applied. model_id holds an id
--                from config/models/*.yaml validated by
--                ModelRegistryLoader.
--   * 'family' — family-level prompt; ONLY family-level rules from
--                config/families.yaml applied. model_id holds a
--                FAMILY id (sdxl/pony/illustrious/flux/chroma/flux2)
--                — `model_id` does double duty here for shape
--                preservation. Validated by FamilyLoader.
--
-- UNIQUE(scene_id, target_kind, model_id, llm_id) enforces "one
-- prompt per (scene, target_kind, target_id, generating-LLM)" so the
-- same scene can carry both family-level and model-level prompts
-- without colliding, plus per-LLM A/B comparison on top.
-- Re-rolling for the same (scene, target_kind, target_id, llm)
-- requires explicit DELETE first (`prepare_prompts --regen-prompts X
-- --llm Y` for model-kind, `--regen-family-prompts X --llm Y` for
-- family-kind).
CREATE TABLE prompts (
    id TEXT PRIMARY KEY,
    series_id TEXT NOT NULL REFERENCES series(id),
    scene_id TEXT REFERENCES scenes(id),
    target_kind TEXT NOT NULL DEFAULT 'model' CHECK (target_kind IN (
        'model','family'
    )),
    model_id TEXT NOT NULL,
    llm_id TEXT NOT NULL,
    prompt_text TEXT NOT NULL,
    negative_prompt TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    content_level TEXT NOT NULL DEFAULT 'T2_implied' CHECK (content_level IN (
        'T1_suggestive','T2_implied','T3_artnude','T4_explicit'
    )),
    -- Phase 4a — version stamp of config/prompt_vocabulary.yaml at
    -- the moment this prompt was composed. Lets us answer "which
    -- vocab produced this prompt?" after the YAML is bumped.
    vocab_version INTEGER NOT NULL DEFAULT 1,
    is_variation BOOLEAN DEFAULT FALSE,
    variation_of TEXT REFERENCES prompts(id),
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN (
        'pending','rendered','failed','rejected'
    )),
    render_attempts INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (scene_id, target_kind, model_id, llm_id)
);

-- ============================================================
-- SCENE_FACETS — per-family LLM expansion of one scene
-- ============================================================
-- A scene's model-agnostic core (pose/camera/lighting/...) lives in
-- the scenes table. Family-shaped LLM output (booru_tags for pony,
-- scene_prose for flux/chroma/illustrious/flux2, camera_spec +
-- clothing for sdxl) lives here, keyed by family. Sibling models in
-- the same family (e.g. juggernaut_ragnarok + gonzalomo_photo_v70, both sdxl)
-- share a single facet row — the per-model differentiation lives in
-- the prompts row via composer + trigger words.
--
-- The CHECK constraint's family list is generated at init time from
-- config/families.yaml so adding a new family is a one-file change
-- (edit families.yaml + re-init DB; pre-stable, no migration).
CREATE TABLE scene_facets (
    scene_id TEXT NOT NULL REFERENCES scenes(id),
    family TEXT NOT NULL CHECK (family IN ({family_check})),
    -- llm_id is a YAML id (config/llm_models.yaml) validated at startup
    -- by LLMRegistryLoader. The (scene, family, llm_id) PRIMARY KEY lets
    -- the same scene be re-expanded by a different LLM (Cydonia vs
    -- Magnum vs Venice) for manual A/B comparison without overwriting
    -- the prior LLM's output. Each prompts row's llm_id matches the
    -- scene_facets row that fed its composition.
    llm_id TEXT NOT NULL,
    booru_tags TEXT,
    source_tag TEXT,
    scene_prose TEXT,
    camera_spec TEXT,
    clothing TEXT,
    -- Phase 4a — abstract concept tags from config/prompt_vocabulary.yaml.
    -- The composer canonicalises these into family-shaped phrases at
    -- compose time. All optional; empty when the LLM didn't pick one.
    realism_camera TEXT,
    realism_lens TEXT,
    realism_film_stock TEXT,
    art_style_reference TEXT,
    lighting_directive TEXT,
    mood_aesthetic TEXT,
    nsfw_anatomy TEXT,
    nsfw_posture TEXT,
    -- Phase 4-bis (Phase B audit fix) — T4 explicit-act concept tag.
    -- Gated to T4_explicit at canonicalize time; canonicalizer drops
    -- at T1/T2/T3 even if the LLM emits a tag.
    nsfw_act TEXT,
    -- Q10 (vocab v4) — composition gap-fill. Camera angle and shot
    -- size / framing concepts. Pony's schema omits these (booru tags
    -- carry them implicitly via `low_angle`, `cowboy_shot`, etc.) but
    -- the DB column accepts NULLs uniformly across families.
    realism_angle TEXT,
    realism_framing TEXT,
    -- Phase 1-4 (vocab v6, 2026-05-19) — environment + narrative +
    -- composition gap-fill. The LLM picks one ENV_* / ATM_* / PROP_*
    -- / NARR_* / COMP_* tag per scene; canonicalizer translates each
    -- into family-shaped prose at compose time. Pre-fix these 5
    -- columns were missing — the Pydantic schemas declared them and
    -- the engine's in-memory `scene_with_facet` consumed them, but
    -- ``scene_facets_repo.insert_facet`` silently dropped them at
    -- persist (round-5 BLOCKER). All families participate; Pony's
    -- composer omits ``composition_principle`` via canonicalizer.
    environment_setting TEXT,
    environment_atmosphere TEXT,
    environment_prop TEXT,
    narrative_moment TEXT,
    composition_principle TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (scene_id, family, llm_id)
);

-- ============================================================
-- IMAGES
-- ============================================================
CREATE TABLE images (
    id TEXT PRIMARY KEY,
    prompt_id TEXT NOT NULL REFERENCES prompts(id),
    series_id TEXT NOT NULL REFERENCES series(id),
    -- model_id is a YAML id (see config/models/*.yaml), validated at
    -- startup by ModelRegistryLoader — no SQL FK.
    model_id TEXT,
    file_path TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    seed INTEGER NOT NULL,
    content_level TEXT NOT NULL DEFAULT 'T2_implied' CHECK (content_level IN (
        'T1_suggestive','T2_implied','T3_artnude','T4_explicit'
    )),
    -- Quality scoring (concrete metrics, not placeholder)
    quality_score REAL,                       -- composite 0.0–1.0
    aesthetic_score REAL,                     -- LAION CLIP aesthetic 0–10
    blur_score REAL,                          -- Laplacian variance
    face_confidence REAL,                     -- insightface det_score
    -- Phase G — human-preference scorers (nullable; populated only when
    -- pipeline.yaml::scoring.use_hps_v2 / use_image_reward are enabled).
    hps_v2_score REAL,                        -- HPS v2 preference probability 0–1
    image_reward_score REAL,                  -- ImageReward raw scalar (~ -2.4 .. +1.0)
    quality_flags TEXT,                       -- JSON: ["low_aesthetic","blurry","no_face","low_hps_v2","low_image_reward"]
    selected BOOLEAN DEFAULT FALSE,
    upscaled_path TEXT,
    rejection_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- SETS
-- ============================================================
CREATE TABLE sets (
    id TEXT PRIMARY KEY,
    series_id TEXT NOT NULL REFERENCES series(id),
    title TEXT NOT NULL,
    description TEXT,
    tags TEXT NOT NULL,                       -- JSON
    content_level TEXT NOT NULL DEFAULT 'T2_implied' CHECK (content_level IN (
        'T1_suggestive','T2_implied','T3_artnude','T4_explicit'
    )),
    image_count INTEGER NOT NULL,
    export_path TEXT NOT NULL,
    manifest_path TEXT,
    status TEXT NOT NULL DEFAULT 'draft',
    platform TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- POSTS (engagement tracking — manual first, API later)
-- ============================================================
CREATE TABLE posts (
    id TEXT PRIMARY KEY,
    set_id TEXT REFERENCES sets(id),
    platform TEXT NOT NULL,
    post_url TEXT,
    posted_at TIMESTAMP,
    content_level TEXT,
    character_id TEXT,
    views_24h INTEGER,
    views_72h INTEGER,
    favorites INTEGER,
    comments INTEGER,
    new_followers INTEGER,
    subscription_conversions INTEGER,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- MEMORY
-- ============================================================
CREATE TABLE generation_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,                       -- 'theme'|'scene'|'prompt'|'style'|'niche'
    content_hash TEXT NOT NULL,
    content_preview TEXT,
    style_profile_id TEXT,
    content_level TEXT,
    character_id TEXT,                        -- For cross-level scene dedup
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- RUN LOG
-- ============================================================
CREATE TABLE run_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mode TEXT NOT NULL,
    content_level TEXT NOT NULL DEFAULT 'T2_implied' CHECK (content_level IN (
        'T1_suggestive','T2_implied','T3_artnude','T4_explicit'
    )),
    series_id TEXT REFERENCES series(id),
    execution_mode TEXT NOT NULL DEFAULT 'manual' CHECK (execution_mode IN (
        'manual','supervised','automated'
    )),
    status TEXT NOT NULL CHECK (status IN (
        'success','failed','partial','aborted','interrupted'
    )),
    images_generated INTEGER DEFAULT 0,
    images_selected INTEGER DEFAULT 0,
    duration_seconds REAL,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- INDEXES
-- ============================================================
CREATE INDEX idx_series_mode ON series(mode);
CREATE INDEX idx_series_status ON series(status);
CREATE INDEX idx_series_character ON series(character_id);
CREATE INDEX idx_series_content_level ON series(content_level);
CREATE INDEX idx_images_series ON images(series_id);
CREATE INDEX idx_images_selected ON images(selected);
CREATE INDEX idx_images_content_level ON images(content_level);
CREATE INDEX idx_images_quality ON images(quality_score);
CREATE INDEX idx_images_model ON images(model_id);
CREATE INDEX idx_prompts_series ON prompts(series_id);
CREATE INDEX idx_prompts_hash ON prompts(prompt_hash);
CREATE INDEX idx_prompts_status ON prompts(status);
CREATE INDEX idx_prompts_model ON prompts(model_id);
CREATE INDEX idx_prompts_llm ON prompts(llm_id);
CREATE INDEX idx_prompts_target_kind ON prompts(target_kind);
CREATE INDEX idx_scene_facets_scene ON scene_facets(scene_id);
CREATE INDEX idx_scene_facets_llm ON scene_facets(llm_id);
CREATE INDEX idx_memory_hash ON generation_memory(content_hash);
CREATE INDEX idx_memory_type ON generation_memory(type);
CREATE INDEX idx_memory_character ON generation_memory(character_id);
CREATE INDEX idx_characters_active ON characters(active);
CREATE INDEX idx_posts_set ON posts(set_id);
CREATE INDEX idx_posts_character ON posts(character_id);
"""


# -----------------------------------------------------------------------------
# DB initialization
# -----------------------------------------------------------------------------

def init_database(db_path: Path) -> None:
    """Create schema into a fresh DB at ``db_path``.

    No seed data — all static rows moved to ``config/*.yaml``. The
    ``scene_facets.family`` CHECK clause is templated from the current
    ``config/families.yaml`` family list so the schema stays in sync
    without a separate migration step.
    """
    schema_sql = SCHEMA_SQL.replace("{family_check}", _family_check_clause())
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(schema_sql)
        conn.commit()
    finally:
        conn.close()


def summarize(db_path: Path) -> None:
    """Print a quick object summary for the freshly created DB."""
    conn = sqlite3.connect(str(db_path))
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )]
        triggers = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' ORDER BY name"
        )]
        indexes = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )]

        print(f"\nDatabase initialized at: {db_path.resolve()}")
        print(f"  Tables   ({len(tables)}): {', '.join(tables)}")
        print(f"  Indexes  ({len(indexes)})")
        print(f"  Triggers ({len(triggers)}): {', '.join(triggers)}")
        print("\nStatic lookup data lives in config/*.yaml "
              "(families, style_profiles, categories, per-model files).")
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path",
        default="nsfw_pipeline.db",
        help="Path to the SQLite database file (default: nsfw_pipeline.db)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the DB file if it already exists.",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)

    if db_path.exists():
        if not args.force:
            print(
                f"ERROR: {db_path} already exists. "
                f"Re-run with --force to overwrite.",
                file=sys.stderr,
            )
            return 1
        print(f"Removing existing database: {db_path}")
        os.remove(db_path)

    try:
        init_database(db_path)
    except sqlite3.Error as exc:
        print(f"ERROR: SQLite error during init: {exc}", file=sys.stderr)
        return 2

    summarize(db_path)
    print("\nOK — database initialization complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
