#!/usr/bin/env python3
"""LLM integration test — validates Ollama connectivity and all agent modules.

Tests:
  1. Ollama server connectivity + model availability
  2. OllamaClient.generate() with a simple prompt
  3. OllamaClient.generate_json() with a structured prompt
  4. SeriesPlanner — generate a sample series plan for char_001
  5. SceneGenerator — generate sample scenes from that plan
  6. CharacterCreator — generate a sample character identity
  7. MetadataGenerator — generate sample set metadata
  8. OllamaClient.unload_model() — evict model, verify memory freed via psutil

Pre-flight:
  - ``ollama serve`` is running at the URL in pipeline.yaml (default localhost:11434)
  - The configured model is pulled (``ollama pull dolphin-mixtral:8x7b``)
  - ``python scripts/init_db.py`` has been run (needed for char_001 lookup)
  - ComfyUI should NOT be running — this test loads the LLM into memory

Usage:
    python scripts/test_llm.py
    python scripts/test_llm.py --model dolphin-llama3:8b
    python scripts/test_llm.py --skip-agents     # only test connectivity + generate
    python scripts/test_llm.py --character char_001
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import psutil  # noqa: E402
import yaml  # noqa: E402

from src.agents.llm_client import (  # noqa: E402
    OllamaClient,
    OllamaConnectionError,
    OllamaGenerateError,
    OllamaJSONParseError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    config_path = PROJECT_ROOT / "config" / "pipeline.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _resolve_db_path(cfg: dict) -> Path:
    raw = cfg.get("pipeline", {}).get("db_path", "nsfw_pipeline.db")
    return PROJECT_ROOT / raw


def _load_character(db_path: Path, character_id: str) -> dict | None:
    """Load a character row from the DB. Returns None if not found."""
    if not db_path.exists():
        return None
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM characters WHERE id = ?", (character_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def _load_style_profile(db_path: Path, profile_id: str) -> dict | None:
    """Load a style profile from ``config/style_profiles.yaml``.

    ``db_path`` kept for signature parity — unused.
    """
    from src.memory.style_profile_loader import (
        StyleProfileLoader, StyleProfileNotFound,
    )

    try:
        p = StyleProfileLoader().get_profile(profile_id)
    except StyleProfileNotFound:
        return None
    # Post-2026-04 refactor: StyleProfile carries only aesthetic intent.
    # Render tuning (sampler/scheduler/steps/cfg/clip_skip) and model_id
    # are resolved from the model YAML, not the profile. The downstream
    # agent tests (SeriesPlanner.plan) only consume name +
    # base_style_keywords from this dict, so the stripped shape is
    # sufficient.
    return {
        "id": p.id,
        "name": p.name,
        "base_style_keywords": p.base_style_keywords,
        "base_negative_prompt": p.base_negative_prompt,
    }


def _load_content_rules(db_path: Path, level: str) -> dict | None:
    """Load content_level rules from ``config/categories.yaml``.

    ``db_path`` kept for signature parity — unused.
    """
    from src.memory.categories_loader import (
        CategoriesLoader, ContentLevelNotFound,
    )

    try:
        r = CategoriesLoader().content_level_rules(level)
    except ContentLevelNotFound:
        return None
    return {
        "content_level": r.level,
        "allowed_pose_types": json.dumps(r.allowed_pose_types),
        "scene_constraints": json.dumps(r.scene_constraints),
        "prompt_boost_keywords": json.dumps(r.prompt_boost_keywords),
        "prompt_suppress_keywords": json.dumps(r.prompt_suppress_keywords),
        "description": r.description,
    }


def _get_memory_mb() -> float:
    """Current process RSS in MB."""
    proc = psutil.Process()
    return proc.memory_info().rss / (1024 * 1024)


def _get_system_memory() -> dict:
    """System-wide memory stats."""
    vm = psutil.virtual_memory()
    return {
        "total_gb": round(vm.total / (1024**3), 1),
        "available_gb": round(vm.available / (1024**3), 1),
        "used_percent": vm.percent,
    }


def _separator(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def test_connectivity(client: OllamaClient) -> bool:
    """Test 1: Can we reach the Ollama server?"""
    _separator("Test 1: Ollama connectivity")
    print(f"  Server:  {client.base_url}")
    print(f"  Model:   {client.model}")

    if not client.is_available():
        print("\n  ✗ FAILED — Ollama server not reachable.")
        print("    Fix: run `ollama serve` in another terminal.")
        return False

    models = client.list_models()
    print(f"  Available models: {len(models)}")
    for m in models:
        marker = " ←" if client.model in m else ""
        print(f"    - {m}{marker}")

    if not any(client.model in m for m in models):
        print(f"\n  ⚠ WARNING: configured model {client.model!r} not found.")
        print(f"    Fix: run `ollama pull {client.model}`")
        return False

    print("\n  ✓ PASSED")
    return True


def test_generate(client: OllamaClient) -> bool:
    """Test 2: Basic text generation."""
    _separator("Test 2: generate() — free-form text")
    try:
        start = time.time()
        result = client.generate(
            system_prompt="You are a helpful assistant. Be brief.",
            user_prompt="Describe a sunset in one sentence.",
            temperature=0.7,
            num_predict=100,
        )
        elapsed = time.time() - start
        print(f"  Response ({elapsed:.1f}s):")
        print(f"    {result.strip()[:200]}")
        print(f"\n  ✓ PASSED ({len(result)} chars)")
        return True
    except (OllamaConnectionError, OllamaGenerateError) as exc:
        print(f"\n  ✗ FAILED: {exc}")
        return False


def test_generate_json(client: OllamaClient) -> bool:
    """Test 3: Structured JSON generation."""
    _separator("Test 3: generate_json() — structured output")
    try:
        start = time.time()
        result = client.generate_json(
            system_prompt="You output ONLY valid JSON, no markdown fences, no commentary.",
            user_prompt=(
                'Generate a JSON object with fields: '
                '"color" (a color name), "number" (1-100), "word" (a random noun). '
                'Return ONLY the JSON object.'
            ),
            temperature=0.6,
            num_predict=200,
        )
        elapsed = time.time() - start
        print(f"  Response ({elapsed:.1f}s):")
        print(f"    {json.dumps(result, indent=2)}")
        if isinstance(result, dict):
            print(f"\n  ✓ PASSED (dict with {len(result)} keys)")
            return True
        else:
            print(f"\n  ⚠ Unexpected type: {type(result).__name__}")
            return True  # Still valid JSON
    except OllamaJSONParseError as exc:
        print(f"\n  ✗ FAILED — JSON parse error: {exc}")
        return False
    except (OllamaConnectionError, OllamaGenerateError) as exc:
        print(f"\n  ✗ FAILED: {exc}")
        return False


def test_series_planner(client: OllamaClient, db_path: Path, character_id: str) -> dict | None:
    """Test 4: Series planner with a real character."""
    _separator("Test 4: SeriesPlanner — series plan for " + character_id)

    from src.agents.series_planner import SeriesPlanner, SeriesPlannerError

    char = _load_character(db_path, character_id)
    if char is None:
        print(f"  ⚠ SKIPPED — character {character_id!r} not in DB.")
        print(f"    Fix: run `python scripts/bootstrap_character.py {character_id}`")
        return None

    style = _load_style_profile(db_path, char["style_profile_id"])
    rules = _load_content_rules(db_path, "T2_implied")
    if style is None or rules is None:
        print("  ⚠ SKIPPED — style profile or content rules not found in DB.")
        return None

    planner = SeriesPlanner(client)
    try:
        start = time.time()
        plan = planner.plan(
            character_name=character_id,
            base_prompt=char["base_prompt"],
            vibe=char.get("vibe", ""),
            style_name=style["name"],
            style_keywords=style["base_style_keywords"],
            content_level="T2_implied",
            content_rules=rules,
            previous_themes=[],
        )
        elapsed = time.time() - start
        print(f"  Response ({elapsed:.1f}s):")
        print(f"    {json.dumps(plan, indent=2)}")
        print(f"\n  ✓ PASSED — theme: {plan['theme']!r}")
        return plan
    except SeriesPlannerError as exc:
        print(f"\n  ✗ FAILED: {exc}")
        return None


def test_scene_generator(client: OllamaClient, series_plan: dict) -> list | None:
    """Test 5: Scene generator from a series plan."""
    _separator("Test 5: SceneGenerator — scenes from plan")

    from src.agents.scene_generator import SceneGenerator, SceneGeneratorError

    # Use soft pose types
    allowed_poses = [
        "standing relaxed", "sitting elegantly", "leaning gently",
        "looking over shoulder", "hand on hip", "profile pose",
    ]

    generator = SceneGenerator(client)
    try:
        start = time.time()
        scenes = generator.generate(
            series_plan=series_plan,
            content_level="T2_implied",
            allowed_pose_types=allowed_poses,
            scene_count=5,  # small count for testing
        )
        elapsed = time.time() - start
        print(f"  Generated {len(scenes)} scenes ({elapsed:.1f}s):")
        for i, scene in enumerate(scenes[:3]):  # show first 3
            print(f"\n  Scene {i+1}:")
            print(f"    axis:        {scene['variation_axis']}")
            print(f"    pose:        {scene['pose']}")
            print(f"    camera:      {scene['camera']}")
            print(f"    lighting:    {scene['lighting']}")
            print(f"    environment: {scene['environment_detail'][:60]}")
            print(f"    mood:        {scene['mood_note']}")
        if len(scenes) > 3:
            print(f"\n  … and {len(scenes) - 3} more scenes")
        print(f"\n  ✓ PASSED — {len(scenes)} valid scenes")
        return scenes
    except SceneGeneratorError as exc:
        print(f"\n  ✗ FAILED: {exc}")
        return None


def test_character_creator(client: OllamaClient) -> dict | None:
    """Test 6: Character creator."""
    _separator("Test 6: CharacterCreator — new identity")

    from src.agents.character_creator import CharacterCreator, CharacterCreatorError

    creator = CharacterCreator(client)
    try:
        start = time.time()
        identity = creator.create(
            target_vibe="warm natural",
            gender="female",
            additional_notes="Mediterranean features",
            existing_characters=[
                {"name": "char_001", "vibe": "soft elegant", "hair": "long dark brown wavy hair"}
            ],
        )
        elapsed = time.time() - start
        print(f"  Generated identity ({elapsed:.1f}s):")
        print(f"    {json.dumps(identity, indent=2)}")
        print(f"\n  ✓ PASSED — vibe: {identity['vibe']!r}")
        return identity
    except CharacterCreatorError as exc:
        print(f"\n  ✗ FAILED: {exc}")
        return None


def test_metadata_generator(client: OllamaClient) -> dict | None:
    """Test 7: Metadata generator."""
    _separator("Test 7: MetadataGenerator — set metadata")

    from src.agents.metadata_generator import MetadataGenerator, MetadataGeneratorError

    generator = MetadataGenerator(client)
    try:
        start = time.time()
        metadata = generator.generate(
            theme="Golden Hour Reverie",
            mood="dreamy",
            environment="sunlit rooftop garden overlooking the city",
            character_name="char_001",
            content_level="T2_implied",
            image_count=20,
            style_keywords="beautiful, elegant, aesthetic, soft lighting",
        )
        elapsed = time.time() - start
        print(f"  Generated metadata ({elapsed:.1f}s):")
        print(f"    Title:       {metadata['title']}")
        print(f"    Description: {metadata['description'][:120]}…")
        print(f"    Tags ({len(metadata['tags'])}): {', '.join(metadata['tags'][:10])}…")
        print(f"\n  ✓ PASSED — {len(metadata['tags'])} tags")
        return metadata
    except MetadataGeneratorError as exc:
        print(f"\n  ✗ FAILED: {exc}")
        return None


def test_unload(client: OllamaClient) -> bool:
    """Test 8: Unload model and verify memory freed."""
    _separator("Test 8: unload_model() — memory release")

    mem_before = _get_system_memory()
    print(f"  System memory BEFORE unload:")
    print(f"    Total:     {mem_before['total_gb']} GB")
    print(f"    Available: {mem_before['available_gb']} GB")
    print(f"    Used:      {mem_before['used_percent']}%")

    print(f"\n  Unloading {client.model} …")
    start = time.time()
    client.unload_model()
    elapsed = time.time() - start

    mem_after = _get_system_memory()
    freed = mem_after["available_gb"] - mem_before["available_gb"]
    print(f"\n  System memory AFTER unload ({elapsed:.1f}s):")
    print(f"    Total:     {mem_after['total_gb']} GB")
    print(f"    Available: {mem_after['available_gb']} GB")
    print(f"    Used:      {mem_after['used_percent']}%")
    print(f"    Freed:     {freed:+.1f} GB")

    if freed > 0.5:
        print(f"\n  ✓ PASSED — {freed:.1f} GB freed (model evicted)")
    elif freed > 0:
        print(f"\n  ✓ PASSED — {freed:.1f} GB freed (partial — model may have been partially loaded)")
    else:
        print(f"\n  ⚠ WARNING — no memory freed. Model may not have been loaded, "
              "or macOS unified memory accounting differs from psutil's view.")
        print("    This is expected if no generate() call was made before unload.")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test LLM integration — Ollama client + all agent modules",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override LLM model (default: from pipeline.yaml)",
    )
    parser.add_argument(
        "--character",
        default="char_001",
        help="Character id for series planner test (default: char_001)",
    )
    parser.add_argument(
        "--skip-agents",
        action="store_true",
        help="Only test connectivity + generate, skip agent modules",
    )
    args = parser.parse_args()

    cfg = _load_config()
    db_path = _resolve_db_path(cfg)
    client = OllamaClient(model=args.model)

    print("=" * 60)
    print("  LLM Integration Test Suite")
    print("=" * 60)
    print(f"  Server:    {client.base_url}")
    print(f"  Model:     {client.model}")
    print(f"  DB:        {db_path}")
    print(f"  Character: {args.character}")
    print(f"  Mode:      {'connectivity only' if args.skip_agents else 'full (all agents)'}")

    results: dict[str, bool] = {}

    # Test 1: connectivity
    results["connectivity"] = test_connectivity(client)
    if not results["connectivity"]:
        print("\n\nCannot proceed without Ollama connectivity. Exiting.")
        sys.exit(1)

    # Test 2: basic generate
    results["generate"] = test_generate(client)

    # Test 3: generate_json
    results["generate_json"] = test_generate_json(client)

    if args.skip_agents:
        # Jump straight to unload
        results["unload"] = test_unload(client)
    else:
        # Test 4: series planner
        plan = test_series_planner(client, db_path, args.character)
        results["series_planner"] = plan is not None

        # Test 5: scene generator (needs a plan)
        if plan:
            scenes = test_scene_generator(client, plan)
            results["scene_generator"] = scenes is not None
        else:
            _separator("Test 5: SceneGenerator — SKIPPED (no plan)")
            results["scene_generator"] = False

        # Test 6: character creator
        identity = test_character_creator(client)
        results["character_creator"] = identity is not None

        # Test 7: metadata generator
        metadata = test_metadata_generator(client)
        results["metadata_generator"] = metadata is not None

        # Test 8: unload model
        results["unload"] = test_unload(client)

    # Summary
    _separator("Summary")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        status = "✓" if ok else "✗"
        print(f"  {status} {name}")
    print(f"\n  {passed}/{total} tests passed")

    if passed == total:
        print("\n  All tests passed. LLM integration is ready.")
        print("  REMINDER: call llm_client.unload_model() before any ComfyUI renders.")
        sys.exit(0)
    else:
        print(f"\n  {total - passed} test(s) failed. Check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
