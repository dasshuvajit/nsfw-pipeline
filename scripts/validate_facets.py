#!/usr/bin/env python3
"""validate_facets — smoke test the SceneFacetGenerator on a handful
of synthetic scenes.

Bypasses the full ``prepare_prompts`` pipeline (planner +
scene_generator + prompt composer + dedup + sanitizer + persistence)
to exercise just the LLM call that produces a family-shaped facet
from a scene-core dict. Useful for:

* Verifying that a schema change actually lands the newly-required
  fields on every facet.
* Spot-checking that the LLM-vocabulary block in the system prompt
  produces sensible tag picks at a given content_level.
* Measuring per-scene wall-clock for facet generation when tuning an
  LLM or schema change.

The script:

1. Loads the requested ``family`` from ``config/families.yaml``.
2. Iterates over up to ``--scenes`` hand-crafted scene-core dicts
   from ``_SAMPLE_SCENES`` (varied pose / lighting / environment /
   mood so the LLM has something to differentiate per scene).
3. Calls ``SceneFacetGenerator.generate(...)`` for each scene at the
   requested ``--level``.
4. Tabulates the resulting facets — one column per scene, one row
   per structured-tag axis — and flags which tier-required fields
   were left null on which scenes.
5. Always unloads the LLM on exit (CLAUDE.md hard invariant — LLM
   and ComfyUI cannot share unified memory).

Usage::

    python scripts/validate_facets.py
    python scripts/validate_facets.py --family flux --level T2_implied
    python scripts/validate_facets.py --scenes 5 --llm cydonia_heretic_24b

The required-field check reads ``_TIER_REQUIRED_FIELDS`` directly from
``src.agents.scene_facet_generator``, so a schema-contract change is
automatically picked up — no need to update this script when the
tier-required set evolves.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.llm_client import LLMClientPool, OllamaClient  # noqa: E402
from src.agents.lm_studio_client import LMStudioClient  # noqa: E402
from src.agents.mlx_client import MlxClient  # noqa: E402
from src.agents.scene_facet_generator import (  # noqa: E402
    _TIER_REQUIRED_FIELDS,
    SceneFacetGenerator,
)
from src.memory.family_loader import FamilyLoader  # noqa: E402
from src.memory.llm_registry import LLMRegistryLoader  # noqa: E402


# Hand-crafted scene-core fields. Picked to span distinct poses /
# environments / lighting setups so the LLM has differentiating signal
# to pick varied tags. Extend this list if you want broader coverage.
_SAMPLE_SCENES: list[dict] = [
    {
        "id": "smoke_scene_001",
        "pose": "standing confident",
        "camera": "medium shot",
        "camera_angle": "eye-level",
        "lighting": "soft golden hour rim light",
        "environment_detail": "Tuscan villa courtyard",
        "mood_note": "Contemplative",
        "expression": "Serene gaze toward horizon",
        "composition_intent": "vertical framing",
    },
    {
        "id": "smoke_scene_002",
        "pose": "reclining expressive",
        "camera": "close-up on torso",
        "camera_angle": "slightly elevated",
        "lighting": "warm side light through window",
        "environment_detail": "Victorian parlour with chaise lounge",
        "mood_note": "Pensive intimacy",
        "expression": "Eyes half-closed in repose",
        "composition_intent": "diagonal composition",
    },
    {
        "id": "smoke_scene_003",
        "pose": "kneeling graceful",
        "camera": "three-quarter body shot",
        "camera_angle": "low angle",
        "lighting": "Rembrandt-style chiaroscuro",
        "environment_detail": "minimalist artist studio",
        "mood_note": "Defiant strength",
        "expression": "Direct confident gaze at camera",
        "composition_intent": "centered hero shot",
    },
    {
        "id": "smoke_scene_004",
        "pose": "side profile arched back",
        "camera": "full body shot",
        "camera_angle": "Dutch tilt",
        "lighting": "split lighting hard shadow",
        "environment_detail": "Berlin loft with exposed brick",
        "mood_note": "Sensual mystery",
        "expression": "Looking back over shoulder",
        "composition_intent": "asymmetric balance",
    },
    {
        "id": "smoke_scene_005",
        "pose": "seated figurative",
        "camera": "upper body shot",
        "camera_angle": "high angle",
        "lighting": "soft window fill",
        "environment_detail": "morning bedroom",
        "mood_note": "Quiet melancholy",
        "expression": "Eyes downcast in thought",
        "composition_intent": "rule of thirds",
    },
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Smoke test for SceneFacetGenerator. Runs the LLM on a "
            "handful of synthetic scenes and reports which tier-required "
            "axes were populated."
        ),
    )
    p.add_argument(
        "--family", default="chroma",
        help="Target family id (default: chroma).",
    )
    p.add_argument(
        "--level", default="T3_artnude",
        choices=["T1_suggestive", "T2_implied", "T3_artnude", "T4_explicit"],
        help="Content level — controls which fields are tier-required.",
    )
    p.add_argument(
        "--scenes", type=int, default=3,
        help=(
            f"Number of scenes to test (1 - {len(_SAMPLE_SCENES)}). "
            f"Default: 3."
        ),
    )
    p.add_argument(
        "--llm", default=None,
        help=(
            "Override the LLM registry id (default: registry default_llm). "
            "Resolved against config/llm_models.yaml."
        ),
    )
    p.add_argument(
        "--out", default="/tmp/facet_smoke.json",
        help="Path to write the raw facet JSON for inspection.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    n = max(1, min(args.scenes, len(_SAMPLE_SCENES)))
    scenes = _SAMPLE_SCENES[:n]

    family = FamilyLoader().get_family(args.family)

    # Resolve LLM model tag from the registry so the CLI takes a
    # registry id (e.g. cydonia_heretic_24b) rather than a raw Ollama
    # tag. Falls back to the registry's default_llm.
    llm_registry = LLMRegistryLoader()
    llm_id = args.llm or llm_registry.default_llm_id
    llm_entry = llm_registry.get_llm(llm_id, require_active=True)
    model_tag = llm_entry.model_tag

    # Build the backend-aware pool so this script handles Ollama /
    # LM Studio / MLX entries uniformly — the SceneFacetGenerator only
    # needs an object with the OllamaClient public-method shape, and
    # LLMClientPool exposes that contract while dispatching internally
    # by backend.
    client = LLMClientPool(
        ollama_client=OllamaClient(),
        lm_studio_client=LMStudioClient(),
        mlx_client=MlxClient(),
        registry=llm_registry,
    )
    gen = SceneFacetGenerator(client)

    print(
        f"family={args.family} level={args.level} llm={llm_id} "
        f"({model_tag}) scenes={n}"
    )

    results: list[dict] = []
    t_all = time.time()
    try:
        for i, scene in enumerate(scenes):
            t_scene = time.time()
            print(f"\n=== scene {i+1}/{n} — generating facet ===")
            facet = gen.generate(
                scene=scene,
                family=family,
                content_level=args.level,
                model=model_tag,
            )
            dt = time.time() - t_scene
            print(f"  ({dt:.1f}s)")
            results.append(facet)
    finally:
        # Hard invariant (CLAUDE.md): LLM must release unified memory
        # before any subsequent ComfyUI run. Always unload, even when
        # the generator raised mid-scene.
        print("\n=== unloading LLM ===")
        client.unload_all()

    total = time.time() - t_all
    print(f"\n=== {n} scenes done in {total:.1f}s ===\n")

    # Tabulate every tier-required field for the active content level.
    # Pulled from _TIER_REQUIRED_FIELDS so the table self-updates as
    # the schema contract evolves.
    required_for_tier: tuple[str, ...] = _TIER_REQUIRED_FIELDS.get(
        args.level, ()
    )
    # Plus a fixed set of "look at this too" optional / encouraged
    # fields for context.
    extra_fields = (
        "realism_framing", "realism_film_stock",
        "environment_prop", "composition_principle",
        "nsfw_posture",
    )
    all_fields = tuple(required_for_tier) + tuple(
        f for f in extra_fields if f not in required_for_tier
    )

    header = f"{'field':30s}" + " ".join(f"s{i+1:02d}" for i in range(n))
    print(header)
    print("-" * len(header))
    for field in all_fields:
        vals = [(r.get(field) or "(null)") for r in results]
        cells = " ".join(f"{v[:18]:18s}" for v in vals)
        marker = " *REQUIRED*" if field in required_for_tier else ""
        print(f"{field:30s} {cells}{marker}")

    print("\n=== tier-required coverage ===")
    missing: dict[str, list[int]] = {f: [] for f in required_for_tier}
    for f in required_for_tier:
        for i, r in enumerate(results):
            if not r.get(f):
                missing[f].append(i + 1)
    all_full = True
    for f, miss in missing.items():
        if miss:
            print(f"  FAIL  {f}: NULL on scenes {miss}")
            all_full = False
        else:
            print(f"  OK    {f}: populated on {n}/{n} scenes")
    print()

    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Raw facets written to {out_path}")
    return 0 if all_full else 1


if __name__ == "__main__":
    raise SystemExit(main())
