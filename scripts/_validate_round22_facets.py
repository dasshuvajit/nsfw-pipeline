"""Round-22 quick validator — invoke SceneFacetGenerator on 3 synthetic
scenes via Ollama Cydonia heretic. Verifies that the freshly-promoted
T3+ required fields (``art_style_reference``, ``realism_angle``) come
back populated.

Avoids the full prepare_prompts pipeline (planner + scene-generator +
prompt composer + dedup + sanitizer) which would take 40+ minutes.

Run: python scripts/_validate_round22_facets.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.scene_facet_generator import SceneFacetGenerator  # noqa: E402
from src.agents.llm_client import OllamaClient  # noqa: E402
from src.memory.family_loader import FamilyLoader  # noqa: E402


SCENES = [
    {
        "id": "test_scene_001",
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
        "id": "test_scene_002",
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
        "id": "test_scene_003",
        "pose": "kneeling graceful",
        "camera": "three-quarter body shot",
        "camera_angle": "low angle",
        "lighting": "Rembrandt-style chiaroscuro",
        "environment_detail": "minimalist artist studio",
        "mood_note": "Defiant strength",
        "expression": "Direct confident gaze at camera",
        "composition_intent": "centered hero shot",
    },
]


def main() -> int:
    loader = FamilyLoader()
    family = loader.get_family("chroma")
    client = OllamaClient()
    gen = SceneFacetGenerator(client)

    results: list[dict] = []
    t0 = time.time()
    try:
        for i, scene in enumerate(SCENES):
            per_t0 = time.time()
            print(f"\n=== Scene {i+1}/3 — generating facet ===")
            facet = gen.generate(
                scene=scene,
                family=family,
                content_level="T3_artnude",
            )
            dt = time.time() - per_t0
            print(f"  ({dt:.1f}s)")
            results.append(facet)
    finally:
        # Hard invariant (CLAUDE.md): LLM must release unified memory
        # before any subsequent ComfyUI run. Always unload, even when
        # the generator raised mid-scene. Cydonia 24B holds ~21.5 GB
        # of VRAM; leaving it loaded blocks Phase B and burns memory
        # on idle.
        print("\n=== Unloading LLM ===")
        client.unload_all()

    total = time.time() - t0
    print(f"\n=== All 3 scenes done in {total:.1f}s ===\n")

    # Tabulate the T3+ required structured-tag fields, with focus on
    # the round-22 newly-promoted ones.
    REQUIRED_FIELDS = [
        "lighting_directive", "mood_aesthetic", "narrative_moment",
        "environment_setting", "environment_atmosphere", "nsfw_anatomy",
        "realism_camera", "realism_lens",
        # Round-22 promotions:
        "realism_angle", "art_style_reference",
    ]
    print(f"{'field':28s} " + " ".join(f"scene_{i+1:02d}" for i in range(3)))
    print("-" * 65)
    for field in REQUIRED_FIELDS:
        vals = [(r.get(field) or "(null)") for r in results]
        marker = "  ← round-22" if field in (
            "realism_angle", "art_style_reference",
        ) else ""
        print(f"{field:28s} " + " ".join(f"{v[:18]:18s}" for v in vals) + marker)

    print()
    # Verdict on the round-22 promotion.
    new_required = ("realism_angle", "art_style_reference")
    missing = {f: [] for f in new_required}
    for f in new_required:
        for i, r in enumerate(results):
            if not r.get(f):
                missing[f].append(i + 1)
    print("=== Round-22 promotion verdict ===")
    for f, miss in missing.items():
        if not miss:
            print(f"  ✓ {f}: populated on 3/3 scenes")
        else:
            print(f"  ✗ {f}: NULL on scenes {miss}")

    # Persist results for follow-up inspection.
    with open("/tmp/round22_facets.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nRaw facets written to /tmp/round22_facets.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
