"""Art-series — end-to-end LLM-direct production flow.

Generate → render → (manual) pick, with NONE of the structured
vocab/composer machinery. Phase 1 calls the art-director prompt engine
(scripts/art_director.py) for N rich, varied prompts; Phase 2 unloads the
LLM and renders each through an external ComfyUI template (the v11 Chroma
refiner by default), saving every candidate plus a manifest so a human
can cherry-pick.

The two phases are sequential (LLM never co-resident with ComfyUI). With
--seeds K>1 each prompt is rendered K times for curation.

Usage:
  python scripts/art_series.py --brief "Mediterranean summer" \
      --tier T3_artnude --count 6
  python scripts/art_series.py --brief "vintage boudoir" --tier T4_explicit \
      --count 8 --seeds 3 --orientation portrait
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import art_director  # noqa: E402
from src.render.workflow_builder import WorkflowBuilder  # noqa: E402
from src.render.comfyui_client import ComfyUIClient  # noqa: E402
from src.niche.selector import (  # noqa: E402
    NicheLibrary, NicheLibraryError, build_selection, build_brief,
)

DEFAULT_TEMPLATE = "templates/chroma/gonzaLomo_Chroma_Refiner_v11.json"

# Persistent seed counter — guarantees every render across runs gets a
# never-before-used seed, defeating ComfyUI's per-node cache collisions
# (which hit us on the seed-9 retry: cache key matched a prior submission).
SEED_COUNTER_FILE = ROOT / "output/art_series/.last_seed"
LEGACY_DEFAULT_SEED = 7

# Persistent niche cursor — advances each --auto run so successive runs
# rotate through evergreen-core niches and periodically inject a trend
# niche (deterministic, no randomness; mirrors the seed counter pattern).
NICHE_CURSOR_FILE = ROOT / "output/art_series/.niche_cursor"

# gonzalomo_chroma_v30 resolution presets.
ORIENTATIONS = {
    "portrait": (896, 1152),
    "square": (1024, 1024),
    "landscape": (1152, 896),
}

# Self-contained default negative (age-safety + multi-subject + anatomy +
# render-risk). Kept here so the production flow does not depend on the DB.
DEFAULT_NEGATIVE = (
    "child, kid, young, minor, teen, teenager, schoolgirl, loli, underage, "
    "baby, toddler, preteen, youthful face, "
    "2girls, multiple girls, multiple subjects, two women, couple, group, "
    "grid, collage, polyptych, split screen, split image, "
    "mirror, reflection, double face, "
    "bad anatomy, bad hands, missing fingers, extra fingers, extra limbs, "
    "deformed, mutated, disfigured, fused fingers, malformed, "
    "lowres, blurry, jpeg artifacts, watermark, text, signature, "
    "cartoon, anime, illustration, 3d render, cgi, plastic skin, airbrushed"
)


def _read_seed_counter() -> int | None:
    """Return the highest seed used by any prior art_series run, or None."""
    try:
        return int(SEED_COUNTER_FILE.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def _next_base_seed(explicit: int | None) -> int:
    """Pick the base seed for this run. Explicit --base-seed overrides; else
    advance one past the persisted counter; else fall back to the legacy
    default."""
    if explicit is not None:
        return explicit
    prior = _read_seed_counter()
    if prior is None:
        return LEGACY_DEFAULT_SEED
    return prior + 1


def _record_max_seed(seed: int) -> None:
    """Write back the highest seed used so the next run advances past it."""
    SEED_COUNTER_FILE.parent.mkdir(parents=True, exist_ok=True)
    prior = _read_seed_counter() or -1
    SEED_COUNTER_FILE.write_text(str(max(prior, seed)))


def _read_niche_cursor() -> int:
    try:
        return int(NICHE_CURSOR_FILE.read_text().strip())
    except (FileNotFoundError, ValueError):
        return 0


def _advance_niche_cursor(cursor: int) -> None:
    NICHE_CURSOR_FILE.parent.mkdir(parents=True, exist_ok=True)
    NICHE_CURSOR_FILE.write_text(str(cursor + 1))


def _unload_llm(model_tag: str) -> None:
    """Free the Ollama model before the render phase (never co-resident
    with ComfyUI). Best-effort: `ollama stop`, then a short grace period."""
    try:
        subprocess.run(["ollama", "stop", model_tag], timeout=30,
                       capture_output=True)
    except Exception:  # noqa: BLE001
        pass
    time.sleep(3)


def main() -> int:
    ap = argparse.ArgumentParser(description="LLM-direct art series (gen→render)")
    # Niche selection (one of --auto / --niche / --brief is required):
    ap.add_argument("--auto", action="store_true",
                    help="auto-pick a niche from config/niche_library.yaml "
                    "(evergreen-core + periodic trend, via the niche cursor)")
    ap.add_argument("--niche", default=None,
                    help="force a specific niche id from the library")
    ap.add_argument("--persona", action="store_true",
                    help="bind a recurring persona (rotated from the pool)")
    ap.add_argument("--persona-name", default=None,
                    help="bind a specific persona by name (e.g. Clara)")
    ap.add_argument("--brief", default=None,
                    help="manual creative brief/theme (overrides niche selection)")
    ap.add_argument("--tier", default="T3_artnude",
                    choices=list(art_director.TIER_DIRECTIVES))
    ap.add_argument("--count", type=int, default=6, help="number of prompts")
    ap.add_argument("--seeds", type=int, default=1,
                    help="renders per prompt (>1 = candidates to pick from)")
    ap.add_argument("--orientation", default="portrait",
                    choices=list(ORIENTATIONS))
    ap.add_argument("--template", default=DEFAULT_TEMPLATE,
                    help="external ComfyUI template (relative to workflow_dir)")
    ap.add_argument("--model-tag", default=art_director.CYDONIA_TAG,
                    help="Ollama LLM tag for prompt generation")
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--word-band", default="110-160",
                    help="prose word band 'lo-hi' (A/B: try 200-300)")
    ap.add_argument("--no-audit-gate", action="store_true",
                    help="disable the inline audit_prompts quality gate")
    ap.add_argument("--base-seed", type=int, default=None,
                    help="explicit base seed; default = auto-advance past the "
                    "highest seed any prior run used (output/art_series/.last_seed)")
    ap.add_argument("--out-dir", default="",
                    help="output dir (default: output/art_series/<timestamp>)")
    args = ap.parse_args()

    if not (args.auto or args.niche or args.brief):
        ap.error("provide one of --auto, --niche <id>, or --brief <text>")

    try:
        lo_s, hi_s = args.word_band.split("-")
        word_band = (int(lo_s), int(hi_s))
    except ValueError:
        ap.error("--word-band must be 'lo-hi', e.g. 110-160 or 200-300")

    cfg = yaml.safe_load((ROOT / "config/pipeline.yaml").read_text())
    cu = cfg["comfyui"]
    workflow_dir = ROOT / cu.get("workflow_dir", "config/comfyui_workflows")
    resolution = ORIENTATIONS[args.orientation]

    # ── Resolve brief + sub-looks via the niche selector (unless --brief) ──
    selection = None
    sub_looks = None
    brief = args.brief
    if not args.brief:
        try:
            library = NicheLibrary.from_yaml()
            niche_cursor = _read_niche_cursor()
            selection = build_selection(
                library, niche_cursor, tier=args.tier,
                force_niche=args.niche,
                persona=args.persona or bool(args.persona_name),
                persona_name=args.persona_name,
            )
        except NicheLibraryError as exc:
            ap.error(f"niche selection failed: {exc}")
        brief = build_brief(selection)
        sub_looks = selection.sub_looks
        _advance_niche_cursor(niche_cursor)
        print(f"=== niche: {selection.niche.id} ({selection.niche.niche_class}) "
              f"| folder={selection.niche.da_folder!r} "
              f"| persona={selection.persona.name if selection.persona else 'none'} "
              f"| tier={args.tier} ===", flush=True)
    base_seed = _next_base_seed(args.base_seed)
    print(f"=== base_seed for this run: {base_seed} "
          f"(seeds {base_seed}..{base_seed + args.seeds - 1}) ===", flush=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (
        ROOT / "output" / "art_series" / ts
    )
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: prompts (LLM) ─────────────────────────────────────────
    print(f"\n=== Phase 1: generating {args.count} prompts via "
          f"{args.model_tag} ===", flush=True)
    rows = art_director.generate_series(
        brief=brief,
        tier=args.tier,
        count=args.count,
        model_tag=args.model_tag,
        temperature=args.temperature,
        sub_looks=sub_looks,
        word_band=word_band,
        audit_gate=not args.no_audit_gate,
    )
    if not rows:
        print("No prompts generated — aborting.", file=sys.stderr)
        return 1

    # ── unload LLM before rendering ────────────────────────────────────
    print("\n=== Unloading LLM before render phase ===", flush=True)
    _unload_llm(args.model_tag)

    # ── Phase 2: render ────────────────────────────────────────────────
    print(f"\n=== Phase 2: rendering {len(rows)} prompts × {args.seeds} "
          f"seed(s) via {Path(args.template).name} ===", flush=True)
    builder = WorkflowBuilder(workflow_dir)
    client = ComfyUIClient(base_url=cu["base_url"], output_dir=cu["output_dir"])

    manifest: list[dict] = []
    for idx, r in enumerate(rows):
        look = r["look"].split()[0]
        entry = {"index": idx, "look": r["look"], "prompt": r["prompt"],
                 "images": []}
        for k in range(args.seeds):
            seed = base_seed + k
            try:
                wf = builder.build_external(
                    external_template=args.template,
                    prompt_text=r["prompt"],
                    negative_prompt=DEFAULT_NEGATIVE,
                    resolution=resolution,
                    seed=seed,
                )
                images = client.render_single_with_retry(wf, timeout=480)
                outs = [im for im in images if im.type == "output"] or images
                name = f"ad{idx + 1:02d}_{look}_s{seed}.png"
                dst = img_dir / name
                shutil.copy(outs[-1].file_path, dst)
                entry["images"].append(str(dst.relative_to(out_dir)))
                print(f"  [{idx + 1}/{len(rows)}] seed {seed} -> {name}",
                      flush=True)
            except Exception as exc:  # noqa: BLE001 — surface, continue series
                print(f"  [{idx + 1}/{len(rows)}] seed {seed} FAILED: {exc}",
                      file=sys.stderr, flush=True)
        manifest.append(entry)

    niche_meta = None
    if selection is not None:
        niche_meta = {
            "id": selection.niche.id,
            "class": selection.niche.niche_class,
            "da_folder": selection.niche.da_folder,
            "tags": selection.niche.tags,
            "aesthetic_lock": {
                "palette": selection.aesthetic_lock.palette,
                "lighting": selection.aesthetic_lock.lighting,
                "photographer": selection.aesthetic_lock.photographer,
            },
            "persona": selection.persona.name if selection.persona else None,
        }
    (out_dir / "manifest.json").write_text(json.dumps({
        "brief": brief, "tier": args.tier, "template": args.template,
        "model_tag": args.model_tag, "orientation": args.orientation,
        "resolution": resolution, "seeds_per_prompt": args.seeds,
        "base_seed": base_seed, "word_band": list(word_band),
        "seeds": list(range(base_seed, base_seed + args.seeds)),
        "niche": niche_meta,
        "prompts": manifest,
    }, indent=2))

    _record_max_seed(base_seed + args.seeds - 1)

    n_img = sum(len(e["images"]) for e in manifest)
    print(f"\nDONE — {len(rows)} prompts, {n_img} images -> {out_dir}", flush=True)
    print("Cherry-pick the keepers from images/; manifest.json maps each "
          "image to its prompt.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
