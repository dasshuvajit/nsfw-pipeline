"""Chroma model A/B — gonzaLomo v3.0 vs UnCanny base vs UnCanny flash.

BASE-ONLY comparison (no SDXL refiner / FaceDetailer — those are
content-dependent overwrites that would mask the base-quality gap).
Each model renders at its OWN native contract (its template), the SAME
prompts + seeds + resolution across all three. We time every render and
score with ImageScorer (aesthetic + composite — prompt-independent;
ImageReward is env-incompatible). Output: per-model median quality +
median sec/image, side-by-side sheets, report.json → pick by
quality-per-time.

Phases (LLM and ComfyUI never co-resident): generate prompts (LLM) →
unload → render all 3 arms (sequenced to minimise model reloads, cold
first-render discarded from timing) → score → report.

Usage:
  python scripts/chroma_model_ab.py --niche modern_boudoir --tier T3_artnude \
      --prompts 4 --seeds 2
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import median

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import art_director as AD          # noqa: E402
import art_series as A             # noqa: E402
from src.niche.selector import NicheLibrary, build_selection, build_brief  # noqa: E402

# Each arm at its native contract (template); per-model negative text.
# gonzaLomo + flash zero the negative internally (cfg 1) → pass "".
# UnCanny base is cfg 3.5 → negatives ACTIVE → pass the real stack.
MODELS = [
    {"name": "gonzalomo",
     "template": "templates/chroma/gonzalomo_chroma_base.json", "negative": ""},
    {"name": "uncanny_base",
     "template": "templates/chroma/uncanny_base.json", "negative": A.DEFAULT_NEGATIVE},
    {"name": "uncanny_flash",
     "template": "templates/chroma/uncanny_flash.json", "negative": ""},
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Chroma 3-model base-only A/B")
    ap.add_argument("--niche", default="modern_boudoir",
                    help="niche id for the shared prompt set")
    ap.add_argument("--tier", default="T3_artnude",
                    choices=list(AD.TIER_DIRECTIVES))
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--base-seed", type=int, default=101)
    ap.add_argument("--resolution", default="1024x1024",
                    help="WxH, equal across arms for fair time/MP")
    ap.add_argument("--model-tag", default=AD.CYDONIA_TAG)
    ap.add_argument("--temperature", type=float, default=0.85)
    args = ap.parse_args()

    w, h = (int(x) for x in args.resolution.lower().split("x"))
    seeds = [args.base_seed + i for i in range(args.seeds)]
    cfg = yaml.safe_load((ROOT / "config/pipeline.yaml").read_text())
    cu = cfg["comfyui"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "output" / "chroma_model_ab" / ts
    out.mkdir(parents=True, exist_ok=True)

    # ── Phase 1 (LLM): one shared prompt set ───────────────────────────
    print(f"=== Phase 1 (LLM): {args.prompts} prompts, niche={args.niche} ===",
          flush=True)
    lib = NicheLibrary.from_yaml()
    sel = build_selection(lib, 0, tier=args.tier, force_niche=args.niche)
    brief = build_brief(sel)
    rows = AD.generate_series(
        brief=brief, tier=args.tier, count=args.prompts, model_tag=args.model_tag,
        temperature=args.temperature, sub_looks=sel.sub_looks, audit_gate=True)
    prompts = [r["prompt"] for r in rows]
    if not prompts:
        print("No prompts — aborting.", file=sys.stderr)
        return 1
    (out / "prompts.json").write_text(json.dumps(
        {"niche": args.niche, "tier": args.tier, "prompts": prompts}, indent=2))

    print("\n=== Unloading LLM before render ===", flush=True)
    A._unload_llm(args.model_tag)

    # ── Phase 2 (render): arm by arm, timed; cold first-render discarded ─
    from src.render.workflow_builder import WorkflowBuilder
    from src.render.comfyui_client import ComfyUIClient
    client = ComfyUIClient(base_url=cu["base_url"], output_dir=cu["output_dir"])
    items: list[dict] = []  # {model, prompt_idx, seed, path, seconds, cold}
    for m in MODELS:
        arm_dir = out / m["name"]
        arm_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Arm: {m['name']} ({Path(m['template']).name}) ===", flush=True)
        first = True
        for pi, prompt in enumerate(prompts):
            for seed in seeds:
                builder = WorkflowBuilder(ROOT / cu.get("workflow_dir",
                                                        "config/comfyui_workflows"))
                try:
                    wf = builder.build_external(
                        external_template=m["template"], prompt_text=prompt,
                        negative_prompt=m["negative"], resolution=(w, h), seed=seed)
                    t0 = time.perf_counter()
                    imgs = client.render_single_with_retry(wf, timeout=600)
                    secs = time.perf_counter() - t0
                    outs = [im for im in imgs if im.type == "output"] or imgs
                    name = f"p{pi:02d}_s{seed}.png"
                    dst = arm_dir / name
                    shutil.copy(outs[-1].file_path, dst)
                    items.append({"model": m["name"], "prompt_idx": pi, "seed": seed,
                                  "path": str(dst), "seconds": round(secs, 1),
                                  "cold": first})
                    print(f"  {m['name']} p{pi} s{seed} -> {name} ({secs:.1f}s"
                          f"{' COLD-discard' if first else ''})", flush=True)
                    first = False
                except Exception as exc:  # noqa: BLE001
                    print(f"  {m['name']} p{pi} s{seed} FAILED: {exc}",
                          file=sys.stderr, flush=True)

    # ── Phase 3 (score): aesthetic + composite (prompt-independent) ────
    print("\n=== Phase 3: scoring (aesthetic + composite) ===", flush=True)
    try:
        from src.scoring.image_scorer import ImageScorer
        scorer = ImageScorer(use_hps_v2=False, use_image_reward=False)
        for it in items:
            s = scorer.score(it["path"], content_level=args.tier)
            it["aesthetic"] = s.get("aesthetic")
            it["composite"] = s.get("composite")
            it["flags"] = s.get("flags")
    except Exception as exc:  # noqa: BLE001
        print(f"!! scoring unavailable: {exc}", file=sys.stderr, flush=True)

    # ── aggregate (warm renders only for timing) ───────────────────────
    def _med(vals):
        vals = [v for v in vals if v is not None]
        return round(median(vals), 4) if vals else None

    by_model: dict[str, dict] = {}
    for m in MODELS:
        mine = [it for it in items if it["model"] == m["name"]]
        warm = [it for it in mine if not it["cold"]]
        by_model[m["name"]] = {
            "n": len(mine),
            "median_aesthetic": _med([it.get("aesthetic") for it in mine]),
            "median_composite": _med([it.get("composite") for it in mine]),
            "median_sec_warm": _med([it["seconds"] for it in warm]),
            "sec_all": sorted(round(it["seconds"], 1) for it in mine),
        }

    report = {"niche": args.niche, "tier": args.tier, "resolution": [w, h],
              "seeds": seeds, "judge": "LAION aesthetic + composite (+ eyeball)",
              "by_model": by_model, "items": items}
    (out / "report.json").write_text(json.dumps(report, indent=2))

    # side-by-side per prompt (one seed: the base_seed), 3 models in a row
    try:
        from src.review.contact_sheet import create_contact_sheet
        for pi in range(len(prompts)):
            cs = []
            for m in MODELS:
                it = next((x for x in items if x["model"] == m["name"]
                           and x["prompt_idx"] == pi and x["seed"] == seeds[0]), None)
                if it:
                    cs.append({"file_path": it["path"],
                               "quality_score": it.get("aesthetic") or 0.0})
            if cs:
                create_contact_sheet(cs, out / f"compare_p{pi:02d}.png", columns=3)
    except Exception as exc:  # noqa: BLE001
        print(f"  (contact sheets skipped: {exc})", file=sys.stderr)

    print("\n" + "=" * 72 + "\nCHROMA MODEL A/B  (base-only, quality-per-time)")
    print(f"  {'model':16s} {'aesthetic':>10s} {'composite':>10s} {'sec/img(warm)':>14s}  n")
    for name, mm in by_model.items():
        print(f"  {name:16s} {str(mm['median_aesthetic']):>10s} "
              f"{str(mm['median_composite']):>10s} {str(mm['median_sec_warm']):>14s}  "
              f"{mm['n']}")
    print(f"\nReport: {out}/report.json  | side-by-sides: {out}/compare_p*.png")
    print("Eyeball compare_p*.png (the sellability authority) + weigh the "
          "sec/img tradeoff, then pick the production base.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
