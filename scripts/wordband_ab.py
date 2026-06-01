"""Word-band A/B — does a wider prose band help Chroma?

For each niche, generate ONE prompt at each band (same niche/sub-look/
aesthetic-lock, only the word budget differs), render with the SAME seed
per pair (prompt text differs → distinct ComfyUI cache keys), then judge.

METHODOLOGY NOTE — the right question is "does a richer 200-300 word prompt
produce a BETTER IMAGE than 110-160?", so the judge must be PROMPT-INDEPENDENT
image quality, not prompt-alignment:
  * LAION aesthetic (prompt-independent) + the composite — the quantitative
    signal. Commercial-safe (Apache/MIT), already in the pipeline.
  * a side-by-side contact sheet per niche — the human eyeball, the final
    arbiter.
Prompt-alignment judges (ImageReward, CLIP-similarity) are deliberately NOT
used: ImageReward is env-incompatible with modern transformers (would risk the
working pipeline to force it), and CLIP's text encoder truncates at 77 tokens
(~60 words) so it literally can't read a 300-word prompt — both are confounded
by prompt length and wrong for THIS question.

Usage:
  python scripts/wordband_ab.py --niches fine_art_figure_study,old_hollywood_glamour,modern_boudoir,poolside_goldenhour \
      --tier T3_artnude --bands 110-160,200-300 --seeds-per 2
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import art_director as AD          # noqa: E402
import art_series as A             # noqa: E402
from src.niche.selector import (   # noqa: E402
    NicheLibrary, build_selection, build_brief,
)


def _parse_band(s: str) -> tuple[int, int]:
    lo, hi = s.split("-")
    return (int(lo), int(hi))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Word-band A/B (LAION-aesthetic + eyeball judged)")
    ap.add_argument("--niches",
                    default="fine_art_figure_study,old_hollywood_glamour,"
                            "modern_boudoir,poolside_goldenhour",
                    help="comma-separated niche ids")
    ap.add_argument("--tier", default="T3_artnude")
    ap.add_argument("--bands", default="110-160,200-300",
                    help="two 'lo-hi' bands to compare")
    ap.add_argument("--seeds-per", type=int, default=1,
                    help="renders per (niche, band)")
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--model-tag", default=AD.CYDONIA_TAG)
    args = ap.parse_args()

    niche_ids = [n.strip() for n in args.niches.split(",") if n.strip()]
    bands = [_parse_band(b) for b in args.bands.split(",")]
    if len(bands) != 2:
        ap.error("--bands needs exactly two, e.g. 110-160,200-300")

    cfg = yaml.safe_load((ROOT / "config/pipeline.yaml").read_text())
    cu = cfg["comfyui"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "output" / "wordband_ab" / ts
    img = out / "images"
    img.mkdir(parents=True, exist_ok=True)

    lib = NicheLibrary.from_yaml()

    # ── Phase 1 (LLM): generate one prompt per (niche, band) ───────────
    print(f"=== Phase 1 (LLM): {len(niche_ids)} niches × {len(bands)} bands ===",
          flush=True)
    plan: list[dict] = []  # {niche, band, prompt, words, look}
    for ni, nid in enumerate(niche_ids):
        sel = build_selection(lib, ni, tier=args.tier, force_niche=nid)
        brief = build_brief(sel)
        look0 = sel.sub_looks[:1] or None
        for band in bands:
            rows = AD.generate_series(
                brief=brief, tier=args.tier, count=1, model_tag=args.model_tag,
                temperature=args.temperature, sub_looks=look0, word_band=band,
                audit_gate=True)
            if not rows:
                print(f"  !! {nid} {band} produced no prompt", file=sys.stderr)
                continue
            p = rows[0]["prompt"]
            plan.append({"niche": nid, "band": f"{band[0]}-{band[1]}",
                         "band_tuple": band, "prompt": p,
                         "words": len(p.split()), "look": rows[0]["look"]})

    print("\n=== Unloading LLM before render ===", flush=True)
    A._unload_llm(args.model_tag)

    # ── Phase 2 (render): same seed per niche-pair ────────────────────
    print("=== Phase 2: rendering ===", flush=True)
    from src.render.workflow_builder import WorkflowBuilder
    from src.render.comfyui_client import ComfyUIClient
    client = ComfyUIClient(base_url=cu["base_url"], output_dir=cu["output_dir"])
    score_items: list[dict] = []
    for entry in plan:
        ni = niche_ids.index(entry["niche"])
        for k in range(args.seeds_per):
            seed = 1000 + ni * 10 + k     # same seed across bands of a niche
            builder = WorkflowBuilder(ROOT / cu.get("workflow_dir",
                                                    "config/comfyui_workflows"))
            try:
                wf = builder.build_external(
                    external_template=A.DEFAULT_TEMPLATE, prompt_text=entry["prompt"],
                    negative_prompt=A.DEFAULT_NEGATIVE, resolution=(896, 1152),
                    seed=seed)
                imgs = client.render_single_with_retry(wf, timeout=480)
                outs = [im for im in imgs if im.type == "output"] or imgs
                name = f"{entry['niche']}__{entry['band']}__s{seed}.png"
                dst = img / name
                shutil.copy(outs[-1].file_path, dst)
                score_items.append({
                    "file_path": str(dst), "prompt_text": entry["prompt"],
                    "content_level": args.tier, "niche": entry["niche"],
                    "band": entry["band"], "words": entry["words"]})
                print(f"  {name}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"  FAILED {entry['niche']} {entry['band']} s{seed}: {exc}",
                      file=sys.stderr, flush=True)

    # ── Phase 3 (score): prompt-INDEPENDENT quality (aesthetic+composite) ──
    print("\n=== Phase 3: scoring (LAION aesthetic + composite) ===", flush=True)
    try:
        from src.scoring.image_scorer import ImageScorer
        scorer = ImageScorer(use_hps_v2=False, use_image_reward=False)
        for it in score_items:
            s = scorer.score(it["file_path"], content_level=it["content_level"])
            it["aesthetic"] = s.get("aesthetic")
            it["composite"] = s.get("composite")
            it["flags"] = s.get("flags")
    except Exception as exc:  # noqa: BLE001
        print(f"!! scoring unavailable: {exc}", file=sys.stderr, flush=True)

    # ── aggregate + verdict ────────────────────────────────────────────
    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    by_band: dict[str, dict] = {}
    for b in (f"{x[0]}-{x[1]}" for x in bands):
        items = [it for it in score_items if it["band"] == b]
        by_band[b] = {
            "n": len(items),
            "mean_words": _mean([it["words"] for it in items]),
            "mean_aesthetic": _mean([it.get("aesthetic") for it in items]),
            "mean_composite": _mean([it.get("composite") for it in items]),
        }

    # per-niche aesthetic deltas (band B − band A) — shows consistency
    ba, bb = (f"{x[0]}-{x[1]}" for x in bands)
    per_niche = []
    for nid in niche_ids:
        a = _mean([it.get("aesthetic") for it in score_items
                   if it["niche"] == nid and it["band"] == ba])
        b = _mean([it.get("aesthetic") for it in score_items
                   if it["niche"] == nid and it["band"] == bb])
        per_niche.append({"niche": nid, f"aesthetic_{ba}": a,
                          f"aesthetic_{bb}": b,
                          "delta": (round(b - a, 3) if a is not None and b is not None else None)})

    report = {"tier": args.tier, "niches": niche_ids, "bands": [ba, bb],
              "judge": "LAION aesthetic (prompt-independent) + composite + eyeball",
              "by_band": by_band, "per_niche": per_niche, "items": score_items}
    (out / "report.json").write_text(json.dumps(report, indent=2))

    # side-by-side contact sheets per niche (the real arbiter)
    try:
        from src.review.contact_sheet import create_contact_sheet
        for nid in niche_ids:
            cs = [{"file_path": it["file_path"],
                   "quality_score": it.get("aesthetic") or 0.0}
                  for it in score_items if it["niche"] == nid]
            if cs:
                create_contact_sheet(cs, out / f"compare_{nid}.png", columns=2)
    except Exception as exc:  # noqa: BLE001
        print(f"  (contact sheets skipped: {exc})", file=sys.stderr)

    print("\n" + "=" * 70 + "\nWORD-BAND A/B RESULT  (judge: prompt-independent aesthetic)")
    for b, m in by_band.items():
        print(f"  band {b}: n={m['n']} words~{m['mean_words']} | "
              f"aesthetic={m['mean_aesthetic']} | composite={m['mean_composite']}")
    print("  per-niche aesthetic delta (B−A):")
    for r in per_niche:
        print(f"    {r['niche']:28s} {ba}={r.get('aesthetic_'+ba)} "
              f"{bb}={r.get('aesthetic_'+bb)}  Δ={r['delta']}")
    a_aes, b_aes = by_band[ba]["mean_aesthetic"], by_band[bb]["mean_aesthetic"]
    if a_aes is not None and b_aes is not None:
        win = bb if b_aes > a_aes else ba
        wins_b = sum(1 for r in per_niche if (r["delta"] or 0) > 0)
        print(f"\n  VERDICT (aesthetic): {win} wins on the mean "
              f"(Δ={round(b_aes - a_aes, 3)}); band {bb} wins {wins_b}/{len(per_niche)} "
              f"niches. EYEBALL compare_*.png — it's the real arbiter.")
    print(f"\nReport: {out}/report.json  | side-by-sides: {out}/compare_*.png",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
