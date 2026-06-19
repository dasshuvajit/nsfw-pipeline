#!/usr/bin/env python3
"""GPU-free diversity report for a prompts-only run dir (or two, for A/B).

Measures whether the prompt engine actually swings wide:
  - distinct-3gram coverage     (unique 3-grams / total 3-grams; higher = less repetition)
  - mean cross-pair similarity  (mean 3-gram Jaccard over all prompt pairs; LOWER = more varied)
  - max cross-pair similarity   (worst near-duplicate pair)
  - opener variety              (distinct first-6-word openers / N)
  - scene/look/garment entropy  (normalized Shannon entropy of the per-image axis labels)

Usage:
  python scripts/diversity_report.py output/art_series/<ts>            # one run
  python scripts/diversity_report.py <baseline_ts_dir> <new_ts_dir>    # A/B delta
"""
import json
import math
import sys
from itertools import combinations
from pathlib import Path


_GARMENT_NOUNS = (
    "kaftan", "kimono", "robe", "slip", "negligee", "babydoll", "teddy", "bodysuit",
    "corset", "bustier", "bralette", "lingerie", "bikini", "swimsuit", "monokini",
    "leotard", "catsuit", "blouse", "shirt", "sweater", "cardigan", "turtleneck",
    "dress", "gown", "skirt", "shorts", "leggings", "jeans", "trousers", "sarong",
    "wrap", "camisole", "tank", "crop top", "bra", "panties", "thong", "stockings",
    "bodystocking", "chemise", "halter", "kaftan", "tunic", "jumpsuit", "romper",
    "overalls", "blazer", "trench", "coat", "veil", "lace", "mesh", "satin gown",
)


def _detect_garment(prompt: str):
    low = prompt.lower()
    for g in _GARMENT_NOUNS:
        if g in low:
            return g
    return ""


def _words(text: str):
    return [w for w in "".join(c.lower() if (c.isalnum() or c.isspace()) else " "
                          for c in text).split() if w]


def _grams(words, n=3):
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _entropy(labels):
    labels = [x for x in labels if x]
    if not labels:
        return 0.0
    counts = {}
    for x in labels:
        counts[x] = counts.get(x, 0) + 1
    total = len(labels)
    h = -sum((c / total) * math.log2(c / total) for c in counts.values())
    hmax = math.log2(len(counts)) if len(counts) > 1 else 1.0
    return h / hmax if hmax else 0.0


def _load(run_dir: Path):
    """Return list of dicts: {prompt, scene, garment, sub_look}. Tolerant of schema."""
    for fname in ("prompts.json", "manifest.json"):
        f = run_dir / fname
        if f.exists():
            raw = json.loads(f.read_text())
            items = raw.get("prompts") or raw.get("scenes") or raw.get("images") or raw
            if isinstance(items, dict):
                items = list(items.values())
            out = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                p = (it.get("prompt") or it.get("text") or it.get("positive") or "")
                if not p:
                    continue
                out.append({
                    "prompt": p,
                    "scene": (it.get("scene") or it.get("scene_label")
                              or it.get("sub_look") or it.get("look") or ""),
                    "garment": (it.get("garment_type") or it.get("garment")
                                or _detect_garment(p)),
                    "sub_look": it.get("sub_look") or it.get("look") or "",
                })
            if out:
                return out
    raise SystemExit(f"no prompts found in {run_dir}")


def report(run_dir: Path):
    items = _load(run_dir)
    grams = [_grams(_words(it["prompt"])) for it in items]
    all_g, uniq_g = 0, set()
    for it in items:
        w = _words(it["prompt"])
        gs = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
        all_g += len(gs)
        uniq_g.update(gs)
    coverage = len(uniq_g) / all_g if all_g else 0.0
    sims = [_jaccard(a, b) for a, b in combinations(grams, 2)] or [0.0]
    openers = {" ".join(_words(it["prompt"])[:6]) for it in items}
    return {
        "n": len(items),
        "distinct_3gram_coverage": round(coverage, 3),
        "mean_pair_similarity": round(sum(sims) / len(sims), 3),
        "max_pair_similarity": round(max(sims), 3),
        "opener_variety": round(len(openers) / len(items), 3),
        "scene_entropy": round(_entropy([it["scene"] for it in items]), 3),
        "garment_entropy": round(_entropy([it["garment"] for it in items]), 3),
        "sublook_entropy": round(_entropy([it["sub_look"] for it in items]), 3),
    }


def _fmt(label, r):
    print(f"\n=== {label}  (n={r['n']}) ===")
    print(f"  distinct-3gram coverage : {r['distinct_3gram_coverage']:.3f}   (higher = less repetition)")
    print(f"  mean pair similarity    : {r['mean_pair_similarity']:.3f}   (LOWER = more varied)")
    print(f"  max  pair similarity    : {r['max_pair_similarity']:.3f}   (worst near-dup pair)")
    print(f"  opener variety          : {r['opener_variety']:.3f}   (distinct openers / N)")
    print(f"  scene entropy           : {r['scene_entropy']:.3f}")
    print(f"  garment entropy         : {r['garment_entropy']:.3f}")
    print(f"  sub_look entropy        : {r['sublook_entropy']:.3f}")


if __name__ == "__main__":
    dirs = [Path(a) for a in sys.argv[1:]]
    if not dirs:
        raise SystemExit(__doc__)
    if len(dirs) == 1:
        _fmt(dirs[0].name, report(dirs[0]))
    else:
        a, b = report(dirs[0]), report(dirs[1])
        _fmt(f"BASELINE {dirs[0].name}", a)
        _fmt(f"NEW      {dirs[1].name}", b)
        print("\n=== DELTA (new - baseline) ===")
        for k in a:
            if k == "n":
                continue
            d = b[k] - a[k]
            arrow = "↑" if d > 0 else ("↓" if d < 0 else "·")
            print(f"  {k:24s}: {d:+.3f} {arrow}")
