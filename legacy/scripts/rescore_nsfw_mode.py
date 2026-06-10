"""Re-score series_83453d626f45 images in both SFW and NSFW modes.

One-off validation script for the NSFW-aware scorer (CompositeWeights
.nsfw_legacy + tier-aware blur/face/aesthetic floors).

Computes:
  * composite_sfw   — content_level=None (legacy behavior)
  * composite_nsfw  — content_level='T4_explicit' (new NSFW mode)
  * delta           — nsfw - sfw

And reports how many additional images now pass the 0.55 cutoff.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from src.scoring.image_scorer import ImageScorer

DB = Path(__file__).resolve().parents[1] / "nsfw_pipeline.db"
SERIES = "series_83453d626f45"
IMG_DIR = Path(
    "/Users/shuvajit/Dev/nsfw-pipeline/output/T4_explicit/series_83453d626f45/"
    "davidau_nemo_thinking_heretic_claude_opus/gonzalomo_chroma_v30/images"
)
CUTOFF = 0.55


def _raw_renders() -> list[Path]:
    """Return one Path per unique raw render (strip NNN_ prefix dupes)."""
    seen: dict[str, Path] = {}
    for p in IMG_DIR.iterdir():
        if not p.suffix == ".png":
            continue
        base = p.name
        if base[:4] == base[:3] + "_" and base[:3].isdigit():
            base = base[4:]
        if base not in seen:
            seen[base] = p
    return sorted(seen.values(), key=lambda p: p.name)


def _prompt_text_for(stem: str, conn: sqlite3.Connection) -> str | None:
    cur = conn.cursor()
    row = cur.execute(
        "SELECT prompt_text FROM prompts WHERE id = ("
        "  SELECT prompt_id FROM images WHERE file_path LIKE ? LIMIT 1"
        ")",
        (f"%{stem}%",),
    ).fetchone()
    return row[0] if row else None


def main() -> None:
    scorer = ImageScorer(use_hps_v2=False, use_image_reward=False)
    paths = _raw_renders()
    print(f"Found {len(paths)} unique raw renders")

    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row

    rows: list[dict] = []
    for p in paths:
        prompt = _prompt_text_for(p.stem, conn)
        sfw = scorer.score(p, prompt=prompt, content_level=None)
        nsfw = scorer.score(p, prompt=prompt, content_level="T4_explicit")
        rows.append({
            "name": p.name[-40:],
            "sfw": sfw["composite"],
            "nsfw": nsfw["composite"],
            "aes": sfw["aesthetic"],
            "blur": sfw["blur"],
            "face": sfw["face_confidence"],
            "sfw_flags": sfw["flags"],
            "nsfw_flags": nsfw["flags"],
        })

    rows.sort(key=lambda r: r["nsfw"], reverse=True)

    print(f"\n{'file':<42}{'sfw':>7}{'nsfw':>7}{'Δ':>7}{'aes':>6}{'blur':>7}{'face':>7}  flags(sfw → nsfw)")
    print("-" * 120)
    flips_in, flips_out, both_pass, both_fail = 0, 0, 0, 0
    for r in rows:
        delta = r["nsfw"] - r["sfw"]
        sfw_pass = r["sfw"] >= CUTOFF
        nsfw_pass = r["nsfw"] >= CUTOFF
        flip = ""
        if not sfw_pass and nsfw_pass:
            flip = " +PASS"
            flips_in += 1
        elif sfw_pass and not nsfw_pass:
            flip = " -FAIL"
            flips_out += 1
        elif sfw_pass and nsfw_pass:
            both_pass += 1
        else:
            both_fail += 1
        print(
            f"{r['name']:<42}{r['sfw']:>7.3f}{r['nsfw']:>7.3f}"
            f"{delta:>+7.3f}{r['aes']:>6.2f}{r['blur']:>7.1f}"
            f"{r['face']:>7.3f}  "
            f"{','.join(r['sfw_flags']) or '-':<35} → "
            f"{','.join(r['nsfw_flags']) or '-'}{flip}"
        )

    print()
    print(f"Pass-cutoff ({CUTOFF}) summary:")
    print(f"  SFW pass:        {sum(1 for r in rows if r['sfw'] >= CUTOFF)}/{len(rows)}")
    print(f"  NSFW pass:       {sum(1 for r in rows if r['nsfw'] >= CUTOFF)}/{len(rows)}")
    print(f"  Flipped IN  (-→+): {flips_in}")
    print(f"  Flipped OUT (+→-): {flips_out}")
    print(f"  Both pass:       {both_pass}")
    print(f"  Both fail:       {both_fail}")


if __name__ == "__main__":
    main()
