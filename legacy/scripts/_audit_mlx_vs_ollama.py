"""One-off A/B audit: MLX Cydonia v3.1 vs Ollama Cydonia-heretic-24B.

Reads the two series rows + scene_facets + prompts via SQL, computes:
  * facet enum-tag diversity per axis
  * tier compliance (NSFW anatomy presence at T3+)
  * scene_prose length distribution
  * canonicalizer drift (unknown concept count from log)
  * wall-clock from external log file

Output goes to stdout — pipe to a markdown file if you want to keep it.

Usage:
    python scripts/_audit_mlx_vs_ollama.py \\
        --baseline series_1722bde99e06 \\
        --candidate <MLX series_id> \\
        --baseline-log /tmp/ollama_run.log \\
        --candidate-log /tmp/mlx_run.log

Round-20 — generated for the MLX integration validation pass.
Intentionally a one-off; delete or move to docs/ after the audit lands.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter
from pathlib import Path


_AXES = [
    "realism_camera",
    "realism_lens",
    "realism_film_stock",
    "art_style_reference",
    "lighting_directive",
    "mood_aesthetic",
    "nsfw_anatomy",
    "nsfw_posture",
]


def _fetch_facets(db, series_id: str) -> list[dict]:
    cols = ", ".join(_AXES + ["scene_prose", "booru_tags"])
    sql = (
        f"SELECT {cols} FROM scene_facets sf "
        f"JOIN scenes s ON s.id = sf.scene_id "
        f"WHERE s.series_id = ?"
    )
    rows = db.execute(sql, (series_id,)).fetchall()
    keys = _AXES + ["scene_prose", "booru_tags"]
    return [dict(zip(keys, r)) for r in rows]


def _fetch_series_meta(db, series_id: str) -> dict:
    row = db.execute(
        "SELECT id, theme, mood, environment, content_level, "
        "actual_count, target_count, completed_at, created_at "
        "FROM series WHERE id = ?",
        (series_id,),
    ).fetchone()
    keys = [
        "id", "theme", "mood", "environment", "content_level",
        "actual_count", "target_count", "completed_at", "created_at",
    ]
    return dict(zip(keys, row)) if row else {}


def _fetch_prompts(db, series_id: str) -> list[dict]:
    sql = (
        "SELECT p.prompt_text, p.negative_prompt, p.target_kind, "
        "p.vocab_version, p.llm_id "
        "FROM prompts p JOIN scenes s ON s.id = p.scene_id "
        "WHERE s.series_id = ?"
    )
    rows = db.execute(sql, (series_id,)).fetchall()
    keys = ["positive", "negative", "target_kind", "vocab_version", "llm_id"]
    return [dict(zip(keys, r)) for r in rows]


def _diversity_block(label: str, facets: list[dict]) -> str:
    out = [f"### {label}", ""]
    for axis in _AXES:
        vals = [f[axis] for f in facets if f[axis]]
        if not vals:
            out.append(f"- **{axis}**: (none populated)")
            continue
        c = Counter(vals)
        unique = len(c)
        top = c.most_common(3)
        line = f"- **{axis}**: {unique} unique / {len(vals)} populated · top: " + ", ".join(
            f"`{v}` ({n})" for v, n in top
        )
        out.append(line)
    out.append("")
    return "\n".join(out)


def _prose_block(label: str, facets: list[dict]) -> str:
    lens = [len((f["scene_prose"] or "").split()) for f in facets if f["scene_prose"]]
    if not lens:
        return f"### {label} prose\n\n- no scene_prose populated\n"
    mean = sum(lens) / len(lens)
    return (
        f"### {label} scene_prose\n\n"
        f"- N = {len(lens)} populated rows\n"
        f"- mean = {mean:.1f} words, min = {min(lens)}, max = {max(lens)}\n\n"
    )


def _tier_block(label: str, facets: list[dict], content_level: str) -> str:
    n_total = len(facets)
    n_anatomy = sum(1 for f in facets if f["nsfw_anatomy"])
    n_posture = sum(1 for f in facets if f["nsfw_posture"])
    return (
        f"### {label} tier compliance (level={content_level})\n\n"
        f"- nsfw_anatomy populated: {n_anatomy}/{n_total}\n"
        f"- nsfw_posture populated: {n_posture}/{n_total}\n\n"
    )


def _log_metrics(label: str, log_path: Path) -> str:
    if not log_path.exists():
        return f"### {label} log\n\n- log file missing: {log_path}\n"
    text = log_path.read_text()
    unknown = text.count("unknown concept")
    retries = text.count("retrying with explicit nudge")
    diversity_retries = text.count("retrying with diversity nudge")
    sanitiser = text.count("Celebrity-likeness")
    return (
        f"### {label} log signals\n\n"
        f"- unknown-concept drift: {unknown}\n"
        f"- tier-required retries: {retries}\n"
        f"- diversity-nudge retries: {diversity_retries}\n"
        f"- celebrity-name sanitiser hits: {sanitiser}\n\n"
    )


def _wall_clock(log_path: Path) -> str:
    if not log_path.exists():
        return "(log missing)"
    text = log_path.read_text()
    # Parse `time` builtin output: "real Xm Ys"
    import re
    m = re.search(r"^real\s+(\d+)m([\d.]+)s", text, re.MULTILINE)
    if m:
        return f"{int(m.group(1))}m{m.group(2)}s"
    # Fallback: parse first vs last timestamp
    ts = re.findall(r"^(\d\d:\d\d:\d\d)", text, re.MULTILINE)
    if len(ts) < 2:
        return "(no timestamps)"
    return f"{ts[0]} → {ts[-1]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--baseline-log", required=True)
    ap.add_argument("--candidate-log", required=True)
    ap.add_argument("--db", default="nsfw_pipeline.db")
    args = ap.parse_args()

    db = sqlite3.connect(args.db)
    base = _fetch_series_meta(db, args.baseline)
    cand = _fetch_series_meta(db, args.candidate)
    if not base or not cand:
        print(f"ERROR: series row missing — baseline={base or 'NULL'} "
              f"candidate={cand or 'NULL'}", file=sys.stderr)
        return 1

    bf = _fetch_facets(db, args.baseline)
    cf = _fetch_facets(db, args.candidate)
    bp = _fetch_prompts(db, args.baseline)
    cp = _fetch_prompts(db, args.candidate)

    print("# MLX Cydonia vs Ollama Cydonia-heretic — Audit")
    print()
    print(f"- Baseline: `{base['id']}` · theme=`{base['theme']}` · "
          f"level={base['content_level']} · scenes={base['actual_count']}/{base['target_count']}")
    print(f"- Candidate: `{cand['id']}` · theme=`{cand['theme']}` · "
          f"level={cand['content_level']} · scenes={cand['actual_count']}/{cand['target_count']}")
    print()
    print(f"- Baseline wall-clock: {_wall_clock(Path(args.baseline_log))}")
    print(f"- Candidate wall-clock: {_wall_clock(Path(args.candidate_log))}")
    print()
    print("## Facet rows")
    print(f"- baseline: {len(bf)}")
    print(f"- candidate: {len(cf)}")
    print()
    print("## Prompt rows")
    print(f"- baseline: {len(bp)}")
    print(f"- candidate: {len(cp)}")
    print()
    print("## Diversity per axis")
    print()
    print(_diversity_block("Baseline (Ollama cydonia-heretic)", bf))
    print(_diversity_block("Candidate (MLX cydonia v3.1 4bit)", cf))
    print(_prose_block("Baseline", bf))
    print(_prose_block("Candidate", cf))
    print(_tier_block("Baseline", bf, base["content_level"]))
    print(_tier_block("Candidate", cf, cand["content_level"]))
    print(_log_metrics("Baseline (Ollama)", Path(args.baseline_log)))
    print(_log_metrics("Candidate (MLX)", Path(args.candidate_log)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
