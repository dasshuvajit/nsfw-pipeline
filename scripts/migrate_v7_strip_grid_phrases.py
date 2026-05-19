"""One-shot migration: clean cross-scene variety phrases from existing
series plans + composed prompts after the vocab v7 anti-grid fix.

Context: prior to vocab v7 (2026-05-20), Cydonia could emit phrases
like "in natural poses across varying compositions" into
``series.llm_series_plan.subject_description`` (theme mode) or
``subject_bias`` / ``core_theme`` / ``visual_elements`` (niche mode),
which then got injected verbatim into every scene's composed prompt
and caused SDXL/Chroma to render 4-panel image-grid hallucinations.

Vocab v7 added (a) the SINGLE-SCENE INVARIANT clause to the planner
system prompts, (b) tightened user-prompt instructions forbidding
variety language, (c) a server-side sanitizer in
``ThemeMode._plan`` / ``NicheMode._plan`` that strips offending
phrases via ``_MULTI_SUBJECT_PATTERN``. New plans land clean.

This script applies the same sanitization to ALREADY-STORED rows:

  - ``series.llm_series_plan`` — subject_description / subject_bias /
    core_theme / visual_elements
  - ``prompts.prompt_text`` — for prompts composed from the bad plan
    that still carry the literal "varying compositions" phrase

Mirror vocab in stored ``scene_facets`` (NSFW_T4_SOLO_MIRROR,
COMP_REFLECTION_PRIMARY, etc.) is NOT touched — the canonicalizer
already silently skips deleted concept tags (verified in F3 of the
verifier audit), so existing rows render with those slots simply
empty rather than corrupt. If the user wants those scenes re-rolled
with NEW tag picks they should run ``--regen-facets`` from the CLI.

Usage::

    python scripts/migrate_v7_strip_grid_phrases.py [--dry-run] [--series <id>]

``--dry-run`` reports counts without writing.
``--series <id>`` limits the sweep to a single series (useful for
verifying scene_021's series ``series_2547fb306a7c`` specifically).
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

# Allow `python scripts/...` invocation from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.prompt.builder import _MULTI_SUBJECT_PATTERN  # noqa: E402

import re

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("migrate_v7")


_WHITESPACE_RUN = re.compile(r"\s{2,}")

# After stripping a multi-subject phrase, remove orphaned connector
# words left mid-sentence — e.g. "in natural poses across varying
# compositions" → first pass strips "varying compositions" → leaves
# "across " → this regex catches that dangling "across" + trailing
# punctuation/space. Bounded to in/across/throughout/with — common
# variety-phrase connectors only, to avoid touching legitimate prose.
_ORPHAN_CONNECTOR = re.compile(
    r"\s+(?:in|across|throughout|with)\s*(?=[.,]|$)",
    re.IGNORECASE,
)


def _sanitize(text: str) -> tuple[str, bool]:
    """Return (cleaned, changed). Strips grid/multi-subject phrases,
    removes orphaned connector words left behind, and collapses
    whitespace + dangling punctuation."""
    if not text:
        return text, False
    cleaned = _MULTI_SUBJECT_PATTERN.sub("", text)
    cleaned = _ORPHAN_CONNECTOR.sub("", cleaned)
    cleaned = _WHITESPACE_RUN.sub(" ", cleaned).strip(" ,.")
    return cleaned, (cleaned != text)


def migrate_series(conn: sqlite3.Connection, dry_run: bool, only_series: str | None) -> dict[str, int]:
    """Clean series.llm_series_plan rows."""
    stats = {"series_scanned": 0, "series_changed": 0, "fields_changed": 0}
    where = "WHERE id = ?" if only_series else ""
    args = (only_series,) if only_series else ()
    cur = conn.execute(
        f"SELECT id, llm_series_plan FROM series {where}", args,
    )
    for row in cur:
        stats["series_scanned"] += 1
        series_id = row[0]
        raw = row[1]
        if not raw:
            continue
        try:
            plan = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("series=%s: llm_series_plan is not JSON (%s) — skipping", series_id, exc)
            continue
        if not isinstance(plan, dict):
            continue

        row_changed = False
        for fld in ("subject_description", "subject_bias", "core_theme"):
            raw_val = plan.get(fld) or ""
            if not isinstance(raw_val, str):
                continue
            cleaned, changed = _sanitize(raw_val)
            if changed:
                logger.info(
                    "series=%s field=%s before=%r after=%r",
                    series_id, fld, raw_val, cleaned,
                )
                plan[fld] = cleaned
                stats["fields_changed"] += 1
                row_changed = True

        ve = plan.get("visual_elements")
        if isinstance(ve, list):
            new_ve = []
            ve_changed = False
            for i, v in enumerate(ve):
                if not isinstance(v, str):
                    new_ve.append(v)
                    continue
                cleaned, changed = _sanitize(v)
                if changed:
                    logger.info(
                        "series=%s visual_elements[%d] before=%r after=%r",
                        series_id, i, v, cleaned,
                    )
                    ve_changed = True
                if cleaned:
                    new_ve.append(cleaned)
            if ve_changed:
                plan["visual_elements"] = new_ve
                stats["fields_changed"] += 1
                row_changed = True

        if row_changed:
            stats["series_changed"] += 1
            if not dry_run:
                conn.execute(
                    "UPDATE series SET llm_series_plan = ? WHERE id = ?",
                    (json.dumps(plan), series_id),
                )
    return stats


def migrate_prompts(conn: sqlite3.Connection, dry_run: bool, only_series: str | None) -> dict[str, int]:
    """Clean prompts.prompt_text + prompts.negative_prompt rows. Even
    though the negative side now has HARD_BLOCK grid coverage, scrub
    any in-body grid phrases that survived from old composed prompts."""
    stats = {"prompts_scanned": 0, "prompts_changed": 0}
    where = "WHERE series_id = ?" if only_series else ""
    args = (only_series,) if only_series else ()
    cur = conn.execute(
        f"SELECT id, prompt_text FROM prompts {where}", args,
    )
    for row in cur:
        stats["prompts_scanned"] += 1
        prompt_id, prompt_text = row[0], row[1]
        cleaned, changed = _sanitize(prompt_text or "")
        if changed:
            logger.info(
                "prompt=%s changed (%d→%d chars)",
                prompt_id, len(prompt_text or ""), len(cleaned),
            )
            stats["prompts_changed"] += 1
            if not dry_run:
                conn.execute(
                    "UPDATE prompts SET prompt_text = ? WHERE id = ?",
                    (cleaned, prompt_id),
                )
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Strip vocab v7 grid/variety phrases from stored series + prompts.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--series", help="Only migrate this series id")
    parser.add_argument(
        "--db",
        default=str(Path(__file__).resolve().parent.parent / "nsfw_pipeline.db"),
        help="Path to the SQLite DB",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        logger.error("DB not found: %s", db_path)
        return 2

    conn = sqlite3.connect(str(db_path))
    try:
        ss = migrate_series(conn, args.dry_run, args.series)
        sp = migrate_prompts(conn, args.dry_run, args.series)
        if not args.dry_run:
            conn.commit()
    finally:
        conn.close()

    logger.info(
        "series: scanned=%d changed=%d (fields_changed=%d)",
        ss["series_scanned"], ss["series_changed"], ss["fields_changed"],
    )
    logger.info(
        "prompts: scanned=%d changed=%d",
        sp["prompts_scanned"], sp["prompts_changed"],
    )
    if args.dry_run:
        logger.info("DRY RUN — no rows were written. Re-run without --dry-run to apply.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
