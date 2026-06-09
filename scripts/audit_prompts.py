#!/usr/bin/env python3
"""Audit script — mechanical scoring of composed prompts for quality issues.

Given a series_id, reads every prompt for the chroma family and runs a
battery of regex-based checks for the issue classes Grok + Claude web
flagged on three separate prompts of series_79ae3b962c8d:

- Single-paragraph coherence (sentence count, canonicalized-fragment count)
- Style stacking (number of distinct photographer / school references)
- Lighting recipe count (multiple competing lighting recipes)
- Camera angle contradictions ("from above" + "low angle")
- B&W vs color contradiction (Lindbergh B&W + warm Caravaggio palette)
- Repetition (exact-match phrase frequency)
- Tag-soup heuristic (sentences with >4 commas)
- Anatomy keywords at T4
- Word count
- Mirror dangling syntax
- Trailing period

Score 1-10 per prompt, list issues. Print summary + per-prompt detail.

Usage:
    python scripts/audit_prompts.py <series_id> [--family chroma]

The script is read-only — it does not modify the DB or any files.
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

# Repo root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "nsfw_pipeline.db"


# ── Issue detectors ───────────────────────────────────────────────────


_PHOTOGRAPHER_REFS = (
    "Helmut Newton", "Peter Lindbergh", "Herb Ritts", "Robert Mapplethorpe",
    "Petter Hegre", "Slim Aarons", "Annie Leibovitz", "Mario Testino",
    "Richard Avedon", "Sarah Moon", "Paolo Roversi", "Bill Henson",
    "Petra Collins", "Gregory Crewdson", "Nan Goldin", "Irving Penn",
)


_LIGHTING_RECIPES = (
    "Rembrandt lighting", "golden hour", "butterfly lighting",
    "low-key", "noir lighting", "split lighting", "rim light",
    "soft fill", "overcast", "volumetric golden", "crepuscular",
    "dappled", "Cinestill", "lens flare", "Caravaggio chiaroscuro",
    "Baroque Caravaggio", "fireplace", "single hard key",
    "window light", "neon",
)


_CAMERA_ANGLES = (
    ("low angle", ("from above", "high angle", "looking down", "overhead")),
    ("high angle", ("low angle", "from below", "looking up", "worm's eye")),
)


_BW_TOKENS = ("B&W", "black and white", "silver gelatin", "monochrome")
_COLOR_TOKENS = (
    "warm honey", "Caravaggio palette", "umber blacks", "sepia",
    "golden hour", "warm flesh-tone", "deep crimson", "neon magenta",
    "kodachrome", "Portra 400",
)


_T4_ANATOMY_KEYWORDS = (
    "nude", "bare body", "bare breasts", "breasts", "nipples", "vulva",
    "labia", "bare chest", "fully nude", "complete nudity",
    "naked", "bare skin", "bare hips", "bare thighs",
)


_PROPS_FOR_BIGRAM_LOCK = (
    "persian rug", "champagne glass", "silk dress", "vinyl record",
    "fountain pen", "letter opener", "ostrich feather",
    "marble pedestal", "moss-covered", "ivy",
)


# Ungrammatical leftovers from the OLD vocab-canonicalizer's mirror-strip (since
# fixed by vocab v7 dropping the mirror entries). Only patterns that CANNOT occur
# in valid prose are kept. The normal-prose ones — `before her,`, `before,`,
# `in a hand,` — were removed: LLM-direct prompts use them legitimately ("the
# stones before her,", "a goblet in a hand,", "moments before,") and they
# false-flagged clean prompts (−2.0). Genuine mirrors are still caught by the
# `_PromptOut` 'mirror' reject + detect_mirror_prose.
_MIRROR_DANGLING_PATTERNS = (
    r"\bat in,",                                     # "looks at [mirror] in," → "at in,"
    r"\bholds the at\b",                             # "holds the [mirror] at" → "holds the at"
    r"\bgilded,\s+her\s+bare\s+body\s+reflected",    # a real mirror-reflection phrase
)


# 2026-05-23 (verifier audit) — sad/crying tokens flagged. User
# explicit ban: commercial adult-art markets sell confidence +
# sensuality, NOT sorrow.
_SAD_TOKENS = (
    "tear", "tears", "tearful", "tear-streaked", "teardrop", "teardrops",
    "crying", "weeping", "sobbing", "mournful",
    "grieving", "grief", "sorrow", "sorrowful",
    "wet eyes", "dried tears",
    "numb detachment", "vacant stare", "blank stare", "vacantly",
    "melancholic", "melancholy", "sad expression",
    "uncertainly", "uncertain gaze", "questioning gaze",
    "tentatively", "hesitantly", "lost expression",
    "anxious", "anxiety", "fearful",
    "lost in a memory", "lost in her own skin", "disconnected",
)


# Mirror references that produce warped-face Chroma artifacts.
_MIRROR_PROSE_PATTERNS = (
    r"\bher reflection\b",
    r"\bher own reflection\b",
    r"\breflecting her\b",
    r"\bdistorted form\b",
    r"\bwarped image\b",
    r"\bher own form reflected\b",
    r"\breflective surface\b",
    r"\breflecting [a-z]+ back\b",
    r"\bmirror[s]?\b",
    r"\bdistorted reflection\b",
    r"\bimage of herself\b",
)


# Fragmented prose detector — verifier found 4/10 scenes had per-axis
# short clauses ("Kneeling. Upper body. Over the shoulder. Soft side
# light.") instead of coherent paragraph prose. The dual-write
# contract requires narrative paragraph form.
def detect_fragmented_prose(text: str) -> bool:
    """True if prose appears fragmented (per-axis clauses, not
    narrative paragraph). Heuristic: >8 sentences AND average
    sentence length < 8 words = fragmented."""
    if not text:
        return False
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) <= 7:
        return False
    words_per_sentence = [len(s.split()) for s in sentences]
    avg = sum(words_per_sentence) / len(words_per_sentence)
    return avg < 8


def detect_sad_tokens(text: str) -> list[str]:
    """Return sad/crying tokens present in text."""
    if not text:
        return []
    text_lower = text.lower()
    return [t for t in _SAD_TOKENS if t in text_lower]


def detect_mirror_prose(text: str) -> list[str]:
    """Return mirror/reflection patterns matched in prose."""
    if not text:
        return []
    hits = []
    for pat in _MIRROR_PROSE_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            hits.append(pat)
    return hits


def count_sentences(text: str) -> int:
    """Approximate sentence count (split on . ! ? followed by space)."""
    if not text:
        return 0
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return sum(1 for s in sentences if s.strip())


def count_photographer_refs(text: str) -> tuple[int, list[str]]:
    """Count how many DISTINCT photographer schools appear in text."""
    found = []
    for ph in _PHOTOGRAPHER_REFS:
        if ph.lower() in text.lower():
            found.append(ph)
    return len(found), found


def count_lighting_recipes(text: str) -> tuple[int, list[str]]:
    """Count distinct lighting recipes in the prompt. >1 = style stacking."""
    found = []
    text_lower = text.lower()
    for recipe in _LIGHTING_RECIPES:
        if recipe.lower() in text_lower:
            found.append(recipe)
    return len(found), found


def detect_camera_angle_contradiction(text: str) -> list[str]:
    """Detect simultaneous incompatible camera angles."""
    text_lower = text.lower()
    contradictions = []
    for angle, opposites in _CAMERA_ANGLES:
        if angle.lower() in text_lower:
            for opp in opposites:
                if opp.lower() in text_lower:
                    contradictions.append(f"{angle!r} + {opp!r}")
    return contradictions


def detect_bw_vs_color(text: str) -> tuple[bool, list[str], list[str]]:
    """Returns (has_conflict, bw_tokens_found, color_tokens_found)."""
    text_lower = text.lower()
    bw_found = [t for t in _BW_TOKENS if t.lower() in text_lower]
    color_found = [t for t in _COLOR_TOKENS if t.lower() in text_lower]
    return bool(bw_found and color_found), bw_found, color_found


def detect_tag_soup_sentences(text: str) -> int:
    """Count sentences with >4 commas (tag-soup heuristic)."""
    if not text:
        return 0
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return sum(1 for s in sentences if s.count(",") > 4)


def detect_t4_anatomy_missing(text: str) -> bool:
    """True if no anatomy keyword in T4 prompt."""
    text_lower = text.lower()
    return not any(kw in text_lower for kw in _T4_ANATOMY_KEYWORDS)


def detect_mirror_dangling(text: str) -> list[str]:
    """Detect dangling-syntax artifacts from mirror strip."""
    hits = []
    for pat in _MIRROR_DANGLING_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            hits.append(pat)
    return hits


# Implausible grounding — the subject reads as ON water / in mid-air / submerged
# (a body hovering or sinking with no solid support; the "sitting on water"
# failure). Mirrors the art_director hard-reject (this is the softer audit-gate
# signal). A pose verb must precede on/atop/upon + a water body / "the water's
# surface", so "kneels AT the water's edge" or "light ON the water" don't trip.
_GROUNDING_WATER = r"(?:water|lake|river|pond|sea|ocean|pool)"
_IMPLAUSIBLE_GROUNDING_PATTERNS = (
    rf"\b(?:sit|sitt|sat|kneel|knelt|kneeling|lie|lying|lay|reclin|perch)\w*\b"
    rf"(?:\W+\w+){{0,2}}?\W+(?:on|atop|upon)\W+(?:the\s+|her\s+)?(?:\w+\s+){{0,2}}"
    rf"(?:{_GROUNDING_WATER}'?s\s+surface|surface\s+of\s+the\s+{_GROUNDING_WATER}|{_GROUNDING_WATER})\b",
    rf"\b(?:float|hover)\w*\b(?:\W+\w+){{0,2}}?\W+(?:on|above|over|upon)\W+(?:the\s+)?"
    rf"(?:\w+\s+){{0,2}}(?:{_GROUNDING_WATER}|air)\b",
    r"\bmid[\s-]?air\b",
    r"\bsuspended\s+in\s+(?:the\s+)?air\b",
    rf"\b(?:sit|sitt|sat|kneel|knelt|kneeling|lie|lying|lay|reclin)\w*\b"
    rf"(?:\W+\w+){{0,3}}?\W+(?:partially\s+|half\s+)?submerged\b",
)


def detect_implausible_grounding(text: str) -> list[str]:
    """Flag the subject reading as on water / mid-air / submerged."""
    hits = []
    for pat in _IMPLAUSIBLE_GROUNDING_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            hits.append(pat)
    return hits


def detect_repeated_clauses(text: str, min_repeats: int = 2) -> list[str]:
    """Find 4+ word phrases that repeat in the SAME prompt."""
    if not text:
        return []
    # Build 4-word shingles, count, return repeats.
    words = re.findall(r"\b\w+\b", text.lower())
    shingles = [
        " ".join(words[i:i+4]) for i in range(len(words) - 3)
    ]
    counts = Counter(shingles)
    return [s for s, c in counts.items() if c >= min_repeats]


def word_count(text: str) -> int:
    return len(text.split()) if text else 0


# ── Scoring ───────────────────────────────────────────────────────────


def score_prompt(text: str, tier: str) -> tuple[float, list[str]]:
    """Score 1-10 + issue list.

    Heavy penalties:
    - Style stacking (>1 photographer): -2
    - Multiple lighting recipes (>1): -2
    - Camera angle contradiction: -2
    - B&W + color contradiction: -1.5
    - Mirror dangling syntax: -2
    - T4 missing anatomy: -2 (T4 only)
    - Tag-soup sentences (>3): -1
    - Repeated 4-grams: -0.5 each

    Light bonuses:
    - Trailing period: +0 (just expected)
    - One coherent paragraph (<7 sentences): +1
    - Concrete anatomy at T4 (>3 anatomy keywords): +0.5
    """
    score = 10.0
    issues: list[str] = []

    # Style stacking
    n_photog, photogs = count_photographer_refs(text)
    if n_photog > 1:
        score -= 2.0
        issues.append(f"STYLE_STACKING: {n_photog} photographer refs: {photogs}")

    # Lighting recipes
    n_lighting, lightings = count_lighting_recipes(text)
    if n_lighting > 1:
        score -= 2.0
        issues.append(f"LIGHTING_COLLISION: {n_lighting} recipes: {lightings}")

    # Camera angle contradiction
    angle_conflicts = detect_camera_angle_contradiction(text)
    if angle_conflicts:
        score -= 2.0
        issues.append(f"CAMERA_ANGLE_CONFLICT: {angle_conflicts}")

    # B&W vs color
    bw_conflict, bw_t, color_t = detect_bw_vs_color(text)
    if bw_conflict:
        score -= 1.5
        issues.append(f"BW_COLOR_CONFLICT: bw={bw_t} color={color_t}")

    # Mirror dangling
    mirror_dang = detect_mirror_dangling(text)
    if mirror_dang:
        score -= 2.0
        issues.append(f"MIRROR_DANGLING: {mirror_dang}")

    # Implausible grounding — subject on water / mid-air / submerged
    grounding = detect_implausible_grounding(text)
    if grounding:
        score -= 2.0
        issues.append(f"IMPLAUSIBLE_GROUNDING: {grounding}")

    # 2026-05-23 verifier audit — sad tokens (user explicit ban)
    sad = detect_sad_tokens(text)
    if sad:
        score -= 2.5
        issues.append(f"SAD_TOKENS: {sad}")

    # Mirror references in prose (subject-mirror, not ambient)
    mirror_prose = detect_mirror_prose(text)
    if mirror_prose:
        score -= 1.5
        issues.append(f"MIRROR_PROSE: {mirror_prose}")

    # Fragmented prose (per-axis clauses)
    if detect_fragmented_prose(text):
        score -= 1.5
        issues.append("FRAGMENTED_PROSE: per-axis clauses not paragraph")

    # T4 anatomy
    if tier == "T4_explicit" and detect_t4_anatomy_missing(text):
        score -= 2.0
        issues.append("T4_NO_ANATOMY: no anatomy keyword in T4 prompt")

    # Tag-soup sentences
    n_soup = detect_tag_soup_sentences(text)
    if n_soup > 3:
        score -= 1.0
        issues.append(f"TAG_SOUP: {n_soup} sentences with >4 commas")

    # Repetition
    repeats = detect_repeated_clauses(text)
    if repeats:
        score -= 0.5 * min(len(repeats), 4)
        issues.append(f"REPETITION: {len(repeats)} repeated 4-grams: {repeats[:3]}")

    # Single-paragraph coherence bonus
    n_sentences = count_sentences(text)
    wc = word_count(text)
    if n_sentences < 7 and wc < 250:
        score += 1.0
        issues.append(f"GOOD_COHERENCE: {n_sentences} sentences, {wc} words")

    # T4 concrete anatomy bonus
    if tier == "T4_explicit":
        text_lower = text.lower()
        n_anatomy = sum(1 for kw in _T4_ANATOMY_KEYWORDS if kw in text_lower)
        if n_anatomy >= 3:
            score += 0.5
            issues.append(f"CONCRETE_ANATOMY: {n_anatomy} anatomy keywords")

    # Trailing period
    if not text.rstrip().endswith((".", "!", "?")):
        score -= 0.5
        issues.append("NO_TRAILING_PERIOD")

    return max(0.0, min(10.0, score)), issues


def find_repeated_props_across_series(prompts: list[tuple]) -> dict[str, int]:
    """Count which props/bigrams appear in multiple prompts.

    Accepts either 2-tuples ``(scene_id, text)`` or 3-tuples
    ``(scene_id, text, generation_kind)`` — only reads the first two
    elements so callers that include extra columns don't break.
    """
    counts: Counter[str] = Counter()
    for row in prompts:
        text = row[1]
        text_lower = (text or "").lower()
        for prop in _PROPS_FOR_BIGRAM_LOCK:
            if prop in text_lower:
                counts[prop] += 1
    return {prop: c for prop, c in counts.items() if c >= 3}


# ── Main ──────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit composed prompts for quality issues (Grok+Claude-web rubric)."
    )
    parser.add_argument("series_id", help="Series ID to audit")
    parser.add_argument("--model", default="gonzalomo_chroma_v30", help="Model ID")
    parser.add_argument(
        "--verbose", action="store_true", help="Print full prompt text per issue"
    )
    args = parser.parse_args()

    conn = sqlite3.connect(str(DB_PATH))
    try:
        # Get series tier
        row = conn.execute(
            "SELECT content_level, theme, style_profile_id FROM series WHERE id = ?",
            (args.series_id,),
        ).fetchone()
        if not row:
            print(f"Series {args.series_id!r} not found.")
            return 1
        tier, theme, style_profile = row

        # Get all prompts
        cur = conn.execute(
            "SELECT scene_id, prompt_text, generation_kind FROM prompts "
            "WHERE scene_id LIKE ? AND model_id = ? ORDER BY scene_id",
            (f"{args.series_id}%", args.model),
        )
        prompts = cur.fetchall()
    finally:
        conn.close()

    if not prompts:
        print(f"No prompts found for series {args.series_id!r} model {args.model!r}")
        return 1

    print("=" * 72)
    print(f"AUDIT — series {args.series_id}")
    print(f"  theme: {theme!r}")
    print(f"  style_profile: {style_profile!r}")
    print(f"  tier: {tier}")
    print(f"  prompts: {len(prompts)}")
    print("=" * 72)
    print()

    scores: list[float] = []
    fallback_count = 0
    llm_success_count = 0
    for scene_id, text, gen_kind in prompts:
        short_id = scene_id[-3:] if scene_id else "?"
        score, issues = score_prompt(text or "", tier)
        scores.append(score)
        rating = "✓" if score >= 7.5 else ("△" if score >= 5 else "✗")
        wc = word_count(text or "")
        ns = count_sentences(text or "")
        # 2026-05-24 — generation_kind tag shows whether the LLM
        # produced the body or the composer used the tier-aware fallback.
        if gen_kind == "llm_success":
            kind_tag = "LLM"
            llm_success_count += 1
        elif gen_kind and gen_kind.startswith("fallback_"):
            kind_tag = "FB"
            fallback_count += 1
        else:
            kind_tag = "?? "
        print(
            f"[{rating}] [{kind_tag}] scene_{short_id} — "
            f"score {score:.1f}/10 ({ns}s, {wc}w)"
        )
        for issue in issues:
            print(f"    {issue}")
        if args.verbose:
            print(f"    PROMPT: {(text or '')[:200]}...")
        print()

    # Series-level: cross-prompt prop repetition
    print("=" * 72)
    print("SERIES-LEVEL PROP LOCK-IN")
    print("-" * 72)
    repeated_props = find_repeated_props_across_series(prompts)
    if repeated_props:
        for prop, count in sorted(repeated_props.items(), key=lambda x: -x[1]):
            print(f"  {prop!r}: {count}/{len(prompts)} scenes")
    else:
        print("  (none — good diversity)")

    print()
    print("=" * 72)
    print("SUMMARY")
    print("-" * 72)
    avg = sum(scores) / len(scores)
    n_excellent = sum(1 for s in scores if s >= 8)
    n_acceptable = sum(1 for s in scores if 5 <= s < 8)
    n_poor = sum(1 for s in scores if s < 5)
    print(f"  Average score: {avg:.2f}/10")
    print(f"  Excellent (≥8): {n_excellent}/{len(scores)}")
    print(f"  Acceptable (5-8): {n_acceptable}/{len(scores)}")
    print(f"  Poor (<5): {n_poor}/{len(scores)}")
    print()
    n = llm_success_count + fallback_count
    if n > 0:
        print(f"  LLM-success:  {llm_success_count}/{n} ({100*llm_success_count/n:.0f}%)")
        print(f"  Fallback:     {fallback_count}/{n} ({100*fallback_count/n:.0f}%)")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
