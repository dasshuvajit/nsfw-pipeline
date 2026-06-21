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
    python scripts/audit_prompts.py output/art_series/<ts>     # manifest.json
    python scripts/audit_prompts.py <dir-or-file> --verbose    # prompts.json works too

(The old <series_id> DB mode was archived 2026-06-10 with the structured
path — the LLM-direct store is per-run manifest.json/prompts.json.)
The script is read-only.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# Repo root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Issue detectors ───────────────────────────────────────────────────


_PHOTOGRAPHER_REFS = (
    "Helmut Newton", "Peter Lindbergh", "Herb Ritts", "Robert Mapplethorpe",
    "Petter Hegre", "Slim Aarons", "Annie Leibovitz", "Mario Testino",
    "Richard Avedon", "Sarah Moon", "Paolo Roversi", "Bill Henson",
    "Petra Collins", "Gregory Crewdson", "Nan Goldin", "Irving Penn",
)


# Split into primary SOURCES (mutually exclusive — two competing recipes is a
# real collision) and MODIFIERS (rim/fill/flare/dappled legitimately layer on
# one source; the system prompt itself teaches "a soft rim separating her from
# the background"). Penalizing modifiers alongside sources false-flagged the
# engine's best layered-light prose.
_LIGHTING_SOURCES = (
    "Rembrandt lighting", "golden hour", "butterfly lighting",
    "noir lighting", "split lighting", "overcast",
    "Caravaggio chiaroscuro", "Baroque Caravaggio", "fireplace",
    "single hard key", "window light", "neon", "Cinestill",
)
_LIGHTING_MODIFIERS = (
    "low-key", "rim light", "soft fill", "volumetric golden",
    "crepuscular", "dappled", "lens flare",
)
# Back-compat alias (callers/tests may import the old name).
_LIGHTING_RECIPES = _LIGHTING_SOURCES + _LIGHTING_MODIFIERS


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

# Tier contracts (2026-06 audit): the gate was tier-blind below T4 — a T3
# "art-nude" face close-up with zero nudity scored 10.0 and shipped inside a
# nude set, and a T4 with mere "bare shoulders" passed on the weak generic
# list above. Strong per-tier token contracts close both holes.
_T3_NUDITY_TOKENS = (
    "nude", "naked", "topless", "bare breast", "bare chest", "bare body",
    "undressed", "unclothed", "fully bare", "wears nothing",
)
_T4_EXPLICIT_TOKENS = (
    "vulva", "labia", "her sex", "mound", "her bare sex",
    "between her parted thighs", "between her thighs",
)
_LOW_TIER_EXPLICIT_TOKENS = (        # never in T1/T2 prose
    "vulva", "labia", "her sex", "nipple", "areola",
)
# Overt-nudity ceiling for BOTH low tiers (2026-06-10): a pre-gate-v2 T2
# prompt rendered fully nude and shipped in a PUBLIC package — T2 means
# IMPLIED (strategic cover), so overt nudity tokens are a violation at T1
# and T2 alike, not just T1.
_LOW_TIER_NUDITY_TOKENS = ("topless", "fully nude", "entirely nude",
                           "completely nude", "bare breast", "naked",
                           "fully bare", "in the nude")
_T1_NUDITY_TOKENS = _LOW_TIER_NUDITY_TOKENS    # back-compat alias

# Cliché-density ladder (2026-06 audit): house words measured saturating the
# corpus — "luminous" 83% of prompts, camera-tail formula 93%, "velvety" 35%.
# A few are fine; a pile is the house style fossilizing. ≥3 distinct hits
# starts costing.
_CLICHE_PHRASES = (
    "luminous", "velvety", "sultry", "tack-sharp", "exquisite",
    "breathtaking", "strikingly beautiful", "her expression is one of",
    "the fine texture of her skin", "toward the lens with a",
    "melts into", "dissolves into", "creamy bokeh", "unretouched",
    "alabaster", "porcelain skin",
)
_CLICHE_REGEXES = (re.compile(r"\bshot on \d+\s?mm", re.IGNORECASE),)

# SDXL/MidJourney "quality booster" + camera-tag soup. MODEL-WRONG for our
# T5/Qwen-prompted cfg-1 engines (Chroma/Z-Image): the art_director system prompt
# bans them, but they were INVISIBLE to this score-side gate, so a pasted SDXL
# prompt shipped at 9.5 (2026-06-18 R&D). Penalized here too. NOTE: a lens woven
# into PROSE ("a short telephoto at f/2 keeps her crisp") is house craft and is NOT
# a booster — only the tag-soup forms below are: bare resolution/quality tags, the
# "shot on …" formula, the "<NN>mm f/<N>" lens-spec adjacency, and named camera
# bodies. All matched on word boundaries (so "18k gold"/"24k" never trip "8k"/"4k").
# DELIBERATELY NOT penalized (regression-checked vs 143 shipped prompts): a lens woven
# into prose, even with a numeric focal length ("an 85mm f/1.4 lens melts the
# background"), and figurative "a masterpiece of …" — both are tolerated house voice.
_BOOSTER_TOKENS = (
    "best quality", "8k", "4k", "uhd", "raw photo",
    "ultra-detailed", "ultra detailed", "ultradetailed", "hyperdetailed",
    "hyper-detailed", "highly detailed", "photorealistic", "hyperrealistic",
    "hyper-realistic", "ultrarealistic", "ultra-realistic", "ultra realistic",
    "trending on artstation", "octane render", "unreal engine",
)
# "masterpiece" is INTENTIONALLY absent: it's common figurative house voice ("a
# jazz-age masterpiece", "a masterpiece of shadow and gold"), and in a real SDXL
# paste it always rides with best-quality/8k/raw-photo/shot-on (caught below), so
# detecting it adds only false positives with zero gain.
_BOOSTER_REGEXES = (
    re.compile(r"\bshot on\b", re.IGNORECASE),                       # "shot on Sony A7R V" / "shot on 85mm"
    re.compile(r"\b(sony\s+a7|canon\s+eos|nikon\s+z|hasselblad|fujifilm|leica\s+m)\b",
               re.IGNORECASE),                                       # named camera bodies
)

# Specificity ladder (excellence-presence, not defect-absence): the audit found
# 10.0 meant only "no defects". These reward the concrete optical/material
# precision the house doctrine demands — absence now costs.
_LIGHT_DIRECTION_TOKENS = (
    "camera-left", "camera-right", "camera left", "camera right",
    "from behind", "from above", "from below", "from the left",
    "from the right", "raking", "backlit", "back-lit", "side-lit",
    "sidelight", "side light", "rim-lit", "overhead", "low-angled",
    "from a high window", "from one side", "obliquely", "low and from",
)
_MATERIAL_NOUNS = (
    "velvet", "marble", "brass", "linen", "silk", "oak", "stone",
    "leather", "lace", "copper", "iron", "mahogany", "satin", "wool",
    "terracotta", "ceramic", "pearl", "chrome", "latex", "cotton",
    "fur", "gilt", "gilded", "bronze", "obsidian", "damask", "tulle",
    "chiffon", "rattan", "wicker", "burlap", "denim", "glass",
    # Scene/architectural materials (2026-06-17): broadened so a prompt can vary
    # its materials BY SCENE rather than cramming the same luxe trio (brass/
    # satin/stone) into every image to hit a quota — the cause of catalog
    # sameness. Paired with a lowered threshold (>=2) below.
    "stucco", "travertine", "teak", "concrete", "canvas", "plaster",
    "granite", "slate", "bamboo", "jute", "sisal", "terrazzo", "suede",
    "steel", "driftwood", "raffia", "cashmere", "organza", "tweed", "sand",
    # Surface-ornament nouns (2026-06-18): stitched/metallic-thread dress detail
    # ("navy lace with gold-thread embroidery") rendered fine but earned no
    # specificity credit. These let couture surface-work count toward the ladder.
    "embroider", "goldwork", "beadwork", "sequin", "filigree", "brocade",
    "applique",
)
_MICRO_TEXTURE_TOKENS = (
    "pores", "vellus", "freckle", "mole", "catchlight", "sheen",
    "grain", "goosebumps", "flush", "dew", "down at her", "fine down",
    "peach fuzz", "translucency",
)


def _ngram_set(text: str, n: int = 3) -> set[str]:
    """Word n-grams for freshness/overlap comparison."""
    words = re.findall(r"[a-z']+", (text or "").lower())
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def freshness_overlap(text: str, context_prompts: "list[str] | None") -> float:
    """Max 3-gram CONTAINMENT of ``text`` vs each context prompt — how much of
    the candidate's phrasing already exists in this run/recent history.
    0.0 when no context."""
    if not context_prompts:
        return 0.0
    cand = _ngram_set(text)
    if not cand:
        return 0.0
    worst = 0.0
    for ref in context_prompts:
        ref_set = _ngram_set(ref)
        if not ref_set:
            continue
        worst = max(worst, len(cand & ref_set) / min(len(cand), len(ref_set)))
    return worst


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


# ── Canonical shared rule lists (single source of truth, 2026-06-10) ──
# Both enforcement layers consume THESE: art_director's hard Pydantic gate
# imports the strict subsets below; this module's soft scoring uses the
# broader _SAD_TOKENS underneath. The lists had drifted across 3 homes —
# the system prompt banned 'wistful' but neither enforcement list had it.
HARD_SAD_EXACT = frozenset({
    "sad", "sadness", "mournful", "melancholic", "melancholy", "sorrow",
    "sorrowful", "crying", "tearful", "forlorn", "woeful", "grief", "doleful",
    "wistful",
})
HARD_SAD_PREFIX = ("griev", "weep", "despair", "anguish", "mourn")

# Refusal-shaped output (2026-06-11 blind panel catch): Skyfall emitted
# "I cannot generate this image as requested..." — right length, no banned
# tokens — and it SHIPPED via keep-best at 4.0. A refusal is never a prompt.
REFUSAL_TOKENS = (
    "i cannot", "i can't", "i am unable", "i'm unable", "i won't",
    "as an ai", "cannot generate", "unable to generate", "cannot create",
    "i must decline", "i apologize", "i'm sorry, but", "i am sorry, but",
)

# 2026-05-23 (verifier audit) — sad/crying tokens flagged. User
# explicit ban: commercial adult-art markets sell confidence +
# sensuality, NOT sorrow.
_SAD_TOKENS = (
    "tear", "tears", "tearful", "tear-streaked", "teardrop", "teardrops",
    "crying", "sobbing", "mournful",  # 'weeping' handled context-aware by the
    #                                   hard gate (exempts weeping willow/stone)
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
    """Count distinct PRIMARY lighting sources in the prompt. >1 = a real
    collision (two competing recipes). Modifiers (rim/fill/flare/dappled)
    legitimately layer on one source and are NOT counted."""
    found = []
    text_lower = text.lower()
    for recipe in _LIGHTING_SOURCES:
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


def score_prompt(
    text: str, tier: str, context_prompts: "list[str] | None" = None,
) -> tuple[float, list[str]]:
    """Score 1-10 + issue list. 10.0 = zero defects AND fully specific.

    Gate v2 (2026-06 audit): the old rubric was all floor-checks with a +1
    coherence bonus that masked up to 1.0 of penalties — 29/36 production
    prompts scored exactly 10.0 and the regenerate branch never fired. The
    bonus is gone; three quality LADDERS spread the top end (cliché density,
    freshness vs ``context_prompts``, specificity-absence), and per-tier
    token CONTRACTS enforce T1-T4 explicitness both ways.

    Defect penalties:
    - Style stacking (>1 photographer): -2
    - Multiple lighting SOURCES (>1; modifiers free): -2
    - Camera angle contradiction: -2
    - B&W + color contradiction: -1.5
    - Mirror dangling / prose, grounding, sad tokens: as before
    - Tag-soup sentences (>3): -1 · Repeated 4-grams: -0.5 each
    - SDXL/MJ boosters + camera-tag soup (masterpiece/8k/raw photo/"shot on
      …"/"<NN>mm f/<N>"): -2 each, cap -6 (a lens woven in prose is NOT a hit)
    Tier contracts:
    - T4 without a strong explicit token (vulva/labia/her sex/...): -4
    - T3 without any nudity token: -3
    - T1/T2 carrying explicit-anatomy tokens: -3 (T1 also bare/topless: -3)
    Quality ladders:
    - ≥3 distinct cliché phrases: -0.5 each beyond the 2nd (cap -2)
    - freshness overlap vs context >0.15: -1.0; >0.30: -2.0
    - no named light direction / <3 material nouns / <2 micro-texture
      tokens: -0.5 each
    """
    score = 10.0
    issues: list[str] = []

    # Refusal-shaped output — floor the score (keep-best must never ship it).
    _low_head = (text or "").lower()[:300]
    refusals = [t for t in REFUSAL_TOKENS if t in _low_head]
    if refusals:
        issues.append(f"REFUSAL: {refusals} — this is a refusal, not a prompt")
        return 0.0, issues

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

    # ── Tier contracts (both directions) ──────────────────────────────
    text_lower = text.lower()
    if tier == "T4_explicit":
        if not any(t in text_lower for t in _T4_EXPLICIT_TOKENS):
            # Strong contract: T4 means the bare sex is described — mere
            # nudity ("bare shoulders") passing the old weak list let a
            # nudity-free prompt ship inside a paid T4 set at 9.0.
            score -= 4.0
            issues.append("T4_NO_EXPLICIT: no explicit-anatomy token in T4 prompt")
        elif detect_t4_anatomy_missing(text):
            score -= 2.0
            issues.append("T4_NO_ANATOMY: no anatomy keyword in T4 prompt")
    elif tier == "T3_artnude":
        if not any(t in text_lower for t in _T3_NUDITY_TOKENS):
            score -= 3.0
            issues.append("T3_NO_NUDITY: no nudity token in an art-nude prompt")
    elif tier in ("T1_suggestive", "T2_implied"):
        explicit_hits = [t for t in _LOW_TIER_EXPLICIT_TOKENS if t in text_lower]
        if explicit_hits:
            score -= 3.0
            issues.append(f"LOW_TIER_EXPLICIT: {explicit_hits} in {tier}")
        nudity_hits = [t for t in _LOW_TIER_NUDITY_TOKENS if t in text_lower]
        if nudity_hits:
            # T1 = clothed/lingerie; T2 = IMPLIED nudity (covered). Overt
            # nudity is a tier violation at both — these sets post PUBLIC.
            score -= 3.0
            issues.append(f"LOW_TIER_NUDITY: {nudity_hits} in {tier}")

    # Tag-soup sentences
    n_soup = detect_tag_soup_sentences(text)
    if n_soup > 3:
        score -= 1.0
        issues.append(f"TAG_SOUP: {n_soup} sentences with >4 commas")

    # Repetition (within-prompt)
    repeats = detect_repeated_clauses(text)
    if repeats:
        score -= 0.5 * min(len(repeats), 4)
        issues.append(f"REPETITION: {len(repeats)} repeated 4-grams: {repeats[:3]}")

    # SDXL/MidJourney booster + camera-tag soup — model-wrong for our engines.
    # A hard defect (not a ladder): even one costs, the full pasted tail craters.
    booster_hits = [t for t in _BOOSTER_TOKENS
                    if re.search(r"\b" + re.escape(t) + r"\b", text_lower)]
    booster_hits += [rx.pattern for rx in _BOOSTER_REGEXES if rx.search(text)]
    if booster_hits:
        score -= min(6.0, 2.0 * len(booster_hits))
        issues.append(f"SDXL_BOOSTERS: {len(booster_hits)} model-wrong tags: "
                      f"{booster_hits[:5]}")

    # ── Quality ladders (excellence presence) ─────────────────────────
    # Cliché density: a few house words are voice; a pile is fossilization.
    # Word-boundary match so "luminous" no longer phantom-fires on "voluminous".
    cliche_hits = [p for p in _CLICHE_PHRASES
                   if re.search(r"\b" + re.escape(p) + r"\b", text_lower)]
    cliche_hits += [rx.pattern for rx in _CLICHE_REGEXES if rx.search(text)]
    if len(cliche_hits) >= 3:
        pen = min(2.0, 0.5 * (len(cliche_hits) - 2))
        score -= pen
        issues.append(f"CLICHE_DENSITY: {len(cliche_hits)} house phrases: "
                      f"{cliche_hits[:4]}")

    # Freshness vs the run/recent history (when the caller supplies context).
    overlap = freshness_overlap(text, context_prompts)
    if overlap > 0.30:
        score -= 2.0
        issues.append(f"STALE_PHRASING: {overlap:.0%} 3-gram overlap with prior prompts")
    elif overlap > 0.15:
        score -= 1.0
        issues.append(f"FAMILIAR_PHRASING: {overlap:.0%} 3-gram overlap with prior prompts")

    # Specificity: named light direction, concrete materials, micro-texture.
    if not any(t in text_lower for t in _LIGHT_DIRECTION_TOKENS):
        score -= 0.5
        issues.append("NO_LIGHT_DIRECTION: light has no named direction")
    # Lowered 2026-06-17 (was >=3 materials / >=2 micro). The old quotas forced
    # the LLM to stuff the same whitelisted nouns into every prompt → catalog
    # sameness + a noun-pile reading. >=2 materials / >=1 micro still rewards
    # concreteness over vague AI mush without dictating WHICH details.
    n_materials = sum(1 for m in _MATERIAL_NOUNS if m in text_lower)
    if n_materials < 2:
        score -= 0.5
        issues.append(f"THIN_MATERIALS: only {n_materials} concrete material nouns")
    n_micro = sum(1 for m in _MICRO_TEXTURE_TOKENS if m in text_lower)
    if n_micro < 1:
        score -= 0.5
        issues.append(f"THIN_MICRO_TEXTURE: no micro-texture token")

    # Trailing period
    if not text.rstrip().endswith((".", "!", "?")):
        score -= 0.5
        issues.append("NO_TRAILING_PERIOD")

    return max(0.0, min(10.0, score)), issues



# ── Main ──────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score a run's prompts (gate-v2 rubric) from its "
                    "manifest.json / prompts.json.")
    parser.add_argument("run_dir",
                        help="output/art_series/<ts> dir, or a direct path to "
                             "manifest.json / prompts.json")
    parser.add_argument("--verbose", action="store_true",
                        help="print full prompt text per entry")
    args = parser.parse_args()

    p = Path(args.run_dir).expanduser()
    f = p
    if p.is_dir():
        f = (p / "manifest.json") if (p / "manifest.json").exists() \
            else (p / "prompts.json")
    if not f.exists():
        print(f"No manifest.json / prompts.json under {p}", file=sys.stderr)
        return 1
    data = json.loads(f.read_text())
    tier = data.get("tier", "T3_artnude")
    rows = (data.get("prompts") or []) + (data.get("covers") or [])
    texts = [(r.get("prompt") or "") for r in rows]

    print("=" * 72)
    print(f"AUDIT — {f}")
    print(f"  tier: {tier}   prompts (incl. covers): {len(rows)}")
    print("=" * 72 + "\n")

    scores: list[float] = []
    for i, txt in enumerate(texts):
        if not txt:
            continue
        # context = the prompts accepted BEFORE this one (freshness ladder)
        score, issues = score_prompt(txt, tier, context_prompts=texts[:i])
        scores.append(score)
        rating = "✓" if score >= 8.5 else ("△" if score >= 7 else "✗")
        print(f"[{rating}] #{i + 1:02d} — score {score:.1f}/10 "
              f"({count_sentences(txt)}s, {word_count(txt)}w)")
        for issue in issues:
            print(f"    {issue}")
        if args.verbose:
            print(f"    PROMPT: {txt[:240]}…")
        print()

    if not scores:
        print("No prompts found.")
        return 1
    print("=" * 72)
    avg = sum(scores) / len(scores)
    print(f"  Average score: {avg:.2f}/10   "
          f"≥8.5: {sum(1 for s in scores if s >= 8.5)}/{len(scores)}   "
          f"<7: {sum(1 for s in scores if s < 7)}/{len(scores)}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
