# Prompt-Variety Overhaul — Session Report (2026-06-20)

Goal you set: *"I'm getting kind of similar images across same niches… cover variety
of everything — lighting, position, face, age, types of women, places, backgrounds,
garments, cultures."* You authorised changing anything, enlarging everything, and
running the whole plan autonomously.

This report covers what shipped, why, and how it was validated (all GPU-free prompt
checks first, then real renders).

---

## TL;DR

The sameness had **three** root causes, not one. All three are fixed and committed:

1. **A structural bug** — every per-image variety axis shared ONE rotation index, so
   the whole ~360k-combination space collapsed to a 1-D line of length 6. Re-runs of a
   niche were near-identical. → **decoupled** (commit `9a69e93`).
2. **Pools too small + too few axes.** → enlarged 2-3× and added wardrobe / pose /
   time-weather axes (same commit).
3. **Within-niche scene redundancy** — most niches had 6 sub_looks, several
   near-duplicates (modern_boudoir = 3 window-light bedrooms; aspirational_luxe =
   all-Mediterranean), so the LLM kept colliding and over-writing. → **doubled every
   niche to 12 sub_looks with deliberately varied lighting + 6 brand-new niches**
   (commit `96578a4`).

Measured result (prompts-only, no GPU): distinct-3gram coverage **0.96**, mean
cross-pair similarity **0.008** (≈ no two prompts alike), opener variety **1.0**,
audits **9.3-9.4** on the new niches, all in the 120-180 word band, 358 tests green.

---

## Phase A+B — the engine (commit `9a69e93`)

**The bug.** `art_director.generate_series` rotated every axis with the same
`(i + run_offset) % len` index — sub_look, framing, opener, craft, concept,
composition, camera all moved in lockstep. Two niches' worth of "variety" axes never
actually crossed; a re-run at the same offset reproduced the same line.

**The fix — `_rotate(seq, i, run_key, axis)`**: per-(run, axis) deterministic shuffle
+ a coprime stride per axis. Each axis now walks its own independent path; re-runs
differ; the combination space is actually explored. (Generalises the mechanism the
look-pools already used.)

**Enlarged pools** (`creative_direction.yaml`): hair 10→28, figure 7→20, face 8→26,
complexion 6→18, age_look 4→7 — every age entry adult-anchored (mid-20s … approaching
forty; "college-age **adult**"). Stride coprimality fixed (figure 5→9).

**Three new per-image axes** (`art_director.py`):
- `GARMENT_TYPES` (28 wardrobe types; tier governs exposure; OFF at T4),
- `POSE_GESTURES` (19 solo poses),
- `ATMOSPHERE` (10 time × weather overlays).

**Bolder + in-band.** Added a "take the bolder swing" directive, and — because the
richer multi-axis prompts initially bloated to ~210 words — a firm **130-175 word cap**
(last-weighted position) framing the axes as *ingredients to select*, not a checklist.
Prompts dropped back to 154-174 words with **higher** diversity.

`scripts/diversity_report.py` (new) is the GPU-free A/B harness used throughout.

## Phase C — the content (commit `96578a4`)

**Every existing niche doubled 6→12 sub_looks**, each new set written for DISTINCT
scenes and DELIBERATELY VARIED lighting (cold pre-dawn, hard noon, overcast, blue hour,
candle/firelight, single shaft, dappled, storm-grey, neon) instead of defaulting to
golden hour — the actual fix for "same niche looks the same".

**Six new niches** opening new visual territory:

| Niche | Lane | Engine | Tiers |
|---|---|---|---|
| `athletic_studio` | gym / yoga / pilates / climbing / boxing / lap-pool | zimage | T1-T3 |
| `wild_nature` | waterfall / forest / snow / alpine / desert / rainforest | zimage | T1-T4 |
| `surreal_dreamscape` | dreamlike, painterly | chroma | T2-T4 |
| `south_asian_editorial` | sari / lehenga heritage-fashion editorial | zimage | T1-T3 |
| `iberian_flamenco` | Andalusian / flamenco glamour | zimage | T1-T3 |
| `slavic_folk` | Russian / Slavic folk-romantic | zimage | T1-T3 |

**Cultural niches are handled with care**: celebratory heritage *fashion* editorial,
tasteful (T1-T3, no explicit T4), never sacred / ritual / caricature. A dedicated
safety gate caught and fixed a "temple-border" reference (→ "contrast zari border")
and verified no sacred terms. Cultural garments live INSIDE these niches (where they
belong) rather than the portable garment axis (which would drop a sari into a
cyberpunk scene).

Every new line passed a safety gate (age = adult, single female, no clichés/boosters,
lighting variety) **and** the catalog-wide invariants (no mirror motif / face-doubling,
no pre-undress verbs) before integration.

---

## Validation (prompts-only, no GPU)

| Run | Mean audit | Word band | Diversity (coverage / pair-sim / opener) |
|---|---|---|---|
| bohemian_naturallight T2 (engine check) | 8.58 | 154-163 ✓ | 0.972 / 0.007 / 1.0 |
| modern_boudoir T2 (was 8.17, worst STALE) | **8.5** | 157-170 ✓ | 0.965 / 0.008 / 1.0 |
| athletic_studio T2 (NEW) | **9.42** | 159-174 ✓ | 0.957 / 0.009 / 1.0 |
| south_asian_editorial T2 (NEW cultural) | **9.33** | 150-172 ✓ | 0.964 / 0.009 / 1.0 |

The tier contract held on the new niches: a T2 "bare breast" attempt and a "shot on"
booster were both caught and regenerated clean automatically.

## Production renders

A showcase batch was rendered (zimage default; chroma for the painterly niche),
each series running the full path: LLM prompts + SFW covers → unload → staged render
(base + SDXL detailers) → curation → tier-split packaging + posting templates.

| Series | Tier | Engine | Result |
|---|---|---|---|
| athletic_studio | T2 | zimage | 6/6 keepers, public=6, cover + posting_templates; tier-truth advisory flagged 3 skin-forward frames for manual QA (the gate working). Frames show real variety — different woman/setting/light/wardrobe/pose per image. |
| wild_nature | T3 | zimage | 6/6 keepers, **gated=6 (clean art-nude) + public=1 (SFW watermarked cover)** — tier-truth verified by eye: gated frames are tasteful figure studies, the public cover is fully clothed. |
| south_asian_editorial | T2 | zimage | 6/6 keepers, public=6, cover; respectful heritage fashion (gold sari, warm diya ambiance) — verified by eye, tasteful + covered. 2 skin-forward frames flagged. |
| modern_boudoir | T2 | zimage | 6/6 keepers, public=6; the fixed-redundant niche renders cleanly. 1 skin-forward frame flagged. |
| surreal_dreamscape | T3 | chroma | 6/6 keepers, gated=6 + public=2 (SFW covers). Chroma path verified — genuinely dreamlike (bioluminescent teal grotto), tasteful art-nude, anatomically clean. |

All five series finished exit 0 (batch 06:53→08:49). Visual QA (sampled across
series): single adult woman, anatomically clean (hands correct; the hand-YOLO guard
auto-rerolled one extra-limb base frame), on-prompt, correct tier-truth — T3 art-nude
to `gated/` (clean) + SFW cover to `public/` (watermarked); T2 to `public/`. Packages
are under `output/art_series/<ts>/package/<Folder>/` with `public/`, `gated/`,
`posting_templates/`, `POSTING_CHECKLIST.md`. Upload is manual (VISUAL tier-truth QA
of `public/` before posting remains the #1 checklist rule).

**One thing for your eye:** the south_asian `festival-diya` frame surrounds a
sari-clad woman with lit diya lamps. It reads as festive *ambiance* (no deity, no
ritual act, woman clothed) and passes the cultural gate, but diya imagery carries
Diwali/festival association — give it your call before posting, and say the word if
you'd rather I soften that one sub_look to a neutral "oil-lamp courtyard".

---

## Notes / decisions you may want to revisit

- **Prose camera language stays.** "an 85mm at f/1.8 on Portra 400" reads in the
  output and is *intentional* house style (natural-language shallow-DOF / film-warmth
  that T5/Qwen parse semantically) — distinct from the banned SDXL tag-soup
  ("shot on Sony A7R V", "masterpiece, 8k"), which is still gated.
- **Engine routing is per-run** (`--engine`), not stored on the niche. Painterly /
  period / B&W niches (surreal_dreamscape, monochrome, the period/fantasy set) want
  `--engine chroma`; modern photoreal niches use the default zimage.
- **Residual STALE** shows up only on a few niches whose remaining scenes still rhyme
  (modern_boudoir still has multiple window-light bedrooms). Could trim/replace those
  specific sub_looks in a later pass; the mean is already back in range.
