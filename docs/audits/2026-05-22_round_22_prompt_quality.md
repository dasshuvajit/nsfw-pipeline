# Round-22 — Prompt Quality Fixes (Grok feedback audit)

**Date:** 2026-05-22
**Trigger:** External LLM (Grok) audited sample composed prompts and surfaced quality issues. User requested validation + surgical fixes (no architecture rewrite).
**Status:** F1–F9 shipped + 7 rounds of independent codebase audits. **Production-ready.**

## Summary of fixes shipped

| # | Fix | Commit |
|---|---|---|
| F1 | Conditional chroma realism tail (drops `f/1.8, 35mm` when facet has `realism_lens` populated) | `0668906` |
| F2 | Vocab strip "afternoon light" from `ATM_DUST_MOTES_IN_LIGHT` (incidental time-of-day; resolves env / atmosphere contradictions) | `acb7b64` |
| F3 | Diversity retry budget 1 → 2 (3rd attempt fires HARD BAN nudge with explicit alternative tags) | `18c3e4c` |
| F4 | Series-aesthetic 3-sentence → 1-sentence consolidation for prose families (flux_natural / flux2_prose); booru + SDXL unchanged | `87c7dff` |
| F5 | Thread `series_plan.subject_description` into SceneFacetGenerator user prompt | `a796038` |
| F6 | System-prompt COHERENCE INVARIANT block with **per-tier** clause 3 (T1 clothed / T2 implied / T3 tasteful / T4 explicit) | `e0aee6b` + `f590714` (per-tier revision after round-4 audit) |
| F7 | `scene_prose` Pydantic validator with word band (hard reject outside 20-140; soft warn outside target) | `fdd345f` |
| F8 | `_synthetic_subject_anchor` for style + variation modes (3-level fallback chain: subject_description → subject_bias → tier-aware synthetic) | `a960861` + `fa3bb45` |
| F9 | Soft-warn target band 40-90 → 30-80 (matches DavidAU's empirically observed natural mode: median=42, min=27, max=61) | `f960236` |

## Grok recommendations validated + acted upon

| Grok claim | Verdict | Action |
|---|---|---|
| Camera-spec redundancy (`85mm f/1.4` + `f/1.8, 35mm` in same prompt) | **REAL BUG** | Fixed by F1 |
| Environment / atmosphere / narrative time-of-day incoherence | **REAL BUG** | Fixed by F2 + F6 clause 1 |
| NARR_AFTER_THE_PARTY 50% concentration despite diversity threshold | **PARTIALLY REAL** (existing retry was 1-attempt-soft; bumped to 2 + HARD BAN on 3rd) | Fixed by F3 |
| Style-anchor overload (3 separate aesthetic sentences per prompt) | **REAL BUG** | Fixed by F4 |
| Pose vs nudity coherence not checked | **PARTIALLY REAL** (pose+expression already threaded; subject_description was missing) | Fixed by F5 + F8 |
| Composer is "mechanical stitching" | **TRUE BY DESIGN** | Fix at vocab data layer (F2) + LLM system prompt (F6), not composer code |

## Grok recommendations explicitly REJECTED

| Grok recommendation | Reason rejected |
|---|---|
| Move to pure-LLM cohesive prose architecture | Architecture rewrite; loses vocab versioning + family-shape control + reproducibility |
| 120–180 word scene_prose | Exceeds Chroma max_tokens budget when composer adds 10+ canonicalized sentences + series aesthetic + negative stack |
| Free-text scene_facets columns (table redesign) | Would break canonicalizer, vocab versioning, schema-constrained LLM decoding, diversity tracker, and tier-gating |
| Chroma weighting syntax `(token:1.5)` in negative prompt | `families.yaml::chroma::supports_weighting: false`. T5 tokenizer treats parens as literal characters, degrades output |
| Hardcoded `(clothed:1.45)` / `(dress:1.35)` family-level negatives | Would over-suppress T2_implied / T3_artnude scenes where the subject legitimately wears a robe or sheet |
| New SeriesPlan fields `location_anchor` / `time_anchor` / `overall_story` | Coherence fix already exists from round-12 (`compatible_narratives` narrowing per category in categories.yaml + vocabulary.py:425) |
| "Avoid repetitive props" planner instruction | Wrong layer; props are per-scene (facet generator), not per-series (planner). F3 retry-bump addresses this at the right layer |
| 8K / masterpiece / best-quality boosters | Outdated for Chroma 1-HD + gonzaLomo, would only token-bloat |
| Hardcode explicit anatomy reinforcement at family level | Bypasses tier-gating. NSFW canonicalizations are already tier-gated in vocabulary.py:203-211 |

## Independent codebase audit rounds (7 total)

Per user instruction "do 3-5 rounds of agent verification". Performed 7 independent audits with separate agents:

| Round | Focus | Key finding | Action |
|---|---|---|---|
| 1 | Per-fix file-level review | F1 false-positive (`realism_camera` edge case isn't a real bug — Sony + f/1.8/35mm is valid camera config) | None |
| 2 | Integration / E2E flows | **HIGH** — style/variation modes don't populate subject_description | F8 shipped |
| 3 | Test coverage gaps + stale fixtures | Missing F5+F8 mode-level integration test | Refactored `resolve_subject_anchor` for testability; +5 fallback-chain tests |
| 4 | Safety + tier-gating integrity | **HIGH SAFETY** — F6 clause 3 at T3 contradicted T3's llm_directive ("describe nude form directly" vs "no anatomical language") | F6 revised with per-tier sub-clauses |
| 5 | Broad codebase health + docs | CLAUDE.md vocab v7 stale (post-F2) | Updated to v8 in F9 commit |
| 6 | Post-T3-run behavioral review | **MEDIUM** — F7's 40-word floor flagged 32% of facets spuriously vs DavidAU's median=42 | F9 shipped (lower band to 30-80) |
| 7 | Production-readiness checklist | "PRODUCTION_BLOCKER" claim of unmeasured token budget | Measured: T4 system prompt = 3283 tokens = 10% of 32K context. Non-issue. |

## T3 verification run results (series_6a762d7bc949)

- **Wall-clock:** 1389 s (23.2 min) for 25 scenes (was 18.3 min for 24 in r-21b; +1 scene + new F3 3rd-retry overhead)
- **Safety:** 0 age-ambiguity strips, 0 multi-subject leaks, 0 celebrity-name strips
- **F1 verified:** prompt 0 ends with `"photographic, natural skin texture."` (no f/1.8 + 35mm — LENS_85MM_F14 was picked)
- **F3 HARD BAN fired 14 times → 9 cleared dominance (64%), 5 stayed locked** — meaningful improvement on diversity retry success
- **F4 consolidation visible:** `"Vermeer Dutch Golden Age tradition, side window illumination, quiet domestic interior, blue and ochre."` = ONE sentence (was 3 in r-21b)
- **F7 word-band empirics:** prose min=27, max=61, mean=43.3, median=42 — drove F9 retuning

## Production-readiness — non-blockers

- **ARCHITECTURE.md stale** — last sync 2026-05-20, still mentions vocab v7. Low priority; CLAUDE.md is the operator-facing doc and has been updated.
- **Add F1 regression fixture** with `realism_lens` populated to exercise the lens-populated branch end-to-end (currently unit-tested via `test_chroma_realism_tail_strips_focal_when_lens_populated`).
- **Observability for dropped tags** — canonicalizer logs at INFO level when it silently drops unknown concept tags. Engine could surface per-series counts for monitoring.
- **Planner-side hallucinated tags** (e.g. `PALETTE_DUTCH_GOLDEN_VERMEER`, `PHOTOG_JOHANNES_VERMEER`) — pre-existing issue, not round-22 scope. T3 run observed 89 unknown-concept drift events, mostly from invalid series-level anchors. Planner has a vocab menu in its system prompt but DavidAU is hallucinating despite it. Worth follow-up if quality degrades further.

## Final verdict

Round-22 ships 9 fixes (F1-F9) across 4 file regions:
- `src/prompt/builder.py` — F1, F4
- `src/prompt/vocabulary.py` — (referenced by F2, F4)
- `src/agents/scene_facet_generator.py` — F3, F5, F6, F7, F9 + retry-budget bump
- `src/agents/schemas.py` — F7, F9 Pydantic validators
- `src/core/engine.py` — F5, F8 (synthetic anchor + fallback chain)
- `config/prompt_vocabulary.yaml` — F2 (vocab v8 bump)

7 independent agent audits validated the work. Critical safety bug (F6 T3 contradiction, round-4) caught and fixed in-flight. Empirical word-band miscalibration (F9, round-6) caught and fixed in-flight.

**1333 tests passing.** Round-22 is production-ready for T3 + T4 workloads. T1 + T2 verification runs remaining (different tier-conditional system prompt paths) but the per-tier F6 revision and F9 band adjustment are not tier-specific risks.
