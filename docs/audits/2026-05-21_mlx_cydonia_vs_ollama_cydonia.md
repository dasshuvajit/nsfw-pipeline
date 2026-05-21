# MLX Cydonia v3.1 4-bit vs Ollama Cydonia-heretic-24B — A/B Audit

**Date:** 2026-05-21
**Backend split:** Phase A (LLM planning) only — both runs feed the same Chroma family default template at render time.
**Mode/level/model:** `theme` · `T3_artnude` · `gonzalomo_chroma_v30`
**Vocab version:** v7

## Series under comparison

| | Baseline | Candidate |
|---|---|---|
| LLM | `cydonia_heretic_24b` (Ollama Q4_K_M) | `mlx_cydonia_v31_4bit` (MLX 4-bit) |
| Series id | `series_1722bde99e06` | `series_c6f6377c5e06` |
| Theme | Nude Figure Study in Sun-Dappled Meadow at Golden Hour | Golden-Hour Nude Study in Olive Grove at Sunset |
| Scenes / facets / prompts | 24 / 24 / 24 | 25 / 24 / 25 |
| Phase A wall-clock | ~1424 s (≈23.7 min) | 1499 s (≈25.0 min) |

## Diversity per axis (unique values / 24 facet rows)

| Axis | Ollama | MLX | Winner |
|---|---|---|---|
| realism_camera | 5 | 3 | Ollama |
| realism_lens | 5 | 3 | Ollama |
| realism_film_stock | 1/3 populated | 1/14 populated | MLX (adoption) |
| art_style_reference | 1/1 populated | 3/12 populated | **MLX (12× adoption)** |
| lighting_directive | 4 | 5 | MLX |
| mood_aesthetic | 5 | 6 | MLX |
| nsfw_anatomy | 4 | 3 | Ollama |
| nsfw_posture | 4/7 populated | 6/21 populated | **MLX (3× adoption)** |

## Tier compliance (T3_artnude — both must populate `nsfw_anatomy`)

| | Ollama | MLX |
|---|---|---|
| nsfw_anatomy populated | 24/24 | 24/24 |
| nsfw_posture populated | 7/24 | 21/24 |

## Prose length (`scene_prose`)

| | Ollama | MLX |
|---|---|---|
| mean words | 51.1 | 44.8 |
| min / max | 39 / 67 | 33 / 66 |

## Run-time signals (MLX log only — Ollama log not retained)

- unknown-concept drift events: 23
- tier-required retries: 11
- diversity-nudge retries: 9
- celebrity-name sanitiser hits: 25

The MLX numbers above are within the same order of magnitude as past Ollama runs of comparable length — the sanitiser + retry safety nets caught everything before it reached prompts/scene_facets.

## Verdict

Both Cydonia variants produce a complete, schema-valid T3 chroma series in essentially identical wall-clock (25 ± 1 min for 25 scenes). Quality is comparable but the axes-by-axes profile is not symmetric:

- **MLX wins on coverage of optional axes** — `art_style_reference` (12× adoption), `nsfw_posture` (3× adoption), `realism_film_stock` (≈5× adoption). The MLX variant evidently treats the structured-tag block as a checklist where the Ollama heretic-tune treats it as a menu to pick from. This is a real, positive signal — those columns are the ones that drive the canonicalizer-translated phrasing in the final prompt, so denser coverage = more textured prompts.
- **Ollama wins on camera + lens diversity** — 5 unique vs 3. The MLX variant concentrates on Leica + Sony where Ollama also reached for Hasselblad / Pentax / Canon.
- **Prose is tighter on MLX** — 44.8 vs 51.1 mean words. Both stay inside the 25–95 band the FluxNatural validator enforces, so this is a stylistic difference, not a quality gap.

## Default recommendation: keep `cydonia_heretic_24b` (Ollama) as project default

The argument is **operational**, not quality:

1. **Phase A → Phase B unload handoff** — Ollama exposes a clean HTTP unload (`/api/generate` with `keep_alive: 0`). MLX `mlx_lm.server` has no unload endpoint; the operator must `pkill -f mlx_lm.server` between Phase A and Phase B or ComfyUI will OOM trying to load the diffusion checkpoint into the same 48 GB. The pipeline's `unload_all()` cascades to MLX but only logs a reminder; it can't actually free the unified memory.
2. **Per-model server process** — `mlx_lm.server` hosts exactly one model. Switching LLMs needs a server restart. Ollama hot-swaps via tag.
3. **Established baseline** — heretic-tune is already the project default and has weeks of production data behind it.

Promote `mlx_cydonia_v31_4bit` to **secondary / A-B-comparison LLM** with these explicit use-cases:

- Per-scene facet re-roll when the Ollama variant has under-populated `art_style_reference` / `nsfw_posture` / `realism_film_stock` on a series — invoke with `--llm mlx_cydonia_v31_4bit --regen-facets chroma`.
- Manual A/B series for engagement testing — same scene plan, two LLMs side-by-side. (The 2026-05 multi-LLM upgrade already supports this — both LLMs' prompts coexist on the same scene rows.)

Re-evaluate this default after either: (a) `mlx_lm.server` ships a server-side unload endpoint, or (b) a future MLX-native checkpoint shows a measurable quality lead over heretic-tune on the same harness.
