# NSFW Content Generation Pipeline

## Project Overview
A fully automated LOCAL pipeline generating artistic adult (NSFW) images of a
single fictional adult female subject, sold on DeviantArt (funnel) + Fanvue
(revenue). The production system is the **LLM-direct path** described below
(2026-05-30 pivot; hardened by the 2026-06-10 eight-lens audit — see
`~/.claude/plans/frolicking-toasting-elephant.md` for that master plan).
The selling strategy lives in docs/DA_GO_TO_MARKET.md.

## The active pipeline (LLM-direct)
```
config/niche_library.yaml ── niche/persona/aesthetic-lock selection
        │                     (src/niche/selector.py — pure function of cursor)
        ▼
scripts/art_director.py ──── Gemma writes COMPLETE prose prompts (120-180 words)
        │                     guarded by Pydantic hard gates + audit-gate v2
        ▼
scripts/art_series.py ────── orchestrates: Phase 1 LLM (prompts + SFW covers +
        │                     metadata) → unload LLM → Phase 2 render (zimage base
        │                     only by default; SDXL detailer refine OPT-IN) → curation →
        │                     tier-split packaging (+ 4k_queue/ + posting templates)
        ▼
scripts/upscale_folder.py ── manual, selective true-4K (USDU, face-true denoise 0.05)
```
- **Prompt engine** (`art_director.py`): system prompt teaches optical
  light-on-form craft, subject-first openers, anatomical-clarity/hands rules,
  T1-T4 tier directives, T4 reveal-style rotation (11 styles), framing rotation
  (8 targets across 5 orientations — portrait/square/landscape + off-native
  widescreen 16:9 & story 9:16, each paired with its suiting shot; the LLM emits
  the final orientation on merit; T4 + covers remap the off-native pair to native),
  opener-lead (5) + craft-placement (7) structural rotations,
  wardrobe/pose/time-weather axes (GARMENT_TYPES 28 / POSE_GESTURES 19 /
  ATMOSPHERE 13 incl. 4-season foliage overlays), and per-image subject looks sampled from
  `config/creative_direction.yaml` look_pools (hair 28/figure 20/face 26/
  complexion 18/age_look 7 — coprime strides + per-run shuffle). **All axes are
  DECOUPLED** via `_rotate(seq, i, run_key, axis)` (per-(run,axis) shuffle +
  coprime stride) — they no longer share one rotation index, so the combination
  space is actually explored and re-runs differ. A firm 130-175 word cap frames
  the many axes as ingredients to SELECT, keeping the richer prompts in-band.
  Anti-repetition is MECHANICAL: 3-gram similarity rejects + banned openers/tails
  enforced in the retry loop, with rejection feedback, temperature escalation,
  and a final-attempt Cydonia fallback. `scripts/diversity_report.py` is the
  GPU-free A/B harness (3-gram coverage, cross-pair similarity, entropy).
- **Quality gate** (`scripts/audit_prompts.py::score_prompt`): defect penalties
  + tier contracts (T4 requires explicit tokens; T3 requires nudity; T1/T2
  reject nudity/explicit) + quality ladders (cliché density, freshness vs run
  history, specificity). Threshold 8.5, keep-best fallback. The CLI scores a
  run dir: `python scripts/audit_prompts.py output/art_series/<ts>`.
  This module is also the single source of truth for the shared rule lists
  (sad-mood, implausible-grounding, mirrors) consumed by art_director's gates.
- **Cross-run memory**: per-run `manifest.json` files are the store (NO DB).
  `_load_niche_history` seeds banned openers/signatures/tails from the last 4
  runs of the same niche (+ covers), a global opener ban from the last 3 runs
  of ANY niche, an overused-house-word budget, and the rotation offset
  (stride count+1). `--brief` runs get a slug key. **Never delete
  output/art_series/*/manifest.json — it is the diversity memory.**
- **Render**: **BASE-ONLY by default (2026-06-21)** — the SDXL detailer refine is
  OPT-IN (`--refine`), exactly like 4K (manual `upscale_folder.py`). One series =
  one resident model, no zimage↔SDXL swap thrash. `enable_refine` defaults False
  (pipeline.yaml + resolver). Staged templates under
  `config/comfyui_workflows/templates/` (chroma/base.json — dpmpp_2m/simple@12/cfg1;
  chroma/refine.json — detailers ONLY hands+nipples, NO face detailer, NO global
  refine; refine_T4.json adds the genital detailer, T4-only via content-based
  tier-purity guard; sdxl/upscale_4k.json — USDU denoise 0.05 face-true).
  **Engine = always `zimage` unless `--engine chroma` is given explicitly — there is
  NO niche-wise/auto engine routing; never auto-pick chroma.** **Seeds are RANDOM
  per render** (logged per image; `--base-seed N` forces a deterministic
  reproducible run) — no persisted counter (the old one only advanced on success →
  an aborted run reused seeds → ComfyUI execution-cache empty renders). Per-render
  ComfyUI timeout 300s (was 1800) so a hang fails fast → reroll/circuit-breaker.
  Extra-limb guard (hand YOLO reroll); ComfyUI pre-flight + circuit breaker.
- **Packaging**: tier-split public/ (SFW only, watermarked) + gated/ (clean)
  + 4k_queue/ (score ≥0.62, flag-free) + per-image posting_templates/ with
  family-serial "Plate" titling + POSTING_CHECKLIST.md.

## Tech Stack
- Python 3.11+, Mac M4 Pro 48GB unified RAM (Apple MPS — no CUDA)
- ComfyUI (separate, http://127.0.0.1:8188). **DEFAULT engine = Z-Image Turbo**
  (`--engine zimage`, 2026-06-19): official `z_image_turbo_bf16.safetensors` (Lumina2-arch,
  **`qwen_3_4b.safetensors` fp16 TE** via stock `CLIPLoader` type=lumina2 (2026-06-29 — swapped
  from the Engineer-V6 Q8 GGUF; fp16 is crisper but ~8GB heavier-to-load → slower first render),
  **`ultrafluxVAEImproved_v10.safetensors`** VAE in `models/vae/zit/` (16ch Flux-compatible, swapped
  from `ae.safetensors`), cfg 1.0) + a LoRA stack (all in `models/loras/zit/` — the zit/ prefix
  is REQUIRED in the workflow or ComfyUI silently fails to load): `zit_fdpo_v1` (**flow-DPO**
  aesthetic LoRA @1.0, FIRST in the chain — F16/z-image-turbo-flow-dpo, Z-Image-native LoKr; counters
  Turbo flatness/washed-out lighting, A/B-verified to add cinematic contrast/polish WITHOUT changing
  composition or burning) → **`NSFW_master_ZIT`** → `dopsd_white`. **BOTH `NSFW_master_ZIT` (anatomy)
  and `dopsd_white` (style) are DISABLED @0.0 as of 2026-07-06 (user request) — only flow-DPO is active.
  Both zimage templates carry lora_nsfw + lora_style at strength_model 0.0, and `_NSFW_LORA_STRENGTH`
  is forced to 0.0 so the render-time tier-gate can't re-apply NSFW at T3/T4 → OFF for every case (all
  tiers + covers). RE-ENABLE: set both templates' lora_nsfw + lora_style strength_model back to 0.8 and
  `_NSFW_LORA_STRENGTH` back to 0.8.** (Background: NSFW_master was RE-ENABLED 2026-06-30 then disabled
  2026-07-06. It restores explicit anatomy on the NSFW-weak official base; WITHOUT it T4 renders as
  tasteful art-nude. The render-time tier-gate clamps lora_nsfw to 0.0 on T1/T2+covers and
  `_NSFW_LORA_STRENGTH` at T3/T4. When on, cumulative T3/T4 strength was 2.6 (A/B-verified, no burn);
  NudeNet package-time gate is the visual backstop for residual T1/T2 drift.) (Skipped LoRAs: alibaba
  Fun-Lora-Distill = redundant on already-distilled Turbo + blur; UltraFlux = FLUX-arch, won't load on
  Lumina2.) + `ModelSamplingAuraFlow` shift 3.0,
  `dpmpp_sde`/beta/8. Template
  `templates/zimage/base.json` (`base_hires.json` for `--hires`). Render at the official
  ~1MP buckets (896×1152/1024²/1152×896 + widescreen 16:9 1360×768 & story 9:16
  768×1360; ÷16; all ≤4096 latent tokens; NEVER 1536×2048 → MPS >12k-token/INT_MAX
  crash). base_resolution lives in 3 lockstep tables (render_pipeline DEFAULTS +
  pipeline.yaml + the --hires table) — base_resolution_for fails OPEN to portrait,
  so every orientation must be in all three. **`--engine chroma`** = gonzaLomo **Chroma v30** (FLUX-arch, T5, flash-heun
  cfg-1) — use for B&W/painterly/period/fantasy niches. Both + SDXL DMD for detailers/4K.
- LLM registry `config/llm_models.yaml` → `LLMClientPool` routes by backend:
  Ollama / LM Studio / MLX / openai_compatible (dormant remote API).
  **Default: `gemma4_26b_a4b_uncensored_hauhaucs_balanced`** (Gemma-4 26B MoE-A4B
  HauhauCS "Balanced", LM Studio — won the 2026-06-15 overnight 3-way A/B: audit
  9.12, all 8 prompts ≥8.5, fastest at 15.3s/prompt). Also registered:
  `gemma_4_31b_it_uncensored_heretic` (llmfan46 31B — same 9.12 audit but ~8x
  slower; the prose fallback), `gemma_4_26b_a4b_heretic` (prior default),
  `deckard_gemma4_31b_heretic` (dense 31B). Fallback `cydonia_heretic_24b`
  (Ollama). `--model-tag` accepts a registry key OR model id (resolves or fails
  loudly); `--llm` override.

## Critical Constraints
- **LLM and ComfyUI NEVER run simultaneously** (48GB). art_series unloads the
  LLM (verified via `lms ps`) before Phase 2.
- **Single adult female subject only** — enforced by the art_director system
  prompt (ABSOLUTE SUBJECT RULE) + Pydantic banned-token validator +
  curation's multiple_faces hard-reject. Age safety likewise (banned tokens +
  adult anchors in prose). NOTE: the render-time negative prompt is **INERT**
  (cfg 1.0 + ConditioningZeroOut) — positive prose + validators carry ALL
  avoidance; do not "fix" by raising cfg.
- **Never mix tiers in a set**: T1_suggestive / T2_implied / T3_artnude /
  T4_explicit. Gate v2 enforces tier contracts at the prompt level; the
  tier-purity guard blocks genital detailing below T4; **Chroma can still
  strip clothing at T1/T2 (render drift) — VISUAL tier-truth QA of public/
  before posting is the #1 checklist rule.**
- **Chroma face mandate**: the refine stage never touches the face (detailers
  only: hands/nipples); 4K runs face-true denoise 0.05. Never reintroduce a
  global refine or face detailer.
- **MPS limits**: RES4LYF res_* samplers crash (float64); stock samplers only.
  Post-upscale detailers OOM at 4K (USDU-only stage 3).
- **Word band 120-180**: flash-merged Chroma prefers ~150-word prose — the T5
  512-token limit is a ceiling, NOT a target; do not widen toward 300.
- Commit/push only when the user says "push it". Never embed
  round/sprint/fix-batch labels in code identifiers.

## Key Files
- CLAUDE.md — this file (project context for Claude Code sessions)
- scripts/art_director.py · scripts/art_series.py · scripts/audit_prompts.py ·
  scripts/upscale_folder.py · scripts/diversity_report.py — the active path (flat
  scripts by design; the shared rule lists live in audit_prompts)
- config/niche_library.yaml — 28 niches (12-15 sub_looks each, varied lighting +
  4-season spread, signature materials, `family:` for the 5-family DA architecture)
  + persona_pool. Includes a modern lane (athletic_studio, wild_nature,
  aspirational_luxe), a wellness lane (thermal_bathhouse — onsen/sauna/hammam/banya/
  hot-spring, tasteful T1-T3, garment axis ON), a surreal lane (surreal_dreamscape →
  chroma), and a heritage-fashion cultural lane (south_asian_editorial,
  iberian_flamenco, slavic_folk — tasteful T1-T3, never sacred/caricature; cultural
  garments live IN these niches, not the portable axis). Seasons live where they
  READ (outdoor niches' sub_looks) + 3 foliage overlays in the ATMOSPHERE axis;
  NOT forced on indoor/period niches (the coherence-override drops them).
- config/creative_direction.yaml — tunable house-style knobs + look_pools
- config/pipeline.yaml — comfyui/llm endpoints, render_pipeline templates,
  watermark; `comfyui.output_dir` is the canonical ComfyUI path source
  (~/AI/apps/ComfyUI on this box — never hardcode ~/ComfyUI)
- config/llm_models.yaml — LLM registry (default/fallback declared here)
- docs/COMFYUI_WORKFLOWS.md — staged render details + template contracts
- docs/DA_GO_TO_MARKET.md — 5-family galleries, pricing, cadence, funnel
- docs/COMPETITOR_INTEL.md — living DA-seller research (append when the user
  drops links)
- PROJECT_GUIDE.md — setup/run/test instructions (update after implementing)
- tests/ — active suite (~325 tests, `pytest -q`); tests/legacy is excluded
  via pytest.ini

## Operating notes
- Run a series: `python scripts/art_series.py --auto --count 6` (niche-cycle
  rotation) or `--niche <id> --tier <tier>`; `--prompts-only` for prompt A/Bs
  (does NOT consume niche-cycle state).
- Batches: wrap in a shell loop with `caffeinate -i`; the circuit breaker +
  pre-flight abort fast if ComfyUI dies (state is only consumed after prompts
  exist).
- LLM A/B harness: `art_director.py --brief <x> --tier T4_explicit --count 8
  --model-tag <tag>` then `audit_prompts.py` + blind qualitative judging.

## Legacy (frozen — do not extend)
The pre-pivot **structured vocab/composer path** (scenes/scene_facets/prompts
DB split, 218KB prompt_vocabulary.yaml canonicalizer, per-family composers,
modes, 9-table SQLite) is archived under `legacy/` with its scripts, config
and tests (`tests/legacy/`, excluded from the default pytest run). See
legacy/README.md. The DB file stays at the repo root as historical data;
nothing active reads it. ARCHITECTURE.md describes that legacy design and is
retained as frozen reference with a banner.
