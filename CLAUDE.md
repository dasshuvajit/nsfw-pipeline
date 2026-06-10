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
        │                     metadata) → unload LLM → Phase 2 staged render
        │                     (Chroma base → SDXL detailer refine) → curation →
        │                     tier-split packaging (+ 4k_queue/ + posting templates)
        ▼
scripts/upscale_folder.py ── manual, selective true-4K (USDU, face-true denoise 0.05)
```
- **Prompt engine** (`art_director.py`): system prompt teaches optical
  light-on-form craft, subject-first openers, anatomical-clarity/hands rules,
  T1-T4 tier directives, T4 reveal-style rotation (11 styles), framing rotation
  (8 targets), opener-lead (5) + craft-placement (7) structural rotations, and
  per-image subject looks sampled from `config/creative_direction.yaml`
  look_pools (hair/figure/face/complexion/age_look — prime strides + per-run
  shuffle). Anti-repetition is MECHANICAL: 3-gram similarity rejects + banned
  openers/tails enforced in the retry loop, with rejection feedback, temperature
  escalation, and a final-attempt Cydonia fallback.
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
- **Render**: staged templates under `config/comfyui_workflows/templates/`
  (chroma/base.json — dpmpp_2m/simple@12/cfg1; chroma/refine.json — detailers
  ONLY hands+nipples, NO face detailer, NO global refine; refine_T4.json adds
  the genital detailer, T4-only via content-based tier-purity guard;
  sdxl/upscale_4k.json — USDU denoise 0.05 face-true). Per-prompt seeds;
  extra-limb guard (hand YOLO reroll); ComfyUI pre-flight + circuit breaker.
- **Packaging**: tier-split public/ (SFW only, watermarked) + gated/ (clean)
  + 4k_queue/ (score ≥0.62, flag-free) + per-image posting_templates/ with
  family-serial "Plate" titling + POSTING_CHECKLIST.md.

## Tech Stack
- Python 3.11+, Mac M4 Pro 48GB unified RAM (Apple MPS — no CUDA)
- ComfyUI (separate, http://127.0.0.1:8188) — image model **gonzaLomo Chroma
  v30** (Chroma = FLUX-arch, T5-prompted, flash-heun LoRA baked in → cfg 1.0
  contract) + SDXL DMD for detailers/4K
- LLM registry `config/llm_models.yaml` → `LLMClientPool` routes by backend:
  Ollama / LM Studio / MLX / openai_compatible (dormant remote API).
  **Default: `gemma_4_26b_a4b_heretic`** (26B MoE, LM Studio, 32K ctx);
  fallback `cydonia_heretic_24b` (Ollama). `--model-tag` / `--llm` override.

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
  scripts/upscale_folder.py — the active path (flat scripts by design; the
  shared rule lists live in audit_prompts)
- config/niche_library.yaml — 20 niches (5-6 sub_looks each, signature
  materials, `family:` for the 5-family DA architecture) + persona_pool
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
