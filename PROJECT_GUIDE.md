# PROJECT GUIDE — Operations Manual

> Operations manual for the ACTIVE **LLM-direct** pipeline (rewritten
> 2026-06-10 when the structured path was archived — the old guide described
> `prepare_prompts`/`render_prompts`/DB ops, all now under `legacy/`; see
> `legacy/README.md` or git history for those). System context: CLAUDE.md.
> Render internals: docs/COMFYUI_WORKFLOWS.md. Selling: docs/DA_GO_TO_MARKET.md.

## 1. Setup
```bash
# 1. Python deps
pip install -r requirements.txt

# 2. LM Studio running with the default LLM downloaded
#    (gemma-4-26b-a4b-it-ultra-uncensored-heretic; server on :1234).
#    Ollama running with the fallback (Cydonia 24B heretic) pulled.
python scripts/list_models.py          # shows the registry + default marker

# 3. ComfyUI running separately (path from config/pipeline.yaml::comfyui.*;
#    this box: ~/AI/apps/ComfyUI). Models needed: gonzaLomo Chroma v30,
#    gonzalomoXLFluxPony v70 DMD, t5xxl, ae VAE, 4x_foolhardy_Remacri,
#    hand_yolov9c + nipple/vagina detectors (see docs/COMFYUI_WORKFLOWS.md).
curl -s http://127.0.0.1:8188/ >/dev/null && echo ComfyUI up

# 4. Watermark handle: config/pipeline.yaml::watermark.text
```
No DB init — the LLM-direct path is DB-free (per-run `manifest.json` is the
store; **never delete `output/art_series/*/manifest.json`** — cross-run
anti-repetition memory lives there, plus `.last_seed` / `.niche_cursor` /
`.used_niches` / `.family_serials` dot-files).

## 2. Run a series
```bash
# Auto niche rotation (exhausts all 20 tier-supporting niches before repeats):
python scripts/art_series.py --auto --count 6

# Specific niche/tier:
python scripts/art_series.py --niche dark_fantasy_vampire --tier T4_explicit --count 6

# Manual brief (gets its own cross-run memory via a brief slug):
python scripts/art_series.py --brief "rainy rooftop garden at dusk" --tier T3_artnude

# Prompt-only iteration / LLM A/B (no render, no niche-cycle state consumed):
python scripts/art_series.py --niche modern_boudoir --prompts-only
```
Each run: Phase 1 LLM (prompts + SFW covers + set metadata) → verified LLM
unload → staged render (Chroma base → SDXL detailer refine) → curation →
package. Output: `output/art_series/<ts>/package/<DA folder>/` with `public/`
(watermarked SFW), `gated/` (clean), `posting_templates/`,
`POSTING_CHECKLIST.md`, plus `4k_queue/` at the run root.

**Batches** (overnight): shell loop + `caffeinate -i`, one
`python scripts/art_series.py --auto --count 6` per series, `|| true` between.
Pre-flight + circuit breaker abort fast (without consuming niche state) if
ComfyUI is down.

## 3. 4K finishing (manual, selective)
```bash
# Review <run>/4k_queue/ (auto-picked keepers: score ≥0.62, flag-free),
# delete any you veto, then:
python scripts/upscale_folder.py output/art_series/<ts>/4k_queue
```
USDU face-true (denoise 0.05) — the Chroma face survives to the sold 4K file.

## 4. Quality auditing
```bash
python scripts/audit_prompts.py output/art_series/<ts>            # score a run
python scripts/audit_prompts.py output/art_series/<ts> --verbose
```
Gate v2 scores defects + tier contracts + cliché/freshness/specificity
ladders; the same scorer gates generation inline (threshold 8.5, keep-best).

**Before posting:** VISUALLY verify every `public/` image is tier-true —
Chroma can strip clothing despite a clean T1/T2 prompt (render drift; the
negative prompt is inert at cfg 1.0). This is the #1 checklist rule.

## 5. Tests
```bash
pytest -q                 # active suite (~325 tests, <5s)
pytest tests/legacy --override-ini addopts=   # frozen legacy suite (expect import errors;
                                              # kept for reference — see legacy/README.md)
```

## 6. Tuning knobs (no code changes)
- `config/creative_direction.yaml` — house style + look_pools (hair / figure /
  face / complexion / age_look; keep lengths coprime with the strides noted
  in art_director._POOL_STRIDES).
- `config/niche_library.yaml` — niches, sub_looks (6+ each), signature
  materials, aesthetics pools, `family:` (DA architecture), persona_pool.
- `config/pipeline.yaml::render_pipeline` — stage templates + base resolutions.
- `config/llm_models.yaml` — LLM registry; `--model-tag` per run to A/B.
- Word band: `--word-band 120-180` default — do NOT widen past ~250 (flash-
  merged Chroma loses adherence).

## 7. LLM A/B protocol (used for every default-LLM decision)
1. Fixed brief/niche, `--tier T4_explicit --count 8 --prompts-only` per model
   (`--model-tag <tag>`; load each in LM Studio at 32K ctx first).
2. Mechanical: `audit_prompts.py` mean, refusal/JSON failures, wall-clock.
3. Qualitative: blind-shuffle, judge vividness/specificity/variety.
4. Decision rule: challenger must win vividness AND hold the mechanical pass
   rate AND stay under ~2 min/prompt.
