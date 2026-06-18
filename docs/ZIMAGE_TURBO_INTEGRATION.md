# Z-Image Turbo — integration, research & audit

> Status: **IN PROGRESS** (overnight 2026-06-18). This doc is the single source of
> truth for adding Z-Image Turbo as an ALTERNATIVE render engine alongside gonzaLomo
> Chroma v30. Chroma stays the production default; Z-Image is additive. The A/B
> results + final verdict sections are filled after the validation renders.

## TL;DR decision
- **Engine:** Z-Image Turbo (Tongyi-MAI / Alibaba; ComfyUI v0.24.1 supports it **natively** — `ZImage` = Lumina2 arch, dim 3840, shift 3.0; TE = **Qwen3-4B**; 16-ch latent reusing the **Flux `ae.safetensors` VAE we already have**).
- **Best NSFW base = gonzaLomo "ZPop v4.0"** (Civitai, creator GBRX — *same creator as our Chroma v30*; its NSFW detailer YOLOs are already on disk). **BLOCKED tonight:** Civitai download returns HTTP 401 (needs a logged-in token we don't have). → see "Manual step for you".
- **Used tonight (fallback, same architecture):** official **`z_image_turbo_bf16.safetensors`** (HF, public). Swapping to ZPop later is a one-line `unet_name` change in the template.
- **Adopt it for QUALITY, not speed.** Real gen time on M4 Pro ≈ **60–160 s / 1024px @ 8 steps** (NOT the "14 s" marketing number) — roughly Chroma-class. The case is realism/skin/composition (Z-Image ranks #1 open-source on Artificial Analysis).

## CRITICAL Mac gotcha (the #1 failure mode)
macOS **15.6** (this machine) has the fp16-attention NaN bug → **pure black images** (ComfyUI #8528). Z-Image must run with ComfyUI launched as:
```
cd ~/AI/apps/ComfyUI
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python main.py --force-fp32
```
- **BF16 weights only** — fp8 throws "Float8_e4m3fn not supported on MPS" (ComfyUI #10292); nvfp4/fp4 are NVIDIA-only.
- `--force-fp32` upcasts the bf16 weights to fp32 internally (correct on MPS). It also slows Chroma, so this is a **Z-Image-session launch mode**, restored to plain `python main.py` afterward. A helper script is provided (`scripts/comfyui_zimage_mode.sh`, if created).
- LLM-unload-before-render invariant is **non-negotiable** (11.5 GB UNET + 8 GB Qwen3-4B + detailers); verify via `lms ps`.

## Files (exact)
| File | Role | Where | Status |
|---|---|---|---|
| `z_image_turbo_bf16.safetensors` (12.3 GB) | base diffusion model (official, tonight) | `models/diffusion_models/` | downloading |
| `gonzalomoZpop_v40.safetensors` (~11.5 GB) | **preferred** NSFW base (gonzaLomo) | `models/diffusion_models/` | **manual — Civitai 401** |
| `qwen_3_4b.safetensors` (8.0 GB) | text encoder (Qwen3-4B) | `models/text_encoders/` | downloading |
| `ae.safetensors` (335 MB) | VAE (Flux 16-ch) | `models/vae/` | **already on disk** ✓ |
| `gonzalomoXLFluxPony_v70PhotoXLDMD.safetensors` | SDXL detailer (nipples/genitals) | `models/checkpoints/` | already on disk ✓ |
| `gonzalomoZImage_*BboxFiles/*.pt` | NSFW detailer YOLOs | `models/ultralytics/bbox/` | already on disk ✓ |
| `realistic_snapshot_zimage_v5.safetensors` | optional realism LoRA (Z-Image-native) | `models/loras/` | skipped for v1 |

### Manual step for you (to use the *preferred* gonzaLomo base)
Logged into Civitai, download **ZPop v4.0** → `https://civitai.com/api/download/models/2932204`
(model page: `https://civitai.com/models/2192562/gonzalomo-zpop`), save as
`~/AI/apps/ComfyUI/models/diffusion_models/gonzalomoZpop_v40.safetensors`, then change
`unet_name` in `config/comfyui_workflows/templates/zimage/base.json` from
`z_image_turbo_bf16.safetensors` → `gonzalomoZpop_v40.safetensors`. Same arch, same params.
(Or set a `CIVITAI_TOKEN` and I can fetch it.)

## Workflow template (base) — `config/comfyui_workflows/templates/zimage/base.json`
Authored to the pipeline's external-template contract (patchable IDs `positive_prompt`,
`negative_prompt`, `ksampler`, `empty_latent`, `save`). Graph:
```
UNETLoader(z_image_turbo_bf16, bf16) → ModelSamplingAuraFlow(shift 3.0) → KSampler
CLIPLoader(qwen_3_4b.safetensors, type=lumina2)  → CLIPTextEncode pos / neg
                                                    neg → ConditioningZeroOut (cfg 1 → inert)
VAELoader(ae.safetensors) → VAEDecode → SaveImage
KSampler: steps 8, cfg 1.0, sampler euler, scheduler simple, denoise 1
EmptySD3LatentImage (16-ch; W/H patched per orientation)
```

### Params (gonzaLomo's own ZPop settings, confirmed by the creator)
- **Recommended:** steps **8** (creator range 5–12), cfg **1.0** (negatives inert — like Chroma flash; never raise), sampler **euler**, scheduler **beta**, shift **3.0**, 1024² native.
- **Creator's sampler options (all work):** `sa_solver`, `euler`, `dpmpp_2m`, `er_sde`, `res_2s` — with scheduler **beta**. We default to `euler` (MPS-safe, ideal for few-step distilled); `dpmpp_2m` is also MPS-proven (Chroma uses it). A/B the others; verify `res_2s` is MPS-clean before trusting it.
- **Higher quality:** steps **10–12** (marginal past 10), optional Realistic-Snapshot LoRA @ 0.6–0.7, res up to 1328² / 1536×1920 (keep ≤2–3K on MPS). cfg stays 1.0.
- Creator's note: "a little light SDXL refinement never hurts" + uses a photoreal SDXL (his MoP v7.1) to refine — we use `gonzalomoXLFluxPony_v70PhotoXLDMD`. Showcase workflow w/ embedded settings: civitai.red/images/130290388.
- **Download the BF16 Unet version** (fp8 unsupported on MPS; GGUF unnecessary on 48 GB).

## Refiner / detailer decision
**Mirror the Chroma staged pattern.** Z-Image base "mangles nipples and genitalia" (creator's words), so Phase 2 = base render (FACE UNTOUCHED — Chroma-face mandate carries over) → **SDXL detailer** pass on hands + nipples + genitalia using the on-disk `gonzalomoZImage` YOLO bbox set driven by `gonzalomoXLFluxPony_v70PhotoXLDMD`. Gate the genital detailer to **T4 only** (reuse the content-based tier-purity guard / `refine_T4` pattern). **No** global refine, **no** face detailer. 4K = existing USDU denoise-0.05 stage (no detailers at 4K → OOM). The realism LoRA is optional, added only if A/B shows waxy skin.

## NSFW approach
Base is natively uncensored (Apache-2.0, no safety filter); ZPop is NSFW-tuned. **No NSFW LoRA needed.** All existing guardrails carry over unchanged: single-adult-female ABSOLUTE SUBJECT RULE + Pydantic banned-token validator + curation multiple_faces reject + age anchors. Tier discipline identical: T1/T2 reject nudity at the prompt gate; **render-time tier-drift still possible → NudeNet package gate + visual public/ QA stays the #1 rule.** cfg-1.0 → negatives inert → avoidance lives in positive prose + validators.

## Prompt engine
Z-Image's Qwen3-4B is instruction-following natural language; our prompts are already
subject-first readable prose (post the 2026-06-17 overhaul), which suits Qwen well. The A/B
uses the **same Chroma-style prompts** (tests the "same prompt works" hypothesis). If
adherence lags, a Qwen-tuned prompt profile is the follow-up (not done tonight).

## A/B plan (vs Chroma v30)
Render an EXISTING Chroma series' prompts through Z-Image (no Chroma re-render — compare to
existing outputs). Blind-judge: skin realism, **face quality (does Z-Image hold the Chroma-face
bar?)**, hand/anatomy (post-detailer), T4 genital fidelity, composition, tier-drift rate
(NudeNet on both), and wall-clock per finished image. Win → promote Z-Image as a selectable
engine for realism-led niches (e.g. aspirational_luxe); Chroma stays default for explicit T4
until ZPop+detailer is proven.

## Open risks
- macOS 15.6 black-image bug → `--force-fp32` mandatory (handled).
- No first-hand "ran on M4" report found — full loader chain smoke-tested before wiring.
- ZPop genital/nipple competence still relies on SDXL detailers (not a one-shot T4 engine).
- Civitai SDXL-detailer-loading issues reported (Dec 2025+) — validate the detailer stage on v0.24.1 before relying on it for T4.

---
## VALIDATION RESULTS (2026-06-18, gonzaLomo ZPop v4.0 BF16 on M4 Pro)

### Big correction vs the research: NO `--force-fp32` needed
The predicted macOS-15.6 black-image blocker **did not occur** on this box. ZPop rendered
clean, correct images on the **existing** ComfyUI launch (plain `python main.py`, no flags) —
torch **2.13.0.dev** evidently mitigates the fp16-attention NaN bug. **No ComfyUI restart was
required.** (Keep the `--force-fp32` recipe documented above as the fallback if a future
torch/macOS combo regresses to black frames.)

### Speed (M4 Pro, 896×1152, 8 steps, euler/beta, cfg 1)
- **~98 s/frame warm**, 121 s cold (first load). Measured over 6 frames.
- **~2.5× faster than Chroma base** (~4 min/img). This is a real, repeatable win — Z-Image
  Turbo is both faster AND (below) at least as photoreal.

### T1 A/B — ZPop vs Chroma v30 (same 6 `aspirational_luxe` prompts, matched seeds)
Both engines are excellent and remarkably aligned (same seed → similar composition). Read:
- **Photorealism:** ZPop slightly ahead — skin texture, golden-hour light and overall
  "real-photograph" feel read a touch more convincing; Chroma is occasionally sharper on
  micro-detail. Net: ZPop ≥ Chroma on realism.
- **Faces:** ZPop faces hold up fully — the Chroma-face-mandate worry was unfounded; varied,
  photoreal, no degradation.
- **Tier-truth (NudeNet):** 5/6 clean; the lone flag was the high-cut one-piece swimsuit
  (`BUTTOCKS_EXPOSED` 0.43) — the same borderline swimwear case Chroma hits. **Parity.**
- **Composition / prompt adherence:** strong on the same subject-first prose — the "same
  Chroma-style prompt works on Z-Image" hypothesis **holds** (no prompt-engine retune needed
  for T1; a Qwen-tuned variant remains an optional future gain).

**T1 verdict:** ZPop is a legitimate upgrade path — equal-or-better realism at ~2.5× speed,
tier-truth parity, faces intact. Strong fit for realism-led niches (aspirational_luxe, poolside,
modern_boudoir).

### T3 NSFW + detailer assessment (6 `aspirational_luxe` artnude prompts → ZPop base)
- **NSFW works natively.** 5/6 frames produced bare breasts (NudeNet-confirmed); all 6 are
  tasteful, photoreal artnude with good variety (topless candid, reclining nude, frontal,
  back-nude, implied). No NSFW LoRA needed. ~99 s/frame.
- **Base anatomy — the detailer verdict:**
  - **Nipples/breasts:** mostly decent at base (a few slightly soft/asymmetric) — a light
    nipple detailer would polish, not rescue.
  - **Genitals/vulva:** render **soft and undetailed** on frontal poses → **confirms the
    research: the SDXL genital detailer is REQUIRED for explicit T4** (ZPop base is not a
    one-shot T4 engine — same limitation as Chroma).
- **Quality vs Chroma T3:** at least on par — ZPop skin/light is very photographic.
- **Tier note:** like Chroma, T3 prompts can render more frontal than the strict "vulva not
  frontal" contract — fine for GATED T3 (nudity allowed); the T1/T2-public + NudeNet-gate
  discipline carries over unchanged.

### FINAL VERDICT — ADOPT as a selectable alternative engine
ZPop (Z-Image Turbo) is a genuine win for this pipeline and directly addresses the Chroma
dissatisfaction: **equal-or-better photorealism at ~2.5× the speed (~98 s vs ~4 min/frame)**,
faces intact, NSFW native, tier-truth parity, and the **same subject-first prompts work** (no
prompt-engine retune required). Recommendation:
- **Use ZPop for realism-led + T1/T2/T3 niches now** (base render; clothed/tasteful-nude). It's
  faster and more photographic.
- **For explicit T4, build the Z-Image SDXL-detailer stage first** (reuse the on-disk
  gonzalomoZImage YOLOs + `gonzalomoXLFluxPony_v70PhotoXLDMD`, gated to T4) — until then run
  T4 on Chroma+`refine_T4`. ZPop base genitals are too soft for one-shot T4.
- Keep Chroma as the default until the T4 detailer + a wider A/B are in; promote ZPop per-niche.

## How to run it — `--engine zimage` (shipped)
```
# T1/T2/T3 — base ZPop + hands/nipples detailer:
python scripts/art_series.py --engine zimage --niche aspirational_luxe --tier T1_suggestive
# T4 explicit — adds the gonzaLomo genital detailer (tier-gated to T4):
python scripts/art_series.py --engine zimage --niche aspirational_luxe --tier T4_explicit
# base-only (skip the detailer):
python scripts/art_series.py --engine zimage --niche <id> --tier T1_suggestive --no-refine
# HIRES hero render — gonzaLomo v11 deep-shrink base at ~1.9MP (forces zimage, ~2x slower on MPS):
python scripts/art_series.py --hires --niche aspirational_luxe --tier T1_suggestive
```

### Hires mode (`--hires`, gonzaLomo v11 deep-shrink)
`PatchModelAddDownscale` (block 3 ×2 over the first 35% of steps) lets Z-Image render
coherently ABOVE its ~1MP native res. Per-orientation hires resolutions: portrait 1216×1536,
square 1344×1344, landscape 1536×1216. A/B'd vs the 896×1152 base (same seed): clearly more
detail (skin/hair/fabric/background), at **~213s vs ~98s** (≈2× slower on MPS). Use it for
max-detail hero/realism content; the standard base stays the default for volume. The SDXL
detailer stage still applies on top. Template: `templates/zimage/base_hires.json`. (Detailer
denoise was also raised 0.25→0.45 to match the creator's Z-Image-tuned value — rebuilds the
softer base anatomy.)
`--engine zimage` swaps the base + refine + refine_T4 templates as a set (explicit
`--base-template`/`--refine-template` still override). ComfyUI needs no special flags on the
current torch 2.13 build; the LLM auto-unloads before render. Ad-hoc A/B harness:
`output/ab_tests/zimage_ab.py`.

## Integration status
- **DONE:** native ComfyUI Z-Image support; `gonzalomoZpop_v40.safetensors` (BF16) +
  `qwen_3_4b.safetensors` TE + `ae` VAE in place; `templates/zimage/{base,refine,refine_T4}.json`
  authored to the pipeline contract; **`--engine zimage` flag** wired (selects base+refine+
  refine_T4 as a set); **SDXL detailer stage** (hands/nipples for T3-and-below; +gonzaLomo
  `vagina-v3.2` genital detailer for T4 only, tier-purity-gated). Validated end-to-end: base
  render (T1+T3), 12-frame A/B vs Chroma, **T4 base→detailer render (genitals rebuilt, nipples
  cleaned, face untouched)**, NudeNet tier-truth, tier-purity guard, full suite (342 tests).
- **PENDING (optional next):** Realistic-Snapshot realism LoRA if skin reads waxy on a wider
  set; 4K USDU pass for Z-Image; per-niche detailer-denoise tuning; optional per-niche
  engine auto-selection (see verdict below).

## WIDER A/B verdict (2026-06-18) — DEFAULT STAYS CHROMA (per-niche, not a flip)
Blind 4-judge panel + synthesis across 4 stylized niches (matched-prompt, base-only ZPop vs
the packaged Chroma gated frames), spot-confirmed by eye:
| Niche | Winner | Why |
|---|---|---|
| `aspirational_luxe` (realism/lifestyle, prior A/B) | **Z-Image** | faster + more photoreal |
| `fine_art_figure_study` T3 (B&W fine-art) | **Chroma** | ZPop renders in **color**, ignores the B&W assignment; Chroma = true monochrome chiaroscuro |
| `old_hollywood_glamour` T3 (B&W vintage) | **Chroma** | Chroma nails period mood; ZPop = generic modern boudoir |
| `renaissance_baroque` T3 (painterly) | **Chroma** | ZPop too glossy/photoreal; Chroma = painterly old-master |
| `goth_romantic` T3 (dark/moody) | **tie** | ZPop better low-key atmosphere, Chroma better faces |

**Core finding:** Z-Image's photoreal crispness is a *liability* when the niche's job is
tonality/mood (B&W, painterly) rather than photorealism — most starkly, **ZPop won't render
true B&W** (defaults to color). So a flat default-flip is wrong; **Chroma remains the default**.

**Recommended engine per niche:**
- **Z-Image (`--engine zimage`):** realism/lifestyle niches — `aspirational_luxe` (proven); plausibly
  `poolside_goldenhour`, `modern_boudoir`, `bohemian_naturallight`, `cottagecore_pastoral`
  (untested — opt in per niche).
- **Chroma (default):** all B&W / painterly / period / fantasy niches — `fine_art_figure_study`,
  `monochrome_fine_art`, `old_hollywood_glamour`, `renaissance_baroque`, `medieval_lady`,
  `mythology_goddess`, `arabian_nights`, `angelic_divine`, `dark_fantasy_vampire`, `goth_romantic`.
- Per-niche AUTO-selection (a `default_engine:` field on each niche read by `--engine`'s default)
  is the clean way to encode this — not yet wired (awaiting go-ahead).
