# ComfyUI workflow templates

> External templates only. After the 2026-05-20 cleanup the pipeline
> loads workflow JSONs you author by hand and drop into
> `config/comfyui_workflows/templates/<family>/`.

## Where templates live

```
config/comfyui_workflows/templates/<family>/<name>.json
```

`<family>` is one of `sdxl / pony / illustrious / flux / flux2 /
chroma`. The directory name IS the family — the render-time family-
match validator reads it from the path.

## Required nodes — the 4-node contract

Every external template MUST expose these top-level semantic IDs.
Each must carry the listed input field.

| Semantic ID         | Input field          | Typical class            |
| ------------------- | -------------------- | ------------------------ |
| `positive_prompt`   | `inputs.text`        | `CLIPTextEncode` (or compatible) |
| `negative_prompt`   | `inputs.text`        | `CLIPTextEncode` (or compatible) |
| `ksampler`          | `inputs.seed`        | `KSampler` / `SamplerCustomAdvanced` / `ClownsharKSampler_Beta` / … |
| `empty_latent`      | `inputs.width`, `inputs.height` | `EmptyLatentImage` / `EmptySD3LatentImage` / `EmptyHunyuanLatentVideo` / … |

Failures here raise `WorkflowTemplateError` at preflight with a
pointer at the missing field.

## Optional refiner contract

Two-stage workflows (e.g. Chroma base + SDXL refiner) declare these
extra IDs. Pair rule: if `refiner_positive_prompt` OR `refiner_ksampler`
is present, BOTH must be.

| Semantic ID                | Patched? | Notes |
| -------------------------- | -------- | ----- |
| `refiner_positive_prompt`  | yes — `inputs.text` ← SAME prompt as base | CLIPTextEncode using the refiner CLIP |
| `refiner_negative_prompt`  | no       | Template-owned; usually empty for denoise ≤ 0.25 |
| `refiner_ksampler`         | yes — `inputs.seed` ← SAME seed as base | Refiner KSampler |
| `refiner_checkpoint_loader`| no — metadata only | Pipeline reads `inputs.ckpt_name` OR `inputs.unet_name` for the PNG `refiner_checkpoint` field |

`gonzaLomo_Chroma_Refiner_v11.json` is the reference example for the
refiner-pair contract. The current production template
(`gonzaLomo_Chroma_4K_v12.json`, below) deliberately OMITS the pair.

## Production Chroma graph — `gonzaLomo_Chroma_4K_v12.json` (true 4K)

The default `art_series` template (`scripts/art_series.py::DEFAULT_TEMPLATE`).
End-to-end single-pass chain that turns one prompt + seed + base
resolution into a finished true-4K image. Authored 2026-06-02 from an
RnD pass (refiner architecture / 4K method / detailer chain / refiner
prompt, adversarially integrated). Pixel-verified on M4 Pro 48GB.

**Stage order** (and measured share of an ~859s portrait render):

| Stage | Nodes | What | Time |
| ----- | ----- | ---- | ---- |
| Base Chroma | `195`/`186`/`188`/`183`/`153` → `positive_prompt`/`negative_prompt`/`130` → `empty_latent` → `ksampler` → `13` | gonzaLomo Chroma, euler/beta/12/cfg1, ae VAE, full T5 prose | 291s (34%) |
| Bridge | `143` (ImageScaleBy 1.25 lanczos) → `132` (VAEEncode into SDXL VAE) | hand pixels into the SDXL arch | ~4s |
| SDXL refine | `refiner_checkpoint_loader` (gonzalomoXLFluxPony_v60PhotoXLDMD) → `refiner_ksampler_local` (lcm/karras/4/cfg1/**denoise 0.15**) → `133` | low-denoise skin/detail pass, NO Deep Shrink | 36s |
| **True 4K** | `skin_upscale_model` (4x Remacri) → `ultimate_upscale` (UltimateSDUpscale, lcm/karras/**6 steps**/denoise 0.18, **tile 1280**, Half-Tile seam fix) | ESRGAN lift + tiled diffusion polish → 2560×3840 | 492s (57%) |
| Detailers | `det_face_detector`(face_yolov8m) → `detailer_face` → `det_eye_detector`(Eyeful_v2) → `detailer_eyes` → `det_hand_detector`(hand_yolov8s) → `detailer_hands`, all on the SDXL model w/ `sam_loader` | face → eyes → hands, serial, lcm/karras | ~50s |
| Save | `171` | final 4K PNG | <1s |

**Resolution math:** base long-edge × 1.25 (bridge) × 2.0 (USDU) → 3840.
`art_series` ORIENTATIONS portrait `(1024,1536)` and landscape
`(1536,1024)` hit true 4K; square `(1024,1024)` → 2560² (raise the
square base to 1536 for a true-4K square at higher base cost).

**Key design decisions (from the RnD):**
- *Keep* the cross-arch SDXL DMD refiner — it must be resident anyway for
  the detailers + USDU polish, so a same-model Chroma refine buys no
  memory and the DMD finetune is the skin engine. (R1's same-model refine
  is the documented fallback if SDXL over-smooths — drop refine denoise
  to 0.12 first.)
- *Generic, content-neutral* refiner/USDU/detailer prompt (`det_pos` /
  `det_neg`) — short, SDXL-77-token-safe, no garment/artist/scene tokens
  (so it never re-dresses a nude base). The base Chroma **prose is never
  routed through SDXL's CLIP** (T5 has no 77-token ceiling; SDXL would
  truncate it). This is why the optional `refiner_positive_prompt` /
  `refiner_ksampler` pair is OMITTED — the refine KSampler is named
  `refiner_ksampler_local` so the pair rule is a no-op.
- *Dropped* PatchModelAddDownscale (Deep Shrink) — harmful at low denoise
  and conflicts with tiled USDU.
- *Face detector fix* — v11 used `Eyeful_v2-Paired.pt` (an EYE detector)
  as the face pass; v12 uses `face_yolov8m.pt` for the face and demotes
  Eyeful to a dedicated eyes pass.
- *MPS-safe only* — every sampler is euler/lcm, every scheduler
  beta/karras. No RES4LYF `res_*` samplers (they crash on Apple MPS).

**Render time:** ~14.3 min/portrait-4K on M4 Pro 48GB (USDU optimized
from tile-1024/8-step ≈ 19 min). Co-resident Chroma+SDXL+T5 swaps a few
GB on the 48GB box but it is compute-bound, not swap-bound (an explicit
model-unload node gave no wall-time win and was removed).

### T4 NSFW variant — `gonzaLomo_Chroma_4K_v12_T4.json`

Base v12 chain + two tier-gated NSFW-region detailers appended before
SaveImage: `det_nipple_detector`(nipples_yolov8s) → `detailer_nipples`
→ `det_vagina_detector`(vagina-v3.2) → `detailer_vagina` (both
lcm/karras/6/denoise 0.25, crop_factor 2.0). `art_series` auto-selects
this for `--tier T4_explicit` **main images only**; SFW covers and
T3-and-below stay on the base v12 template (their detectors would not
fire and NSFW-region inpainting is unwanted there). Pin `--template` to
override.

## Per-family default

`config/families.yaml::<family>::default_template` is the path
`render_prompts.py` falls back to when `--templates` is omitted.
Chroma's default is set to the gonzaLomo refiner. Other families
default to `null` — set yours after authoring.

```yaml
chroma:
  default_render_model: gonzalomo_chroma_v30
  default_template: templates/chroma/gonzaLomo_Chroma_4K_v12.json
```

`null` means "no default; `--templates` required at render time" — the
engine raises a clear error.

## Authoring a new template

1. Build the workflow in the ComfyUI UI. Tune sampler / cfg / steps /
   LoRAs / VAE / CLIP / post-processing exactly how you want every
   render to look.
2. `Save (API Format)` → JSON file.
3. Open the JSON in your editor. Find/replace four node IDs to the
   contract names. Inside `inputs` arrays the IDs appear as
   `["<numeric_id>", N]` — replace the numeric ID in BOTH the
   top-level key AND every reference array.
4. Drop the file into `config/comfyui_workflows/templates/<family>/`.
5. (Optional) Set as default in `config/families.yaml`.

## Verify

```bash
python -m pytest tests/test_workflow_builder_external_template.py \
                  tests/test_workflow_builder_refiner.py -q
```

A successful run confirms the 4-node contract + refiner-pair rule fire
correctly. To smoke-test against ComfyUI, run a small render:

```bash
python scripts/prepare_prompts.py --mode theme --level T2_implied
python scripts/render_prompts.py --series-id <new_id>
```
