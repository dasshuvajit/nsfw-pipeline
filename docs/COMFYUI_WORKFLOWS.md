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

## Staged pipeline (default for `art_series`, 2026-06-02)

`art_series` now renders in **separate per-model-domain stages** instead of
the v12 monolith. The monolith co-resided Chroma (17G) + T5 (9G) + SDXL DMD
(6.5G) every render → ~21 GB swap on the 48 GB M4 Pro, and spent the 4K
UltimateSDUpscale (≈66 % of render time) on every image including culls.
Staged execution splits at the only real model boundary — **Chroma (base) ↔
SDXL (everything after)** — and gates 4K behind manual selection.

| Stage | Template | Model | Run by | Output |
|-------|----------|-------|--------|--------|
| 1 Base | `templates/chroma/base.json` | Chroma + T5 + ae | `art_series` (all prompts) | ~896×1152 |
| 2 Refine | `templates/chroma/refine.json` | SDXL DMD (detailer crops only) | `art_series` (all base outputs) | review ~1120×1440 |
| 3 4K-finish | `templates/sdxl/upscale_4k.json` (+`_T4`) | SDXL DMD | **`scripts/upscale_folder.py` (manual, keepers)** | true 4K ≥3840 |

`art_series` runs stage 1 for the whole series (Chroma resident once), then
stage 2 for every base output (SDXL resident once) — the two domains never
co-reside, so each batch stays well under 48 GB. It produces **review**
images and curates/packages those. **4K is never auto-run.** Detailers +
USDU live in stage 3 (the proven detail-after-upscale ordering), so the
series render is tier-neutral.

### Extra-limb guard (auto-retry, `--anatomy-retries`, default 2)
A FaceDetailer cannot remove an extra limb, so a base render with a 3rd hand would
survive to the keepers. After each base render `art_series` runs `hand_yolov9c`
(CPU, conf 0.5) on it; if it counts **>2 hands** the render is rerolled with a new
seed up to `--anatomy-retries` times, keeping the first clean one (or the
fewest-hands render, flagged `CULL THIS CANDIDATE`, if all rerolls fail — an image
is never dropped). Cost is paid only when a defect is found; clean renders render
once. `--anatomy-retries 0` disables it. The check is hands-only — no bare-foot
detector works (see the no-foot-detailer note above), so feet still rely on prompt
guidance + manual culling.

### Stage-3 manual 4K — `scripts/upscale_folder.py`
Eyeball the review images, copy favourites into a folder, then:
```bash
python scripts/upscale_folder.py output/art_series/<ts>/keepers
python scripts/upscale_folder.py my_picks --tier T4_explicit   # NSFW detailers
```
It reads each image's dims, computes `upscale_by = ceil((target/long_edge)*100)/100`
(clamped ≥1.0, default target 3840), and runs the stage-3 template. **Stateless
+ base-model-agnostic** — it upscales output from any source, not just this
pipeline.

### Image-stage contract (`build_image_stage`)
Stages 2 and 3 consume an INPUT IMAGE (no `empty_latent`), so they use
`WorkflowBuilder.build_image_stage`, not `build_external`. Semantic IDs:

| ID | Patched? | Class |
|----|----------|-------|
| `load_image` | yes — `inputs.image` ← absolute path | `VHS_LoadImagePath` |
| `save` | no — presence-checked sink | `SaveImage` |
| `stage_ksampler` | optional — `inputs.seed` | a stage KSampler if present (refine is now detailer-only — see below; kept for custom templates) |
| `upscale` | optional — `inputs.upscale_by` + `seed` | `UltimateSDUpscale` |
| `prelift` | optional — `inputs.largest_size` | `ImageScaleToMaxDimension` |

### Config — `pipeline.yaml::render_pipeline`
`base_template` / `refine_template` / `upscale_template` / `upscale_template_t4`,
`enable_refine` (false → review = raw base), `target_4k_long_edge`, and
`base_resolution` per orientation (portrait reverted to native **896×1152**
— 4K is reached in stage 3 so the base no longer needs 1024×1536). Per-model
override + CLI (`--base-template` / `--refine-template` / `--no-refine`) layer
over it. `--template <monolith>` is the back-compat escape hatch to the old
single-pass v12 render.

### Refine = Chroma-face preserved (2026-06-09)
The refine stage runs **NO SDXL over the image**. The global img2img refine
(`stage_ksampler` + its VAE encode/decode) AND the dedicated face detailer were
**removed** — even at low denoise they shifted the warm/soft Chroma face toward a
sharper/cooler SDXL look (the recurring *"I prefer the chroma base face"*
complaint). Stage 2 now only **upscales 1.25× (lanczos)** and runs three
**targeted detailer crops** on the SDXL DMD model (DifferentialDiffusion-wrapped,
lcm/karras): `detailer_hands` (denoise **0.45 ×2 cycles** — restructures bad
fingers, not just polish) → `detailer_nipples` (`nipples_yolov8s`, light 0.25; a
no-op when none detected). The face and the rest of the body pass through as the
(upscaled) Chroma base, so **the keeper face == the base face**. Pinned by
`tests/test_stage_templates_integrity.py::test_refine_contract_and_values`.

**No foot detailer:** neither bare-foot YOLO on disk (`foot-yolov8l`,
`FootYolov8x_v20`) detects *nude* feet — 0 detections even at conf 0.05 — so a
foot detailer never fires. Feet rely on the prompt's "keep feet tucked / out of
frame, five toes" guidance + manual culling. Likewise a detailer cannot remove an
*extra* limb (e.g. a 3rd hand the base generated) — those are culled by hand.

**T4 variant — `refine_T4.json`:** for **T4_explicit MAIN images only** (routed via
`render_pipeline.refine_template_t4`; SFW covers and T1/T2/T3 stay on the base
`refine.json` for tier purity), the chain gains a light vagina detailer
(`vagina-v3.2`, denoise 0.25, after the nipple detailer). Unlike the bare-foot
models, `vagina-v3.2` detects reliably (0.86–0.91 conf), and the light pass adds
natural labia/skin texture to Chroma's slightly-soft vulva. It is otherwise an
exact superset of `refine.json` (pinned by a drift-guard test), so the Chroma face
is still untouched.

Pose grounding (the *"sitting on water"* failure) is fixed upstream in the prompt
engine — `scripts/art_director.py` has a STABLE GROUNDING system-prompt block + a
hard-reject validator (`_IMPLAUSIBLE_GROUNDING_RE`), `scripts/audit_prompts.py`
penalises it, and water sub-looks in `config/niche_library.yaml` were grounded on
a solid bank.

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
