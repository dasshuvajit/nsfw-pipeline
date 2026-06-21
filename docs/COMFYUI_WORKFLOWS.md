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

> **Retired (2026-06-13):** the optional single-pass *embedded-refiner*
> contract (`refiner_positive_prompt` / `refiner_negative_prompt` /
> `refiner_ksampler` / `refiner_checkpoint_loader`) was removed with the
> v11-era workflow it served. Two-pass refinement is now done by the
> separate image-stage contract (`load_image` / `save` +
> `stage_ksampler` / `upscale`; see `refine.json` / `refine_T4.json` /
> `upscale_4k.json`), not by a refiner baked into the base graph.

## Staged pipeline (the `art_series` render path, 2026-06-02)

`art_series` renders in **separate per-model-domain stages**. (This replaced
an earlier single-pass monolith — `gonzaLomo_Chroma_4K_v12.json`, retired
2026-06-13 — which co-resided Chroma 17G + T5 9G + SDXL DMD 6.5G on every
render → ~21 GB swap on the 48 GB M4 Pro, and spent the 4K UltimateSDUpscale
(≈66 % of render time) on every image including culls.) Staged execution
splits at the only real model boundary — **Chroma (base) ↔ SDXL (everything
after)** — and gates 4K behind manual selection.

| Stage | Template | Model | Run by | Output |
|-------|----------|-------|--------|--------|
| 1 Base | `templates/chroma/base.json` | Chroma + T5 + ae | `art_series` (all prompts) | ~896×1152 |
| 2 Refine | `templates/chroma/refine.json` | SDXL DMD (detailer crops only) | `art_series` (all base outputs) | review ~1120×1440 |
| 3 4K-finish | `templates/sdxl/upscale_4k.json` (tier-neutral; the `_T4` variant was deleted 2026-06-10 — it had become byte-identical) | SDXL DMD | **`scripts/upscale_folder.py` (manual, `4k_queue/`)** | true 4K ≥3840 |

`art_series` runs stage 1 for the whole series (Chroma resident once), then
stage 2 for every base output (SDXL resident once) — the two domains never
co-reside, so each batch stays well under 48 GB. It produces **review**
images and curates/packages those. **4K is never auto-run.** Stage 3 is
**USDU-only** (the post-upscale detailers were removed — they OOM on MPS at
4K; detailers live in stage 2), so the 4K stage is tier-neutral.

**Face-true 4K (2026-06-10 A/B):** USDU `denoise` is **0.05** (seam_fix
0.10). The old 0.18 visibly restructured the Chroma face on the sold 4K
product — higher than the 0.15 global refine that was removed for exactly
that drift. At 0.05 Remacri does the upscaling and SDXL only blends seams;
the face stays the reviewed Chroma face. (0.10 = slightly more texture with
slight eye/brow crispening — rejected under the face mandate.)

**Negative prompts are INERT in this staged path:** base = cfg 1.0 +
`ConditioningZeroOut` (the gonzaLomo flash-heun contract); refine detailers
= cfg 1 lcm/DMD. `DEFAULT_NEGATIVE` does nothing at render time — avoidance
is carried entirely by positive prose + the art_director validators. Do not
raise cfg to revive negatives (doubles base time); if real negative guidance
is ever needed, A/B ComfyUI-NAG (NAGCFGGuider supports Chroma at cfg=1).

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
Packaging now emits a **`4k_queue/`** (keepers with quality_score ≥ 0.62 and
no flags — the images that earn the ~10-min pass). Review it, delete any you
veto, then:
```bash
python scripts/upscale_folder.py output/art_series/<ts>/4k_queue
```
(`--tier` is a retained no-op; the 4K stage is tier-neutral.)
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
`base_template` / `refine_template` / `refine_template_t4` / `upscale_template`,
`enable_refine` (false → review = raw base), `target_4k_long_edge`, and
`base_resolution` per orientation (portrait **896×1152**, square 1024², landscape
1152×896 native; + off-native lanes widescreen 16:9 **1360×768** and story 9:16
**768×1360**, 2026-06-22 — all five ÷16, ~1MP, MPS-safe; 4K is reached in stage 3
so the base no longer needs 1024×1536). Per-model
override + CLI (`--base-template` / `--refine-template` / `--no-refine`) layer
over it.

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

**Detailer no-op contract:** every region detailer (nipple, vagina, hand) is an
Impact-pack `FaceDetailer` — when its bbox detector returns zero detections it
passes the image through unchanged (no diffusion), so e.g. the nipple/vagina
detailers are harmless on a clothed SFW cover. **Tier routing is enforced in
code regardless** (`art_series._refine_templates_for` + the content-based
`_template_has_genital_detailer` guard), so explicit detailers never even reach a
sub-T4 main image or any cover — the no-op is a second line of defence, not the
first. Routing/purity is tested in `tests/test_sellable_pipeline.py`
(`test_refine_templates_for_keeps_covers_sfw`,
`test_genital_detailer_detection_drives_tier_purity_guard`).

Pose grounding (the *"sitting on water"* failure) is fixed upstream in the prompt
engine — `scripts/art_director.py` has a STABLE GROUNDING system-prompt block + a
hard-reject validator (`_IMPLAUSIBLE_GROUNDING_RE`), `scripts/audit_prompts.py`
penalises it, and water sub-looks in `config/niche_library.yaml` were grounded on
a solid bank.

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
python -m pytest tests/test_workflow_builder_external_template.py -q
```

A successful run confirms the 4-node external contract fires
correctly. To smoke-test against ComfyUI, run a small render:

```bash
python scripts/prepare_prompts.py --mode theme --level T2_implied
python scripts/render_prompts.py --series-id <new_id>
```
