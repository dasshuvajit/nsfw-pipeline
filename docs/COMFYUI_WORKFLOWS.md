# ComfyUI workflow templates

> External templates only. After the 2026-05-20 cleanup the pipeline
> loads workflow JSONs you author by hand and drop into
> `config/comfyui_workflows/templates/<family>/`.

## Where templates live

```
config/comfyui_workflows/templates/<family>/<name>.json
```

Live families: **`zimage`** (the production engine) and `ideogram` (a
separate MPS fp8 experiment, not part of the series pipeline).

> **Archived 2026-08:** the `chroma` base/refine/refine_T4 templates, the
> `zimage` refine/refine_T4 templates and the `sdxl` upscale_4k template
> were pruned with the refine/4K/chroma stages (their model weights were
> deleted from disk). They live under `legacy/config_templates_chroma/`
> and `legacy/config_templates_pruned/`.

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

> **Retired contracts:** the single-pass *embedded-refiner* contract was
> removed 2026-06-13; the two-pass *image-stage* contract
> (`build_image_stage` — `load_image`/`save`/`stage_ksampler`/`upscale`/
> `prelift`, used by the SDXL refine + 4K stages) was archived 2026-08 with
> those stages. `build_external` is the only live entrypoint.

## Render path (the `art_series` pipeline) — zimage BASE-ONLY (2026-08)

One Z-Image Turbo render per (prompt × seed); the base output IS the
review image that curation + packaging consume. One resident model, no
stage swaps, well under the 48 GB budget.

| Template | Model stack | Run by | Output |
|----------|-------------|--------|--------|
| `templates/zimage/base.json` | z_image_turbo_bf16 + qwen_3_4b fp16 TE (CLIPLoader type=lumina2) + ultrafluxVAEImproved_v10 VAE + zit/ LoRA chain (zit_fdpo_v1 @1.0; lora_nsfw + lora_style present @0.0) — ModelSamplingAuraFlow shift 3.0, dpmpp_sde/beta/8, cfg 1.0 | `art_series` (all prompts) | native ~1MP buckets |
| `templates/zimage/base_hires.json` | same + PatchModelAddDownscale (deep-shrink) | `art_series --hires` | higher per-orientation buckets |

History: the earlier staged path (Chroma base → SDXL DMD detailer refine →
manual USDU 4K via `upscale_folder.py`, 2026-06-02) was archived 2026-08
when its weights were deleted. Face-mandate/USDU tuning notes live with the
archived templates and tests (`tests/legacy/`).

**Negative prompts are INERT in this path:** cfg 1.0 +
`ConditioningZeroOut`. `DEFAULT_NEGATIVE` does nothing at render time —
avoidance is carried entirely by positive prose + the art_director
validators. Do not raise cfg to revive negatives (doubles base time); if
real negative guidance is ever needed, A/B ComfyUI-NAG.

**Render-time tier gate:** `art_series` clamps the `lora_nsfw` node's
`strength_model` per tier (`_NSFW_LORA_STRENGTH`; currently 0.0 everywhere —
both NSFW_master_ZIT and dopsd_white are disabled per the 2026-07-06 user
directive, flow-DPO only).

### Extra-limb guard (auto-retry, `--anatomy-retries`, default 2)
A detailer cannot remove an extra limb, so a base render with a 3rd hand would
survive to the keepers. After each base render `art_series` runs `hand_yolov9c`
(CPU, conf 0.5) on it; if it counts **>2 hands** the render is rerolled with a new
seed up to `--anatomy-retries` times, keeping the first clean one (or the
fewest-hands render, flagged `CULL THIS CANDIDATE`, if all rerolls fail — an image
is never dropped). Cost is paid only when a defect is found; clean renders render
once. `--anatomy-retries 0` disables it. The check is hands-only — no bare-foot
detector works, so feet still rely on prompt guidance + manual culling.

### Config — `pipeline.yaml::render_pipeline`
`base_template` and `base_resolution` per orientation (portrait **896×1152**,
square 1024², landscape 1152×896 native; + off-native lanes widescreen 16:9
**1360×768** and story 9:16 **768×1360**, 2026-06-22 — all five ÷16, ~1MP,
MPS-safe). Per-model override + CLI (`--base-template`, `--hires`) layer
over it.

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
4. Drop the file into `config/comfyui_workflows/templates/<family>/`
   and point `render_pipeline.base_template` (or `--base-template`) at it.

## Verify

```bash
python -m pytest tests/test_workflow_builder_external_template.py \
    tests/test_stage_templates_integrity.py -q
```

A successful run confirms the 4-node external contract fires correctly and
the live zimage templates are MPS-safe/acyclic. To smoke-test against
ComfyUI, run a small series:

```bash
python scripts/art_series.py --niche modern_boudoir --count 1 --no-package
```
