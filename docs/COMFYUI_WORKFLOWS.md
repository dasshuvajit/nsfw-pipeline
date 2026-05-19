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

`gonzaLomo_Chroma_Refiner_v11.json` is the reference example.

## Per-family default

`config/families.yaml::<family>::default_template` is the path
`render_prompts.py` falls back to when `--templates` is omitted.
Chroma's default is set to the gonzaLomo refiner. Other families
default to `null` — set yours after authoring.

```yaml
chroma:
  default_render_model: gonzalomo_chroma_v30
  default_template: templates/chroma/gonzaLomo_Chroma_Refiner_v11.json
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
