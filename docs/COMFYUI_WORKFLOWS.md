# ComfyUI Workflow Templates

Canonical reference for every workflow JSON template under
`config/comfyui_workflows/`. Read this before you rebuild, edit, or
add a template — the node IDs, class types, and connection order
documented here are the contract that `src/render/workflow_builder.py`
relies on at render time. If you rename a node or swap a class type,
WorkflowBuilder raises `WorkflowTemplateError` on the next render.

## Table of Contents

1. [Template contract (what WorkflowBuilder expects)](#template-contract)
2. [Workflow → semantic ID rename flow](#workflow--semantic-id-rename-flow)
3. [External templates (`templates/{family}/`)](#external-templates)
4. [`sdxl/base.json`](#sdxlbasejson) — **built**
5. [`sdxl/ipadapter.json`](#sdxlipadapterjson) — **built**
6. [`sdxl/upscale.json`](#sdxlupscalejson) — Phase 4, not yet built
7. [`sdxl/face_detail.json`](#sdxlface_detailjson) — Phase 4, not yet built
8. [`pony/base.json`](#ponybasejson) — **built**
9. [`illustrious/base.json`](#illustriousbasejson) — **built**
10. [`illustrious/ipadapter.json`](#illustriousipadapterjson) — **built**
11. [`chroma/base.json`](#chromabasejson) — **built**
12. [`flux/base.json`](#fluxbasejson) — **built**
13. [`flux2/klein_base.json`](#flux2klein_basejson) — **built**

---

## Template contract

WorkflowBuilder loads a template by path
`config/comfyui_workflows/{family}/{name}.json` and injects per-render
parameters by **semantic node ID** (the top-level keys in the JSON
dict). A fresh ComfyUI API export keys every node by an opaque
numeric ID — you rename those to semantic names with
`scripts/rename_workflow_nodes.py` (see next section) before placing
the file in `config/comfyui_workflows/`.

### Required semantic IDs (every template)

Defined in `src/render/workflow_builder.py:_REQUIRED_NODES_BASE`:

| ID | ComfyUI class_type | What WorkflowBuilder injects |
|---|---|---|
| `load_checkpoint` | `CheckpointLoaderSimple` | `inputs.ckpt_name = style_profile.model_checkpoint` |
| `empty_latent` | `EmptyLatentImage` | `inputs.width`, `inputs.height`, `inputs.batch_size = 1` |
| `positive_prompt` | `CLIPTextEncode` | `inputs.text = prompt_text` |
| `negative_prompt` | `CLIPTextEncode` | `inputs.text = negative_prompt` |
| `ksampler` | `KSampler` | `sampler_name`, `scheduler`, `steps`, `cfg`, `seed` |

Missing any of these → `WorkflowTemplateError` on first render.

### Optional IDs

| ID | When required | What WorkflowBuilder injects |
|---|---|---|
| `lora_loader_0` | if `style_profile.lora_stack` has ≥1 entry | `lora_name`, `strength_model`, `strength_clip` |
| `lora_loader_1` | if `style_profile.lora_stack` has ≥2 entries (cap is 2) | same |
| `load_reference_image` | IPAdapter templates only | `inputs.image = <basename in ComfyUI/input/>` |
| `ipadapter_unified_loader` | IPAdapter templates only | nothing — preset baked in template |
| `ipadapter_apply` | IPAdapter templates only | `inputs.weight = ipadapter_weight` |

### Cap-2 LoRA invariant + per-family template requirement

Every LoRA-supporting family's `base.json` template **must** define
both `lora_loader_0` and `lora_loader_1` nodes with the canonical
wiring (KSampler model input drains through `lora_loader_1.MODEL`,
prompt encoders read `lora_loader_1.CLIP`). When the resolved stack
has fewer than 2 enabled LoRAs, the WorkflowBuilder strips the unused
loader nodes and re-routes through the next-most-recent model carrier.

The hard cap of **2 LoRAs per render** is a CLAUDE.md project
invariant, enforced at YAML-load time in
`src/memory/model_registry.py:106-110` (raises `ModelRegistryError`
when a `lora_stack:` declares >2 enabled entries).

Families covered: sdxl, pony, illustrious, flux, flux2 (5 of 6).
Chroma deliberately has **no** LoRA wiring — the chroma workflow
graph uses `UnetLoaderGGUF + SamplerCustomAdvanced` instead of the
SDXL-style LoraLoader chain, so chroma model YAMLs declare
`supports_lora: false` and don't attempt to template the loaders.

### What WorkflowBuilder does NOT touch

Anything it doesn't inject is baked into the template. That includes:

- The IPAdapter preset (`PLUS (high strength)`, `PLUS FACE`, …) —
  set it in the UI before exporting.
- VAE routing (always read from `load_checkpoint` output 2 for SDXL).
- `save_image.inputs.filename_prefix` — just pick something distinct
  per template so `~/AI/apps/ComfyUI/output/` stays searchable.
- Any conditioning refinements (Detail Daemon, `PerpNeg`, etc.) — add
  them in the UI and connect them into the `positive_prompt` /
  `negative_prompt` outputs as the template author intends.

---

## Workflow → semantic ID rename flow

The ComfyUI UI assigns arbitrary numeric IDs to nodes. Doing the
rename by hand is error-prone (every `[node_id, slot]` reference
inside `inputs` has to be updated too), so this repo ships
`scripts/rename_workflow_nodes.py`. Flow:

1. **Build the workflow in the ComfyUI UI** at
   <http://127.0.0.1:8188>. Wire everything up, test render once,
   make sure it produces the image you want with a placeholder
   prompt.
2. **Enable "Dev Mode Options"** in the ComfyUI settings gear
   (top-right) → scroll down to `Enable Dev Mode Options` → toggle
   on. Without this, the "Save (API Format)" button is hidden.
3. **Export via "Save (API Format)"** — NOT "Save". The regular
   save produces a UI-layout JSON; the API format produces the
   workflow-submission JSON WorkflowBuilder expects.
4. **Edit the node map** at
   `config/workflow_node_maps/{family}_{name}.yaml`: put each
   numeric ID on the left and the semantic name (from the tables
   in this doc) on the right. Quote the integer keys (`"3":
   ksampler`) so YAML doesn't coerce the string to an int.
5. **Run the rename tool:**
   ```bash
   python scripts/rename_workflow_nodes.py \
       --input ~/Downloads/sdxl_base_api.json \
       --map config/workflow_node_maps/sdxl_base.yaml \
       --output config/comfyui_workflows/sdxl/base.json
   ```
6. **Diff against the previous version** before committing — make
   sure no semantic IDs moved or disappeared. WorkflowBuilder has
   its own `_assert_required_nodes` check that fires at first
   render, but a local diff catches issues earlier.

### Exit codes for the rename tool

| Exit | Meaning |
|---|---|
| 0 | Success — output JSON written. |
| 1 | `--input` file missing or not valid JSON. |
| 2 | `--map` file missing or not valid YAML. |
| 3 | Input has node IDs with no entry in the map — the message lists the unmapped IDs. Add them to the YAML. |
| 4 | A `[node_id, slot]` connection points at a source that isn't in the map — input was inconsistent. |

---

## External templates

Storage: `config/comfyui_workflows/templates/{family}/{name}.json`.

External templates are user-authored or community-sourced ComfyUI
workflows (Civitai release pages, HuggingFace model cards,
hand-built graphs) that you want the pipeline to render through
without rewriting. They ship their own checkpoint, their own LoRAs
(if any), their own sampler + scheduler + CFG + steps, and whatever
post-processing the author baked in. The pipeline respects all of
that and only injects four fields per render.

The `templates/{family}/` subdirectory is **organizational only** —
the family your run is tagged under is always driven by `--model`'s
YAML (`config/models/{id}.yaml::family`), never inferred from the
path. `templates/chroma/foo.json` does NOT imply Chroma family.

### The 4-node contract

External templates must expose these semantic top-level IDs,
uniform across all families (including Chroma — the user's
convention is `ksampler` for the seed-carrying node even when the
actual ComfyUI class is `KSampler`, `SamplerCustomAdvanced`,
`ClownsharKSampler_Beta`, or any other sampler node):

| Semantic ID | Required inputs | What the pipeline writes |
|---|---|---|
| `positive_prompt` | `inputs.text` | LLM-generated / CLI positive prompt |
| `negative_prompt` | `inputs.text` | LLM-generated / CLI negative prompt |
| `ksampler` | `inputs.seed` | Per-scene seed |
| `empty_latent` | `inputs.width`, `inputs.height` | Per-scene resolution |

Missing any top-level ID or input field →
`WorkflowTemplateError` at preflight (before Ollama is pinged,
before rendering), non-zero exit. No fallback to built-in
templates.

Canonical example:
`config/comfyui_workflows/templates/chroma/chroma_done_properly.json`.

### What the pipeline does NOT touch

- `load_checkpoint` / `load_unet` / any checkpoint loader — the
  template's baked-in model file is what renders. `--model`
  selects the LLM prompt style; it does NOT rewrite the
  template's checkpoint.
- LoRA loader nodes — the template's LoRAs run as authored.
- Sampler name / scheduler / CFG / steps / denoise — whatever the
  template has baked in.
- IPAdapter nodes. IPAdapter is **forced off** under
  `--template`, regardless of the character's
  `reference_image_path` (an INFO line is logged). If you need
  IPAdapter, use the built-in `{family}/ipadapter.json` path.
- VAE, CLIP, any other loader.
- Post-processing nodes (face detailer, upscaler). The external
  template runs its own, if any.

### Authoring flow

1. Build / paste the workflow in ComfyUI UI
   (<http://127.0.0.1:8188>). **Render it manually once** with
   the model + custom nodes it references so you know it works on
   your box (the pipeline can't preflight custom-node
   availability — e.g. `ClownsharKSampler_Beta` needs the
   Clownshark pack installed).
2. Enable **Dev Mode Options** in the gear menu.
3. Export via **Save (API Format)**.
4. Write a node map at
   `config/workflow_node_maps/external_{name}.yaml` with the
   numeric ID → semantic ID table (same format as built-in
   templates, see § Workflow → semantic ID rename flow). The four
   IDs above are mandatory; any others can be left with their
   numeric keys — the pipeline ignores them.
5. Run `scripts/rename_workflow_nodes.py --input ... --map ...
   --output config/comfyui_workflows/templates/{family}/{name}.json`.
6. Test: `python scripts/run_once.py --mode character --level
   T2_implied --character char_001 --model <id> --template
   templates/{family}/{name}.json --verbose`.

### Role of `--model` under `--template`

- Drives Phase A (LLM): family-specific prompt style, trigger
  words, avoid words, structure rules (via
  `config/families.yaml` + `config/models/{id}.yaml`).
- Drives output directory naming for DB tracking.
- Does **NOT** override the external template's checkpoint, VAE,
  CLIP, LoRAs, or sampler.
- Still required — without it the pipeline can't pick a prompt
  style for the LLM phase.

### Scripts

- `scripts/run_once.py --template PATH` — single full cycle.
- `scripts/render_set.py --template PATH` — iteration workflow
  for a real character; `--reference-image` is ignored with an
  INFO log.
- `scripts/render_prompts.py --series-id S --models M --templates PATH`
  — Phase 4: render existing DB prompts. `--templates` pairs
  positionally with `--models` (model[i] uses template[i]); a
  single `--templates` value broadcasts to every model.
- `scripts/compare_models.py --models <single>
  --templates system,path1.json,path2.json` — A/B the built-in
  template against one or more externals for a single model.
  See "Phase 5 template-vs-model rule" below for the multi-model
  pairing semantics (the **N==N case is positional pairing**, not
  Cartesian — breaking change vs. pre-Phase-5).

#### Phase 5 template-vs-model rule (compare_models + render_prompts)

| Models len | Templates len | Behavior |
|:-:|:-:|:--|
| N | (none) | each model uses its `system` template |
| N | 1 | broadcast — every model uses that one template |
| 1 | M | one model rendered M times, once per template |
| **N | N** | **positional pairing — model[i] ↔ template[i]** (was: Cartesian) |
| N | M (mismatched, both >1) | error |

Example: `--models juggernaut_ragnarok,chroma_v10HD --templates system,templates/chroma/x.json`
renders **2** outputs (juggernaut_ragnarok with system, chroma_v10HD with the
chroma template), not 4. Users who want Cartesian should run the
script once per template.

External-template + multi-model is no longer rejected at compare-
time — the pipeline trusts the user's pairing. If you broadcast a
checkpoint-specific template across mixed-architecture models, it'll
likely produce broken images for the mismatched ones; that's the
operator's responsibility.

### Custom-node caveat

The pipeline cannot preflight custom-node availability without
pinging ComfyUI during Phase A (which it deliberately doesn't —
ComfyUI stays out of RAM while the LLM plans). If the template
references a custom node you haven't installed, the render fails
at execution time with ComfyUI's own error message. Mitigate by
running the template manually in the UI once before pasting.

## Refiner pipelines (optional two-stage templates)

Added 2026-05-15. The external-template contract supports an
**optional** second-stage refiner pass — typically a Chroma /
Flux / Pony base + SDXL refiner combo. The pipeline patches the
refiner stage with the same prompt + same seed as the base, so a
refiner template is fully deterministic for a given (scene, model,
llm, seed) tuple.

### Why no LLM-generated refiner prompt

SDXL refiners at the standard denoise band (0.10–0.25) are *polish
passes* — they enforce skin texture, sharp edges, photographic
detail; they don't re-imagine the scene. The community convention
(A1111 SDXL base+refiner, Civitai workflows, ComfyUI examples) is
to feed the refiner the **same positive prompt** as the base,
optionally with a short SDXL-keyword booster ("realistic skin
detail, sharp focus") appended.

The pipeline mirrors this exactly: at render time it patches
`refiner_positive_prompt.inputs.text` with the same string it
patched into `positive_prompt.inputs.text`. The template author
owns the static keyword booster — write whatever SDXL keywords you
prefer into a separate node (e.g. node `45` in the gonzaLomo
template) and wire a `ConditioningConcat` to combine them with the
refiner positive. The pipeline never touches that booster node.

### Optional refiner contract IDs

Add these top-level keys to a template to wire a refiner stage.
Backward-compatible: templates without these keys (every shipped
`<family>/base.json` and `chroma_done_properly.json`) continue to
validate and render unchanged.

| Semantic ID | Required input fields | Pipeline patches | Notes |
|---|---|---|---|
| `refiner_positive_prompt` | `inputs.text` | `inputs.text` ← base prompt | CLIPTextEncode using the refiner's CLIP |
| `refiner_negative_prompt` | _(none)_ | **NOT patched** | Template-owned; usually empty for denoise < 0.25 |
| `refiner_ksampler` | `inputs.seed` | `inputs.seed` ← base seed | KSampler / variant for the refiner pass |
| `refiner_checkpoint_loader` | _(none)_ | **NOT patched (metadata only)** | Pipeline reads `inputs.ckpt_name` OR `inputs.unet_name` for the PNG `refiner_checkpoint` field |

### The all-or-none pair rule

If either `refiner_positive_prompt` or `refiner_ksampler` is
present, **both must be**. This catches half-renamed templates —
the bug where you change one ID and forget the other, producing
silent broken renders. `refiner_negative_prompt` and
`refiner_checkpoint_loader` are fully optional and don't
participate in the pair check.

The preflight error looks like:

```
External template MyRefiner.json has 'refiner_positive_prompt' but
is missing 'refiner_ksampler'. The refiner pair must be present
together (refiner stage wired) or both absent (no-refiner
template).
```

### Refiner negative prompt — left empty, by design

Even when `refiner_negative_prompt` is present in the template,
the pipeline does **not** patch its `inputs.text`. At denoise =
0.15 the refiner barely touches the image; negative-prompt
influence is near-zero. Reusing the base negative would add a
meaningless CLIP-encoding overhead. A1111 and Civitai conventions
both default to empty refiner negatives. If you want a non-empty
refiner negative for some specific effect, write the text directly
into the template — the pipeline preserves it.

### Refiner checkpoint metadata

`refiner_checkpoint_loader` exists so the pipeline can record
which checkpoint actually refined the image — written into the
PNG's `nsfw_pipeline` chunk as `refiner_checkpoint`. The pipeline
reads either `inputs.ckpt_name` (CheckpointLoaderSimple) or
`inputs.unet_name` (UNETLoader) — whichever your loader uses.
Without this node, `refiner_checkpoint` lands as `null`; the
forensic record is incomplete but the render still works.

### Renaming a fresh ComfyUI-export with a refiner stage

For a typical Chroma-base + SDXL-refiner workflow saved as API
JSON from the ComfyUI UI, you'll typically have all-numeric IDs.
Rename:

1. The Chroma base CLIPTextEncode (the one feeding the base
   KSampler's `positive` input) → `positive_prompt`
2. The Chroma base CLIPTextEncode for negative (if zeroed via
   `ConditioningZeroOut`, the *source* node) → `negative_prompt`
3. The Chroma base KSampler → `ksampler`
4. The latent shape node (EmptyLatentImage / EmptySD3LatentImage)
   → `empty_latent`
5. The SDXL CLIPTextEncode that re-encodes the base text →
   `refiner_positive_prompt`
6. The SDXL CLIPTextEncode for refiner negative (usually empty) →
   `refiner_negative_prompt`
7. The SDXL refiner KSampler → `refiner_ksampler`
8. The SDXL CheckpointLoaderSimple → `refiner_checkpoint_loader`

Reference: `config/comfyui_workflows/templates/chroma/gonzaLomo_Chroma_Refiner_v11.json`
is a worked example of this exact pattern.

### Cross-family templates

The contract is family-agnostic. A flux series + SDXL refiner
needs a `templates/flux/<author>_FluxBase_SDXLRefiner.json` with
the same 8 semantic IDs. Same patching code path — the pipeline
just sees the contract and does its thing. The flux base prompt
flows into both encoders fine because SDXL CLIP handles prose
input.

### Opt-in (no new CLI flag)

To render through a refiner template:

```bash
python scripts/render_prompts.py --series-id <id> \
    --families chroma --render-with-model <some_chroma_model> \
    --templates templates/chroma/gonzaLomo_Chroma_Refiner_v11.json \
    --llm cydonia_heretic_24b
```

Choosing the refiner template via `--templates` IS the opt-in. No
new flag at `prepare_prompts` or `render_prompts`. A/B-compare
without refiner: point `--templates` at `chroma_done_properly.json`
on the same series.

### Recommended denoise band

SDXL refiners on top of a Chroma/Flux/Pony base typically run at
`denoise = 0.10–0.25`. The gonzaLomo template uses **0.15** — a
gentle skin/texture polish that doesn't alter composition. Edit
`refiner_ksampler.inputs.denoise` to tune. The pipeline doesn't
touch that field.

---

## `sdxl/base.json`

**Status:** built — lives at `config/comfyui_workflows/sdxl/base.json`.

**Purpose:** the no-reference render path. Used by every `render_set.py`
and `test_comfyui.py` run that doesn't pass `--reference-image`. This
is what the pipeline uses for the bulk of T1–T4 SDXL renders.

### Node graph

```
load_checkpoint (CheckpointLoaderSimple)
    ├── MODEL ──> lora_loader_0 (LoraLoader, optional)
    │                 ├── MODEL ──> lora_loader_1 (LoraLoader, optional)
    │                 │                 ├── MODEL ──> ksampler.model
    │                 │                 └── CLIP  ──> positive_prompt.clip
    │                 │                                negative_prompt.clip
    │                 ├── [if only 1 LoRA]
    │                 │      MODEL ──> ksampler.model
    │                 │      CLIP  ──> positive_prompt.clip / negative_prompt.clip
    ├── [if no LoRAs]
    │      MODEL ──> ksampler.model
    │      CLIP  ──> positive_prompt.clip / negative_prompt.clip
    └── VAE ─────────────────────────────────────────> vae_decode.vae

empty_latent (EmptyLatentImage) ──> ksampler.latent_image
positive_prompt (CLIPTextEncode) ──> ksampler.positive
negative_prompt (CLIPTextEncode) ──> ksampler.negative

ksampler (KSampler) ──> vae_decode (VAEDecode) ──> save_image (SaveImage)
```

### Build it in the ComfyUI UI

Menu paths are from ComfyUI's right-click "Add Node" menu:

1. **loaders → Load Checkpoint** — pick any checkpoint as the
   placeholder (WorkflowBuilder overwrites `ckpt_name` on every
   render). Rename to `load_checkpoint` via the YAML map.
2. **loaders → Load LoRA** ×2 — chain them. Leave
   `strength_model=0.5`, `strength_clip=0.5` as UI defaults;
   WorkflowBuilder will overwrite per render. Wire node 1's MODEL+CLIP
   into node 2's MODEL+CLIP inputs. Rename to `lora_loader_0` and
   `lora_loader_1`. These are only populated if the style profile
   declares LoRAs — if it has none, WorkflowBuilder skips them but
   the nodes must still exist in the graph (keep the passthrough
   wiring so MODEL/CLIP still flow).
3. **latent → Empty Latent Image** — default 512×512, the per-render
   dimensions are overwritten by WorkflowBuilder. Rename to
   `empty_latent`.
4. **conditioning → CLIP Text Encode (Prompt)** ×2 — one wired to
   `ksampler.positive`, one to `ksampler.negative`. Both take CLIP
   from the last LoRA loader (or directly from `load_checkpoint`
   if no LoRAs). Rename to `positive_prompt` and `negative_prompt`.
5. **sampling → KSampler** — defaults: `seed=0`, `steps=24`,
   `cfg=5.0`, `sampler_name=dpmpp_2m`, `scheduler=karras`,
   `denoise=1.0`. WorkflowBuilder overwrites the first five per
   render. Rename to `ksampler`.
6. **latent → VAE Decode** — wire `samples` ← `ksampler`,
   `vae` ← `load_checkpoint` (output slot 2). Rename to `vae_decode`.
7. **image → Save Image** — wire `images` ← `vae_decode`. Set
   `filename_prefix="base"` so `test_comfyui.py`-style smoke tests
   are easy to find in `~/AI/apps/ComfyUI/output/`. Rename to
   `save_image`.

**Sanity check:** queue one test prompt from the UI itself before
exporting. If the UI render works, the API render will work.

### Rename map

`config/workflow_node_maps/sdxl_base.yaml` — one entry per numeric ID
the API export assigns. See the file for the canonical list of
semantic targets.

---

## `sdxl/ipadapter.json`

**Status:** built — lives at `config/comfyui_workflows/sdxl/ipadapter.json`.

**Purpose:** the reference-image render path. Used when `render_set.py`
is called with `--reference-image <path>`, and by
`tests/integration/test_ipadapter.py`. Identical to `base.json` with three extra
nodes injected between `lora_loader_1` and `ksampler.model` to apply
the IPAdapter conditioning.

### Pre-flight: ComfyUI installs

Install **IPAdapter Plus (cubiq)** via ComfyUI-Manager:
`https://github.com/cubiq/ComfyUI_IPAdapter_plus`. Restart ComfyUI
after install. You'll also need the CLIP Vision and IPAdapter model
files — the "PLUS (high strength)" preset downloads them
automatically on first use via the `IPAdapterUnifiedLoader` node.

### Node graph

```
(same base.json prefix through lora_loader_1)
    │
    └── MODEL ──> ipadapter_unified_loader (IPAdapterUnifiedLoader)
                      ├── MODEL ───────────> ipadapter_apply (IPAdapterAdvanced)
                      └── IPADAPTER ───────> ipadapter_apply.ipadapter

load_reference_image (LoadImage) ──> ipadapter_apply.image

ipadapter_apply ──> MODEL ──> ksampler.model
```

`positive_prompt` / `negative_prompt` / `empty_latent` / `vae_decode` /
`save_image` wire up the same way as in `base.json`.

### Build it in the ComfyUI UI

Start from a working `base.json` graph, then insert:

1. **loaders → Load Image** — placeholder file is fine;
   WorkflowBuilder overwrites `inputs.image` per render. The caller
   (`render_set.py`) stages the actual reference image into
   `~/AI/apps/ComfyUI/input/` before submission. Rename to
   `load_reference_image`.
2. **ipadapter → IPAdapter Unified Loader** — set `preset` to
   `PLUS (high strength)` in the UI. Wire `model` from
   `lora_loader_1.model` (or the last model-carrying node in your
   chain). Rename to `ipadapter_unified_loader`.
3. **ipadapter → IPAdapter Advanced** — this is the node that
   actually applies the conditioning. Wire `model` and `ipadapter`
   from `ipadapter_unified_loader`, `image` from
   `load_reference_image`. Leave the other inputs at their UI
   defaults (`weight_type=linear`, `combine_embeds=concat`,
   `start_at=0`, `end_at=1`, `embeds_scaling="V only"`).
   WorkflowBuilder overwrites `weight` per render with
   `args.ipadapter_weight` (default 0.7). Rename to `ipadapter_apply`.
4. **Rewire** `ksampler.model` to come from `ipadapter_apply` (not
   from `lora_loader_1` directly).

### Reference image requirements

Square, 512×512 PNG, face-centered. Non-square references get
center-cropped by CLIP Vision and may drop the top or bottom of the
face, causing identity drift across a set. Pre-crop the source image
in any tool of your choice (Photoshop, ImageMagick `-gravity center
-crop`, etc.) before running `scripts/map_reference.py
--character-id <id> --image <path>` to register it.

### Weight tuning

WorkflowBuilder's default is `0.7` — strong enough to lock identity,
soft enough to let pose/lighting/expression vary across a set. At
`1.0` the reference dominates and every image in the set looks like
the same shot; at `0.4` the face drifts. Override with
`--ipadapter-weight` on the render command.

### Rename map

`config/workflow_node_maps/sdxl_ipadapter.yaml`.

---

## `sdxl/upscale.json`

**Status:** **NOT BUILT** — Phase 4 placeholder. Do not build yet.

**Purpose:** upscale a rendered image from (e.g.) 1024×1536 → 2048×3072
via a 4x ESRGAN model plus a low-denoise tile-diffusion re-pass.
Separate workflow (not merged into `base.json`) because upscaling is a
post-step run against finished renders, not a per-prompt render path.

### Intended node graph (when you build it)

```
load_image (LoadImage)
    └── IMAGE ──> upscale_model_loader (UpscaleModelLoader, 4x-UltraSharp)
                      └── UPSCALE_MODEL ──> image_upscale_with_model
                                                ├── IMAGE ──> vae_encode
                                                └── (4x larger image)

load_checkpoint (same SDXL checkpoint as the source render)
    ├── MODEL ──> ksampler_tiled.model
    ├── CLIP  ──> positive_prompt.clip / negative_prompt.clip
    └── VAE   ──> vae_encode.vae / vae_decode.vae

vae_encode (VAEEncode) ──> ksampler_tiled.latent_image
positive_prompt ──> ksampler_tiled.positive
negative_prompt ──> ksampler_tiled.negative

ksampler_tiled (KSampler, denoise=0.25) ──> vae_decode ──> save_image
```

Key design: `denoise=0.25` keeps the ESRGAN detail intact while
letting SDXL refine skin/eyes at the new resolution. Full denoise
would regenerate from scratch and lose identity.

### Required nodes (WorkflowBuilder will need to inject into these)

- `load_image`
- `load_checkpoint`
- `upscale_model_loader`
- `image_upscale_with_model`
- `vae_encode`
- `positive_prompt`, `negative_prompt`
- `ksampler_tiled`
- `vae_decode`
- `save_image`

See `config/workflow_node_maps/sdxl_upscale.yaml` for the placeholder
map. This template is deferred until Phase 4; documentation here is
for future-us.

---

## `sdxl/face_detail.json`

**Status:** **NOT BUILT** — Phase 4 placeholder. Do not build yet.

**Purpose:** a second-pass workflow that detects the face bbox in an
already-rendered image and re-samples just that region at higher
detail using the impact-pack `FaceDetailer` node. Separate from
`upscale.json` because you often want one without the other — full
detail on the face but no 4x upscale, or 4x upscale with no face pass.

### Intended node graph

```
load_image (LoadImage) ──> face_detailer.image
load_checkpoint ──> face_detailer.model, .clip, .vae
positive_prompt / negative_prompt ──> face_detailer.positive/.negative

ultralytics_face_detector (UltralyticsDetectorProvider, 'face_yolov8m.pt')
    └── BBOX_DETECTOR ──> face_detailer.bbox_detector

sam_loader (SAMLoader, 'sam_vit_b_01ec64.pth')
    └── SAM_MODEL ──> face_detailer.sam_model_opt

face_detailer (FaceDetailer, impact-pack)
    └── IMAGE ──> save_image
```

`FaceDetailer` handles the detect-mask-sample-composite pipeline
internally; you just wire the inputs and it emits a final image with
the face region re-rendered at a denoise level you configure in the
UI (typically 0.35 — high enough to sharpen, low enough to preserve
identity).

### Pre-flight

Install **ComfyUI-Impact-Pack**:
`https://github.com/ltdrdata/ComfyUI-Impact-Pack`. It bundles the
`FaceDetailer`, `UltralyticsDetectorProvider`, and `SAMLoader` nodes.
First run downloads the YOLO + SAM weights.

See `config/workflow_node_maps/sdxl_face_detail.yaml` for the
placeholder map.

---

## `flux/base.json`

**Status:** **BUILT** — ships with the `gonzalomo_flux_v30` registry row
(`gonzalomoXLFluxPony_v30FluxDAIO.safetensors`). Routed through
`WorkflowBuilder._build_flux()` via `workflow_family='flux'`.

**Purpose:** base flux render path. Flux is a materially different
architecture from SDXL — no single CheckpointLoader, no classical
CFG, and negative prompts are effectively ignored. `_REQUIRED_NODES_FLUX`
replaces `_REQUIRED_NODES_BASE` for this family.

### Architectural differences from SDXL

- **No single CheckpointLoader.** Flux loads the UNet via
  `UnetLoaderGGUF` (GGUF Q8_0), text encoders via `DualCLIPLoader`
  (T5-XXL + CLIP-L, `type='flux'`), and VAE via `VAELoader` —
  three separate files living in `models/unet/`, `models/clip/`,
  `models/vae/`.
- **Latent is SD3 style.** Use `EmptySD3LatentImage`, not
  `EmptyLatentImage`. Width/height still work the same way.
- **No classical CFG.** `KSampler.cfg` is hardcoded to `1.0` in the
  template. The user-tunable "guidance" lives on `FluxGuidance` and
  is driven by `style_profile.cfg` (default 3.5).
- **`ModelSamplingFlux` required.** Injects flux-dev's flow-matching
  schedule with `max_shift=1.15`, `base_shift=0.5` (tuned for 1024²).
  Width/height must match `empty_latent` — `_build_flux` writes both.
- **Sampler defaults:** `dpmpp_2m` + `beta`, `steps=28`. Explicit
  creator recommendations for the `fluxedUpFluxNSFW_71Q8` model.
- **Negative prompts are wired but ignored.** `KSampler` requires a
  `negative` input, so `negative_prompt` stays in the graph. At
  `cfg=1.0` it's a no-op. `model_registry.supports_negative_prompt=0`
  reflects this.
- **LoRAs supported.** `lora_loader_0` and `lora_loader_1` are
  scaffolded in the template and chained into both the MODEL and
  CLIP paths. `_build_flux` strips unused loader nodes and rewires
  downstream consumers. Activate by setting
  `style_profiles.lora_stack` to a JSON list of
  `{"name", "strength"}` entries.

### Node graph (13 nodes)

```
load_unet (UnetLoaderGGUF)
    └── MODEL ──> lora_loader_0.model ──> lora_loader_1.model
                                              └── MODEL ──> model_sampling
                                                                └── ksampler.model

load_clip (DualCLIPLoader, type='flux', t5xxl + clip_l)
    └── CLIP ──> lora_loader_0.clip ──> lora_loader_1.clip
                                            └── CLIP ──> positive_prompt / negative_prompt

load_vae (VAELoader, 'ae.safetensors')
    └── VAE ──> vae_decode.vae

empty_latent (EmptySD3LatentImage) ──> ksampler.latent_image
positive_prompt (CLIPTextEncode) ──> flux_guidance (FluxGuidance, guidance=3.5)
                                         └──> ksampler.positive
negative_prompt (CLIPTextEncode) ──> ksampler.negative (ignored at cfg=1.0)

ksampler (KSampler, dpmpp_2m / beta / steps=28 / cfg=1.0)
    └──> vae_decode ──> save_image
```

### Required semantic IDs (`_REQUIRED_NODES_FLUX`)

`load_unet`, `model_sampling`, `load_clip`, `load_vae`,
`positive_prompt`, `negative_prompt`, `flux_guidance`, `empty_latent`,
`ksampler`. LoRA loaders are optional (stripped if unused).

### Pre-flight

- `~/AI/apps/ComfyUI/models/unet/fluxedUpFluxNSFW_71Q8GGUF.gguf`
- `~/AI/apps/ComfyUI/models/clip/t5xxl_fp16.safetensors`
- `~/AI/apps/ComfyUI/models/clip/clip_l.safetensors`
- `~/AI/apps/ComfyUI/models/vae/ae.safetensors`
- `~/AI/apps/ComfyUI/custom_nodes/ComfyUI-GGUF/` (provides
  `UnetLoaderGGUF`)

### Rename map

`config/workflow_node_maps/flux_base.yaml` documents the semantic
IDs for reference — the template is hand-authored with the right
keys, no rename pass needed.

---

## `illustrious/base.json`

**Status:** **BUILT** — ships with the `perfection_realistic_ilxl`
registry row (`perfectionRealisticILXL_70.safetensors`). Routed
through the standard SDXL `WorkflowBuilder.build()` path via
`workflow_family='illustrious'` — Illustrious XL shares the SDXL
node graph shape, so no new dispatch branch or `_build_illustrious`
method is needed.

**Purpose:** base Illustrious render path. Same 9-node graph as
`sdxl/base.json`, tuned with the CivitAI-recommended settings
(`dpmpp_3m_sde` + `simple` at 24 steps, cfg 4) from the
`perfection_realistic_ilxl` model card.

### Architectural notes

- **SDXL-compatible graph.** `CheckpointLoaderSimple` +
  `EmptyLatentImage` + standard `KSampler`. VAE and CLIP are baked
  into the checkpoint, so no separate loaders are required.
- **Sampler defaults:** `dpmpp_3m_sde` + `simple` (explicitly not
  `normal` per the model card), `steps=24`, `cfg=4.0`. Native
  resolution is 1024² (same as SDXL); portrait/landscape use
  832×1216 / 1216×832.
- **LoRAs supported.** `lora_loader_0` and `lora_loader_1` are
  scaffolded and chained into MODEL + CLIP, exactly like
  `sdxl/base.json`. The shared
  `WorkflowBuilder._strip_unused_lora_loaders()` helper removes
  unused slots at build time. `illustrious_quality_nsfw` ships with
  `detail_tweaker_xl` (0.4) + `NSFW_V3` (0.6) pre-wired;
  `illustrious_speed_dmd2` ships with `dmd2_sdxl_4step_lora` (1.0)
  and uses cfg=1.0 / steps=8 for a ~3× speed boost.
- **Negative prompts supported.** `supports_negative_prompt=1`.
- **IPAdapter supported.** `supports_ipadapter=1` — Illustrious
  shares SDXL's CLIP-vision backbone, so cubiq IPAdapter SDXL models
  work identically.
- **Prompt style:** `sdxl_keywords` (comma-separated Danbooru tags
  with Illustrious quality prefix `masterpiece, best quality, very
  aware, newest`). Do NOT use Pony `score_X` tags — they degrade
  Illustrious output.

### Node graph (9 nodes)

```
load_checkpoint (CheckpointLoaderSimple)
    ├── MODEL ──> lora_loader_0.model ──> lora_loader_1.model
    │                                         └── MODEL ──> ksampler.model
    ├── CLIP ──> lora_loader_0.clip ──> lora_loader_1.clip
    │                                       └── CLIP ──> positive_prompt / negative_prompt
    └── VAE ──> vae_decode.vae

empty_latent (EmptyLatentImage) ──> ksampler.latent_image
positive_prompt (CLIPTextEncode) ──> ksampler.positive
negative_prompt (CLIPTextEncode) ──> ksampler.negative

ksampler (KSampler, dpmpp_3m_sde / simple / steps=24 / cfg=4.0)
    └──> vae_decode ──> save_image
```

### Required semantic IDs (`_REQUIRED_NODES_BASE`)

`load_checkpoint`, `positive_prompt`, `negative_prompt`,
`empty_latent`, `ksampler` (shared with SDXL — no Illustrious-specific
required-nodes tuple needed). LoRA loaders are optional (stripped if
unused via the shared helper).

### Pre-flight

- `~/AI/apps/ComfyUI/models/checkpoints/perfectionRealisticILXL_70.safetensors`
- (Quality profile) `~/AI/apps/ComfyUI/models/loras/detail_tweaker_xl.safetensors`
- (Quality profile) `~/AI/apps/ComfyUI/models/loras/NSFW_V3.safetensors`
- (Speed profile) `~/AI/apps/ComfyUI/models/loras/dmd2_sdxl_4step_lora.safetensors`

### Rename map

`config/workflow_node_maps/illustrious_base.yaml` documents the
semantic IDs for reference — the template is hand-authored with the
right keys, no rename pass needed.

---

## `illustrious/ipadapter.json`

**Status:** **BUILT** — clone of `sdxl/ipadapter.json` with
Illustrious defaults (checkpoint filename, sampler/scheduler). The
SDXL IPAdapter chain (`ipadapter_unified_loader` +
`ipadapter_apply` + `load_reference_image`) works on Illustrious
because they share the SDXL CLIP-vision backbone.

Same required nodes as `sdxl/ipadapter.json` (`_REQUIRED_NODES_IPADAPTER`).
Activated by passing `ipadapter_image` to `WorkflowBuilder.build()`,
same contract as the SDXL family.

---

## Adding a new architecture

If you ever want a sixth family (SD 3.5, Pixart, Lumina, whatever),
the steps are:

1. Pick a `family` string (`sd35`, `pixart`, …) and add a block for
   it under `families:` in `config/families.yaml` — declare its
   `prompt_style`, `quality_prefix/suffix`, negatives, capabilities,
   and `llm_hint`.
2. Add a directory `config/comfyui_workflows/{family}/` and put the
   renamed `base.json` there.
3. Add a rename map at `config/workflow_node_maps/{family}_base.yaml`.
4. Add a section to this doc describing the architecture's quirks
   and the intended node graph.
5. Drop a per-model YAML at `config/models/{model_id}.yaml` with
   `family: {family}` and any `prompt.extend:` hooks it needs.
6. Render once via `python scripts/render_set.py --model <new_id>
   --dry-run` to confirm WorkflowBuilder finds the template.
