# PROJECT GUIDE — Operations Manual

> Operations manual. System design lives in
> [`ARCHITECTURE.md`](./ARCHITECTURE.md); Claude Code rules live in
> [`CLAUDE.md`](./CLAUDE.md). This file is the end-to-end command
> reference + setup walkthrough after the 2026-05-20 cleanup
> (custom-workflow generation + IPAdapter + character mode + 8 legacy
> scripts removed; external templates only).

## 1. Setup

```bash
# 1. Python deps
pip install -r requirements.txt

# 2. Init the DB (9 tables: series, scenes, scene_facets, prompts,
#    images, sets, posts, generation_memory, run_log)
python scripts/init_db.py

# 3. Confirm Ollama is up
curl http://localhost:11434/api/tags

# 4. Confirm ComfyUI is up (separate process)
curl http://127.0.0.1:8188/queue

# 5. List registered models + LLMs
python scripts/list_models.py
```

## 2. Quick start

```bash
# Plan a series (uses pipeline.default_model_id = gonzalomo_chroma_v30
# + cydonia_heretic_24b default LLM + theme/style/niche/variation
# weighted random mode + T2_implied tier).
python scripts/prepare_prompts.py

# Render (uses families.yaml::<family>::default_template; chroma falls
# back to templates/chroma/gonzaLomo_Chroma_Refiner_v11.json).
python scripts/render_prompts.py --series-id <id from step above>
```

## 3. `prepare_prompts.py` — full CLI

| Flag                                 | Required?                            | Default                                                                                | Purpose |
| ------------------------------------ | ------------------------------------ | -------------------------------------------------------------------------------------- | ------- |
| `--mode {theme,style,niche,variation}` | optional                           | weighted random from `pipeline.mode_weights` (theme 0.50 / niche 0.25 / style 0.125 / variation 0.125) | Pipeline mode |
| `--level {T1_suggestive,T2_implied,T3_artnude,T4_explicit}` | optional        | `T2_implied` for new series; inherited from DB for `--series-id` re-runs (mismatch rejected) | Content tier |
| `--style-profile <id>`              | optional                            | `pipeline.default_style_profile_id`                                                    | Style profile YAML id |
| `--models <id,id,...>`              | optional (XOR with `--families`)     | `pipeline.default_model_id` = `gonzalomo_chroma_v30`                                  | Model-kind prep |
| `--families <id,id,...>`            | optional (XOR with `--models`)       | none                                                                                   | Family-kind prep |
| `--series-id <id>`                  | optional                             | new series                                                                             | Re-target an existing series — skips plan + scenes |
| `--regen-facets <family,...>`       | optional                             | none                                                                                   | DELETE + regen scene_facets rows for these families |
| `--regen-prompts <model,...>`       | optional                             | none                                                                                   | DELETE + recompose model-kind prompts |
| `--regen-family-prompts <fam,...>`  | optional                             | none                                                                                   | DELETE + recompose family-kind prompts |
| `--llm <id>`                        | optional                             | `default_llm` from `config/llm_models.yaml` (`cydonia_heretic_24b`)                  | Override LLM for this run |
| `--verbose / -v`                    | optional                             | off                                                                                    | DEBUG-level logging |

### Examples

```bash
# Friction-free default — chroma + theme + T2 + cydonia
python scripts/prepare_prompts.py

# Explicit T4 chroma model-kind prep
python scripts/prepare_prompts.py --mode theme --level T4_explicit --models gonzalomo_chroma_v30

# Family-kind prep (broader; renders later via --render-with-model)
python scripts/prepare_prompts.py --mode theme --level T4_explicit --families chroma

# Re-prep facets for an existing series after a vocab bump
python scripts/prepare_prompts.py --series-id series_abc --regen-facets chroma --llm cydonia_heretic_24b

# A/B comparison — same series, different LLM
python scripts/prepare_prompts.py --series-id series_abc --regen-prompts gonzalomo_chroma_v30 --llm qwen3_abliterated_30b
```

## 4. `render_prompts.py` — full CLI

| Flag                              | Required?                                                  | Default                                                | Purpose |
| --------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------ | ------- |
| `--series-id <id>`                | one of `--series-id` / `--scene-id` required               | —                                                      | Render every pending prompt in series |
| `--scene-id <id>`                 | one of `--series-id` / `--scene-id` required               | —                                                      | Render a single scene's prompt |
| `--models <id,id,...>`            | optional (XOR with `--families`)                           | `pipeline.default_model_id` = `gonzalomo_chroma_v30` | Render with these models (model-kind prompts) |
| `--families <id,id,...>`          | optional (XOR with `--models`)                             | none                                                   | Render family-kind prompts (requires `--render-with-model`) |
| `--render-with-model <id>`        | REQUIRED when `--families` given                           | —                                                      | Which checkpoint actually renders the family-kind prompt; family must match |
| `--templates <path,...>`          | optional                                                   | per-family `default_template` from `config/families.yaml` | Override template; family extracted from `templates/<family>/X.json` path; mismatch raises |
| `--llm <id>`                      | REQUIRED when the (series, target, target_kind) has prompts | —                                                      | Filter to one LLM's prompts |
| `--no-export`                     | optional                                                   | off                                                    | Skip Phase C (DA / Patreon export) |
| `--db-path <path>`                | optional                                                   | `nsfw_pipeline.db`                                    | DB location |
| `--verbose / -v`                  | optional                                                   | off                                                    | DEBUG-level logging |

### Examples

```bash
# Friction-free default — render with chroma family default template
python scripts/render_prompts.py --series-id series_abc

# Explicit model + default template (chroma → gonzaLomo refiner)
python scripts/render_prompts.py --series-id series_abc --models gonzalomo_chroma_v30

# Family-kind prompts rendered via a specific checkpoint
python scripts/render_prompts.py --series-id series_abc --families chroma --render-with-model gonzalomo_chroma_v30

# Override template (must be same family as prompt)
python scripts/render_prompts.py --series-id series_abc \
   --templates templates/chroma/chroma_done_properly.json

# Multi-LLM series: must filter to one LLM
python scripts/render_prompts.py --series-id series_abc --llm cydonia_heretic_24b

# Single scene only
python scripts/render_prompts.py --scene-id series_abc_scene_007
```

### Error cases (verbatim)

- `EngineError: No default template set for family '<X>'. Set families.yaml::<X>::default_template or pass --templates <path>.` — family has `default_template: null` and you didn't pass `--templates`.
- `EngineError: Template '<path>' belongs to family '<X>' but prompt is for family '<Y>'. Render mismatched families is not supported.` — family mismatch.
- `ERROR: --families requires --render-with-model <model_id>.` (rc=2)
- `ERROR: --render-with-model '<id>' belongs to family '<X>', not '<Y>'.` (rc=2)
- `ERROR: --models and --families are mutually exclusive.` (rc=2)
- `ERROR: <target_desc> has prompts from LLM(s): '<llm1>', '<llm2>'. Specify --llm <id> to choose which LLM's prompts to render.` (rc=2)

## 5. Authoring templates

See [`docs/COMFYUI_WORKFLOWS.md`](./docs/COMFYUI_WORKFLOWS.md) for the
4-node contract + optional refiner-pair contract + step-by-step
authoring guide.

In one paragraph: build the workflow in ComfyUI UI, `Save (API Format)`,
rename four node IDs to `positive_prompt` / `negative_prompt` /
`ksampler` / `empty_latent` (BOTH the top-level keys AND every reference
array), drop the file under `config/comfyui_workflows/templates/<family>/`,
optionally set as the family's `default_template` in `config/families.yaml`.

## 6. DB schema (9 tables)

| Table                | Purpose                                                                 |
| -------------------- | ----------------------------------------------------------------------- |
| `series`             | One row per planned image set (theme/mood/environment + LLM plan JSON)  |
| `scenes`             | Model-agnostic scene rows (pose / camera / lighting / etc.)             |
| `scene_facets`       | Per-(scene, family, llm) LLM expansion (booru_tags / scene_prose / structured tags) |
| `prompts`            | Per-(scene, target_kind, model, llm) composed text                      |
| `images`             | Per-render output metadata + quality score                              |
| `sets`               | Filter / scoring grouping for export                                    |
| `posts`              | Cross-platform post tracking (DA / Patreon / Fanvue)                    |
| `generation_memory`  | Anti-repetition tracker (themes / scenes / prompts)                     |
| `run_log`            | One row per render cycle (mode / status / duration / image counts)      |

## 7. Config files

| File                                              | Purpose                                                          |
| ------------------------------------------------- | ---------------------------------------------------------------- |
| `config/pipeline.yaml`                            | Run cadence, default model + style profile, mode_weights, ComfyUI / LLM endpoints |
| `config/families.yaml`                            | 6 families (sdxl / pony / illustrious / flux / flux2 / chroma); per-family prompt rules, negative axes, default_render_model, default_template |
| `config/models/*.yaml`                            | Per-checkpoint identity + resolution + license. After 2026-05-20: workflow-tuning fields (sampler / scheduler / steps / cfg / VAE / etc.) moved into the template JSONs |
| `config/llm_models.yaml`                          | LLM registry — registry id → ollama tag; `default_llm`; per-role routing in `pipeline.yaml::llm.routing` |
| `config/style_profiles.yaml`                      | Aesthetic intent profiles (lighting / color / camera bias)       |
| `config/categories.yaml`                          | Theme / niche / style categories + tier directives               |
| `config/prompt_vocabulary.yaml`                   | Abstract concept tag library; canonicalizer translates to family phrasing |
| `config/comfyui_workflows/templates/<family>/`    | Your external workflow templates                                 |

## 8. Troubleshooting

- **`No default template set for family 'X'.`** — set `families.yaml::X::default_template` or pass `--templates <path>`.
- **`Template '<path>' belongs to family 'X' but prompt is for family 'Y'.`** — your `--templates` path was under `templates/X/` but the prompt's model belongs to family Y. Use the right family directory.
- **`WorkflowTemplateError: missing required semantic node IDs: ['positive_prompt']`** — your template has numeric IDs; rename four nodes to the contract names (see `docs/COMFYUI_WORKFLOWS.md`).
- **`WorkflowTemplateError: refiner pair incomplete`** — either both `refiner_positive_prompt` AND `refiner_ksampler` must exist, or neither.
- **`Series has prompts from LLM(s): X, Y. Specify --llm`** — A/B series; pick which LLM to render.
- **DB-stale errors after vocab bumps** — wipe + re-init: `rm nsfw_pipeline.db && python scripts/init_db.py`.

## 9. Render output layout

```
output/
  └── <content_level>/
      └── <series_id>/
          └── <llm_id>/
              └── <target_id>/              # model id or family id
                  ├── images/
                  │   └── <prompt_id>_<comfy_filename>.png
                  ├── manifest.json
                  └── metadata.json
```

Per-LLM + per-target segmentation so multi-LLM A/B + model-vs-family
renders on the same series don't overwrite each other.

## 10. Tests

```bash
# Full suite
python -m pytest -q --ignore=scripts

# Workflow contract + refiner contract only
python -m pytest tests/test_workflow_builder_external_template.py tests/test_workflow_builder_refiner.py -q

# Anti-grid / anti-mirror regression (vocab v7)
python -m pytest tests/test_anti_grid_regression.py -q
```
