# PROJECT GUIDE — Operations Manual

> Operations manual. System design lives in
> [`ARCHITECTURE.md`](./ARCHITECTURE.md); Claude Code rules live in
> [`CLAUDE.md`](./CLAUDE.md). This file is the end-to-end command
> reference — every script, every flag, a combined-parameter example
> for each, and a matrix of which flags accept multiple values.

---

## 1. Prerequisites

- Python 3.11+
- ComfyUI running at `http://127.0.0.1:8188` (dev box path: `~/AI/apps/ComfyUI/`)
- Ollama running at `http://localhost:11434` with `dolphin-mixtral:8x7b`
- Mac M4 Pro 48 GB (or a Linux box with ≥24 GB VRAM + 32 GB RAM)

```bash
pip install -r requirements.txt
ollama pull dolphin-mixtral:8x7b
```

### Image scorer weights

```bash
mkdir -p models/aesthetic
curl -L -o models/aesthetic/sac+logos+ava1-l14-linearMSE.pth \
  "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth"
```

`insightface/buffalo_l` auto-downloads to `~/.insightface/models/` on
first scorer run.

---

## 2. First-time setup

Since the 2026-04 refactor, all static config (families, models, style
profiles, categories) lives in `config/*.yaml`. The DB holds runtime
state only — no seed step.

```bash
python scripts/init_db.py                        # create nsfw_pipeline.db
python scripts/list_models.py                    # confirm 17 models load
python scripts/bootstrap_character.py characters/char_001/identity.json
python scripts/test_comfyui.py                   # 1 smoke render
python scripts/test_llm.py --skip-agents         # Ollama connectivity
```

Edit `config/pipeline.yaml` if your ComfyUI install or Ollama URL
differs. Key keys: `comfyui.output_dir`, `comfyui.input_dir`,
`llm.base_url`, `compliance.commercial_mode` (see §11).

---

## 3. Architecture in 60 seconds

| Layer | Count | Source of truth |
|---|---|---|
| Runtime tables | 10 | `scripts/init_db.py` schema (characters, series, scenes, scene_facets, prompts, images, sets, posts, generation_memory, run_log) |
| Prompt families | 6 | `config/families.yaml` (sdxl, pony, illustrious, flux, chroma, flux2) |
| Prompt composers | 5 | `src/prompt/builder.py` (sdxl_keywords, pony_danbooru, illustrious_tags, flux_natural, flux2_prose) |
| Vocabulary concepts | ~50 | `config/prompt_vocabulary.yaml` v2 — realism: lighting (13), camera (7), lens (7), film_stock (7), art_style (8), mood (8); nsfw: anatomy/posture (T3-gated, 6) + act (T4-gated, 5) |
| Model YAMLs | 17 | `config/models/*.yaml` |
| Style profiles | 10 | `config/style_profiles.yaml` |
| Content tiers | 4 | `T1_suggestive`, `T2_implied`, `T3_artnude`, `T4_explicit` |
| Workflow templates | 13 | `config/comfyui_workflows/{family}/{name}.json` (9 base/ipadapter + 3 upscale + 1 external) |
| Test suite | 896 | `tests/` — Phase 4 + 4-bis + 2026-05-02 NSFW-output-path fix (content_level + tier-directive surfacing, nsfw_act schema completion, tier-aware vocab block, T3/T4 boost_keyword cleanup); smoke tests under `scripts/` skipped by default |

Deep dive: `ARCHITECTURE.md §16` (file layout) and `§§5–14`
(tiers, families, composers, scoring).

---

## 4. Character bootstrap

### `characters/{id}/identity.json` schema

Required: `gender`, `age_appearance`, `face`, `hair`, `body_type`.
Optional (flow into `base_prompt`): `distinguishing_features`, `vibe`.
Other fields (`name`, `reference_image_path`, `locked_features`,
`allowed_shift_axes`) are informational.

```json
{
  "name": "char_001",
  "gender": "female",
  "age_appearance": "late 20s",
  "face": "oval face, sharp jawline, almond eyes, defined cheekbones",
  "hair": "long dark brown wavy hair",
  "body_type": "slim athletic",
  "distinguishing_features": "light freckles across nose",
  "vibe": "soft elegant",
  "reference_image_path": "characters/char_001/reference.png",
  "locked_features": {"eye_color": "hazel", "skin_tone": "warm fair"},
  "allowed_shift_axes": ["outfit", "pose", "expression", "lighting", "background"]
}
```

### `bootstrap_character.py`

```bash
# Minimum
python scripts/bootstrap_character.py characters/char_001/identity.json

# Combined — every flag set
python scripts/bootstrap_character.py characters/char_001/identity.json \
  --db-path nsfw_pipeline.db \
  --style-profile boudoir_noir \
  --model-id juggernaut_ragnarok \
  --character-id char_001 \
  --force
```

| Flag | Required | Repeats | Notes |
|---|---|---|---|
| `identity_path` (positional) | yes | no | path to identity.json |
| `--db-path` | no | no | default: `pipeline.db_path` |
| `--style-profile` | no | no | id from `config/style_profiles.yaml`; default `golden_hour_natural` |
| `--model-id` | no | no | id from `config/models/*.yaml`; default NULL (falls back to `pipeline.default_model_id`) |
| `--character-id` | no | no | default: `identity['name']` |
| `--force` | no | no | delete-and-reinsert (bypasses `base_prompt` immutability) |

`base_prompt` is immutable via DB trigger — `--force` is the only way
to re-bootstrap a character.

### `map_reference.py` — point a character at its reference image

```bash
python scripts/map_reference.py \
  --character-id char_001 \
  --image characters/char_001/reference.png \
  --db-path nsfw_pipeline.db
```

| Flag | Required | Repeats | Notes |
|---|---|---|---|
| `--character-id` | yes | no | existing row |
| `--image` | yes | no | absolute or project-relative path |
| `--db-path` | no | no | default: `pipeline.db_path` |

---

## 5. Rendering commands

### 5.1 `dry_run.py` — LLM planning only, no render

```bash
# Minimum (weighted-random mode, T2_implied)
python scripts/dry_run.py

# Combined — every flag set
python scripts/dry_run.py \
  --mode character \
  --character char_001 \
  --level T2_implied \
  --style-profile golden_hour_natural \
  --model chroma_v10HD \
  --verbose
```

Or one-shot a theme-mode dry run at a specific tier:

```bash
python scripts/dry_run.py --mode theme --level T3_artnude
```

| Flag | Required | Repeats | Notes |
|---|---|---|---|
| `--mode` | no | no | `character` / `theme` / `style` / `niche` / `variation`; default weighted-random |
| `--character` | no | no | character-mode only; default LRU pick |
| `--level` | no | no | `T1_suggestive` / `T2_implied` / `T3_artnude` / `T4_explicit`; default `T2_implied` |
| `--style-profile` | no | no | default `pipeline.default_style_profile_id` |
| `--model` | no | no | default `pipeline.default_model_id` |
| `--verbose` / `-v` | no | no | DEBUG logging |

Ollama must be up. No DB writes, no ComfyUI call.

### 5.2 `render_set.py` — manual Mode-1 renderer (primary entry)

Hand-written scenes + PromptBuilder + sanitizer + ComfyUI + scorer.
No LLM. IPAdapter is opt-in via `--reference-image`.

```bash
# Minimum
python scripts/render_set.py --character char_001 --level T2_implied

# Combined — every flag set
python scripts/render_set.py \
  --character char_001 \
  --theme "vintage pinup" \
  --level T2_implied \
  --count 5 \
  --ratio portrait_23 \
  --model juggernaut_ragnarok \
  --reference-image characters/char_001/reference.png \
  --ipadapter-weight 0.70 \
  --db-path nsfw_pipeline.db \
  --timeout 600 \
  --dry-run \
  --no-score
```

| Flag | Required | Repeats | Notes |
|---|---|---|---|
| `--character` | yes | no | must exist in DB |
| `--level` | yes | no | 4-tier set |
| `--theme` | no | no | free-form label stored on `series.theme`; default `manual baseline` |
| `--count` | no | no | default 15 |
| `--ratio` | no | no | `portrait_23` / `portrait_916` / `square` / `landscape`; default = per-scene auto |
| `--model` | no | no | overrides character/style-profile default; uses that model's YAML tuning |
| `--reference-image` | no | no | switches template to `{family}/ipadapter.json` (sdxl / pony / illustrious only); requires `supports_ipadapter: true` on the model — flux / chroma / flux2 raise |
| `--ipadapter-weight` | no | no | 0.4–1.0; pairs with `--reference-image` |
| `--template` | no | no | path to external workflow (e.g. `templates/chroma/chroma_done_properly.json`); forces IPAdapter off, injects only prompt/negative/seed/resolution — see `docs/COMFYUI_WORKFLOWS.md § External templates` |
| `--db-path` | no | no | default: `pipeline.db_path` |
| `--timeout` | no | no | per-render seconds; default `comfyui.render_timeout_seconds` |
| `--dry-run` | no | no | print prompts, skip ComfyUI + DB |
| `--no-score` | no | no | skip aesthetic scorer |

Output: `output/{level}/{series_id}/images/`.

External template example:

```bash
python scripts/render_set.py \
  --character char_001 --level T2_implied \
  --model chroma_v10HD \
  --template templates/chroma/chroma_done_properly.json \
  --count 5
```

### 5.3 `prepare_prompts.py` — Phase A only (LLM → DB; no render)

Generates a series + scenes + per-(scene, family) facets + per-model
prompts; persists to DB; **no ComfyUI calls**. Pairs with
`render_prompts.py` for the second half of the cycle.

Use this when you want to:
- Generate prompts for one or more models without rendering yet
  (e.g. supervised mode lets you eyeball the plan before GPU time).
- **Re-target** an existing series for a new model (`--series-id`
  skips planning + scene generation; reuses scenes; generates the
  new family's facets if missing; composes per-model prompts).
- Re-roll family-shaped facets (`--regen-facets`) or per-model
  prompts (`--regen-prompts`) without affecting other models.

```bash
# Fresh series for one model (default = pipeline.default_model_id)
python scripts/prepare_prompts.py --mode character --level T2_implied

# Multi-model fan-out (sibling-family models share the family's facet row)
python scripts/prepare_prompts.py --character char_001 --level T3_artnude \
  --models gonzalomo_photo_v70,chroma_v10HD

# Re-target an existing series for a new model
python scripts/prepare_prompts.py --series-id ser_abc \
  --models gonzalomo_flux_v30

# Re-roll the SDXL facets + gonzalomo_photo_v70 prompts on an existing series
python scripts/prepare_prompts.py --series-id ser_abc \
  --models gonzalomo_photo_v70 --regen-facets sdxl --regen-prompts gonzalomo_photo_v70
```

| Flag | Required | Repeats | Notes |
|---|---|---|---|
| `--mode` | no | no | character / theme / style / niche / variation; default weighted-random |
| `--character` | no | no | character id for character mode |
| `--level` | no | no | 4-tier set; default `T2_implied` |
| `--style-profile` | no | no | default `pipeline.default_style_profile_id` |
| `--models` | no | **values** | comma-separated; default `[pipeline.default_model_id]`. Sibling-family models share `scene_facets` rows. |
| `--series-id` | no | no | re-target an existing series (skip planning + scene-gen) |
| `--regen-facets` | no | **values** | comma-separated family ids; DELETE existing facet rows before regenerating |
| `--regen-prompts` | no | **values** | comma-separated model ids; DELETE existing prompts before re-composing |
| `--verbose` / `-v` | no | no | DEBUG logging |

Exit codes: `0` success · `1` engine error · `2` duplicate-prompt
collision (use `--regen-prompts`) · `3` supervisor abort · `4` interrupt.

Stdout summary:
```
Phase A complete in 1m 47s — status: complete
  Series:           ser_xyz (new)
  Scenes created:   20
  Facets created:   40
  Prompts inserted: 40
  Models completed: gonzalomo_photo_v70, chroma_v10HD

Next:
  python scripts/render_prompts.py --series-id ser_xyz --models gonzalomo_photo_v70,chroma_v10HD
```

### 5.4 `render_prompts.py` — Phase B + C only (render existing prompts)

Renders prompts already in the DB for one or more target models.
Pairs with `prepare_prompts.py`. Loads prompts by `(series_id, model_id)`,
renders each, scores, packages, exports.

```bash
# Whole series, single model
python scripts/render_prompts.py --series-id ser_abc --models gonzalomo_photo_v70

# Single scene, single model
python scripts/render_prompts.py --scene-id ser_abc_scene_003 \
  --models gonzalomo_photo_v70

# Multi-model fan-out (sequential — one ComfyUI checkpoint at a time)
python scripts/render_prompts.py --series-id ser_abc \
  --models gonzalomo_photo_v70,chroma_v10HD,gonzalomo_flux_v30

# Per-model templates (positional pairing: model[i] uses template[i])
python scripts/render_prompts.py --series-id ser_abc \
  --models gonzalomo_photo_v70,chroma_v10HD \
  --templates system,templates/chroma/chroma_done_properly.json
```

| Flag | Required | Repeats | Notes |
|---|---|---|---|
| `--series-id` | XOR `--scene-id` | no | render whole series |
| `--scene-id` | XOR `--series-id` | no | render single scene |
| `--models` | no | **values** | comma-separated; default `[pipeline.default_model_id]` |
| `--templates` | no | **values** | comma-separated; positional pair with `--models`; `system` = built-in |
| `--no-export` | no | no | skip Phase C (set build + watermark + export) |
| `--db-path` | no | no | default: `pipeline.db_path` |
| `--verbose` / `-v` | no | no | DEBUG logging |

**Pre-flight validation**: counts prompts in the DB for every requested
model BEFORE any expensive ComfyUI work. If any model has 0 prompts,
the script aborts (exit 2) with a hint pointing at `prepare_prompts.py`.

Exit codes: `0` all models complete · `1` any model failed ·
`2` missing prompts · `3` supervisor abort · `4` interrupt.

### 5.5 `run_once.py` — full cycle (plan → render → package), single model

Internally composes `run_phase_a([model]) → run_phase_b → run_phase_c`
(see ARCHITECTURE.md §3). No external behavior change vs. pre-Phase-3.
Use `prepare_prompts` + `render_prompts` for multi-model production
runs; `run_once` is the convenience wrapper for single-model end-to-end.

Ollama **and** ComfyUI must both be up; engine unloads Ollama before
calling ComfyUI (48 GB budget).

```bash
# Minimum
python scripts/run_once.py

# Combined — every flag set
python scripts/run_once.py \
  --mode character \
  --level T3_artnude \
  --style-profile boudoir_noir \
  --model juggernaut_ragnarok \
  --verbose
```

Or run a theme-mode cycle on a specific style profile + model:

```bash
python scripts/run_once.py \
  --mode theme \
  --level T3_artnude \
  --style-profile golden_hour_natural \
  --model gonzalomo_photo_v70
```

| Flag | Required | Repeats | Notes |
|---|---|---|---|
| `--mode` | no | no | same enum as `dry_run.py` |
| `--level` | no | no | 4-tier set; default `T2_implied` |
| `--style-profile` | no | no | default from `pipeline.yaml` |
| `--model` | no | no | default from `pipeline.yaml` |
| `--template` | no | no | path to external workflow; forces IPAdapter off, injects only prompt/negative/seed/resolution — see `docs/COMFYUI_WORKFLOWS.md § External templates` |
| `--verbose` / `-v` | no | no | DEBUG logging |

Note: `run_once.py` has **no** `--character` flag — character pick is
LRU. Use `render_set.py` for hand-written scenes against a specific
character, or the `prepare_prompts.py` + `render_prompts.py` pair when
you want full control over series id, model fan-out, or re-targeting.

External template example:

```bash
python scripts/run_once.py \
  --mode theme --level T3_artnude \
  --style-profile boudoir_noir \
  --model chroma_v10HD \
  --template templates/chroma/chroma_done_properly.json \
  --verbose
```

---

## 6. Model tools

### 6.1 `list_models.py` — print the registry

```bash
python scripts/list_models.py                    # all active
python scripts/list_models.py --family sdxl      # filter (sdxl|pony|illustrious|flux|chroma|flux2)
python scripts/list_models.py --all              # include active: false
```

### 6.2 `compare_models.py` — same prompt (or stored series/scene) across N models

**4-way mutually exclusive** input mode:
- `--prompt "..."` — raw text; no LoRAs; same prompt for every model.
- `--character ID` — pulls base_prompt + style-profile LoRAs; same prompt for every model.
- `--series-id S` (Phase 5) — render every scene in S; **each model uses its own per-(scene, model) prompt from the DB**.
- `--scene-id Sc` (Phase 5) — render one scene across every model; same per-(scene, model) prompt lookup.

The `--series-id` / `--scene-id` paths require `prepare_prompts.py`
to have run first for every `--models` entry. Missing prompts → exit 2 with a hint.

```bash
# Raw-prompt mode (every model gets the same text)
python scripts/compare_models.py \
  --prompt "a confident woman, soft window light, 85mm lens" \
  --models juggernaut_ragnarok,gonzalomo_photo_v70,chroma_v10HD \
  --count 3 --seed 42 --ratio portrait_23
```

```bash
# Stored-scene mode — each model uses its own DB prompt for the scene
python scripts/compare_models.py \
  --scene-id ser_abc_scene_007 \
  --models gonzalomo_photo_v70,chroma_v10HD,gonzalomo_flux_v30 \
  --seed 42
```

```bash
# Stored-series mode — render every scene in the series across every model
python scripts/compare_models.py \
  --series-id ser_abc \
  --models gonzalomo_photo_v70,chroma_v10HD
```

```bash
# Character-mode variant — uses base_prompt + style-profile LoRAs
python scripts/compare_models.py \
  --character char_001 \
  --models juggernaut_ragnarok,gonzalomo_photo_v70 \
  --count 2 --seed 123
```

| Flag | Required | Repeats | Notes |
|---|---|---|---|
| `--prompt` | XOR | no | raw string; no LoRAs |
| `--character` | XOR | no | uses char row + style-profile LoRAs |
| `--series-id` | XOR | no | every scene in the series; per-model DB prompts |
| `--scene-id` | XOR | no | single scene across models; per-model DB prompts |
| `--models` | yes | **values** | comma-separated model ids |
| `--template` | XOR with `--templates` | no | one external template path |
| `--templates` | XOR with `--template` | **values** | comma-separated; see template rule below |
| `--count` | no | no | images per candidate (only used in `--prompt` / `--character` mode); default 3. Ignored for series/scene mode (count = number of stored prompts). |
| `--seed` | no | no | base seed; per-image = base+i; default fresh random |
| `--ratio` | no | no | force a single ratio (per-model resolution lookup); default 1024×1024 in prompt/character mode, scene's stored aspect_ratio in series/scene mode |
| `--db-path` / `--comfyui-url` / `--comfyui-output-dir` / `--timeout` | no | no | defaults from `pipeline.yaml` |

#### Template-vs-model rule (Phase 5 BREAKING CHANGE)

| `--models` len | `--templates` len | Behavior |
|---|---|---|
| N | (none) | each model uses its `system` template |
| N | 1 | broadcast — every model uses that one template |
| 1 | M | one model rendered M times, once per template |
| **N | N** | **NEW**: positional pairing — `model[i]` ↔ `template[i]` |
| N | M (mismatched, both >1) | error |

> **Breaking change**: `--models a,b --templates system,system` previously rendered 4 outputs (Cartesian product); under the Phase 5 rule it renders 2 (paired). Users who want Cartesian behavior should run `compare_models` once per template.

Output: `output/comparisons/{timestamp}/{model_id}__{slug}/{label}.png`
where `slug` is `system` for the built-in or the template filename stem
for externals (`__2`, `__3` suffixes on collision). `label` is the per-image
counter `00,01,...` in prompt/character mode, or the scene_id tail
(`007`, `008`, ...) in series/scene mode for traceability. Nothing
scored, nothing written to the DB.

External template example — one model, built-in vs one external:

```bash
python scripts/compare_models.py \
  --character char_001 \
  --models chroma_v10HD \
  --templates system,templates/chroma/chroma_done_properly.json \
  --count 2 --seed 42 --ratio portrait_23
```

---

## 7. IPAdapter workflow

### 7.1 `tests/integration/test_ipadapter.py` — A/B visual diagnostic

Renders each scene twice (WITH and WITHOUT IPAdapter) at the same
seed so any difference is the adapter, not RNG. Not scored, not
persisted. Lives under `tests/integration/` because it needs ComfyUI
running and a bootstrapped character — pytest skips it during the
hermetic unit-test run.

```bash
# Minimum (needs characters.reference_image_path set, or pass --reference-image)
python tests/integration/test_ipadapter.py

# Combined — every flag set
python tests/integration/test_ipadapter.py \
  --character char_001 \
  --reference-image characters/char_001/reference.png \
  --count 5 \
  --level T2_implied \
  --ipadapter-weight 0.75 \
  --db-path nsfw_pipeline.db \
  --timeout 600
```

| Flag | Required | Repeats | Notes |
|---|---|---|---|
| `--character` | no | no | default `char_001` |
| `--reference-image` | no | no | overrides `characters.reference_image_path` |
| `--count` | no | no | A/B pairs per run; default 5 |
| `--level` | no | no | 4-tier set; default `T2_implied` |
| `--ipadapter-weight` | no | no | default 0.7 |
| `--db-path` / `--timeout` | no | no | defaults from `pipeline.yaml` |

Output: `output/test/ipadapter_AB/{timestamp}/{no_ipadapter,with_ipadapter}/`.

---

## 8. Smoke tests

### 8.1 `test_comfyui.py` — 1 render round-trip

```bash
python scripts/test_comfyui.py \
  --comfyui-url http://127.0.0.1:8188 \
  --comfyui-output-dir ~/AI/apps/ComfyUI/output \
  --checkpoint juggernautXL_ragnarokBy.safetensors \
  --width 1024 --height 1024 \
  --seed 42 \
  --timeout 300
```

All flags optional; defaults from `pipeline.yaml`. Hardcoded SDXL test
prompt, no LoRAs.

### 8.2 `test_llm.py` — Ollama + agent modules

```bash
python scripts/test_llm.py                            # full run
python scripts/test_llm.py --skip-agents              # connectivity only
python scripts/test_llm.py --model dolphin-llama3:8b  # override model
python scripts/test_llm.py --character char_002       # series-planner test
```

| Flag | Required | Repeats | Notes |
|---|---|---|---|
| `--model` | no | no | default: `llm.model` |
| `--character` | no | no | default: `char_001` |
| `--skip-agents` | no | no | skip planner / scene / metadata agents |

### 8.3 `test_scorer.py` — aesthetic MLP + face + blur + resolution

```bash
python scripts/test_scorer.py output/test/some_image.png \
  --mlp-weights models/aesthetic/sac+logos+ava1-l14-linearMSE.pth \
  --device mps
```

| Flag | Required | Repeats | Notes |
|---|---|---|---|
| `image_path` (positional) | yes | no | image to score |
| `--mlp-weights` | no | no | override LAION checkpoint path |
| `--device` | no | no | `cpu` / `mps` / `cuda`; auto-detect default |

---

## 9. DB + workflow maintenance

### 9.1 `init_db.py` — fresh database

```bash
python scripts/init_db.py                          # create
python scripts/init_db.py --db-path custom.db      # custom location
python scripts/init_db.py --force                  # drop + recreate
```

9 tables, indexes, `base_prompt` immutability trigger. No seed rows.

The project has no migration scripts — schema changes are made
directly to `init_db.py`'s `SCHEMA_SQL` block, and existing databases
are re-initialized (`python scripts/init_db.py --force`). Stable
release (post-v1) will reintroduce migrations; until then, treat the
DB as disposable.

### 9.2 `rename_workflow_nodes.py` — rebuild a template

Takes a ComfyUI "Save (API Format)" export, renames numeric node IDs
to semantic names via a YAML map. **All three flags are required.**

```bash
python scripts/rename_workflow_nodes.py \
  --input ~/Downloads/sdxl_base_api.json \
  --map config/workflow_node_maps/sdxl_base.yaml \
  --output config/comfyui_workflows/sdxl/base.json
```

Exit codes: `0` ok · `1` bad input · `2` bad map · `3` unmapped ids ·
`4` dangling `[node_id, slot]` refs. See `docs/COMFYUI_WORKFLOWS.md`
for the end-to-end workflow build guide.

### Inspecting the DB

```bash
sqlite3 nsfw_pipeline.db ".tables"
sqlite3 nsfw_pipeline.db "SELECT id, base_prompt FROM characters;"
sqlite3 nsfw_pipeline.db "SELECT count(*) FROM images;"
sqlite3 nsfw_pipeline.db "SELECT mode, status, images_generated, images_selected \
  FROM run_log ORDER BY created_at DESC LIMIT 10;"
```

---

## 10. FLUX.2 family (no models currently registered)

The `flux2` family infrastructure stays wired (builder, schema,
template, negative-axes, families.yaml entry) — but no flux2 model
YAMLs are currently in `config/models/`. When a flux2 checkpoint
(e.g. Klein 9B, or a future BFL release) is reintroduced, drop the
YAML into `config/models/` and the registry will pick it up.

### Distilled-contract clamps

Distilled flux2 checkpoints (Klein 9B is the canonical example, but
any step- or guidance-distilled member of the family will share the
profile) MUST run with **cfg=1.0, steps≤6 (clamps to 4),
sampler=euler, scheduler=simple**. `_build_flux2` warn-and-clamps any
override back to these — protects against accidental melt.

### License gate

Any FLUX NCL-licensed checkpoint MUST declare
`license: flux_ncl` + `commercial_use: false` in its YAML. For paid-
tier workflows (DA Premium / Patreon / Fanvue):

```yaml
# config/pipeline.yaml
compliance:
  commercial_mode: true
```

When `true`, the registry refuses to register any NCL model at
startup (`ModelNotFound` on any attempt to resolve it).

### Prompting style (`flux2_prose`)

Prose only, BFL 5-anchor order (subject → setting → details →
lighting → atmosphere), 30–80 words. No comma tag lists, no weighting
syntax, no `BREAK`. Trigger phrases (e.g. Ultra Real's
`"This is a high-quality photograph of"`) are surfaced via
`prompt.extend.trigger_words` — the LLM may place them, not the
composer.

---

## 11. Testing

```bash
pytest tests/ -v                                       # 896 collected (post 2026-05-02 NSFW-output-path fix)
pytest tests/test_workflow_builder.py -v               # one file
pytest tests/ -v -k "flux2"                            # keyword filter
pytest tests/ -v -k "commercial_gate or lora_stack"    # boolean filter
pytest tests/ -v -k "vocabulary"                       # Phase 4a vocab library + canonicalizer
```

The `tests/` directory covers: composers, negatives, dedup, family
loader, model registry (commercial-mode gate + LoRA stack parse),
ratio selector + multi-axis scoring, safety sanitizer, tier sanitizer,
**Scene + SceneFacet Pydantic schemas (per-family dispatcher,
Phase 4b model_validator invariants)**, style profile loader, workflow
builder (all 6 families incl. flux2 clamps), token-budget enforcement
(positive **and negative** sides — Phase 3a), upscale workflow shape,
HPS v2 + ImageReward composite, **engine phase split
(`run_phase_a/b/c`)**, **`_save_dry_run` model_id persistence**,
**`OllamaClient.generate_json(schema=...)` Pydantic path + Ollama
`format:schema` constrained decoding (Phase 4b)**, **`scene_facets`
repo CRUD (incl. Phase 4a structured enum-tag columns)**,
**`SceneFacetGenerator` per-family LLM dispatch**, **realism +
NSFW vocabulary library + canonicalizer (Phase 4a)**, **per-family
adult_anchor + structure_intro (Phase 3b/3c)**, **negative TI
embeddings as first-class YAML field (Phase 1) — incl. 3-form
normalization sugar (bare name / name:weight / canonical
embedding:Foo)**, **per-family LoRA stack support (Phase 2)**, **PNG
parameters + nsfw_pipeline metadata chunks (Phase 4b)**,
**prepare_prompts CLI** (collision + multi-model + regen flags),
**render_prompts CLI** (XOR validation, missing-prompts hint,
multi-model loop, template positional pairing), and the Phase H
regression harness.

No network, no ComfyUI, no Ollama required — tests run cold.

### Regression harness (Phase H)

`tests/test_regression_render_smoke.py` pins prompt composition + ratio
selection + token counts against frozen fixtures (3 cases per family
× 6 families = 18 cases). It catches drift in any of:

* Composer changes (`src/prompt/builder.py`).
* Family YAML changes (`config/families.yaml` — quality prefix/suffix,
  negative axes, avoid_words, tokenizer_id, max_tokens).
* Per-model YAML changes (`config/models/*.yaml`).
* `config/ratio_signals.yaml` (audience/composition/family bonuses).
* Tokenizer backend changes (`src/prompt/tokenizer.py`).

When you intentionally change one of those, expect ~3 fixtures (one
per content tier × the affected family) to fail. Re-bake the baseline:

```bash
python scripts/regenerate_regression_fixtures.py            # write all
python scripts/regenerate_regression_fixtures.py --check    # dry-run
python scripts/regenerate_regression_fixtures.py --family sdxl
git diff tests/fixtures/regression/                         # review
```

Fixture pairs live at
`tests/fixtures/regression/{family}/case_{1..3}.input.yaml` (hand-
authored, version-controlled) +
`tests/fixtures/regression/{family}/case_{1..3}.expected.yaml`
(regenerated; reviewed via `git diff` before commit).

---

## 12. Multi-parameter matrix (cheat-sheet)

| Script | Required | XOR | Repeats within one flag |
|---|---|---|---|
| `bootstrap_character.py` | `identity_path` | — | none |
| `map_reference.py` | `--character-id`, `--image` | — | none |
| `dry_run.py` | — | — | none |
| `render_set.py` | `--character`, `--level` | — | none |
| `prepare_prompts.py` (Phase 4) | — | — | `--models` / `--regen-facets` / `--regen-prompts` (CSV) |
| `render_prompts.py` (Phase 4) | one of `--series-id` / `--scene-id` | `--series-id` XOR `--scene-id` | `--models` / `--templates` (CSV; positional pairing) |
| `run_once.py` | — | — | none |
| `list_models.py` | — | — | none |
| `compare_models.py` | `--models` | `--prompt` XOR `--character` XOR `--series-id` XOR `--scene-id` | `--models` / `--templates` (CSV; positional pairing for N==N) |
| `tests/integration/test_ipadapter.py` | — | — | none |
| `test_comfyui.py` | — | — | none |
| `test_llm.py` | — | — | none |
| `test_scorer.py` | `image_path` | — | none |
| `init_db.py` | — | — | none |
| `regenerate_regression_fixtures.py` | — | — | none |
| `rename_workflow_nodes.py` | `--input`, `--map`, `--output` | — | none |

Several flags accept multi-values now (post-Phase-4 / Phase-5):
`compare_models.py --models a,b,c`, `prepare_prompts.py --models`/
`--regen-facets`/`--regen-prompts`, `render_prompts.py --models`/
`--templates`. All other
flag is single-valued.

---

## 13. Config reference

### `config/pipeline.yaml`

| Key | Controls |
|---|---|
| `pipeline.db_path` | SQLite path (default `nsfw_pipeline.db`) |
| `pipeline.output_dir` | set-export root |
| `pipeline.default_model_id` | fallback when no character model / CLI override (`juggernaut_ragnarok`) |
| `pipeline.default_style_profile_id` | fallback for non-character modes (`golden_hour_natural`) |
| `mode_weights` | character/theme/style/niche/variation probability |
| `content_level_weights` | T1 / T2 / T3 / T4 probability |
| `aspect_ratio_weights` | per-tier ratio probability |
| `set_builder.quality_cutoff` | min composite score (0.55) |
| `scoring.use_hps_v2` | Phase G — flip on to enable HPS v2 (~600MB model, opt-in install via `pip install hpsv2`) |
| `scoring.use_image_reward` | Phase G — flip on to enable BAAI ImageReward (~400MB model, opt-in via `pip install image-reward`) |
| `scoring.composite_weights` | 6-signal weights when either Phase-G flag on (defaults: hps=0.30, image_reward=0.25, aesthetic=0.20, face=0.10, blur=0.10, resolution=0.05) |
| `scoring.legacy_weights` | 4-signal fallback weights when both Phase-G flags off (0.40 / 0.25 / 0.25 / 0.10) |
| `llm.base_url` / `llm.unload_after_phase` / `llm.keep_alive_seconds` | Ollama transport config |
| `llm.routing` | Per-role LLM routing. Disabled by default (2026-05-18) — every role falls through to `default_llm` (`cydonia_heretic_24b`). Set per-role entries here if a future evaluation shows a specific role benefits from a different LLM. See §16 |
| `comfyui.base_url` / `output_dir` / `input_dir` / `render_timeout_seconds` / `workflow_dir` | ComfyUI config |
| `watermark.*`, `postprocess.*`, `variation_mode.*` | tier-export + Phase-2/4 knobs |
| `postprocess.upscale_enabled` | Phase F — pure-ESRGAN upscale (sdxl/pony/illustrious only); raises eagerly for untemplated families |
| `compliance.commercial_mode` | drop NCL-licensed models at registry load (§10) |

### `config/families.yaml`

Six families: `sdxl`, `pony`, `illustrious`, `flux`, `chroma`, `flux2`.
Each declares `prompt_style` (one of `sdxl_keywords`, `pony_danbooru`,
`illustrious_tags`, `flux_natural`, `flux2_prose`), quality
prefix/suffix, **`structure_intro`** (Phase 3c — comma-tokens emitted
between quality_prefix and body; Pony realism finetunes use
`[source_photograph, "photo (medium)", realistic]`),
**`adult_anchor`** (Phase 3b — `{keyword, prose}` injection text for
the positive-side age-safety scan; Pony overrides to
`1girl, mature_female, adult` since `1girl` + `mature_female` is the
canonical Danbooru adult-female pair (`1woman` is not a real tag)
verbatim), negatives (7-axis taxonomy), clip_skip, max_tokens,
structure rules, avoid words, LLM hints, capability flags.

### `config/prompt_vocabulary.yaml` (Phase 4a + 4-bis, vocab_version 2)

Versioned realism + NSFW concept library. Top-level `version:` integer
is captured in `prompts.vocab_version` per row.

**Two namespaces:**

* `realism.{lighting, camera, lens, film_stock, art_style, mood}` —
  always-on. ~50 concepts at version 2. Phase 4-bis broadened the
  initial Phase 4a set (added LIGHT_SPLIT / LIGHT_CANDLELIGHT /
  LIGHT_BLUE_HOUR; CAMERA_PHASE_ONE_IQ4 / CAMERA_PENTAX_67_FILM;
  LENS_100MM_MACRO / LENS_70_200_F28; FILM_VELVIA_50 /
  FILM_KODAK_VISION3; ART_HELMUT_NEWTON / ART_HERB_RITTS_BW /
  ART_IRVING_PENN_MINIMALISM; MOOD_SENSUAL / MOOD_SERENE /
  MOOD_MELANCHOLIC).
* `nsfw.{anatomy, posture, act}` — tier-gated. Phase 4a shipped
  T3_artnude anatomy + posture (NSFW_BREAST_NATURAL / NSFW_HIPS_THIGHS
  / NSFW_GLUTES / NSFW_RECLINED_NUDE / NSFW_KNEELING_NUDE /
  NSFW_INTIMATE). Phase 4-bis added the **T4_explicit `act` namespace**
  (NSFW_T4_EMBRACE_NUDE / NSFW_T4_KISS_PASSIONATE / NSFW_T4_SOLO_TOUCH
  / NSFW_T4_PARTNERED_INTIMATE / NSFW_T4_AFTERGLOW). The canonicalizer
  silently drops below-tier concepts; phrasing is anatomically focused
  and tastefully worded across every prose family.

The LLM emits abstract concept tags from this menu (e.g.
`LIGHT_REMBRANDT`, `CAMERA_85MM_F14`, `FILM_PORTRA_400`,
`NSFW_T4_EMBRACE_NUDE`); the canonicalizer in
`src/prompt/vocabulary.py` translates each tag into family-shaped
phrasing at compose time. Pony omits camera / lens / film_stock /
art_style namespaces — booru tagging carries those implicitly via
`source_photograph + booru_tags`.

### `config/models/{id}.yaml`

Per-model registry row. Required: `id`, `display_name`, `filename`,
`architecture`, `family`, `default_sampler`, `default_scheduler`,
`default_steps`, `default_cfg`. Optional: `resolution_*`,
`vae_filename`, `text_encoder`, `supports_ipadapter`, `supports_lora`,
`license`, `commercial_use`, `active`, `notes`, `lora_stack` (max 2
enabled — generalized across all LoRA-supporting families in Phase 2),
`prompt.extend` / `prompt.override` (merged over the family). Phase 1
adds **`prompt.extend.negative_embeddings:`** (typed list of TI
embedding tokens — accepts bare names like `epiCNegative`, weighted
sugar like `Foo:0.8`, or canonical `embedding:Bar` form). Phase 3c
adds **`prompt.override.structure_intro:`** (Pony realism finetunes
use this to pin source_photograph + realistic phrasing in the
leading window).

### `config/style_profiles.yaml`

10 aesthetic archetypes — `boudoir_noir`, `old_hollywood_glamour`,
`golden_hour_natural`, `cinematic_wet_set`, `fine_art_figurative`,
`vintage_pinup_kodachrome`, `editorial_fashion_nude`, `moody_bw`,
`fantasy_castlecore`, `neo_noir_neon`. Each pins palette, lighting,
`suited_tiers`, `suited_families`, base keywords, and a base
negative. **No sampler/steps/cfg** — render tuning comes from the
model YAML.

### `config/categories.yaml`

`themes:`, `styles:`, `niches:`, `content_levels:` — used by
`dry_run.py --mode theme|style|niche` and by `PromptSanitizer` (tier
policies). Each theme/style/niche carries an `aesthetic_affinity`
list pointing at `style_profiles.yaml` ids.

### DB schema notes (per-model prompts work)

The 10 runtime tables live in `scripts/init_db.py::SCHEMA_SQL`. The
non-obvious columns + constraints worth knowing:

- `scenes` — model-agnostic scene core (pose / camera / lighting /
  env / mood / expression / composition_intent / framing_hint /
  audience_target).
- `scene_facets` — per-(scene, family) LLM expansion. PRIMARY KEY
  `(scene_id, family)`; the `family` CHECK clause is **templated
  from `config/families.yaml`** at init time so adding a family is a
  one-yaml-edit + DB re-init away. Phase 4a added 8 persistent
  enum-tag columns; the 2026-05-02 NSFW-output-path audit fix added
  a 9th (`nsfw_act`, T4-gated). Full set: `realism_camera`,
  `realism_lens`, `realism_film_stock`, `art_style_reference`,
  `lighting_directive`, `mood_aesthetic`, `nsfw_anatomy`,
  `nsfw_posture`, `nsfw_act`. Each holds an abstract concept tag
  from `config/prompt_vocabulary.yaml` that the composer
  canonicalises into family-shaped phrasing at compose time.
- `prompts` — per-(scene, model). `prompts.model_id` is **NOT NULL**;
  `UNIQUE(scene_id, model_id)` enforces "one prompt per (scene,
  model)". Re-rolling on the same model requires explicit
  `prepare_prompts.py --regen-prompts <model>` to DELETE + re-insert
  (otherwise IntegrityError surfaces with a helpful CLI hint).
  Phase 4a added `prompts.vocab_version` (INTEGER NOT NULL DEFAULT 1)
  — captures which `prompt_vocabulary.yaml` version produced the
  prompt so a YAML bump preserves audit trail for older rows.
- `images.model_id` — already there (was unaffected by the per-model
  prompts work); records which model rendered each file. Note that
  PNG output also carries the same reproduction parameters in two
  tEXt chunks (Phase 4b): `parameters` (AUTOMATIC1111 / Civitai
  format) and `nsfw_pipeline` (pipeline-native JSON).

### `config/comfyui_workflows/`

13 active templates (9 base/ipadapter + 3 upscale + 1 external):

| Template | Purpose |
|---|---|
| `sdxl/base.json` | standard SDXL t2i; `lora_loader_0/1` slots |
| `sdxl/ipadapter.json` | SDXL + IPAdapter unified loader / apply |
| `sdxl/upscale.json` | Phase F: pure ESRGAN 4× → 0.35 (1.4× source); `prefix=upscale_sdxl` |
| `pony/base.json` | Pony + `CLIPSetLastLayer` (clip_skip 2) |
| `pony/ipadapter.json` | Pony + IPAdapter |
| `pony/upscale.json` | Phase F: pure ESRGAN, `prefix=upscale_pony` |
| `illustrious/base.json` | Illustrious XL (SDXL-shaped) |
| `illustrious/ipadapter.json` | Illustrious + IPAdapter |
| `illustrious/upscale.json` | Phase F: pure ESRGAN, `prefix=upscale_illustrious` |
| `chroma/base.json` | GGUF Q8 + SamplerCustomAdvanced + beta scheduler |
| `flux/base.json` | FLUX.1 GGUF + ModelSamplingFlux + FluxGuidance |
| `flux2/base.json` | FLUX.2 family (no models currently registered) — UNETLoader + CLIPLoader(type=flux2) + KSampler @ cfg=1/steps=4 (clamped for distilled checkpoints) |
| `templates/chroma/chroma_done_properly.json` | community external template, used via `--template` flag |

Build process: export from ComfyUI UI as "Save (API Format)" →
`rename_workflow_nodes.py` to semantic ids (§9.4) → commit under
`config/comfyui_workflows/{family}/`. External (user/community)
templates live under `config/comfyui_workflows/templates/{family}/`
and are passed via `--template` rather than auto-loaded by family.

---

## 14. Troubleshooting

| Symptom | Fix |
|---|---|
| `Ollama not reachable` | `ollama serve`; `curl localhost:11434/api/tags` |
| `Checkpoint file not found` | preflight looks in a family-specific folder: SDXL / Pony / Illustrious → `models/checkpoints/`; Chroma & FLUX.1 GGUF → `models/unet/`; FLUX.2 → `models/diffusion_models/`. The error message prints the exact path it expected |
| `Workflow template not found` | the model's `family:` has no template under `config/comfyui_workflows/{family}/` |
| `base_prompt is immutable after creation` | rerun bootstrap with `--force` (DELETE + re-INSERT; trigger only fires on UPDATE) |
| `Model X does not support IPAdapter` | `supports_ipadapter: false` in that model's YAML; pick a different model or drop `--reference-image` |
| `style_profile_id is not in config/style_profiles.yaml` | `python -c "from src.memory.style_profile_loader import StyleProfileLoader; print([p.id for p in StyleProfileLoader().list_profiles()])"` |
| FLUX.2 output melted | something overrode the distilled contract; `_build_flux2` logs will show which field clamped. Don't pass `--model <flux2_id>` with a style-profile that ships cfg/steps overrides |
| `ModelNotFound` for an NCL-licensed model | `compliance.commercial_mode: true` dropped it at load; flip to `false` if this run isn't commercial |
| Memory error mid-render | Ollama + ComfyUI compete for 48 GB — engine unloads LLM; if it fails: `curl -X POST localhost:11434/api/generate -d '{"model":"dolphin-mixtral:8x7b","prompt":"","keep_alive":0}'` |
| `ComfyUI returned no images: every node was served from cache` | vary the seed or the prompt — identical inputs return cached output with no file |
| `HPS v2 disabled for the rest of this run` (or same for ImageReward) | `pip install hpsv2` (or `image-reward`); both flagged optional in `requirements.txt`. First call downloads ~600MB / ~400MB of weights. To opt out entirely: `scoring.use_hps_v2: false` / `use_image_reward: false` in `pipeline.yaml` |
| Phase G column missing (`hps_v2_score` / `image_reward_score`) on existing DB | Re-init: `python scripts/init_db.py --force` (the project has no migrations during pre-stable; see §9.1). Existing data is disposable until v1. |
| `Upscaler: family 'flux' not in _SUPPORTED_FAMILIES` | Phase F templates only ship for sdxl/pony/illustrious — flux/chroma/flux2 already render at 1024+. Either disable `postprocess.upscale_enabled`, or pick a templated family |
| `prepare_prompts.py: ERROR: prompts for ... already exist on series` | UNIQUE(scene_id, model_id, llm_id) blocks accidental double-insert. Re-run with `--regen-prompts <model_id> --llm <id>` to DELETE + re-compose for that specific (model, LLM) pair. Other LLMs' prompts on the same scenes are untouched. |
| `render_prompts.py: ERROR: series 'S' has prompts from LLM(s): ...` | Strict-ambiguity check (plan §3.5b) — when 1+ LLMs have prompts on the series, `--llm <id>` becomes required. Pass `--llm cydonia_heretic_24b` (or whichever LLM you want to render). |
| `render_prompts.py: ERROR: no prompts in DB for ... on series` | The model's prompts haven't been generated yet. Run `python scripts/prepare_prompts.py --series-id <S> --models <M>` first; the error message includes the exact command. |
| `LLM 'foo' is not in your registry` | `--llm` was passed an id not declared under `llms:` in `config/llm_models.yaml`. The error lists the available ids. Either fix the typo, or `ollama pull <tag>` and add the entry. |
| `pipeline.yaml::llm.routing.X -> 'Y' is not a valid active registry id` | Routing block references a missing or inactive LLM. Either remove the routing entry, mark `active: true` in the registry, or change the value to a valid id. Validation fires at engine startup so the typo never reaches the LLM call. |
| `compare_models.py: ERROR: --templates count must equal --models count` | Phase 5 changed N==N from Cartesian to positional pairing; mismatched counts (both >1) are now rejected. Either pass exactly N templates for N models, or pass 1 template (broadcast). |
| `SceneFacetGeneratorError: facet generation failed` | LLM returned malformed JSON or schema-invalid fields for the facet. Check Ollama logs. Re-roll with `prepare_prompts --regen-facets <family>` (the bad facet row will be DELETEd + regenerated). |
| **T4 series rendered tasteful boudoir, not actually NSFW/explicit** | 2026-05-17/18 — fully resolved by 5-layer fix + LLM swap: (1) `prepare_prompts.py --series-id` retarget now inherits `content_level` from DB instead of silently downgrading to T2_implied when `--level` is omitted; (2-3) theme/niche/style/character modes and SeriesPlanner/SceneGenerator now inject the rich `categories.yaml::content_levels.<tier>.llm_directive` into both plan and scene templates inside a `══ CONTENT TIER ══` banner; (4) `families.yaml` T4 few-shot exemplars rewritten across all 6 families with explicit anatomical vocabulary; (5) `scene_facet_generator.py` post-validation enforces tier-required NSFW fields — `nsfw_anatomy` at T3+, both `nsfw_anatomy + nsfw_act` at T4 — with retry-nudge then ship-with-warning if still missing. The 2026-05-18 LLM swap (default → `cydonia_heretic_24b`, a heretic-tuned refusal-removed variant) further reduces facet-side self-censoring at T3/T4. **For DBs created pre-2026-05-17:** drop stale T4 facets+prompts (`DELETE FROM scene_facets / prompts WHERE llm_id='<old_llm_id>'` on the affected series) and re-run `prepare_prompts.py --series-id <s> --regen-facets <fam> --regen-family-prompts <fam>` — facets regenerate via the new default LLM and populate the required NSFW tags. |
| `no such column: nsfw_act` | Your DB was created pre-2026-05-02. Run `python scripts/init_db.py --force` to add the column. Per project no-migration policy, existing series data is disposable. |

---

## 15. Project layout (pointers)

```
nsfw-pipeline/
├── config/
│   ├── pipeline.yaml              # 15-section runtime config
│   ├── families.yaml              # 6 families
│   ├── llm_models.yaml            # LLM registry (multi-LLM upgrade, 2026-05)
│   ├── style_profiles.yaml        # 10 aesthetic archetypes
│   ├── categories.yaml            # themes / styles / niches / 4-tier rules
│   ├── ratio_signals.yaml         # Phase A: audience/composition/family bonuses
│   ├── models/                    # 17 per-model YAMLs
│   ├── comfyui_workflows/         # 13 templates (see §13)
│   └── workflow_node_maps/        # inputs to rename_workflow_nodes.py
├── scripts/                       # 15 CLI entry points (see §§4–9)
│                                  # incl. prepare_prompts (Phase A) +
│                                  # render_prompts (Phase B+C) added in Phase 4
├── src/
│   ├── main.py                                # supervised/automated scheduler entry
│   ├── core/                      # engine (4 phase methods: run_cycle/_a/_b/_c),
│   │                                           # generation_context, ratio_selector,
│   │                                           # aspect_ratio_buckets (Phase A), content_level,
│   │                                           # mode_selector, merge_overrides, style_profile_adapter
│   ├── agents/                    # llm_client (Phase 4b: Ollama format:schema; multi-LLM upgrade
│   │                                           # 2026-05 added per-call model + unload_all + loaded_models),
│   │                                           # llm_router (multi-LLM upgrade — resolves role/family
│   │                                           # → registry id with override→routing→default chain),
│   │                                           # series_planner, scene_generator (model-agnostic core only),
│   │                                           # scene_facet_generator (Phase 2 — per-family LLM expansion),
│   │                                           # metadata_generator, character_creator,
│   │                                           # schemas (Pydantic: Scene + 5 SceneFacet* incl.
│   │                                           # Phase 4b model_validator family invariants)
│   ├── prompt/                    # builder (5 composers + canonicalizer thread),
│   │                                           # tokenizer (Phase C), negative_axes (Phase D),
│   │                                           # sanitizer, deduplicator,
│   │                                           # vocabulary (Phase 4a — concept-tag → family phrase translator)
│   ├── render/                    # comfyui_client, workflow_builder (incl. _build_flux2 clamp),
│   │                                           # metadata (Phase 4b — A1111 parameters + nsfw_pipeline PNG chunks)
│   ├── scoring/                   # image_scorer (Phase G: 6-signal composite, opt-in HPS+IR)
│   ├── memory/                    # character_manager, memory_manager, model_registry,
│   │                                           # llm_registry (multi-LLM upgrade, 2026-05 — YAML loader
│   │                                           # for config/llm_models.yaml + default_llm validation),
│   │                                           # family_loader, style_profile_loader, categories_loader,
│   │                                           # scene_facets_repo (Phase 1 — per-family facet CRUD;
│   │                                           # multi-LLM upgrade extended to per-(scene, family, llm_id))
│   ├── filter/                    # set_builder, level_purity_check
│   ├── modes/                     # base_mode + 5 mode implementations
│   ├── postprocess/               # watermarker (preserves PNG chunks), upscaler (Phase F), face_refiner
│   ├── export/                    # exporter
│   ├── review/                    # supervisor, contact_sheet
│   └── analytics/                 # placeholder (engagement loop pending)
├── tests/                         # 896 collected (post 2026-05-02 NSFW-output-path fix:
│                                  # content_level + tier-directive surfacing, nsfw_act
│                                  # schema completion, tier-aware vocab block, T3/T4
│                                  # boost_keyword cleanup), no network deps
│   ├── _regression_harness.py                 # Phase H shared compute_expected
│   ├── integration/test_ipadapter.py          # ComfyUI-required; pytest-skip by default
│   └── fixtures/regression/{family}/case_*    # Phase H baseline (6 families × 3 cases)
├── characters/                    # identity.json + reference images
├── models/aesthetic/              # LAION predictor weights
├── output/                        # renders, comparisons, IPAdapter A/B
│                                  # (multi-LLM: output/<level>/<series>/<llm_id>/images)
├── docs/COMFYUI_WORKFLOWS.md      # template build walkthrough
├── ARCHITECTURE.md                # system design
├── CLAUDE.md                      # Claude Code rules
└── PROJECT_GUIDE.md               # this file
```

---

## 16. Multi-LLM workflow (2026-05)

The pipeline supports multiple Ollama-installed LLMs via a registry at
`config/llm_models.yaml`. Each entry maps a stable registry id (used by
`--llm` flag, `pipeline.yaml::llm.routing`, and `prompts.llm_id`) to the
Ollama tag installed locally.

### 16.1 Two operating modes

**Single-default mode (shipped 2026-05-18).** `pipeline.yaml::llm.routing`
is empty by default — every role falls through to `default_llm`
(`cydonia_heretic_24b`, the heretic-tuned Cydonia 24B v4.3 variant).
The heretic tune already carries a low refusal floor, so the prior
per-role split is no longer needed. To switch LLMs for a single
command, pass `--llm <id>` on the CLI — that overrides every role
uniformly for the run. Recommended starting point.

```bash
# Default (every role uses cydonia_heretic_24b):
python scripts/prepare_prompts.py --character char_001 --level T4_explicit \
    --models gonzalomo_photo_v70

# Explicit default LLM (same as above; useful when documenting):
python scripts/prepare_prompts.py --character char_001 --level T4_explicit \
    --models gonzalomo_photo_v70 --llm cydonia_heretic_24b

# Override to Hermes 3 for every role (A/B test against smaller model):
python scripts/prepare_prompts.py --character char_001 --level T4_explicit \
    --models gonzalomo_photo_v70 --llm hermes3
```

**Per-role routing mode (opt-in tuning).** Edit
`pipeline.yaml::llm.routing` to send specific roles to specific
LLMs:

```yaml
llm:
  routing:
    scene_facet_generator:
      default:          cydonia_heretic_24b   # NSFW phrasing (default LLM)
      flux_natural:     hermes3                # smaller/faster for flux/chroma prose
      flux2_prose:      hermes3                # BFL 5-anchor prose
    metadata_generator: cydonia_heretic_24b   # platform-metadata role
    # series_planner / scene_generator / character_creator stay on default_llm
```

Each role automatically gets the best-fit LLM. `--llm` is reserved for
explicit A/B testing where you want a single LLM doing every role.

**Fallback retry.** `config/llm_models.yaml::fallback_llm: hermes3`
configures the second-chance retry inside `OllamaClient.generate_json`
— if the primary LLM (`cydonia_heretic_24b`) fails constrained
decoding twice in a row, the agent retries once with Hermes 3.
Different lineage (Llama 3.1 vs Mistral) gives meaningful diversity
on the retry. Setting `fallback_llm` equal to `default_llm` collapses
the fallback dimension (single-LLM operation).

### 16.2 A/B-compare flow (the headline use case)

Re-prompt the same series with a different LLM; both sets coexist on
the same scene rows.

```bash
# 1. Generate with Cydonia Heretic (default).
python scripts/prepare_prompts.py \
    --character char_001 --level T4_explicit --models gonzalomo_photo_v70

# 2. Re-prompt the same series with Hermes 3 (scenes reused; new
#    facets+prompts written for llm_id=hermes3).
python scripts/prepare_prompts.py \
    --series-id ser_xxx --models gonzalomo_photo_v70 --llm hermes3

# 3. Inspect the DB — both LLMs' prompts coexist:
sqlite3 nsfw_pipeline.db "SELECT llm_id, COUNT(*) FROM prompts \
    WHERE series_id='ser_xxx' GROUP BY llm_id;"
# cydonia_heretic_24b|25
# hermes3|25

# 4. Render each LLM separately. --llm is REQUIRED here (strict
#    ambiguity check, plan §3.5b) since both LLMs have prompts.
python scripts/render_prompts.py --series-id ser_xxx \
    --models gonzalomo_photo_v70 --llm cydonia_heretic_24b
python scripts/render_prompts.py --series-id ser_xxx \
    --models gonzalomo_photo_v70 --llm hermes3

# 5. Output paths disambiguate — each LLM gets its own subdir:
ls output/T4_explicit/ser_xxx/
# cydonia_heretic_24b/  hermes3/

# 6. Compare manually — pick whichever LLM produced better images.
```

Scene-level workflow is identical with `--scene-id` instead of
`--series-id`.

### 16.3 Adding a new LLM

```bash
# 1. Pull the Ollama tag.
ollama pull <ollama_tag>

# 2. Add an entry to config/llm_models.yaml under llms:
#    (mirror an existing entry; required keys: ollama_id, display_name)

# 3. (Optional) Set as default by editing default_llm at the bottom.

# 4. Use it: --llm <your_registry_id>
```

The registry validates at engine startup — a typo in `default_llm` or
in any `pipeline.yaml::llm.routing.*` target raises `LLMRegistryError`
before any agent fires. Inactive entries (`active: false`) are
visible in `list_llms(include_inactive=True)` but rejected by
`get_llm()` and routing.

### 16.4 Resolution-display logging

Every LLM-using CLI prints a resolution table to the log at run start
showing which LLM each role resolves to and the source of resolution
(`--llm override` / `routing` / `routing.default` / `default`). When
`--llm` overrides explicit routing, the table annotates each row with
"routing was: X" so the user sees what they overrode. Closes the
silent-misconfiguration class.

### 16.5 PNG metadata

Renders include the generating `llm_id` in the `nsfw_pipeline` tEXt
chunk so a forensic reader can identify the LLM from the PNG alone
(no DB access needed). The standard `parameters` (A1111) chunk is
unchanged — interop-safe.

### 16.6 Switching LLMs (cheat-sheet)

Each Phase A LLM call has a **role**. The active LLM for a role is
resolved in priority order:

1. `--llm <id>` on the CLI — single-run override for every role.
2. `pipeline.yaml::llm.routing.<role>` — per-role override.
3. `config/llm_models.yaml::default_llm` — global fallback.

Roles: `series_planner`, `scene_generator`, `scene_facet_generator`,
`character_creator`, `metadata_generator`.

| Goal | What to edit |
|---|---|
| Use a different model as the default everywhere | `config/llm_models.yaml::default_llm` |
| One-off run with a different LLM | `--llm <id>` on the CLI (no config change) |
| Route a specific family through a different facet LLM | `pipeline.yaml::llm.routing.scene_facet_generator.<prompt_style>: <id>` (e.g. `flux_natural: hermes3`) |
| Re-enable per-role routing | Add entries under `pipeline.yaml::llm.routing` (it's `{}` by default — every role currently falls through to `default_llm`) |
| Swap the fallback LLM | `config/llm_models.yaml::fallback_llm: <id>` (used after two consecutive primary failures) |
| Add a new model | `ollama pull <tag>`, add an entry to `config/llm_models.yaml::llms` |

The shipped default (2026-05-18) is **Cydonia 24B v4.3 Heretic
Vision (Q4_K_M)** for every role, with **Hermes 3 (latest)** as
the second-chance fallback. See §16.7 for why.

### 16.7 When to switch LLMs (and what breaks if you do)

- **Cydonia Heretic Vision 24B (Q4_K_M, ~14 GB) — default for
  every role.** Heretic-tuned (refusal-removed) variant of
  Cydonia 24B v4.3, built on Mistral Small 24B. Initiative-taking
  creative-writing layer + RP-tuned booru tag handling makes it
  a single LLM that competently serves all five Phase A roles
  (planner / scene-gen / facet / character / metadata) without
  the prior per-role split. The 2026-05-18 LLM swap collapsed
  routing entirely on this basis. Pull:
  `ollama pull Fermi/Cydonia-24B-v4.3-heretic-vision:Q4_K_M`.

- **Hermes 3 (latest tag, ~5 GB Q4_K_M) — fallback.** Nous
  Research Hermes 3 on Llama 3.1 (`:latest` defaults to the 8B
  variant). Smaller, faster, different lineage — a meaningful
  diversification target for `OllamaClient.generate_json`'s
  second-chance retry. Set as `fallback_llm` so the agent retries
  once with Hermes 3 if the primary fails constrained decoding
  twice in a row. Can also be the global default for batch /
  speed-sensitive runs via `--llm hermes3`. Pull:
  `ollama pull hermes3:latest`.

**Rule of thumb:** keep the default at `cydonia_heretic_24b` for
quality and `fallback_llm: hermes3` for diversity on retry.
Re-enable per-role routing in `pipeline.yaml::llm.routing` only
when a future evaluation shows a specific role benefits from a
different LLM — the heretic tune obsoletes the prior need for a
separate low-refusal model on facet generation.

## 17. Family-level prompt preparation (2026-05)

Prepare prompts at the **family** level (sdxl / pony / illustrious /
flux / chroma / flux2) without applying any per-model rules.
A family-level prompt is checkpoint-agnostic — any model in that
family can render it. Useful when:

- You want to compose prompts once and render them through several
  sibling checkpoints in the same family (e.g. compare juggernaut_ragnarok
  vs gonzalomo_photo_v70 on identical SDXL text).
- You want clean family-level baseline prompts without trigger words,
  per-model LoRAs, or model-specific negative embeddings.

Model-level and family-level prompts coexist on the same scene; the
schema discriminates via `prompts.target_kind` ∈ {`model`, `family`}.
The `scene_facets` row is shared — facet generation (the expensive
LLM hop) runs once per (scene, family, llm) regardless of how many
target_kinds consume it.

### 17.1 Flag matrix

| CLI | Flag | Behaviour |
|---|---|---|
| `prepare_prompts.py` | `--models <m1,m2>` | Per-model rules (trigger / avoid / LoRAs). target_kind='model'. |
| `prepare_prompts.py` | `--families <f1,f2>` | Family-level only — no per-model overlay. target_kind='family'. |
| `prepare_prompts.py` | both | Both kinds prepared on same series; coexist in DB. |
| `prepare_prompts.py` | `--regen-prompts <m>` | Re-roll model-kind only. |
| `prepare_prompts.py` | `--regen-family-prompts <f>` | Re-roll family-kind only. |
| `render_prompts.py` | `--models <m1,m2>` | Render model-kind prompts; the `model_id` IS the checkpoint. |
| `render_prompts.py` | `--families <f1,f2> --render-with-model <m>` | Render family-kind prompts via `<m>` (must belong to the family). |
| `render_prompts.py` | mutex | `--models` XOR `--families`; `--render-with-model` REQUIRES `--families`, FORBIDDEN with `--models`. |

### 17.2 Walkthrough

```bash
# 1. Prepare both kinds on a fresh series (one LLM hop generates the
#    flux scene_facet; both prompt-kinds compose from it).
python scripts/prepare_prompts.py \
    --mode theme --level T2_implied \
    --models gonzalomo_photo_v70 \
    --families flux \
    --llm cydonia_heretic_24b

# 2. Inspect — expect ('family','flux',N), ('model','gonzalomo_photo_v70',N).
sqlite3 nsfw_pipeline.db \
  "SELECT target_kind, model_id, COUNT(*) FROM prompts GROUP BY 1, 2"

# 3. Render the family-kind rows through a chosen flux checkpoint.
python scripts/render_prompts.py \
    --series-id <id> \
    --families flux --render-with-model gonzalomo_flux_v30 \
    --llm cydonia_heretic_24b

# 4. Render the model-kind rows separately (paths don't collide).
python scripts/render_prompts.py \
    --series-id <id> \
    --models gonzalomo_photo_v70 \
    --llm cydonia_heretic_24b

# 5. Output paths — symmetric for both kinds.
ls output/T2_implied/<id>/cydonia_heretic_24b/flux/images/
ls output/T2_implied/<id>/cydonia_heretic_24b/gonzalomo_photo_v70/images/
```

### 17.3 Family-membership validation

`render_prompts.py` validates at parse time that
`--render-with-model` belongs to the family supplied via
`--families`. Pony booru tags don't render correctly through a flux
checkpoint, etc. — this rejects the combination with rc=2 and a
clear "gonzalomo_photo_v70 belongs to family 'sdxl', not 'flux'" message
before any LLM or ComfyUI work fires.

### 17.4 PNG metadata

Family-kind renders carry both `target_kind='family'`,
`model_id='flux'` (the family) AND `render_model_id='gonzalomo_flux_v30'`
(the actual checkpoint that produced the image) in the
`nsfw_pipeline` tEXt chunk. A forensic reader can answer "was this
prompt family-level or model-level? which LLM produced the prompt?
which checkpoint actually rendered it?" from the PNG alone.

### 17.5 Out-of-scope CLIs

`compare_models.py`, `dry_run.py`, `run_once.py`, and `src/main.py`
remain model-only for now. Family-aware variants are deferred — the
engine plumbing is in place (`run_phase_a` / `run_phase_b` accept
`target_kind`), so adding a `--families` flag to those scripts is a
small follow-up rather than a deep refactor.
