# NSFW Content Generation Pipeline — System Architecture

> **Platform:** Mac M4 Pro, 48 GB unified RAM  
> **Target:** DeviantArt, Patreon  
> **Stack:** Python 3.11, SQLite, ComfyUI, Ollama 0.5+
> **Last sync:** 2026-05-20 (External-templates-only cleanup —
> custom-workflow generation + IPAdapter + character mode + 8 legacy
> scripts + postprocess Upscaler/FaceRefiner + workflow capability
> flags ALL deleted. Pipeline now renders ONLY through external
> templates under `config/comfyui_workflows/templates/<family>/`.
> Per-family `default_template` in `config/families.yaml`. Family-
> match validator at render time. Default render model = gonzalomo_chroma_v30.
> mode_weights rebalanced to theme 0.50 / niche 0.25 / style 0.125 /
> variation 0.125. DB drops to 9 tables (characters removed).
> See `PROJECT_GUIDE.md` for the new CLI reference.)
> **Prior sync:** 2026-05-20 (Anti-grid / anti-mirror cleanup — vocab v7
> dropped 6 mirror / reflection / frame-within-frame entries.)
> **Earlier sync:** 2026-05-19 (Creative-uplift overhaul — vocab v6 +
> 6 namespaces + series-level aesthetic inheritance. User reported
> rendered images were "boring, non-creative, static boring
> backgrounds, can't generate any profit selling these". Root-cause
> audit found three structural gaps: (1) ``SeriesPlanner`` locked
> ONE ``environment`` per series, forcing all 24 scenes into the
> same room; (2) ``SceneGenerator`` system prompt explicitly
> enforced "same location" across the set; (3) vocab was photo-
> technical only — zero coverage for environments / props /
> atmosphere / color palettes / photographer refs / art movements /
> narrative moments / advanced composition. Six-phase fix shipped
> across 5 commits + verifier-agent round-trip:
> Phase 0 — fixed ``all_concepts_for_family`` hardcode (was iterating
>   only ``realism`` + ``nsfw`` top-level keys; new namespaces were
>   invisible to the LLM menu). Verifier B1 blocker.
> Phase 1 — ``environment.setting`` (41 location archetypes) +
>   ``environment.atmosphere`` (24 atmospheric elements). Tier-required
>   at T3+. ``SceneGenerator`` updated to push for per-scene environment
>   variety. Pony participates (locations have natural booru forms).
> Phase 2 — ``narrative.moment`` (30 tags). The #1 leverage axis per
>   market research — "she reads a letter at dawn" forces window +
>   chair + envelope + stillness in one tag. Tier-required at every
>   tier. Placed on SceneFacet (not Scene) per verifier B2/B3 so
>   validator + retry-nudge actually fires.
> Phase 3 — series-level aesthetic anchors:
>   ``aesthetic.color_palette`` (17 cinematic grades) +
>   ``aesthetic.photographer_ref`` (15 named photographer signatures,
>   PetaPixel-verified Midjourney-prompt-frequency) +
>   ``aesthetic.art_movement`` (15 art-history traditions). Pinned ONCE
>   per series by ``SeriesPlanner``, threaded into every scene via new
>   ``canonicalize_series_aesthetic`` helper. Per-style-profile
>   ``compatible_*`` lists narrow the menu to coherent combinations.
>   This is the "signature look" layer — the commercial differentiator
>   every top Patreon boudoir creator has. Pony omits photographer +
>   art_movement (no booru equivalents); participates in color_palette.
> Phase 4 — ``environment.prop`` (30 named-prop tags) +
>   ``composition.principle`` (18 higher-order composition rules:
>   frame-within-frame, reflection-primary, leading-lines, etc.).
>   Optional polish layer; LLM picks when they add value.
> Phase 5 — magic-word audit. Verifier I7 finding confirmed: chroma/
>   flux/flux2 already have empty quality_prefix/suffix +
>   correct avoid_words; ``cinematic depth of field`` usage in
>   example_prompts is descriptive prose, not boost. No action.
> Phase 6 — docs sync + 8 new regression tests. 1281 tests pass
>   (1269 baseline + 12 new). Full plan + verifier critique at
>   ``.claude/plans/creative-uplift.md``.
>
> End-to-end smoke: a T4 chroma facet with all Phase 1-4 fields
> populated composes to 9 distinct phrasings (vs. 4 pre-Phase-1) —
> golden hour + intimate mood + Tuscan villa + dust motes + reading
> letter + peonies + frame-within-frame + full nude + sensual gaze.
> All coherent around the "Tuscan-villa morning letter-reading" theme.
>
> Prior sync: 2026-05-18 (Prompt-generation architectural pass —
> creativity + reliability + correctness. Five intertwined fixes
> shipped after a thorough audit of the prompt generation pipeline,
> all driven by quality + artistic + creativity goals for NSFW art
> output: (1) ``nsfw.act`` vocab expanded from 1 usable solo tag to
> 8 — pre-fix every T4 scene was locked to ``NSFW_T4_SOLO_TOUCH``
> (the only non-banned act in the solo-only pipeline); seven new
> SOLO_* acts (DISPLAY / RECLINING / MIRROR / BATH / GAZE / OUTDOOR
> / PERFORMER) give T4 real creative variation. (2) ``nsfw.posture``
> expanded from 3 to 8 — added STANDING / SEATED / DRAPED /
> BACK_VIEW / SIDE_PROFILE canonical figure-study poses for richer
> T3+ compositional variety. ``vocab_version`` bumped 4 → 5.
> (3) All 18 few-shot examples (3 per family × 6 families) rewritten
> to demonstrate the structured enum-tag fields — pre-fix the
> examples only populated free-text fields (``camera_spec`` /
> ``scene_prose`` / ``booru_tags``), so the LLM mirrored the pattern
> and nulled every structured tag. Now every example fills
> ``lighting_directive`` / ``mood_aesthetic`` and (where the schema
> has them) ``realism_camera`` / ``realism_lens`` /
> ``realism_film_stock`` / ``art_style_reference`` / ``nsfw_*``.
> Also fixed a Flux.2 example bug where ``lighting_directive`` held
> free-text prose instead of an enum tag (the canonicalizer silently
> dropped it, teaching wrong convention). (4) ``mood_aesthetic``
> joined ``_TIER_REQUIRED_FIELDS`` at every tier alongside
> ``lighting_directive`` — same canonicalizer rationale, same retry
> nudge with example tags inlined via ``_FIELD_EXAMPLE_TAGS``.
> (5) Validator hardening — ``_missing_required_fields`` is now
> schema-aware (skips fields not in the facet dict so future
> required-field additions don't break Pony's narrower schema) AND
> rejects unknown enum tags (LLM inventions like ``MOOD_ETHEREAL``
> route to the retry-nudge with valid menu values inlined).
> Discovered + fixed a bug where ``_attempt`` stripped None values
> too eagerly, defeating the schema-aware check; None-filter moved
> to a single end-of-``generate()`` step. Empirical result on a
> regen-facets smoke: lighting_directive + mood_aesthetic
> population went from 0/24 to 24/24 (100%); retry-fired-but-
> recovered rate is 24/24 (every facet gets retried for the missing
> fields and the retry succeeds with the inline-example nudge).
> Total tests 1269 (was 1265; +4 unknown-tag cases).)
>
> Prior sync: 2026-05-18 (Facet post-validation hardening + PNG
> seed metadata fix — two narrow corrections shipped after end-to-end
> validation of the heretic-vision LLM exposed gaps: (1)
> ``lighting_directive`` joined the tier-required field list at every
> content tier (T1-T4) in
> ``scene_facet_generator._TIER_REQUIRED_FIELDS``. Pre-fix, every
> realism enum-tag field (camera/lens/film_stock/lighting/mood/
> art_style) was Optional[str] + unenforced; heretic-tuned LLMs
> empirically nulled them all, which left the canonicalizer (the
> whole point of vocab_version 2) as dead weight. Adding just
> lighting_directive — the single biggest factor in image quality
> + present in all 5 family schemas — lifts canonicalizer coverage
> from 0% to ~87% on heretic-vision with at most one retry per
> facet. The retry-nudge inlines 4-5 concrete tag examples per
> missing field via the new ``_FIELD_EXAMPLE_TAGS`` table so the LLM
> doesn't have to recall the menu from the long system prompt;
> empirically this lifts retry hit rate 14→21 / 24 on the same
> series. Function rename ``_missing_required_nsfw_fields`` →
> ``_missing_required_fields`` (back-compat alias kept).
> (2) PNG ``nsfw_pipeline`` chunk's ``seed`` field now reads from
> the built workflow dict via the new
> ``engine._extract_seed_from_workflow`` helper, not from the
> ComfyUI ``RenderedImage`` response — the dataclass has no seed
> attribute, so pre-fix every PNG recorded ``seed: 0``. Helper
> handles both ``ksampler.seed`` (SDXL/Pony/Illustrious/Flux/Flux.2
> + every external template) and ``random_noise.noise_seed``
> (Chroma's built-in base.json). 25 new focused tests cover both
> fixes; full suite 1257 passing.)
>
> Prior sync: 2026-05-18 (LLM registry overhaul + routing disabled
> — the previous 3-LLM setup (`cydonia_24b_v43` + `venice_24b` +
> `magnum_v4_22b`) is retired. New 2-LLM registry: (1) primary +
> default for every role: `cydonia_heretic_24b` — Ollama tag
> `Fermi/Cydonia-24B-v4.3-heretic-vision:Q4_K_M`, the heretic-tuned
> (refusal-removed) variant of Cydonia 24B v4.3 on Mistral Small 24B.
> The "heretic" tune drops the refusal floor low enough that the
> prior per-role split (venice for facets) is no longer needed —
> one LLM serves planner / scene-gen / facet / character / metadata.
> (2) fallback: `qwen3_abliterated_30b` — Ollama tag
> `huihui_ai/qwen3-abliterated:30b-a3b`, the huihui_ai abliterated
> (refusal-removed) variant of Qwen3 30B-A3B. MoE: ~30B total
> parameters, ~3B active per token, ~18GB on disk at Q4. Different
> lineage (Qwen vs Mistral) gives meaningful diversity on
> ``OllamaClient.generate_json``'s second-chance retry after two
> consecutive constrained-decoding failures. Routing in
> `pipeline.yaml::llm.routing` is now `{}` (intentionally empty);
> every role falls through to `default_llm`. The 2026-05-19 LLM
> swap replaced the Hermes 3 8B variants (Q4 + Q8) with Qwen3
> 30B-A3B — the Hermes variants failed empirically to populate
> structured enum-tag fields under constrained decoding on
> T4_explicit (0–8% structured-tag fill rate vs Cydonia's 96%),
> so a different lineage at higher capacity replaces them.)
>
> Prior sync: 2026-05-17 (Global single-female subject enforcement
> — pipeline-wide constraint that EVERY render targets exactly one
> adult female subject (multi-subject is explicitly deferred).
> 7-layer defence-in-depth mirroring the existing age-safety
> pattern: (1) tier directives (`categories.yaml` T1-T4) gain a
> SOLO clause — light-touch at T1/T2, strict at T3/T4 (T4 directive
> additionally forbids partnered `nsfw_act` tags); (2) mode planner
> SYSTEM_PROMPTs (`theme/niche/style/character` modes +
> `SeriesPlanner` + `SceneGenerator` + `SceneFacetGenerator`)
> inject a SOLO operating principle so the LLM sees the constraint
> from multiple angles, with Pony's booru system prompt explicitly
> forbidding `2girls / multiple_girls / NSFW_T4_PARTNERED_*`; (3)
> `families.yaml` gains a `solo_anchor` field — separate from
> `adult_anchor` — with booru-shaped `1girl, solo` (Pony,
> Illustrious) vs the default single `solo` token (SDXL realism
> finetunes still recognise booru subject vocabulary) vs prose
> sentence for Flux / Chroma / Flux.2; (4) `negative_axes` taxonomy
> bumps from 7 axes to 8 with new `subject_count` axis (booru
> families emit `2girls / multiple_girls / multiple_subjects /
> group / crowd`; Chroma emits prose `two people / multiple people
> / a couple / group / crowd / another person`; Flux / Flux.2 stay
> empty since CFG-distilled families ignore negatives — also
> exempt from `filter_conflicts` so positive `solo` doesn't cancel
> the negative-side suppression); (5) `HARD_BLOCK_NEGATIVE` adds
> `2girls, multiple_girls, multiple_subjects` so every family that
> supports negatives gets the multi-subject gate unconditionally;
> (6) `_positive_subject_count_scan` strips multi-subject vocab
> from any LLM drift before composition (ERROR log so the drift is
> visible); (7) `vocabulary.py::_SOLO_MODE_BANNED_TAGS` filters the
> 4 partnered T4 `nsfw_act` tags both from the LLM's concept menu
> (`all_concepts_for_family`) and from the canonicalizer's output
> (defence-in-depth — even if a stale directive slips a banned tag
> through). End-to-end validation on a fresh T4 chroma series
> (25/25 scenes) confirmed zero multi-subject leakage. Illustrious
> `adult_anchor` corrected from prose-shaped default to booru-shaped
> `1girl, mature_female, adult` to match its composer convention.
> 1240 tests pass.)
>
> Prior sync: 2026-05-17 (T4 softness fix + Venice routing default —
> 5-layer fix lands: (1) `prepare_prompts.py --series-id` retarget
> now inherits `content_level` from DB instead of silently
> downgrading to T2_implied; (2-3) theme/niche/style/character modes
> and SeriesPlanner/SceneGenerator inject the rich
> `categories.yaml::content_levels.<tier>.llm_directive` into both
> plan and scene templates inside a `══ CONTENT TIER ══` banner;
> (4) `families.yaml` T4 few-shot exemplars rewritten across all 6
> families with explicit `fully_nude / breasts / nipples / vulva /
> anatomically_correct` (booru) and `fully nude / bare breasts /
> natural nipples` (prose) language; (5) `scene_facet_generator.py`
> post-validation enforces tier-required NSFW fields —
> `nsfw_anatomy` at T3+, both `nsfw_anatomy + nsfw_act` at T4 —
> retries with explicit nudge then ships with warning if still
> missing. Plus `pipeline.yaml::llm.routing.scene_facet_generator
> .default: venice_24b` (2.2% refusal floor) pinned by default so
> the facet LLM doesn't self-censor — empirically resolves the
> "tasteful boudoir at T4" symptom. Pony `families.yaml::adult_anchor`
> corrected from non-Danbooru `1woman` to canonical
> `1girl, mature_female`. `init_db.py::series.status` CHECK
> constraint now includes `aborted` (fixed latent crash on
> supervisor reject path).)
>
> Prior sync: 2026-05-06 (Family-level prompt prep — `prompts` schema
> gains `target_kind` discriminator (`'model'` | `'family'`); UNIQUE
> extends to `(scene_id, target_kind, model_id, llm_id)`. New
> `prepare_prompts --families <f>` and `render_prompts --families <f>
> --render-with-model <m>` flag pair lets a series carry both
> family-kind (checkpoint-agnostic, no per-model overlay) and
> model-kind (full trigger / avoid / LoRA stack) prompts on the same
> scenes; `scene_facets` rows are shared across both kinds since the
> facet table is family-keyed. `GenerationContext` gains
> `target_kind` field + `__post_init__` invariant guard;
> `build_family_context` factory produces family-kind ctxs for
> Phase A's per-target loop. Output paths symmetric:
> `output/<level>/<series>/<llm_id>/<target_id>/{images,preview}/`
> where `target_id` = `model_id` for model-kind, `family_id` for
> family-kind. PNG `nsfw_pipeline` chunk records `target_kind` and
> `render_model_id` for forensic reproducibility.
> Out-of-scope: `compare_models.py` / `dry_run.py` / `run_once.py` /
> `src/main.py` stay model-only for now (engine-side wiring is done;
> CLI flags are deferred). See PROJECT_GUIDE.md §17.)
>
> Prior sync: 2026-05-04 (Multi-LLM cleanup + quality lifts — F1-F4
> + Q6-Q11 shipped. Per-role routing now fires for every agent (not
> just facet generator); `OllamaClient` is a pure transport with
> per-call model required; constrained decoding wired for Scene/
> Metadata/Character generators; Pattern A persona + assistant-prefill
> via `/api/chat`; booru-tag persona for Pony/Illustrious; registry-
> aware fallback LLM with second-chance retry; vocabulary v4 added
> `realism.angle` + `realism.framing` namespaces; tier-stratified
> few-shot examples — every family ships ≥3 T2+T4-covering examples.
> 1176 tests passing. See PROJECT_GUIDE.md §16 for the multi-LLM
> workflow.)
>
> Prior sync: 2026-05-03 (Multi-LLM upgrade foundation — `LLMRegistryLoader`
> + `LLMRouter` + `cli_llm_override` plumbed through `run_cycle` /
> `run_phase_a` / `run_phase_b`. Schema bumped: `scene_facets` PK
> extends to `(scene_id, family, llm_id)`; `prompts` UNIQUE extends
> to `(scene_id, model_id, llm_id)`. Output paths gain `<llm_id>`
> segment so two LLMs A/B-rendering the same series don't overwrite
> each other.)
>
> Prior sync: 2026-05-02 (NSFW-output-path fix — `SceneFacetGenerator`
> now sees `content_level` + tier-specific `llm_directive` from
> `categories.yaml`. Phase B closed Phase 4-bis gap by adding the
> `nsfw_act` field/column.)

---

## 1. System Overview

A local pipeline that generates themed image sets for adult content
platforms. The system plans a cohesive series with an LLM, renders the
images with ComfyUI (Stable Diffusion checkpoints), scores them for
quality, selects the best, and exports a ready-to-post package.

The fundamental hardware constraint: **LLM and ComfyUI never run
simultaneously.** Both compete for the same 48 GB of unified memory.
The pipeline enforces this by running in three sequential phases:

```
Phase A (LLM)      Phase B (Render)      Phase C (Package)
───────────────     ────────────────      ─────────────────
plan series         memory preflight      build set
generate scenes     render images         generate metadata (brief LLM reload)
build prompts       score images          watermark
sanitize/dedup      postprocess           export
save dry_run        supervised review     persist to DB
unload LLM ──────▶  (LLM is gone)        record to memory
```

The LLM is explicitly unloaded (`keep_alive: 0` + 2 s sleep) between
Phase A and B so macOS releases the memory before ComfyUI loads its
checkpoint.

---

## 2. Database Schema

SQLite, single file: `nsfw_pipeline.db`. Created by
`scripts/init_db.py`. The DB holds **runtime-mutable state only** —
static lookup data (models, families, style profiles, categories) lives
in YAML under `config/`. See §3 for the storage split.

### 10 Tables

| Table | Purpose | Key columns |
|-------|---------|-------------|
| `characters` | Character identities | `id`, `style_profile_id` (weak-ref TEXT), `base_prompt` (immutable), `negative_prompt`, `reference_image_path`, `locked_features` (JSON), `allowed_shift_axes` (JSON), `outfit_pool` (JSON), `version`, `active` |
| `series` | One concept-level row per series | `id`, `mode`, `content_level`, `character_id`, `style_profile_id`, `theme`, `llm_series_plan` (JSON — re-targeting reads this back), `status` |
| `scenes` | Per-scene model-agnostic core | `id`, `series_id`, `pose`, `camera`, `camera_angle`, `lighting`, `environment_detail`, `mood_note`, `expression`, `aspect_ratio`, `resolution_w/h`. **No family-shaped fields** — those moved to `scene_facets`. |
| `scene_facets` (Phase 1) | Per-(scene, family) LLM expansion | PRIMARY KEY `(scene_id, family)`. Holds free-text family fields — `booru_tags`, `source_tag`, `scene_prose`, `camera_spec`, `clothing` — **plus** the structured concept-tag enum fields added in 2026-04 Phase 4a (`realism_camera`, `realism_lens`, `realism_film_stock`, `art_style_reference`, `lighting_directive`, `mood_aesthetic`, `nsfw_anatomy`, `nsfw_posture`). Set by `SceneFacetGenerator`. Sibling models in the same family share one row. The `family` CHECK clause is **templated from `config/families.yaml`** at init time. |
| `prompts` | Per-(scene, target_kind, target_id, llm) composed text | `id`, `series_id`, `scene_id`, **`target_kind` (NOT NULL, `'model'` \| `'family'` — added 2026-05)**, **`model_id` (NOT NULL — dual semantic: model id when target_kind='model', family id when target_kind='family')**, **`llm_id` (NOT NULL)**, `prompt_text`, `negative_prompt`, `prompt_hash`, `content_level`, **`vocab_version` (Phase 4a — records the `prompt_vocabulary.yaml` version that produced the row)**, `status`. **`UNIQUE(scene_id, target_kind, model_id, llm_id)`** enforces "one prompt per (scene, kind, target, generating-LLM)" so model-kind and family-kind prompts coexist on the same scene. Re-rolling requires `prepare_prompts --regen-prompts <model> --llm <id>` (model-kind) or `prepare_prompts --regen-family-prompts <family> --llm <id>` (family-kind). |
| `images` | Rendered images + scores | `id`, `prompt_id`, `series_id`, `model_id` (weak-ref TEXT), `file_path`, `seed`, `quality_score`, `aesthetic_score`, `blur_score`, `face_confidence`, `hps_v2_score`, `image_reward_score`, `quality_flags` (JSON). The two Phase-G score columns are nullable — populated only when `scoring.use_hps_v2` / `use_image_reward` are flipped on |
| `sets` | Exported set metadata | `id`, `series_id`, `title`, `tags` (JSON), `export_path` |
| `posts` | Engagement tracking | `id`, `set_id`, `platform`, `views_24h/72h`, `favorites` |
| `generation_memory` | Anti-repetition hashes | `type`, `content_hash`, `character_id` |
| `run_log` | Audit trail | `mode`, `content_level`, `series_id`, `status`, `images_generated/selected`, `duration_seconds`. One row per (series, model) render. |

### Indexes

On: `series(mode, status, character_id, content_level)`,
`images(series_id, selected, content_level, quality_score, model_id)`,
`prompts(series_id, prompt_hash, status, model_id)`,
`scene_facets(scene_id)`,
`generation_memory(content_hash, type, character_id)`,
`characters(active)`, `posts(set_id, character_id)`,
`sets(series_id)`, `run_log(status, created_at)`.

### 1 Trigger

```sql
CREATE TRIGGER protect_character_base_prompt
BEFORE UPDATE ON characters
WHEN OLD.base_prompt != NEW.base_prompt
BEGIN
    SELECT RAISE(ABORT, 'base_prompt is immutable after creation.');
END;
```

### Cross-ref Chain (weak FKs validated at startup)

```
config/models/{id}.yaml            (model registry, 17 files)
config/families.yaml               (6 families referenced by family:
                                    sdxl, pony, illustrious, flux,
                                    chroma, flux2)
config/style_profiles.yaml         (10 profiles)
  └── characters.style_profile_id  (plain TEXT, validated at startup)
        └── series.character_id
              ├── scenes.series_id
              │     └── scene_facets.scene_id (+ family → YAML)
              ├── prompts.series_id (+ prompts.model_id → YAML)
              ├── images.series_id (+ images.model_id → YAML)
              └── sets.series_id
```

`characters.style_profile_id`, `images.model_id`, and
`prompts.model_id` are plain TEXT columns — the YAML loaders
fail-fast at startup if any reference points to a nonexistent id.
This is strictly tighter than SQLite's FK enforcement (which the
codebase disables per-connection). `scene_facets.family` is the one
weak-ref enforced via SQL CHECK (templated from `families.yaml` at
init time).

### Removed in the 2026-04 refactor

- 7 static lookup tables moved to YAML: `model_registry`,
  `model_prompt_guides`, `style_profiles`, `theme_categories`,
  `style_categories`, `niche_clusters`, `content_level_rules`.
- 3 dead columns on `characters`: `experimental_lane`,
  `character_lora_path`, `lora_overrides` (Phase-2 placeholders with
  zero readers).

---

## 3. Pipeline Engine

`src/core/engine.py` — `PipelineEngine` class. **Four public methods** exposed (the Phase 3 split):

```
run_cycle()       — composition wrapper; calls phase_a → phase_b → phase_c
                    for a single model. Behavior identical to pre-Phase-3.
run_phase_a(ctx, *, models, series_id_existing,
            regen_facets, regen_prompts,
            style_profile, style_profile_id) -> dict
                  — LLM phase: plan + scene gen + per-(scene, family)
                    facet + per-model prompts. Multi-model fan-out.
                    No ComfyUI calls. Single LLM unload at end.
                    Skips `run_log` (Phase A is not a render run).
run_phase_b(*, series_id, model_id, scene_ids,
            template_override) -> (rendered_images, ctx_state)
                  — Render phase for one (series, model). Reloads
                    series + scenes + prompts from DB; rebuilds ctx
                    for the target model; recomputes resolution
                    per-target-model at render time; IPAdapter
                    staging; ComfyUI loop with retries; score;
                    postprocess.
run_phase_c(*, series_id, model_id, content_level,
            rendered_images, series_plan, scenes, prompts,
            ctx, style_profile, style_profile_id,
            elapsed_seconds) -> dict
                  — Set build + watermark + export + persist images
                    + memory record + character usage + one
                    `run_log` row per (series, model).
```

The `prepare_prompts.py` CLI wraps `run_phase_a` directly (LLM-only).
The `render_prompts.py` CLI wraps `run_phase_b` + `run_phase_c` per
model (DB→render). `run_once.py` wraps all three for the single-model
end-to-end case. The same code paths underlie all three CLIs.

### Phase A — LLM Planning

1. **Build context** — `build_context()` resolves the model id via
   the precedence `--model CLI override → character.model_id →
   pipeline.default_model_id`, looks up the model in
   `config/models/*.yaml` (via `ModelRegistryLoader`), loads the
   aesthetic-only `StyleProfile` for the character (via
   `StyleProfileLoader`), then merges the family rules from
   `config/families.yaml` with the model's
   `prompt.extend`/`prompt.override` hooks. Produces a
   `GenerationContext` carrying both the model entry and the
   effective `FamilyConfig`. (Post-2026-04 refactor: the style profile
   no longer carries `model_id` — checkpoint selection is bound to
   the character or the pipeline default, not to the aesthetic.)

2. **Preflight** — Checks: Ollama reachable, checkpoint file exists,
   workflow template exists, IPAdapter compatibility if character mode.

3. **Mode selection** — `ModeSelector` picks one of 5 modes by weighted
   random draw (configurable in `pipeline.yaml`). `--mode` CLI flag
   overrides.

4. **Plan** — The selected mode's `plan(ctx)` calls `SeriesPlanner` to
   generate a series plan (theme, mood, environment, variation_axes).

5. **Scene generation** — `generate_scenes(plan, ctx)` calls
   `SceneGenerator` to produce 20-30 scene dicts of the
   **model-agnostic core** (post Phase 2 split): 7 required fields
   (`variation_axis`, `pose`, `camera`, `camera_angle`, `lighting`,
   `environment_detail`, `mood_note`) + 4 optional model-agnostic
   fields (`expression`, `composition_intent`, `framing_hint`,
   `audience_target` — the last three feed the Phase A ratio scorer).
   **Family-shaped fields no longer come from this step.** Validation
   uses `Scene(BaseModel)` from `src/agents/schemas.py`; failures
   trigger an LLM retry with the Pydantic error path inlined as a nudge.

6. **Ratio assignment** — `RatioSelector.select()` picks an aspect
   ratio per scene via 5-axis additive scoring (Phase A): content-
   level base weights + audience bonus
   (`deviantart`/`patreon`) + composition-intent bonus (`close-up` /
   `medium` / `full-body` / `wide`) + pose-signal bump (+0.15
   default) + family-quality bonus, with each ratio clamped to
   `[0.05, 2.0]` then drawn deterministically from the per-scene
   hash. Hard overrides survive: `pose ∋ "full body"|"standing"` →
   `portrait_23`; `pose ∋ "reclining"|"lying"` → `landscape`.
   Resolution lookup is 3-tier: per-model `resolution_*` →
   per-family megapixel bucket (`src/core/aspect_ratio_buckets.py`)
   → base `RATIO_TO_RESOLUTION` SDXL defaults.

7. **Per-model fan-out** (multi-model is the new norm; single-model
   `run_cycle` passes `models=[ctx.model_id]`):

   For each `model_id` in `models`:

   a. **Build per-model ctx** — look up model → entry → family →
      prompt_guide. The ctx's family + trigger / avoid words may
      differ from the baseline.
   b. **Optional `--regen-facets <family>`** — bulk-DELETE existing
      `scene_facets` rows for that family across the series's scenes.
   c. **Per-scene facet** — for each scene, ensure a `scene_facets`
      row exists for this model's family. If missing, call
      `SceneFacetGenerator` (`src/agents/scene_facet_generator.py`)
      — a focused LLM call that produces the family-shaped fields
      (`booru_tags`, `source_tag`, `scene_prose`, `camera_spec`,
      `clothing`) per the per-family Pydantic schema in
      `SCENE_FACET_SCHEMA_BY_STYLE`. Sibling models in the same
      family share the row.
   d. **Optional `--regen-prompts <model>`** — DELETE existing
      prompts for this `(series, model)` pair before composing.
   e. **Per-scene prompt build** — `PromptBuilder.build_one()`
      receives the merged `scene_with_facet` dict and dispatches to
      one of **five** composers (`sdxl_keywords`, `pony_danbooru`,
      `illustrious_tags`, `flux_natural`, `flux2_prose`) selected by
      `family.prompt_style`. The composer injects quality prefix /
      suffix and strips `avoid_words`. **Phase C** token-trim runs
      after composition: `fit_to_budget(text, max_tokens,
      tokenizer_id, break_marker)` — CLIP for
      sdxl/pony/illustrious, T5 for flux/chroma, word-count heuristic
      for flux2; trim from the middle, BREAK-window-aware for Pony.
      **5-layer negative assembly** (Phase D + E): TI embeddings
      hoisted, `HARD_BLOCK_NEGATIVE` (age + multi-subject), family
      `negative_axes` (8-axis taxonomy with conflict-filter; the
      `subject_count` axis is exempt from filter_conflicts),
      per-model `prompt.extend.negative_*`, character-level.
   f. **Sanitize** — `PromptSanitizer` enforces tier suppress / boost.
   g. **Dedup** — `PromptDeduplicator` per-model (hash + scene
      structural similarity against `generation_memory`).
   h. **Persist** — INSERT into `prompts` with `model_id` set;
      UNIQUE(scene_id, model_id) catches accidental double-insert
      and the CLI surfaces the violation as a `--regen-prompts` hint.
   i. **Supervised pause** — once per model when
      `execution.mode = supervised`.

8. **Single LLM unload at end** — regardless of how many models
   ran, `OllamaClient.unload_model()` is called once before any
   ComfyUI work. CRITICAL boundary on 48 GB unified memory.

11. **Supervised pause 1** — In supervised mode, prints plan summary and
    waits for human y/n.

### Phase B — Render

1. **Memory preflight** — Checks ≥14 GB free via `psutil`.

2. **Build style-for-workflow wrapper** — `_StyleProfileForWorkflow`
   merges the style profile and the model YAML entry. Two-tier
   precedence: if `--model` CLI was used, sampler/scheduler/steps/cfg
   come from the model's YAML (Tier 1); otherwise from
   `config/style_profiles.yaml` (Tier 2).

3. **IPAdapter staging** — For character mode when model supports it
   and character has a reference image: copies the image to ComfyUI's
   `input/` directory.

4. **Render loop** — For each prompt:
   - `WorkflowBuilder.build()` produces a ComfyUI-ready workflow JSON.
   - `ComfyUIClient.queue_prompt()` submits it.
   - `ComfyUIClient.wait_for_completion()` polls `/history/{prompt_id}`.
   - Up to 3 retries per image.

5. **Score** — `ImageScorer.score()` evaluates each rendered image
   across **6 signals** (Phase G): HPS v2 + ImageReward (both prompt-
   conditioned, opt-in via `scoring.use_hps_v2` /
   `use_image_reward`), LAION aesthetic, face confidence
   (insightface), Laplacian blur, resolution pass. Default Phase-G
   composite weights: `0.30·hps + 0.25·image_reward + 0.20·aesthetic
   + 0.10·face + 0.10·blur + 0.05·resolution`. When both Phase-G
   flags are off, the scorer auto-falls-back to `CompositeWeights.legacy()`
   (`0.40·aesthetic + 0.25·blur + 0.25·face + 0.10·resolution`) so
   users without the optional `hpsv2` / `image-reward` packages
   installed are unaffected. `score()` accepts an optional `prompt`
   kwarg; `score_batch` reads `prompt_text` from each img dict.
   When a Phase-G predictor fails to load mid-batch (e.g. weights
   not downloaded), `ImageScorer` logs once at WARNING and silently
   disables that signal for the rest of the run via
   `_hps_v2_disabled` / `_image_reward_disabled`. Weight
   redistribution in `_compose` keeps the composite in [0, 1]
   regardless of which signals fired.

6. **Postprocess** — Optional **pure-ESRGAN upscale** (4×-UltraSharp
   model → `ImageScaleBy(0.35)` = 1.4× source); Phase F shipped
   templates for `sdxl`/`pony`/`illustrious` only — `flux` /
   `chroma` / `flux2` already render natively at 1024+ and don't
   ship templates. `Upscaler.__init__` validates template existence
   eagerly so flipping `upscale_enabled: true` for an untemplated
   family fails fast at construction. Optional face-refine
   (FaceDetailer) is still gated behind an unbuilt
   `{family}/face_detail.json` template. Both disabled by default
   in `pipeline.yaml::postprocess`.

7. **Supervised pause 2** — Contact sheet + quality stats, human review.

### Phase C — Package

1. **Set builder** — Filters images by quality threshold (0.55),
   enforces level purity, selects 10-25 images.

2. **Metadata** — Brief LLM reload. `MetadataGenerator` produces
   title, description, and tags. Content-level aware: T1/T2 lean
   tasteful/aesthetic copy; T3/T4 lean bolder/more direct. LLM
   unloaded again immediately.

3. **Watermark** — Soft-level images get a corner watermark
   (`@YourDAHandle`, 30% opacity). Full-level exported unwatermarked.

4. **Export** — `Exporter` writes to
   `output/{content_level}/{series_id}/` with `images/`, `preview/`,
   `metadata.json`, `manifest.json`.

5. **Persist** — Images saved to DB. Character usage updated.
   Series/scenes/prompts recorded to `generation_memory`. Run logged
   to `run_log`.

---

## 4. Generation Modes

Five modes, selected by weighted random or `--mode` CLI:

| Mode | Weight | Entry | Description |
|------|--------|-------|-------------|
| **character** | 60% | `CharacterMode` | Character-based sets. IPAdapter ON when model supports it. LRU character rotation. Cross-level scene dedup. |
| **theme** | 20% | `ThemeMode` | Theme-driven sets from `config/categories.yaml:themes`. `subject_description` from plan leads the prompt. |
| **niche** | 10% | `NicheMode` | SEO-driven sets from `config/categories.yaml:niches`. Top 3 keywords appended as extra_keywords. |
| **style** | 5% | `StyleMode` | Style-driven sets from `config/categories.yaml:styles`. `style_keywords` lead the prompt. |
| **variation** | 5% | `VariationMode` | Re-imagines scenes from a previous series. Varies pose/camera/lighting/expression axes. |

All modes extend `BaseMode` (abstract class with `plan()` and
`generate_scenes()` methods). Each mode delegates to `SeriesPlanner`
and `SceneGenerator` for LLM calls.

### Character Mode specifics

- **LRU selection**: `CharacterManager.get_least_recently_used()` picks
  the character with the oldest `last_used_at`. NULL (never used) wins.
  Tiebreaker: `created_at ASC`.
- **IPAdapter**: When `ctx.supports_ipadapter` is True and the character
  has `reference_image_path`, the render uses the `ipadapter.json`
  workflow template instead of `base.json`. The reference image is copied
  to ComfyUI's `input/` dir before rendering.
- **Cross-level dedup**: For character mode, `MemoryManager.is_novel()`
  checks across ALL content levels for the same `character_id`.

---

## 5. Content Level System

**Four tiers**, defined in `src/core/content_level.py` and enforced
by SQL CHECK constraints on `series`, `scenes`, `prompts`, `images`,
`sets`, `run_log`:

| Tier | ID | Spectrum | Default mix |
|------|----|----------|-------------|
| 1 | `T1_suggestive` | clothed, mood / pose / lighting carry intent | 10% |
| 2 | `T2_implied` | tasteful nude / partial nudity, posed for art | 35% |
| 3 | `T3_artnude` | full nudity, fine-art framing | 45% |
| 4 | `T4_explicit` | explicit content, sale-tier-only | 10% |

Per-tier rules live under
`config/categories.yaml::content_levels`, loaded by
`ContentLevelLoader`. Each tier defines:
- `allowed_pose_types` — JSON array of permitted poses.
- `scene_constraints` — `mood_range`, `environment_constraint`,
  `max_skin_exposure`.
- `prompt_boost_keywords` — appended to every prompt at this tier.
- `prompt_suppress_keywords` — stripped from every prompt at this tier.

`PromptSanitizer` enforces keyword policies. Word-boundary regex
prevents substring corruption (e.g. "aggressive" inside
"aggressively"). Idempotent.

Per-tier `aspect_ratio_weights` in `pipeline.yaml` skew T3/T4 toward
portrait (full-body nudes and explicit framing sell tall) and T1/T2
toward variety (more square / landscape).

**Content tiers never mix within a series.** `assert_level_purity()`
in `src/filter/level_purity_check.py` validates this invariant on
the rendered image set before export.

The project ships no migration scripts during pre-stable development:
schema changes go directly into `init_db.py::SCHEMA_SQL` and existing
DBs are re-initialized (`python scripts/init_db.py --force`). The
4-tier set has been the only valid CHECK constraint set in the
schema since the post-soft/full era.

---

## 6. Model Profile System

### Storage split

Models are declarative YAML under `config/models/*.yaml` (one file per
checkpoint). Each file references one of **six** families declared in
`config/families.yaml`. Nothing about models is in the DB — the
runtime reads YAML via `ModelRegistryLoader` / `FamilyLoader`.

### 8 Registered Models

| Model ID | Architecture | Family | Sampler | Scheduler | Steps | CFG | Neg? | License |
|----------|-------------|--------|---------|-----------|-------|-----|------|---------|
| `pony_realism_v23_ultra` | pony | pony | dpmpp_2m_sde | karras | 30 | 6.5 | Y | open |
| `cyberrealistic_pony_v170` | pony | pony | dpmpp_sde | karras | 30 | 5.0 | Y | open |
| `juggernaut_ragnarok` | sdxl | sdxl | dpmpp_2m_sde | karras | 35 | 4.0 | Y | open |
| `gonzalomo_photo_v70` | sdxl | sdxl | dpmpp_sde | karras | 8 | 2.0 | Y | open |
| `gonzalomo_flux_v30` | flux | flux | euler | simple | 12 | 1.0 | N | open |
| `gonzalomo_chroma_v30` | chroma | chroma | euler | simple | 26 | 3.8 | Y | open |
| `chroma_v10HD` | chroma | chroma | euler | simple | 26 | 3.8 | Y | open |
| `perfection_realistic_ilxl` | illustrious | illustrious | dpmpp_3m_sde | simple | 24 | 4.0 | Y | open |

> **Note (2026-05-18):** the user removed 9 model checkpoints from
> their local install on this date (`cyberrealistic_pony_v160`,
> `flux2_klein_9b`, `flux_nsfw_71q8`, `jib`, `lustify_endgame`,
> `lustify_olt`, `lustify_v7`, `mop`, `pony_realism_v23`). Their
> YAMLs and all code/test/doc references were deleted alongside.
> The flux2 family infrastructure remains (builder, schema,
> negative-axes, families.yaml entry) so a future flux2 model can be
> reintroduced by dropping a YAML into `config/models/`. Same for the
> commercial-mode license gate — no NCL-licensed model is currently
> registered, but the gate fires on any future YAML carrying
> `commercial_use: false`.

Flux and Chroma both use `models/unet/` + `UnetLoaderGGUF` (or
safetensors UNET). Flux uses `ModelSamplingFlux` + `FluxGuidance` +
standard `KSampler`; Chroma uses `ModelSamplingAuraFlow` + `CFGGuider`
+ `SamplerCustomAdvanced`. For Flux, `default_cfg` stores
`FluxGuidance.guidance`; `KSampler.cfg` is hardcoded to `1.0` in the
template. Illustrious XL uses the same graph shape as SDXL
(`CheckpointLoaderSimple` + `KSampler`), so it routes through the
standard `WorkflowBuilder.build()` path — no dedicated builder
method; only the `illustrious/` template directory and per-model
YAML profile. **Flux.2 family** wiring stays in place
(`_build_flux2`, FLUX2 schema, dedicated template) — when a new
flux2 checkpoint is added, the registered YAML drops into
`models/diffusion_models/`, loaded via `UNETLoader` + a single
`CLIPLoader(type=flux2)` (Qwen3-8B text encoder); distilled models
are warn-and-clamped to `cfg=1.0`, `steps≤6` (effectively 4),
`sampler=euler`, `scheduler=simple` — any override melts output.

### 6 Families (`config/families.yaml`)

| Family | `prompt_style` | Composer output | Neg? | Weighting? | CLIP skip | Tokens | Tokenizer |
|--------|----------------|-----------------|------|------------|-----------|--------|-----------|
| `sdxl` | `sdxl_keywords` | Comma-separated keywords | Y | Y | 1 | 77 | clip |
| `pony` | `pony_danbooru` | 6-tier `score_*` prefix + `BREAK` + Danbooru tags | Y | Y | 2 | 77 (×2 windows via BREAK) | clip |
| `illustrious` | `illustrious_tags` | Danbooru tags + short prose, with `masterpiece, best quality, amazing quality, very aesthetic, newest` suffix | Y | Y | 1 | 77 | clip |
| `flux` | `flux_natural` | Natural-language prose | **N** (CFG-distilled) | N | 1 | 512 | t5 |
| `chroma` | `flux_natural` | Natural-language prose, period-joined realism tail | Y (Chroma restored CFG) | N | 1 | 256 | t5 |
| `flux2` | `flux2_prose` | BFL 5-anchor prose (subject → setting → details → lighting → atmosphere, 30–80 words) | **N** (distilled) | N | 1 | 256 | heuristic (Qwen3 vocab not bundled) |

Research sources verified 2025–2026: AstraliteHeart Pony V6 model card
(mandates the full 6-tier score prefix, `clip_skip=2`); OnomaAI
Illustrious XL card (quality suffix string including the literal
`"very aesthetic"` — not `"very aware"`); Black Forest Labs Flux docs
(negatives are no-ops on vanilla Flux); lodestones Chroma HF repo
(re-trained with CFG, T5-256 window inherited from Schnell); BFL
Flux.2 Klein release notes (4-step contract, no negatives, 5-anchor
prose structure).

### YAML Profiles

Each model has a flat YAML file in `config/models/{model_id}.yaml`:

```yaml
id: juggernaut_ragnarok
display_name: Juggernaut XL Ragnarok
filename: juggernautXL_ragnarokBy.safetensors
architecture: sdxl
family: sdxl                        # → config/families.yaml
default_sampler: dpmpp_2m_sde
default_scheduler: karras
default_steps: 35
default_cfg: 4.0
default_clip_skip: null
supports_ipadapter: true
supports_lora: true
resolution_portrait: [768, 1152]
resolution_square: [1024, 1024]
resolution_landscape: [1152, 768]
active: true

# Optional: per-model VAE / text encoder + license metadata
vae_filename: null                   # use checkpoint's bundled VAE
text_encoder: null                   # use checkpoint's bundled CLIP
license: open                        # open|cc-by|flux_ncl|...
commercial_use: true                 # see compliance.commercial_mode gate

# Optional: stacked LoRAs (max 2 enabled), validated at registry load.
# Per-model defaults; can be replaced per-render via style profile.
lora_stack:
  - name: ultra_real_v4.safetensors
    strength: 0.70
    enabled: true
  - name: klein_slider_anatomy.safetensors
    strength: 2.0
    enabled: false                   # off for SDXL; example shape only

prompt:
  extend:
    trigger_words: ["shot on Canon EOS 5D", "glamour photography"]
    negative_embeddings: ["embedding:BadDream", "embedding:UnrealisticDream"]
    avoid_words: ["painting", "illustration"]
    negative_axes:                   # 8-axis taxonomy (anatomy / medium /
                                     # skin / quality / watermark / safety /
                                     # censor / subject_count)
      skin: ["plastic skin"]
      quality: ["jpeg artifacts"]
  override:
    # max_tokens: 60                 # replaces family's value
    example_prompt: "confident woman, ..."
```

`prompt.extend` appends to the family's values; `prompt.override`
replaces them wholesale (per-axis under `negative_axes`).
Conflicting keys (same field in both blocks) raise at load time.

**Commercial-license gate.** When
`pipeline.yaml::compliance.commercial_mode: true`,
`ModelRegistryLoader` filters out any model whose YAML declares
`commercial_use: false` (any future FLUX NCL-licensed checkpoint
would land here). Trying to resolve the model under the gate raises
`ModelNotFound` at startup, not mid-render — protects paid-tier
exports from accidentally feeding through a non-commercial
checkpoint. No NCL-licensed model is currently registered, but the
gate remains active for future additions.

### Two-Tier Sampler Precedence

When `--model` CLI is used, sampler/scheduler/steps/cfg come from the
model YAML (the model author's recommended settings). When no
override, they come from `config/style_profiles.yaml` (aesthetic
tuning measured for that profile's original model).

This is implemented in `_StyleProfileForWorkflow` (`engine.py`,
`render_set.py`) — the adapter exposes `.workflow_family` (now sourced
from the model's `family:` field) for `WorkflowBuilder`.

### Prompt Style Dispatch

`PromptBuilder` dispatches on `ctx.family.prompt_style`, set by the
family YAML:

- **`sdxl_keywords`** — Comma-separated keyword list with
  case-insensitive token dedup. Trigger words appended at the end.
- **`pony_danbooru`** — Danbooru-style tags with the family's 6-tier
  score prefix (`score_9, score_8_up, score_7_up, score_6_up,
  score_5_up, score_4_up, BREAK`). `BREAK` tells CLIP to encode the
  quality tags in a separate attention window. `clip_skip=2` is
  applied at the workflow level via the pony template. Embedding
  trigger words (e.g. `embedding:CyberRealisticPony_POSV1`) are
  appended via per-model `prompt.extend.trigger_words`.
- **`illustrious_tags`** — Hybrid: Danbooru tags first, then short
  prose phrases. Quality suffix (`masterpiece, best quality, amazing
  quality, very aesthetic, newest`) is appended by the composer.
- **`flux_natural`** — Natural-language paragraph. Segments become
  capitalized sentence fragments joined with `. `. Trigger words
  appended as a final sentence. Chroma uses the same composer with
  `realism_tail_style: period` so the camera/lens/photoreal tokens
  attach as period-joined fragments rather than comma-stitched.
- **`flux2_prose`** — BFL 5-anchor prose: subject → setting →
  details → lighting → atmosphere, target 30–80 words. No comma
  tag lists, no weighting syntax, no `BREAK`. The Ultra Real
  trigger phrase is surfaced via `prompt.extend.trigger_words` —
  the LLM may place it, the composer does not auto-prepend.

All five composers run a final `family.avoid_words` strip pass so
LLM drift (e.g. a rogue `score_9` in a Flux prompt) can't corrupt
the output. Every composed prompt is then handed to
`fit_to_budget` (Phase C) so over-long output is middle-trimmed
against the family's real tokenizer rather than CLIP-tail-truncated.

### 5-Layer Negative Prompt Assembly

`PromptBuilder.assemble_negative_prompt()` merges five sources, in
this order (`embedding:` tokens hoisted to the front so ComfyUI's
position-weighted encoder gives them their full weight):

1. **TI embeddings** (Phase E) — `negative_embeddings:` from per-
   model YAML, e.g. `["embedding:BadDream", "embedding:UnrealisticDream"]`.
   Identity-match dedup so `embedding:Foo` ≠ `embedding:Foo:0.8`.
2. **`HARD_BLOCK_NEGATIVE`** — age-ambiguity vocabulary, prepended
   to every render unconditionally (`child, kid, young, minor,
   teen, schoolgirl, loli, shota, underage, ...`). Belt-and-braces
   alongside the positive-side `_AGE_AMBIGUITY_PATTERN` scan and
   the post-render `PromptSanitizer`.
3. **`family.negative_axes`** flattened — 8-axis taxonomy
   (`anatomy / medium / skin / quality / watermark / safety /
   censor / subject_count`); each axis is a list of tokens. Before
   flattening, `filter_conflicts()` drops any token that appears in
   the positive prompt to avoid the classic `"naked"`-in-negatives-
   while-positive-says-`"nude pose"` foot-gun. `subject_count` is
   on the `_CONFLICT_FILTER_EXEMPT_AXES` allowlist — even when the
   positive carries `solo`, the negative-side multi-subject
   suppression must NOT be filtered out (it's intentional
   "intentional-suppression" rather than a conflict). Dropped tokens
   are logged at WARNING.
4. **per-model `prompt.extend.negative_axes`** / `negative_prompt`
   — additive overrides on top of the family taxonomy.
5. **`characters.negative_prompt`** — character-level negatives.

All non-TI segments are comma-split and case-insensitive deduped
via `_keyword_dedup`. Returns empty string when
`family.supports_negative_prompt` is False (Flux / Flux.2).

### Global single-female subject enforcement

Pipeline-wide invariant: every render targets **exactly one adult
female subject**. Multi-subject generation is explicitly deferred —
the constraint is encoded across 7 enforcement layers (mirrors the
existing age-safety pattern) so a drift at any one layer is caught
by the next:

1. **Tier directives** (`categories.yaml` T1-T4) — every
   `llm_directive` carries a SOLO clause. Light-touch at T1/T2
   ("exactly one subject"), strict at T3/T4 (T4 additionally forbids
   the partnered `nsfw_act` tags by name).
2. **Mode + agent SYSTEM_PROMPTs** — `ThemeMode` / `NicheMode` /
   `StyleMode` (PLAN and SCENE templates) + `SeriesPlanner` +
   `SceneGenerator` + `SceneFacetGenerator` all carry a SOLO
   operating principle so the LLM sees the constraint from multiple
   angles. `PONY_BOORU_SYSTEM_PROMPT` additionally forbids
   `2girls / multiple_girls / NSFW_T4_PARTNERED_*` tags explicitly.
3. **`families.yaml::solo_anchor`** (separate from `adult_anchor`)
   — positive-side injection. Booru families (Pony, Illustrious)
   override to `1girl, solo`; SDXL realism inherits the default
   single `solo` token (realism finetunes still recognise booru
   subject vocabulary); Flux / Chroma / Flux.2 use a prose sentence.
   `PromptBuilder._positive_solo_anchor_inject` runs
   unconditionally (unlike `_positive_age_safety_scan`, which fires
   only when an age-ambiguity term is present) and respects
   `family.break_marker` for Pony — the anchor lands in CLIP
   window 2 (post-BREAK) so it stays adjacent to the booru body.
4. **`negative_axes.subject_count`** — 8th axis on the taxonomy.
   Booru families emit `2girls / multiple_girls /
   multiple_subjects / group / crowd`; Chroma emits prose tokens
   (`two people / multiple people / a couple / group / crowd /
   another person`); Flux / Flux.2 stay empty (CFG-distilled
   families ignore negatives). On the `_CONFLICT_FILTER_EXEMPT_AXES`
   allowlist so positive `solo` doesn't trigger
   `filter_conflicts` to drop the negative-side suppression.
5. **`HARD_BLOCK_NEGATIVE` extension** — adds
   `2girls, multiple_girls, multiple_subjects` to the unconditional
   prepend block, so every render with `supports_negative=True`
   carries the multi-subject gate even before family negatives
   are flattened.
6. **`_positive_subject_count_scan`** — composer-side scan that
   strips multi-subject vocab (`2girls`, `multiple subjects`,
   `her partner`, `another woman`, …) from the assembled positive
   if any LLM drifted through layers 1-2. Logs at ERROR so drift
   surfaces in operator visibility rather than silent leakage.
7. **`vocabulary.py::_SOLO_MODE_BANNED_TAGS`** — frozenset of the
   4 partnered T4 `nsfw_act` concept tags. Filtered both from
   `all_concepts_for_family()` (hides them from the LLM's concept
   menu) and from `canonicalize()` (drops them if a stale system
   prompt slips one through). ERROR-logged on drop.

The whole stack is deliberate defence-in-depth: each layer is
narrow on its own but the combination makes single-subject leakage
exceptionally hard to produce. Adding multi-subject support later
means relaxing all 7 layers in lockstep behind an explicit opt-in
flag — never silently.

### LLM Model Awareness (Phase 2 split)

After the per-model prompts work (Phase 2), the family-prompting
hints live on **`SceneFacetGenerator`**, NOT on `SceneGenerator`:

- `SceneGenerator.generate()` — model-agnostic. One LLM call per
  series produces 20-30 scene cores (pose / camera / lighting / env
  / mood). The system prompt does NOT mention trigger words,
  structure rules, or family.guide.
- `SceneFacetGenerator.generate(scene, family, prompt_guide)` —
  one LLM call per (scene, family) produces the family-shaped
  fields (`booru_tags`, `source_tag`, `scene_prose`, `camera_spec`,
  `clothing`). The system prompt inherits the model-aware hints
  that used to live in SceneGenerator:
  - `family.structure_rules` — family prompting shape
  - `family.llm_hint` — one-line guidance per composer
  - `family.avoid_words` + per-model extensions — emitted as "AVOID:"
  - `family.example_prompt` — one concrete example
  - `family.guide` — Flux.2 5-anchor order + 30–80 word band
  - `family.llm_temperature` — per-family default (Pony 0.5,
    Flux/Chroma 0.7, Flux.2 ~0.8)
  - per-model `prompt.extend.trigger_words` — "use naturally when
    they fit"

Output of either generator is a plain `dict` (Pydantic validation
runs, then `model_dump()` hands the dict to downstream consumers).

Because hints live in YAML, adding a new family requires:
1. Add to `config/families.yaml`.
2. Add a `SceneFacet*` Pydantic schema in `src/agents/schemas.py`.
3. Add an entry to `SCENE_FACET_SCHEMA_BY_STYLE`.
4. Add a row to `_SCHEMA_BODY_BY_STYLE` in `scene_facet_generator.py`.

Plus the three hard-coded family touch-points listed under
"Adding a new family" below.

**Pydantic contract.** All `series_planner`, `scene_generator`,
**and `scene_facet_generator`** outputs are parsed through Pydantic
schemas in `src/agents/schemas.py` (`SeriesPlan`, `Scene`,
`SceneList`, **`SceneFacetSDXL` / `Pony` / `Illustrious` /
`FluxNatural` / `Flux2`**). Field types and ranges are checked at
parse time, so a malformed LLM output fails fast with a typed error
path (`OllamaJSONParseError` → retry-with-nudge).

### Per-(scene, target, llm) prompts (Phases 1–5 + 2026-05 family-level)

The DB carries **one prompt per (scene, target_kind, target_id, llm)**
quad rather than the old one prompt per scene. Four coordinated
properties underpin this:

1. **`scenes` is model-agnostic** — only the universal scene core.
   Family-shaped fields no longer live here; they moved to
   `scene_facets`.
2. **`scene_facets` is per-(scene, family, llm)** — sibling models
   in the same family share one row; the per-family LLM expansion
   is reused across them. **Shared between model-kind and family-kind
   prompt prep** — facet generation runs once regardless of how many
   target_kinds consume it.
3. **`prompts.target_kind`** (added 2026-05) is `'model'` or
   `'family'`. `prompts.model_id` is dual-purpose: it carries an
   image-model id when target_kind='model' and a family id when
   target_kind='family'. Both columns are NOT NULL, plus
   `UNIQUE(scene_id, target_kind, model_id, llm_id)` enforces the
   invariant.
4. **Re-rolling** for model-kind requires
   `prepare_prompts --regen-prompts <model>`; for family-kind,
   `prepare_prompts --regen-family-prompts <family>` (separate
   flags so a typo can't cross-delete the other kind). Without
   regen, IntegrityError surfaces with a CLI hint.

This split lets the **same scene concept** feed multiple model
families without re-querying the LLM for the model-agnostic core
(saving ~one LLM call per scene per added model), AND lets the
same scene carry both checkpoint-agnostic family-level prompts
(no per-model overlay) and full per-model prompts simultaneously.
The SDXL-vs-Flux "render the same series two ways" use case is
native:
`prepare_prompts --series-id S --models gonzalomo_flux_v30` adds a
new model-kind row; `prepare_prompts --series-id S --families flux`
adds a new family-kind row. `render_prompts --families flux
--render-with-model gonzalomo_flux_v30` then renders the family-level
prompts through any flux-family checkpoint (validated at parse
time).

### Realism vocabulary library (Phase 4a + 4-bis + v6 creative-uplift + v7 anti-grid cleanup)

`config/prompt_vocabulary.yaml` is the single source of truth for
realism + NSFW + environment + narrative + aesthetic + composition
phrasing per family. The LLM emits abstract concept tags (e.g.
`LIGHT_REMBRANDT`, `CAMERA_SONY_A7RV`, `FILM_PORTRA_400`,
`ENV_TUSCAN_VILLA_RENAISSANCE`, `ATM_DUST_MOTES_IN_LIGHT`,
`NARR_READING_LETTER_AT_DAWN`, `PALETTE_BAROQUE_CARAVAGGIO`,
`PHOTOG_HELMUT_NEWTON`, `ART_MOVE_FILM_NOIR_1940S`,
`COMP_LEADING_LINES_FLOOR`, `PROP_CHAISE_LOUNGE_VELVET`,
`NSFW_T4_SOLO_DISPLAY`); the canonicalizer in
`src/prompt/vocabulary.py` translates each tag to family-shaped
phrasing at `PromptBuilder` compose time.

**vocab_version 7 (2026-05-20, anti-grid / anti-mirror cleanup):**
After a Cydonia-planned T4 series shipped a 4-panel image-grid
hallucination (caused by the LLM writing "in natural poses across
varying compositions" into the series-level `subject_description`,
which gets injected verbatim into every scene) and four other scenes
in the same series rendered warped mirror reflections, vocab v7
removed six entries that were proven to cause these failure modes:
`NSFW_T4_SOLO_MIRROR`, `PROP_CHEVAL_MIRROR`,
`PROP_VANITY_TRIPTYCH_MIRROR` (its prose literally instructed
"three-panel ... central panel framing her face ... side panels
giving fragments"), `COMP_FRAME_WITHIN_FRAME`,
`COMP_REFLECTION_PRIMARY` (instructed "primary subject visible only
as a reflection ... real subject out of frame"),
`COMP_REFLECTION_SECONDARY` (instructed "subject doubled by
reflection ... doubled presence"). The four mirror-mentioning
environment entries (`ENV_ART_DECO_HOTEL_SUITE`,
`ENV_CLAWFOOT_BATHROOM`, `ENV_TOKYO_LOVE_HOTEL`,
`ENV_BACKSTAGE_DRESSING_ROOM`) had their family prose rewritten to
drop "mirrored ceiling / fogged mirror / makeup mirror / mirrored
side table" mentions. The composition `COMP_OVER_SHOULDER` lost its
"mirror reflection of her face in sharp focus" clause. End-to-end
defence: `HARD_BLOCK_NEGATIVE` (`src/prompt/builder.py:85`) extended
with `grid, collage, diptych, triptych, polyptych, split_screen,
multiple_views, panels, tiled, contact_sheet, frame_within_frame,
mirror, mirrored, reflection, double_exposure`; the positive-side
`_positive_subject_count_scan` extended with the LLM-generated grid
phrases (`varying compositions, various poses, across compositions,
across scenes, throughout the series, diptych of, doubled presence,
doubled by reflection`, etc.); the theme-mode `subject_description`
LLM instruction tightened to forbid those exact words and capped at
18 words to prevent multi-clause variety language from leaking into
every scene's prompt. The seven remaining `NSFW_T4_SOLO_*` acts +
28 props + 15 composition principles still give the LLM plenty of
creative range without the failure modes.

**Six top-level namespaces (vocab_version 7, 2026-05-20):**

* **`realism.{lighting, camera, lens, film_stock, art_style, mood,
  angle, framing}`** — always-on, ~70 concepts. Photo-technical
  axes: camera body, lens spec, film stock, lighting setup, mood,
  art-style reference, camera angle, shot size.
* **`nsfw.{anatomy, posture, act}`** — tier-gated. `anatomy` +
  `posture` at T3_artnude; `act` at T4_explicit. Solo-only pipeline
  filters partnered acts.
* **`environment.{setting, atmosphere, prop}`** (Phase 1 + 4, v6) —
  per-scene location + atmospheric element + named prop pull
  (~95 tags). Tier-required `setting` + `atmosphere` at T3+; `prop`
  is optional polish.
* **`narrative.moment`** (Phase 2, v6) — captured-moment anchor
  (~30 tags). The #1 leverage axis per market research; "she reads
  a letter at dawn" forces window light + chair + envelope +
  stillness in one tag. Tier-required at EVERY tier.
* **`aesthetic.{color_palette, photographer_ref, art_movement}`**
  (Phase 3, v6) — SERIES-level inherited (47 tags). Pinned ONCE by
  `SeriesPlanner`, threaded into every scene via new
  `canonicalize_series_aesthetic` helper. Per-style-profile
  `compatible_*` filter lists narrow the menu to coherent combos
  (no Helmut-Newton + Wes-Anderson-pastel). Pony omits
  photographer_ref + art_movement (no booru equivalents); Pony
  DOES participate in color_palette.
* **`composition.principle`** (Phase 4, v6) — higher-order
  composition pulls beyond angle + framing (~18 tags:
  frame-within-frame, reflection-primary, leading-lines,
  negative-space-dominant, etc.). Optional. Pony omits — booru
  tags carry composition implicitly via positional tags.

The canonicalizer silently drops below-tier concepts (a T2 scene
cannot leak T3 vocabulary), Pony-omitted namespaces, and unknown
tags (LLM drift) — defence-in-depth on top of content-tier guards.

Per-scene canonicalization via `canonicalize_facet(scene_facet,
family_id, content_level=)`; series-level inherited canonicalization
via `canonicalize_series_aesthetic(series_plan, family_id,
content_level=)`. The composer threads both into the prompt body:
series-aesthetic phrases land right after the base_prompt (visual
world established first), scene vocab phrases land after scene
fields (per-scene specifics).

Three benefits over the pre-Phase-4a "LLM invents realism vocabulary
every call" world:

1. **Consistent phrasing.** Two scenes that both pick `LIGHT_REMBRANDT`
   produce byte-identical lighting clauses, regardless of LLM noise.
2. **Per-family tuning lives in one file.** Flux2's BFL-style 5-anchor
   phrasing for Rembrandt ("Rembrandt key light from camera-left at
   45 degrees…") and SDXL's tag-style ("rembrandt lighting, triangle
   of light on cheek, dramatic shadow") sit beside each other, easy
   to diff.
3. **NSFW tier-gating is structural, not editorial.** The canonicalizer
   reads `tier_min:` per concept and drops at the data layer; T4 acts
   never reach a T3 prompt regardless of what the LLM writes.

Pony deliberately omits the camera / lens / film_stock / art_style
namespaces — booru tagging carries those implicitly via
`source_photograph + booru_tags`. The canonicalizer logs INFO and
skips any concept that has no Pony phrasing (LLM drift becomes a
no-op rather than a crash).

The `prompts.vocab_version` column captures which YAML version
produced each row, so a YAML bump preserves audit trail for older
prompts without forcing a re-render.

### Adding a new family

The family list is YAML-driven, but **3 hardcoded touch points** in
src/ need a one-line edit each when a 7th family lands. They aren't
load-bearing for correctness — they silently degrade if missed —
but the user-facing behavior is suboptimal:

| File | What's hardcoded | Effect if missed |
|------|------|------|
| `src/core/aspect_ratio_buckets.py` | per-family megapixel tier map (sdxl=1MP, flux=1.5MP, etc.) | Falls through to base SDXL 1MP defaults; renders work but at wrong megapixel tier |
| `src/postprocess/upscaler.py::_SUPPORTED_FAMILIES` | frozenset of families with `upscale.json` templates | New family silently can't use `postprocess.upscale_enabled` — `Upscaler.__init__` raises eagerly with the family name |
| `src/render/workflow_builder.py::build()` | per-family dispatch (`_build_chroma`, `_build_flux`, `_build_flux2`) | New family without a `_build_<id>` method falls through to the standard SDXL-shaped path; works for SDXL-architecture families, breaks for novel architectures |

The `scene_facets.family` CHECK constraint is **NOT** on this list —
it's regenerated from `families.yaml` at `init_db.py` time, so it
stays in sync automatically (cost: re-init the DB).

---

## 7. Prompt Construction

`src/prompt/builder.py` — `PromptBuilder` class.

### Template Order

```
[character.base_prompt]           ← identity (highest CLIP weight)
[scene.expression]
[scene.pose]
[scene.camera]
[scene.camera_angle]
[scene.lighting]
[scene.environment_detail]
[scene.mood_note]
[style_profile.base_style_keywords]
[extra_keywords]                  ← niche SEO, etc.
[trigger_words]                   ← model-specific quality cues
```

Leading tokens get heavier CLIP weighting, so character identity comes
first. Empty segments are dropped. Duplicate tokens are removed
(case-insensitive, first occurrence kept).

### Token-budget enforcement (Phase C)

After composition, every family runs through
`fit_to_budget(text, max_tokens, tokenizer_id, break_marker)` in
`src/prompt/tokenizer.py`. Three real backends:

- **`clip`** — `open_clip` SimpleTokenizer, used by sdxl / pony /
  illustrious (CLIP-L 77-token window).
- **`t5`** — Hugging Face `T5Tokenizer` for `google/t5-v1_1-base`,
  used by flux (512 tokens) and chroma (256 tokens). Pulls the
  SentencePiece vocab on first call (cached under
  `~/.cache/huggingface`); same vocab as T5-XXL/Schnell so token
  counts match the actual encoders.
- **`heuristic`** — `len(words) × 1.4` fallback for Flux.2 / Qwen3
  (Qwen vocab not bundled).

Trim is from the **middle** so the subject prefix (~30 tokens) and
quality / camera suffix (~20 tokens) survive — CLIP otherwise
silently truncates the *tail*, dropping exactly the lens / lighting
tokens we care about. Trimming is comma-aligned where possible so
keywords aren't cut in half. Pony's `BREAK` marker (set on
`family.break_marker`) splits the prompt into two independent CLIP
windows of 75 tokens each; each side is budgeted independently.

### Mode-Specific Build Methods

| Method | Used by | Lead segment |
|--------|---------|-------------|
| `build_character_prompt` | CharacterMode | `character.base_prompt` |
| `build_theme_prompt` | ThemeMode | `subject_description` or `subject_detail` |
| `build_style_prompt` | StyleMode | `style_keywords, subject_detail` |
| `build_niche_prompt` | NicheMode | `subject_bias`, extra_keywords = top 3 SEO |
| `build_variation_prompt` | VariationMode | `base_scene.environment_detail` |

All delegate to `build_one()` which handles assembly, style dispatch,
negative prompt override, and SHA256 hashing.

---

## 8. Aspect Ratio & Resolution

`src/core/ratio_selector.py` — `RatioSelector` class.

### Four Ratio Buckets

| Key | SDXL Default | Flux / Chroma | Flux.2 Klein |
|-----|-------------|---------------|--------------|
| `portrait_23` | 768 × 1152 | 832 × 1216 | 832 × 1216 |
| `portrait_916` | 768 × 1366 | 768 × 1366 | 832 × 1480 |
| `square` | 1024 × 1024 | 1024 × 1024 | 1024 × 1024 |
| `landscape` | 1152 × 768 | 1216 × 832 | 1216 × 832 |

Per-family megapixel buckets (1MP / 1.5MP / 2MP tiers) live in
`src/core/aspect_ratio_buckets.py`. Per-model `resolution_*`
overrides in `config/models/{id}.yaml` win over both.

### Selection Algorithm — 5-axis additive scoring (Phase A)

```
score(ratio) = base[content_level][ratio]                   # tier weights
             + audience_bonus[audience][ratio]              # DA / Patreon
             + composition_bonus[intent][ratio]             # close-up / medium / full-body / wide
             + pose_signal_bump[ratio] (× scene-keyword match)
             + family_quality_bonus[family][ratio]          # SDXL hates 9:16, etc.
clamp to [0.05, 2.0], drop ≤0, normalize, weighted draw.
```

Inputs:
- **Content-level base weights** — `pipeline.yaml::aspect_ratio_weights.{T1..T4}`.
  T1/T2 skew tasteful (variety); T3/T4 skew portrait (full-body
  nudes and explicit framing sell tall).
- **Audience bonus** — `config/ratio_signals.yaml::audience_bonus`
  keyed on `select(audience=...)`. DA's grid favors classic 2:3
  portraits; Patreon's mobile feed rewards 9:16 vertical.
- **Composition-intent bonus** — read from the optional
  `scene["composition_intent"]` field (close-up / medium /
  full-body / wide), which `SceneGenerator` produces when the
  family's `llm_hint` requests it.
- **Pose signal bump** — `+0.15` (configurable) for any ratio whose
  `signals` list matches a keyword in the scene text. One bump per
  ratio per scene.
- **Family-quality bonus** — `family_quality_bonus[family][ratio]`.
  SDXL/Pony/Illustrious dislike 9:16 (UNet off-distribution);
  Flux/Chroma flat; Flux.2 favors portrait.

**Hard overrides** survive unchanged and short-circuit the score:
- `pose ∋ "full body"|"standing"` → `portrait_23`.
- `pose ∋ "reclining"|"lying"` → `landscape`.

The weighted draw uses a deterministic per-scene RNG seeded by
`SHA1(content_level || audience || family || scene_text)` so the
same scene under the same audience+family always picks the same
ratio (reproducible tests, replay after config tweak).

### Resolution Precedence

1. Per-model `resolution_portrait/square/landscape` from
   `config/models/{id}.yaml`, surfaced via
   `model_resolution_overrides()` in `ratio_selector.py`.
2. Per-family megapixel bucket via
   `src/core/aspect_ratio_buckets.py::get_family_resolution(family,
   ratio)` — Flux/Flux.2 default to 1.5MP, Chroma 1MP, SDXL/Pony/
   Illustrious 1MP.
3. `RATIO_TO_RESOLUTION` dict (base SDXL 1MP defaults).

---

## 9. LLM Agents

Three LLM agents, all using `OllamaClient` (Ollama `/api/generate`):

### SeriesPlanner (`src/agents/series_planner.py`)

- **Input**: character name, base_prompt, vibe, style info, content
  level, content rules, previous themes.
- **Output**: JSON with `theme`, `mood`, `environment`, `variation_axes`.
- **Temperature**: 0.6 (structured output).
- **Retries**: 3 attempts if JSON parse fails or required fields missing.

### SceneGenerator (`src/agents/scene_generator.py`)

- **Input**: series plan, content level, allowed pose types, scene count.
- **Output**: Pydantic-validated `Scene` objects via `SceneList =
  RootModel[list[Scene]]` (`src/agents/schemas.py`). 7 required
  fields (`variation_axis`, `pose`, `camera`, `camera_angle`,
  `lighting`, `environment_detail`, `mood_note`) plus 9 optional
  fields populated when the family's `llm_hint` requests them:
  `expression`, `composition_intent`
  (close-up/medium/full-body/wide — feeds Phase A ratio scoring),
  `framing_hint`, `audience_target`
  (deviantart/patreon/either — feeds Phase A audience bonus),
  `camera_spec` (sdxl primary), `clothing` (sdxl primary),
  `booru_tags` (pony/illustrious primary), `source_tag` (pony only),
  `scene_prose` (flux/chroma/illustrious/flux2 primary).
- **Temperature**: from `family.llm_temperature` if set, else 0.6.
  Pony tends to want ~0.5 for tag predictability; Flux.2 ~0.8 for
  varied prose.
- **Retries**: 3 attempts. Pydantic errors are inlined into the
  retry nudge so the LLM sees exactly which field failed.
- **Model awareness**: `_build_system_prompt()` appends prompt guide
  rules (trigger words, avoid words, structure rules, example
  prompts, family-specific guide block such as Flux.2's 5-anchor
  ordering) to the base system prompt.

### MetadataGenerator (`src/agents/metadata_generator.py`)

- **Input**: theme, mood, environment, character name, content level,
  image count, style keywords.
- **Output**: JSON with `title` (≤80 chars), `description` (2-3
  sentences), `tags` (15-25 tags).
- **Temperature**: 0.6.
- Content-level aware: T1/T2 → aesthetic / tasteful copy;
  T3/T4 → bolder, more direct copy.
- Called in Phase C with a brief LLM reload.

### OllamaClient (`src/agents/llm_client.py`)

- `generate()` — free-form text (temperature 0.7).
- `generate_json(schema=...)` — text + markdown fence stripping +
  JSON parse (temperature 0.6). When `schema` is passed (a Pydantic
  `BaseModel` subclass), output is validated via
  `schema.model_validate_json(cleaned)` instead of bare
  `json.loads`. Existing call sites that don't pass `schema` keep
  raw-dict semantics.
- `unload_model()` — `keep_alive: 0` + 2 s sleep.
- `is_available()` — health check via `/api/tags`.
- Config from `pipeline.yaml` → `llm:` block. Default model:
  `dolphin-mixtral:8x7b`, fallback: `dolphin-llama3:8b`.

---

## 10. ComfyUI Integration

### WorkflowBuilder (`src/render/workflow_builder.py`)

Loads workflow JSON templates from `config/comfyui_workflows/{family}/`,
caches them, and returns deep copies with per-render values injected.

Templates use **semantic node IDs** (renamed from ComfyUI's numeric
IDs):

**Base template required nodes**: `load_checkpoint`, `positive_prompt`,
`negative_prompt`, `empty_latent`, `ksampler`.

**IPAdapter template adds**: `ipadapter_unified_loader`,
`ipadapter_apply`, `load_reference_image`.

**Upscale template** (Phase F, sdxl/pony/illustrious only):
`load_image`, `upscale_model_loader`, `upscale_with_model`,
`downscale_to_target` (`ImageScaleBy(0.35)` → 1.4× source),
`save_image`. Pure ESRGAN — no prompt encoder, no sampler,
deterministic.

Injected values:
- `empty_latent` → width, height, batch_size
- `positive_prompt` → prompt text
- `negative_prompt` → negative text (conditional — skipped if node absent)
- `load_checkpoint` → checkpoint filename
- `ksampler` → sampler, scheduler, steps, cfg, seed
- LoRA loaders (`lora_loader_0`, `lora_loader_1`) → name, strength
- IPAdapter nodes → reference image filename, weight (default 0.7)

**Per-family dispatch**:

- `sdxl` / `pony` / `illustrious` — standard `build()` path:
  `CheckpointLoaderSimple` + `KSampler` (Pony adds
  `CLIPSetLastLayer` for `clip_skip=2`).
- `flux` — `_build_flux()` — `UnetLoaderGGUF` +
  `ModelSamplingFlux` + `FluxGuidance` + standard `KSampler`.
- `chroma` — `_build_chroma()` — `UnetLoaderGGUF` +
  `ModelSamplingAuraFlow` + `CFGGuider` +
  `SamplerCustomAdvanced` + `BetaSamplingScheduler` +
  `RandomNoise`.
- `flux2` — `_build_flux2()` — `UNETLoader` +
  `CLIPLoader(type=flux2, clip_name=qwen_3_8b...)` + `KSampler`.
  **Distilled-contract clamp**: cfg forced to `1.0`, `steps≤6`
  (warn-and-clamp to 4), `sampler=euler`, `scheduler=simple`.
  Any override emits a WARNING — Klein 9B is step-distilled and
  guidance-distilled and melts under standard sampler settings.

**Capability guards**: IPAdapter request + `supports_ipadapter=False` →
raises. LoRA stack + `supports_lora=False` → raises. Max 2 LoRAs.
Upscaler request for an untemplated family (flux/chroma/flux2) →
`Upscaler.__init__` raises eagerly (`_validate_template_path()`
against `_SUPPORTED_FAMILIES = {sdxl, pony, illustrious}`); flipping
`upscale_enabled: true` for a wrong family fails at construction,
not mid-render.

**External templates (`template_override` / `--template`)**: For
user-authored or community-sourced ComfyUI workflows, callers can pass
`template_override="templates/{family}/{name}.json"` to
`PipelineEngine.__init__` (wired from `--template` on `run_once.py`,
`render_set.py`, `compare_models.py`). Under this path,
`WorkflowBuilder.build_external` is used instead of `build()`; the
pipeline validates four semantic node IDs (`positive_prompt`,
`negative_prompt`, `ksampler`, `empty_latent`) + their required input
fields at preflight, and injects ONLY prompt/negative/seed/resolution
— the template's baked-in checkpoint, VAE, CLIP, LoRAs, sampler, and
post-processing all run as authored. IPAdapter is forced off.
`--model` still drives Phase A prompt style but does not override the
template's checkpoint. See `docs/COMFYUI_WORKFLOWS.md § External
templates` for the full contract.

### ComfyUIClient (`src/render/comfyui_client.py`)

HTTP client for the ComfyUI API:

- `queue_prompt(workflow)` → POST `/prompt`, returns `prompt_id`.
- `wait_for_completion(prompt_id, timeout)` → polls
  `/history/{prompt_id}`, returns `list[RenderedImage]`.
- `render_single_with_retry(workflow, max_attempts)` → submit + wait
  with retry on `RenderTimeout` only.

File path resolution: ComfyUI's history returns relative filenames. The
client resolves them against `output_dir` (configurable, default
`~/AI/apps/ComfyUI/output/`). A file-existence guard polls up to 5
times with 1 s intervals to handle the race where ComfyUI reports done
before the file is flushed to disk.

Cached-prompt detection: if ComfyUI served every node from cache
(identical inputs), the client raises with a specific message rather
than a generic "no images" error.

### Workflow Templates

```
config/comfyui_workflows/
  sdxl/
    base.json          # SDXL t2i, lora_loader_0/1 slots
    ipadapter.json     # SDXL + IPAdapter unified loader / apply
    upscale.json       # Phase F: pure ESRGAN 4× → 0.35 (1.4× source)
  pony/
    base.json          # Pony + CLIPSetLastLayer (clip_skip 2)
    ipadapter.json     # Pony + IPAdapter
    upscale.json       # Phase F: pure ESRGAN, prefix=upscale_pony
  illustrious/
    base.json          # Illustrious XL (SDXL-shaped)
    ipadapter.json     # Illustrious + IPAdapter
    upscale.json       # Phase F: pure ESRGAN, prefix=upscale_illustrious
  flux/
    base.json          # FLUX.1 GGUF + ModelSamplingFlux + FluxGuidance
  flux2/
    base.json          # FLUX.2 Klein 9B — UNETLoader + CLIPLoader(type=flux2)
                       # + KSampler clamped to cfg=1, steps=4, euler+simple
  chroma/
    base.json          # GGUF + SamplerCustomAdvanced + BetaSampling
  templates/           # user-provided external workflow templates
    chroma/
      chroma_done_properly.json   # community external (--template flag)
```

13 active templates total: 9 per-family base/ipadapter + 3 upscale +
1 external. Chroma uses a fundamentally different node graph from
SDXL/Pony: three separate loaders (UnetLoaderGGUF for GGUF Q8_0
UNET, CLIPLoader for T5, VAELoader), CFGGuider instead of KSampler,
BetaSamplingScheduler, RandomNoise, and SamplerCustomAdvanced.
WorkflowBuilder dispatches to `_build_chroma()` for the `chroma`
family, which injects into 8 nodes instead of the SDXL/Pony pattern
of `load_checkpoint` + `ksampler`. Requires city96's ComfyUI-GGUF
extension. Flux.2 routes through `_build_flux2()` with the
distilled-contract clamp described above.

---

## 11. Image Quality Scoring

`src/scoring/image_scorer.py` — `ImageScorer` class.

### Six Signals + Resolution Check

| Signal | Method | Range | Source | Phase |
|--------|--------|-------|--------|-------|
| `hps_v2` | HPS v2 (CLIP-H + classifier head) | [0, 1] | `HPSv2Predictor` | G (opt-in) |
| `image_reward` | BAAI ImageReward (CLIP-H + reward head) | ~[-2.4, +1.0] | `ImageRewardPredictor` | G (opt-in) |
| `aesthetic` | LAION CLIP ViT-L/14 + MLP head | [0, 10] | `AestheticPredictor` | original |
| `face_confidence` | insightface buffalo_l `det_score` | [0, 1] | `_FaceAnalyzerWrapper` | original |
| `blur` | OpenCV Laplacian variance | [0, 500 clamped] | `cv2.Laplacian` | original |
| `resolution` | h ≥ 1024 AND w ≥ 768 | binary | pixel check | original |

### Composite Formula — Phase G default

```
composite = 0.30 · hps_v2_norm
          + 0.25 · image_reward_sigmoid
          + 0.20 · aesthetic / 10
          + 0.10 · face_confidence
          + 0.10 · min(blur, 500) / 500
          + 0.05 · resolution_pass
```

`image_reward_sigmoid(raw) = 1 / (1 + exp(-raw / 2))` (centered at
0, divisor 2 → -2 → ~0.12, +2 → ~0.88). `hps_v2_norm` is identity
clamp to [0, 1].

### Composite Formula — legacy fallback

When **both** Phase-G flags (`scoring.use_hps_v2`,
`scoring.use_image_reward`) are off, `ImageScorer.__init__`
auto-selects `CompositeWeights.legacy()`:

```
composite = 0.40 · aesthetic / 10
          + 0.25 · min(blur, 500) / 500
          + 0.25 · face_confidence
          + 0.10 · resolution_pass
```

This keeps the scoring path back-compat for users who haven't
installed the optional `hpsv2` / `image-reward` packages.

### Weight redistribution

When a Phase-G signal is `None` for an individual image (predictor
disabled, weights missing, prompt omitted), `_compose` redistributes
that signal's weight by dividing the used numerator by the active
denominator. Composite stays in [0, 1] regardless of which signals
fired — no zero-collapse, no scale shift.

### Quality Flags

| Flag | Condition |
|------|-----------|
| `low_aesthetic` | aesthetic < 4.5 |
| `blurry` | blur < 80 |
| `no_face` | no face detected |
| `multiple_faces` | >1 faces detected |
| `low_hps_v2` | hps_v2 < 0.20 (only when HPS v2 enabled) |
| `low_image_reward` | image_reward < -1.5 (only when ImageReward enabled) |

### Prompt-conditioning

HPS v2 and ImageReward both score image-vs-prompt alignment, not the
image alone. `score(path, prompt=...)` accepts the positive prompt
explicitly; `score_batch` reads `prompt_text` from each img dict
(populated by `engine.py` during render). When prompt is missing
those two signals are skipped and the composite falls back to the
legacy weighting **for that one image** — no error, just degraded
scoring.

### Graceful degradation

When a flag is on but the predictor fails to load (e.g. `hpsv2`
package not installed, weights missing, network down on first call),
`ImageScorer` logs once at WARNING and silently disables that signal
for the rest of the run via `_hps_v2_disabled` /
`_image_reward_disabled`. The pipeline doesn't crash mid-batch over
a missing scorer.

### Model loading

Models are lazy-loaded on first `score()` call and cached for the
process lifetime. CLIP runs on MPS (Apple Silicon GPU); insightface
uses CPU onnxruntime; HPS v2 + ImageReward use torch and inherit the
same MPS/CPU device pick. Total VRAM at full Phase-G is ~3.5GB
(CLIP-L for aesthetic + CLIP-H for HPS + CLIP-H for ImageReward —
each backbone loads its own copy).

---

## 12. Post-Processing

### Watermarker (`src/postprocess/watermarker.py`)

Semi-transparent text watermark on T1/T2 ("preview-tier") exports
intended for free posting. T3/T4 ("sale-tier") exports go out
unwatermarked because they ship to paid platforms.

Config from `pipeline.yaml`:
```yaml
watermark:
  enabled: true
  text: "@YourDAHandle"
  position: "bottom-right"
  opacity: 0.3
```

### Upscaler (`src/postprocess/upscaler.py`)

Pure-ESRGAN 4× upscale via ComfyUI workflow, downscaled by 0.35 to
land at 1.4× source. Disabled by default
(`postprocess.upscale_enabled: false`). **Phase F** shipped templates
for `sdxl`, `pony`, and `illustrious` only — `flux`, `chroma`,
`flux2` already render natively at 1024+ and the post-hoc upscale
adds marginal value, so no template ships for them.
`Upscaler.__init__` runs `_validate_template_path()` against
`_SUPPORTED_FAMILIES = {sdxl, pony, illustrious}`; flipping the flag
on for an untemplated family raises `UpscaleError` at construction,
not mid-render. Each `upscale.json` requires three semantic nodes
(`load_image`, `upscale_model_loader`, `save_image`) or
`_load_template()` raises with a clear error. The chain is
deterministic — `LoadImage → UpscaleModelLoader → ImageUpscaleWithModel
→ ImageScaleBy(0.35) → SaveImage` — no prompt encoder, no sampler,
not a denoise-based hires-fix.

### FaceRefiner (`src/postprocess/face_refiner.py`)

FaceDetailer re-render via ComfyUI workflow (Impact Pack node).
Disabled by default (`postprocess.face_refine_enabled: false`).
Requires `config/comfyui_workflows/{family}/face_detail.json`
template (not yet built — the module exists but is gated until a
template is authored).

---

## 13. Character Management

`src/memory/character_manager.py` — `CharacterManager` class.

Three operations:
- `get_character(id)` — direct lookup.
- `get_least_recently_used()` — LRU rotation. NULL `last_used_at` wins
  (never-used character). Tiebreaker: `created_at ASC`.
- `update_usage(id, image_count)` — increments
  `total_images_generated`, sets `last_used_at`.

Character JSON identity files live in `characters/{char_id}/`. Loaded
into DB by `scripts/bootstrap_character.py`. Key fields:

- `base_prompt` — **immutable** after creation (DB trigger enforced).
- `negative_prompt` — character-specific negatives.
- `reference_image_path` — for IPAdapter identity lock.
- `locked_features` — JSON list of features that must not change.
- `allowed_shift_axes` — JSON list of axes the LLM can vary.
- `outfit_pool` — JSON list of outfit options.

---

## 14. Anti-Repetition

`src/memory/memory_manager.py` — `MemoryManager` class.

Records themes, scenes, and prompts to `generation_memory` via SHA256
content hashes. `is_novel()` checks the hash before recording.

Character mode checks scenes across ALL 4 content tiers for the same
`character_id` (subscribers see every tier; cross-tier scene reuse
breaks the illusion of distinct content).

`src/prompt/deduplicator.py` — `PromptDeduplicator` class.

Two strategies:
1. **Prompt hash** — exact match via SHA256 (threshold 0.9).
2. **Scene structural similarity** — compares environment + lighting +
   camera. Uses `sentence-transformers` when available, falls back to
   hash-based approach. Threshold 0.75.

---

## 15. Configuration

### pipeline.yaml

`config/pipeline.yaml` — **15 top-level sections**:

```yaml
pipeline:               # runs_per_day, output_dir, db_path,
                        #   default_model_id, default_style_profile_id
execution:              # mode (manual|supervised|automated),
                        #   pause_after_dryrun, pause_after_filter
mode_weights:           # character: 0.60, theme: 0.20, niche: 0.10,
                        #   style: 0.05, variation: 0.05
content_level_weights:  # T1_suggestive: 0.10, T2_implied: 0.35,
                        #   T3_artnude: 0.45, T4_explicit: 0.10
aspect_ratio_weights:   # per tier (T1/T2/T3/T4) → {portrait_23,
                        #   portrait_916, square, landscape}
generation:             # batch_size
set_builder:            # min_images: 10, max_images: 25,
                        #   quality_cutoff: 0.55
scoring:                # use_hps_v2: false (Phase G opt-in)
                        # use_image_reward: false (Phase G opt-in)
                        # composite_weights:
                        #   {hps_v2: 0.30, image_reward: 0.25,
                        #    aesthetic: 0.20, face: 0.10,
                        #    blur: 0.10, resolution: 0.05}
                        # legacy_weights:
                        #   {aesthetic_weight: 0.40, blur_weight: 0.25,
                        #    face_weight: 0.25, resolution_weight: 0.10}
dedup:                  # prompt_similarity_threshold: 0.9,
                        #   scene_structural_threshold: 0.75
llm:                    # model, fallback_model, base_url,
                        #   unload_after_phase, keep_alive_seconds
comfyui:                # base_url, output_dir, input_dir,
                        #   render_timeout_seconds, max_retry_per_image,
                        #   workflow_dir
watermark:              # enabled, text, position, opacity (T1/T2 only)
postprocess:            # upscale_enabled (Phase F: sdxl/pony/illustrious
                        #   templates), face_refine_enabled,
                        #   upscale_model, face_denoise
variation_mode:         # axis_weights, multi_base.{enabled, max_bases,
                        #   min_source_quality}
compliance:             # commercial_mode (drops NCL-licensed models at
                        #   ModelRegistryLoader load when true)
```

### Generation Context

`src/core/generation_context.py` — `GenerationContext` dataclass.

Built once at the top of `run_cycle()`. Contains: `mode`,
`content_level`, `execution_mode`, `style_profile`, `content_rules`,
`model_id`, `model_config` (`ModelRegistryEntry`),
`model_prompt_guide` (effective `ModelPromptGuide` — family +
per-model `prompt.extend`/`override` merge), `family`
(`FamilyConfig`), `character`, `character_id`, `db_path`.

Properties: `family` (the `FamilyConfig` dataclass — carries
`prompt_style`, `supports_negative_prompt`, `supports_weighting`,
`clip_skip`, quality prefix/suffix, negatives), `supports_ipadapter`,
`supports_lora`.

Method: `augment_system_prompt(base)` — injects family rules,
structure hints, example prompts, and avoid-word warnings into LLM
system prompts.

---

## 16. File Structure

```
nsfw-pipeline/
├── config/
│   ├── pipeline.yaml                      # 15-section runtime config
│   ├── families.yaml                      # 6 prompt families
│                                          #   (sdxl/pony/illustrious/flux/chroma/flux2)
│   ├── style_profiles.yaml                # 10 aesthetic archetypes
│   ├── categories.yaml                    # themes/styles/niches + 4-tier content rules
│   ├── ratio_signals.yaml                 # Phase A: audience/composition/family bonuses
│   ├── models/                            # 17 per-model YAML profiles
│   ├── comfyui_workflows/
│   │   ├── sdxl/{base,ipadapter,upscale}.json
│   │   ├── pony/{base,ipadapter,upscale}.json
│   │   ├── illustrious/{base,ipadapter,upscale}.json
│   │   ├── flux/base.json                 # FLUX.1 GGUF + ModelSamplingFlux + FluxGuidance
│   │   ├── flux2/base.json                # Phase F: distilled-contract template
│   │   ├── chroma/base.json               # GGUF + SamplerCustomAdvanced + BetaSampling
│   │   └── templates/                     # external (community) templates
│   │       └── chroma/chroma_done_properly.json
│   └── workflow_node_maps/                # 6 semantic-rename YAMLs
├── scripts/                               # 15 user-facing CLI scripts
│   ├── init_db.py                         # create 10-table schema + trigger (no migrations during pre-stable)
│   ├── regenerate_regression_fixtures.py  # Phase H: bake test fixture baseline
│   ├── bootstrap_character.py             # load identity.json into characters
│   ├── render_set.py                      # manual set render (hand-written scenes)
│   ├── run_once.py                        # full single-model cycle (phase_a→_b→_c)
│   ├── prepare_prompts.py                 # Phase 4: Phase A only — multi-model fan-out + DB persist
│   ├── render_prompts.py                  # Phase 4: Phase B+C — DB→render per model
│   ├── dry_run.py                         # Phase A only (LLM, no render)
│   ├── compare_models.py                  # side-by-side: --prompt|--character|--series-id|--scene-id
│   ├── list_models.py                     # registry dump
│   ├── map_reference.py                   # copy reference image to ComfyUI input
│   ├── rename_workflow_nodes.py           # numeric → semantic node IDs
│   ├── test_comfyui.py                    # smoke ComfyUI connection
│   ├── test_llm.py                        # smoke Ollama connection
│   └── test_scorer.py                     # smoke image scorer
├── src/
│   ├── main.py                            # supervised/automated scheduler entry
│   ├── core/
│   │   ├── engine.py                      # PipelineEngine — 4 public methods:
│   │   │                                  #   run_cycle (single-model wrapper)
│   │   │                                  #   run_phase_a (LLM, multi-model fan-out)
│   │   │                                  #   run_phase_b (render + score + postprocess)
│   │   │                                  #   run_phase_c (set + watermark + export + persist + log)
│   │   ├── generation_context.py
│   │   ├── mode_selector.py               # weighted-random mode selection
│   │   ├── ratio_selector.py              # Phase A: 5-axis additive scoring
│   │   ├── aspect_ratio_buckets.py        # Phase A: per-family megapixel buckets (HARDCODED — see §6 Adding a new family)
│   │   ├── content_level.py               # 4-tier definitions
│   │   ├── merge_overrides.py             # YAML extend/override semantics
│   │   └── style_profile_adapter.py
│   ├── agents/
│   │   ├── llm_client.py                  # OllamaClient + generate_json(schema=...)
│   │   ├── series_planner.py
│   │   ├── scene_generator.py             # Phase 2: model-agnostic core only
│   │   ├── scene_facet_generator.py       # Phase 2: per-(scene, family) LLM expansion
│   │   ├── metadata_generator.py
│   │   ├── character_creator.py
│   │   └── schemas.py                     # Pydantic: SeriesPlan, Scene, SceneList,
│   │                                      #   SceneFacet{SDXL,Pony,Illustrious,FluxNatural,Flux2}
│   ├── prompt/
│   │   ├── builder.py                     # 5-composer dispatcher + 5-layer negative
│   │   ├── tokenizer.py                   # Phase C: CLIP/T5/heuristic + fit_to_budget
│   │   ├── negative_axes.py               # Phase D: 7-axis taxonomy + conflict filter
│   │   ├── sanitizer.py                   # 4-tier suppress/boost
│   │   └── deduplicator.py
│   ├── render/
│   │   ├── comfyui_client.py
│   │   └── workflow_builder.py            # per-family dispatch incl. _build_flux2 clamp (HARDCODED dispatch — see §6 Adding a new family)
│   ├── scoring/
│   │   └── image_scorer.py                # Phase G: 6-signal composite, opt-in HPS+IR
│   ├── memory/
│   │   ├── character_manager.py
│   │   ├── memory_manager.py
│   │   ├── model_registry.py              # 17-model registry + commercial gate
│   │   ├── family_loader.py               # 6-family loader
│   │   ├── style_profile_loader.py
│   │   ├── categories_loader.py
│   │   └── scene_facets_repo.py           # Phase 1: scene_facets CRUD (per-family LLM expansion)
│   ├── filter/
│   │   ├── set_builder.py
│   │   └── level_purity_check.py
│   ├── modes/
│   │   ├── base_mode.py
│   │   ├── character_mode.py
│   │   ├── theme_mode.py
│   │   ├── style_mode.py
│   │   ├── niche_mode.py
│   │   └── variation_mode.py
│   ├── postprocess/
│   │   ├── watermarker.py                 # T1/T2 watermark
│   │   ├── upscaler.py                    # Phase F: eager template validation; _SUPPORTED_FAMILIES is HARDCODED — see §6 Adding a new family
│   │   └── face_refiner.py                # gated; needs face_detail.json template
│   ├── export/
│   │   └── exporter.py                    # output/{tier}/{series_id}/ tree
│   ├── review/
│   │   ├── supervisor.py                  # supervised pause points
│   │   └── contact_sheet.py
│   └── analytics/
│       └── __init__.py                    # placeholder (engagement loop pending)
├── tests/                                 # 607 in tests/ (537 functions × parametrize; +8 smoke tests under scripts/)
│   ├── _regression_harness.py             # Phase H: shared compute_expected
│   ├── fixtures/regression/               # Phase H: 6 families × 3 cases
│   │   └── {sdxl,pony,illustrious,flux,chroma,flux2}/
│   │       └── case_{1..3}.{input,expected}.yaml
│   ├── integration/
│   │   └── test_ipadapter.py              # IPAdapter A/B diagnostic (skip-by-default; needs ComfyUI)
│   └── test_*.py                          # 30+ test files
├── characters/                            # identity.json + reference images
├── models/aesthetic/                      # LAION predictor MLP weights
├── output/                                # exported sets, comparisons, IPAdapter A/B
├── docs/COMFYUI_WORKFLOWS.md              # workflow build walkthrough
├── nsfw_pipeline.db                       # SQLite (10 runtime tables)
├── requirements.txt
├── ARCHITECTURE.md                        # this file
├── CLAUDE.md                              # Claude Code rules
└── PROJECT_GUIDE.md                       # operations manual
```

---

## 17. Hard Invariants

These rules are enforced in code and must never be violated:

1. **LLM and ComfyUI never run simultaneously.** `unload_model()` is
   called between Phase A and B, and after Phase C metadata generation.

2. **`base_prompt` is immutable.** DB trigger prevents UPDATE. Create a
   new character version instead.

3. **Content levels never mix within a series.** `assert_level_purity()`
   validates every image in a set matches the series content level.

4. **Max 2 LoRAs per render.** `WorkflowBuilder` raises if
   `len(lora_stack) > 2`.

5. **Capability guards are fatal, not silent.** IPAdapter request on a
   model that doesn't support it → raises. LoRA stack on a model that
   doesn't support LoRAs → raises. No silent downgrades.

6. **Seeds are 32-bit unsigned integers.** `seed=None` → random,
   `seed=0` is a valid honored seed (not treated as "no seed").

7. **Static config lives in YAML; the DB is runtime-mutable only.**
   Models, families, style profiles, and categories are declarative
   YAML under `config/`. Do not add new static-only tables. Cross-refs
   from DB → YAML (`characters.style_profile_id`, `images.model_id`)
   are plain TEXT columns validated at startup by the YAML loaders.

8. **FLUX.2 distilled contract is clamped at workflow build.**
   `WorkflowBuilder._build_flux2` forces `cfg=1.0`, `steps≤6`
   (warn-and-clamp to 4), `sampler=euler`, `scheduler=simple`. Any
   override is overwritten with a WARNING — Klein 9B is
   step-distilled and guidance-distilled and melts under standard
   sampler settings. This invariant lives at the workflow layer so
   even hand-edited templates can't bypass it.

9. **Commercial-licensed-only mode is a registry-load gate.** When
   `pipeline.yaml::compliance.commercial_mode: true`,
   `ModelRegistryLoader` refuses any model whose YAML declares
   `commercial_use: false` (any FLUX NCL-licensed checkpoint would
   land here). The gate raises `ModelNotFound` at startup, not at
   render — protects paid-tier exports (DA Premium, Patreon, Fanvue)
   from accidentally feeding through a non-commercial checkpoint.
   No NCL-licensed models are currently registered; the gate remains
   active for future additions.

10. **One prompt per (scene, model) — DB-enforced uniqueness.**
    `prompts.UNIQUE(scene_id, model_id)` blocks accidental
    double-insert during the multi-model fan-out. Re-rolling on the
    same model requires explicit
    `prepare_prompts.py --regen-prompts <model>`, which DELETEs first
    so the re-INSERT can land. Plain INSERT without `--regen-prompts`
    surfaces the IntegrityError at the CLI level with a hint pointing
    at the right command.
