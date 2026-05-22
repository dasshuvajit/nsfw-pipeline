# Plan — Creative Uplift: Diverse Backgrounds, Color Contrast, Eye-Stunning Output

> **Status:** DRAFT for user approval after verifier-agent review.
> **No code changes until user green-lights.**
> **Author:** Claude + research-agent + verifier-agent (TBD).
> **Date:** 2026-05-19.

---

## 1. Problem statement (verbatim user complaint + diagnosis)

User reports rendered images are **"boring, non-creative, static boring backgrounds, no diversity, can't generate any profit selling these"**. Wants:
- Artistic, aesthetic, eye-stunning, mind-blowing images
- Strong color contrast + color variety across scenes
- Background variety + creative atmospheric details
- Intelligent placement (not random) — each element matches the chosen scene/mood
- Coverage for **every category**, not just boudoir

**Root-cause diagnosis** (from codebase analysis on `series_4e981ac4c0d9`):

1. **SeriesPlanner locks ONE `environment` per series.** `environment = "Sun-drenched bedroom with floor-to-ceiling windows..."` → all 24 scenes constrained to that one bedroom.
2. **SceneGenerator system prompt enforces single-location.** Explicit rule (line 141): `"Environment details should feel like they belong in the same location"`.
3. **Vocabulary is photo-technical-only.** 9 namespaces cover camera/lens/film/lighting/mood/art_style/angle/framing/nsfw. **Zero** namespaces for: environments, props, atmospheric elements, color palettes, photographer references, art movements, narrative moments, advanced composition.
4. **Categories are narrow.** "Boudoir Lingerie" → "Intimate bedroom / studio" → variations `[silk robe, lace bodysuit, vintage stockings]`. The category encodes "bedroom" into the very setup.
5. **DB evidence of the failure mode** (`series_4e981ac4c0d9.scenes[0-7].environment_detail`): `white linen sheets / pillow / floor-to-ceiling window / sheer fabric / white duvet / mirror / thick carpet / bed frame` — every single one is bedroom furniture.

**Top-3 research findings** (from parallel agent's 18-source review):

- **#1 by leverage: `narrative_moment`** as a required scene field. "She reads a letter at dawn" forces window + chair + envelope + stillness → solves 5 axes from one tag. Editorial photography convention (Hegre, Crewdson, Newton) is built on this.
- **#2: environment + prop + atmosphere namespaces** (~95 tags total) — the structural backbone of visual interest, entirely absent from current vocab.
- **#3: series-level inheritance of `color_palette` + `photographer_ref` + `art_movement`** (~50 tags) — gives the series a coherent "signature look" (the commercial differentiator every top Patreon boudoir creator has). Vary location at scene level; hold aesthetic constant at series level — matches editorial convention.

**Bonus finding:** quality magic words (`masterpiece, best quality, 8k, ultradetailed, intricate, cinematic, sharp focus`) **measurably degrade Chroma/Flux/Flux2 output** because their T5 encoder is trained on natural language. Strip them from those three families' quality prefix/suffix. SDXL/Pony/Illustrious still benefit from them — keep.

---

## 2. Design principles

The plan adheres to three principles surfaced from the research:

1. **Series-level coherence + scene-level diversity.** Aesthetic (palette / photographer / art-movement / film-stock / lighting style) is pinned ONCE on the series and inherited by every scene. Location / props / atmosphere / narrative moment / composition principle vary per-scene. This is exactly how Hegre's *Tuscany Nudes* works — one estate, one visual language, but each photograph has its own moment.

2. **Intelligent placement via category-side compatibility filters.** The LLM doesn't pick from the full 200-tag vocabulary on every call — each theme/category/style_profile declares `compatible_*` lists that filter the menu to coherent options. Helmut Newton + Wes Anderson pastel is never offered as a combination because the noir-themed categories whitelist Newton and exclude pastel palettes.

3. **Tier-required fields force usage.** Adding vocab without making the LLM use it = no change (we already learned this with optional realism fields, where Qwen3 left 0/24 NULL). Critical new fields (`narrative_moment`, `environment_setting`, `color_palette`, `photographer_ref`) become tier-required at T3+ so the validator's retry-nudge fires when the LLM skips them.

---

## 3. The new vocabulary (the data layer)

All under `config/prompt_vocabulary.yaml`. Version bumps **5 → 6**.

### 3.1 Scene-level namespaces (LLM picks per-scene via SceneFacet)

| Namespace | Tag count | Tier-required? | Picks per scene | Purpose |
|---|---:|:-:|:-:|---|
| `environment.setting` | 40 | ✅ T3+ | 1 | The location archetype (`ENV_VICTORIAN_CONSERVATORY`, `ENV_TUSCAN_VILLA_RENAISSANCE`, `ENV_BRUTALIST_LOFT`, `ENV_DESERT_DUNE_GOLDEN_HOUR`) |
| `environment.prop` | 30 | optional | 0-3 | Set-dressing (`PROP_CHEVAL_MIRROR`, `PROP_VELVET_CURTAIN_HEAVY`, `PROP_HANDWRITTEN_LETTER`, `PROP_PEONIES_OVERBLOWN`) |
| `environment.atmosphere` | 25 | ✅ T3+ | 1-2 | Atmospheric elements (`ATM_DUST_MOTES_IN_LIGHT`, `ATM_RAIN_ON_GLASS`, `ATM_STEAM_FROM_BATH`, `ATM_CREPUSCULAR_RAYS`) |
| `narrative.moment` | 30 | ✅ T1+ | 1 | What's happening (`NARR_READING_LETTER_AT_DAWN`, `NARR_STEPPING_FROM_BATH`, `NARR_LACING_CORSET_BACK`) |
| `composition.principle` | 20 | optional | 0-1 | Advanced composition (`COMP_FRAME_THROUGH_DOORWAY`, `COMP_NEGATIVE_SPACE_DOMINANT`, `COMP_REFLECTION_PRIMARY`) |
| **Subtotal scene-level** | **145** | | | |

### 3.2 Series-level inherited namespaces (SeriesPlanner picks once)

| Namespace | Tag count | Required? | Picks per series | Purpose |
|---|---:|:-:|:-:|---|
| `aesthetic.color_palette` | 17 | ✅ | 1 | Cinematic grade (`PALETTE_TEAL_ORANGE_BLOCKBUSTER`, `PALETTE_BAROQUE_CARAVAGGIO`, `PALETTE_WES_ANDERSON_PASTEL`) |
| `aesthetic.photographer_ref` | 15 | ✅ | 1 | Photographer signature (`PHOTOG_HELMUT_NEWTON`, `PHOTOG_PETER_LINDBERGH`, `PHOTOG_PETTER_HEGRE`, `PHOTOG_SLIM_AARONS`) |
| `aesthetic.art_movement` | 15 | optional | 0-1 | Art movement (`ART_PRE_RAPHAELITE`, `ART_DUTCH_GOLDEN_VERMEER`, `ART_FILM_NOIR_1940S`, `ART_WES_ANDERSON_SYMMETRY`) |
| **Subtotal series-level** | **47** | | | |

**Total new tags: ~192** (with per-family phrasings, ~960 total family-phrased entries across SDXL/Pony/Illustrious/Flux/Chroma/Flux2). Pony omits art_movement + photographer_ref + composition.principle namespaces (booru tagging carries those implicitly via its convention).

### 3.3 Concrete tag examples (full lists in §11 appendix)

**Sample environment.setting (with chroma phrasing):**
```yaml
ENV_VICTORIAN_CONSERVATORY:
  sdxl:        "abandoned Victorian conservatory, broken glass panels, ivy creeping through iron frame, dappled green light"
  pony:        "abandoned_building, conservatory, broken_glass, ivy, dappled_light, victorian_architecture"
  illustrious: "victorian_conservatory, broken_glass, ivy_overgrown, dappled_light, ruined_elegance"
  flux:        "An abandoned Victorian conservatory with broken glass panels, ivy creeping through the wrought-iron framework, dappled green light filtering through the remaining panes"
  chroma:      "abandoned Victorian conservatory, broken glass panels, ivy through iron frame, dappled green light through remaining panes"
  flux2:       "Set inside an abandoned Victorian conservatory, broken glass panels above, ivy creeping through the wrought-iron frame at chest height, dappled green light filtering through the surviving panes, humid still air"

ENV_TUSCAN_VILLA_RENAISSANCE:
  sdxl:        "16th-century Tuscan villa, frescoed walls, terracotta floors, ancient marble statuary"
  pony:        "tuscan_villa, frescoed_walls, terracotta_floor, marble_statue, renaissance_interior"
  illustrious: "tuscan_villa, frescoed_walls, terracotta_floors, marble_statuary, renaissance_setting"
  flux:        "A 16th-century Tuscan villa interior with frescoed walls, terracotta floors, ancient marble statuary in alcoves"
  chroma:      "16th-century Tuscan villa, frescoed walls, terracotta tile floor, ancient marble statuary"
  flux2:       "Set inside a 16th-century Tuscan villa, frescoed walls catching afternoon light, terracotta floor tiles, ancient marble statuary placed in alcoves, cool stone interior atmosphere"
```

**Sample narrative.moment (with chroma phrasing):**
```yaml
NARR_READING_LETTER_AT_DAWN:
  sdxl:        "reading a handwritten letter by morning light, fountain pen on the side table"
  pony:        "reading_letter, morning_light, fountain_pen, sitting, contemplative"
  illustrious: "reading_letter, morning_light, fountain_pen, contemplative_moment"
  flux:        "She is reading a handwritten letter by the morning light, a fountain pen lying on the side table beside her"
  chroma:      "reading a handwritten letter by morning light, fountain pen resting on the side table, contemplative pause"
  flux2:       "The subject is reading a handwritten letter in the soft morning light, a fountain pen lying on the polished side table beside her, a contemplative pause caught mid-thought"

NARR_STEPPING_FROM_BATH:
  sdxl:        "stepping from a clawfoot bath, towel half-wrapped, steam rising from the water"
  pony:        "stepping_out_of_bath, clawfoot_bathtub, towel, steam, wet_skin, candid_moment"
  illustrious: "stepping_out_of_bath, clawfoot_bathtub, half_wrapped_towel, steam, wet_skin"
  flux:        "She is stepping from a clawfoot bath, a linen towel half-wrapped around her, steam still rising from the warm water behind her"
  chroma:      "stepping from a clawfoot bath, linen towel half-wrapped, steam rising from warm water behind her, wet skin catching the light"
  flux2:       "The subject is stepping from a clawfoot porcelain bath, a thick linen towel half-wrapped around her body, steam still rising from the warm water behind her, water droplets catching the soft window light on her skin"
```

**Sample color_palette (with chroma phrasing):**
```yaml
PALETTE_BAROQUE_CARAVAGGIO:
  sdxl:        "Caravaggio palette, deep umber blacks, single warm flesh-tone highlight, dramatic shadow"
  pony:        "caravaggio_palette, dark_background, dramatic_chiaroscuro, warm_skin_highlight"
  illustrious: "caravaggio_palette, deep_blacks, warm_flesh_highlight, dramatic_shadow"
  flux:        "Caravaggio colour palette — deep umber blacks dominating the frame, a single warm flesh-tone highlight on the subject, dramatic chiaroscuro shadow falloff"
  chroma:      "Caravaggio palette, deep umber blacks, warm flesh-tone highlight, dramatic chiaroscuro shadow"
  flux2:       "Caravaggio colour palette — deep umber blacks dominating the frame, a single warm flesh-tone highlight on the subject's skin, dramatic chiaroscuro shadow falloff into pure black"

PALETTE_WES_ANDERSON_PASTEL:
  sdxl:        "Wes Anderson pastel palette, muted yellows, soft pinks, mint green, symmetrical color blocks"
  pony:        "wes_anderson_palette, pastel_yellow, soft_pink, mint_green, symmetrical_color_blocks"
  illustrious: "wes_anderson_palette, muted_pastels, color_block_symmetry"
  flux:        "Wes Anderson pastel colour palette — muted yellows, soft pinks, and mint greens arranged as symmetrical colour blocks throughout the frame"
  chroma:      "Wes Anderson pastel palette, muted yellows, soft pinks, mint green, symmetrical colour blocks"
  flux2:       "Wes Anderson pastel colour palette — muted lemon yellows, soft blush pinks, mint greens arranged as deliberate symmetrical colour blocks across the composition"
```

(Full 192-tag YAML supplied in §11; not pasted here to keep this plan readable.)

---

## 4. Schema extensions (the code data layer)

### 4.1 Scene & SceneFacet (`src/agents/schemas.py`)

**Scene** gets one new field:
```python
narrative_moment: str | None = None  # required when content_level >= T1 (validator-enforced)
```

**Each SceneFacet*** gets new Optional[str] fields:
```python
environment_setting: str | None = Field(default=None, ...)      # all families
environment_props: str | None = Field(default=None, ...)        # all families — comma-joined tag list
environment_atmosphere: str | None = Field(default=None, ...)   # all families
composition_principle: str | None = Field(default=None, ...)    # SDXL/Flux/Chroma/Flux2/Illustrious — Pony omits
```

**SeriesPlan** (lives as JSON inside `series.llm_series_plan` column) gets:
```python
color_palette: str             # series-level inherited, required
photographer_ref: str          # series-level inherited, required
art_movement: str | None       # series-level inherited, optional
environment_archetype: str     # broad category — e.g. "natural_outdoor", "period_luxury", "mixed_vignettes"
environment_diversity: str     # one of: "single_location", "varied_within_archetype", "mixed_vignettes"
# the existing `environment: str` field stays as a default-environment fallback
```

No DB column changes needed — the planner output already lives in a JSON blob.

### 4.2 Canonicalizer (`src/prompt/vocabulary.py`)

Five new entries in `_FIELD_TO_NAMESPACE`:
```python
_FIELD_TO_NAMESPACE: dict[str, tuple[str, str]] = {
    ...existing 11...
    "environment_setting":    ("environment", "setting"),
    "environment_props":      ("environment", "prop"),
    "environment_atmosphere": ("environment", "atmosphere"),
    "composition_principle":  ("composition", "principle"),
    "narrative_moment":       ("narrative", "moment"),
}
```

Plus three new series-level lookups (canonicalized via a new helper `canonicalize_series_aesthetic(series_plan, family_id, content_level)`):
```python
_SERIES_FIELD_TO_NAMESPACE: dict[str, tuple[str, str]] = {
    "color_palette":     ("aesthetic", "color_palette"),
    "photographer_ref":  ("aesthetic", "photographer_ref"),
    "art_movement":      ("aesthetic", "art_movement"),
}
```

The existing canonicalizer code handles tier-gating and family-omission automatically — no new logic needed.

### 4.3 PromptBuilder (`src/prompt/builder.py`)

`build_one` reads the SeriesPlan from the scene's parent series and adds the series-aesthetic phrases ahead of the scene-facet phrases:

```python
# NEW: thread series-level aesthetic phrases
series_phrases = canonicalize_series_aesthetic(
    series_plan, family.id, content_level=content_level,
)
# existing scene-facet canonicalize
vocab_phrases = canonicalize_facet(scene, family.id, content_level=content_level)
# Compose: scene_body → series_aesthetic → scene_vocab → style → extras
segments = [base_prompt] + [series_phrases] + [scene_fields] + [vocab_phrases] + [style_keywords]
```

Why series-aesthetic ahead of scene-vocab: it sets the visual world before the per-scene details fill it in. Diffusion model attention weights earlier tokens more.

### 4.4 Family-level magic-word strip (`config/families.yaml`)

For `chroma`, `flux`, `flux2`: audit `quality_prefix` + `quality_suffix` and strip:
- `masterpiece`
- `best quality`
- `8k`
- `ultra HD`
- `ultradetailed`
- `intricate background`
- `rich atmospheric details`
- `highly detailed background`
- `cinematic` (when standalone)
- `sharp focus` (when standalone)

Replace with specificity-anchored text where useful (e.g. `"f/1.8. 35mm. photographic. natural skin texture."` for chroma is fine — those are real-world specifics, not magic words).

For `sdxl`, `pony`, `illustrious`: keep their existing quality boilerplate.

---

## 5. Agent / system-prompt changes (the LLM-driving layer)

### 5.1 SeriesPlanner (`src/agents/series_planner.py`)

**System prompt addition:**
```
EVERY series MUST establish a coherent VISUAL WORLD via three series-level
aesthetic anchors held constant across every scene:
  - color_palette: pick one PALETTE_* tag from the menu. This is the
    series's colour grading; every scene inherits it.
  - photographer_ref: pick one PHOTOG_* tag. This is the photographer
    whose signature style the series emulates.
  - art_movement: pick one ART_* tag (optional but encouraged). The
    art movement / aesthetic tradition that informs the series.

COHERENCE RULE: these three tags must form a sensible combination. Some
known-coherent pairings:
  - PHOTOG_HELMUT_NEWTON + PALETTE_MONOCHROME_HIGH_CONTRAST + ART_FILM_NOIR_1940S
  - PHOTOG_PETTER_HEGRE + PALETTE_TUSCAN_EARTH + ART_DUTCH_GOLDEN_VERMEER
  - PHOTOG_SLIM_AARONS + PALETTE_LUBEZKI_NATURAL_GOLDEN + (no art_movement)
  - PHOTOG_PETRA_COLLINS + PALETTE_WES_ANDERSON_PASTEL + (no art_movement)
  - PHOTOG_BILL_HENSON + PALETTE_BAROQUE_CARAVAGGIO + ART_BAROQUE_CARAVAGGIO_CHIAROSCURO
Avoid mixing incompatible worlds (e.g. PHOTOG_HELMUT_NEWTON + PALETTE_WES_ANDERSON_PASTEL).

The active style_profile narrows the menu to compatible options — pick
from within the narrowed menu.

ENVIRONMENT DIVERSITY: pick ONE of three modes for this series:
  - "single_location": all scenes happen in the same specific setting
    (legacy behaviour — appropriate for intimate-bedroom series)
  - "varied_within_archetype": scenes vary across compatible locations
    within one archetype (e.g. Tuscan villa: kitchen, terrace, bathroom,
    garden, library — all coherent location-family)
  - "mixed_vignettes": scenes span across archetypes (e.g. a "fine-art
    nude study" series mixing studio + outdoor-classical + interior-period)
Default: "varied_within_archetype" — gives visual diversity while
holding the series's visual world coherent.
```

**New user-prompt fields:**
```
Series-level aesthetic anchors menu (filtered to style_profile compatibility):
  color_palette: <list from style_profile.compatible_palettes>
  photographer_ref: <list from style_profile.compatible_photographers>
  art_movement: <list from style_profile.compatible_art_movements>

Environment menu (filtered to theme.compatible_environments):
  environment_archetype: <list>
  available_settings_in_archetype: <list of ENV_* tags>
```

**JSON schema body grows:**
```json
{
  "theme": "...",
  "mood": "...",
  "environment": "...",  // legacy fallback for single_location mode
  "environment_archetype": "<one of: natural_outdoor, urban_loft, period_luxury, decay_patina, studio_minimal, mixed_vignettes>",
  "environment_diversity": "<one of: single_location, varied_within_archetype, mixed_vignettes>",
  "color_palette": "<PALETTE_*>",
  "photographer_ref": "<PHOTOG_*>",
  "art_movement": "<ART_* or null>",
  "variation_axes": [...]
}
```

### 5.2 SceneGenerator (`src/agents/scene_generator.py`)

**System prompt rewrite (creative-variety push):**
```
DIVERSITY REQUIREMENT (NEW):
- When series.environment_diversity == "varied_within_archetype" or
  "mixed_vignettes", each scene MUST set environment_setting to a
  DIFFERENT ENV_* tag from the others. Repeating the same location
  twice in 24 scenes is a FAIL.
- Vary atmosphere across scenes: dust-motes scene / rain-on-glass
  scene / steam-from-bath scene / golden-crepuscular scene / fog
  scene — don't cluster atmosphere on one type.
- Vary narrative_moment across scenes — never repeat the same
  moment twice in a set.
- Vary time-of-day across scenes (morning gold, midday flat, blue
  hour, midnight tungsten) when the location allows it.

CREATIVE-WORLD COHERENCE (NEW):
The series carries fixed aesthetic anchors (color_palette,
photographer_ref, art_movement) — see series plan. Each scene's
choices (environment_setting, props, atmosphere) MUST be compatible
with those anchors. A Helmut-Newton + film-noir series should pick
ENV_PARIS_ATTIC_APARTMENT / ENV_HOTEL_LOBBY_NOIR / ENV_INDOOR_POOL_NIGHT
— not ENV_WILDFLOWER_MEADOW. The composer will canonicalize these tags
into family-shaped phrasing; the LLM's job is choosing the right tags.

INTELLIGENT PLACEMENT (NEW):
Props and atmosphere are picked for each scene to MATCH that scene's
pose + lighting + narrative_moment + environment_setting. A
NARR_READING_LETTER_AT_DAWN scene logically gets PROP_HANDWRITTEN_LETTER
+ PROP_FOUNTAIN_PEN + ATM_DUST_MOTES_IN_LIGHT. A NARR_STEPPING_FROM_BATH
scene gets PROP_CHEVAL_MIRROR + ATM_STEAM_FROM_BATH. Don't pick
random props that don't fit the moment.
```

**New scene fields (in the schema body):**
```json
{
  ...existing 10 fields...
  "narrative_moment": "<NARR_* tag — required at every tier>",
  "environment_setting": "<ENV_* tag — required at T3+ when environment_diversity != single_location>"
}
```

**Validator nudge-retry**: when LLM omits narrative_moment / environment_setting in a tier-required context, the existing nudge-retry mechanism in SceneFacetGenerator (already proven to work at 100% with Cydonia) fires for these fields too.

### 5.3 SceneFacetGenerator (`src/agents/scene_facet_generator.py`)

**Tier-required field expansion**:
```python
_TIER_REQUIRED_FIELDS: dict[str, frozenset[str]] = {
    "T1_suggestive": frozenset({"lighting_directive", "mood_aesthetic", "narrative_moment"}),
    "T2_implied":    frozenset({"lighting_directive", "mood_aesthetic", "narrative_moment"}),
    "T3_artnude":    frozenset({"lighting_directive", "mood_aesthetic", "narrative_moment",
                                 "nsfw_anatomy", "environment_setting", "environment_atmosphere"}),
    "T4_explicit":   frozenset({"lighting_directive", "mood_aesthetic", "narrative_moment",
                                 "nsfw_anatomy", "nsfw_act", "environment_setting", "environment_atmosphere"}),
}
```

`_FIELD_EXAMPLE_TAGS` grows entries for each new tier-required field (inlined into retry-nudge prompt for the second attempt).

---

## 6. Categories taxonomy expansion (`config/categories.yaml`)

Each theme grows four new `compatible_*` filter lists. Example:

```yaml
themes:
  - id: boudoir_lingerie
    name: Boudoir Lingerie
    weight: 1.0
    description: Intimate boudoir — bedroom / dressing room / hotel suite / period parlour
    variations: [silk robe, lace bodysuit, vintage stockings, sheer kimono, satin chemise]
    aesthetic_affinity: [boudoir_noir, old_hollywood_glamour]
    suited_tiers: [T1_suggestive, T2_implied, T3_artnude, T4_explicit]
    # NEW: compatibility filters that narrow the LLM's tag menu
    compatible_environments:
      - ENV_MORNING_BEDROOM
      - ENV_OLD_HOLLYWOOD_BOUDOIR
      - ENV_ART_DECO_HOTEL_SUITE
      - ENV_VICTORIAN_PARLOUR
      - ENV_GEORGIAN_LIBRARY
      - ENV_VANITY_DRESSING_ROOM
      - ENV_READING_NOOK
      - ENV_CLAWFOOT_BATHROOM
    compatible_palettes:
      - PALETTE_DEAKINS_AMBER_TUNGSTEN
      - PALETTE_GREAT_GATSBY_JEWEL
      - PALETTE_MONOCHROME_LOW_KEY
      - PALETTE_SEPIA_PLATINUM
      - PALETTE_BAROQUE_CARAVAGGIO
    compatible_photographers:
      - PHOTOG_HELMUT_NEWTON
      - PHOTOG_RICHARD_AVEDON
      - PHOTOG_PAOLO_ROVERSI
      - PHOTOG_MARIO_TESTINO
      - PHOTOG_PETER_LINDBERGH
    compatible_art_movements:
      - ART_FILM_NOIR_1940S
      - ART_ART_DECO_1920S
      - ART_HOPPER_AMERICAN_REALIST
      - ART_DUTCH_GOLDEN_VERMEER
```

Similar compatibility filters for: `golden_hour`, `editorial_fashion_nude`, `fine_art_studio_nude`, `wet_set_cinematic`, `vintage_pinup`, `dark_boudoir_neonoir`, `fantasy_castlecore`, `old_hollywood`, `implied_nude_fineart`.

**3 NEW themes** added (premium-tier categories surfaced by research):
- `decay_patina_fineart` — abandoned conservatories, derelict theatres, overgrown greenhouses (high atmosphere yield per research §2)
- `mediterranean_villa` — Tuscan / Spanish / Greek-island light + architecture (Hegre-template)
- `metropolitan_neon` — Tokyo / Manhattan / Berlin urban noir (Henson / neo-noir lineage)

Each new theme ships with full compatibility lists.

**style_profiles.yaml** parallel changes: each profile grows `compatible_palettes` + `compatible_photographers` + `compatible_art_movements` lists (narrows what the SeriesPlanner picks). Existing `palette_hint` / `lighting_hint` freetext stays — those are post-LLM composer fallbacks for series that don't pick structured tags.

---

## 7. Composer changes summary (`src/prompt/builder.py`)

Net code changes (no architectural shift, just additive):

1. **New helper** `canonicalize_series_aesthetic(series_plan, family_id, content_level)` — returns list[str] of family-phrased aesthetic anchors. Mirrors existing `canonicalize_facet`.

2. **`build_one`** loads the series plan (passed in by caller alongside scene + character + style_profile), calls the new helper, prepends the series-aesthetic phrases at the front of the prompt body (right after base_prompt, before scene fields).

3. **Engine threading** (`src/core/engine.py::run_phase_a`): `_compose_prompts_for_scene` already has access to the series — just pass `series_plan` to `build_one`. Trivial.

4. **No order change** to scene-facet vocab phrases — they continue to land between scene fields and style keywords.

---

## 8. Test plan

### 8.1 New unit tests

- `tests/test_vocabulary_v6.py` — round-trip every new tag through canonicalize_facet / canonicalize_series_aesthetic, per family.
- `tests/test_categories_compatibility.py` — every `compatible_*` entry in categories.yaml must resolve to a real vocab tag (no dangling refs).
- `tests/test_series_planner_aesthetic_anchors.py` — generated plans always have `color_palette`, `photographer_ref`, `environment_archetype`, `environment_diversity` populated.
- `tests/test_scene_generator_narrative_moment.py` — generated scenes always have `narrative_moment` populated at every tier.
- `tests/test_scene_generator_environment_variety.py` — `varied_within_archetype` series produce 24 scenes with N>=12 distinct `environment_setting` picks.
- `tests/test_builder_series_aesthetic_threading.py` — composed prompt for a Helmut-Newton series contains "Helmut Newton" phrasing; same scene re-composed for a Slim-Aarons series contains "Slim Aarons" phrasing.

### 8.2 Regression tests

- All 1239+ existing tests stay green. Schema changes are additive (Optional fields default null); existing scene_facets rows continue to validate.
- The validator's tier-required field check now nudges retry on new fields; existing tier-required tests pass after `_TIER_REQUIRED_FIELDS` update.

### 8.3 Manual smoke

Generate a fresh series:
```bash
python scripts/prepare_prompts.py --mode theme --level T4_explicit --models gonzalomo_chroma_v30
```

Verify:
- `series.llm_series_plan` JSON contains `color_palette`, `photographer_ref`, `environment_archetype`, `environment_diversity`
- `scenes.narrative_moment` populated on all 24 (NEW field)
- `scene_facets.environment_setting` populated on all 24, with >= 8 distinct picks (when varied_within_archetype)
- `scene_facets.environment_atmosphere` populated on all 24, with >= 4 distinct picks
- `scene_facets.environment_props` populated on >=18 of 24 (optional but encouraged)
- Composed `prompts.prompt_text` contains the series's photographer_ref phrase, the series's color_palette phrase, the scene's environment_setting phrase, the scene's narrative_moment phrase

Then render 4-6 of the 24 chroma prompts and eyeball them for:
- Background diversity (each renders a different setting)
- Color contrast (the chosen palette is visibly driving the grade)
- Narrative legibility (the chosen moment is visible — books / letters / bath steam / mirror reflection actually appear)
- Coherence (the series's photographer-signature visible across all renders)

### 8.4 Acceptance criterion

User reviews 6 fresh renders and confirms they no longer look like "studio test shots". Visual diversity + color contrast + narrative interest must be present without any further prompt engineering on the user's side. If 6/6 are still "boring", the plan is rejected and we iterate. If 5/6 work, ship.

---

## 9. Phasing (the implementation order)

The full plan is ~6-8 hours of focused work. Phased so we can validate each layer before adding the next:

| Phase | Scope | Validates | Time |
|---|---|---|---|
| **Phase 1 — MVP** | `narrative.moment` namespace (30 tags) + Scene schema field + SceneGenerator system prompt addition. Tier-required at T1+. | Whether narrative_moment alone moves the needle as research claims (top leverage). 1 series + manual smoke. | 1.5h |
| **Phase 2 — Environment** | `environment.setting/prop/atmosphere` namespaces (95 tags) + SceneFacet schema fields + tier-required at T3+. SceneGenerator diversity directives. | Whether per-scene environment variety produces visually distinct images within a series. | 2h |
| **Phase 3 — Series aesthetic** | `aesthetic.color_palette/photographer_ref/art_movement` namespaces (47 tags) + SeriesPlanner JSON fields + composer threading. | Whether series-level coherence + series-level color palette gives the "signature look". | 1.5h |
| **Phase 4 — Categories & filters** | `compatible_*` filters in `categories.yaml` + `style_profiles.yaml` + 3 new themes. SceneFacetGenerator nudge updates. | Whether intelligent placement (vs random tag selection) holds aesthetic coherence. | 1.5h |
| **Phase 5 — Magic-word strip** | Audit `families.yaml` chroma/flux/flux2 quality_prefix/suffix; strip the degrading magic words. | Whether removing magic words improves chroma/flux output (research claims yes). | 0.5h |
| **Phase 6 — Tests + commit** | New tests (~6 files), regression run, ARCHITECTURE.md + PROJECT_GUIDE.md sync, single PR. | All green. | 1h |

**Total: ~8h**. Each phase ends with a commit and a manual smoke. You can stop after any phase if the output meets your bar.

---

## 10. Risk + mitigation

| Risk | Severity | Mitigation |
|---|:-:|---|
| LLM (cydonia) struggles to navigate the bigger ~20-namespace menu — fill rate drops below 100% | medium | Tier-required fields force retry-nudge (proven on prior structured-field validator). Optional fields stay optional. Expect 80-95% fill at 200-tag scale. If <80%, prune low-leverage tags. |
| LLM picks aesthetically incoherent combinations (Newton + Wes Anderson pastel) | low | `compatible_*` filters in categories/style_profiles narrow the menu BEFORE the LLM sees it. System prompt enforces coherence as a rule. Random incompatible picks ~impossible after the filter. |
| Phase A time grows from 18min → 25-30min due to more LLM thinking on bigger menu | medium | Acceptable — final-quality LLM choice was already cydonia (slow but thorough). User can use --llm qwen3_abliterated_30b for fast brainstorm and switch to cydonia for production. |
| Tag explosion: 192 new tags × 6 families = 1152 YAML entries to write | low | Yes, but mechanical work. The research agent provided 60+ tag candidates with phrasings; I'll fill the remaining 130 by analogy. ChatGPT-style mechanical expansion, low intellectual load. |
| Existing scene_facets rows have null on new fields → some old prompts will render with the new pipeline missing the series aesthetic | low | All new fields are Optional with null defaults. Existing rows render as before. Only NEW series benefits from the new vocab. Users who want old series re-rendered with the new aesthetic call `--regen-facets chroma` to re-prompt them. |
| Magic-word strip on chroma/flux/flux2 backfires (we're wrong about the research) | low-med | Phase 5 ships isolated commit. If output gets worse, revert with one commit. A/B render before/after on a 4-scene smoke. |
| 3 new themes (`decay_patina_fineart`, `mediterranean_villa`, `metropolitan_neon`) don't sell — wasted category work | low | Themes are weighted; weights start at 0.5 (below median). If they don't get picked, no harm. |
| User wants vary-environment but also "intelligent placement" and the LLM picks weird props (book in a desert dune scene) | low | Coherence rules in scene-generator system prompt explicitly call this out. Verifier-agent will sanity-check. Worst case: the prop is dropped at canonicalize-time if the family-phrasing makes it incoherent. |

---

## 11. Appendix — full tag lists (deferred)

Full 192-tag YAML written separately and attached to the actual implementation PR. Sample shown in §3.3 above. Tag-list size:
- environment.setting: 40
- environment.prop: 30
- environment.atmosphere: 25
- aesthetic.color_palette: 17
- aesthetic.photographer_ref: 15
- aesthetic.art_movement: 15
- narrative.moment: 30
- composition.principle: 20

Each tag: per-family phrasing (6 families = sdxl/pony/illustrious/flux/chroma/flux2). Pony omits art_movement / photographer_ref / composition.principle entirely (booru tagging carries those implicitly). Total YAML entries: ~960.

---

## 12. Migration: NONE

User said "do not worry about migration". The plan is entirely additive:
- New Optional[str] fields → existing rows default null → existing prompts compose as before
- New vocab namespaces → existing scene_facets without them simply don't trigger canonicalization
- New series-plan JSON fields → existing series without them fall back to legacy single-environment mode
- New categories.yaml `compatible_*` filters → missing filters default to "any" (no narrowing) → existing themes unaffected
- vocab_version bumps 5 → 6 → existing rows keep their vocab_version=5 stamp for audit
- DB schema: no changes

Zero migration burden. Existing series re-renderable as-is. New series benefit from the new vocab.
