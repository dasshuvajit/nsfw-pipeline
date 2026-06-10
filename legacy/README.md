# legacy/ — the FROZEN structured vocab/composer path

Archived 2026-06-10 (W4 of the audit master plan). This is the pre-pivot
pipeline: scenes/scene_facets/prompts DB split, vocab canonicalizer, composer,
per-family negative axes, modes, exporter. It was replaced in production by the
**LLM-direct path** (scripts/art_director.py + scripts/art_series.py — see
CLAUDE.md) on 2026-05-30 after audits scored its prompt quality 4.88/10 vs the
LLM-direct path's ~10/10, and it last wrote the DB on 2026-05-29.

**Feature-frozen: do not extend, do not import from active code.**
Kept (not deleted) because it contains working reference implementations
(constrained-decoding schemas, per-family composers, the 9-table DB design)
and its history explains many active-path decisions.

Contents:
- `src/` — core/engine, prompt builder + vocabulary canonicalizer, modes,
  facet/scene generators + schemas (MetadataSchema was lifted OUT to
  src/agents/metadata_generator.py — the one class the active path used),
  family/model/categories/style loaders, supervisor, exporter, filters.
- `scripts/` — prepare_prompts, render_prompts, run_once, init_db, etc.
- `config/` — prompt_vocabulary.yaml (218KB), families.yaml, categories.yaml,
  style_profiles.yaml, ratio_signals.yaml, models/*.yaml.
- `../tests/legacy/` — its test suite (excluded from the default pytest run by
  pytest.ini; to run: `pytest tests/legacy --override-ini addopts=` — imports
  will need `legacy/` on sys.path or a checkout of a pre-archive commit).

The SQLite DB (`nsfw_pipeline.db`, 9 tables) stays at the repo root untouched
as historical data; nothing active reads or writes it.
