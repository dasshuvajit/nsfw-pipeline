#!/usr/bin/env python3
"""prepare_prompts — Phase A only (LLM planning + per-model prompts).

Persists series + scenes + per-(scene, family) facets + per-model
prompts to the DB. **No ComfyUI calls.** Pairs with
``render_prompts.py`` for the second half of the cycle.

Use this when you want to:
  - Generate a series and prepare prompts for one or more models
    without rendering yet (e.g. so a human can review the plan
    in supervised mode before committing GPU time).
  - Re-target an existing series for a new model (``--series-id``
    skips planning + scene generation; reuses scenes; generates
    that family's facets if missing; composes per-model prompts).
  - Re-roll family-shaped facets (``--regen-facets``) or per-model
    prompts (``--regen-prompts``) without affecting other models.

Pre-flight:
  - ``ollama serve`` is running.
  - ``python scripts/init_db.py`` has been run.
  - Each ``--models`` id resolves under ``config/models/*.yaml``.

Usage:
    # Fresh series for one model (default = pipeline.default_model_id)
    python scripts/prepare_prompts.py --mode theme --level T2_implied

    # Multi-model fan-out (sibling-family models share facet rows)
    python scripts/prepare_prompts.py --mode theme --level T3_artnude \\
        --models juggernaut_ragnarok,chroma_v10HD

    # Re-target an existing series for a new model
    python scripts/prepare_prompts.py --series-id ser_abc \\
        --models gonzalomo_flux_v30

    # Re-roll the SDXL facets + juggernaut_ragnarok prompts on an existing series
    python scripts/prepare_prompts.py --series-id ser_abc \\
        --models juggernaut_ragnarok --regen-facets sdxl --regen-prompts juggernaut_ragnarok

Exit codes:
    0 = success
    1 = engine error (preflight, LLM failure, missing model id)
    2 = duplicate prompts (already exist for this (series, model);
        use --regen-prompts to re-roll)
    3 = supervisor abort (human rejected one model's plan)
    4 = keyboard interrupt
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.content_level import (  # noqa: E402
    ContentLevelLoader,
    UnknownContentLevel,
)
from src.core.engine import (  # noqa: E402
    EngineError,
    PipelineEngine,
    PreflightError,
    _load_config,
    _load_style_profile,
)
from src.core.generation_context import build_context  # noqa: E402


def _parse_csv(value: str | None) -> list[str]:
    """Parse a comma-separated CLI value into a deduped, ordered list.

    ``--models juggernaut_ragnarok,chroma_v10HD`` → ``["juggernaut_ragnarok", "chroma_v10HD"]``.
    Empty / None → ``[]``.
    """
    if not value:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for tok in value.split(","):
        tok = tok.strip()
        if tok and tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def _resolve_models(args_models: str | None, config: dict[str, Any]) -> list[str]:
    """``--models a,b`` → list; empty + no default → ``[]``.

    Post family-level prompt-prep (2026-05): when both ``--models``
    and ``--families`` are absent, this returns ``[]`` rather than
    raising. The post-parse validator in ``main()`` enforces "at
    least one of models/families non-empty" — that path emits a
    clearer error pointing at both flags.
    """
    parsed = _parse_csv(args_models)
    if parsed:
        return parsed
    default_id = config.get("pipeline", {}).get("default_model_id")
    if default_id:
        return [default_id]
    return []


def _resolve_families(args_families: str | None) -> list[str]:
    """``--families flux,pony`` → list; empty → ``[]``.

    There is no ``pipeline.default_family_id`` (out of scope for the
    initial family-level feature). The user explicitly opts in by
    passing ``--families``; absence means "model-mode only".
    """
    return _parse_csv(args_families)


def _build_baseline_ctx(
    *,
    engine: PipelineEngine,
    args: argparse.Namespace,
    models: list[str],
    families: list[str],
    style_profile: dict[str, Any],
    style_profile_id: str,
) -> Any:
    """Build the baseline GenerationContext that ``run_phase_a`` uses
    for series-level concerns (planning + scene generation).

    The per-target loop inside ``run_phase_a`` rebuilds a fresh ctx
    per (target_kind, target_id) — the baseline only matters for the
    new-series path (it's ignored on re-target).

    **The baseline ctx is always model-kind** even in family-only
    invocations (verifier patch B1 — `run_phase_a:621,647-654` reads
    `ctx.model_config.filename` / `ctx.model_config.family` BEFORE
    the per-target loop fires). When `--families` is supplied without
    `--models`, we pick the first model registered to the first
    family as a baseline planning checkpoint. The per-family loop
    inside `run_phase_a` then rebuilds family-kind ctx for prompt
    generation.
    """
    # Mode resolution: --mode wins; else weighted-random via ModeSelector
    # (the engine's selector, configured from pipeline.yaml::mode_weights).
    mode_name = engine.mode_selector.select(force_mode=args.mode)

    # Content-level resolution (2026-05-17 bug fix). When retargeting an
    # existing series, the tier is whatever the series row in DB says —
    # NOT the CLI default. This closes the silent-downgrade bug where
    # `prepare_prompts --series-id <T4-series> --families chroma` (no
    # --level) was producing T2_implied prompts despite the series being
    # T4_explicit.
    #
    # The DB read is defensive: a fresh DB (or a test fixture) may not
    # carry the `series` table yet. In that case we leave effective_level
    # at whatever args.level resolved to and let run_phase_a's own
    # "series not found" check fire.
    effective_level = args.level
    if args.series_id:
        import sqlite3 as _sqlite3
        db_level: str | None = None
        try:
            _conn = _sqlite3.connect(str(engine.db_path))
            try:
                row = _conn.execute(
                    "SELECT content_level FROM series WHERE id = ?",
                    (args.series_id,),
                ).fetchone()
                if row is not None:
                    db_level = row[0]
            finally:
                _conn.close()
        except _sqlite3.DatabaseError:
            # Schema not initialised / DB unreadable — fall through and
            # let downstream code surface the real error if it matters.
            db_level = None

        if db_level is not None:
            if effective_level is None:
                effective_level = db_level
                logging.getLogger(__name__).info(
                    "Retarget: inheriting content_level %r from series %s",
                    effective_level, args.series_id,
                )
            elif effective_level != db_level:
                raise EngineError(
                    f"--level {effective_level!r} does not match the "
                    f"existing series' content_level ({db_level!r}). "
                    f"You cannot change the tier of an existing series "
                    f"— omit --level to inherit from the series, or "
                    f"create a new series at the desired tier."
                )

    if effective_level is None:
        # New series (or unreadable DB) — apply the documented default.
        effective_level = "T2_implied"

    content_rules = ContentLevelLoader(engine.db_path).load(effective_level)

    # Baseline ctx uses the FIRST model in --models (but the per-target
    # loop in run_phase_a rebuilds per iteration, so this is mostly
    # cosmetic for new-series + plan/scene-gen). When --models is
    # empty (family-only invocation), fall back to the first model
    # registered to the first --families entry — keeps the baseline
    # ctx model-kind so the planning hop has a concrete model_config.
    if models:
        baseline_model_id = models[0]
    elif families:
        from src.memory.model_registry import ModelRegistryLoader
        loader = ModelRegistryLoader(
            engine.db_path, commercial_mode=engine._commercial_mode,
        )
        first_family_models = loader.list_models(family=families[0])
        if not first_family_models:
            raise EngineError(
                f"--families {families[0]!r} has no registered models in "
                f"config/models/*.yaml — can't pick a baseline checkpoint "
                f"for the planning hop. Add a model with "
                f"family: {families[0]!r} or pass --models explicitly."
            )
        baseline_model_id = first_family_models[0].id
    else:
        # Should be unreachable — main() validates at least one of
        # models / families is non-empty before calling here.
        raise EngineError(
            "_build_baseline_ctx requires non-empty models or families"
        )
    ctx = build_context(
        mode=mode_name,
        content_level=effective_level,
        execution_mode=engine.execution_mode,
        style_profile=style_profile,
        content_rules=content_rules,
        db_path=engine.db_path,
        model_id=baseline_model_id,
        commercial_mode=engine._commercial_mode,
    )
    return ctx


def _print_summary(
    result: dict[str, Any],
    *,
    elapsed: float,
    re_target: bool,
) -> None:
    """Stdout-friendly summary of what Phase A produced."""
    series_id = result.get("series_id", "?")
    status = result.get("status", "?")
    models_completed = result.get("models_completed", []) or []
    families_completed = result.get("families_completed", []) or []
    print()
    print(f"Phase A complete in {elapsed:.1f}s — status: {status}")
    print(f"  Series:           {series_id}{' (re-targeted)' if re_target else ' (new)'}")
    print(f"  Scenes created:   {result.get('scenes_created', 0)}"
          + (" (reused from existing series)" if re_target else ""))
    print(f"  Facets created:   {result.get('facets_created', 0)}")
    print(f"  Prompts inserted: {result.get('prompts_created', 0)}")
    if models_completed:
        print(f"  Models completed: {', '.join(models_completed)}")
    if families_completed:
        print(f"  Families completed: {', '.join(families_completed)}")
    if status == "complete":
        print()
        print("Next:")
        if models_completed:
            print(
                f"  python scripts/render_prompts.py --series-id {series_id} "
                f"--models {','.join(models_completed)}"
            )
        if families_completed:
            # Family-kind renders need an explicit checkpoint via
            # --render-with-model. Suggest "<one of your models>" as
            # a placeholder; the user picks per family.
            for fam in families_completed:
                print(
                    f"  python scripts/render_prompts.py --series-id "
                    f"{series_id} --families {fam} "
                    f"--render-with-model <a model in '{fam}' family>"
                )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Phase A only — generate series + scenes + per-(scene,family) "
            "facets + per-model prompts; persist to DB; no ComfyUI."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["theme", "style", "niche", "variation"],
        default=None,
        help="Pipeline mode (default: weighted random selection)",
    )
    parser.add_argument(
        "--level",
        choices=["T1_suggestive", "T2_implied", "T3_artnude", "T4_explicit"],
        default=None,
        help=(
            "Content tier. New-series default: T2_implied. Retargeting "
            "an existing series (--series-id): the tier is INHERITED "
            "from the series row in DB and --level must match (or be "
            "omitted). Mismatch is rejected — you cannot retarget a "
            "T4 series at T2."
        ),
    )
    parser.add_argument(
        "--style-profile",
        default=None,
        help="Style profile id (default: pipeline.default_style_profile_id)",
    )
    parser.add_argument(
        "--models",
        default=None,
        help=(
            "Comma-separated model ids to fan out across "
            "(e.g. 'juggernaut_ragnarok,chroma_v10HD'). Per-model rules apply "
            "(trigger words, avoid words, negative_embeddings). "
            "Sibling-family models share scene_facets "
            "rows. Default when neither --models nor --families is "
            "given: [pipeline.default_model_id]."
        ),
    )
    parser.add_argument(
        "--families",
        default=None,
        help=(
            "Comma-separated family ids to fan out across "
            "(e.g. 'flux,pony,sdxl,illustrious,chroma,flux2'). "
            "Family-level prompt prep — only family-level rules "
            "apply, NO per-model trigger words / avoid words / "
            "negative_embeddings. Render the resulting "
            "prompts with `render_prompts --families <F> "
            "--render-with-model <M>`. May be combined with --models "
            "to prepare both kinds in one invocation."
        ),
    )
    parser.add_argument(
        "--series-id",
        default=None,
        help=(
            "Re-target an existing series — skip planning + scene "
            "generation, only generate this run's facets + prompts on "
            "the existing scenes."
        ),
    )
    parser.add_argument(
        "--regen-facets",
        default=None,
        help=(
            "Comma-separated family ids whose scene_facets rows should "
            "be DELETEd before regenerating "
            "(e.g. 'sdxl,pony'). No-op for families not in the run."
        ),
    )
    parser.add_argument(
        "--regen-prompts",
        default=None,
        help=(
            "Comma-separated MODEL ids whose model-kind prompts on "
            "this series should be DELETEd before re-composing. "
            "Required when re-running with the same model on the same "
            "series. Does NOT affect family-kind prompts (see "
            "--regen-family-prompts)."
        ),
    )
    parser.add_argument(
        "--regen-family-prompts",
        default=None,
        help=(
            "Comma-separated FAMILY ids whose family-kind prompts on "
            "this series should be DELETEd before re-composing. "
            "Required when re-running with the same family on the "
            "same series. Does NOT affect model-kind prompts (see "
            "--regen-prompts) — kept distinct so a typo can't "
            "accidentally cross-delete."
        ),
    )
    parser.add_argument(
        "--llm",
        default=None,
        help=(
            "Override LLM from config/llm_models.yaml. When set, every "
            "agent role uses this LLM (overrides routing). Use to "
            "re-prompt the same series with a different LLM for A/B "
            "comparison: --series-id S --llm qwen3_abliterated_30b. "
            "Default: "
            "registry's default_llm."
        ),
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    try:
        config = _load_config()
        # Resolve targets — both flags optional, but at least one
        # must produce a non-empty list (post-parse validation
        # below). _resolve_models falls back to
        # pipeline.default_model_id ONLY when --models is empty AND
        # --families is also empty (kept here as the
        # "implicit-default" path; explicit --families opts out).
        families = _resolve_families(args.families)
        if families and not args.models:
            # User explicitly chose family-mode; do NOT pull in the
            # pipeline default model. Only apply the implicit default
            # when both flags are absent.
            models: list[str] = []
        else:
            models = _resolve_models(args.models, config)
        if not models and not families:
            print(
                "ERROR: pass --models <model_ids> or --families "
                "<family_ids> (or both). At least one is required.\n"
                "    --models juggernaut_ragnarok,chroma_v10HD   # model-level\n"
                "    --families flux,pony               # family-level\n"
                "    --models X --families Y            # both\n"
                "If neither is set, pipeline.default_model_id is used "
                "as a fallback when configured.",
                file=sys.stderr,
            )
            return 2
        regen_facets = _parse_csv(args.regen_facets) or None
        regen_prompts = _parse_csv(args.regen_prompts) or None
        regen_family_prompts = (
            _parse_csv(args.regen_family_prompts) or None
        )

        # Validate --llm at the CLI boundary. Engine will also validate
        # via the router but failing here gives a tighter feedback loop
        # before the heavy registry+context construction.
        if args.llm is not None:
            from src.memory.llm_registry import (
                LLMRegistryLoader,
                LLMNotFound,
            )
            try:
                LLMRegistryLoader().get_llm(args.llm, require_active=True)
            except LLMNotFound as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 2

        # Style profile resolution (shared across all models).
        style_profile_id = (
            args.style_profile
            or config.get("pipeline", {}).get("default_style_profile_id")
        )
        if not style_profile_id:
            raise EngineError(
                "Pass --style-profile or set "
                "pipeline.default_style_profile_id in config/pipeline.yaml."
            )

        engine = PipelineEngine(
            config=config,
            dry_run=False,  # we want real DB writes
            force_mode=args.mode,
        )
        style_profile = _load_style_profile(engine.db_path, style_profile_id)

        ctx = _build_baseline_ctx(
            engine=engine,
            args=args,
            models=models,
            families=families,
            style_profile=style_profile,
            style_profile_id=style_profile_id,
        )

        re_target = args.series_id is not None
        logger.info(
            "prepare_prompts: %s, models=%s, families=%s",
            f"re-target series={args.series_id}" if re_target else "new series",
            ",".join(models) or "-",
            ",".join(families) or "-",
        )

        start = time.time()
        try:
            result = engine.run_phase_a(
                ctx,
                models=models or None,
                families=families or None,
                series_id_existing=args.series_id,
                regen_facets=regen_facets,
                regen_prompts=regen_prompts,
                regen_family_prompts=regen_family_prompts,
                style_profile=style_profile,
                style_profile_id=style_profile_id,
                cli_llm_override=args.llm,
            )
        except sqlite3.IntegrityError as exc:
            if "UNIQUE" in str(exc).upper():
                target_series = args.series_id or "<this series>"
                # Build a remediation hint that mentions BOTH kinds
                # since the failed INSERT could be model-kind OR
                # family-kind.
                hints: list[str] = []
                if models:
                    hints.append(f"  --regen-prompts {','.join(models)}")
                if families:
                    hints.append(
                        f"  --regen-family-prompts {','.join(families)}"
                    )
                print(
                    f"\nERROR: prompts for one or more targets already "
                    f"exist on series {target_series}.\n"
                    f"To re-roll, re-run with:\n"
                    + "\n".join(hints),
                    file=sys.stderr,
                )
                return 2
            raise

        elapsed = time.time() - start

        if result.get("status") == "aborted":
            print(
                "\nPhase A aborted (supervisor rejected one model's plan).",
                file=sys.stderr,
            )
            return 3

        _print_summary(result, elapsed=elapsed, re_target=re_target)
        return 0

    except UnknownContentLevel as exc:
        print(f"\nUnknown content level: {exc}", file=sys.stderr)
        return 1
    except PreflightError as exc:
        print(f"\nPreflight failed:\n{exc}", file=sys.stderr)
        return 1
    except EngineError as exc:
        print(f"\nEngine error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 4


if __name__ == "__main__":
    sys.exit(main())
