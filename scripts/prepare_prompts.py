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
    python scripts/prepare_prompts.py --mode character --level T2_implied

    # Multi-model fan-out (sibling-family models share facet rows)
    python scripts/prepare_prompts.py --character char_001 --level T3_artnude \\
        --models lustify_v7,chroma_v10HD

    # Re-target an existing series for a new model
    python scripts/prepare_prompts.py --series-id ser_abc \\
        --models flux_nsfw_71q8

    # Re-roll the SDXL facets + lustify_v7 prompts on an existing series
    python scripts/prepare_prompts.py --series-id ser_abc \\
        --models lustify_v7 --regen-facets sdxl --regen-prompts lustify_v7

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
from src.memory.character_manager import (  # noqa: E402
    CharacterManager,
    CharacterNotFound,
)


def _parse_csv(value: str | None) -> list[str]:
    """Parse a comma-separated CLI value into a deduped, ordered list.

    ``--models lustify_v7,chroma_v10HD`` → ``["lustify_v7", "chroma_v10HD"]``.
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
    """``--models a,b`` → list; empty → ``[pipeline.default_model_id]``."""
    parsed = _parse_csv(args_models)
    if parsed:
        return parsed
    default_id = config.get("pipeline", {}).get("default_model_id")
    if not default_id:
        raise EngineError(
            "No --models given and pipeline.default_model_id is unset in "
            "config/pipeline.yaml. Pass --models explicitly."
        )
    return [default_id]


def _build_baseline_ctx(
    *,
    engine: PipelineEngine,
    args: argparse.Namespace,
    models: list[str],
    style_profile: dict[str, Any],
    style_profile_id: str,
) -> Any:
    """Build the baseline GenerationContext that ``run_phase_a`` uses
    for series-level concerns (planning + scene generation).

    The per-model loop inside ``run_phase_a`` rebuilds a fresh ctx
    per model_id — the baseline only matters for the new-series path
    (it's ignored on re-target).
    """
    # Mode resolution: --mode wins; else weighted-random via ModeSelector
    # (the engine's selector, configured from pipeline.yaml::mode_weights).
    mode_name = engine.mode_selector.select(force_mode=args.mode)
    content_rules = ContentLevelLoader(engine.db_path).load(args.level)

    # Character resolution (character mode + --character flag).
    baseline_character: dict[str, Any] | None = None
    if args.character:
        if mode_name != "character":
            # --character only meaningful in character mode; if mode was
            # randomly chosen as something else, log + ignore.
            logging.getLogger(__name__).info(
                "--character given but mode is %r (not character); "
                "character will be ignored for this run", mode_name,
            )
        else:
            try:
                baseline_character = CharacterManager(
                    engine.db_path
                ).get_character(args.character)
            except CharacterNotFound as exc:
                raise EngineError(
                    f"--character {args.character!r} not found in DB: {exc}"
                ) from exc

    # Baseline ctx uses the FIRST model in --models (but the per-model
    # loop in run_phase_a rebuilds per iteration, so this is mostly
    # cosmetic for new-series + plan/scene-gen).
    baseline_model_id = models[0]
    if baseline_character and baseline_character.get("model_id"):
        # If the character has its own preferred model, use that as
        # baseline (matches run_cycle behavior).
        baseline_model_id = baseline_character["model_id"]

    ctx = build_context(
        mode=mode_name,
        content_level=args.level,
        execution_mode=engine.execution_mode,
        style_profile=style_profile,
        content_rules=content_rules,
        db_path=engine.db_path,
        model_id=baseline_model_id,
        commercial_mode=engine._commercial_mode,
    )
    if baseline_character is not None:
        ctx.character = baseline_character
        ctx.character_id = args.character
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
    print()
    print(f"Phase A complete in {elapsed:.1f}s — status: {status}")
    print(f"  Series:           {series_id}{' (re-targeted)' if re_target else ' (new)'}")
    print(f"  Scenes created:   {result.get('scenes_created', 0)}"
          + (" (reused from existing series)" if re_target else ""))
    print(f"  Facets created:   {result.get('facets_created', 0)}")
    print(f"  Prompts inserted: {result.get('prompts_created', 0)}")
    print(f"  Models completed: {', '.join(result.get('models_completed', []))}")
    if status == "complete":
        print()
        print("Next:")
        models_csv = ",".join(result.get("models_completed", []))
        print(
            f"  python scripts/render_prompts.py --series-id {series_id} "
            f"--models {models_csv}"
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
        choices=["character", "theme", "style", "niche", "variation"],
        default=None,
        help="Pipeline mode (default: weighted random selection)",
    )
    parser.add_argument(
        "--character",
        default=None,
        help=(
            "Character id for character mode (default: LRU pick made by "
            "CharacterMode.plan; here we don't pre-resolve it)"
        ),
    )
    parser.add_argument(
        "--level",
        choices=["T1_suggestive", "T2_implied", "T3_artnude", "T4_explicit"],
        default="T2_implied",
        help="Content tier (default: T2_implied)",
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
            "(e.g. 'lustify_v7,chroma_v10HD'). Default: "
            "[pipeline.default_model_id]. Sibling-family models share "
            "scene_facets rows."
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
            "Comma-separated model ids whose prompts on this series "
            "should be DELETEd before re-composing. Required when "
            "re-running with the same model on the same series."
        ),
    )
    parser.add_argument(
        "--llm",
        default=None,
        help=(
            "Override LLM from config/llm_models.yaml. When set, every "
            "agent role uses this LLM (overrides routing). Use to "
            "re-prompt the same series with a different LLM for A/B "
            "comparison: --series-id S --llm magnum_v4_22b. Default: "
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
        models = _resolve_models(args.models, config)
        regen_facets = _parse_csv(args.regen_facets) or None
        regen_prompts = _parse_csv(args.regen_prompts) or None

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
            force_character=args.character,
        )
        style_profile = _load_style_profile(engine.db_path, style_profile_id)

        ctx = _build_baseline_ctx(
            engine=engine,
            args=args,
            models=models,
            style_profile=style_profile,
            style_profile_id=style_profile_id,
        )

        re_target = args.series_id is not None
        logger.info(
            "prepare_prompts: %s, models=%s",
            f"re-target series={args.series_id}" if re_target else "new series",
            ",".join(models),
        )

        start = time.time()
        try:
            result = engine.run_phase_a(
                ctx,
                models=models,
                series_id_existing=args.series_id,
                regen_facets=regen_facets,
                regen_prompts=regen_prompts,
                style_profile=style_profile,
                style_profile_id=style_profile_id,
                cli_llm_override=args.llm,
            )
        except sqlite3.IntegrityError as exc:
            if "UNIQUE" in str(exc).upper():
                target_series = args.series_id or "<this series>"
                print(
                    f"\nERROR: prompts for one or more of "
                    f"[{', '.join(models)}] already exist on series "
                    f"{target_series}.\n"
                    f"To re-roll, re-run with:\n"
                    f"  --regen-prompts {','.join(models)}",
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
