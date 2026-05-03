#!/usr/bin/env python3
"""Dry run — full Phase A (LLM planning) with no rendering.

Runs the complete LLM planning pipeline:
  1. Builds GenerationContext (mode, content_level, style_profile, model)
  2. Preflight checks (Ollama reachable, checkpoint exists, workflow exists)
  3. Series planning via LLM (mode-specific)
  4. Scene generation via LLM (or deterministic for variation mode)
  5. Ratio assignment per scene
  6. Prompt building + sanitizing + dedup
  7. Saves everything to DB with status='dry_run'
  8. Unloads the LLM

Does NOT render images (Phase B) or package a set (Phase C).

This is the primary way to validate that the LLM planning produces
sensible output before committing a real render cycle (~20+ minutes).

Pre-flight:
  - ``ollama serve`` is running
  - ``python scripts/init_db.py`` has been run
  - For character mode: ``python scripts/bootstrap_character.py`` has populated the character
  - ComfyUI does NOT need to be running (no rendering happens)

Usage:
    python scripts/dry_run.py --character char_001 --level T2_implied
    python scripts/dry_run.py --mode character --level T2_implied
    python scripts/dry_run.py --mode theme --level T2_implied
    python scripts/dry_run.py --mode style --level T3_artnude
    python scripts/dry_run.py --mode niche --level T1_suggestive
    python scripts/dry_run.py --mode variation --level T2_implied
    python scripts/dry_run.py --level T2_implied  # random mode (weighted)

Exit codes:
    0 = success
    1 = engine error (preflight, LLM, DB)
    2 = keyboard interrupt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.engine import (  # noqa: E402
    EngineError,
    InsufficientMemoryError,
    PipelineEngine,
    PreflightError,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dry run — Phase A only (LLM planning, no rendering)",
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
        help="Character id — only used in character mode (default: LRU selection)",
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
        "--model",
        default=None,
        help="Override model from registry (default: pipeline.default_model_id)",
    )
    parser.add_argument(
        "--llm",
        default=None,
        help=(
            "Override LLM from config/llm_models.yaml. When set, every "
            "agent role uses this LLM (overrides routing). Default: the "
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

    try:
        # If --character is given without --mode, default to character mode
        force_mode = args.mode
        if args.character and not force_mode:
            force_mode = "character"

        # Validate --llm against the registry at the CLI boundary so the
        # error fires before engine construction (faster signal than
        # waiting for the engine's own router validation).
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

        engine = PipelineEngine(
            dry_run=True,
            model_override=args.model,
            force_mode=force_mode,
            force_character=args.character,
        )

        result = engine.run_cycle(
            content_level=args.level,
            style_profile_id=args.style_profile,
            cli_llm_override=args.llm,
        )

        print(f"\nDry run result: {result['status']}")
        print(f"  Series:    {result['series_id']}")
        print(f"  Theme:     {result.get('theme')}")
        print(f"  Scenes:    {result.get('scenes')}")
        print(f"  Prompts:   {result.get('prompts')}")
        sys.exit(0)

    except PreflightError as exc:
        print(f"\nPreflight failed:\n{exc}", file=sys.stderr)
        sys.exit(1)
    except EngineError as exc:
        print(f"\nEngine error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(2)


if __name__ == "__main__":
    main()
