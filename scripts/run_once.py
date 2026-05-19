#!/usr/bin/env python3
"""Run once — execute a single full pipeline cycle (plan + render + package).

Unlike ``dry_run.py`` (Phase A only), this runs the complete pipeline:
  1. Phase A: LLM planning (mode selection, series plan, scenes, prompts)
  2. Phase B: ComfyUI rendering + scoring + filtering
  3. Phase C: Set packaging + watermark + export

This is the primary way to test a full cycle without starting the
scheduler loop.  Equivalent to ``python src/main.py --once`` but
with more granular CLI flags.

Pre-flight:
  - ``ollama serve`` is running (for Phase A)
  - ``python scripts/init_db.py`` has been run
  - ComfyUI is running at the configured URL (for Phase B)

Usage:
    python scripts/run_once.py --level T2_implied
    python scripts/run_once.py --mode theme --level T3_artnude
    python scripts/run_once.py --mode niche --level T1_suggestive
    python scripts/run_once.py --level T2_implied --verbose

Exit codes:
    0 = success
    1 = engine error (preflight, LLM, render, DB)
    2 = keyboard interrupt
    3 = supervisor abort (human rejected in supervised mode)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.engine import (  # noqa: E402
    EngineError,
    InsufficientMemoryError,
    PipelineEngine,
    PreflightError,
)
from src.filter.set_builder import SetTooSmall  # noqa: E402
from src.review.supervisor import SupervisorAbort  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run once — single full pipeline cycle (plan + render + package)",
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
        "--template",
        default=None,
        help=(
            "Path to an external ComfyUI workflow template "
            "(e.g. templates/chroma/chroma_done_properly.json). The "
            "template's baked-in checkpoint / LoRAs / sampler all run "
            "as authored; the pipeline injects only positive/negative "
            "prompt, seed, and resolution. IPAdapter is forced off. "
            "See docs/COMFYUI_WORKFLOWS.md § External templates."
        ),
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

    logger = logging.getLogger(__name__)

    try:
        start = time.time()

        # Validate --llm against the registry at the CLI boundary.
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
            dry_run=False,
            model_override=args.model,
            force_mode=args.mode,
            template_override=args.template,
        )

        result = engine.run_cycle(
            content_level=args.level,
            style_profile_id=args.style_profile,
            cli_llm_override=args.llm,
        )

        duration = time.time() - start
        status = result.get("status", "unknown")

        print(f"\nCycle complete in {duration:.1f}s")
        print(f"  Status:          {status}")
        print(f"  Series:          {result.get('series_id')}")
        print(f"  Mode:            {result.get('mode')}")
        print(f"  Images rendered: {result.get('images_generated', 0)}")
        print(f"  Images selected: {result.get('images_selected', 0)}")
        sys.exit(0)

    except PreflightError as exc:
        print(f"\nPreflight failed:\n{exc}", file=sys.stderr)
        sys.exit(1)
    except InsufficientMemoryError as exc:
        print(f"\nInsufficient memory: {exc}", file=sys.stderr)
        sys.exit(1)
    except SetTooSmall as exc:
        logger.warning("Set too small: %s", exc)
        print(f"\nSet too small (partial run): {exc}", file=sys.stderr)
        sys.exit(1)
    except SupervisorAbort:
        print("\nCycle aborted by supervisor (human rejected).")
        sys.exit(3)
    except EngineError as exc:
        print(f"\nEngine error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(2)


if __name__ == "__main__":
    main()
