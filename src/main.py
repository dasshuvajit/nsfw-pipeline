#!/usr/bin/env python3
"""Pipeline scheduler — execution mode router.

ARCHITECTURE.md Section 18 (Execution Modes) and Section 19 (Error
Handling & Recovery).

Three execution modes from ``pipeline.yaml → execution.mode``:

  - **manual**: print a message and exit. Use individual scripts.
  - **supervised**: start the scheduler. Engine pauses for human review.
  - **automated**: start the scheduler. Engine runs without pauses.

The scheduler:
  - Runs every N hours (calculated from ``pipeline.runs_per_day``)
  - Lock file at ``/tmp/nsfw_pipeline.lock`` prevents overlapping runs
  - Zombie series cleanup before each cycle
  - All exceptions caught and logged to ``run_log`` table — never crashes

Usage:
    python src/main.py
    python src/main.py --once           # single cycle then exit
    python src/main.py --mode character  # force a specific mode

Stop gracefully with Ctrl+C.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sqlite3
import sys
import time
from pathlib import Path

import yaml

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

logger = logging.getLogger(__name__)

LOCK_FILE = "/tmp/nsfw_pipeline.lock"
_PIPELINE_YAML = PROJECT_ROOT / "config" / "pipeline.yaml"

# Graceful shutdown flag
_shutdown = False


def _signal_handler(signum: int, frame: object) -> None:
    global _shutdown
    _shutdown = True
    logger.info("Shutdown signal received. Will exit after current cycle.")


def _load_config() -> dict:
    if _PIPELINE_YAML.exists():
        with open(_PIPELINE_YAML) as f:
            return yaml.safe_load(f) or {}
    return {}


def _resolve_db_path(cfg: dict) -> Path:
    raw = cfg.get("pipeline", {}).get("db_path", "nsfw_pipeline.db")
    return PROJECT_ROOT / raw


def _cleanup_zombies(db_path: Path) -> int:
    """Mark zombie series as failed.

    Any series stuck in ``rendering``/``filtering``/``packaging`` for
    >6 hours is considered dead from a crashed run.

    Returns the number of zombies cleaned up.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.execute(
            """
            UPDATE series SET status = 'failed'
            WHERE status IN ('rendering', 'filtering', 'packaging')
            AND created_at < datetime('now', '-6 hours')
            """
        )
        count = cursor.rowcount
        conn.commit()
        if count:
            logger.warning("Cleaned up %d zombie series", count)
        return count
    finally:
        conn.close()


def _log_run(
    db_path: Path,
    *,
    mode: str = "unknown",
    content_level: str = "T2_implied",
    series_id: str | None = None,
    execution_mode: str = "automated",
    status: str = "failed",
    images_generated: int = 0,
    images_selected: int = 0,
    duration_seconds: float | None = None,
    error_message: str | None = None,
) -> None:
    """Write a row to the run_log table."""
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute(
                """
                INSERT INTO run_log (
                    mode, content_level, series_id, execution_mode,
                    status, images_generated, images_selected,
                    duration_seconds, error_message
                ) VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    mode, content_level, series_id, execution_mode,
                    status, images_generated, images_selected,
                    duration_seconds, error_message,
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("Failed to write run_log: %s", exc)


def _select_content_level(cfg: dict) -> str:
    """Weighted random content level selection from pipeline.yaml."""
    import random
    weights = cfg.get("content_level_weights", {
        "T1_suggestive": 0.10,
        "T2_implied": 0.35,
        "T3_artnude": 0.45,
        "T4_explicit": 0.10,
    })
    levels = list(weights.keys())
    w = [weights[l] for l in levels]
    return random.choices(levels, weights=w, k=1)[0]


def run_safe(
    cfg: dict,
    db_path: Path,
    *,
    force_mode: str | None = None,
    force_level: str | None = None,
    cli_llm_override: str | None = None,
) -> None:
    """Execute one pipeline cycle with full error handling.

    Uses a lock file to prevent overlapping runs. Logs all outcomes
    to the ``run_log`` table.
    """
    execution_mode = cfg.get("execution", {}).get("mode", "supervised")

    # Lock file check
    if os.path.exists(LOCK_FILE):
        logger.warning("Previous cycle still running (lock file exists), skipping")
        return

    # Acquire lock
    try:
        with open(LOCK_FILE, "w") as f:
            f.write(str(os.getpid()))
    except OSError as exc:
        logger.error("Cannot create lock file %s: %s", LOCK_FILE, exc)
        return

    start = time.time()
    content_level = force_level or _select_content_level(cfg)

    try:
        # Zombie cleanup
        _cleanup_zombies(db_path)

        # Build engine
        engine = PipelineEngine(
            config=cfg,
            db_path=db_path,
            dry_run=False,
            model_override=None,
            force_mode=force_mode,
        )

        # Run cycle
        result = engine.run_cycle(
            content_level=content_level,
            cli_llm_override=cli_llm_override,
        )

        duration = time.time() - start
        status = result.get("status", "unknown")

        if status == "complete":
            logger.info(
                "Cycle complete: %s — %d images in %.1fs",
                result["series_id"], result.get("images_selected", 0), duration,
            )
        elif status == "aborted":
            logger.info("Cycle aborted by supervisor: %s", result["series_id"])
            _log_run(
                db_path,
                mode=result.get("mode", force_mode or "unknown"),
                content_level=content_level,
                series_id=result.get("series_id"),
                execution_mode=execution_mode,
                status="aborted",
                duration_seconds=duration,
                error_message="Human rejected",
            )
        else:
            logger.warning("Cycle ended with status: %s", status)

    except PreflightError as exc:
        duration = time.time() - start
        logger.error("Preflight failed: %s", exc)
        _log_run(
            db_path, mode=force_mode or "unknown",
            content_level=content_level, execution_mode=execution_mode,
            status="failed", duration_seconds=duration,
            error_message=f"Preflight: {exc}",
        )

    except InsufficientMemoryError as exc:
        duration = time.time() - start
        logger.error("Insufficient memory: %s", exc)
        _log_run(
            db_path, mode=force_mode or "unknown",
            content_level=content_level, execution_mode=execution_mode,
            status="failed", duration_seconds=duration,
            error_message=f"Memory: {exc}",
        )

    except SetTooSmall as exc:
        duration = time.time() - start
        logger.warning("Set too small: %s", exc)
        _log_run(
            db_path, mode=force_mode or "unknown",
            content_level=content_level, execution_mode=execution_mode,
            status="partial", duration_seconds=duration,
            error_message=str(exc),
        )

    except SupervisorAbort:
        duration = time.time() - start
        logger.info("Cycle aborted by supervisor")
        _log_run(
            db_path, mode=force_mode or "unknown",
            content_level=content_level, execution_mode=execution_mode,
            status="aborted", duration_seconds=duration,
            error_message="Human rejected",
        )

    except EngineError as exc:
        duration = time.time() - start
        logger.error("Engine error: %s", exc)
        _log_run(
            db_path, mode=force_mode or "unknown",
            content_level=content_level, execution_mode=execution_mode,
            status="failed", duration_seconds=duration,
            error_message=str(exc),
        )

    except KeyboardInterrupt:
        duration = time.time() - start
        logger.info("Interrupted by user")
        _log_run(
            db_path, mode=force_mode or "unknown",
            content_level=content_level, execution_mode=execution_mode,
            status="interrupted", duration_seconds=duration,
            error_message="KeyboardInterrupt",
        )
        raise

    except Exception as exc:
        duration = time.time() - start
        logger.exception("Unexpected error in pipeline cycle")
        _log_run(
            db_path, mode=force_mode or "unknown",
            content_level=content_level, execution_mode=execution_mode,
            status="failed", duration_seconds=duration,
            error_message=str(exc),
        )

    finally:
        # Release lock
        try:
            if os.path.exists(LOCK_FILE):
                os.remove(LOCK_FILE)
        except OSError as exc:
            logger.warning("Failed to remove lock file: %s", exc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NSFW Pipeline scheduler",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run exactly one cycle then exit",
    )
    parser.add_argument(
        "--mode",
        choices=["character", "theme", "style", "niche", "variation"],
        default=None,
        help="Force a specific mode (default: weighted random)",
    )
    parser.add_argument(
        "--level",
        choices=["T1_suggestive", "T2_implied", "T3_artnude", "T4_explicit"],
        default=None,
        help="Force a content tier (default: weighted random)",
    )
    parser.add_argument(
        "--llm",
        default=None,
        help=(
            "Override LLM from config/llm_models.yaml. Applies to "
            "every scheduler iteration uniformly — to switch LLMs "
            "mid-run, kill the scheduler and restart with the new "
            "--llm. Default: registry's default_llm."
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

    cfg = _load_config()
    db_path = _resolve_db_path(cfg)
    execution_mode = cfg.get("execution", {}).get("mode", "supervised")
    runs_per_day = cfg.get("pipeline", {}).get("runs_per_day", 3)

    # Check DB exists
    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        print("Run: python scripts/init_db.py", file=sys.stderr)
        sys.exit(1)

    # Mode routing
    if execution_mode == "manual":
        print("Execution mode is 'manual'.")
        print("Use individual scripts (prepare_prompts.py, render_prompts.py, dry_run.py) instead.")
        print("To start the scheduler, set execution.mode to 'supervised' or 'automated' in pipeline.yaml.")
        sys.exit(0)

    # Validate --llm against the registry at the CLI boundary so a
    # typo fails fast — before the scheduler installs jobs.
    if args.llm is not None:
        from src.memory.llm_registry import (
            LLMRegistryLoader,
            LLMNotFound,
        )
        try:
            LLMRegistryLoader().get_llm(args.llm, require_active=True)
        except LLMNotFound as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(2)

    # Calculate interval
    interval_hours = 24.0 / runs_per_day
    logger.info(
        "Starting scheduler: mode=%s, %d runs/day (every %.1f hours)",
        execution_mode, runs_per_day, interval_hours,
    )

    # Install signal handlers
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    if args.once:
        # Single cycle mode
        logger.info("Running single cycle …")
        run_safe(
            cfg, db_path,
            force_mode=args.mode,
            force_level=args.level,
            cli_llm_override=args.llm,
        )
        logger.info("Single cycle complete.")
        sys.exit(0)

    # Scheduler loop using schedule library
    try:
        import schedule
    except ImportError:
        print(
            "The 'schedule' library is required for the scheduler.\n"
            "Install it: pip install schedule>=1.2.1\n"
            "Or uncomment it in requirements.txt.",
            file=sys.stderr,
        )
        sys.exit(1)

    def _scheduled_run() -> None:
        if _shutdown:
            return
        # --llm applies uniformly to every scheduler iteration. To
        # switch mid-run, kill the scheduler and restart.
        run_safe(
            cfg, db_path,
            force_mode=args.mode,
            force_level=args.level,
            cli_llm_override=args.llm,
        )

    # Schedule at the calculated interval
    schedule.every(interval_hours).hours.do(_scheduled_run)

    # Run immediately on startup
    logger.info("Running initial cycle …")
    _scheduled_run()

    # Main loop
    logger.info("Scheduler active. Ctrl+C to stop.")
    while not _shutdown:
        schedule.run_pending()
        time.sleep(60)

    logger.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
