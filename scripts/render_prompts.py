#!/usr/bin/env python3
"""render_prompts — Phase B (+ Phase C) only. Renders prompts already
in the DB for a given series (or single scene) across one or more
target models.

Pairs with ``prepare_prompts.py``: prepare_prompts persists the
prompts; render_prompts loads them by ``(series_id, model_id)``,
renders, scores, and (unless ``--no-export``) packages + exports.

Use this when you want to:
  - Render an existing series with the same model used to create it.
  - Render the same series through a different model
    (after ``prepare_prompts --series-id S --models <other>``
    has populated that model's prompts).
  - Render a single scene across multiple models for side-by-side
    comparison without going through ``compare_models.py``.

Pre-flight:
  - ComfyUI is running at the configured URL.
  - For each ``--models`` entry, prompts already exist in the DB.
    Missing → script aborts with a hint to run ``prepare_prompts`` first.

Usage:
    # Whole series, single model
    python scripts/render_prompts.py --series-id ser_abc --models lustify_v7

    # Single scene, single model
    python scripts/render_prompts.py --scene-id ser_abc_scene_003 \\
        --models lustify_v7

    # Multi-model fan-out (sequential render — one ComfyUI checkpoint at a time)
    python scripts/render_prompts.py --series-id ser_abc \\
        --models lustify_v7,chroma_v10HD,flux_nsfw_71q8

    # Per-model templates (positional pairing: model[i] uses template[i])
    python scripts/render_prompts.py --series-id ser_abc \\
        --models lustify_v7,chroma_v10HD \\
        --templates system,templates/chroma/chroma_done_properly.json

Exit codes:
    0 = success (every requested model rendered)
    1 = engine error (preflight, render, missing series)
    2 = missing prompts for one or more models
        (run prepare_prompts first)
    3 = supervisor abort (human rejected images mid-pipeline)
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

from src.core.engine import (  # noqa: E402
    EngineError,
    InsufficientMemoryError,
    PipelineEngine,
    PreflightError,
    _load_config,
)
from src.review.supervisor import SupervisorAbort  # noqa: E402


# ── helpers ─────────────────────────────────────────────────────────


def _parse_csv(value: str | None) -> list[str]:
    """Comma-separated CLI value → ordered, deduped list."""
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


def _resolve_models(
    args_models: str | None, config: dict[str, Any],
) -> list[str]:
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


def _resolve_templates(
    args_templates: str | None, models: list[str],
) -> list[str | None]:
    """Templates pair positionally with ``models``.

    Rules:
      * No --templates                → ``[None] * len(models)``     (all use system)
      * 1 template, N models          → that template applied to all
      * N models, N templates         → positional pair (i ↔ i)
      * else                          → ``EngineError``
    'system' as a token → ``None`` (built-in family template).
    """
    parsed = _parse_csv(args_templates)
    if not parsed:
        return [None] * len(models)
    if len(parsed) == 1:
        # Apply to all (matches compare_models.py's "single template" rule).
        return [None if parsed[0] == "system" else parsed[0]] * len(models)
    if len(parsed) != len(models):
        raise EngineError(
            f"--templates count ({len(parsed)}) must equal --models count "
            f"({len(models)}) for positional pairing, or be 1 (applied to "
            f"all). Got: models={models}, templates={parsed}"
        )
    return [None if t == "system" else t for t in parsed]


def _validate_prompts_exist(
    db_path: Path,
    *,
    series_id: str | None,
    scene_id: str | None,
    models: list[str],
    llm_id: str | None = None,
) -> dict[str, int]:
    """For each model: count prompts matching ``(series_id, scene_id)``.

    When ``llm_id`` is set, only prompts from that LLM are counted —
    matching the run_phase_b filter. ``None`` skips the LLM filter
    (legacy single-LLM path).

    Returns ``{model_id: count}``. Caller decides whether to abort
    when any value is 0 (we always abort for the missing-prompts path
    so the operator gets a clear hint pointing at ``prepare_prompts``).
    """
    counts: dict[str, int] = {}
    conn = sqlite3.connect(str(db_path))
    try:
        for model_id in models:
            base = "SELECT COUNT(*) FROM prompts WHERE model_id = ?"
            params: list[Any] = [model_id]
            if scene_id:
                base += " AND scene_id = ?"
                params.append(scene_id)
            else:
                base += " AND series_id = ?"
                params.append(series_id)
            if llm_id is not None:
                base += " AND llm_id = ?"
                params.append(llm_id)
            row = conn.execute(base, params).fetchone()
            counts[model_id] = row[0] if row else 0
    finally:
        conn.close()
    return counts


def _llms_present_on_series(
    db_path: Path,
    *,
    series_id: str | None,
    scene_id: str | None,
    models: list[str],
) -> list[str]:
    """Return the distinct llm_ids present on the prompts for this
    target (series_id or scene_id) across the requested models.

    Used by the strict-ambiguity check (plan §3.5b): when 1+ LLMs are
    represented on a non-empty series, ``--llm`` becomes required so
    the user makes an explicit choice instead of silently rendering
    one LLM's data.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        placeholders = ", ".join(["?"] * len(models))
        if scene_id:
            base = (
                f"SELECT DISTINCT llm_id FROM prompts "
                f"WHERE scene_id = ? AND model_id IN ({placeholders})"
            )
            params: list[Any] = [scene_id, *models]
        else:
            base = (
                f"SELECT DISTINCT llm_id FROM prompts "
                f"WHERE series_id = ? AND model_id IN ({placeholders})"
            )
            params = [series_id, *models]
        rows = conn.execute(base, params).fetchall()
        return sorted(r[0] for r in rows if r[0])
    finally:
        conn.close()


def _resolve_series_id_for_scene(
    db_path: Path, scene_id: str,
) -> str | None:
    """Look up the series_id that owns this scene, or None if missing."""
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT series_id FROM scenes WHERE id = ?", (scene_id,),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _print_per_model_summary(
    summaries: list[dict[str, Any]], *, total_elapsed: float,
) -> None:
    """One-block stdout summary of every model that ran."""
    print()
    print(f"render_prompts complete in {total_elapsed:.1f}s")
    for s in summaries:
        model = s.get("model_id", "?")
        status = s.get("status", "?")
        if status == "complete":
            print(
                f"  {model:30s} status=complete   "
                f"rendered={s.get('images_rendered', 0):>3} "
                f"selected={s.get('images_selected', 0):>3} "
                f"-> {s.get('export_dir', '')}"
            )
        else:
            print(
                f"  {model:30s} status={status:10s} "
                f"reason={s.get('reason') or s.get('error', '?')}"
            )


# ── main ────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Phase B + Phase C — render existing DB prompts for one or "
            "more target models. Prompts must already exist (run "
            "prepare_prompts first)."
        ),
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--series-id",
        default=None,
        help="Render every scene in this series.",
    )
    target.add_argument(
        "--scene-id",
        default=None,
        help="Render a single scene (within whatever series owns it).",
    )
    parser.add_argument(
        "--models",
        default=None,
        help=(
            "Comma-separated model ids to render through "
            "(e.g. 'lustify_v7,chroma_v10HD'). Default: "
            "[pipeline.default_model_id]. Models render sequentially."
        ),
    )
    parser.add_argument(
        "--templates",
        default=None,
        help=(
            "Per-model external workflow templates (positional pairing "
            "with --models). 'system' = built-in family template. A "
            "single value applies to all models. Otherwise must equal "
            "--models count."
        ),
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help=(
            "Skip Phase C (set build + watermark + export + persist + "
            "memory + run_log). Useful for fast iteration when you only "
            "want to look at the rendered files in ComfyUI's output/."
        ),
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="SQLite DB path (default: pipeline.db_path)",
    )
    parser.add_argument(
        "--llm",
        default=None,
        help=(
            "Filter prompts by their generating LLM (id from "
            "config/llm_models.yaml). REQUIRED when the series has "
            "prompts from any LLM — silent fallback to a single LLM is "
            "a footgun. Omit only on empty/fresh series; in that case "
            "the registry's default_llm is used. See plan §3.5b."
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
        templates = _resolve_templates(args.templates, models)

        engine = PipelineEngine(
            config=config,
            db_path=Path(args.db_path) if args.db_path else None,
            dry_run=False,
        )

        # Resolve series_id (from --series-id directly, or via the
        # scene's owning series for --scene-id).
        if args.scene_id:
            series_id = _resolve_series_id_for_scene(
                engine.db_path, args.scene_id,
            )
            if series_id is None:
                print(
                    f"\nERROR: scene_id {args.scene_id!r} not found in DB.",
                    file=sys.stderr,
                )
                return 1
            scene_ids: list[str] | None = [args.scene_id]
        else:
            series_id = args.series_id
            scene_ids = None

        # Strict ambiguity check (plan §3.5b). When --llm is omitted
        # and the DB has prompts on this target, REQUIRE --llm so the
        # user makes a deliberate choice — silent fallback to one LLM
        # when several have prompts is a footgun. Only fresh / empty
        # targets fall through to default_llm.
        present_llms = _llms_present_on_series(
            engine.db_path,
            series_id=args.series_id,
            scene_id=args.scene_id,
            models=models,
        )
        if args.llm is None and present_llms:
            target_desc = (
                f"scene '{args.scene_id}'" if args.scene_id
                else f"series '{args.series_id}'"
            )
            print(
                f"\nERROR: {target_desc} has prompts from LLM(s): "
                f"{', '.join(repr(x) for x in present_llms)}.\n"
                f"Specify --llm <id> to choose which LLM's prompts to "
                f"render.\n",
                file=sys.stderr,
            )
            return 2

        # Validate --llm is in the registry when explicitly set.
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

        # Validate prompts exist for every model under the chosen LLM
        # (or all-LLMs when fresh) BEFORE kicking off ComfyUI.
        counts = _validate_prompts_exist(
            engine.db_path,
            series_id=args.series_id,
            scene_id=args.scene_id,
            models=models,
            llm_id=args.llm,
        )
        missing = [m for m, n in counts.items() if n == 0]
        if missing:
            target_desc = (
                f"scene '{args.scene_id}'" if args.scene_id
                else f"series '{args.series_id}'"
            )
            llm_hint = f" --llm {args.llm}" if args.llm else ""
            print(
                f"\nERROR: no prompts in DB for "
                f"{', '.join(repr(m) for m in missing)} on {target_desc}"
                f"{f' (llm={args.llm})' if args.llm else ''}.\n"
                f"Run prepare_prompts first:\n"
                f"  python scripts/prepare_prompts.py --series-id "
                f"{args.series_id or '<series>'} "
                f"--models {','.join(missing)}{llm_hint}",
                file=sys.stderr,
            )
            return 2

        logger.info(
            "render_prompts: series=%s%s, models=%s",
            series_id,
            f" scene={args.scene_id}" if args.scene_id else "",
            ",".join(models),
        )

        # Per-model render loop (sequential — ComfyUI loads one
        # checkpoint at a time; parallelism here would just thrash
        # VRAM).
        run_start = time.time()
        summaries: list[dict[str, Any]] = []

        for model_id, template in zip(models, templates):
            t0 = time.time()
            try:
                rendered_images, ctx_state = engine.run_phase_b(
                    series_id=series_id,
                    model_id=model_id,
                    scene_ids=scene_ids,
                    template_override=template,
                    cli_llm_override=args.llm,
                )
            except EngineError as exc:
                logger.error(
                    "Render failed for %s: %s", model_id, exc,
                )
                summaries.append({
                    "model_id": model_id,
                    "status": "failed",
                    "error": str(exc),
                })
                continue

            if ctx_state.get("aborted"):
                summaries.append({"model_id": model_id, "status": "aborted"})
                continue
            if not rendered_images:
                summaries.append({
                    "model_id": model_id,
                    "status": "failed",
                    "reason": "zero rendered",
                })
                continue

            if args.no_export:
                summaries.append({
                    "model_id": model_id,
                    "status": "complete",
                    "images_rendered": len(rendered_images),
                    "images_selected": len(rendered_images),
                    "export_dir": "(skipped via --no-export)",
                })
                continue

            elapsed_for_model = time.time() - t0
            try:
                result = engine.run_phase_c(
                    series_id=series_id,
                    model_id=ctx_state["ctx"].model_id,
                    content_level=ctx_state["ctx"].content_level,
                    rendered_images=rendered_images,
                    series_plan=ctx_state["series_plan"],
                    scenes=ctx_state["scenes"],
                    prompts=ctx_state["prompts"],
                    ctx=ctx_state["ctx"],
                    style_profile=ctx_state["style_profile"],
                    style_profile_id=ctx_state["style_profile_id"],
                    elapsed_seconds=elapsed_for_model,
                    llm_id=ctx_state.get("llm_id"),
                )
                summaries.append(result)
            except Exception as exc:
                logger.error(
                    "Phase C failed for %s: %s", model_id, exc,
                )
                summaries.append({
                    "model_id": model_id,
                    "status": "failed",
                    "error": f"phase_c: {exc}",
                })

        total = time.time() - run_start
        _print_per_model_summary(summaries, total_elapsed=total)

        # Exit code: 0 only if every requested model completed.
        any_failed = any(s.get("status") != "complete" for s in summaries)
        return 1 if any_failed else 0

    except PreflightError as exc:
        print(f"\nPreflight failed:\n{exc}", file=sys.stderr)
        return 1
    except InsufficientMemoryError as exc:
        print(f"\nInsufficient memory: {exc}", file=sys.stderr)
        return 1
    except SupervisorAbort:
        print("\nRender aborted by supervisor (human rejected images).")
        return 3
    except EngineError as exc:
        print(f"\nEngine error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 4


if __name__ == "__main__":
    sys.exit(main())
