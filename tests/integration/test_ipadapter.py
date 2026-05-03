#!/usr/bin/env python3
"""IPAdapter A/B visual test — Phase 1 step 14/15.

ARCHITECTURE.md Section 7 describes Mode 1 (Character) renders as
"IPAdapter ON" — the reference face/identity is locked across every
scene in a set so a series feels like the same person in different
poses, not 15 different people.

Before we trust IPAdapter in the auto-pipeline we need an eyeball test:
render the *same* prompt + the *same* seed twice — once with IPAdapter
and once without — and look at them side by side. If IPAdapter is
working, the WITH-set should hold a consistent face across all five
scenes and the NO-set should drift.

What this script does:

  1. Loads a character (default: char_001) from the DB.
  2. Resolves a reference image:
       - ``--reference-image`` overrides everything
       - else falls back to ``characters.reference_image_path`` column
       - else exits with a helpful error
  3. Picks N scenes (default 5) from the matching content-level pool in
     ``scripts/render_set.py`` (so the two scripts can't drift apart).
  4. Generates N **deterministic** seeds (12345, 13345, 14345, ...) so
     the A and B renders for scene i use the *same* seed — that's the
     whole point: any visual difference is the IPAdapter, not the
     RNG.
  5. For each scene+seed:
       - Render WITHOUT IPAdapter via ``sdxl/base.json``
       - Render WITH IPAdapter via ``sdxl/ipadapter.json`` (weight 0.7)
     and copies both files into a side-by-side directory tree under
     ``output/test/ipadapter_AB/{timestamp}/{no_ipadapter,with_ipadapter}/``.
  6. Prints both directories so you can `open` them in Finder and
     compare frame-by-frame.

This script is **not** scored, **not** persisted to the DB, and **not**
hooked into ``CharacterManager.update_usage`` — it's a transient
visual diagnostic. Run it once after the IPAdapter template lands,
once after any IPAdapter weight tune, and once whenever you swap the
reference image.

Pre-flight:
  - ``python scripts/init_db.py`` has been run.
  - ``python scripts/bootstrap_character.py`` has populated the
    character.
  - ``config/comfyui_workflows/sdxl/base.json`` exists with
    ``lora_loader_0`` + ``lora_loader_1`` nodes (Known Issues).
  - ``config/comfyui_workflows/sdxl/ipadapter.json`` exists (see
    PROJECT_GUIDE.md → "Building the SDXL ipadapter.json workflow
    template").
  - ``--reference-image`` points at a real file, OR the character row's
    ``reference_image_path`` column is set.
  - ComfyUI is running.

Examples:
    # Easy: char_001 has reference_image_path set, default 5 scenes
    python tests/integration/test_ipadapter.py

    # Override the reference image
    python tests/integration/test_ipadapter.py \\
        --reference-image characters/char_001/reference.png

    # Tweak count, weight, level
    python tests/integration/test_ipadapter.py --count 8 --ipadapter-weight 0.85 --level T3_artnude
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Make src/ AND scripts/ importable when running this file as a plain script.
# tests/integration/<file>.py → ../../.. is the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.content_level import ContentLevelLoader, UnknownContentLevel  # noqa: E402
from src.core.ratio_selector import RatioSelector  # noqa: E402
from src.memory.character_manager import (  # noqa: E402
    CharacterManager,
    CharacterNotFound,
)
from src.prompt.builder import PromptBuilder  # noqa: E402
from src.prompt.sanitizer import PromptSanitizer  # noqa: E402
from src.render.comfyui_client import ComfyUIClient, ComfyUIError  # noqa: E402
from src.render.workflow_builder import (  # noqa: E402
    WorkflowBuilder,
    WorkflowTemplateError,
)

# Reuse the scene pools + staging helpers from render_set.py so the two
# scripts can never drift on what counts as a "soft scene" or how the
# reference image is staged into ComfyUI's input dir.
from scripts.render_set import (  # noqa: E402
    FULL_SCENES,
    SOFT_SCENES,
    _load_config,
    _load_style_profile,
    _resolve_comfy_input_dir,
    _resolve_db_path,
    _resolve_model,
    _stage_reference_image,
)
from src.core.style_profile_adapter import StyleProfileForWorkflow  # noqa: E402
from src.memory.family_loader import FamilyLoader  # noqa: E402
from src.memory.model_registry import ModelRegistryLoader  # noqa: E402

# Integration script — invoked manually as a CLI; needs ComfyUI + a
# bootstrapped character. Skipped by default during `pytest tests/` so
# the unit-test run stays hermetic. Run via `python
# tests/integration/test_ipadapter.py ...` to exercise.
import pytest  # noqa: E402
pytestmark = pytest.mark.skip(
    reason="Integration script; run manually as `python tests/integration/test_ipadapter.py`."
)

WORKFLOW_DIR = PROJECT_ROOT / "config" / "comfyui_workflows"

# Deterministic seed base. Each scene i uses BASE + i*1000 so seeds are
# spaced far enough apart that any modulo collisions in samplers don't
# alias them, and so the A and B renders for the same scene get the
# *exact same seed*. Hardcoded so two runs of the script produce
# byte-comparable seeds (the *images* will still differ run-to-run if
# the IPAdapter weight or reference changes).
_SEED_BASE = 12345
_SEED_STRIDE = 1000


@dataclass
class _PairRecord:
    """One A/B pair's bookkeeping."""

    index: int
    scene: dict[str, Any]
    ratio: str
    resolution: tuple[int, int]
    prompt_text: str
    negative_prompt: str
    seed: int

    no_ipadapter_path: Path | None = None
    with_ipadapter_path: Path | None = None
    no_ipadapter_error: str | None = None
    with_ipadapter_error: str | None = None


# ─── reference resolution ──────────────────────────────────────────────────


def _resolve_reference_image(
    character: dict[str, Any], cli_override: str | None
) -> Path:
    """Pick the reference image path: CLI > character row > error."""
    if cli_override:
        path = Path(cli_override).expanduser()
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        return path
    column = character.get("reference_image_path")
    if column:
        path = Path(column).expanduser()
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        return path
    raise SystemExit(
        f"ERROR: no reference image. Pass --reference-image PATH, or "
        f"set characters.reference_image_path for {character.get('id')!r}."
    )


# ─── render helpers ────────────────────────────────────────────────────────


def _do_render(
    client: ComfyUIClient,
    wf_builder: WorkflowBuilder,
    style_for_wf: StyleProfileForWorkflow,
    rec: _PairRecord,
    *,
    ipadapter_image: str | None,
    ipadapter_weight: float,
    timeout: int,
    max_attempts: int,
) -> tuple[Path | None, str | None]:
    """Build + submit one render. Returns (rendered_path, error_message)."""
    try:
        workflow = wf_builder.build(
            prompt_text=rec.prompt_text,
            negative_prompt=rec.negative_prompt,
            style_profile=style_for_wf,
            resolution=rec.resolution,
            seed=rec.seed,
            ipadapter_image=ipadapter_image,
            ipadapter_weight=ipadapter_weight,
        )
        rendered = client.render_single_with_retry(
            workflow, max_attempts=max_attempts, timeout=timeout
        )
        if not rendered:
            return None, "ComfyUI returned no images"
        return rendered[0].file_path, None
    except WorkflowTemplateError as exc:
        return None, f"WorkflowTemplateError: {exc}"
    except ComfyUIError as exc:
        return None, f"ComfyUIError: {exc}"


def _save_pair_file(
    rendered: Path | None,
    dest_dir: Path,
    rec: _PairRecord,
    label: str,
) -> Path | None:
    """Copy a rendered image into the side-by-side dest dir."""
    if not rendered:
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    pose_slug = (rec.scene.get("pose") or "scene").replace(" ", "_")
    dest = (
        dest_dir / f"{rec.index:02d}_{pose_slug}_{label}{rendered.suffix}"
    )
    shutil.copy2(rendered, dest)
    return dest


# ─── main ──────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--character",
        default="char_001",
        help="Character id (default: %(default)s).",
    )
    parser.add_argument(
        "--reference-image",
        default=None,
        help="Reference image path. Overrides characters.reference_image_path.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of A/B pairs to render. Default: %(default)s.",
    )
    parser.add_argument(
        "--level",
        choices=("T1_suggestive", "T2_implied", "T3_artnude", "T4_explicit"),
        default="T2_implied",
        help="Content tier (picks the matching scene pool). Default: %(default)s.",
    )
    parser.add_argument(
        "--ipadapter-weight",
        type=float,
        default=0.7,
        help="IPAdapter weight (0.0-1.0). Default: %(default)s.",
    )
    parser.add_argument(
        "--db-path", default=None, help="Override DB path. Default: pipeline.yaml"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Per-image render timeout (seconds). Default: pipeline.yaml or 300",
    )
    args = parser.parse_args()

    cfg = _load_config()
    db_path = _resolve_db_path(cfg, args.db_path)
    if not db_path.exists():
        print(
            f"ERROR: database not found at {db_path}.\n"
            f"Run `python scripts/init_db.py` first.",
            file=sys.stderr,
        )
        return 1

    # ----- 1. Character + style profile + content rules -------------------
    try:
        character = CharacterManager(db_path).get_character(args.character)
    except CharacterNotFound as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    style_row = _load_style_profile(db_path, character["style_profile_id"])
    # test_ipadapter.py never overrides the model — it resolves from
    # characters.model_id (or pipeline.default_model_id) and holds it
    # constant across A/B. The comparison only varies IPAdapter presence,
    # not the checkpoint.
    default_model_id = (cfg.get("pipeline") or {}).get("default_model_id")
    if not default_model_id:
        print(
            "ERROR: pipeline.default_model_id is not set in "
            "config/pipeline.yaml.",
            file=sys.stderr,
        )
        return 1
    try:
        model_entry = _resolve_model(
            db_path, style_row, character, default_model_id, None
        )
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 8
    style_for_wf = StyleProfileForWorkflow(style_row, model_entry)

    registry = ModelRegistryLoader()
    family = FamilyLoader().get_family(model_entry.family)
    prompt_guide = registry.get_prompt_guide(model_entry.id)
    avoid_words = prompt_guide.avoid_words if prompt_guide else None

    try:
        rules = ContentLevelLoader(db_path).load(args.level)
    except UnknownContentLevel as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    # ----- 2. Reference image staging -------------------------------------
    try:
        ref_path = _resolve_reference_image(character, args.reference_image)
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 7

    if not ref_path.exists():
        print(f"ERROR: reference image not found: {ref_path}", file=sys.stderr)
        return 7

    comfy_input_dir = _resolve_comfy_input_dir(cfg)
    ipadapter_basename = _stage_reference_image(ref_path, comfy_input_dir)

    # ----- 3. Scene + prompt + ratio (shared between A and B) -------------
    pool = (
        SOFT_SCENES if args.level in ("T1_suggestive", "T2_implied") else FULL_SCENES
    )
    if args.count <= 0:
        print(f"ERROR: --count must be > 0 (got {args.count})", file=sys.stderr)
        return 4
    if args.count > len(pool):
        print(
            f"ERROR: --count={args.count} but only {len(pool)} {args.level} "
            f"scenes are defined in render_set.py. Pick a smaller count or "
            f"add more scenes to the pool.",
            file=sys.stderr,
        )
        return 4
    scenes = [dict(s) for s in pool[: args.count]]

    builder = PromptBuilder()
    sanitizer = PromptSanitizer.from_rules(rules)

    aspect_weights = cfg.get("aspect_ratio_weights")
    if not aspect_weights:
        print(
            "ERROR: pipeline.yaml is missing aspect_ratio_weights.",
            file=sys.stderr,
        )
        return 4
    selector = RatioSelector(aspect_weights)

    pairs: list[_PairRecord] = []
    for i, scene in enumerate(scenes):
        prompt_dict = builder.build_one(
            character, scene, style_row,
            family=family, avoid_words=avoid_words,
        )
        sanitized_text = sanitizer.sanitize_text(prompt_dict["prompt_text"])
        pick = selector.select(
            scene,
            content_level=args.level,
            model_checkpoint=style_for_wf.model_checkpoint,
        )
        pairs.append(
            _PairRecord(
                index=i,
                scene=scene,
                ratio=pick.ratio,
                resolution=pick.resolution,
                prompt_text=sanitized_text,
                negative_prompt=prompt_dict["negative_prompt"],
                seed=_SEED_BASE + i * _SEED_STRIDE,
            )
        )

    # ----- 4. Output dirs --------------------------------------------------
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    test_root = (
        PROJECT_ROOT / "output" / "test" / "ipadapter_AB" / timestamp
    )
    no_dir = test_root / "no_ipadapter"
    with_dir = test_root / "with_ipadapter"

    # ----- 5. Render loop --------------------------------------------------
    comfy_cfg = cfg.get("comfyui") or {}
    comfy_url = comfy_cfg.get("base_url", "http://127.0.0.1:8188")
    comfy_output_dir = comfy_cfg.get(
        "output_dir", str(Path.home() / "AI" / "apps" / "ComfyUI" / "output")
    )
    timeout = args.timeout or int(comfy_cfg.get("render_timeout_seconds", 300))
    max_attempts = int(comfy_cfg.get("max_retry_per_image", 3))

    wf_builder = WorkflowBuilder(WORKFLOW_DIR)
    client = ComfyUIClient(base_url=comfy_url, output_dir=comfy_output_dir)

    print(f"IPAdapter A/B test — {len(pairs)} pairs")
    print(f"  character:        {args.character}")
    print(f"  style:            {style_row['id']}")
    print(f"  level:            {args.level}")
    print(f"  reference image:  {ref_path}")
    print(f"  staged as:        {ipadapter_basename}")
    print(f"  weight:           {args.ipadapter_weight}")
    print(f"  output:           {test_root}")
    print()

    render_start = time.time()
    for rec in pairs:
        print(
            f"[{rec.index + 1:02d}/{len(pairs)}] {rec.ratio} "
            f"{rec.resolution[0]}x{rec.resolution[1]} seed={rec.seed} "
            f"pose={rec.scene.get('pose','?')!r}"
        )
        # A) NO IPAdapter (base.json) - always render this first so a
        # broken IPAdapter template doesn't prevent us from at least
        # getting the baseline frames.
        rendered_no, err_no = _do_render(
            client, wf_builder, style_for_wf, rec,
            ipadapter_image=None, ipadapter_weight=args.ipadapter_weight,
            timeout=timeout, max_attempts=max_attempts,
        )
        rec.no_ipadapter_path = _save_pair_file(rendered_no, no_dir, rec, "noipa")
        rec.no_ipadapter_error = err_no
        if err_no:
            print(f"     A (base):     ERROR {err_no}", file=sys.stderr)
        else:
            print(f"     A (base):     {rec.no_ipadapter_path.name}")

        # B) WITH IPAdapter
        rendered_with, err_with = _do_render(
            client, wf_builder, style_for_wf, rec,
            ipadapter_image=ipadapter_basename,
            ipadapter_weight=args.ipadapter_weight,
            timeout=timeout, max_attempts=max_attempts,
        )
        rec.with_ipadapter_path = _save_pair_file(
            rendered_with, with_dir, rec, "ipa"
        )
        rec.with_ipadapter_error = err_with
        if err_with:
            print(f"     B (ipadapter): ERROR {err_with}", file=sys.stderr)
        else:
            print(f"     B (ipadapter): {rec.with_ipadapter_path.name}")

    render_seconds = time.time() - render_start

    # ----- 6. Summary ------------------------------------------------------
    no_ok = sum(1 for r in pairs if r.no_ipadapter_path)
    with_ok = sum(1 for r in pairs if r.with_ipadapter_path)
    print()
    print("─" * 60)
    print(f"IPAdapter A/B test complete — {render_seconds:.1f}s")
    print("─" * 60)
    print(f"  pairs total:        {len(pairs)}")
    print(f"  no_ipadapter ok:    {no_ok}/{len(pairs)}")
    print(f"  with_ipadapter ok:  {with_ok}/{len(pairs)}")
    print()
    print(f"Side-by-side directories:")
    print(f"  A (no IPAdapter):   {no_dir}")
    print(f"  B (with IPAdapter): {with_dir}")
    print()
    print(
        "Open both directories in Finder and step through frame by frame.\n"
        "If IPAdapter is working, every B frame should hold the same face\n"
        "across all scenes; the A frames will drift between scenes."
    )

    # If at least one B render succeeded the test is "useful" — we
    # exit 0. If every B render failed (typical: missing template,
    # wrong node names) exit 6 so CI shells can branch on it.
    if with_ok == 0:
        return 6
    return 0


if __name__ == "__main__":
    sys.exit(main())
