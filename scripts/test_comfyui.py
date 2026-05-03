#!/usr/bin/env python3
"""Smoke-test the ComfyUI client + workflow builder.

What it does:
  1. Loads config/comfyui_workflows/sdxl/base.json (must exist — see
     PROJECT_GUIDE.md → "Building the SDXL base.json workflow template").
  2. Builds a minimal workflow with a hardcoded test prompt and NO LoRAs.
  3. Submits it to ComfyUI at http://127.0.0.1:8188.
  4. Waits for the render to complete (default 5 minute timeout).
  5. Copies the resulting image into output/test/.
  6. Prints the render time and the destination path.

Pre-flight checklist:
  - ComfyUI running at http://127.0.0.1:8188
  - juggernautXL_ragnarokBy.safetensors in ~/AI/apps/ComfyUI/models/checkpoints/
  - config/comfyui_workflows/sdxl/base.json exists with semantic node IDs

Usage:
    python scripts/test_comfyui.py
    python scripts/test_comfyui.py --comfyui-url http://192.168.1.5:8188
    python scripts/test_comfyui.py --checkpoint other_model.safetensors
    python scripts/test_comfyui.py --seed 42 --width 1024 --height 1024
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import yaml

# Make src/ importable when running this file as a plain script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.render.comfyui_client import ComfyUIClient, ComfyUIError  # noqa: E402
from src.render.workflow_builder import (  # noqa: E402
    WorkflowBuilder,
    WorkflowTemplateError,
)


TEST_PROMPT = "a beautiful portrait photo, studio lighting"
TEST_NEGATIVE = (
    "blurry, low quality, bad anatomy, distorted face, extra limbs, "
    "deformed hands, watermark, text, logo"
)

CONFIG_PATH = PROJECT_ROOT / "config" / "pipeline.yaml"


def _load_comfyui_config() -> dict:
    """Read the comfyui block from pipeline.yaml.

    Falls back to an empty dict if the file is missing so the script still
    runs with hardcoded argparse defaults — useful before init/setup. We
    deliberately read this once at startup, not in ComfyUIClient itself,
    because the client should remain config-system-agnostic for unit tests.
    """
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("comfyui") or {}


def main() -> int:
    comfyui_cfg = _load_comfyui_config()
    default_url = comfyui_cfg.get("base_url", "http://127.0.0.1:8188")
    default_output_dir = comfyui_cfg.get(
        "output_dir", str(Path.home() / "AI" / "apps" / "ComfyUI" / "output")
    )
    default_timeout = int(comfyui_cfg.get("render_timeout_seconds", 300))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comfyui-url",
        default=default_url,
        help="ComfyUI HTTP endpoint (default from pipeline.yaml: %(default)s)",
    )
    parser.add_argument(
        "--comfyui-output-dir",
        default=default_output_dir,
        help="ComfyUI's on-disk output directory (default from pipeline.yaml: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint",
        default="juggernautXL_ragnarokBy.safetensors",
        help="Checkpoint filename (default: %(default)s)",
    )
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=1152)
    # Default seed is random so back-to-back smoke runs don't hit ComfyUI's
    # execution cache (identical workflow → empty `outputs` → confusing
    # "no images" failure). Pass --seed to pin it for debugging.
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducibility. Default: random per run.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=default_timeout,
        help="Render timeout in seconds (default from pipeline.yaml: %(default)s)",
    )
    args = parser.parse_args()

    workflow_dir = PROJECT_ROOT / "config" / "comfyui_workflows"
    output_dir = PROJECT_ROOT / "output" / "test"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Minimal duck-typed style profile. No LoRAs — keeps the test runnable
    # against the simplest possible base.json template (the seeded
    # realistic_soft / realistic_bold profiles include LoRAs and require
    # corresponding lora_loader_N nodes; that comes later).
    #
    # Synthetic capability flags: supports_ipadapter/supports_lora are set
    # True so the WorkflowBuilder's capability guards stay out of the way;
    # this is a checkpoint smoke test, not a model-registry test.
    style_profile = SimpleNamespace(
        workflow_family="sdxl",
        model_checkpoint=args.checkpoint,
        sampler="dpmpp_2m",
        scheduler="karras",
        steps=24,
        cfg=5.0,
        lora_stack=None,
        supports_ipadapter=True,
        supports_lora=True,
    )

    builder = WorkflowBuilder(workflow_dir)
    client = ComfyUIClient(
        base_url=args.comfyui_url,
        output_dir=args.comfyui_output_dir,
    )

    # Resolve the actual seed up-front so we can print it (WorkflowBuilder
    # also generates one internally when seed is None, but we want to log
    # whatever ends up in the workflow).
    effective_seed = (
        args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    )

    print(f"Building workflow from {workflow_dir / 'sdxl' / 'base.json'}")
    try:
        workflow = builder.build(
            prompt_text=TEST_PROMPT,
            negative_prompt=TEST_NEGATIVE,
            style_profile=style_profile,
            resolution=(args.width, args.height),
            seed=effective_seed,
        )
    except WorkflowTemplateError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Submitting test render to {client.base_url} ...")
    print(f"  Prompt:     {TEST_PROMPT!r}")
    print(f"  Resolution: {args.width} x {args.height}")
    print(f"  Checkpoint: {style_profile.model_checkpoint}")
    print(
        f"  Sampler:    {style_profile.sampler}/{style_profile.scheduler} "
        f"steps={style_profile.steps} cfg={style_profile.cfg} seed={effective_seed}"
    )

    start = time.time()
    try:
        # max_attempts=1: a smoke test should not retry on hangs — let the
        # user see the actual failure mode immediately.
        rendered = client.render_single_with_retry(
            workflow, max_attempts=1, timeout=args.timeout
        )
    except ComfyUIError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2
    elapsed = time.time() - start

    if not rendered:
        print(
            "\nERROR: ComfyUI returned no images. Does your base.json "
            "include a SaveImage node?",
            file=sys.stderr,
        )
        return 3

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    saved: list[Path] = []
    for i, img in enumerate(rendered):
        dest = output_dir / f"{timestamp}_test_{i}{img.file_path.suffix}"
        shutil.copy2(img.file_path, dest)
        saved.append(dest)

    print(f"\nOK — render complete in {elapsed:.1f}s")
    for path in saved:
        print(f"  saved: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
