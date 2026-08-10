#!/usr/bin/env python3
"""Standalone true-4K upscaler — the manual, keepers-only gate of the staged
render pipeline.

Workflow: `art_series` renders a series to non-4K *review* images. You eyeball
them, copy the favourites into a folder, then run this on that folder. Each
image is upscaled to >= the target long edge (default 3840) via the SDXL
UltimateSDUpscale + detailer stage template — so 4K compute (~66% of a render)
is spent only on images you actually want, never on culls.

Stateless and base-model-agnostic: it only needs an input image + a stage-3
template, so it upscales output from ANY source (this pipeline or otherwise).

    python scripts/upscale_folder.py output/art_series/<ts>/keepers
    python scripts/upscale_folder.py my_picks --tier T4_explicit --out-dir my_picks_4k
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from src.render.comfyui_client import ComfyUIClient  # noqa: E402
from src.render.render_pipeline import resolve_render_pipeline  # noqa: E402
from src.render.workflow_builder import WorkflowBuilder  # noqa: E402


def compute_upscale_by(long_edge: int, target: int) -> float:
    """Multiplier to bring ``long_edge`` to >= ``target``.

    2-decimal CEIL (so rounding never lands just under target) and clamped to
    >= 1.0 (never downscale an already-large image)."""
    if long_edge <= 0:
        return 1.0
    return max(1.0, math.ceil((target / long_edge) * 100) / 100.0)


def upscale_folder(
    input_dir: Path, out_dir: Path, *, builder, client, template: str,
    target: int, seed: int | None, glob: str, skip_existing: bool,
    timeout: int = 2400,
) -> list[dict]:
    """Upscale every ``glob`` match in ``input_dir`` to ``out_dir`` as
    ``{stem}_4k.png``. Returns a per-image result list (for a summary /
    optional manifest). Failures are recorded and skipped, not fatal."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for p in sorted(input_dir.glob(glob)):
        if not p.is_file():
            continue
        dst = out_dir / f"{p.stem}_4k.png"
        if skip_existing and dst.exists():
            results.append({"src": str(p), "dst": str(dst), "skipped": True})
            continue
        w, h = Image.open(p).size
        upscale_by = compute_upscale_by(max(w, h), target)
        try:
            wf = builder.build_image_stage(
                external_template=template, image_path=str(p.resolve()),
                upscale_by=upscale_by, seed=seed)
            images = client.render_single_with_retry(wf, timeout=timeout)
            outs = [im for im in images if im.type == "output"] or images
            shutil.copy(outs[-1].file_path, dst)
            # Carry the source's tEXt chunks (incl. the A1111 'parameters'
            # block the refine stage embeds) onto the 4K master + note the
            # postprocess — the SOLD file stays self-describing.
            try:
                from PIL.PngImagePlugin import PngInfo
                with Image.open(p) as src_im:
                    src_text = dict(getattr(src_im, "text", None) or {})
                with Image.open(dst) as dst_im:
                    merged = {**dict(getattr(dst_im, "text", None) or {}),
                              **src_text}
                    info = PngInfo()
                    for k, v in merged.items():
                        if isinstance(v, str):
                            info.add_text(k, v)
                    info.add_text("postprocess",
                                  f"USDU x{upscale_by} Remacri "
                                  f"(upscale_folder.py)")
                    dst_im.load()
                    dst_im.save(dst, pnginfo=info)
            except Exception as exc:  # noqa: BLE001 — metadata never fatal
                print(f"  (metadata passthrough skipped: {exc})",
                      file=sys.stderr, flush=True)
            results.append({"src": str(p), "dst": str(dst), "upscale_by": upscale_by,
                            "src_size": [w, h]})
            print(f"  {p.name} ({w}x{h}) x{upscale_by} -> {dst.name}", flush=True)
        except Exception as exc:  # noqa: BLE001 — surface, continue
            results.append({"src": str(p), "error": str(exc)})
            print(f"  {p.name} FAILED: {exc}", file=sys.stderr, flush=True)
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="Standalone true-4K folder upscaler")
    ap.add_argument("input_dir", help="folder of selected review images to 4K")
    ap.add_argument("--out-dir", default="",
                    help="output dir (default: <input_dir>_4k)")
    ap.add_argument("--tier", default="T3_artnude",
                    help="(no-op since 2026-06-10 — the 4K stage is tier-neutral)")
    ap.add_argument("--template", default=None,
                    help="override the stage-3 template (relative to workflow_dir)")
    ap.add_argument("--target-long-edge", type=int, default=0,
                    help="target long edge px (default: render_pipeline.target_4k_long_edge)")
    ap.add_argument("--seed", type=int, default=None,
                    help="fixed seed for the upscale pass (default: random per image)")
    ap.add_argument("--glob", default="*.png", help="input glob (default *.png)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip images whose _4k output already exists")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).expanduser()
    if not input_dir.is_dir():
        ap.error(f"input_dir is not a directory: {input_dir}")

    cfg = yaml.safe_load((ROOT / "config/pipeline.yaml").read_text())
    cu = cfg["comfyui"]
    rp = resolve_render_pipeline(cfg.get("render_pipeline"))
    # --tier is a retained no-op alias: the T4 4K variant was deleted (it had
    # become byte-identical after the MPS post-upscale-detailer removal — the
    # 4K stage is tier-neutral; explicit detail comes from the review stage).
    template = args.template or rp["upscale_template"]
    target = args.target_long_edge or int(rp["target_4k_long_edge"])
    out_dir = (Path(args.out_dir).expanduser() if args.out_dir
               else input_dir.parent / f"{input_dir.name}_4k")

    builder = WorkflowBuilder(ROOT / cu.get("workflow_dir", "config/comfyui_workflows"))
    client = ComfyUIClient(base_url=cu["base_url"], output_dir=cu["output_dir"])

    print(f"=== Upscaling {input_dir} → {out_dir} via {Path(template).name} "
          f"(target {target}px) ===", flush=True)
    results = upscale_folder(
        input_dir, out_dir, builder=builder, client=client, template=template,
        target=target, seed=args.seed, glob=args.glob,
        skip_existing=args.skip_existing)
    ok = sum(1 for r in results if r.get("dst") and not r.get("skipped")
             and not r.get("error"))
    skipped = sum(1 for r in results if r.get("skipped"))
    failed = sum(1 for r in results if r.get("error"))
    print(f"\nupscaled {ok}/{len(results)} (skipped {skipped}, failed {failed}) "
          f"→ {out_dir}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
