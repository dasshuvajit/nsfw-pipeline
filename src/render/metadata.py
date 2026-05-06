"""PNG metadata embedding for rendered images.

Per Phase 4b of the prompt-quality plan, every exported PNG carries
two chunks of reproduction metadata so a downstream tool (Civitai,
sd-prompt-reader, AUTOMATIC1111) can recover the parameters that
produced it without going back to the SQLite database:

* ``parameters`` — AUTOMATIC1111 / Civitai format. Single string with
  the positive prompt, ``Negative prompt:`` line, and a comma-separated
  ``Steps: N, Sampler: X, ...`` line.
* ``nsfw_pipeline`` — pipeline-native JSON. Carries the structured
  facets, vocab_version, scene_id linkage, content_level — fields that
  don't fit AUTOMATIC1111's flat shape.

The two chunks coexist; ``parameters`` is for interop, ``nsfw_pipeline``
is for round-tripping back into our own DB.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, PngImagePlugin


logger = logging.getLogger(__name__)


def build_a1111_parameters(
    *,
    prompt_text: str,
    negative_prompt: str | None,
    steps: int | None,
    sampler: str | None,
    scheduler: str | None,
    cfg_scale: float | None,
    seed: int | None,
    model: str | None,
    size: tuple[int, int] | None = None,
    clip_skip: int | None = None,
) -> str:
    """Render the AUTOMATIC1111-format ``parameters`` string.

    Civitai / sd-prompt-reader / A1111 webui all read this format::

        <positive prompt>
        Negative prompt: <negative>
        Steps: 30, Sampler: DPM++ 2M Karras, CFG scale: 6.0, Seed: 12345,
        Size: 832x1216, Model: lustify_v7, Clip skip: 2

    Missing fields are silently dropped from the trailing line — the
    consumer libraries are tolerant of partial data.
    """
    parts = [prompt_text.strip()]
    if negative_prompt and negative_prompt.strip():
        parts.append(f"Negative prompt: {negative_prompt.strip()}")

    tail_pairs: list[str] = []
    if steps is not None:
        tail_pairs.append(f"Steps: {steps}")
    if sampler:
        sampler_str = sampler if not scheduler else f"{sampler} {scheduler}"
        tail_pairs.append(f"Sampler: {sampler_str}")
    if cfg_scale is not None:
        tail_pairs.append(f"CFG scale: {cfg_scale}")
    if seed is not None:
        tail_pairs.append(f"Seed: {seed}")
    if size:
        tail_pairs.append(f"Size: {size[0]}x{size[1]}")
    if model:
        tail_pairs.append(f"Model: {model}")
    if clip_skip is not None:
        tail_pairs.append(f"Clip skip: {clip_skip}")
    if tail_pairs:
        parts.append(", ".join(tail_pairs))

    return "\n".join(parts)


def build_pipeline_metadata(
    *,
    vocab_version: int,
    family: str,
    model_id: str,
    scene_id: str | None,
    series_id: str | None,
    prompt_hash: str | None,
    seed: int | None,
    sampler: str | None,
    scheduler: str | None,
    steps: int | None,
    cfg: float | None,
    content_level: str | None,
    structured_facet: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    llm_id: str | None = None,
    target_kind: str = "model",
    render_model_id: str | None = None,
) -> str:
    """Build the ``nsfw_pipeline`` JSON chunk content.

    Carries the fields A1111 format can't represent: structured facet
    enum tags, vocab_version (for cross-version reproduction), scene
    linkage, content_level, the generating LLM (``llm_id``), and the
    target_kind / render_model_id pair so a forensic reader can
    answer "which LLM produced this prompt? was the prompt
    family-level or model-level? which checkpoint actually rendered?"
    from the PNG alone. Returned as a JSON string ready to insert
    via ``PngInfo.add_text``.

    For family-kind renders (``target_kind='family'``), ``model_id``
    holds the family id (e.g. 'flux') and ``render_model_id`` holds
    the actual checkpoint that produced the PNG (e.g.
    'flux_nsfw_71q8'). For model-kind, the two collapse —
    ``render_model_id`` may be ``None`` and readers use ``model_id``.
    """
    payload: dict[str, Any] = {
        "vocab_version": vocab_version,
        "family": family,
        "model_id": model_id,
        "target_kind": target_kind,
        "render_model_id": render_model_id,
        "llm_id": llm_id,
        "scene_id": scene_id,
        "series_id": series_id,
        "prompt_hash": prompt_hash,
        "seed": seed,
        "sampler": sampler,
        "scheduler": scheduler,
        "steps": steps,
        "cfg": cfg,
        "content_level": content_level,
    }
    if structured_facet:
        # Drop None values to keep the chunk small.
        payload["structured_facet"] = {
            k: v for k, v in structured_facet.items() if v is not None
        }
    if extra:
        for k, v in extra.items():
            if v is not None and k not in payload:
                payload[k] = v
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def write_png_metadata(
    image_path: Path | str,
    *,
    a1111_parameters: str | None = None,
    pipeline_metadata: str | None = None,
) -> None:
    """Re-save ``image_path`` (PNG) with the two metadata chunks attached.

    No-op for non-PNG files (JPEG doesn't carry tEXt chunks; we'd need
    EXIF UserComment for those, which is out of scope for Phase 4b).
    Existing chunks are preserved when not overwritten by the args.
    """
    image_path = Path(image_path)
    if image_path.suffix.lower() != ".png":
        logger.debug(
            "write_png_metadata: skipping non-PNG file %s", image_path,
        )
        return
    if a1111_parameters is None and pipeline_metadata is None:
        return

    try:
        with Image.open(image_path) as img:
            img_copy = img.copy()
            existing = getattr(img, "text", {}) or {}
    except (FileNotFoundError, OSError) as exc:
        logger.warning(
            "write_png_metadata: cannot open %s: %s", image_path, exc,
        )
        return

    info = PngImagePlugin.PngInfo()
    # Carry over any chunks already present that we're not overwriting.
    for key, value in existing.items():
        if key in {"parameters", "nsfw_pipeline"}:
            continue
        info.add_text(str(key), str(value))
    if a1111_parameters is not None:
        info.add_text("parameters", a1111_parameters)
    if pipeline_metadata is not None:
        info.add_text("nsfw_pipeline", pipeline_metadata)
    img_copy.save(image_path, "PNG", pnginfo=info)


def read_png_metadata(image_path: Path | str) -> dict[str, str]:
    """Return all tEXt chunks on a PNG (parameters + nsfw_pipeline + others).

    Convenience wrapper used by tests to confirm round-trip integrity.
    Returns an empty dict for non-PNG files or read failures.
    """
    image_path = Path(image_path)
    if image_path.suffix.lower() != ".png":
        return {}
    try:
        with Image.open(image_path) as img:
            return dict(getattr(img, "text", {}) or {})
    except (FileNotFoundError, OSError):
        return {}
