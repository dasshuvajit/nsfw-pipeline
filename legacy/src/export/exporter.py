"""Set exporter — output/{content_level}/{series_id}/ directory tree.

Creates the level-split export directory per ARCHITECTURE.md Section 17:

    output/{content_level}/{series_id}/
        images/          — final selected images
        preview/         — contact sheet + teaser crop (future)
        metadata.json    — title, description, tags (for platform posting)
        manifest.json    — full production data (prompts, seeds, scores, model)

See ARCHITECTURE.md Section 17 (Export).
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ExportError(Exception):
    """Export failed."""


class Exporter:
    """Export a finished set to the output directory tree."""

    def __init__(self, output_root: str | Path = "output") -> None:
        self.output_root = Path(output_root)

    def export(
        self,
        *,
        series_id: str,
        content_level: str,
        images: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
        series_plan: dict[str, Any] | None = None,
        model_id: str | None = None,
        style_profile_id: str | None = None,
        llm_id: str | None = None,
        target_id: str | None = None,
        target_kind: str = "model",
    ) -> Path:
        """Export the set and return the export directory path.

        Parameters
        ----------
        series_id : str
            Series identifier.
        content_level : str
            One of the 4 tiers: ``T1_suggestive``, ``T2_implied``,
            ``T3_artnude``, ``T4_explicit``.
        images : list[dict]
            Selected images. Each must have ``file_path`` (absolute or
            relative to project root).
        metadata : dict | None
            LLM-generated metadata (title, description, tags).
        series_plan : dict | None
            The series plan for the manifest.
        model_id : str | None
            Which model checkpoint produced these images. For
            family-kind renders, this is the chosen
            ``--render-with-model`` checkpoint.
        style_profile_id : str | None
            Which style profile was used.
        llm_id : str | None
            Which generating LLM produced the prompts these renders came
            from. When set, exports land under the per-LLM segment.
        target_id : str | None
            Discriminator segment between ``llm_id`` and ``images/``.
            Equals ``model_id`` for model-kind renders, ``family_id``
            for family-kind renders. Falls back to ``model_id`` when
            ``None`` so model-kind callers needn't pass it explicitly.
        target_kind : str
            ``'model'`` (default) or ``'family'``. Recorded in the
            manifest JSON for forensic reproducibility.

        Symmetric path shape (model-kind and family-kind alike):
        ``output_root / content_level / series_id / llm_id / target_id /``.
        ``llm_id`` is omitted when ``None`` (no per-LLM segment); when
        omitted but ``target_id`` is set, the path collapses to
        ``output_root / content_level / series_id / target_id /``.
        """
        # Resolve the target_id segment — defaults to model_id for
        # model-kind callers that don't explicitly pass it.
        effective_target_id = target_id or model_id
        export_dir = self.output_root / content_level / series_id
        if llm_id:
            export_dir = export_dir / llm_id
        if effective_target_id:
            export_dir = export_dir / effective_target_id
        images_dir = export_dir / "images"
        preview_dir = export_dir / "preview"

        images_dir.mkdir(parents=True, exist_ok=True)
        preview_dir.mkdir(parents=True, exist_ok=True)

        # Copy images
        exported_paths = []
        for i, img in enumerate(images):
            src = Path(img["file_path"])
            if not src.exists():
                logger.warning("Export: source image not found: %s", src)
                continue
            dst = images_dir / f"{i+1:03d}_{src.name}"
            shutil.copy2(src, dst)
            exported_paths.append(str(dst))

        if not exported_paths:
            raise ExportError(
                f"No images could be exported for series {series_id}. "
                f"Check that the file_path values in the image list are valid."
            )

        # Write metadata.json (for platform posting)
        meta = metadata or {}
        metadata_out = {
            "series_id": series_id,
            "content_level": content_level,
            "title": meta.get("title", ""),
            "description": meta.get("description", ""),
            "tags": meta.get("tags", []),
            "image_count": len(exported_paths),
        }
        meta_path = export_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata_out, f, indent=2)

        # Write manifest.json (full production data)
        manifest = {
            "series_id": series_id,
            "content_level": content_level,
            "model_id": model_id,
            "target_kind": target_kind,
            "target_id": effective_target_id,
            "llm_id": llm_id,
            "style_profile_id": style_profile_id,
            "series_plan": series_plan,
            "image_count": len(exported_paths),
            "images": [
                {
                    "export_path": path,
                    "original_path": img.get("file_path"),
                    "seed": img.get("seed"),
                    "quality_score": img.get("quality_score"),
                    "aesthetic_score": img.get("aesthetic_score"),
                    "prompt_text": img.get("prompt_text"),
                    "aspect_ratio": img.get("aspect_ratio"),
                    "width": img.get("width"),
                    "height": img.get("height"),
                }
                for path, img in zip(exported_paths, images)
            ],
        }
        manifest_path = export_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(
            "Exported %d images to %s (metadata + manifest written)",
            len(exported_paths),
            export_dir,
        )
        return export_dir
