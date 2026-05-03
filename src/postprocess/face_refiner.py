"""Face refiner — FaceDetailer via ComfyUI workflow.

Loads ``sdxl/face_detail.json`` workflow template and runs a
single-image face refinement through the ComfyUI API. The template
uses Impact Pack's FaceDetailer node to detect and re-render faces
at higher fidelity.

Required semantic node IDs in the template:

  - ``load_image``: input image path (basename, must be in ComfyUI input/)
  - ``face_detailer``: the FaceDetailer node (guide_size, denoise, etc.)
  - ``save_image``: output prefix

The workflow template is Phase 4 — it does not exist yet. This module
is ready to use once ``config/comfyui_workflows/sdxl/face_detail.json``
is built per ``docs/COMFYUI_WORKFLOWS.md``.

See ARCHITECTURE.md Section 17.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from src.render.comfyui_client import ComfyUIClient

logger = logging.getLogger(__name__)


class FaceRefineError(Exception):
    """Error during face refinement."""


class FaceRefiner:
    """FaceDetailer post-processing via ComfyUI.

    Parameters
    ----------
    workflow_dir : Path
        Directory containing workflow templates.
    comfy_client : ComfyUIClient
        Initialized ComfyUI client.
    comfy_input_dir : Path
        ComfyUI's input directory for staging images.
    denoise_strength : float
        How aggressively to re-render the face (0.0–1.0). Lower values
        preserve more of the original; higher values re-render more.
        Default 0.35 is conservative.
    timeout : int
        Render timeout in seconds.
    """

    def __init__(
        self,
        workflow_dir: Path,
        comfy_client: ComfyUIClient,
        comfy_input_dir: Path,
        *,
        denoise_strength: float = 0.35,
        timeout: int = 600,
    ) -> None:
        self.workflow_dir = workflow_dir
        self.comfy = comfy_client
        self.comfy_input_dir = comfy_input_dir
        self.denoise_strength = denoise_strength
        self.timeout = timeout
        self._template: dict[str, Any] | None = None

    def _load_template(self) -> dict[str, Any]:
        """Load and cache the face_detail workflow template."""
        if self._template is not None:
            return self._template

        import json
        template_path = self.workflow_dir / "sdxl" / "face_detail.json"
        if not template_path.exists():
            raise FaceRefineError(
                f"Face detail workflow template not found: {template_path}\n"
                f"Build it per docs/COMFYUI_WORKFLOWS.md → sdxl/face_detail.json"
            )
        with open(template_path) as f:
            self._template = json.load(f)
        return self._template

    def refine(
        self,
        image_path: str | Path,
        output_path: str | Path | None = None,
    ) -> Path:
        """Refine faces in a single image via ComfyUI FaceDetailer.

        Parameters
        ----------
        image_path : str | Path
            Source image to refine.
        output_path : str | Path | None
            Where to save. If None, saves alongside the source with
            ``_facefix`` suffix.

        Returns
        -------
        Path
            The refined image path.
        """
        import copy

        image_path = Path(image_path)
        if not image_path.exists():
            raise FaceRefineError(f"Source image not found: {image_path}")

        if output_path is None:
            output_path = image_path.parent / f"{image_path.stem}_facefix{image_path.suffix}"
        output_path = Path(output_path)

        # Stage input image
        staged = self.comfy_input_dir / image_path.name
        if staged != image_path:
            shutil.copy2(image_path, staged)

        # Load and inject
        template = copy.deepcopy(self._load_template())

        if "load_image" in template:
            template["load_image"]["inputs"]["image"] = image_path.name
        if "face_detailer" in template:
            template["face_detailer"]["inputs"]["denoise"] = self.denoise_strength
        if "save_image" in template:
            template["save_image"]["inputs"]["filename_prefix"] = f"facefix_{image_path.stem}"

        # Render
        try:
            prompt_id = self.comfy.queue_prompt(template)
            results = self.comfy.wait_for_completion(prompt_id, timeout=self.timeout)
        except Exception as exc:
            raise FaceRefineError(
                f"ComfyUI face refinement failed for {image_path.name}: {exc}"
            ) from exc

        if not results:
            raise FaceRefineError(
                f"ComfyUI returned no images for face refinement of {image_path.name}"
            )

        result_path = Path(results[0].file_path) if hasattr(results[0], "file_path") else Path(str(results[0]))
        if result_path.exists():
            shutil.copy2(result_path, output_path)
            logger.info("Face refined: %s → %s", image_path.name, output_path.name)
            return output_path
        else:
            raise FaceRefineError(f"Refined file not found on disk: {result_path}")

    def refine_batch(
        self,
        images: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Refine faces in a batch of image dicts.

        Each dict must have ``file_path``. On failure, the original is
        kept and a warning is logged.

        Returns the same list for chaining.
        """
        count = 0
        for img in images:
            fp = img.get("file_path")
            if not fp:
                continue
            try:
                refined = self.refine(fp)
                img["file_path"] = str(refined)
                img["face_refined"] = True
                count += 1
            except FaceRefineError as exc:
                logger.warning("Face refinement failed for %s: %s", fp, exc)
                img["face_refined"] = False

        logger.info("Face-refined %d/%d images", count, len(images))
        return images
