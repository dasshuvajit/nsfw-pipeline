"""Image upscaler — 4x-UltraSharp via ComfyUI workflow.

Loads ``{family}/upscale.json`` workflow template and runs a single-image
upscale through the ComfyUI API. The template must have semantic node
IDs that this module injects values into:

  - ``load_image``: input image path (basename, must be in ComfyUI input/)
  - ``upscale_model_loader``: the upscale model name (default: ``4x-UltraSharp``)
  - ``save_image``: output prefix

Templates ship for the three CLIP-encoder families that benefit from a
post-hoc upscale pass — sdxl / pony / illustrious — at
``config/comfyui_workflows/{sdxl,pony,illustrious}/upscale.json``. Flux,
Chroma, and FLUX.2 native renders are already at 1024+ and post-hoc
upscaling gives marginal returns; those families are intentionally not
templated. ``commercial_mode`` is irrelevant here — upscaling does not
re-sample with the source model, only an ESRGAN backbone.

Phase F validates the template at construction time so a misconfigured
``upscale_enabled: true`` fails before any render begins, not mid-batch.

See ARCHITECTURE.md Section 17.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from src.render.comfyui_client import ComfyUIClient
from src.render.workflow_builder import WorkflowBuilder, WorkflowTemplateError

logger = logging.getLogger(__name__)


# Families that ship an upscale.json template. Calling Upscaler with a
# family outside this set raises at construction. Flux/Chroma/Flux2
# render natively at 1024+ so a post-hoc 1.4× upscale gives marginal
# benefit and isn't worth the maintenance.
_SUPPORTED_FAMILIES: frozenset[str] = frozenset({"sdxl", "pony", "illustrious"})

# Required semantic node IDs the template must declare. The upscaler
# injects values into ``inputs`` of each.
_REQUIRED_NODES: tuple[str, ...] = ("load_image", "upscale_model_loader", "save_image")


class UpscaleError(Exception):
    """Error during upscaling."""


class Upscaler:
    """4x-UltraSharp upscale via ComfyUI.

    Parameters
    ----------
    workflow_dir : Path
        Directory containing workflow templates (e.g. ``config/comfyui_workflows``).
    comfy_client : ComfyUIClient
        Initialized ComfyUI client.
    comfy_input_dir : Path
        ComfyUI's input directory for staging images.
    family_id : str
        Family of the source render (sdxl/pony/illustrious). Picks which
        ``{family}/upscale.json`` template to load.
    upscale_model : str
        Name of the upscale model file in ComfyUI's ``models/upscale_models/``.
    timeout : int
        Render timeout in seconds (upscaling can take longer than base renders).
    """

    def __init__(
        self,
        workflow_dir: Path,
        comfy_client: ComfyUIClient,
        comfy_input_dir: Path,
        *,
        family_id: str = "sdxl",
        upscale_model: str = "4x-UltraSharp.pth",
        timeout: int = 600,
    ) -> None:
        self.workflow_dir = workflow_dir
        self.comfy = comfy_client
        self.comfy_input_dir = comfy_input_dir
        self.family_id = family_id
        self.upscale_model = upscale_model
        self.timeout = timeout
        self._template: dict[str, Any] | None = None
        # Eager validation — fail before any render begins, not mid-batch.
        self._validate_template_path()

    @property
    def template_path(self) -> Path:
        return self.workflow_dir / self.family_id / "upscale.json"

    def _validate_template_path(self) -> None:
        """Verify the family is supported and the template file exists.

        Raised at construction time so a misconfigured ``upscale_enabled``
        flag is caught before any render starts.
        """
        if self.family_id not in _SUPPORTED_FAMILIES:
            raise UpscaleError(
                f"Upscale not supported for family {self.family_id!r}. "
                f"Templates ship only for {sorted(_SUPPORTED_FAMILIES)}; "
                f"flux / chroma / flux2 render natively at 1024+ and "
                f"don't benefit from a post-hoc upscale pass."
            )
        if not self.template_path.exists():
            raise UpscaleError(
                f"Upscale workflow template not found: {self.template_path}\n"
                f"Expected at config/comfyui_workflows/{self.family_id}/upscale.json"
            )

    def _load_template(self) -> dict[str, Any]:
        """Load and cache the upscale workflow template."""
        if self._template is not None:
            return self._template

        import json
        with open(self.template_path) as f:
            self._template = json.load(f)
        missing = [n for n in _REQUIRED_NODES if n not in self._template]
        if missing:
            raise UpscaleError(
                f"Template {self.template_path} missing required semantic "
                f"node IDs: {missing}. Expected: {list(_REQUIRED_NODES)}"
            )
        return self._template

    def upscale(
        self,
        image_path: str | Path,
        output_path: str | Path | None = None,
    ) -> Path:
        """Upscale a single image via ComfyUI.

        Parameters
        ----------
        image_path : str | Path
            Source image to upscale.
        output_path : str | Path | None
            Where to save the result. If None, saves alongside the source
            with ``_upscaled`` suffix.

        Returns
        -------
        Path
            The upscaled image path.
        """
        import copy
        import json

        image_path = Path(image_path)
        if not image_path.exists():
            raise UpscaleError(f"Source image not found: {image_path}")

        if output_path is None:
            output_path = image_path.parent / f"{image_path.stem}_upscaled{image_path.suffix}"
        output_path = Path(output_path)

        # Stage input image into ComfyUI input dir
        staged = self.comfy_input_dir / image_path.name
        if staged != image_path:
            shutil.copy2(image_path, staged)

        # Load and inject into template
        template = copy.deepcopy(self._load_template())

        # Inject values into semantic nodes
        if "load_image" in template:
            template["load_image"]["inputs"]["image"] = image_path.name
        if "upscale_model_loader" in template:
            template["upscale_model_loader"]["inputs"]["model_name"] = self.upscale_model
        if "save_image" in template:
            template["save_image"]["inputs"]["filename_prefix"] = f"upscale_{image_path.stem}"

        # Render
        try:
            prompt_id = self.comfy.queue_prompt(template)
            results = self.comfy.wait_for_completion(prompt_id, timeout=self.timeout)
        except Exception as exc:
            raise UpscaleError(f"ComfyUI upscale failed for {image_path.name}: {exc}") from exc

        if not results:
            raise UpscaleError(f"ComfyUI returned no images for upscale of {image_path.name}")

        # Copy the result to output path
        result_path = Path(results[0].file_path) if hasattr(results[0], "file_path") else Path(str(results[0]))
        if result_path.exists():
            shutil.copy2(result_path, output_path)
            logger.info("Upscaled: %s → %s", image_path.name, output_path.name)
            return output_path
        else:
            raise UpscaleError(f"Upscaled file not found on disk: {result_path}")

    def upscale_batch(
        self,
        images: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Upscale a batch of image dicts.

        Each dict must have ``file_path``. The upscaled version replaces
        the original file_path in the dict. On failure, the original is
        kept and a warning is logged.

        Returns the same list for chaining.
        """
        count = 0
        for img in images:
            fp = img.get("file_path")
            if not fp:
                continue
            try:
                upscaled = self.upscale(fp)
                img["file_path"] = str(upscaled)
                img["upscaled"] = True
                count += 1
            except UpscaleError as exc:
                logger.warning("Upscale failed for %s: %s", fp, exc)
                img["upscaled"] = False

        logger.info("Upscaled %d/%d images", count, len(images))
        return images
