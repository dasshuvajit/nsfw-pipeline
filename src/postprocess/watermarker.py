"""Subtle corner watermark for public-gallery-visible tiers.

Adds a semi-transparent text watermark (e.g. "@YourHandle") to the
bottom-right corner of images. Applied ONLY to tiers that can surface
on DA's public gallery (T1_suggestive, T2_implied) — the paywalled
tiers (T3_artnude, T4_explicit) are sold unwatermarked because buyers
expect a clean file and the paywall itself does the attribution work.

Configuration from ``pipeline.yaml``:

.. code-block:: yaml

    watermark:
      enabled: true
      text: "@YourDAHandle"
      position: "bottom-right"
      opacity: 0.3

See ARCHITECTURE.md Section 17 (Metadata & Packaging).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Default watermark config
_DEFAULTS = {
    "enabled": True,
    "text": "@YourDAHandle",
    "position": "bottom-right",
    "opacity": 0.3,
    "font_size_ratio": 0.025,  # font size as fraction of image width
    "margin_ratio": 0.02,      # margin as fraction of image dimensions
}


class WatermarkError(Exception):
    """Error during watermarking."""


class Watermarker:
    """Apply a subtle text watermark to images.

    Parameters
    ----------
    config : dict
        The ``watermark`` section from ``pipeline.yaml``.
        Falls back to ``_DEFAULTS`` for missing keys.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = {**_DEFAULTS, **(config or {})}
        self.enabled: bool = bool(cfg["enabled"])
        self.text: str = str(cfg["text"])
        self.position: str = str(cfg["position"])
        self.opacity: float = float(cfg["opacity"])
        self.font_size_ratio: float = float(cfg.get("font_size_ratio", 0.025))
        self.margin_ratio: float = float(cfg.get("margin_ratio", 0.02))

    # Tiers that receive a watermark. T3/T4 are paywalled, so we hand over
    # clean files; T1/T2 can appear on DA's public gallery and deserve a
    # subtle attribution mark.
    WATERMARKED_TIERS = frozenset({"T1_suggestive", "T2_implied"})

    def apply(
        self,
        image_path: str | Path,
        output_path: str | Path | None = None,
        *,
        content_level: str = "T2_implied",
    ) -> Path:
        """Apply watermark to an image file.

        Watermarks only tiers in :attr:`WATERMARKED_TIERS` (T1_suggestive,
        T2_implied). Paywalled tiers (T3_artnude, T4_explicit) are
        returned as-is.

        Parameters
        ----------
        image_path : str | Path
            Source image to watermark.
        output_path : str | Path | None
            Where to save. If None, overwrites the source.
        content_level : str
            Content tier; only public-gallery tiers get watermarked.

        Returns
        -------
        Path
            The output file path.
        """
        image_path = Path(image_path)
        output_path = Path(output_path) if output_path else image_path

        if not self.enabled:
            logger.debug("Watermarker: disabled, skipping %s", image_path.name)
            return output_path

        if content_level not in self.WATERMARKED_TIERS:
            logger.debug(
                "Watermarker: level=%s not in WATERMARKED_TIERS, skipping",
                content_level,
            )
            return output_path

        if not image_path.exists():
            raise WatermarkError(f"Image not found: {image_path}")

        try:
            src_img = Image.open(image_path)
            # Capture PNG tEXt chunks BEFORE convert() — the conversion
            # creates a new Image without the .text dict.
            existing_text = (
                dict(getattr(src_img, "text", {}) or {})
                if image_path.suffix.lower() == ".png"
                else {}
            )
            img = src_img.convert("RGBA")
        except Exception as exc:
            raise WatermarkError(f"Cannot open {image_path}: {exc}") from exc

        w, h = img.size

        # Create transparent overlay
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Calculate font size relative to image width
        font_size = max(12, int(w * self.font_size_ratio))
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", font_size)
            except (OSError, IOError):
                font = ImageFont.load_default()

        # Measure text
        bbox = draw.textbbox((0, 0), self.text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # Calculate position
        margin_x = int(w * self.margin_ratio)
        margin_y = int(h * self.margin_ratio)

        if self.position == "bottom-right":
            x = w - text_w - margin_x
            y = h - text_h - margin_y
        elif self.position == "bottom-left":
            x = margin_x
            y = h - text_h - margin_y
        elif self.position == "top-right":
            x = w - text_w - margin_x
            y = margin_y
        elif self.position == "top-left":
            x = margin_x
            y = margin_y
        else:
            # Default to bottom-right
            x = w - text_w - margin_x
            y = h - text_h - margin_y

        # Draw with configured opacity
        alpha = max(0, min(255, int(255 * self.opacity)))
        color = (255, 255, 255, alpha)
        draw.text((x, y), self.text, fill=color, font=font)

        # Composite and save
        watermarked = Image.alpha_composite(img, overlay)
        # Convert back to RGB for JPEG/PNG output
        result = watermarked.convert("RGB")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Preserve PNG metadata chunks (parameters / nsfw_pipeline) that
        # were attached upstream by the engine after Phase 4b. Lossy
        # without this — a watermarker round-trip would silently drop
        # reproduction parameters.
        save_kwargs: dict[str, Any] = {"quality": 95}
        if (
            output_path.suffix.lower() == ".png"
            and existing_text
        ):
            from PIL import PngImagePlugin
            info = PngImagePlugin.PngInfo()
            for key, value in existing_text.items():
                info.add_text(str(key), str(value))
            save_kwargs["pnginfo"] = info
        result.save(output_path, **save_kwargs)
        logger.debug("Watermarked: %s → %s", image_path.name, output_path.name)
        return output_path

    def apply_batch(
        self,
        images: list[dict[str, Any]],
        *,
        content_level: str = "T2_implied",
    ) -> list[dict[str, Any]]:
        """Apply watermark to a batch of image dicts (in place).

        Each dict must have a ``file_path`` key. The watermark is applied
        directly to the file (overwriting the original).

        Returns the same list for chaining.
        """
        if not self.enabled or content_level not in self.WATERMARKED_TIERS:
            return images

        count = 0
        for img in images:
            fp = img.get("file_path")
            if not fp:
                continue
            try:
                self.apply(fp, content_level=content_level)
                count += 1
            except WatermarkError as exc:
                logger.warning("Watermark failed for %s: %s", fp, exc)

        if count:
            logger.info("Watermarked %d/%d images", count, len(images))
        return images
