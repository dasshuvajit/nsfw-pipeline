"""Contact sheet generator — grid of images with quality score labels.

Takes a list of image paths and quality scores and creates a 4-column
grid image for human review during supervised mode.

See ARCHITECTURE.md Section 4 (review/contact_sheet.py).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

COLUMNS = 4
THUMB_SIZE = (384, 384)
PADDING = 8
LABEL_HEIGHT = 24
BG_COLOR = (32, 32, 32)
LABEL_BG = (0, 0, 0, 180)
LABEL_COLOR = (255, 255, 255)


def create_contact_sheet(
    images: list[dict[str, Any]],
    output_path: str | Path,
    *,
    columns: int = COLUMNS,
    thumb_size: tuple[int, int] = THUMB_SIZE,
) -> Path:
    """Create a contact sheet grid and save to ``output_path``.

    Parameters
    ----------
    images : list[dict]
        Each dict needs ``file_path`` (str) and optionally
        ``quality_score`` (float) for the label.
    output_path : str | Path
        Where to write the PNG.
    columns : int
        Number of columns in the grid.
    thumb_size : tuple[int, int]
        Size of each thumbnail cell.

    Returns
    -------
    Path
        The written file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(images)
    if n == 0:
        logger.warning("Contact sheet: no images to render")
        # Create a minimal placeholder
        placeholder = Image.new("RGB", (400, 100), BG_COLOR)
        draw = ImageDraw.Draw(placeholder)
        draw.text((20, 40), "No images", fill=LABEL_COLOR)
        placeholder.save(output_path)
        return output_path

    rows = (n + columns - 1) // columns
    cell_w = thumb_size[0] + PADDING
    cell_h = thumb_size[1] + LABEL_HEIGHT + PADDING
    sheet_w = columns * cell_w + PADDING
    sheet_h = rows * cell_h + PADDING

    sheet = Image.new("RGB", (sheet_w, sheet_h), BG_COLOR)
    draw = ImageDraw.Draw(sheet)

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for idx, img_data in enumerate(images):
        row = idx // columns
        col = idx % columns
        x = PADDING + col * cell_w
        y = PADDING + row * cell_h

        file_path = Path(img_data.get("file_path", ""))
        score = img_data.get("quality_score")

        # Load and resize thumbnail
        try:
            if file_path.exists():
                thumb = Image.open(file_path)
                thumb.thumbnail(thumb_size, Image.Resampling.LANCZOS)
                # Center in cell
                offset_x = x + (thumb_size[0] - thumb.width) // 2
                offset_y = y + (thumb_size[1] - thumb.height) // 2
                sheet.paste(thumb, (offset_x, offset_y))
            else:
                # Missing image placeholder
                draw.rectangle(
                    [x, y, x + thumb_size[0], y + thumb_size[1]],
                    fill=(64, 64, 64),
                )
                draw.text(
                    (x + 10, y + thumb_size[1] // 2),
                    "missing",
                    fill=(128, 128, 128),
                    font=font,
                )
        except Exception as exc:
            logger.warning("Contact sheet: failed to load %s: %s", file_path, exc)
            draw.rectangle(
                [x, y, x + thumb_size[0], y + thumb_size[1]],
                fill=(64, 0, 0),
            )

        # Quality score label
        label_y = y + thumb_size[1]
        score_text = f"#{idx+1}  {score:.3f}" if score is not None else f"#{idx+1}  --"
        draw.text((x + 4, label_y + 4), score_text, fill=LABEL_COLOR, font=font)

    sheet.save(output_path, quality=90)
    logger.info("Contact sheet: %d images → %s (%dx%d)", n, output_path, sheet_w, sheet_h)
    return output_path
