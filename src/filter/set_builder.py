"""Set builder — select the final image set from scored renders.

Applies level purity, quality threshold, min/max count, and optional
regeneration from unrendered prompts if the set is too small.

See ARCHITECTURE.md Section 17 (Set Builder).
"""

from __future__ import annotations

import logging
from typing import Any

from src.filter.level_purity_check import LevelPurityError, assert_level_purity

logger = logging.getLogger(__name__)


class SetTooSmall(Exception):
    """Set has fewer than MIN_IMAGES even after a regeneration attempt."""


class SetBuilder:
    """Build a final image set from scored images.

    Parameters
    ----------
    min_images : int
        Minimum images for a valid set (default 10).
    max_images : int
        Maximum images to include (default 25).
    quality_cutoff : float
        Minimum composite quality score (default 0.55).
    """

    def __init__(
        self,
        min_images: int = 10,
        max_images: int = 25,
        quality_cutoff: float = 0.55,
    ) -> None:
        self.min_images = min_images
        self.max_images = max_images
        self.quality_cutoff = quality_cutoff

    def build(
        self,
        images: list[dict[str, Any]],
        content_level: str,
        *,
        regenerate_fn: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Build the final set.

        Parameters
        ----------
        images : list[dict]
            Scored image dicts. Must have ``quality_score`` and
            ``content_level`` keys.
        content_level : str
            Expected content level. Raises ``LevelPurityError`` on mismatch.
        regenerate_fn : callable | None
            Optional callback ``fn(deficit: int) -> list[dict]`` that
            renders additional images from unrendered prompts. Called once
            if the set is below ``min_images``.

        Returns
        -------
        list[dict]
            Selected images, sorted by quality (best first), capped at
            ``max_images``.

        Raises
        ------
        LevelPurityError
            If any image has the wrong content level.
        SetTooSmall
            If the set is still below ``min_images`` after regeneration.
        """
        # Level purity check
        assert_level_purity(images, content_level)

        # Filter by quality threshold
        above = [
            img for img in images
            if (img.get("quality_score") or 0) >= self.quality_cutoff
        ]
        dropped = len(images) - len(above)
        if dropped > 0:
            logger.info(
                "SetBuilder: %d/%d images below quality cutoff %.2f",
                dropped, len(images), self.quality_cutoff,
            )

        # Try regeneration if below minimum
        if len(above) < self.min_images and regenerate_fn is not None:
            deficit = self.min_images - len(above)
            logger.info(
                "SetBuilder: %d images, need %d — regenerating %d",
                len(above), self.min_images, deficit,
            )
            new_images = regenerate_fn(deficit)
            if new_images:
                assert_level_purity(new_images, content_level)
                good = [
                    img for img in new_images
                    if (img.get("quality_score") or 0) >= self.quality_cutoff
                ]
                above.extend(good)
                logger.info(
                    "SetBuilder: regeneration produced %d/%d passing images",
                    len(good), len(new_images),
                )

        if len(above) < self.min_images:
            raise SetTooSmall(
                f"Only {len(above)} images above quality cutoff "
                f"{self.quality_cutoff} after retry. "
                f"Need at least {self.min_images}. Aborting set."
            )

        # Sort by quality (best first) and cap
        above.sort(key=lambda x: x.get("quality_score", 0), reverse=True)
        selected = above[: self.max_images]

        logger.info(
            "SetBuilder: selected %d images (quality range %.3f–%.3f)",
            len(selected),
            selected[-1].get("quality_score", 0) if selected else 0,
            selected[0].get("quality_score", 0) if selected else 0,
        )
        return selected
