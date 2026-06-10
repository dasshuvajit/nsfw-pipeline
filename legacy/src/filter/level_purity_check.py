"""Level purity assertion — ARCHITECTURE.md Section 17 rule 1.

Never mix soft and full images in the same set. This module provides a
simple assertion that checks every image in a list has the expected
content level.
"""

from __future__ import annotations

from typing import Any


class LevelPurityError(Exception):
    """One or more images have the wrong content level."""


def assert_level_purity(
    images: list[dict[str, Any]],
    expected_level: str,
) -> None:
    """Raise ``LevelPurityError`` if any image doesn't match ``expected_level``.

    Parameters
    ----------
    images : list[dict]
        Each dict must have a ``content_level`` key.
    expected_level : str
        One of the 4 tiers: ``T1_suggestive``, ``T2_implied``,
        ``T3_artnude``, ``T4_explicit``.
    """
    violations = []
    for i, img in enumerate(images):
        img_level = img.get("content_level")
        if img_level != expected_level:
            violations.append(
                f"  image[{i}] ({img.get('id', '?')}): "
                f"content_level={img_level!r} (expected {expected_level!r})"
            )

    if violations:
        detail = "\n".join(violations)
        raise LevelPurityError(
            f"Level purity check failed: {len(violations)} image(s) "
            f"don't match expected level {expected_level!r}:\n{detail}"
        )
