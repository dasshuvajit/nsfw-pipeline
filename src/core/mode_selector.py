"""Mode selector — picks which pipeline mode runs next.

Weighted random selection from ``MODE_WEIGHTS``. The weights can be
overridden via ``pipeline.yaml → mode_weights``.

See ARCHITECTURE.md Section 4 and Sections 7–11.
"""

from __future__ import annotations

import logging
import random
from typing import Any

logger = logging.getLogger(__name__)

# Default weights per ARCHITECTURE.md.
# pipeline.yaml → mode_weights can override individual entries.
MODE_WEIGHTS: dict[str, float] = {
    "character": 0.60,
    "theme": 0.20,
    "niche": 0.10,
    "style": 0.05,
    "variation": 0.05,
}

VALID_MODES = set(MODE_WEIGHTS)


class ModeSelector:
    """Select the next mode for a pipeline cycle via weighted random.

    Parameters
    ----------
    config : dict
        Full ``pipeline.yaml`` config dict. Reads ``mode_weights`` to
        override the defaults.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        overrides = config.get("mode_weights", {})
        self.weights: dict[str, float] = {**MODE_WEIGHTS}
        for k, v in overrides.items():
            if k in self.weights:
                self.weights[k] = float(v)

    def select(self, *, force_mode: str | None = None) -> str:
        """Return the mode name for the next cycle.

        Parameters
        ----------
        force_mode : str | None
            If set, bypass selection and return this mode directly.
            Used by ``--mode`` CLI flag and ``dry_run.py``.
        """
        if force_mode:
            if force_mode not in VALID_MODES:
                raise ValueError(
                    f"Unknown mode {force_mode!r}. "
                    f"Valid modes: {sorted(VALID_MODES)}"
                )
            logger.info("ModeSelector: forced mode=%r", force_mode)
            return force_mode

        modes = list(self.weights.keys())
        weights = [self.weights[m] for m in modes]
        selected = random.choices(modes, weights=weights, k=1)[0]
        logger.info(
            "ModeSelector: weighted random → %r (weights: %s)",
            selected, {m: f"{w:.0%}" for m, w in self.weights.items()},
        )
        return selected
