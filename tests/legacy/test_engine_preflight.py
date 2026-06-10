"""Preflight folder-selection for UNET-style model families.

Regression: before the fix, ``flux2`` fell through the chroma/flux
branch and hit the ``checkpoints/`` default — but FLUX.2 Klein ships
as a safetensors UNET that ComfyUI's ``UNETLoader`` reads from
``diffusion_models/``. The preflight check was asking the user to
download the file into the wrong folder.
"""

from __future__ import annotations

import pytest

from src.core.engine import _model_subfolder


@pytest.mark.parametrize(
    "family, expected",
    [
        ("sdxl", "checkpoints"),
        ("pony", "checkpoints"),
        ("illustrious", "checkpoints"),
        ("chroma", "unet"),
        ("flux", "unet"),
        ("flux2", "diffusion_models"),
    ],
)
def test_model_subfolder_per_family(family, expected):
    assert _model_subfolder(family) == expected


def test_flux2_does_not_route_to_checkpoints():
    # Explicit guard against the original bug — even if someone
    # rewrites _model_subfolder, flux2 must never end up in
    # checkpoints/ (which is where CheckpointLoaderSimple reads from,
    # not UNETLoader).
    assert _model_subfolder("flux2") != "checkpoints"


def test_unknown_family_falls_back_to_checkpoints():
    assert _model_subfolder("totally_new_family") == "checkpoints"
