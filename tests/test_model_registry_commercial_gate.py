"""ModelRegistryLoader commercial_mode license gate.

When ``compliance.commercial_mode`` is true, the registry refuses to
register any model whose YAML declares ``commercial_use: false`` (e.g.
FLUX NCL-licensed Klein 9B). This prevents the pipeline from rendering
with a non-commercial model when output is destined for a paid platform.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.memory.model_registry import (
    ModelNotFound,
    ModelRegistryLoader,
)


_NON_COMMERCIAL_YAML = textwrap.dedent(
    """\
    id: flux2_klein_fixture
    display_name: Flux2 Klein (fixture)
    filename: flux-2-klein-9b.safetensors
    architecture: flux2
    family: flux2
    default_sampler: euler
    default_scheduler: simple
    default_steps: 4
    default_cfg: 1.0
    default_clip_skip: null
    supports_ipadapter: false
    supports_lora: true
    vae_filename: flux2-vae.safetensors
    text_encoder: qwen_3_8b_fp8mixed.safetensors
    license: flux_ncl
    commercial_use: false
    active: true
    """
)

_COMMERCIAL_YAML = textwrap.dedent(
    """\
    id: sdxl_commercial_fixture
    display_name: SDXL (fixture)
    filename: sdxl_commercial.safetensors
    architecture: sdxl
    family: sdxl
    default_sampler: dpmpp_2m
    default_scheduler: karras
    default_steps: 28
    default_cfg: 6.5
    default_clip_skip: 2
    supports_ipadapter: true
    supports_lora: true
    license: creativeml_openrail_m
    commercial_use: true
    active: true
    """
)


@pytest.fixture
def models_dir(tmp_path: Path) -> Path:
    d = tmp_path / "models"
    d.mkdir()
    (d / "flux2_klein_fixture.yaml").write_text(_NON_COMMERCIAL_YAML)
    (d / "sdxl_commercial_fixture.yaml").write_text(_COMMERCIAL_YAML)
    return d


def test_non_commercial_mode_registers_everything(models_dir):
    loader = ModelRegistryLoader(
        models_dir=models_dir, commercial_mode=False
    )
    # Both models visible.
    assert loader.get_model("flux2_klein_fixture").commercial_use is False
    assert loader.get_model("sdxl_commercial_fixture").commercial_use is True


def test_commercial_mode_drops_non_commercial_model(models_dir, caplog):
    import logging
    caplog.set_level(logging.ERROR, logger="src.memory.model_registry")
    loader = ModelRegistryLoader(
        models_dir=models_dir, commercial_mode=True
    )
    # Non-commercial model is unreachable.
    with pytest.raises(ModelNotFound):
        loader.get_model("flux2_klein_fixture")
    # Commercial model still resolves.
    assert loader.get_model("sdxl_commercial_fixture").id == "sdxl_commercial_fixture"
    # Error was logged.
    assert any(
        "flux2_klein_fixture" in r.message and "commercial_use=False" in r.message
        for r in caplog.records
    )


def test_legacy_yaml_without_license_keys_defaults_to_commercial(tmp_path):
    # Legacy YAMLs (pre-license-gate) don't carry `license:` /
    # `commercial_use:` keys. The loader must treat the missing
    # `commercial_use` as True so commercial_mode=True doesn't
    # accidentally drop every existing model.
    d = tmp_path / "models"
    d.mkdir()
    (d / "legacy_fixture.yaml").write_text(textwrap.dedent(
        """\
        id: legacy_fixture
        display_name: Legacy (no license keys)
        filename: legacy.safetensors
        architecture: sdxl
        family: sdxl
        default_sampler: dpmpp_2m
        default_scheduler: karras
        default_steps: 28
        default_cfg: 6.5
        default_clip_skip: 2
        supports_ipadapter: false
        supports_lora: false
        active: true
        """
    ))
    loader = ModelRegistryLoader(models_dir=d, commercial_mode=True)
    entry = loader.get_model("legacy_fixture")
    assert entry.commercial_use is True
    assert entry.license is None
