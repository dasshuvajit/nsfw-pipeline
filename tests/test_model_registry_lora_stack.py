"""ModelRegistryEntry parses the model YAML ``lora_stack:`` block.

Only entries with ``enabled: true`` land on the typed dataclass, and
the list is capped at two at load time (CLAUDE.md invariant — max 2
LoRAs per render).

Phase 2 of the prompt-quality plan generalises the schema across
LoRA-supporting families (sdxl, pony, illustrious, flux, flux2).
Chroma deliberately has no LoRA wiring (the workflow uses
SamplerCustomAdvanced, not the lora_loader_0/1 SDXL graph), so its
models declare ``supports_lora: false`` and skip the field entirely.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.memory.model_registry import (
    ModelRegistryError,
    ModelRegistryLoader,
)


_BASE_YAML = """\
id: {mid}
display_name: {mid} fixture
filename: {mid}.safetensors
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


# Per-family fixture skeletons — same lora_stack schema, different
# sampler / scheduler / family / architecture defaults so the
# registry validation passes without faking irrelevant fields.
_FAMILY_FIXTURES: dict[str, str] = {
    "sdxl": """\
id: {mid}
display_name: {mid} fixture
filename: {mid}.safetensors
architecture: sdxl
family: sdxl
default_sampler: dpmpp_2m
default_scheduler: karras
default_steps: 30
default_cfg: 6.0
default_clip_skip: null
supports_ipadapter: true
supports_lora: true
vae_filename: null
text_encoder: null
resolution_portrait: [832, 1216]
resolution_square: [1024, 1024]
resolution_landscape: [1216, 832]
active: true
""",
    "pony": """\
id: {mid}
display_name: {mid} fixture
filename: {mid}.safetensors
architecture: pony
family: pony
default_sampler: dpmpp_sde
default_scheduler: karras
default_steps: 30
default_cfg: 5.0
default_clip_skip: 2
supports_ipadapter: true
supports_lora: true
vae_filename: null
text_encoder: null
resolution_portrait: [896, 1152]
resolution_square: [1024, 1024]
resolution_landscape: [1152, 896]
active: true
""",
    "illustrious": """\
id: {mid}
display_name: {mid} fixture
filename: {mid}.safetensors
architecture: illustrious
family: illustrious
default_sampler: euler_a
default_scheduler: normal
default_steps: 28
default_cfg: 6.0
default_clip_skip: 2
supports_ipadapter: false
supports_lora: true
vae_filename: null
text_encoder: null
resolution_portrait: [896, 1152]
resolution_square: [1024, 1024]
resolution_landscape: [1152, 896]
active: true
""",
    "flux": """\
id: {mid}
display_name: {mid} fixture
filename: {mid}.safetensors
architecture: flux
family: flux
default_sampler: euler
default_scheduler: simple
default_steps: 20
default_cfg: 1.0
default_clip_skip: null
supports_ipadapter: false
supports_lora: true
vae_filename: ae.safetensors
text_encoder: clip_l.safetensors
resolution_portrait: [832, 1216]
resolution_square: [1024, 1024]
resolution_landscape: [1216, 832]
active: true
""",
}


def _write_yaml(d: Path, mid: str, extra: str) -> None:
    (d / f"{mid}.yaml").write_text(
        _BASE_YAML.format(mid=mid) + textwrap.dedent(extra)
    )


def _write_family_yaml(d: Path, family: str, mid: str, extra: str) -> None:
    """Write a fixture for a specific family (sdxl/pony/illustrious/flux)."""
    skeleton = _FAMILY_FIXTURES[family]
    (d / f"{mid}.yaml").write_text(
        skeleton.format(mid=mid) + textwrap.dedent(extra)
    )


@pytest.fixture
def models_dir(tmp_path: Path) -> Path:
    d = tmp_path / "models"
    d.mkdir()
    return d


def test_lora_stack_filters_enabled(models_dir):
    _write_yaml(models_dir, "flux2_mixed", """
        lora_stack:
          - name: ultra_real_v4
            filename: ultra_real_v4.safetensors
            strength: 0.70
            enabled: true
          - name: klein_slider_anatomy
            filename: klein_slider_anatomy.safetensors
            strength: 2.0
            enabled: true
          - name: portrait_engine_disabled
            filename: V2_flux_klein_4.safetensors
            strength: 1.0
            enabled: false
    """)
    loader = ModelRegistryLoader(models_dir=models_dir)
    entry = loader.get_model("flux2_mixed")
    assert entry.lora_stack == (
        {"name": "ultra_real_v4", "strength": 0.70},
        {"name": "klein_slider_anatomy", "strength": 2.0},
    )


def test_lora_stack_cap_enforced_at_load(models_dir):
    _write_yaml(models_dir, "flux2_overstack", """
        lora_stack:
          - name: a
            filename: a.safetensors
            strength: 0.5
            enabled: true
          - name: b
            filename: b.safetensors
            strength: 0.5
            enabled: true
          - name: c
            filename: c.safetensors
            strength: 0.5
            enabled: true
    """)
    with pytest.raises(ModelRegistryError, match="max 2 per render"):
        ModelRegistryLoader(models_dir=models_dir)


def test_legacy_yaml_without_lora_stack_defaults_to_empty(models_dir):
    _write_yaml(models_dir, "flux2_bare", "")
    loader = ModelRegistryLoader(models_dir=models_dir)
    assert loader.get_model("flux2_bare").lora_stack == ()


# ----- Phase 2: lora_stack works on every LoRA-supporting family ----------


@pytest.mark.parametrize("family", ["sdxl", "pony", "illustrious", "flux"])
def test_lora_stack_parses_for_every_lora_supporting_family(family, models_dir):
    """The same ``lora_stack:`` schema works on sdxl/pony/illustrious/flux.
    chroma intentionally has no LoRA wiring (supports_lora: false on its
    YAMLs); flux2 currently has no registered models but the wiring is
    family-agnostic — the SDXL/pony/illustrious/flux cases prove it."""
    _write_family_yaml(models_dir, family, f"{family}_with_loras", """
        lora_stack:
          - name: realism_skin
            filename: realism_skin_v3.safetensors
            strength: 0.65
            enabled: true
          - name: detail_boost
            filename: detail_boost_v2.safetensors
            strength: 0.85
            enabled: true
          - name: experimental_dropout
            filename: experimental.safetensors
            strength: 1.0
            enabled: false
    """)
    loader = ModelRegistryLoader(models_dir=models_dir)
    entry = loader.get_model(f"{family}_with_loras")
    assert entry.family == family
    # Disabled entry filtered; both enabled entries land in declaration order.
    assert entry.lora_stack == (
        {"name": "realism_skin", "strength": 0.65},
        {"name": "detail_boost", "strength": 0.85},
    )


@pytest.mark.parametrize("family", ["sdxl", "pony", "illustrious", "flux"])
def test_lora_stack_cap_2_enforced_per_family(family, models_dir):
    """Cap-2 invariant fires regardless of family."""
    _write_family_yaml(models_dir, family, f"{family}_overstack", """
        lora_stack:
          - name: a
            filename: a.safetensors
            strength: 0.5
            enabled: true
          - name: b
            filename: b.safetensors
            strength: 0.5
            enabled: true
          - name: c
            filename: c.safetensors
            strength: 0.5
            enabled: true
    """)
    with pytest.raises(ModelRegistryError, match="max 2 per render"):
        ModelRegistryLoader(models_dir=models_dir)


@pytest.mark.parametrize("family", ["sdxl", "pony", "illustrious", "flux"])
def test_no_lora_stack_yields_empty_tuple_per_family(family, models_dir):
    """A model YAML without ``lora_stack:`` still loads cleanly."""
    _write_family_yaml(models_dir, family, f"{family}_bare", "")
    loader = ModelRegistryLoader(models_dir=models_dir)
    entry = loader.get_model(f"{family}_bare")
    assert entry.lora_stack == ()


def test_chroma_supports_lora_false_on_real_registry():
    """Both on-disk Chroma YAMLs deliberately have supports_lora: false
    because the chroma workflow graph (UnetLoaderGGUF +
    SamplerCustomAdvanced) does not template lora_loader_0/1. Phase 2's
    LoRA-stack generalisation explicitly excludes Chroma."""
    loader = ModelRegistryLoader()
    chroma_v10 = loader.get_model("chroma_v10HD")
    assert chroma_v10.supports_lora is False
    assert chroma_v10.lora_stack == ()
    gonzalomo_chroma = loader.get_model("gonzalomo_chroma_v30")
    assert gonzalomo_chroma.supports_lora is False
    assert gonzalomo_chroma.lora_stack == ()
