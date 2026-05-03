"""Regression: GonzaLomo Chroma v3.0 YAML loads clean and inherits
chroma family defaults.

This is a YAML-typo-catcher for ``config/models/gonzalomo_chroma_v30.yaml``.
The registry + family-merge pipeline is already covered by
``test_model_registry_commercial_gate.py`` and
``test_model_registry_lora_stack.py``; these two assertions just make
sure this specific model stays registered with the researched settings
(Civitai #2182526, lodestones/Chroma1-HD) and that the empty
``prompt.extend`` block leaves the chroma family's flux_natural
composer defaults untouched.
"""

from __future__ import annotations

import pytest

from src.memory.model_registry import ModelRegistryLoader


def test_gonzalomo_chroma_v30_registers():
    loader = ModelRegistryLoader()
    entry = loader.get_model("gonzalomo_chroma_v30")
    assert entry.family == "chroma"
    assert entry.filename == "gonzalomoChroma_v30.gguf"
    assert entry.default_cfg == pytest.approx(1.15)
    assert entry.default_steps == 12
    assert entry.default_sampler == "euler"
    assert entry.default_scheduler == "beta"
    assert entry.commercial_use is True


def test_gonzalomo_chroma_v30_prompt_guide_inherits_chroma_family():
    guide = ModelRegistryLoader().get_prompt_guide("gonzalomo_chroma_v30")
    assert guide is not None
    assert guide.prompt_style == "flux_natural"
    assert guide.supports_negative_prompt is True
    # Chroma is Flux-lineage prose; A1111-style (word:1.2) weighting is off.
    assert guide.supports_weighting is False
    assert guide.trigger_words == []
