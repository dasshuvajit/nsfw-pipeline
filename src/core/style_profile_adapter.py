"""Shared style-profile → workflow adapter.

``WorkflowBuilder`` duck-types against a flat bag of attributes
(``workflow_family``, ``model_checkpoint``, ``sampler``,
``supports_ipadapter`` …). That bag is assembled from two sources:

  1. The **style profile** row — pure aesthetic intent (LoRA stack,
     base keywords, palette / lighting hints). Render tuning is NOT
     here any more after the 2026-04 Phase A overhaul.
  2. The **resolved model** row — which checkpoint, which workflow
     family, capability flags, AND the sampler / scheduler / steps /
     cfg tuning measured for that checkpoint.

Render tuning always comes from the model YAML. The old two-tier
precedence (CLI → model, default → style profile) was retired once
profiles stopped carrying sampler/steps/cfg; buyers ask for
aesthetics, the model asks for tuning, the two never collide now.

``lora_stack``, ``base_style_keywords``, and ``base_negative_prompt``
come from the style profile. Archetypes may declare an empty
lora_stack — per-model LoRA defaults (if any) should be added via
the model YAML's ``prompt.extend`` hook, not re-introduced on the
profile.
"""

from __future__ import annotations

import json
from typing import Any

from src.memory.model_registry import ModelRegistryEntry


class StyleProfileForWorkflow:
    """Duck-typed view of (style profile row + resolved model).

    Render tuning (sampler, scheduler, steps, cfg) always resolves
    from the model YAML. Aesthetic fields (keywords, negatives,
    lora_stack) always resolve from the profile row.
    """

    def __init__(
        self,
        row: dict[str, Any],
        model: ModelRegistryEntry,
    ) -> None:
        self.model_id = model.id
        self.model_checkpoint = model.filename
        # Attribute kept as `workflow_family` because the WorkflowBuilder
        # duck-types against that name; the value is the `family` field
        # from the model YAML.
        self.workflow_family = model.family
        self.supports_ipadapter = model.supports_ipadapter
        self.supports_lora = model.supports_lora

        self.sampler = model.default_sampler
        self.scheduler = model.default_scheduler
        self.steps = model.default_steps
        self.cfg = model.default_cfg

        # LoRA stack resolution: profile wins, else fall back to the
        # model YAML's enabled entries (parsed into model.lora_stack by
        # the registry loader). Keeps the "render tuning comes from the
        # model YAML" precedence consistent with sampler/steps/cfg
        # above, while still letting a style profile override a model's
        # default stack (e.g. a no-LoRA smoke-test profile).
        profile_stack = row.get("lora_stack")
        if profile_stack:
            self.lora_stack = profile_stack
        elif model.lora_stack:
            self.lora_stack = json.dumps(list(model.lora_stack))
        else:
            self.lora_stack = None

    def __repr__(self) -> str:
        return (
            f"StyleProfileForWorkflow(model_id={self.model_id!r}, "
            f"checkpoint={self.model_checkpoint!r}, "
            f"family={self.workflow_family!r}, "
            f"sampler={self.sampler}/{self.scheduler}, "
            f"steps={self.steps}, cfg={self.cfg})"
        )
