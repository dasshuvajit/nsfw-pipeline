"""Tests for ``src.core.engine._extract_seed_from_workflow``.

Pre-2026-05-18 the engine read seed from the ComfyUI history response
(``RenderedImage``) — which has no ``seed`` attribute — so PNG metadata
always recorded ``seed=0``. The helper added on 2026-05-18 reads the
authoritative seed straight out of the built workflow dict instead.

Coverage:
  * ksampler-shaped workflows (SDXL / Pony / Illustrious / Flux /
    Flux.2 / every external template per ``_REQUIRED_NODES_EXTERNAL``).
  * random_noise-shaped workflows (Chroma's built-in graph).
  * Missing both nodes → 0 (defensive fallback).
  * Non-int seed values (defensive — should not raise).
"""

from __future__ import annotations

from src.core.engine import _extract_seed_from_workflow


def test_ksampler_workflow_returns_inputs_seed():
    workflow = {
        "ksampler": {"inputs": {"seed": 4242, "steps": 30}},
        "positive_prompt": {"inputs": {"text": "x"}},
    }
    assert _extract_seed_from_workflow(workflow) == 4242


# random_noise / SamplerCustomAdvanced seed extraction deleted with
# Chroma's built-in base.json (2026-05-20). External templates always
# carry the seed on `ksampler.inputs.seed` (the 4-node contract enforces it).


def test_neither_node_returns_zero():
    """Hand-crafted template with no recognised seed node — fall back
    to 0 rather than raise. Callers can treat 0 as 'unrecoverable'."""
    workflow = {"some_node": {"inputs": {}}}
    assert _extract_seed_from_workflow(workflow) == 0


def test_seed_field_missing_falls_back():
    """ksampler present but missing inputs.seed → 0."""
    workflow = {"ksampler": {"inputs": {"steps": 30}}}  # no seed key
    assert _extract_seed_from_workflow(workflow) == 0


def test_non_int_seed_returns_zero():
    """If ksampler.seed is a non-int (defensive — should never happen
    in practice), fall back to 0 rather than propagate a garbage value."""
    workflow = {
        "ksampler": {"inputs": {"seed": "not-a-number"}},
    }
    assert _extract_seed_from_workflow(workflow) == 0


def test_inputs_missing_entirely_returns_zero():
    """``ksampler`` node with no ``inputs`` dict at all (malformed
    template) → 0, no AttributeError."""
    workflow = {"ksampler": {}}
    assert _extract_seed_from_workflow(workflow) == 0


def test_refiner_workflow_uses_base_ksampler_not_refiner_ksampler():
    """The refiner contract patches ``refiner_ksampler.seed`` to match
    the base ``ksampler.seed`` (deterministic refiner pass). The PNG
    metadata records the BASE seed — that's the seed identifying the
    render. Verifies the helper reads ksampler, not refiner_ksampler."""
    workflow = {
        "ksampler": {"inputs": {"seed": 555}},
        "refiner_ksampler": {"inputs": {"seed": 555}},
        "refiner_positive_prompt": {"inputs": {"text": "x"}},
    }
    assert _extract_seed_from_workflow(workflow) == 555
