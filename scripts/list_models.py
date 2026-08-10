#!/usr/bin/env python3
"""Print the LLM registry (config/llm_models.yaml).

Thin read-only frontend over ``LLMRegistryLoader``.

(The image-model registry half was archived 2026-06-10 with the legacy
structured path; the per-role routing table (--routing) was archived
2026-08 with src/agents/llm_router.py — see legacy/.)

Example:
    python scripts/list_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.memory.llm_registry import (  # noqa: E402
    LLMRegistryError,
    LLMRegistryLoader,
)


def _print_llms(loader: LLMRegistryLoader) -> int:
    """Print the LLM registry block. Returns exit code."""
    entries = loader.list_llms(include_inactive=True)
    if not entries:
        print("No LLMs in registry (config/llm_models.yaml).")
        return 0

    headers = ("id", "backend", "model_tag", "quant", "size_gb", "active", "default")
    rows: list[tuple[str, ...]] = []
    for entry in entries:
        rows.append((
            entry.id,
            entry.backend,
            entry.model_tag,
            entry.quant or "-",
            f"{entry.size_gb:.0f}" if entry.size_gb is not None else "-",
            "Y" if entry.active else "N",
            "Y" if entry.id == loader.default_llm_id else "",
        ))

    widths = [
        max(len(h), *(len(r[i]) for r in rows))
        for i, h in enumerate(headers)
    ]

    def fmt(values: tuple[str, ...]) -> str:
        return "  ".join(v.ljust(w) for v, w in zip(values, widths))

    print("LLM registry (config/llm_models.yaml):")
    print(fmt(headers))
    print(fmt(tuple("─" * w for w in widths)))
    for r in rows:
        print(fmt(r))

    print(f"\n{len(rows)} LLM(s); default = {loader.default_llm_id!r}")
    return 0


def main() -> int:
    try:
        llm_loader = LLMRegistryLoader()
    except LLMRegistryError as exc:
        print(f"ERROR: LLM registry — {exc}", file=sys.stderr)
        return 1
    return _print_llms(llm_loader)


if __name__ == "__main__":
    sys.exit(main())
