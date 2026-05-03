#!/usr/bin/env python3
"""Print the model registry as a formatted table.

Thin read-only frontend over
``src.memory.model_registry.ModelRegistryLoader`` (YAML-backed since
the 2026-04 refactor).

Useful for:
  - Seeing which ids exist before passing ``--model X`` to render_set.py
  - Confirming a just-added model YAML is loadable
  - Filtering by family (``--family flux`` etc.)

Default output shows only rows with ``active: true``. Pass ``--all`` to
include inactive rows too.

Example:
    python scripts/list_models.py
    python scripts/list_models.py --family sdxl
    python scripts/list_models.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.memory.model_registry import (  # noqa: E402
    ModelRegistryError,
    ModelRegistryLoader,
    ModelPromptGuide,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        default=None,
        help="Filter by family (sdxl|pony|illustrious|flux|chroma|flux2).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include inactive models (active: false).",
    )
    args = parser.parse_args()

    try:
        loader = ModelRegistryLoader()
    except ModelRegistryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    models = loader.list_models(
        family=args.family, include_inactive=args.all
    )
    if not models:
        where = []
        if args.family:
            where.append(f"family={args.family!r}")
        if not args.all:
            where.append("active=1")
        suffix = f" ({', '.join(where)})" if where else ""
        print(f"No models in registry{suffix}.")
        return 0

    # Build a model_id → prompt_guide lookup for the extra columns.
    guides: dict[str, ModelPromptGuide | None] = {}
    for m in models:
        guides[m.id] = loader.get_prompt_guide(m.id)

    headers = (
        "id", "family", "sampler", "sched", "steps", "cfg",
        "neg", "prompt_style", "ipa", "lora", "active", "notes",
    )
    rows: list[tuple[str, ...]] = []
    for m in models:
        g = guides.get(m.id)
        rows.append((
            m.id,
            m.family,
            m.default_sampler,
            m.default_scheduler,
            str(m.default_steps),
            f"{m.default_cfg:.1f}",
            "Y" if (g and g.supports_negative_prompt) else "N",
            g.prompt_style if g else "sdxl_keywords",
            "Y" if m.supports_ipadapter else "N",
            "Y" if m.supports_lora else "N",
            "Y" if m.active else "N",
            (m.notes or "")[:50],
        ))

    widths = [
        max(len(h), *(len(r[i]) for r in rows))
        for i, h in enumerate(headers)
    ]

    def fmt(values: tuple[str, ...]) -> str:
        return "  ".join(v.ljust(w) for v, w in zip(values, widths))

    print(fmt(headers))
    print(fmt(tuple("─" * w for w in widths)))
    for r in rows:
        print(fmt(r))

    print(f"\n{len(rows)} model(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
