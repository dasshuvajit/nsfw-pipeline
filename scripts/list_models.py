#!/usr/bin/env python3
"""Print the LLM registry + optional per-role routing resolution.

Thin read-only frontend over:
  - ``LLMRegistryLoader`` for ``config/llm_models.yaml``
  - ``LLMRouter`` (constructed against ``pipeline.yaml::llm.routing``)
    for the per-role resolution table

(The image-model registry half was archived 2026-06-10 with the legacy
structured path — ``config/models/*.yaml`` now lives under ``legacy/config/``;
the active render path hardcodes its checkpoint in the workflow templates.)

Flags:
  --routing           print the per-role LLM resolution table + reverse mapping
  --llms-only         retained no-op (back-compat; the LLM registry is all
                      there is now)

Examples:
    python scripts/list_models.py
    python scripts/list_models.py --routing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.llm_router import LLMRouter  # noqa: E402
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


def _load_routing_config() -> dict[str, Any]:
    """Read ``pipeline.yaml::llm.routing`` (or {} if absent)."""
    pipeline_yaml = PROJECT_ROOT / "config" / "pipeline.yaml"
    if not pipeline_yaml.exists():
        return {}
    with open(pipeline_yaml) as f:
        cfg = yaml.safe_load(f) or {}
    llm_cfg = cfg.get("llm") or {}
    routing = llm_cfg.get("routing")
    return routing if isinstance(routing, dict) else {}


def _print_routing(loader: LLMRegistryLoader) -> int:
    """Print the per-role resolution table + reverse mapping."""
    try:
        routing = _load_routing_config()
        router = LLMRouter(loader, routing_config=routing)
    except LLMRegistryError as exc:
        print(f"ERROR: routing config invalid — {exc}", file=sys.stderr)
        return 1

    rows = router._build_resolution_rows(cli_llm_override=None)

    # Forward table: role → llm_id with Source column.
    role_w = max(len("Role"), *(len(r[0]) for r in rows))
    llm_w = max(len("LLM resolved"), *(len(r[1]) for r in rows))
    src_w = max(len("Source"), *(len(r[2]) for r in rows))

    print("Resolved LLMs by role (no --llm override):")
    print(
        f"{'Role'.ljust(role_w)}  {'LLM resolved'.ljust(llm_w)}  "
        f"{'Source'.ljust(src_w)}"
    )
    print(
        f"{'─' * role_w}  {'─' * llm_w}  {'─' * src_w}"
    )
    for role, llm_id, source in rows:
        print(
            f"{role.ljust(role_w)}  {llm_id.ljust(llm_w)}  "
            f"{source.ljust(src_w)}"
        )

    # Reverse mapping: llm_id → roles served.
    reverse: dict[str, list[str]] = {}
    for role, llm_id, _src in rows:
        reverse.setdefault(llm_id, []).append(role)

    print("\nReverse mapping (LLM → roles served):")
    for llm_id in sorted(reverse):
        try:
            entry = loader.get_llm(llm_id, require_active=False)
            tags = ["active"] if entry.active else ["inactive"]
            if entry.id == loader.default_llm_id:
                tags.append("default")
            tag_str = ", ".join(tags)
        except LLMRegistryError:
            tag_str = "unknown"
        print(f"  {llm_id} ({tag_str}):")
        for role in reverse[llm_id]:
            print(f"    - {role}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--llms-only",
        action="store_true",
        help="Retained no-op (the LLM registry is the only registry now).",
    )
    parser.add_argument(
        "--routing",
        action="store_true",
        help="Print per-role LLM resolution table + reverse mapping.",
    )
    args = parser.parse_args()

    try:
        llm_loader = LLMRegistryLoader()
    except LLMRegistryError as exc:
        print(f"ERROR: LLM registry — {exc}", file=sys.stderr)
        return 1
    rc = _print_llms(llm_loader)

    if args.routing:
        print()  # spacer
        rc = _print_routing(llm_loader) or rc

    return rc


if __name__ == "__main__":
    sys.exit(main())
