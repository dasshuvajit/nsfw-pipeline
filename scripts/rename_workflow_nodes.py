#!/usr/bin/env python3
"""Rename numeric ComfyUI node IDs to semantic names via a YAML map.

When you export a workflow from ComfyUI's UI via "Save (API Format)" you
get a JSON blob like ``{"3": {...}, "6": {...}, "10": {...}}`` where
every node is keyed by an opaque integer ID. ``WorkflowBuilder`` in
this project expects semantic IDs (``ksampler``, ``positive_prompt``,
``lora_loader_0``) so it can inject per-render parameters by name. That
means every time you rebuild a template you have to hand-rename a
dozen keys AND every ``[node_id, slot]`` connection reference inside
``inputs``. Doing this by hand is error-prone — forget one and
WorkflowBuilder raises an opaque KeyError mid-render.

This tool does the rename mechanically:

  1. Load the raw API export (input JSON).
  2. Load the rename map (YAML): ``{"3": "ksampler", "6": "positive_prompt", ...}``.
  3. Validate: every numeric key in the input must have a mapping entry.
     Unmapped keys → exit 3 with the list of unmapped IDs. This is
     deliberately strict: a partial rename would leave dangling numeric
     refs that WorkflowBuilder can't resolve.
  4. Rewrite top-level keys AND every ``[node_id, slot]`` connection
     reference (dangling refs after rename → exit 4).
  5. Pretty-print to the output path with 2-space indent.

Example:
    python scripts/rename_workflow_nodes.py \\
        --input ~/Downloads/sdxl_base_api.json \\
        --map config/workflow_node_maps/sdxl_base.yaml \\
        --output config/comfyui_workflows/sdxl/base.json

Exit codes:
    0 — success
    1 — --input file missing or not JSON
    2 — --map file missing or not YAML
    3 — unmapped node IDs in the input (strict — see above)
    4 — dangling [node_id, slot] ref after rename
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"ERROR: --input not found: {path}")
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"ERROR: --input is not valid JSON: {exc}")
    if not isinstance(data, dict):
        raise SystemExit(
            f"ERROR: --input must be a top-level JSON object "
            f"(got {type(data).__name__}). ComfyUI API exports always "
            f"are — did you export via 'Save' instead of 'Save (API "
            f"Format)'?"
        )
    return data


def _load_map(path: Path) -> dict[str, str]:
    if not path.exists():
        raise SystemExit(f"ERROR: --map not found: {path}")
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise SystemExit(f"ERROR: --map is not valid YAML: {exc}")
    if not isinstance(data, dict):
        raise SystemExit(
            f"ERROR: --map must be a top-level YAML mapping "
            f"(got {type(data).__name__}). Expected format:\n"
            f"    '3': ksampler\n    '6': positive_prompt\n"
        )
    # YAML auto-coerces '3' → int 3 unless quoted. Normalize to strings
    # so the caller doesn't have to remember to quote every key.
    normalized: dict[str, str] = {}
    for raw_key, raw_val in data.items():
        key = str(raw_key)
        if not isinstance(raw_val, str):
            raise SystemExit(
                f"ERROR: --map value for key {key!r} must be a string, "
                f"got {type(raw_val).__name__}. Every node ID must map "
                f"to a semantic name like 'ksampler' or 'lora_loader_0'."
            )
        normalized[key] = raw_val
    return normalized


def _validate_mapping(
    workflow: dict[str, Any], node_map: dict[str, str]
) -> None:
    """Every top-level key in ``workflow`` must have a map entry.

    Warns on unused map entries (harmless — map is a superset) but
    aborts on unmapped workflow keys because those are the dangerous
    ones: they'd leave numeric IDs floating around in the output.
    """
    unmapped = [k for k in workflow if k not in node_map]
    if unmapped:
        raise SystemExit(
            f"EXIT 3: --input has {len(unmapped)} node(s) with no entry "
            f"in --map: {unmapped}. Add them to the YAML map or drop "
            f"them from the raw export."
        )
    unused = [k for k in node_map if k not in workflow]
    if unused:
        print(
            f"WARN: --map has {len(unused)} unused entries (not "
            f"present in --input): {unused}",
            file=sys.stderr,
        )


def _rewrite_references(
    workflow: dict[str, Any], node_map: dict[str, str]
) -> None:
    """Walk every ``inputs`` dict and rewrite ``[node_id, slot]`` refs.

    ComfyUI encodes a connection as a two-element list
    ``[source_node_id, output_slot]`` inside the ``inputs`` of the
    receiving node. After we rename top-level keys, these refs still
    point at the old numeric IDs — this function rewrites them in
    place. Raises SystemExit (exit 4) if a ref points at a node_id
    that isn't in the rename map (which means the input was
    inconsistent to begin with: a node referencing a non-existent
    source).
    """
    for node_key, node in workflow.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for input_name, input_val in inputs.items():
            # ComfyUI connection refs are exactly a 2-list
            # [str_or_int_node_id, int_slot]. Anything else is a
            # literal parameter value, leave it alone.
            if not (
                isinstance(input_val, list)
                and len(input_val) == 2
                and isinstance(input_val[1], int)
            ):
                continue
            src = str(input_val[0])
            if src not in node_map:
                raise SystemExit(
                    f"EXIT 4: node {node_key!r} input {input_name!r} "
                    f"points at source node_id={src!r} which is not "
                    f"in --map. This is a dangling ref — either the "
                    f"input JSON is inconsistent, or the map is "
                    f"missing that ID."
                )
            inputs[input_name] = [node_map[src], input_val[1]]


def _rewrite_top_level(
    workflow: dict[str, Any], node_map: dict[str, str]
) -> dict[str, Any]:
    """Return a new dict with top-level keys replaced by semantic names.

    Uses a fresh dict (rather than in-place mutation) to preserve
    original key order where the mapping is identity, and to make the
    transformation easy to reason about. Duplicate target names (two
    source IDs mapping to the same semantic name) are caught here with
    a clear error — they'd otherwise silently clobber each other.
    """
    renamed: dict[str, Any] = {}
    for old_key, node in workflow.items():
        new_key = node_map[old_key]
        if new_key in renamed:
            raise SystemExit(
                f"ERROR: --map produces duplicate semantic name "
                f"{new_key!r} (at least two numeric IDs map to it). "
                f"Every target name must be unique."
            )
        renamed[new_key] = node
    return renamed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Raw API JSON exported from ComfyUI ('Save (API Format)').",
    )
    parser.add_argument(
        "--map", dest="map_path", required=True, type=Path,
        help="YAML rename map: {'numeric_id': 'semantic_name'}.",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Where to write the renamed workflow JSON.",
    )
    args = parser.parse_args()

    try:
        workflow = _load_json(args.input)
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        node_map = _load_map(args.map_path)
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        _validate_mapping(workflow, node_map)
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 3

    try:
        _rewrite_references(workflow, node_map)
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 4

    try:
        renamed = _rewrite_top_level(workflow, node_map)
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 4

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(renamed, f, indent=2)
        f.write("\n")

    print(
        f"OK — renamed {len(renamed)} node(s), wrote {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
