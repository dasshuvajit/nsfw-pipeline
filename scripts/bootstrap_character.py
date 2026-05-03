#!/usr/bin/env python3
"""Bootstrap a character from a hand-written identity JSON file.

Per ARCHITECTURE.md Section 12, this is the manual character creation
ritual that runs 3-5 times in Phase 1. **No LLM yet** — you write the
identity by hand so you understand what makes a good character before
trying to automate it.

What this does:
  1. Loads `identity.json` from disk and validates the required fields.
  2. Builds the character's `base_prompt` by joining the identity fields
     in a fixed, reproducible order.
  3. INSERTs a new row into the `characters` table with sensible defaults
     for the JSON columns (locked_features={}, allowed_shift_axes=[]).
  4. Reads the row back via `CharacterManager` and pretty-prints it so
     you can verify the prompt looks right.

The character ID defaults to `identity['name']`, falling back to the
parent directory name if `name` isn't set. Override with `--character-id`.

`base_prompt` is IMMUTABLE after creation (DB trigger). To re-bootstrap
the same character with a new prompt, pass `--force` — this DELETEs the
existing row and re-inserts. (Trigger only fires on UPDATE, not DELETE.)

Example:
    python scripts/bootstrap_character.py characters/char_001/identity.json
    python scripts/bootstrap_character.py characters/char_001/identity.json \\
        --style-profile boudoir_noir --model-id juggernaut_ragnarok
    python scripts/bootstrap_character.py characters/char_001/identity.json \\
        --force --db-path /tmp/test.db
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import yaml

# Make src/ importable when running this file as a plain script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.memory.character_manager import (  # noqa: E402
    CharacterManager,
    CharacterNotFound,
)
from src.memory.model_registry import (  # noqa: E402
    ModelNotFound,
    ModelRegistryLoader,
)
from src.memory.style_profile_loader import (  # noqa: E402
    StyleProfileLoader,
    StyleProfileNotFound,
)


CONFIG_PATH = PROJECT_ROOT / "config" / "pipeline.yaml"

# Required identity fields per the user's spec for this script — these
# must be present and non-empty in identity.json.
REQUIRED_FIELDS = ("gender", "age_appearance", "face", "hair", "body_type")

# Optional fields that flow into base_prompt if present.
OPTIONAL_PROMPT_FIELDS = ("distinguishing_features", "vibe")


# ----- DB path resolution ---------------------------------------------------


def _load_db_path() -> Path:
    """Read pipeline.db_path from config/pipeline.yaml.

    Resolved relative to PROJECT_ROOT so the script works regardless of
    cwd. Falls back to nsfw_pipeline.db at the project root if the YAML
    or the field is missing — same default as `scripts/init_db.py`.
    """
    default = PROJECT_ROOT / "nsfw_pipeline.db"
    if not CONFIG_PATH.exists():
        return default
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f) or {}
    raw = (cfg.get("pipeline") or {}).get("db_path")
    if not raw:
        return default
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate


# ----- identity validation + prompt building --------------------------------


def validate_identity(identity: dict[str, Any]) -> list[str]:
    """Return a list of validation problems. Empty list = OK."""
    problems: list[str] = []
    if not isinstance(identity, dict):
        return [f"identity must be a JSON object, got {type(identity).__name__}"]

    for field in REQUIRED_FIELDS:
        value = identity.get(field)
        if value is None or value == "":
            problems.append(f"missing required field: {field!r}")
            continue
        if not isinstance(value, str):
            problems.append(
                f"field {field!r} must be a string, got {type(value).__name__}"
            )
    return problems


def build_base_prompt(identity: dict[str, Any]) -> str:
    """Compose base_prompt from identity fields in a fixed order.

    Order matters because base_prompt is immutable — every render of this
    character will reuse this exact string, so a stable ordering keeps
    seeds reproducible across pipeline runs.
    """
    parts: list[str] = []
    for field in REQUIRED_FIELDS:
        parts.append(identity[field].strip())
    for field in OPTIONAL_PROMPT_FIELDS:
        value = identity.get(field)
        if value:
            parts.append(str(value).strip())
    return ", ".join(parts)


# ----- DB write -------------------------------------------------------------


def _row_exists(conn: sqlite3.Connection, character_id: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM characters WHERE id = ? LIMIT 1", (character_id,)
        ).fetchone()
        is not None
    )


def insert_character(
    db_path: Path,
    character_id: str,
    identity: dict[str, Any],
    base_prompt: str,
    style_profile_id: str,
    model_id: str | None,
    *,
    force: bool,
) -> None:
    """INSERT a character row, optionally replacing an existing one.

    ``style_profile_id`` is a YAML id validated against
    ``config/style_profiles.yaml`` before the INSERT runs.
    ``model_id`` is optional — NULL falls back to
    ``pipeline.default_model_id`` at run time.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA foreign_keys = ON")

        if _row_exists(conn, character_id):
            if not force:
                raise SystemExit(
                    f"Character {character_id!r} already exists. "
                    f"base_prompt is immutable — pass --force to delete "
                    f"and re-insert (this destroys total_images_generated "
                    f"and last_used_at, so use it only during bootstrap)."
                )
            conn.execute("DELETE FROM characters WHERE id = ?", (character_id,))

        conn.execute(
            """
            INSERT INTO characters (
                id, name, style_profile_id, model_id, base_prompt,
                negative_prompt, reference_image_path,
                locked_features, allowed_shift_axes, outfit_pool,
                version, active
            )
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                character_id,
                identity.get("name", character_id),
                style_profile_id,
                model_id,
                base_prompt,
                identity.get("negative_prompt"),
                identity.get("reference_image_path"),
                json.dumps(identity.get("locked_features") or {}),
                json.dumps(identity.get("allowed_shift_axes") or []),
                _json_or_none(identity.get("outfit_pool")),
                int(identity.get("version", 1)),
                1,  # active
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _json_or_none(value: Any) -> str | None:
    """Serialize a Python value to JSON, or None if absent."""
    if value is None:
        return None
    return json.dumps(value)


# ----- pretty printer -------------------------------------------------------


def print_character(record: dict[str, Any]) -> None:
    """Pretty-print a character row in fixed-width key/value form."""
    label_width = 24
    print("\nCharacter record:")
    for key in (
        "id",
        "name",
        "style_profile_id",
        "model_id",
        "base_prompt",
        "negative_prompt",
        "reference_image_path",
        "locked_features",
        "allowed_shift_axes",
        "outfit_pool",
        "version",
        "active",
        "total_images_generated",
        "last_used_at",
        "created_at",
    ):
        value = record.get(key)
        if value is None or value == "":
            display: str = "(none)"
        elif key == "active":
            display = "True" if value else "False"
        elif key == "last_used_at" and not value:
            display = "(never)"
        else:
            display = str(value)
        print(f"  {key:<{label_width}} {display}")


# ----- main -----------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "identity_path",
        help="Path to the identity JSON file (e.g. characters/char_001/identity.json).",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Override DB path. Default: read from config/pipeline.yaml.",
    )
    parser.add_argument(
        "--style-profile",
        default="golden_hour_natural",
        help="style_profile id from config/style_profiles.yaml to "
        "attach this character to. Default: %(default)s.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="model id from config/models/*.yaml to attach this "
        "character to. Default: NULL (falls back to "
        "pipeline.default_model_id at run time).",
    )
    parser.add_argument(
        "--character-id",
        default=None,
        help="Override the character primary key. "
        "Default: identity['name'] or the identity file's parent directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="DELETE any existing row with this id before inserting. "
        "Required because base_prompt is immutable on UPDATE.",
    )
    args = parser.parse_args()

    identity_path = Path(args.identity_path).expanduser()
    if not identity_path.exists():
        print(f"ERROR: identity file not found: {identity_path}", file=sys.stderr)
        return 1

    try:
        identity = json.loads(identity_path.read_text())
    except json.JSONDecodeError as exc:
        print(f"ERROR: {identity_path} is not valid JSON: {exc}", file=sys.stderr)
        return 2

    problems = validate_identity(identity)
    if problems:
        print(
            f"ERROR: {identity_path} failed validation:", file=sys.stderr
        )
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print(
            f"\nRequired fields: {', '.join(REQUIRED_FIELDS)}",
            file=sys.stderr,
        )
        return 3

    character_id = (
        args.character_id
        or identity.get("name")
        or identity_path.parent.name
    )
    base_prompt = build_base_prompt(identity)

    db_path = Path(args.db_path).expanduser() if args.db_path else _load_db_path()
    if not db_path.exists():
        print(
            f"ERROR: database not found at {db_path}.\n"
            f"Run `python scripts/init_db.py` first.",
            file=sys.stderr,
        )
        return 4

    # Validate style_profile_id against config/style_profiles.yaml before
    # touching the DB. There is no SQL FK since static config lives in
    # YAML now — the loader is the authority.
    try:
        StyleProfileLoader().get_profile(args.style_profile)
    except StyleProfileNotFound:
        known = ", ".join(p.id for p in StyleProfileLoader().list_profiles())
        print(
            f"ERROR: --style-profile {args.style_profile!r} not in "
            f"config/style_profiles.yaml.\n"
            f"Known profiles: {known}",
            file=sys.stderr,
        )
        return 5

    if args.model_id:
        try:
            ModelRegistryLoader(db_path).get_model(args.model_id)
        except ModelNotFound as exc:
            print(
                f"ERROR: --model-id {args.model_id!r} not in "
                f"config/models/: {exc}",
                file=sys.stderr,
            )
            return 5

    print(f"Bootstrapping character {character_id!r}")
    print(f"  identity:      {identity_path}")
    print(f"  db:            {db_path}")
    print(f"  style profile: {args.style_profile}")
    print(f"  model id:      {args.model_id or '(pipeline default)'}")
    print(f"  base_prompt:   {base_prompt}")

    try:
        insert_character(
            db_path=db_path,
            character_id=character_id,
            identity=identity,
            base_prompt=base_prompt,
            style_profile_id=args.style_profile,
            model_id=args.model_id,
            force=args.force,
        )
    except sqlite3.IntegrityError as exc:
        print(f"\nERROR: DB integrity violation: {exc}", file=sys.stderr)
        return 5
    except SystemExit as exc:
        # Reraise our own user-facing duplicate-row exit message.
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 6

    # If identity.json provided a reference_image_path, the INSERT above
    # already persisted it verbatim. Nothing else to do on disk —
    # reference-image selection is now a manual workflow: generate candidates
    # with `scripts/render_set.py --ratio square ...`, pick one visually, and
    # run `scripts/map_reference.py` to re-point the DB at the chosen file.
    ref_stored = identity.get("reference_image_path")
    if ref_stored:
        ref_path = Path(ref_stored).expanduser()
        if not ref_path.is_absolute():
            ref_path = (PROJECT_ROOT / ref_path).resolve()
        if not ref_path.exists():
            print(
                f"\nNOTE: identity.reference_image_path does not exist yet "
                f"on disk: {ref_path}. The character row is created; set a "
                f"reference image with:\n"
                f"  python scripts/render_set.py --character {character_id} "
                f"--level T2_implied --count 15 --ratio square\n"
                f"  # pick one, copy it to characters/{character_id}/, then:\n"
                f"  python scripts/map_reference.py --character-id "
                f"{character_id} --image <chosen_path>",
                file=sys.stderr,
            )

    # Read it back via CharacterManager and pretty-print so the user can
    # verify the prompt looks right.
    cm = CharacterManager(db_path)
    try:
        record = cm.get_character(character_id)
    except CharacterNotFound as exc:
        print(f"\nERROR: insert succeeded but read-back failed: {exc}", file=sys.stderr)
        return 7

    print_character(record)
    print("\nOK — character bootstrapped.")
    if not ref_stored:
        print(
            "\nTo set a reference image for IPAdapter renders:\n"
            f"  1. python scripts/render_set.py --character {character_id} "
            f"--level T2_implied --count 15 --ratio square\n"
            f"  2. pick the best render visually, copy it to "
            f"characters/{character_id}/\n"
            f"  3. python scripts/map_reference.py --character-id "
            f"{character_id} --image <chosen_path>",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
