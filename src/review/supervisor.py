"""Supervised pause points — human approval gates.

In ``supervised`` execution mode, the engine pauses at two points:

  1. **After planning** (``approve_plan``): prints the series plan,
     sample scenes, sample prompts, and asks y/n.
  2. **After rendering** (``review_images``): generates a contact sheet,
     prints quality stats, and asks y/n or lets the human reject
     specific images.

See ARCHITECTURE.md Section 18 (Execution Modes).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from src.review.contact_sheet import create_contact_sheet

logger = logging.getLogger(__name__)


class SupervisorAbort(Exception):
    """Human rejected the plan or images."""


class Supervisor:
    """Interactive supervisor for the pipeline's pause points.

    When ``auto_approve=True``, both gates pass without prompting.
    This is used by ``dry_run.py`` (which only runs Phase A) and by
    ``automated`` execution mode (Phase 3+).
    """

    def __init__(self, *, auto_approve: bool = False) -> None:
        self.auto_approve = auto_approve

    def approve_plan(
        self,
        *,
        mode: str,
        content_level: str,
        character_id: str | None,
        model_id: str,
        series_plan: dict[str, Any],
        scenes: list[dict[str, Any]],
        prompts: list[dict[str, Any]],
    ) -> bool:
        """Pause point 1: human reviews the LLM plan.

        Prints a summary and asks y/n. Returns True if approved.
        """
        print("\n" + "=" * 60)
        print("  SUPERVISED PAUSE POINT 1: Plan Review")
        print("=" * 60)
        print(f"  Mode:           {mode}")
        print(f"  Content level:  {content_level}")
        print(f"  Character:      {character_id or '(none)'}")
        print(f"  Model:          {model_id}")
        print(f"  Theme:          {series_plan.get('theme', '?')}")
        print(f"  Mood:           {series_plan.get('mood', '?')}")
        print(f"  Environment:    {series_plan.get('environment', '?')}")
        print(f"  Variation axes: {series_plan.get('variation_axes', [])}")
        print(f"  Scenes:         {len(scenes)}")
        print(f"  Prompts:        {len(prompts)}")

        # Show first 3 scenes
        print(f"\n  Sample scenes (first 3 of {len(scenes)}):")
        for i, scene in enumerate(scenes[:3]):
            print(f"    [{i+1}] pose={scene.get('pose', '?')}, "
                  f"camera={scene.get('camera', '?')}, "
                  f"lighting={scene.get('lighting', '?')}")
            print(f"         env={scene.get('environment_detail', '?')[:50]}")

        # Show first 2 prompts
        print(f"\n  Sample prompts (first 2 of {len(prompts)}):")
        for i, p in enumerate(prompts[:2]):
            text = p.get("prompt_text", "")
            print(f"    [{i+1}] {text[:100]}…" if len(text) > 100 else f"    [{i+1}] {text}")

        print()

        if self.auto_approve:
            print("  [auto-approve: YES]")
            return True

        return self._ask_yes_no("  Approve this plan? [y/n]: ")

    def review_images(
        self,
        images: list[dict[str, Any]],
        *,
        series_id: str,
        preview_dir: Path | None = None,
    ) -> list[dict[str, Any]]:
        """Pause point 2: human reviews rendered images.

        Generates a contact sheet, prints stats, and asks for approval.
        The human can type ``y`` to accept all, ``n`` to reject all,
        or comma-separated 1-based indices to reject specific images.

        Returns the (potentially filtered) image list.
        """
        print("\n" + "=" * 60)
        print("  SUPERVISED PAUSE POINT 2: Image Review")
        print("=" * 60)
        print(f"  Series:     {series_id}")
        print(f"  Images:     {len(images)}")

        scores = [img.get("quality_score", 0) for img in images if img.get("quality_score")]
        if scores:
            print(f"  Avg score:  {sum(scores) / len(scores):.3f}")
            print(f"  Min score:  {min(scores):.3f}")
            print(f"  Max score:  {max(scores):.3f}")

        # Generate contact sheet
        if preview_dir and images:
            sheet_path = preview_dir / "contact_sheet.png"
            try:
                create_contact_sheet(images, sheet_path)
                print(f"  Sheet:      {sheet_path}")
                print(f"  (open in Finder to review)")
            except Exception as exc:
                logger.warning("Failed to create contact sheet: %s", exc)

        print()

        if self.auto_approve:
            print("  [auto-approve: YES, keeping all images]")
            return images

        response = self._ask_input(
            "  Accept all? [y/n/comma-separated indices to reject]: "
        )
        response = response.strip().lower()

        if response == "y" or response == "yes":
            return images
        elif response == "n" or response == "no":
            raise SupervisorAbort("Human rejected all images at pause point 2.")

        # Parse rejection indices (1-based)
        try:
            reject_indices = {int(x.strip()) - 1 for x in response.split(",")}
            filtered = [
                img for i, img in enumerate(images) if i not in reject_indices
            ]
            rejected_count = len(images) - len(filtered)
            print(f"  Rejected {rejected_count} images, {len(filtered)} remaining.")
            return filtered
        except ValueError:
            print("  Could not parse rejection indices, keeping all images.")
            return images

    @staticmethod
    def _ask_yes_no(prompt: str) -> bool:
        """Ask a y/n question. Returns True for y/yes."""
        try:
            response = input(prompt).strip().lower()
            return response in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            print("\n  Interrupted — treating as rejection.")
            return False

    @staticmethod
    def _ask_input(prompt: str) -> str:
        """Read a line of input."""
        try:
            return input(prompt)
        except (EOFError, KeyboardInterrupt):
            print("\n  Interrupted — treating as rejection.")
            return "n"
