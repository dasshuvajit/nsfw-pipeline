"""ComfyUI HTTP API client.

Implements the per-image submit/poll/retry contract from ARCHITECTURE.md
Section 14. ComfyUI runs separately from this project at
http://127.0.0.1:8188 and writes its output files to ~/AI/apps/ComfyUI/output/
(or wherever ComfyUI is configured to save them).

Three small deliberate deviations from the spec in Section 14:

1. ``wait_for_completion`` returns a list of :class:`RenderedImage` dataclasses
   with an absolute, resolved ``file_path`` rather than raw history dicts.
   Every downstream caller needs the resolved path; resolving it once at the
   API boundary is cleaner than threading the output dir everywhere.

2. The spec only defines :class:`RenderTimeout`. We also raise
   :class:`ComfyUIError` (server unreachable / non-200 response) and
   :class:`RenderFailed` (history reports ``status_str == 'error'``).
   Without these, those failure modes would surface as cryptic ``KeyError``\\s.

3. The file-existence guard *raises* :class:`ComfyUIError` if the file never
   appears, instead of silently returning a path that doesn't exist on disk.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


DEFAULT_BASE_URL = "http://127.0.0.1:8188"
DEFAULT_OUTPUT_DIR = Path.home() / "AI" / "apps" / "ComfyUI" / "output"


# ----- exceptions -----------------------------------------------------------


class ComfyUIError(Exception):
    """ComfyUI is unreachable, returned an error response, or sent garbage."""


class RenderTimeout(ComfyUIError):
    """A render did not complete within the configured timeout."""


class RenderFailed(ComfyUIError):
    """ComfyUI reported the prompt finished with status != 'success'."""


# ----- result type ----------------------------------------------------------


@dataclass
class RenderedImage:
    """A single image produced by a ComfyUI render."""

    filename: str
    subfolder: str
    type: str             # 'output' | 'temp' | 'input'
    file_path: Path       # absolute path on disk


# ----- client ---------------------------------------------------------------


class ComfyUIClient:
    """HTTP client for the ComfyUI API."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        output_dir: Path | str = DEFAULT_OUTPUT_DIR,
        poll_interval: float = 2.0,
        request_timeout: float = 30.0,
        file_existence_retries: int = 5,
        file_existence_interval: float = 1.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.output_dir = Path(output_dir).expanduser()
        self.poll_interval = poll_interval
        self.request_timeout = request_timeout
        self.file_existence_retries = file_existence_retries
        self.file_existence_interval = file_existence_interval

    # ----- public API -------------------------------------------------------

    def queue_prompt(self, workflow: dict[str, Any]) -> str:
        """Submit a workflow JSON to ComfyUI's /prompt; return prompt_id."""
        try:
            resp = requests.post(
                f"{self.base_url}/prompt",
                json={"prompt": workflow},
                timeout=self.request_timeout,
            )
        except requests.RequestException as exc:
            raise ComfyUIError(
                f"Could not reach ComfyUI at {self.base_url}: {exc}"
            ) from exc

        if resp.status_code != 200:
            raise ComfyUIError(
                f"ComfyUI /prompt returned HTTP {resp.status_code}: "
                f"{resp.text[:500]}"
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise ComfyUIError(
                f"ComfyUI /prompt returned non-JSON: {resp.text[:500]}"
            ) from exc

        if "prompt_id" not in data:
            raise ComfyUIError(
                f"ComfyUI /prompt response missing prompt_id: {data}"
            )
        return data["prompt_id"]

    def wait_for_completion(
        self, prompt_id: str, timeout: int = 300
    ) -> list[RenderedImage]:
        """Poll /history/{prompt_id} until the render is done or timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            entry = self._fetch_history(prompt_id)
            if entry is not None:
                self._check_status(prompt_id, entry)
                images = self._collect_images(entry)
                if not images:
                    self._raise_empty_outputs(prompt_id, entry)
                resolved = [self._resolve(img) for img in images]
                self._wait_for_files(resolved)
                return resolved
            time.sleep(self.poll_interval)

        raise RenderTimeout(
            f"Render {prompt_id} did not complete within {timeout}s"
        )

    def render_single_with_retry(
        self,
        workflow: dict[str, Any],
        max_attempts: int = 3,
        timeout: int = 300,
        retry_sleep: float = 10.0,
    ) -> list[RenderedImage]:
        """Submit and wait, retrying ONLY on RenderTimeout (per spec).

        Connection errors and explicit ComfyUI failures (RenderFailed) are
        configuration / workflow problems and propagate immediately — retrying
        them just hides the root cause.
        """
        for attempt in range(max_attempts):
            try:
                prompt_id = self.queue_prompt(workflow)
                return self.wait_for_completion(prompt_id, timeout=timeout)
            except RenderTimeout:
                if attempt == max_attempts - 1:
                    raise
                time.sleep(retry_sleep)
        # Unreachable — kept to satisfy type checkers / linters.
        raise RenderTimeout(f"Render failed after {max_attempts} attempts")

    # ----- internals --------------------------------------------------------

    def _fetch_history(self, prompt_id: str) -> dict[str, Any] | None:
        """Return the history entry for prompt_id, or None if not yet done."""
        try:
            resp = requests.get(
                f"{self.base_url}/history/{prompt_id}",
                timeout=self.request_timeout,
            )
        except requests.RequestException as exc:
            raise ComfyUIError(
                f"Could not reach ComfyUI at {self.base_url}: {exc}"
            ) from exc

        if resp.status_code != 200:
            raise ComfyUIError(
                f"ComfyUI /history returned HTTP {resp.status_code}: "
                f"{resp.text[:500]}"
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise ComfyUIError(
                f"ComfyUI /history returned non-JSON: {resp.text[:500]}"
            ) from exc

        return data.get(prompt_id)

    @staticmethod
    def _check_status(prompt_id: str, entry: dict[str, Any]) -> None:
        status = entry.get("status") or {}
        status_str = status.get("status_str")
        # 'success' is the happy path. Older ComfyUI builds may omit the field
        # entirely; treat missing as success and trust the outputs.
        if status_str and status_str != "success":
            messages = status.get("messages") or []
            raise RenderFailed(
                f"ComfyUI render {prompt_id} status={status_str!r}: {messages}"
            )

    @staticmethod
    def _collect_images(entry: dict[str, Any]) -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        outputs = entry.get("outputs") or {}
        for node_output in outputs.values():
            for img in node_output.get("images") or []:
                collected.append(img)
        return collected

    @staticmethod
    def _raise_empty_outputs(prompt_id: str, entry: dict[str, Any]) -> None:
        """Distinguish 'fully cached prompt' from 'workflow has no SaveImage'.

        ComfyUI caches node outputs by input hash. If a workflow is submitted
        twice with byte-identical inputs (same seed, same prompt, same
        everything), the second submission fires an ``execution_cached``
        message for every node and returns with ``outputs == {}`` — nothing
        was re-run, so nothing is reported. The usual workaround is to vary
        the seed (which production does naturally; only smoke tests with
        pinned seeds trip over this).

        We detect the cached case by looking for an execution_cached message
        in the status and give the caller a pointed error. Otherwise we fall
        back to the generic 'no SaveImage' guidance.
        """
        status = entry.get("status") or {}
        messages = status.get("messages") or []
        cached_nodes: list[str] = []
        for msg in messages:
            # Each message is [event_name, payload_dict].
            if not isinstance(msg, list) or len(msg) < 2:
                continue
            if msg[0] == "execution_cached":
                payload = msg[1] or {}
                cached_nodes = list(payload.get("nodes") or [])
                break

        if cached_nodes:
            raise ComfyUIError(
                f"ComfyUI returned no images for prompt {prompt_id}: every "
                f"node was served from the execution cache "
                f"({len(cached_nodes)} cached nodes: {cached_nodes[:5]}...). "
                f"This happens when a workflow is submitted with "
                f"byte-identical inputs (same seed, prompt, resolution, etc.) "
                f"as a previous run. Vary the seed to force re-execution."
            )
        raise ComfyUIError(
            f"ComfyUI returned no images for prompt {prompt_id} and the "
            f"history entry has empty outputs. Does your workflow template "
            f"include a SaveImage node?"
        )

    def _resolve(self, img: dict[str, Any]) -> RenderedImage:
        subfolder = img.get("subfolder") or ""
        filename = img["filename"]
        path = self.output_dir / subfolder / filename
        return RenderedImage(
            filename=filename,
            subfolder=subfolder,
            type=img.get("type", "output"),
            file_path=path,
        )

    def _wait_for_files(self, images: list[RenderedImage]) -> None:
        """Race-condition guard.

        ComfyUI can report a prompt 'done' a beat before the SaveImage node
        has flushed the file to disk. Poll each path up to
        ``file_existence_retries`` times. If any file is still missing after
        that, raise — silently returning a non-existent path is worse.
        """
        for img in images:
            for _ in range(self.file_existence_retries):
                if img.file_path.exists():
                    break
                time.sleep(self.file_existence_interval)
            else:
                raise ComfyUIError(
                    f"ComfyUI reported success but file never appeared on "
                    f"disk: {img.file_path}. Is ComfyUIClient.output_dir "
                    f"({self.output_dir}) pointing at the right ComfyUI "
                    f"output folder?"
                )
