"""Art-series — end-to-end LLM-direct production flow.

Generate → render → (manual) pick, with NONE of the structured
vocab/composer machinery. Phase 1 calls the art-director prompt engine
(scripts/art_director.py) for N rich, varied prompts; Phase 2 unloads the
LLM and renders each through an external ComfyUI template (the v11 Chroma
refiner by default), saving every candidate plus a manifest so a human
can cherry-pick.

The two phases are sequential (LLM never co-resident with ComfyUI). With
--seeds K>1 each prompt is rendered K times for curation.

Usage:
  python scripts/art_series.py --brief "Mediterranean summer" \
      --tier T3_artnude --count 6
  python scripts/art_series.py --brief "vintage boudoir" --tier T4_explicit \
      --count 8 --seeds 3 --orientation portrait
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import art_director  # noqa: E402
from src.render.workflow_builder import WorkflowBuilder  # noqa: E402
from src.render.comfyui_client import ComfyUIClient  # noqa: E402
from src.render.render_pipeline import (  # noqa: E402
    resolve_render_pipeline, base_resolution_for,
)
from src.niche.selector import (  # noqa: E402
    NicheLibrary, NicheLibraryError, build_selection, build_brief,
    select_niche_cycle, persona_locked_look,
)

# Seeds are RANDOM per render by default (2026-06-21). A fresh random int makes
# every submitted workflow JSON unique → no ComfyUI execution-cache collision, and
# nothing to desync on abort (the old persisted counter only advanced on success,
# so an aborted run let the next reuse its seeds → cache-served empty renders).
# The exact seed is logged per image (manifest + filename) for reproducibility;
# pass --base-seed N to force a fully deterministic, reproducible run.
_SEED_MAX = 2**31 - 1
# Per-render ComfyUI timeout. Was 1800s (30 min) — a hung render then ate the full
# half hour before failing. 300s comfortably covers a real base/chroma render
# (~2.5–4 min) while failing a hang fast so the reroll/circuit-breaker kicks in.
_RENDER_TIMEOUT_S = 300

# Persistent niche cursor — advances each --auto run; drives the per-series
# aesthetic-lock + persona rotation (deterministic, no randomness; mirrors the
# seed counter pattern).
NICHE_CURSOR_FILE = ROOT / "output/art_series/.niche_cursor"

# Persistent used-niche set for --auto: the cursor alone over-samples
# high-weight niches and repeats early, so --auto instead tracks which niches
# it has already shot and picks the next UNUSED one — exhausting every
# tier-supporting niche before repeating. Cleared automatically when the cycle
# completes (select_niche_cycle signals the reset).
USED_NICHES_FILE = ROOT / "output/art_series/.used_niches"

# Valid --orientation choices. The actual base render resolution per orientation
# comes from render_pipeline.base_resolution (config/pipeline.yaml); 4K is reached
# only in the separate manual upscale stage (scripts/upscale_folder.py).
ORIENTATIONS = ("portrait", "square", "landscape")

# ⚠️ INERT AT RENDER TIME (2026-06 Chroma R&D): the staged Chroma base runs
# cfg=1.0 (the gonzaLomo flash-heun contract) AND routes this through
# ConditioningZeroOut; the refine detailers run cfg=1 lcm/DMD. At cfg=1 the
# negative branch is never evaluated — every token below does NOTHING in the
# staged path. Safety/avoidance is carried ENTIRELY by positive prose + the
# art_director Pydantic gates + curation (which is the working reality).
# Kept as interop metadata (recorded in the manifest). Do NOT "fix" by raising
# cfg — that doubles base render time for marginal benefit; see
# docs/COMFYUI_WORKFLOWS.md.
DEFAULT_NEGATIVE = (
    "child, kid, young, minor, teen, teenager, schoolgirl, loli, underage, "
    "baby, toddler, preteen, youthful face, "
    "2girls, multiple girls, multiple subjects, two women, couple, group, "
    "grid, collage, polyptych, split screen, split image, "
    "mirror, reflection, double face, "
    "bad anatomy, bad hands, missing fingers, extra fingers, extra limbs, "
    "deformed, mutated, disfigured, fused fingers, malformed, "
    "lowres, blurry, jpeg artifacts, watermark, text, signature, "
    "cartoon, anime, illustration, 3d render, cgi, plastic skin, airbrushed"
)


def _read_niche_cursor() -> int:
    try:
        return int(NICHE_CURSOR_FILE.read_text().strip())
    except (FileNotFoundError, ValueError):
        return 0


def _advance_niche_cursor(cursor: int) -> None:
    NICHE_CURSOR_FILE.parent.mkdir(parents=True, exist_ok=True)
    NICHE_CURSOR_FILE.write_text(str(cursor + 1))


def _read_used_niches() -> list[str]:
    """Niche ids --auto has already shot this cycle (one per line)."""
    try:
        return [ln.strip() for ln in USED_NICHES_FILE.read_text().splitlines()
                if ln.strip()]
    except FileNotFoundError:
        return []


def _record_used_niche(used: list[str], niche_id: str) -> None:
    """Append ``niche_id`` to the used-set. ``used`` is the list as it stood
    for THIS pick (already emptied by the caller on a cycle reset), so this
    writes the fresh cycle's first entry after a wrap."""
    USED_NICHES_FILE.parent.mkdir(parents=True, exist_ok=True)
    new = used + ([niche_id] if niche_id not in used else [])
    USED_NICHES_FILE.write_text("\n".join(new) + "\n")


# Cross-series variety: how many prior series of the SAME niche feed the avoid
# lists. 2026-06 audit: at depth 2 a niche's older runs were free to be echoed
# (several niches have 4-5 runs); 4 runs ≈ ~32 fragments — still cheap context.
NICHE_HISTORY_RECENT = 4

# House words measured saturating the corpus ("luminous" 83% of prompts,
# "velvety" 35%, "sultry" 34% — and RISING run over run). When a word appears
# in >40% of a niche's recent prompts it gets a one-use-per-series budget.
_HOUSE_WORD_WATCHLIST = (
    "luminous", "velvety", "sultry", "chiaroscuro", "exquisite",
    "breathtaking", "unretouched", "tack-sharp", "alabaster",
    "her expression is one of", "the fine texture of her skin",
    "toward the lens with a", "discarded",
)


def _load_niche_history(
    niche_id: str, recent_series: int = NICHE_HISTORY_RECENT,
) -> "tuple[list[str], list[str], int, list[str]]":
    """Cross-series memory from prior series of the SAME niche.

    Returns ``(banned_openers, avoid_signatures, prior_series_count,
    overused_words)``:
    - banned_openers / avoid_signatures — first-8-words / first-40-words AND
      last-14-words of each prior prompt (most-recent ``recent_series`` runs),
      to seed the art_director anti-repetition lists so a new run DIFFERS from
      past ones. Covers are included (the DA shopfront converged catalog-wide
      because they had no memory).
    - prior_series_count — how many times this niche has been run (ALL prior
      runs); drives the per-run rotation offset.
    - overused_words — watchlist words present in >40% of the recent prompts;
      the LLM gets a one-use-per-series budget for them.

    Reads the existing per-series ``manifest.json`` files (the prompt store — no
    DB). Robust: skips malformed/partial manifests."""
    runs: list[tuple[str, list[dict]]] = []  # (timestamp-dirname, prompts+covers)
    for mf in (ROOT / "output/art_series").glob("*/manifest.json"):
        try:
            m = json.loads(mf.read_text())
            if (m.get("niche") or {}).get("id") == niche_id:
                rows = (m.get("prompts") or []) + (m.get("covers") or [])
                runs.append((mf.parent.name, rows))
        except Exception:  # noqa: BLE001 — skip partial/broken manifests
            continue
    runs.sort(key=lambda r: r[0])  # oldest -> newest by timestamp dirname
    prior_count = len(runs)
    recent = runs[-recent_series:] if recent_series > 0 else runs
    banned, avoid, texts = [], [], []
    for _, prompts in recent:
        for p in prompts:
            txt = p.get("prompt") or ""
            if txt:
                banned.append(art_director._opener(txt))
                avoid.append(art_director._signature(txt))
                avoid.append(art_director._tail_signature(txt))
                texts.append(txt.lower())
    overused = []
    if texts:
        for w in _HOUSE_WORD_WATCHLIST:
            if sum(1 for t in texts if w in t) / len(texts) > 0.4:
                overused.append(w)
    return banned, avoid, prior_count, overused


def _load_global_openers(recent_runs: int = 3) -> list[str]:
    """Openers from the most recent runs of ANY niche — cross-NICHE opener
    convergence was unguarded (the same 'A tight, intimate portrait captures…'
    opener shipped in 3 different niches the same week). Openers are cheap to
    ban globally; settings/poses stay niche-scoped."""
    dirs = sorted((ROOT / "output/art_series").glob("*/manifest.json"),
                  key=lambda p: p.parent.name)[-recent_runs:]
    openers: list[str] = []
    for mf in dirs:
        try:
            m = json.loads(mf.read_text())
            for p in (m.get("prompts") or []) + (m.get("covers") or []):
                txt = p.get("prompt") or ""
                if txt:
                    openers.append(art_director._opener(txt))
        except Exception:  # noqa: BLE001
            continue
    return openers


# This pipeline's prompts run ~5K tokens (the T4 explicit system+reveal prompt is
# the largest); LM Studio's JIT default context (often 4096) is too small and
# returns HTTP 400. Ensure the model is resident at a large context before gen.
LLM_MIN_CONTEXT = 8192       # reload if the loaded context is smaller than this
LLM_LOAD_CONTEXT = 32768     # the registry-native context we (re)load at


def _ensure_llm_loaded(model_tag: str) -> None:
    """Make an LM Studio model resident at a LARGE-enough context BEFORE gen — LM
    Studio JIT-loads at its app default (often 4096), which truncates this
    pipeline's ~5K-token prompts (T4 especially) into an HTTP 400. Non-LM-Studio
    tags (Ollama / MLX / remote OpenAI-compatible API) are skipped — only LM Studio
    needs this preload. Best-effort + graceful."""
    try:
        from src.memory.llm_registry import LLMRegistryLoader, BACKEND_LM_STUDIO
        if LLMRegistryLoader().backend_for_tag(model_tag) != BACKEND_LM_STUDIO:
            return
    except Exception:  # noqa: BLE001 — registry unavailable → fall back to a tag heuristic
        if "/" in model_tag or ":" in model_tag:   # slash/colon → ollama_id or remote 'vendor/model', not LM Studio
            return
    lms = shutil.which("lms") or os.path.expanduser("~/.lmstudio/bin/lms")
    if not os.path.exists(lms):
        return
    try:  # already resident with enough context? (CONTEXT column in `lms ps`)
        ps = subprocess.run([lms, "ps"], timeout=20, capture_output=True, text=True).stdout
        for line in ps.splitlines():
            if model_tag in line:
                if any(int(t) >= LLM_MIN_CONTEXT for t in line.split() if t.isdigit()):
                    return
                break
    except Exception:  # noqa: BLE001
        pass
    try:  # (re)load at the large context
        subprocess.run([lms, "unload", "--all"], timeout=30, capture_output=True)
        res = subprocess.run(
            [lms, "load", model_tag, "--context-length", str(LLM_LOAD_CONTEXT), "-y"],
            timeout=300, capture_output=True, text=True)
        if res.returncode != 0:
            # Previously this printed success unconditionally — a typoed tag /
            # OOM then burned 4 attempts × count scenes against a 4096-ctx JIT
            # load (the exact HTTP-400 failure this function exists to prevent).
            print(f"  !! lms load FAILED (rc={res.returncode}) for {model_tag} — "
                  f"generation will fall back to JIT (small context; T4 prompts "
                  f"may 400). stderr: {(res.stderr or '')[-300:]}",
                  file=sys.stderr, flush=True)
        else:
            print(f"  (ensured {model_tag} resident at {LLM_LOAD_CONTEXT} context)",
                  flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  (could not ensure LLM context — load it manually at "
              f"{LLM_LOAD_CONTEXT}: {exc})", file=sys.stderr, flush=True)


def _unload_llm(model_tag: str) -> None:
    """Free the LLM before the render phase (never co-resident with ComfyUI).
    Belt-and-braces across backends: (1) the pool's unload_all(), (2) a DIRECT
    `lms unload --all` for LM Studio, (3) a direct `ollama stop`. The direct CLI
    calls are essential because the pool's unload_all() no-ops when this fresh
    client never tracked loading the model (Gemma is pre-/JIT-loaded), which left
    ~16.7 GB resident through every render and finally OOM-crashed ComfyUI. All
    are graceful no-ops if nothing is loaded. Best-effort + grace period."""
    try:
        from src.agents.llm_client import LLMClientPool
        LLMClientPool().unload_all()
    except Exception:  # noqa: BLE001
        pass
    # LM Studio — unconditional CLI evict (the pool can't see externally-loaded models).
    lms = shutil.which("lms") or os.path.expanduser("~/.lmstudio/bin/lms")
    if os.path.exists(lms):
        try:
            subprocess.run([lms, "unload", "--all"], timeout=30, capture_output=True)
        except Exception:  # noqa: BLE001
            pass
    try:
        subprocess.run(["ollama", "stop", model_tag], timeout=30,
                       capture_output=True)
    except Exception:  # noqa: BLE001
        pass
    time.sleep(3)
    # VERIFY the eviction (the documented OOM cause was exactly this gap: a
    # resident ~16.7GB Gemma surviving "unload" and crashing ComfyUI mid-render).
    # Poll `lms ps` for the SPECIFIC tag (the tiny embedding model legitimately
    # stays resident). Warn loudly — the operator can abort before Phase 2.
    if os.path.exists(lms):
        for _ in range(3):
            try:
                ps = subprocess.run([lms, "ps"], timeout=20, capture_output=True,
                                    text=True).stdout
            except Exception:  # noqa: BLE001
                break
            if model_tag not in ps:
                break
            try:
                subprocess.run([lms, "unload", "--all"], timeout=30,
                               capture_output=True)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(3)
        else:
            print(f"  !! WARNING: {model_tag} STILL RESIDENT after unload — "
                  f"rendering now risks an OOM crash. Consider aborting and "
                  f"evicting manually (`lms unload --all`).",
                  file=sys.stderr, flush=True)


def _template_hashes(workflow_dir: Path, run_template) -> "dict | None":
    """Short sha256 per template file — a template edit can no longer silently
    change what an old manifest 'means' (manifests only recorded paths)."""
    import hashlib
    paths = (list(run_template.values()) if isinstance(run_template, dict)
             else [run_template] if run_template else [])
    out: dict = {}
    for rel in paths:
        try:
            out[str(rel)] = hashlib.sha256(
                (workflow_dir / rel).read_bytes()).hexdigest()[:16]
        except Exception:  # noqa: BLE001
            out[str(rel)] = None
    return out or None


def _comfyui_up(base_url: str, timeout: int = 5) -> bool:
    """Cheap reachability probe — used BEFORE burning LLM hours on a run whose
    render phase is doomed (a dead ComfyUI previously produced zero-image
    series that still consumed prompts, seeds and niche-cycle state)."""
    try:
        import requests
        return requests.get(base_url, timeout=timeout).status_code == 200
    except Exception:  # noqa: BLE001
        return False


def _comfyui_free(base_url: str, timeout: int = 30) -> bool:
    """Best-effort: ask ComfyUI to unload its models + free memory. Called at
    pre-flight (before Phase 1 loads the LLM) AND at the END of every render run
    (2026-06-21) so the resident model drops immediately instead of lingering.
    ComfyUI keeps the engine model (+ T5/SDXL when used) RESIDENT after a render,
    so a back-to-back series whose Phase-1 LLM (~17 GB) loads on top blows past
    the 48 GB box and the OS kills ComfyUI (cause of three crashes 2026-06-15:
    `unload LLM before Phase 2` handles the within-series direction). ComfyUI
    stays UP and reloads the model on the next render.
    Never raises — a failure just means we proceed without the free."""
    try:
        import requests
        r = requests.post(f"{base_url.rstrip('/')}/free",
                          json={"unload_models": True, "free_memory": True},
                          timeout=timeout)
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


# Circuit breaker: consecutive CONNECTIVITY failures (ComfyUI unreachable —
# not per-image render failures) before a stage aborts the run loudly instead
# of grinding through every remaining image/series producing nothing.
COMFYUI_BREAKER_LIMIT = 3


def _classify_conn_failure(exc: Exception) -> bool:
    """True if ``exc`` looks like ComfyUI-unreachable (breaker-countable)
    rather than a single bad render."""
    try:
        from src.render.comfyui_client import ComfyUIError, RenderFailed, RenderTimeout
    except Exception:  # noqa: BLE001
        return False
    return isinstance(exc, ComfyUIError) and not isinstance(
        exc, (RenderFailed, RenderTimeout))


def _embed_parameters(png: Path, *, prompt: str, seed: int, steps: int = 14,
                      sampler: str = "euler", scheduler: str = "beta",
                      cfg_scale: float = 1.0,
                      model: str = "gonzalomo_chroma_v30") -> None:
    """Add an A1111-interop ``parameters`` tEXt chunk so review/keeper PNGs are
    self-describing (prompt + seed + sampler — previously only the base PNGs
    carried metadata, and only as a ComfyUI graph with absolute paths).
    Existing chunks (e.g. the ComfyUI graph) are preserved. Best-effort."""
    try:
        from PIL import Image
        from PIL.PngImagePlugin import PngInfo
        with Image.open(png) as im:
            info = PngInfo()
            for k, v in (getattr(im, "text", None) or {}).items():
                if k != "parameters" and isinstance(v, str):
                    info.add_text(k, v)
            w, h = im.size
            info.add_text("parameters", (
                f"{prompt}\nSteps: {steps}, Sampler: {sampler} {scheduler}, "
                f"CFG scale: {cfg_scale}, Seed: {seed}, Size: {w}x{h}, "
                f"Model: {model}"))
            im.load()
            im.save(png, pnginfo=info)
    except Exception as exc:  # noqa: BLE001 — metadata is never run-fatal
        print(f"  (parameters embed skipped for {png.name}: {exc})",
              file=sys.stderr, flush=True)


# Curation reject flags that drop a candidate from keeper contention
# regardless of score (composition-fatal, not taste).
_HARD_REJECT_FLAGS = {"multiple_faces", "no_face", "scorer_error"}

# 4K-queue admission: composite quality_score floor for the ~10-min manual 4K
# pass (calibrated on recent manifests: ~top-40% of keepers; 'blurry' etc.
# flags disqualify regardless).
FOURK_QUEUE_MIN_SCORE = 0.62


def _curate(
    manifest: list[dict],
    out_dir: Path,
    *,
    content_level: str,
    keep_top: int,
    use_hps_v2: bool,
    use_image_reward: bool,
    contact_sheet_name: str = "contact_sheet.png",
) -> list[Path]:
    """Score every rendered candidate (ImageScorer), rank per prompt, flag
    rejects, copy keepers to keepers/, and build a contact sheet for human
    review. Mutates the manifest image dicts with quality_score / flags /
    keeper. Graceful: if scoring is unavailable (missing weights, etc.) every
    candidate is kept and flagged so the run never fails on curation."""
    flat: list[dict] = []
    for entry in manifest:
        for im in entry["images"]:
            flat.append({
                "file_path": str(out_dir / im["path"]),
                "prompt_text": entry["prompt"],
                "content_level": content_level,
                "_im": im,
            })
    if not flat:
        return []

    scored_ok = False
    try:
        from src.scoring.image_scorer import ImageScorer
        scorer = ImageScorer(use_hps_v2=use_hps_v2, use_image_reward=use_image_reward)
        scorer.score_batch(flat)
        scored_ok = True
    except Exception as exc:  # noqa: BLE001 — never fail a run on curation
        print(f"  (curation: scoring unavailable — keeping all candidates: {exc})",
              file=sys.stderr, flush=True)

    for f in flat:
        im = f["_im"]
        im["quality_score"] = f.get("quality_score")
        im["quality_flags"] = f.get("quality_flags")

    keeper_paths: list[Path] = []
    for entry in manifest:
        imgs = entry["images"]
        def _flags(im: dict) -> set[str]:
            try:
                return set(json.loads(im.get("quality_flags") or "[]"))
            except (ValueError, TypeError):
                return set()
        eligible = [im for im in imgs if not (_flags(im) & _HARD_REJECT_FLAGS)]
        pool = eligible or imgs  # all flagged → fall back to all
        pool_sorted = sorted(pool, key=lambda im: im.get("quality_score") or 0.0,
                             reverse=True)
        keep_n = keep_top if scored_ok else len(pool_sorted)
        keep_ids = {id(im) for im in pool_sorted[:keep_n]}
        for im in imgs:
            im["keeper"] = id(im) in keep_ids
            if im["keeper"]:
                keeper_paths.append(out_dir / im["path"])

    keepers_dir = out_dir / "keepers"
    keepers_dir.mkdir(parents=True, exist_ok=True)
    for p in keeper_paths:
        if p.exists():
            shutil.copy(p, keepers_dir / p.name)

    try:
        from src.review.contact_sheet import create_contact_sheet
        cs = [{"file_path": str(out_dir / im["path"]),
               "quality_score": im.get("quality_score") or 0.0}
              for entry in manifest for im in entry["images"]]
        create_contact_sheet(cs, out_dir / contact_sheet_name)
    except Exception as exc:  # noqa: BLE001
        print(f"  (contact sheet skipped: {exc})", file=sys.stderr, flush=True)

    n_keep = len(keeper_paths)
    n_all = sum(len(e["images"]) for e in manifest)
    print(f"  curation: {n_keep}/{n_all} keepers "
          f"({'scored' if scored_ok else 'unscored — kept all'})", flush=True)
    return keeper_paths


# ── tier-split labels ───────────────────────────────────────────────
_EXPLICIT_TIERS = {"T3_artnude", "T4_explicit"}
# Z-Image default base.json bakes in an NSFW LoRA (lora_nsfw) that restores explicit
# anatomy on the NSFW-weak official base. It is a LIABILITY at the funnel tiers — it
# strips clothing even on clean T1/T2 prompts (2026-06-20: bohemian T2 drifted 5/6 nude).
# So it is TIER-GATED at render time: full at T3/T4, OFF at T1/T2 + covers (which are T1).
_NSFW_LORA_STRENGTH = 0.8
AI_DISCLOSURE_LABEL = "Created using AI tools"

# ── Plate numbering (collectability — DA_GO_TO_MARKET.md §4) ─────────
# Family serial: "Atelier No. 014 — 'Marble Light'", with the set's gated
# images as Plates I-VI. Gaps in a buyer's collection create completion
# pressure (the proven competitor pattern). Counter persists per family.
FAMILY_SERIALS_FILE = ROOT / "output/art_series/.family_serials"
_FAMILY_DISPLAY = {
    "atelier": "The Atelier", "gilded_glamour": "Gilded Glamour",
    "golden_hour": "Golden Hour", "myth_crown": "Myth & Crown",
    "nocturne": "Nocturne",
}
_ROMAN = ("I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
          "XI", "XII")


def _next_family_serial(family: str) -> int:
    """Advance + persist the per-family set serial (1-based)."""
    try:
        serials = json.loads(FAMILY_SERIALS_FILE.read_text())
    except (FileNotFoundError, ValueError):
        serials = {}
    n = int(serials.get(family, 0)) + 1
    serials[family] = n
    FAMILY_SERIALS_FILE.parent.mkdir(parents=True, exist_ok=True)
    FAMILY_SERIALS_FILE.write_text(json.dumps(serials, indent=2))
    return n


# Per-persona appearance serial — a recurring persona's body-of-work index
# ("Clara — Golden Hour No. 07"), the collectability hook for a future per-
# persona "Room". Distinct from the family/Plate numbering above: this counts a
# persona's SETS across all niches. Keyed by lowercased name (matches the
# case-insensitive persona lookup in select_persona).
PERSONA_SERIALS_FILE = ROOT / "output/art_series/.persona_serials"


def _next_persona_serial(name: str) -> int:
    """Advance + persist the per-persona appearance serial (1-based)."""
    try:
        serials = json.loads(PERSONA_SERIALS_FILE.read_text())
    except (FileNotFoundError, ValueError):
        serials = {}
    key = name.lower()
    n = int(serials.get(key, 0)) + 1
    serials[key] = n
    PERSONA_SERIALS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PERSONA_SERIALS_FILE.write_text(json.dumps(serials, indent=2))
    return n


# Anatomy guard (auto-retry): a single female has at most TWO hands, so the hand
# detector counting >2 is a reliable EXTRA-LIMB signal a detailer CAN'T fix (it
# refines every hand it finds). We reroll the base seed until the render is clean
# (or retries run out). hand_yolov9c is the same model the refine detailer uses;
# run on CPU so it never contends with ComfyUI's GPU. No bare-foot detector works,
# so feet are not guarded — see [[reference_anatomy_detailer_limits]].
_HAND_DETECTORS: dict = {}
EXPECTED_MAX_HANDS = 2


def _hand_detector(model_path: "str | None"):
    """Lazily load + cache the hand YOLO; None (→ anatomy guard disabled) when the
    model file or ultralytics is unavailable."""
    if not model_path:
        return None
    if model_path in _HAND_DETECTORS:
        return _HAND_DETECTORS[model_path]
    det = None
    if Path(model_path).exists():
        try:
            from ultralytics import YOLO  # heavy import — only when the guard is on
            det = YOLO(model_path)
        except Exception as exc:  # noqa: BLE001
            print(f"  (anatomy guard off — can't load hand detector: {exc})",
                  file=sys.stderr, flush=True)
    else:
        print(f"  (anatomy guard off — hand detector not found: {model_path})",
              file=sys.stderr, flush=True)
    _HAND_DETECTORS[model_path] = det
    return det


def _count_hands(detector, image_path: Path, conf: float = 0.5) -> int:
    """Confident hand-detection count (the extra-limb signal). Returns -1 when no
    detector (guard disabled) — callers treat -1 as 'clean'."""
    if detector is None:
        return -1
    try:
        res = detector(str(image_path), conf=conf, device="cpu", verbose=False)[0]
        return len(res.boxes)
    except Exception as exc:  # noqa: BLE001
        print(f"  (anatomy guard error on {image_path.name}: {exc})",
              file=sys.stderr, flush=True)
        return -1


def _render_stage_base(
    rows: list[dict], *, builder, client, base_template: str, negative: str,
    resolution: tuple[int, int], base_seed: int, seeds: int,
    dest_dir: Path, out_dir: Path, prefix: str,
    rp: dict | None = None, default_orientation: str = "portrait",
    anatomy_retries: int = 0, hand_detector_path: "str | None" = None,
    reroll_start: "int | None" = None, nsfw_lora_strength: float = 0.0,
    random_seeds: bool = True,
) -> list[dict]:
    """Stage 1 (Chroma): base gen for every (prompt × seed) into dest_dir.

    Each prompt is rendered at ITS OWN orientation (the LLM chose it per prompt
    via art_director) → portrait/square/landscape variety instead of one fixed
    ratio. ``rp`` is the render_pipeline config (base_resolution per orientation);
    falls back to ``resolution`` / ``default_orientation`` when absent.

    Manifest images carry ``{base_path, seed}``; ``path`` (what curation +
    packaging read) is set later by the refine stage. Chroma is loaded ONCE
    by ComfyUI and stays resident across all base submissions — SDXL is not
    touched here, so this whole batch runs without co-residing Chroma+SDXL
    in VRAM (and the swap that caused).

    ``anatomy_retries`` > 0 enables the extra-limb guard: each base render is
    checked with the hand detector and rerolled (new seed) up to that many times
    when >2 hands are found; the least-bad render is kept if all rerolls fail."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    hand_det = _hand_detector(hand_detector_path) if anatomy_retries > 0 else None
    # Reroll seeds live past the whole candidate span (callers pass a
    # reroll_start beyond their cover range so rerolls can't collide with it).
    reroll_seed = (reroll_start if reroll_start is not None
                   else base_seed + len(rows) * seeds)
    conn_failures = 0          # circuit breaker — consecutive connectivity deaths
    manifest: list[dict] = []
    for idx, r in enumerate(rows):
        look = r["look"].split()[0]
        orientation = r.get("orientation") or default_orientation
        res = base_resolution_for(rp, orientation) if rp else resolution
        entry = {"index": idx, "look": r["look"], "prompt": r["prompt"],
                 "audit_score": r.get("audit_score"),
                 "orientation": orientation, "shot_type": r.get("shot_type"),
                 "framing_rationale": r.get("framing_rationale"),
                 "resolution": list(res), "images": []}
        for k in range(seeds):
            kept: "tuple[str, int, int] | None" = None  # (rel_path, seed, hand_count)
            for attempt in range(anatomy_retries + 1):
                if random_seeds:
                    # Fresh random seed per render (incl. rerolls) — unique workflow
                    # JSON, no cache collision; logged in the filename for repro.
                    seed = random.randint(1, _SEED_MAX)
                elif attempt == 0:
                    # Deterministic --base-seed path: per-PROMPT seeds (2026-06
                    # audit) so prompts don't share one initial latent noise.
                    seed = base_seed + idx * seeds + k
                else:
                    seed = reroll_seed
                    reroll_seed += 1
                try:
                    wf = builder.build_external(
                        external_template=base_template, prompt_text=r["prompt"],
                        negative_prompt=negative, resolution=res, seed=seed)
                    # Tier-gate the Z-Image NSFW LoRA (only present in the default
                    # zimage base) — full at T3/T4, OFF at T1/T2 + covers (anti-drift).
                    if "lora_nsfw" in wf:
                        wf["lora_nsfw"]["inputs"]["strength_model"] = nsfw_lora_strength
                    images = client.render_single_with_retry(wf, timeout=_RENDER_TIMEOUT_S)
                    outs = [im for im in images if im.type == "output"] or images
                    name = f"{prefix}{idx + 1:02d}_{look}_s{seed}.png"
                    dst = dest_dir / name
                    shutil.copy(outs[-1].file_path, dst)
                except Exception as exc:  # noqa: BLE001 — surface, continue
                    if _classify_conn_failure(exc):
                        conn_failures += 1
                        if conn_failures >= COMFYUI_BREAKER_LIMIT:
                            raise RuntimeError(
                                f"ComfyUI unreachable ({conn_failures} consecutive "
                                f"connectivity failures) — aborting the run instead "
                                f"of grinding through empty renders. Restart ComfyUI "
                                f"and re-run.") from exc
                    print(f"  [base {idx + 1}/{len(rows)}] seed {seed} FAILED: {exc}",
                          file=sys.stderr, flush=True)
                    continue
                conn_failures = 0
                hands = _count_hands(hand_det, dst)
                rel = str(dst.relative_to(out_dir))
                tag = f" (reroll {attempt})" if attempt else ""
                if hands <= EXPECTED_MAX_HANDS:          # clean (or guard off: -1)
                    if kept is not None:
                        (out_dir / kept[0]).unlink(missing_ok=True)  # drop earlier reject
                    kept = (rel, seed, hands)
                    print(f"  [base {idx + 1}/{len(rows)}] {orientation} "
                          f"{res[0]}x{res[1]} seed {seed} -> {name}{tag}", flush=True)
                    break
                # extra limb — keep the least-bad so far, then reroll a new seed
                print(f"  [base {idx + 1}/{len(rows)}] seed {seed}: {hands} hands "
                      f"(extra limb) — rerolling {attempt + 1}/{anatomy_retries}",
                      file=sys.stderr, flush=True)
                if kept is None or hands < kept[2]:
                    if kept is not None:
                        (out_dir / kept[0]).unlink(missing_ok=True)
                    kept = (rel, seed, hands)
                else:
                    dst.unlink(missing_ok=True)
            if kept is not None:
                rel, seed, hands = kept
                entry["images"].append({"base_path": rel, "seed": seed})
                if hands > EXPECTED_MAX_HANDS:
                    print(f"  [base {idx + 1}/{len(rows)}] !! still {hands} hands after "
                          f"{anatomy_retries} rerolls — kept best; CULL THIS CANDIDATE",
                          file=sys.stderr, flush=True)
        manifest.append(entry)
    return manifest


def _select_refine_template(tier: str, rp: dict) -> str:
    """Stage-2 refine template for a tier's MAIN images. T4_explicit gets the
    variant with the vagina detailer (``refine_template_t4``); every lower tier
    uses ``refine_template`` so a tasteful T3 nude never has its genitals detailed
    (tier purity)."""
    base = rp["refine_template"]
    return rp.get("refine_template_t4", base) if tier == "T4_explicit" else base


def _refine_templates_for(tier: str, rp: dict) -> "tuple[str, str]":
    """``(main_template, cover_template)`` for the staged refine. MAIN follows the
    tier (T4 → vagina-detailer variant); COVERS are always SFW so they ALWAYS use
    the base refine — even on a T4 run — so a public/teaser image is never
    genital-detailed. The whole routing decision lives here so the cover-purity
    invariant is tested in one place."""
    return _select_refine_template(tier, rp), rp["refine_template"]


def _template_has_genital_detailer(workflow_dir: Path, template_rel: "str | None") -> bool:
    """True if a workflow template contains a vagina/genital detailer node.
    CONTENT-based (not filename-based) so a renamed template can't smuggle explicit
    detailing past the tier-purity guard. Missing/unreadable template → False."""
    if not template_rel:
        return False
    try:
        wf = json.loads((workflow_dir / template_rel).read_text())
    except Exception:  # noqa: BLE001 — absent/bad template → treat as clean
        return False
    for nid, nd in wf.items():
        blob = (nid + " " + str(nd.get("inputs", {}).get("model_name", ""))).lower()
        if "vagina" in blob:
            return True
    return False


def _violates_tier_purity(tier: str, main_template: "str | None", workflow_dir: Path) -> bool:
    """True if rendering would run a genital detailer outside T4_explicit — e.g. a
    ``--refine-template`` override leaking the T4 template into a lower tier. The
    staged render aborts when this is True."""
    return tier != "T4_explicit" and _template_has_genital_detailer(workflow_dir, main_template)


def _render_stage_refine(
    manifest: list[dict], *, builder, client, refine_template: str,
    dest_dir: Path, out_dir: Path,
) -> None:
    """Stage 2 (SDXL): refine each stage-1 base image into a review image.

    Sets ``images[].path`` (== ``review_path``) to the review image so
    curation + packaging operate on the finished non-4K images the human
    picks from. SDXL DMD is loaded ONCE on the first refine submit and stays
    resident for the rest of the batch (Chroma is no longer referenced) — the
    swap win. On refine failure the base image is used as the review path so
    a usable image is never silently dropped."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    conn_failures = 0          # circuit breaker — consecutive connectivity deaths
    for entry in manifest:
        for im in entry["images"]:
            base_rel = im.get("base_path")
            if not base_rel:
                continue
            base_abs = (out_dir / base_rel).resolve()
            try:
                wf = builder.build_image_stage(
                    external_template=refine_template,
                    image_path=str(base_abs), seed=im["seed"])
                images = client.render_single_with_retry(wf, timeout=_RENDER_TIMEOUT_S)
                outs = [i for i in images if i.type == "output"] or images
                name = Path(base_rel).name  # mirror base filename under images/
                dst = dest_dir / name
                shutil.copy(outs[-1].file_path, dst)
                im["path"] = str(dst.relative_to(out_dir))
                im["review_path"] = im["path"]
                conn_failures = 0
                # Self-describing review/keeper PNG (A1111-interop chunk) —
                # these are the files that get packaged and sold.
                _embed_parameters(dst, prompt=entry["prompt"], seed=im["seed"])
                print(f"  [refine] {name}", flush=True)
            except Exception as exc:  # noqa: BLE001 — surface, fall back to base
                if _classify_conn_failure(exc):
                    conn_failures += 1
                    if conn_failures >= COMFYUI_BREAKER_LIMIT:
                        raise RuntimeError(
                            f"ComfyUI unreachable ({conn_failures} consecutive "
                            f"connectivity failures in refine) — aborting; every "
                            f"image would silently 'fall back to base'.") from exc
                im["path"] = base_rel  # base image is still a usable review image
                im["review_path"] = base_rel
                print(f"  [refine] {base_rel} FAILED ({exc}); using base as review",
                      file=sys.stderr, flush=True)


def _gen_metadata(selection, brief: str, tier: str, image_count: int,
                  model_tag: str) -> dict:
    """Set-level DA metadata (title/description/tags) via MetadataGenerator,
    with the niche's 5-axis tags merged in + the mandatory AI/Mature labels.
    Falls back to a minimal stub if the LLM call fails (never blocks)."""
    if selection is not None:
        theme = selection.niche.da_folder
        mood = ", ".join(selection.niche.tags[:4])
        env = selection.niche.brief_seed
        lk = selection.aesthetic_lock
        style_kw = "; ".join(x for x in (lk.palette, lk.lighting, lk.photographer) if x)
        seed_tags = list(selection.niche.tags)
    else:
        theme = (brief or "Untitled")[:60]
        mood, env, style_kw, seed_tags = "", brief or "", "", []

    meta: dict
    try:
        from src.agents.metadata_generator import MetadataGenerator
        from src.agents.llm_client import LLMClientPool
        # Route by the tag's backend (LM Studio / Ollama / MLX) via the registry,
        # exactly like the prompt path. Hardcoding OllamaClient here 404'd whenever
        # model_tag was an LM Studio tag (e.g. the default Gemma), silently
        # dropping the LLM-bespoke set title/description back to the niche stub.
        meta = MetadataGenerator(LLMClientPool()).generate(
            theme=theme, mood=mood, environment=env, content_level=tier,
            image_count=image_count, style_keywords=style_kw, model=model_tag)
    except Exception as exc:  # noqa: BLE001
        print(f"  (metadata fallback — generator failed: {exc})",
              file=sys.stderr, flush=True)
        meta = {"title": (theme or "Untitled")[:80],
                "description": (selection.niche.brief_seed if selection else brief or ""),
                "tags": []}
    # merge niche 5-axis tags, dedup, cap 25; mandatory discovery tags
    merged = list(dict.fromkeys([*meta.get("tags", []), *seed_tags, "aiart"]))[:25]
    meta["tags"] = merged
    meta["labels"] = {"ai_tools": AI_DISCLOSURE_LABEL, "mature": True}
    return meta


# DA price card by tier (docs/DA_GO_TO_MARKET.md §1) — impulse-low; the volume +
# subscription/Fanvue funnel is the business, not per-image margin.
_PRICE_BY_TIER = {
    "T1_suggestive": "Free public teaser / $2 single",
    "T2_implied": "$2-3 single / $5-8 themed set",
    "T3_artnude": "$5-8 single (4K-finished) / $5-8 Premium Gallery set",
    "T4_explicit": "$5-8 single / $10-mo explicit subscription (DA gated or Fanvue)",
}


def _suggested_groups(niche_tags: list[str]) -> list[str]:
    """Generic DA Group suggestions by genre — Groups are the main reach
    multiplier (the DA feed is weak). Refine per niche over time."""
    t = set(niche_tags)
    g = ["AI-Generated-Art", "DigitalArt"]
    if t & {"fineart", "figurestudy", "classical", "renaissance", "baroque", "oldmaster"}:
        g.append("Fine-Art-Nude-Groups")
    if t & {"boudoir", "lingerie", "glamour", "sensual"}:
        g.append("Glamour-and-Boudoir-Groups")
    if t & {"pinup", "oldhollywood", "vintageglamour", "artdeco"}:
        g.append("Pinup-and-Vintage-Groups")
    if t & {"fantasy", "goddess", "vampire", "medieval", "gothic", "angel", "mythology"}:
        g.append("Fantasy-Art-Groups")
    if t & {"blackandwhite", "monochrome"}:
        g.append("Black-and-White-Art-Groups")
    return g[:5]


def _emit_posting_templates(pkg: Path, metadata: dict, tier: str,
                            is_explicit: bool) -> None:
    """Per-image copy-paste posting templates (TITLE/FOLDER/DESCRIPTION/TAGS/
    GROUPS/MATURE/AI/PRICE) so manual DA upload is trivial — one .txt per image
    under package/<niche>/posting_templates/. Titles are series-numbered for
    collectibility (the competitor pattern)."""
    tdir = pkg / "posting_templates"
    tdir.mkdir(exist_ok=True)
    folder = metadata["da_folder"]
    persona = metadata.get("persona")
    persona_serial = metadata.get("persona_serial")
    groups = ", ".join(metadata.get("da_groups", []))
    ai = metadata["labels"]["ai_tools"]
    fam_disp = metadata.get("family_display") or ""
    serial = metadata.get("family_serial")
    pub_advisories = metadata["public"].get("advisories") or {}

    def _write(images: list[str], meta: dict, is_gated: bool) -> None:
        base = meta.get("title") or folder
        if persona:
            # Persona body-of-work index ("Clara — <title> No. 07"). ARABIC so it
            # never visually collides with the Roman per-Plate numerals below.
            base = (f"{persona} — {base} No. {persona_serial:02d}"
                    if persona_serial else f"{persona} — {base}")
        price = _PRICE_BY_TIER.get(tier if is_gated else "T1_suggestive", "$2-3")
        where = "gated" if is_gated else "public"
        note = "CLEAN file (no watermark)" if is_gated else "watermarked SFW teaser"
        for n, img in enumerate(images, 1):
            # Plate titling (collectability): gated images are Plates of a
            # family-serialed Set; covers stay simply titled.
            plate = _ROMAN[n - 1] if n <= len(_ROMAN) else str(n)
            if is_gated and serial:
                title = f"{fam_disp} No. {serial:03d} — {base} · Plate {plate}"
                desc_prefix = (f"Plate {plate} of Set {serial}: \"{base}\" "
                               f"({fam_disp} collection). ")
            else:
                title = f"{base} {n}"
                desc_prefix = ""
            txt = (
                f"TITLE: {title}\n"
                f"FOLDER: {folder}"
                f"{'  [Premium Gallery / Subscription]' if is_gated else '  [public SFW]'}\n"
                f"DESCRIPTION: {desc_prefix}{meta.get('description', '')}\n"
                f"TAGS: {', '.join(meta.get('tags', []))}\n"
                f"GROUPS: {groups}\n"
                f"MATURE: yes\n"
                f"AI-LABEL: {ai}\n"
                f"PRICE: {price}\n"
                f"FILE: {where}/{img}  ({note})\n"
            )
            if not is_gated and pub_advisories.get(img):
                txt += f"REVIEW: ⚠ {pub_advisories[img]}\n"
            (tdir / f"{Path(img).stem}.txt").write_text(txt)

    _write(metadata["public"]["images"], metadata["public"]["metadata"], False)
    if is_explicit:
        _write(metadata["gated"]["images"], metadata["gated"]["metadata"], True)


def _keepers_of(manifest: list[dict], out_dir: Path) -> list[Path]:
    return [out_dir / im["path"] for e in manifest for im in e["images"]
            if im.get("keeper")]


# Secondary-screen YOLO floors (the detailer detectors — weak as scene-level
# classifiers: a clear color nude scored only 0.36 and B&W nudes 0.00, so
# these are belt-and-braces behind NudeNet, not the primary gate).
_NUDITY_DETECTOR_CONF = {"nipples": 0.30, "vagina": 0.45}

# Primary screen: NudeNet (dedicated NSFW-region classifier). Calibrated
# 2026-06-12 on 15 labeled production images from two nights: flagged 9/9
# nudes INCLUDING every B&W frame the YOLOs missed (scores 0.44-0.83) and
# passed 6/6 cleans incl. lingerie/draped edge cases (zero detections).
_NUDENET_EXPOSED = frozenset({
    "FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED", "ANUS_EXPOSED",
})
_NUDENET_CONF = 0.35
_NUDENET_CACHE: dict = {"det": None, "tried": False}


def _nudenet_detector():
    """Lazy singleton NudeDetector; None when nudenet isn't installed."""
    if not _NUDENET_CACHE["tried"]:
        _NUDENET_CACHE["tried"] = True
        try:
            from nudenet import NudeDetector
            _NUDENET_CACHE["det"] = NudeDetector()
        except Exception as exc:  # noqa: BLE001
            print(f"  (NudeNet unavailable — tier-drift screen falls back to "
                  f"the weaker YOLO pair: {exc})", file=sys.stderr, flush=True)
    return _NUDENET_CACHE["det"]


def _nudity_hit(image_path: Path, detector_paths: "tuple | list") -> bool:
    """True when a PUBLIC-bound image contains exposed nudity. Chroma render
    drift put fully-nude frames in public packages on two consecutive nights
    despite clean prompts — prompt-side gates cannot see the RENDER, so the
    public folder gets a vision gate. PRIMARY: NudeNet (handles color AND
    B&W). SECONDARY: the refine-stage nipple/vagina YOLOs. Conservative
    direction: a false positive merely quarantines to _tier_drift/ for the
    operator to restore. No detectors available → False (graceful)."""
    nd = _nudenet_detector()
    if nd is not None:
        try:
            for d in nd.detect(str(image_path)):
                if (d.get("class") in _NUDENET_EXPOSED
                        and d.get("score", 0) >= _NUDENET_CONF):
                    return True
            # NudeNet's verdict is FINAL when it loaded: a 112-image catalog
            # sweep (2026-06-12) showed the YOLO secondary contributes only
            # false positives on full scenes (15/16 of its flags had zero
            # NudeNet detections) while NudeNet alone went 9/9 nudes + 6/6
            # cleans on the labeled set. YOLOs below = fallback-only.
            return False
        except Exception as exc:  # noqa: BLE001 — screening is best-effort
            print(f"  (NudeNet screen error on {image_path.name}: {exc})",
                  file=sys.stderr, flush=True)
    for dp in (detector_paths or []):
        det = _hand_detector(dp)        # generic lazy YOLO loader/cache
        if det is None:
            continue
        conf = next((c for k, c in _NUDITY_DETECTOR_CONF.items()
                     if k in str(dp).lower()), 0.45)
        try:
            res = det(str(image_path), conf=conf, device="cpu", verbose=False)[0]
            if len(res.boxes):
                return True
        except Exception as exc:  # noqa: BLE001 — screening is best-effort
            print(f"  (tier-drift screen error on {image_path.name}: {exc})",
                  file=sys.stderr, flush=True)
    return False


# Advisory nudge (NOT a gate). Calibrated 2026-06-17 on the aspirational_luxe T2
# render: NudeNet CANNOT separate a topless draped-shirt frame from a clean
# bare-shouldered dress — both score FEMALE_BREAST_COVERED ~0.55-0.66 + armpits
# exposed (the clean dress scored HIGHER on both). So no threshold catches the
# topless without also catching the clean frame. Rather than quarantine good
# frames (useless) or do nothing, we flag SKIN-FORWARD public frames for the
# operator's mandatory visual QA — a "look harder at these" triage, honestly not
# a topless detector. Floor matched to the calibration (armpits 0.53-0.76).
_ADVISORY_SKIN_CONF = 0.40
_ADVISORY_SKIN_CTX = ("ARMPITS_EXPOSED", "BELLY_EXPOSED")


def _tier_truth_advisory(image_path: Path) -> str:
    """Non-blocking advisory for a public-bound frame that PASSED ``_nudity_hit``
    but is skin-forward (bare shoulders/torso while the chest reads only
    'covered') — the exact configuration where NudeNet's covered-vs-exposed call
    is unreliable. Returns a short reason for the operator to eyeball, or '' (no
    detector / hard-exposed-and-already-quarantined / modest frame). Never
    quarantines and never claims a problem."""
    nd = _nudenet_detector()
    if nd is None:
        return ""
    try:
        dets = nd.detect(str(image_path))
    except Exception:  # noqa: BLE001 — advisory is best-effort
        return ""
    # Hard exposed → _nudity_hit already quarantined it; no advisory needed.
    if any(d.get("class") in _NUDENET_EXPOSED and d.get("score", 0) >= _NUDENET_CONF
           for d in dets):
        return ""
    covered = any(d.get("class") == "FEMALE_BREAST_COVERED"
                  and d.get("score", 0) >= _ADVISORY_SKIN_CONF for d in dets)
    skin = sorted({d["class"] for d in dets
                   if d.get("class") in _ADVISORY_SKIN_CTX
                   and d.get("score", 0) >= _ADVISORY_SKIN_CONF})
    if covered and skin:
        where = ", ".join(c.replace("_EXPOSED", "").lower() for c in skin)
        return (f"skin-forward (bare {where}, chest read as merely covered) — "
                f"verify by eye the top isn't open/sheer; NudeNet's covered-vs-"
                f"exposed call is unreliable on draped or dim frames")
    return ""


def _package(
    out_dir: Path, selection, tier: str,
    main_manifest: list[dict], cover_manifest: list[dict],
    gated_meta: dict, public_meta: dict, watermark_cfg: dict,
    nudity_detector_paths: "tuple | list" = (),
    postgrade_cfg: "dict | None" = None,
) -> Path:
    """Assemble a publish-ready package with a tier-split public/gated layout.

    HARD RULE (DA SFW-shopfront ToS): the public folder + cover NEVER contain a
    T3/T4 image. For explicit runs the public set is sourced ONLY from the
    dedicated SFW (T1) cover renders; the explicit keepers go to gated/.
    Every PUBLIC-bound image additionally passes the render-level nudity
    screen (``_nudity_hit``) — drifted frames land in ``_tier_drift/``."""
    is_explicit = tier in _EXPLICIT_TIERS
    folder = (selection.niche.da_folder if selection else "Series").replace("/", "-")
    if selection and selection.persona:
        folder = f"{folder} - {selection.persona.name}"
    pkg = out_dir / "package" / folder
    public_dir = pkg / "public"
    gated_dir = pkg / "gated"
    public_dir.mkdir(parents=True, exist_ok=True)
    gated_dir.mkdir(parents=True, exist_ok=True)

    main_keepers = _keepers_of(main_manifest, out_dir)
    cover_keepers = _keepers_of(cover_manifest, out_dir)

    public_advisories: dict[str, str] = {}

    def _copy_into(paths: list[Path], dest: Path, *, screen: bool = False) -> list[str]:
        names = []
        for p in paths:
            if not p.exists():
                continue
            if screen and _nudity_hit(p, nudity_detector_paths):
                drift = pkg / "_tier_drift"
                drift.mkdir(exist_ok=True)
                shutil.copy(p, drift / p.name)
                print(f"  !! TIER DRIFT: {p.name} rendered NUDE despite a "
                      f"clean {tier} prompt — quarantined to _tier_drift/ "
                      f"(NEVER post publicly; restore manually only if the "
                      f"detector is wrong).", file=sys.stderr, flush=True)
                continue
            if screen:  # passed the gate — flag skin-forward frames for visual QA
                adv = _tier_truth_advisory(p)
                if adv:
                    public_advisories[p.name] = adv
            shutil.copy(p, dest / p.name)
            names.append(p.name)
        return names

    if is_explicit:
        gated_names = _copy_into(main_keepers, gated_dir)
        public_names = _copy_into(cover_keepers, public_dir, screen=True)
        public_tier = "T1_suggestive"
    else:
        public_names = _copy_into(main_keepers, public_dir, screen=True)
        gated_names = []
        public_tier = tier

    cover_name = public_names[0] if public_names else None  # SFW by construction

    # ── 4K selection rule (the manual 4K stage finally has criteria) ────
    # keepers/ is NOT a shortlist (with --keep-top 1 every image becomes a
    # keeper, including 'blurry'-flagged ones). 4k_queue/ = keepers that
    # actually earn the ~10-min 4K pass: scored ≥ threshold AND flag-free.
    # Operator vetoes by deleting files, then runs
    #   python scripts/upscale_folder.py <run>/4k_queue
    queue: list[Path] = []
    for e in main_manifest:
        for im in e["images"]:
            if not im.get("keeper"):
                continue
            score = im.get("quality_score") or 0.0
            try:
                flags = set(json.loads(im.get("quality_flags") or "[]"))
            except (ValueError, TypeError):
                flags = set()
            if score >= FOURK_QUEUE_MIN_SCORE and not flags:
                p = out_dir / im["path"]
                if p.exists():
                    queue.append(p)
    if queue:
        qdir = out_dir / "4k_queue"
        qdir.mkdir(exist_ok=True)
        for p in queue:
            shutil.copy(p, qdir / p.name)
        print(f"  4k_queue: {len(queue)} image(s) earned the 4K pass "
              f"(score ≥ {FOURK_QUEUE_MIN_SCORE}, no flags)", flush=True)

    # Cinematic post-grade (filmic colour/bloom/vignette) — the finishing layer
    # that adds the look Z-Image/Chroma under-render. Runs on public AND gated,
    # AFTER render and BEFORE watermark (so branding sits on the graded frame); 4K
    # is graded post-USDU in upscale_folder, never here (2026-06-18 R&D).
    if postgrade_cfg and postgrade_cfg.get("enabled", True):
        try:
            from src.postprocess.grader import Grader
            grader = Grader(postgrade_cfg)
            n_graded = 0
            for d in (public_dir, gated_dir):
                for p in sorted(d.glob("*.png")):
                    grader.apply(p, p)
                    n_graded += 1
            print(f"  post-grade: {n_graded} image(s) graded "
                  f"(strength {grader.strength})", flush=True)
        except Exception as exc:  # noqa: BLE001 — images are safe; grade is optional polish
            print(f"  (post-grade skipped: {exc})", file=sys.stderr, flush=True)

    # Watermark the PUBLIC teasers (branding/funnel); gated stays clean for buyers.
    try:
        from src.postprocess.watermarker import Watermarker
        wm = Watermarker(watermark_cfg)
        for p in sorted(public_dir.glob("*.png")):
            wm.apply(p, p, content_level=public_tier)
    except Exception as exc:  # noqa: BLE001
        print(f"  (watermark skipped: {exc})", file=sys.stderr, flush=True)

    # Plate numbering: per-family set serial for collectability (§4 of the
    # go-to-market doc). Falls back to a "series" family for --brief runs.
    family = (selection.niche.family if (selection and selection.niche.family)
              else "series")
    family_serial = _next_family_serial(family)
    family_display = _FAMILY_DISPLAY.get(family, family.title())
    # Per-persona appearance serial — incremented HERE only (packaging runs once
    # per series; --prompts-only returns before _package, so prompt A/Bs never
    # mint a number). Re-running a packaged series mints the next index by design.
    persona_serial = (_next_persona_serial(selection.persona.name)
                      if (selection and selection.persona) else None)

    metadata = {
        "da_folder": folder,
        "tier": tier,
        "niche": selection.niche.id if selection else None,
        "family": family,
        "family_display": family_display,
        "family_serial": family_serial,
        "persona": selection.persona.name if (selection and selection.persona) else None,
        "persona_serial": persona_serial,
        "labels": {"ai_tools": AI_DISCLOSURE_LABEL, "mature": True},
        "price_by_tier": _PRICE_BY_TIER.get(tier),
        "da_groups": _suggested_groups(selection.niche.tags if selection else []),
        "posting_strategy": ("public SFW teasers → Premium Gallery / Subscription"
                             if is_explicit else "public SFW set (T1/T2)"),
        "watermark_status": "public watermarked; gated clean",
        "cover_image": cover_name,
        "public": {"count": len(public_names), "images": public_names,
                   "metadata": public_meta, "advisories": public_advisories},
        "gated": {"count": len(gated_names), "images": gated_names,
                  "metadata": gated_meta},
    }
    (pkg / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (pkg / "POSTING_CHECKLIST.md").write_text(_posting_checklist(metadata, is_explicit))
    _emit_posting_templates(pkg, metadata, tier, is_explicit)
    print(f"  packaged: {pkg}  (public={len(public_names)}, gated={len(gated_names)}, "
          f"cover={cover_name}, +posting_templates/)", flush=True)
    if public_advisories:
        print(f"  ⚠ visual-QA advisory: {len(public_advisories)} skin-forward "
              f"public frame(s) flagged for a double-check (see POSTING_CHECKLIST.md) "
              f"— {', '.join(public_advisories)}", flush=True)
    return pkg


def _posting_checklist(meta: dict, is_explicit: bool) -> str:
    pub, gat = meta["public"], meta["gated"]
    folder = meta["da_folder"]
    lines = [
        f"# Posting checklist — {folder}",
        "",
        "> Generated by art_series.py. Upload is MANUAL and human-paced "
        "(bulk/automated posting is the #1 DeviantArt ban vector). Keep local "
        "masters — DeviantArt may delete content at its sole discretion. See "
        "docs/DA_GO_TO_MARKET.md for the full pricing/shop/series strategy.",
        "",
        f"**Price:** {meta.get('price_by_tier', 'see DA_GO_TO_MARKET.md')}  ·  "
        f"**Groups:** {', '.join(meta.get('da_groups', []))}  ·  "
        f"**4K-finish heroes:** review `4k_queue/` (auto-picked: score ≥ "
        f"{FOURK_QUEUE_MIN_SCORE}, flag-free), delete any you veto, then "
        f"`python scripts/upscale_folder.py <run>/4k_queue`  ·  "
        "**Per-image copy-paste:** `posting_templates/`",
        "",
        "## Hard rules (verified ToS, do not skip)",
        "- [ ] **VISUALLY verify every `public/` image is tier-true** — Chroma "
        "sometimes strips clothing despite a clean T1/T2 prompt (render drift; "
        "two fully-nude 'T2' images were caught in public folders on "
        "2026-06-10). A nude image posted public is the ToS breach that kills "
        "the account.",
        f"- [ ] Apply the **\"{meta['labels']['ai_tools']}\"** label on EVERY "
        "for-sale piece (DA AI-disclosure requirement).",
        "- [ ] Set **Mature Content** on every piece.",
        "- [ ] The COVER / thumbnail / tier-cover MUST be SFW — use only images "
        "from `public/` (never `gated/`).",
        "- [ ] Do NOT use \"hyperrealistic\"/\"realistic\"/\"real woman\" in "
        "title/tags — frame as art/render/digital art.",
        "- [ ] Human-paced cadence (a few posts/day max); submit to relevant "
        "Groups (the main reach multiplier; DA feed is weak).",
        "",
        f"## PUBLIC post (top-of-funnel) — {pub['count']} image(s) in `public/`",
        f"- Cover: `{meta['cover_image']}`",
        f"- Title: {pub['metadata'].get('title','')}",
        f"- Description: {pub['metadata'].get('description','')}",
        f"- Tags: {', '.join(pub['metadata'].get('tags', []))}",
        "",
    ]
    advisories = pub.get("advisories") or {}
    if advisories:
        lines += [
            "### ⚠ Double-check these skin-forward frames (ADVISORY, not a block)",
            "> They passed the nudity screen but show bare shoulders/torso where "
            "NudeNet's covered-vs-exposed call is unreliable. The detector cannot "
            "tell a clean bare-shouldered dress from an open/draped top — so "
            "**eyeball each by hand** before posting (it does NOT mean they're "
            "nude).",
        ]
        lines += [f"- [ ] `{img}` — {reason}" for img, reason in advisories.items()]
        lines.append("")
    if is_explicit:
        lines += [
            f"## GATED set (Subscription / Premium Gallery) — {gat['count']} "
            "image(s) in `gated/`",
            "- [ ] Route to a paid Subscription tier or Premium Gallery "
            "(requires DA Core membership to sell). NEVER post explicit "
            "publicly.",
            f"- Title: {gat['metadata'].get('title','')}",
            f"- Description: {gat['metadata'].get('description','')}",
            f"- Tags: {', '.join(gat['metadata'].get('tags', []))}",
            "- [ ] (Funnel option) Mirror the gated set to Fanvue with the "
            "AI-disclosure in bio/caption (Patreon bans synthetic AI nudes).",
            "",
        ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="LLM-direct art series (gen→render)")
    # Niche selection (one of --auto / --niche / --brief is required):
    ap.add_argument("--auto", action="store_true",
                    help="auto-pick a niche from config/niche_library.yaml "
                    "(evergreen-core + periodic trend, via the niche cursor)")
    ap.add_argument("--niche", default=None,
                    help="force a specific niche id from the library")
    ap.add_argument("--persona", action="store_true",
                    help="bind a recurring persona (rotated from the pool)")
    ap.add_argument("--persona-name", default=None,
                    help="bind a specific persona by name (e.g. Clara)")
    ap.add_argument("--brief", default=None,
                    help="manual creative brief/theme (overrides niche selection)")
    ap.add_argument("--tier", default="T3_artnude",
                    choices=list(art_director.TIER_DIRECTIVES))
    ap.add_argument("--count", type=int, default=6, help="number of prompts")
    ap.add_argument("--seeds", type=int, default=1,
                    help="renders per prompt (>1 = candidates to pick from)")
    ap.add_argument("--anatomy-retries", type=int, default=2,
                    help="extra-limb guard: reroll a base render up to N times when "
                         "the hand detector finds >2 hands (0 = off; default 2)")
    ap.add_argument("--orientation", default="portrait",
                    choices=list(ORIENTATIONS))
    ap.add_argument("--base-template", default=None,
                    help="staged stage-1 base template (default: pipeline.yaml "
                    "render_pipeline.base_template)")
    ap.add_argument("--refine-template", default=None,
                    help="staged stage-2 refine template (default: pipeline.yaml "
                    "render_pipeline.refine_template)")
    ap.add_argument("--refine", action="store_true",
                    help="OPT-IN: run the stage-2 SDXL detailer refine. Default is "
                    "base-only (like 4K) — keeps a series on one resident model.")
    ap.add_argument("--no-refine", action="store_true",
                    help="(default) skip stage-2 refine; review images = raw base "
                    "render. Kept for back-compat; base-only is now the default.")
    ap.add_argument("--no-postgrade", action="store_true",
                    help="skip the cinematic post-grade pass (pipeline.yaml postgrade)")
    ap.add_argument("--engine", choices=["chroma", "zimage"], default="zimage",
                    help="render engine: zimage (Z-Image Turbo — DEFAULT; official base "
                         "+ NSFW_master + dopsd_white LoRA stack, dpmpp_sde) or chroma "
                         "(gonzaLomo Chroma v30 — for B&W/painterly/period/fantasy niches). "
                         "Swaps the base + refine + refine_T4 templates as a set; explicit "
                         "--base-template / --refine-template still override. (ZPop base is "
                         "templates/zimage/base_zpop.json via --base-template.)")
    ap.add_argument("--hires", action="store_true",
                    help="Z-Image deep-shrink HIRES base (gonzaLomo v11: "
                         "PatchModelAddDownscale + higher per-orientation resolution) "
                         "for max-detail hero renders. Forces --engine zimage; "
                         "~2x slower on MPS. The SDXL detailer stage still applies.")
    ap.add_argument("--model-tag", default=art_director.DEFAULT_LLM_TAG,
                    help="LLM tag for prompt generation; routed to its backend "
                         "(LM Studio / Ollama / MLX) by config/llm_models.yaml. "
                         "Default = registry default_llm (Gemma, LM Studio). "
                         "Pass art_director.CYDONIA_TAG for the Ollama/Cydonia path.")
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--word-band", default="120-180",
                    help="prose word band 'lo-hi' (flash-merged Chroma prefers "
                         "~150-word prose; do not exceed ~250)")
    ap.add_argument("--no-audit-gate", action="store_true",
                    help="disable the inline audit_prompts quality gate")
    ap.add_argument("--no-curate", action="store_true",
                    help="skip ImageScorer curation (keep every render)")
    ap.add_argument("--keep-top", type=int, default=1,
                    help="keepers per prompt after scoring (default 1)")
    ap.add_argument("--covers", type=int, default=2,
                    help="SFW cover/teaser prompts to render for explicit (T3/T4) "
                    "runs — the public shopfront (default 2; 0 to skip)")
    ap.add_argument("--no-package", action="store_true",
                    help="skip the publish-ready packaging step")
    ap.add_argument("--prompts-only", action="store_true",
                    help="generate + save prompts and STOP (no render) — for "
                    "prompt analysis / LLM A-B (writes prompts.json)")
    ap.add_argument("--base-seed", type=int, default=None,
                    help="explicit base seed for a DETERMINISTIC, reproducible run "
                    "(per-prompt seeds = base_seed + offset). Default = a fresh "
                    "RANDOM seed per render (logged per image for reproducibility).")
    ap.add_argument("--out-dir", default="",
                    help="output dir (default: output/art_series/<timestamp>)")
    args = ap.parse_args()

    if not (args.auto or args.niche or args.brief):
        ap.error("provide one of --auto, --niche <id>, or --brief <text>")

    # Resolve --model-tag: accept a registry key OR a backend model id, and FAIL
    # LOUDLY on an unknown value instead of silently routing to Ollama → all-
    # Cydonia output (the 2026-06-15 canary footgun). Registry-unreadable is
    # tolerated (DEFAULT_LLM_TAG already degrades gracefully there).
    from src.memory.llm_registry import LLMRegistryLoader, LLMNotFound
    try:
        args.model_tag = LLMRegistryLoader().resolve_model_tag(args.model_tag)
    except LLMNotFound as exc:
        ap.error(str(exc))          # wrong tag → loud CLI fail, not silent fallback
    except Exception:               # registry unreadable/malformed → graceful pass-through
        pass

    try:
        lo_s, hi_s = args.word_band.split("-")
        word_band = (int(lo_s), int(hi_s))
    except ValueError:
        ap.error("--word-band must be 'lo-hi', e.g. 110-160 or 200-300")

    cfg = yaml.safe_load((ROOT / "config/pipeline.yaml").read_text())
    cu = cfg["comfyui"]
    workflow_dir = ROOT / cu.get("workflow_dir", "config/comfyui_workflows")

    # Pre-flight: a render run with a dead ComfyUI previously burned the whole
    # LLM phase (and consumed niche-cycle state) to produce zero images.
    if not args.prompts_only and not _comfyui_up(cu["base_url"]):
        sys.exit(f"PRE-FLIGHT ABORT: ComfyUI unreachable at {cu['base_url']} — "
                 f"start it (or use --prompts-only) and re-run.")
    # Release ComfyUI's resident models (~32 GB) BEFORE Phase 1 loads the LLM, so
    # back-to-back series start with a clean slate and the LLM never OOM-kills
    # ComfyUI by stacking on top of the previous render's models (2026-06-15).
    if not args.prompts_only:
        freed = _comfyui_free(cu["base_url"])
        print(f"  (pre-flight: ComfyUI memory {'freed' if freed else 'free request failed — proceeding'})",
              flush=True)

    # Engine selection: a convenience that swaps the base + refine + refine_T4
    # templates as a SET so --engine zimage carries its own detailer stage (incl.
    # the T4 genital detailer). Explicit --base-template / --refine-template still
    # win (None is ignored by resolve_render_pipeline, so chroma falls through to
    # the pipeline.yaml defaults).
    _ENGINE_TEMPLATES = {
        "zimage": {
            "base_template": "templates/zimage/base.json",
            "refine_template": "templates/zimage/refine.json",
            "refine_template_t4": "templates/zimage/refine_T4.json",
        },
    }
    # --hires: gonzaLomo deep-shrink base at a higher per-orientation resolution
    # (Z-Image only — deep-shrink is a Z-Image feature; ~2x slower on MPS).
    _ZIMAGE_HIRES_BASE = "templates/zimage/base_hires.json"
    _HIRES_BASE_RESOLUTION = {"portrait": [1216, 1536], "square": [1344, 1344],
                              "landscape": [1536, 1216]}
    if args.hires and args.engine != "zimage":
        print("  (--hires forces --engine zimage: deep-shrink is a Z-Image feature)",
              flush=True)
        args.engine = "zimage"
    eng = _ENGINE_TEMPLATES.get(args.engine, {})
    base_tmpl_cli = args.base_template or (
        _ZIMAGE_HIRES_BASE if args.hires else eng.get("base_template"))
    rp_cli = {
        "base_template": base_tmpl_cli,
        "refine_template": args.refine_template or eng.get("refine_template"),
        "refine_template_t4": eng.get("refine_template_t4"),
    }
    if args.hires:
        rp_cli["base_resolution"] = _HIRES_BASE_RESOLUTION
    # Refine is OPT-IN (base-only default). --refine turns it on; --no-refine is
    # the (now-default) off state, kept for back-compat. --refine wins if both given.
    if args.refine:
        rp_cli["enable_refine"] = True
    elif args.no_refine:
        rp_cli["enable_refine"] = False
    rp = resolve_render_pipeline(cfg.get("render_pipeline"), None, rp_cli)
    # Base renders at the model's native resolution (render_pipeline.base_resolution
    # — portrait 896×1152); true 4K is reached only in the separate manual upscale
    # stage (scripts/upscale_folder.py).
    resolution = base_resolution_for(rp, args.orientation)

    # ── Resolve brief + sub-looks via the niche selector (unless --brief) ──
    selection = None
    sub_looks = None
    brief = args.brief
    used_niches: list[str] = []
    auto_niche = not args.niche  # --auto / no explicit --niche → use the cycle
    if not args.brief:
        try:
            library = NicheLibrary.from_yaml()
            niche_cursor = _read_niche_cursor()
            chosen_niche_id = args.niche
            if auto_niche:
                # Exhaust EVERY tier-supporting niche before repeating any.
                used_niches = _read_used_niches()
                picked, cycle_reset = select_niche_cycle(
                    library, used_niches, tier=args.tier)
                if cycle_reset:
                    used_niches = []  # wrapped — start a fresh rotation
                    print("=== niche cycle complete — starting a fresh "
                          "rotation ===", flush=True)
                chosen_niche_id = picked.id
            selection = build_selection(
                library, niche_cursor, tier=args.tier,
                force_niche=chosen_niche_id,
                persona=args.persona or bool(args.persona_name),
                persona_name=args.persona_name,
            )
        except NicheLibraryError as exc:
            ap.error(f"niche selection failed: {exc}")
        brief = build_brief(selection)
        sub_looks = selection.sub_looks
        # NOTE: cursor/used-niche state is committed AFTER prompts exist (below)
        # — previously a run that failed in Phase 1 still consumed its
        # niche-cycle slot and produced nothing.
        print(f"=== niche: {selection.niche.id} ({selection.niche.niche_class}) "
              f"| folder={selection.niche.da_folder!r} "
              f"| persona={selection.persona.name if selection.persona else 'none'} "
              f"| tier={args.tier} ===", flush=True)
    # Seeds: RANDOM per render by default; an explicit --base-seed switches to a
    # deterministic, reproducible layout. The deterministic layout below is only
    # consulted when random_seeds is False (the render stage draws random otherwise).
    random_seeds = args.base_seed is None
    base_seed = 0 if random_seeds else args.base_seed
    is_explicit = args.tier in _EXPLICIT_TIERS
    n_covers = args.covers if (is_explicit and not args.no_package) else 0
    cover_seeds = 1
    # Deterministic seed layout (per-prompt seeds): [main: count*seeds][covers]
    # [main anatomy-rerolls][cover anatomy-rerolls] — non-overlapping spans so a
    # reroll never collides with a cover.
    main_span = args.count * args.seeds
    cover_base = base_seed + main_span
    cover_end = cover_base + n_covers * cover_seeds
    main_reroll_start = cover_end
    cover_reroll_start = cover_end + main_span * args.anatomy_retries
    if random_seeds:
        print("=== seeds: RANDOM per render (reproduce a specific run with "
              "--base-seed N) ===", flush=True)
    else:
        print(f"=== base_seed {base_seed} (deterministic; main {base_seed}.."
              f"{cover_base - 1}"
              f"{f'; covers {cover_base}..{cover_end - 1}' if n_covers else ''}) ===",
              flush=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (
        ROOT / "output" / "art_series" / ts
    )
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: ALL LLM work (prompts + SFW covers + metadata) ────────
    # Done in one LLM session so we unload exactly once before rendering
    # (LLM and ComfyUI never co-resident). Metadata is set-level, so it can
    # be generated up-front from the niche/brief — no post-render reload.
    print(f"\n=== Phase 1 (LLM): {args.count} prompts"
          f"{f' + {n_covers} SFW covers' if n_covers else ''} + metadata via "
          f"{args.model_tag} ===", flush=True)
    _ensure_llm_loaded(args.model_tag)   # large context — JIT default 4096 truncates T4 prompts
    # Cross-series variety: seed the anti-repetition lists with prior same-niche
    # prompts + offset the per-run rotations, so re-running a niche yields a
    # DISTINCT series instead of reproducing it. --brief runs use a slug of the
    # brief as the memory key (previously they had NO memory and shipped
    # near-duplicate series three runs in a row).
    _niche_id = selection.niche.id if selection else None
    if _niche_id is None and args.brief:
        _niche_id = ("brief-"
                     + re.sub(r"[^a-z0-9]+", "-", args.brief.lower()).strip("-")[:40])
    seed_banned, seed_avoid, prior_count, overused = [], [], 0, []
    if _niche_id:
        seed_banned, seed_avoid, prior_count, overused = _load_niche_history(_niche_id)
        if prior_count:
            print(f"  (cross-series variety: run #{prior_count + 1} of '{_niche_id}'"
                  f" — steering away from {len(seed_avoid)} prior fragments"
                  f"{f'; overused: {overused}' if overused else ''})", flush=True)
    # Cross-NICHE opener ban: openers from the last few runs of ANY niche
    # (the same opener template was shipping across 3 niches the same week).
    seed_banned = list(dict.fromkeys(seed_banned + _load_global_openers()))
    # Rotation offset strides past the whole previous run (count+1, not 1):
    # stride-1 re-issued 5/6 of the assignment tuples on every consecutive run.
    run_offset = prior_count * (args.count + 1)
    # A bound persona LOCKS the per-image look to one identity (the same woman in
    # every image of the series, replacing the look-pool rotation); non-persona
    # runs pass "" → rotation as before. None-safe for the --brief path.
    locked_look = (persona_locked_look(selection.persona)
                   if (selection and selection.persona) else "")
    rows = art_director.generate_series(
        brief=brief, tier=args.tier, count=args.count, model_tag=args.model_tag,
        temperature=args.temperature, sub_looks=sub_looks, word_band=word_band,
        audit_gate=not args.no_audit_gate,
        seed_avoid=seed_avoid, seed_banned_openers=seed_banned, run_offset=run_offset,
        seed_overused=overused, locked_look=locked_look,
        lock_wardrobe=(selection.niche.lock_wardrobe if selection else False),
    )
    if not rows:
        print("No prompts generated — aborting.", file=sys.stderr)
        return 1

    # --prompts-only: stop after prompt generation (no render). Saves the rows
    # (prompt + orientation/shot_type/framing + audit_score) for analysis/A-B.
    if args.prompts_only:
        out_dir.mkdir(parents=True, exist_ok=True)
        from collections import Counter
        oris = Counter(r.get("orientation") for r in rows)
        shots = Counter(r.get("shot_type") for r in rows)
        scores = [r.get("audit_score") for r in rows if r.get("audit_score") is not None]
        payload = {"brief": brief, "tier": args.tier, "model_tag": args.model_tag,
                   "niche": selection.niche.id if selection else None,
                   "orientations": dict(oris), "shot_types": dict(shots),
                   "mean_audit": round(sum(scores) / len(scores), 2) if scores else None,
                   "prompts": rows}
        (out_dir / "prompts.json").write_text(json.dumps(payload, indent=2))
        print(f"\n=== PROMPTS ONLY (no render) — {len(rows)} prompts ===", flush=True)
        print(f"  orientations: {dict(oris)}  shot_types: {dict(shots)}", flush=True)
        print(f"  mean audit: {payload['mean_audit']}  -> {out_dir / 'prompts.json'}",
              flush=True)
        return 0

    # ── State commit (post-prompts): the run has real output now, so the
    # niche-cycle slot may be consumed. prompts-only runs returned above and
    # never mutate rotation state.
    if not args.brief and selection is not None:
        _advance_niche_cursor(niche_cursor)
        if auto_niche:  # only --auto tracks the used-niche cycle
            _record_used_niche(used_niches, selection.niche.id)

    # niche_meta built early so the post-render manifest checkpoint has it.
    niche_meta = None
    if selection is not None:
        niche_meta = {
            "id": selection.niche.id, "class": selection.niche.niche_class,
            "da_folder": selection.niche.da_folder, "tags": selection.niche.tags,
            "family": selection.niche.family or None,
            "aesthetic_lock": {
                "palette": selection.aesthetic_lock.palette,
                "lighting": selection.aesthetic_lock.lighting,
                "photographer": selection.aesthetic_lock.photographer,
            },
            "persona": selection.persona.name if selection.persona else None,
        }
    elif _niche_id:
        # --brief run: the slug is the cross-run memory key so future runs of
        # the same brief inherit banned openers / signatures / rotation offset.
        niche_meta = {"id": _niche_id, "kind": "brief"}

    cover_rows: list[dict] = []
    if n_covers:
        print(f"\n=== Phase 1b (LLM): {n_covers} SFW (T1) cover prompts ===",
              flush=True)
        # Covers are the DA shopfront — they previously got NO memory and
        # cloned main scenes 0-1's assignments (cycle-wide, every niche's first
        # cover converged on the same look+framing). They now inherit the full
        # avoid lists EXTENDED with this run's accepted prompts, and rotate
        # from a window past the main scenes.
        cover_banned = seed_banned + [art_director._opener(r["prompt"]) for r in rows]
        cover_avoid = (seed_avoid
                       + [art_director._signature(r["prompt"]) for r in rows]
                       + [art_director._tail_signature(r["prompt"]) for r in rows])
        cover_rows = art_director.generate_series(
            brief=brief, tier="T1_suggestive", count=n_covers,
            model_tag=args.model_tag, temperature=args.temperature,
            sub_looks=sub_looks, word_band=word_band,
            audit_gate=not args.no_audit_gate,
            seed_avoid=cover_avoid, seed_banned_openers=cover_banned,
            run_offset=run_offset + args.count,
            seed_overused=overused, locked_look=locked_look,
            require_sfw=True,
            extra_directive=art_director.SFW_COVER_DIRECTIVE,
        )

    gated_meta = public_meta = None
    if not args.no_package:
        print("\n=== Phase 1c (LLM): set metadata ===", flush=True)
        gated_meta = _gen_metadata(selection, brief, args.tier, args.count,
                                   args.model_tag)
        public_meta = (
            _gen_metadata(selection, brief, "T2_implied", max(n_covers, 1),
                          args.model_tag)
            if is_explicit else gated_meta
        )

    # ── unload LLM before rendering ────────────────────────────────────
    print("\n=== Unloading LLM before render phase ===", flush=True)
    _unload_llm(args.model_tag)

    # ── Phase 2: render ────────────────────────────────────────────────
    # Default = STAGED: base (Chroma) for ALL prompts, then refine (SDXL) for
    # ALL base outputs → review images. The two model domains never co-reside
    # (no swap). 4K is NEVER auto-run here — it is a manual keepers-only step
    # via scripts/upscale_folder.py. Detailers + 4K live in the upscale stage
    # (proven detail-after-upscale ordering), so the series render is
    # tier-NEUTRAL.
    builder = WorkflowBuilder(workflow_dir)
    client = ComfyUIClient(base_url=cu["base_url"], output_dir=cu["output_dir"])
    # Anatomy guard: the hand detector lives under the ComfyUI models dir
    # (<comfyui_root>/models/ultralytics/bbox/hand_yolov9c.pt). output_dir is
    # <comfyui_root>/output, so its parent is the root.
    hand_det_path = str(Path(os.path.expanduser(cu["output_dir"])).parent
                        / "models/ultralytics/bbox/hand_yolov9c.pt")
    # Render-level tier-drift screen for PUBLIC-bound images (same detectors
    # the refine detailers use; missing files → graceful no-op).
    _ultra = Path(os.path.expanduser(cu["output_dir"])).parent / "models/ultralytics/bbox"
    nudity_det_paths = tuple(str(_ultra / f) for f in
                             ("nipples_yolov8s.pt", "vagina-v3.2.pt"))
    base_dir = out_dir / "base"
    cover_dir = out_dir / "covers"
    cover_manifest: list[dict] = []

    base_tmpl = rp["base_template"]
    # T4-ONLY: explicit main images get the refine variant with the vagina
    # detailer; SFW covers (and T1/T2/T3) use the base refine so a tasteful
    # nude (or a public teaser) never has its genitals detailed (tier purity).
    refine_tmpl_main, refine_cover_tmpl = _refine_templates_for(args.tier, rp)
    # Tier-purity guard (content-based): refuse to render explicit genital
    # detailing below T4 — catches a --refine-template override that injects a
    # T4 template into a lower tier. Covers always use the base refine (safe).
    if _violates_tier_purity(args.tier, refine_tmpl_main, workflow_dir):
        sys.exit(f"TIER-PURITY ABORT: refine template '{refine_tmpl_main}' has a "
                 f"genital detailer but --tier {args.tier} (not T4_explicit). "
                 f"Explicit detailing is T4-only.")
    enable_refine = rp.get("enable_refine", True)
    run_template = {"base": base_tmpl, "refine": refine_tmpl_main,
                    "upscale": rp["upscale_template"]}
    print(f"\n=== Phase 2a (base, {args.engine}): {Path(base_tmpl).name} ===",
          flush=True)
    manifest = _render_stage_base(
        rows, builder=builder, client=client, base_template=base_tmpl,
        negative=DEFAULT_NEGATIVE, resolution=resolution, base_seed=base_seed,
        seeds=args.seeds, dest_dir=base_dir, out_dir=out_dir, prefix="ad",
        rp=rp, default_orientation=args.orientation,
        anatomy_retries=args.anatomy_retries, hand_detector_path=hand_det_path,
        reroll_start=main_reroll_start, random_seeds=random_seeds,
        nsfw_lora_strength=(_NSFW_LORA_STRENGTH if args.tier in _EXPLICIT_TIERS else 0.0))
    if cover_rows:
        cover_manifest = _render_stage_base(
            cover_rows, builder=builder, client=client, base_template=base_tmpl,
            negative=DEFAULT_NEGATIVE, resolution=resolution,
            base_seed=cover_base, seeds=cover_seeds,
            dest_dir=base_dir, out_dir=out_dir, prefix="cover",
            rp=rp, default_orientation=args.orientation,
            anatomy_retries=args.anatomy_retries, hand_detector_path=hand_det_path,
            reroll_start=cover_reroll_start, random_seeds=random_seeds,
            nsfw_lora_strength=0.0)   # covers are SFW T1
    if enable_refine:
        print(f"\n=== Phase 2b (refine, SDXL): {Path(refine_tmpl_main).name} "
              f"→ review images ===", flush=True)
        _render_stage_refine(manifest, builder=builder, client=client,
                             refine_template=refine_tmpl_main, dest_dir=img_dir,
                             out_dir=out_dir)
        if cover_manifest:   # covers are SFW → always the base refine
            _render_stage_refine(cover_manifest, builder=builder, client=client,
                                 refine_template=refine_cover_tmpl, dest_dir=cover_dir,
                                 out_dir=out_dir)
    else:
        print("\n=== Phase 2b skipped (--no-refine): review = raw base ===",
              flush=True)
        for m in (manifest, cover_manifest):
            for e in m:
                for im in e["images"]:
                    im["path"] = im.get("base_path")
                    im["review_path"] = im["path"]

    # ── Post-render checkpoint ──────────────────────────────────────────
    # Manifest commits RIGHT AFTER rendering: a crash in curation/packaging
    # previously lost manifest.json (the cross-run anti-repetition memory!)
    # even though every image existed on disk. (Seeds are random per render now,
    # so there is no counter to advance — each seed is logged per image.)
    def _write_manifest(pkg_path: "Path | None" = None) -> None:
        (out_dir / "manifest.json").write_text(json.dumps({
            "brief": brief, "tier": args.tier, "template": run_template,
            "template_sha256": _template_hashes(workflow_dir, run_template),
            "model_tag": args.model_tag, "orientation": args.orientation,
            "resolution": resolution, "seeds_per_prompt": args.seeds,
            "base_seed": (None if random_seeds else base_seed),
            "random_seeds": random_seeds, "word_band": list(word_band),
            "niche": niche_meta,
            "package": str(pkg_path.relative_to(out_dir)) if pkg_path else None,
            "prompts": manifest, "covers": cover_manifest,
        }, indent=2))

    _write_manifest()

    # ── Phase 3: curation (ImageScorer triage) ─────────────────────────
    scoring_cfg = cfg.get("scoring", {})
    use_hps = bool(scoring_cfg.get("use_hps_v2", False))
    use_ir = bool(scoring_cfg.get("use_image_reward", False))
    if not args.no_curate:
        print("\n=== Phase 3: curating (ImageScorer) ===", flush=True)
        _curate(manifest, out_dir, content_level=args.tier,
                keep_top=args.keep_top, use_hps_v2=use_hps, use_image_reward=use_ir)
        if cover_manifest:
            # Distinct sheet name so the covers pass doesn't clobber the main
            # keepers' contact_sheet.png (the QA montage the checklist references).
            _curate(cover_manifest, out_dir, content_level="T1_suggestive",
                    keep_top=1, use_hps_v2=use_hps, use_image_reward=use_ir,
                    contact_sheet_name="contact_sheet_covers.png")
    else:  # keep all so packaging has candidates
        for m in (manifest, cover_manifest):
            for e in m:
                for im in e["images"]:
                    im["keeper"] = True

    # ── Phase 4: tier-split packaging (wrapped — every sibling phase is;
    # a packaging OSError previously killed the run post-render) ────────
    pkg_dir = None
    if not args.no_package:
        print("\n=== Phase 4: packaging ===", flush=True)
        try:
            pg_cfg = dict(cfg.get("postgrade", {}) or {})
            if args.no_postgrade:
                pg_cfg["enabled"] = False
            pkg_dir = _package(out_dir, selection, args.tier, manifest,
                               cover_manifest, gated_meta, public_meta,
                               cfg.get("watermark", {}),
                               nudity_detector_paths=nudity_det_paths,
                               postgrade_cfg=pg_cfg)
        except Exception as exc:  # noqa: BLE001 — images + manifest are safe
            print(f"  !! packaging FAILED ({exc}) — images + manifest are intact; "
                  f"re-package manually from {out_dir}", file=sys.stderr, flush=True)

    # Final manifest rewrite — now carries curation + packaging results.
    _write_manifest(pkg_dir)

    n_img = sum(len(e["images"]) for e in manifest) + \
        sum(len(e["images"]) for e in cover_manifest)
    print(f"\nDONE — {len(rows)} prompts, {n_img} images -> {out_dir}", flush=True)
    if pkg_dir:
        print(f"Publish-ready package: {pkg_dir} "
              f"(see POSTING_CHECKLIST.md). Upload is manual.", flush=True)

    # Auto-free ComfyUI's resident model the moment the series finishes (2026-06-21)
    # — not just at the next run's pre-flight — so memory drops right away. ComfyUI
    # stays UP and reloads the model on the next render.
    freed = _comfyui_free(cu["base_url"])
    print(f"  (post-run: ComfyUI memory {'freed' if freed else 'free request failed'})",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
