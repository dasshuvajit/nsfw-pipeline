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
    select_niche_cycle,
)

DEFAULT_TEMPLATE = "templates/chroma/gonzaLomo_Chroma_4K_v12.json"
# T4 variant: base v12 chain + tier-gated NSFW-region detailers (nipples,
# vagina). Auto-selected for T4_explicit main images only; SFW covers and
# T3-and-below stay on the base template (their detectors would not fire and
# NSFW-region inpainting is unwanted there).
T4_TEMPLATE = "templates/chroma/gonzaLomo_Chroma_4K_v12_T4.json"

# Persistent seed counter — guarantees every render across runs gets a
# never-before-used seed, defeating ComfyUI's per-node cache collisions
# (which hit us on the seed-9 retry: cache key matched a prior submission).
SEED_COUNTER_FILE = ROOT / "output/art_series/.last_seed"
LEGACY_DEFAULT_SEED = 7

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

# gonzalomo_chroma_v30 base resolution presets. These feed the v12 4K
# template, whose chain is base ->(ImageScaleBy 1.25)-> SDXL refine
# ->(UltimateSDUpscale 2.0)-> detailers. Final long edge = base * 1.25 * 2.0.
# Portrait/landscape bases give a TRUE-4K (>=3840px) long edge; square at
# 1024 yields 2560px (raise to 1536 for a true-4K square at higher base cost).
ORIENTATIONS = {
    "portrait": (1024, 1536),   # -> 2560 x 3840  (true 4K)
    "square": (1024, 1024),     # -> 2560 x 2560  (2.5K; see note above)
    "landscape": (1536, 1024),  # -> 3840 x 2560  (true 4K)
}

# Self-contained default negative (age-safety + multi-subject + anatomy +
# render-risk). Kept here so the production flow does not depend on the DB.
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


def _read_seed_counter() -> int | None:
    """Return the highest seed used by any prior art_series run, or None."""
    try:
        return int(SEED_COUNTER_FILE.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def _next_base_seed(explicit: int | None) -> int:
    """Pick the base seed for this run. Explicit --base-seed overrides; else
    advance one past the persisted counter; else fall back to the legacy
    default."""
    if explicit is not None:
        return explicit
    prior = _read_seed_counter()
    if prior is None:
        return LEGACY_DEFAULT_SEED
    return prior + 1


def _record_max_seed(seed: int) -> None:
    """Write back the highest seed used so the next run advances past it."""
    SEED_COUNTER_FILE.parent.mkdir(parents=True, exist_ok=True)
    prior = _read_seed_counter() or -1
    SEED_COUNTER_FILE.write_text(str(max(prior, seed)))


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


def _unload_llm(model_tag: str) -> None:
    """Free the LLM before the render phase (never co-resident with ComfyUI).
    Cascades across every backend (Ollama + LM Studio + MLX) via the pool, so
    an LM-Studio-resident Gemma is freed too — not just Ollama models. Then a
    direct `ollama stop` belt-and-braces for the Ollama path. Best-effort + grace."""
    try:
        from src.agents.llm_client import LLMClientPool
        LLMClientPool().unload_all()
    except Exception:  # noqa: BLE001
        pass
    try:
        subprocess.run(["ollama", "stop", model_tag], timeout=30,
                       capture_output=True)
    except Exception:  # noqa: BLE001
        pass
    time.sleep(3)


# Curation reject flags that drop a candidate from keeper contention
# regardless of score (composition-fatal, not taste).
_HARD_REJECT_FLAGS = {"multiple_faces", "no_face", "scorer_error"}


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
AI_DISCLOSURE_LABEL = "Created using AI tools"


def _render_rows(
    rows: list[dict], *, builder, client, template: str, negative: str,
    resolution: tuple[int, int], base_seed: int, seeds: int,
    dest_dir: Path, out_dir: Path, prefix: str,
) -> list[dict]:
    """Render every (prompt × seed) into dest_dir; return a manifest list
    of {index, look, prompt, audit_score, images:[{path, seed}]}."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for idx, r in enumerate(rows):
        look = r["look"].split()[0]
        entry = {"index": idx, "look": r["look"], "prompt": r["prompt"],
                 "audit_score": r.get("audit_score"), "images": []}
        for k in range(seeds):
            seed = base_seed + k
            try:
                wf = builder.build_external(
                    external_template=template, prompt_text=r["prompt"],
                    negative_prompt=negative, resolution=resolution, seed=seed)
                # v12 4K chain (base + SDXL refine + UltimateSDUpscale to
                # 3840px + 3 detailers) runs ~8-15 min/img on the M4 Pro;
                # 1800s leaves headroom so a slow render is not killed mid-pass.
                images = client.render_single_with_retry(wf, timeout=1800)
                outs = [im for im in images if im.type == "output"] or images
                name = f"{prefix}{idx + 1:02d}_{look}_s{seed}.png"
                dst = dest_dir / name
                shutil.copy(outs[-1].file_path, dst)
                entry["images"].append(
                    {"path": str(dst.relative_to(out_dir)), "seed": seed})
                print(f"  [{prefix} {idx + 1}/{len(rows)}] seed {seed} -> {name}",
                      flush=True)
            except Exception as exc:  # noqa: BLE001 — surface, continue
                print(f"  [{prefix} {idx + 1}/{len(rows)}] seed {seed} FAILED: {exc}",
                      file=sys.stderr, flush=True)
        manifest.append(entry)
    return manifest


def _render_stage_base(
    rows: list[dict], *, builder, client, base_template: str, negative: str,
    resolution: tuple[int, int], base_seed: int, seeds: int,
    dest_dir: Path, out_dir: Path, prefix: str,
    rp: dict | None = None, default_orientation: str = "portrait",
) -> list[dict]:
    """Stage 1 (Chroma): base gen for every (prompt × seed) into dest_dir.

    Each prompt is rendered at ITS OWN orientation (the LLM chose it per prompt
    via art_director) → portrait/square/landscape variety instead of one fixed
    ratio. ``rp`` is the render_pipeline config (base_resolution per orientation);
    falls back to ``resolution`` / ``default_orientation`` when absent.

    Manifest images carry ``{base_path, seed}``; ``path`` (what curation +
    packaging read) is set later by the refine stage. Chroma is loaded ONCE
    by ComfyUI and stays resident across all base submissions — SDXL is not
    touched here, so this whole batch runs without the monolith's Chroma+SDXL
    co-residence (and its swap)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
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
            seed = base_seed + k
            try:
                wf = builder.build_external(
                    external_template=base_template, prompt_text=r["prompt"],
                    negative_prompt=negative, resolution=res, seed=seed)
                images = client.render_single_with_retry(wf, timeout=1800)
                outs = [im for im in images if im.type == "output"] or images
                name = f"{prefix}{idx + 1:02d}_{look}_s{seed}.png"
                dst = dest_dir / name
                shutil.copy(outs[-1].file_path, dst)
                entry["images"].append(
                    {"base_path": str(dst.relative_to(out_dir)), "seed": seed})
                print(f"  [base {idx + 1}/{len(rows)}] {orientation} {res[0]}x{res[1]}"
                      f" seed {seed} -> {name}", flush=True)
            except Exception as exc:  # noqa: BLE001 — surface, continue
                print(f"  [base {idx + 1}/{len(rows)}] seed {seed} FAILED: {exc}",
                      file=sys.stderr, flush=True)
        manifest.append(entry)
    return manifest


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
                images = client.render_single_with_retry(wf, timeout=1800)
                outs = [i for i in images if i.type == "output"] or images
                name = Path(base_rel).name  # mirror base filename under images/
                dst = dest_dir / name
                shutil.copy(outs[-1].file_path, dst)
                im["path"] = str(dst.relative_to(out_dir))
                im["review_path"] = im["path"]
                print(f"  [refine] {name}", flush=True)
            except Exception as exc:  # noqa: BLE001 — surface, fall back to base
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
        from src.agents.llm_client import OllamaClient
        meta = MetadataGenerator(OllamaClient()).generate(
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
    groups = ", ".join(metadata.get("da_groups", []))
    ai = metadata["labels"]["ai_tools"]

    def _write(images: list[str], meta: dict, is_gated: bool) -> None:
        base = meta.get("title") or folder
        if persona:
            base = f"{persona} — {base}"
        price = _PRICE_BY_TIER.get(tier if is_gated else "T1_suggestive", "$2-3")
        where = "gated" if is_gated else "public"
        note = "CLEAN file (no watermark)" if is_gated else "watermarked SFW teaser"
        for n, img in enumerate(images, 1):
            txt = (
                f"TITLE: {base} {n}\n"
                f"FOLDER: {folder}"
                f"{'  [Premium Gallery / Subscription]' if is_gated else '  [public SFW]'}\n"
                f"DESCRIPTION: {meta.get('description', '')}\n"
                f"TAGS: {', '.join(meta.get('tags', []))}\n"
                f"GROUPS: {groups}\n"
                f"MATURE: yes\n"
                f"AI-LABEL: {ai}\n"
                f"PRICE: {price}\n"
                f"FILE: {where}/{img}  ({note})\n"
            )
            (tdir / f"{Path(img).stem}.txt").write_text(txt)

    _write(metadata["public"]["images"], metadata["public"]["metadata"], False)
    if is_explicit:
        _write(metadata["gated"]["images"], metadata["gated"]["metadata"], True)


def _keepers_of(manifest: list[dict], out_dir: Path) -> list[Path]:
    return [out_dir / im["path"] for e in manifest for im in e["images"]
            if im.get("keeper")]


def _package(
    out_dir: Path, selection, tier: str,
    main_manifest: list[dict], cover_manifest: list[dict],
    gated_meta: dict, public_meta: dict, watermark_cfg: dict,
) -> Path:
    """Assemble a publish-ready package with a tier-split public/gated layout.

    HARD RULE (DA SFW-shopfront ToS): the public folder + cover NEVER contain a
    T3/T4 image. For explicit runs the public set is sourced ONLY from the
    dedicated SFW (T1) cover renders; the explicit keepers go to gated/."""
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

    def _copy_into(paths: list[Path], dest: Path) -> list[str]:
        names = []
        for p in paths:
            if p.exists():
                shutil.copy(p, dest / p.name)
                names.append(p.name)
        return names

    if is_explicit:
        gated_names = _copy_into(main_keepers, gated_dir)
        public_names = _copy_into(cover_keepers, public_dir)
        public_tier = "T1_suggestive"
    else:
        public_names = _copy_into(main_keepers, public_dir)
        gated_names = []
        public_tier = tier

    cover_name = public_names[0] if public_names else None  # SFW by construction

    # Watermark the PUBLIC teasers (branding/funnel); gated stays clean for buyers.
    try:
        from src.postprocess.watermarker import Watermarker
        wm = Watermarker(watermark_cfg)
        for p in sorted(public_dir.glob("*.png")):
            wm.apply(p, p, content_level=public_tier)
    except Exception as exc:  # noqa: BLE001
        print(f"  (watermark skipped: {exc})", file=sys.stderr, flush=True)

    metadata = {
        "da_folder": folder,
        "tier": tier,
        "niche": selection.niche.id if selection else None,
        "persona": selection.persona.name if (selection and selection.persona) else None,
        "labels": {"ai_tools": AI_DISCLOSURE_LABEL, "mature": True},
        "price_by_tier": _PRICE_BY_TIER.get(tier),
        "da_groups": _suggested_groups(selection.niche.tags if selection else []),
        "posting_strategy": ("public SFW teasers → Premium Gallery / Subscription"
                             if is_explicit else "public SFW set (T1/T2)"),
        "watermark_status": "public watermarked; gated clean",
        "cover_image": cover_name,
        "public": {"count": len(public_names), "images": public_names,
                   "metadata": public_meta},
        "gated": {"count": len(gated_names), "images": gated_names,
                  "metadata": gated_meta},
    }
    (pkg / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (pkg / "POSTING_CHECKLIST.md").write_text(_posting_checklist(metadata, is_explicit))
    _emit_posting_templates(pkg, metadata, tier, is_explicit)
    print(f"  packaged: {pkg}  (public={len(public_names)}, gated={len(gated_names)}, "
          f"cover={cover_name}, +posting_templates/)", flush=True)
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
        f"**4K-finish heroes:** `python scripts/upscale_folder.py <selected folder>`  ·  "
        "**Per-image copy-paste:** `posting_templates/`",
        "",
        "## Hard rules (verified ToS, do not skip)",
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
    ap.add_argument("--orientation", default="portrait",
                    choices=list(ORIENTATIONS))
    ap.add_argument("--template", default=None,
                    help="MONOLITH escape hatch: run a single-pass v12-style "
                    "template (e.g. gonzaLomo_Chroma_4K_v12.json) instead of the "
                    "default staged base+refine pipeline")
    ap.add_argument("--base-template", default=None,
                    help="staged stage-1 base template (default: pipeline.yaml "
                    "render_pipeline.base_template)")
    ap.add_argument("--refine-template", default=None,
                    help="staged stage-2 refine template (default: pipeline.yaml "
                    "render_pipeline.refine_template)")
    ap.add_argument("--no-refine", action="store_true",
                    help="skip stage-2 refine; review images = raw base render")
    ap.add_argument("--model-tag", default=art_director.DEFAULT_LLM_TAG,
                    help="LLM tag for prompt generation; routed to its backend "
                         "(LM Studio / Ollama / MLX) by config/llm_models.yaml. "
                         "Default = registry default_llm (Gemma, LM Studio). "
                         "Pass art_director.CYDONIA_TAG for the Ollama/Cydonia path.")
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--word-band", default="110-160",
                    help="prose word band 'lo-hi' (A/B: try 200-300)")
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
                    help="explicit base seed; default = auto-advance past the "
                    "highest seed any prior run used (output/art_series/.last_seed)")
    ap.add_argument("--out-dir", default="",
                    help="output dir (default: output/art_series/<timestamp>)")
    args = ap.parse_args()

    if not (args.auto or args.niche or args.brief):
        ap.error("provide one of --auto, --niche <id>, or --brief <text>")

    try:
        lo_s, hi_s = args.word_band.split("-")
        word_band = (int(lo_s), int(hi_s))
    except ValueError:
        ap.error("--word-band must be 'lo-hi', e.g. 110-160 or 200-300")

    cfg = yaml.safe_load((ROOT / "config/pipeline.yaml").read_text())
    cu = cfg["comfyui"]
    workflow_dir = ROOT / cu.get("workflow_dir", "config/comfyui_workflows")

    # Staged pipeline is the default; --template pins the monolith escape hatch.
    staged = args.template is None
    rp_cli = {"base_template": args.base_template,
              "refine_template": args.refine_template}
    if args.no_refine:
        rp_cli["enable_refine"] = False
    rp = resolve_render_pipeline(cfg.get("render_pipeline"), None, rp_cli)
    # Monolith expects its 4K-tuned ORIENTATIONS (portrait 1024×1536); staged
    # uses render_pipeline.base_resolution (portrait reverted to native 896×1152,
    # 4K reached in the separate manual upscale stage).
    resolution = (ORIENTATIONS[args.orientation] if not staged
                  else base_resolution_for(rp, args.orientation))

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
        _advance_niche_cursor(niche_cursor)
        if auto_niche:  # only --auto tracks the used-niche cycle
            _record_used_niche(used_niches, selection.niche.id)
        print(f"=== niche: {selection.niche.id} ({selection.niche.niche_class}) "
              f"| folder={selection.niche.da_folder!r} "
              f"| persona={selection.persona.name if selection.persona else 'none'} "
              f"| tier={args.tier} ===", flush=True)
    base_seed = _next_base_seed(args.base_seed)
    is_explicit = args.tier in _EXPLICIT_TIERS
    n_covers = args.covers if (is_explicit and not args.no_package) else 0
    cover_seeds = 1
    main_span = args.count * args.seeds
    cover_base = base_seed + main_span         # covers use non-overlapping seeds
    max_seed = cover_base + (n_covers * cover_seeds) - 1 if n_covers else \
        base_seed + args.seeds - 1
    print(f"=== base_seed {base_seed} (main seeds {base_seed}..{base_seed + args.seeds - 1}"
          f"{f'; cover seeds {cover_base}..{cover_base + n_covers - 1}' if n_covers else ''}) ===",
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
    rows = art_director.generate_series(
        brief=brief, tier=args.tier, count=args.count, model_tag=args.model_tag,
        temperature=args.temperature, sub_looks=sub_looks, word_band=word_band,
        audit_gate=not args.no_audit_gate,
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

    cover_rows: list[dict] = []
    if n_covers:
        print(f"\n=== Phase 1b (LLM): {n_covers} SFW (T1) cover prompts ===",
              flush=True)
        cover_rows = art_director.generate_series(
            brief=brief, tier="T1_suggestive", count=n_covers,
            model_tag=args.model_tag, temperature=args.temperature,
            sub_looks=sub_looks, word_band=word_band,
            audit_gate=not args.no_audit_gate,
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
    # tier-NEUTRAL. --template pins the old monolith single-pass escape hatch.
    builder = WorkflowBuilder(workflow_dir)
    client = ComfyUIClient(base_url=cu["base_url"], output_dir=cu["output_dir"])
    base_dir = out_dir / "base"
    cover_dir = out_dir / "covers"
    cover_manifest: list[dict] = []
    run_template = args.template  # for the manifest record

    if staged:
        base_tmpl = rp["base_template"]
        refine_tmpl = rp["refine_template"]
        enable_refine = rp.get("enable_refine", True)
        run_template = {"base": base_tmpl, "refine": refine_tmpl,
                        "upscale": rp["upscale_template"]}
        print(f"\n=== Phase 2a (base, Chroma): {Path(base_tmpl).name} ===",
              flush=True)
        manifest = _render_stage_base(
            rows, builder=builder, client=client, base_template=base_tmpl,
            negative=DEFAULT_NEGATIVE, resolution=resolution, base_seed=base_seed,
            seeds=args.seeds, dest_dir=base_dir, out_dir=out_dir, prefix="ad",
            rp=rp, default_orientation=args.orientation)
        if cover_rows:
            cover_manifest = _render_stage_base(
                cover_rows, builder=builder, client=client, base_template=base_tmpl,
                negative=DEFAULT_NEGATIVE, resolution=resolution,
                base_seed=cover_base, seeds=cover_seeds,
                dest_dir=base_dir, out_dir=out_dir, prefix="cover",
                rp=rp, default_orientation=args.orientation)
        if enable_refine:
            print(f"\n=== Phase 2b (refine, SDXL): {Path(refine_tmpl).name} "
                  f"→ review images ===", flush=True)
            _render_stage_refine(manifest, builder=builder, client=client,
                                 refine_template=refine_tmpl, dest_dir=img_dir,
                                 out_dir=out_dir)
            if cover_manifest:
                _render_stage_refine(cover_manifest, builder=builder, client=client,
                                     refine_template=refine_tmpl, dest_dir=cover_dir,
                                     out_dir=out_dir)
        else:
            print("\n=== Phase 2b skipped (--no-refine): review = raw base ===",
                  flush=True)
            for m in (manifest, cover_manifest):
                for e in m:
                    for im in e["images"]:
                        im["path"] = im.get("base_path")
                        im["review_path"] = im["path"]
    else:
        # Monolith escape hatch — old single-pass via _render_rows.
        mono_res = ORIENTATIONS[args.orientation]
        print(f"\n=== Phase 2 (monolith): {Path(args.template).name} ===",
              flush=True)
        manifest = _render_rows(
            rows, builder=builder, client=client, template=args.template,
            negative=DEFAULT_NEGATIVE, resolution=mono_res, base_seed=base_seed,
            seeds=args.seeds, dest_dir=img_dir, out_dir=out_dir, prefix="ad")
        if cover_rows:
            cover_manifest = _render_rows(
                cover_rows, builder=builder, client=client, template=args.template,
                negative=DEFAULT_NEGATIVE, resolution=mono_res,
                base_seed=cover_base, seeds=cover_seeds,
                dest_dir=cover_dir, out_dir=out_dir, prefix="cover")

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

    # ── Phase 4: tier-split packaging ──────────────────────────────────
    pkg_dir = None
    if not args.no_package:
        print("\n=== Phase 4: packaging ===", flush=True)
        pkg_dir = _package(out_dir, selection, args.tier, manifest,
                           cover_manifest, gated_meta, public_meta,
                           cfg.get("watermark", {}))

    niche_meta = None
    if selection is not None:
        niche_meta = {
            "id": selection.niche.id, "class": selection.niche.niche_class,
            "da_folder": selection.niche.da_folder, "tags": selection.niche.tags,
            "aesthetic_lock": {
                "palette": selection.aesthetic_lock.palette,
                "lighting": selection.aesthetic_lock.lighting,
                "photographer": selection.aesthetic_lock.photographer,
            },
            "persona": selection.persona.name if selection.persona else None,
        }
    (out_dir / "manifest.json").write_text(json.dumps({
        "brief": brief, "tier": args.tier, "template": run_template,
        "model_tag": args.model_tag, "orientation": args.orientation,
        "resolution": resolution, "seeds_per_prompt": args.seeds,
        "base_seed": base_seed, "word_band": list(word_band),
        "niche": niche_meta,
        "package": str(pkg_dir.relative_to(out_dir)) if pkg_dir else None,
        "prompts": manifest, "covers": cover_manifest,
    }, indent=2))

    _record_max_seed(max_seed)

    n_img = sum(len(e["images"]) for e in manifest) + \
        sum(len(e["images"]) for e in cover_manifest)
    print(f"\nDONE — {len(rows)} prompts, {n_img} images -> {out_dir}", flush=True)
    if pkg_dir:
        print(f"Publish-ready package: {pkg_dir} "
              f"(see POSTING_CHECKLIST.md). Upload is manual.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
