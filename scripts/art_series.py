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
from src.niche.selector import (  # noqa: E402
    NicheLibrary, NicheLibraryError, build_selection, build_brief,
)

DEFAULT_TEMPLATE = "templates/chroma/gonzaLomo_Chroma_Refiner_v11.json"

# Persistent seed counter — guarantees every render across runs gets a
# never-before-used seed, defeating ComfyUI's per-node cache collisions
# (which hit us on the seed-9 retry: cache key matched a prior submission).
SEED_COUNTER_FILE = ROOT / "output/art_series/.last_seed"
LEGACY_DEFAULT_SEED = 7

# Persistent niche cursor — advances each --auto run so successive runs
# rotate through evergreen-core niches and periodically inject a trend
# niche (deterministic, no randomness; mirrors the seed counter pattern).
NICHE_CURSOR_FILE = ROOT / "output/art_series/.niche_cursor"

# gonzalomo_chroma_v30 resolution presets.
ORIENTATIONS = {
    "portrait": (896, 1152),
    "square": (1024, 1024),
    "landscape": (1152, 896),
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


def _unload_llm(model_tag: str) -> None:
    """Free the Ollama model before the render phase (never co-resident
    with ComfyUI). Best-effort: `ollama stop`, then a short grace period."""
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
        create_contact_sheet(cs, out_dir / "contact_sheet.png")
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
                images = client.render_single_with_retry(wf, timeout=480)
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
        "cover_image": cover_name,
        "public": {"count": len(public_names), "images": public_names,
                   "metadata": public_meta},
        "gated": {"count": len(gated_names), "images": gated_names,
                  "metadata": gated_meta},
    }
    (pkg / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (pkg / "POSTING_CHECKLIST.md").write_text(_posting_checklist(metadata, is_explicit))
    print(f"  packaged: {pkg}  (public={len(public_names)}, gated={len(gated_names)}, "
          f"cover={cover_name})", flush=True)
    return pkg


def _posting_checklist(meta: dict, is_explicit: bool) -> str:
    pub, gat = meta["public"], meta["gated"]
    folder = meta["da_folder"]
    lines = [
        f"# Posting checklist — {folder}",
        "",
        "> Generated by art_series.py. Upload is MANUAL and human-paced "
        "(bulk/automated posting is the #1 DeviantArt ban vector). Keep local "
        "masters — DeviantArt may delete content at its sole discretion.",
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
    ap.add_argument("--template", default=DEFAULT_TEMPLATE,
                    help="external ComfyUI template (relative to workflow_dir)")
    ap.add_argument("--model-tag", default=art_director.CYDONIA_TAG,
                    help="Ollama LLM tag for prompt generation")
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
    resolution = ORIENTATIONS[args.orientation]

    # ── Resolve brief + sub-looks via the niche selector (unless --brief) ──
    selection = None
    sub_looks = None
    brief = args.brief
    if not args.brief:
        try:
            library = NicheLibrary.from_yaml()
            niche_cursor = _read_niche_cursor()
            selection = build_selection(
                library, niche_cursor, tier=args.tier,
                force_niche=args.niche,
                persona=args.persona or bool(args.persona_name),
                persona_name=args.persona_name,
            )
        except NicheLibraryError as exc:
            ap.error(f"niche selection failed: {exc}")
        brief = build_brief(selection)
        sub_looks = selection.sub_looks
        _advance_niche_cursor(niche_cursor)
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

    cover_rows: list[dict] = []
    if n_covers:
        print(f"\n=== Phase 1b (LLM): {n_covers} SFW (T1) cover prompts ===",
              flush=True)
        cover_rows = art_director.generate_series(
            brief=brief, tier="T1_suggestive", count=n_covers,
            model_tag=args.model_tag, temperature=args.temperature,
            sub_looks=sub_looks, word_band=word_band,
            audit_gate=not args.no_audit_gate,
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

    # ── Phase 2: render (main + SFW covers) ────────────────────────────
    print(f"\n=== Phase 2: rendering via {Path(args.template).name} ===",
          flush=True)
    builder = WorkflowBuilder(workflow_dir)
    client = ComfyUIClient(base_url=cu["base_url"], output_dir=cu["output_dir"])
    manifest = _render_rows(
        rows, builder=builder, client=client, template=args.template,
        negative=DEFAULT_NEGATIVE, resolution=resolution, base_seed=base_seed,
        seeds=args.seeds, dest_dir=img_dir, out_dir=out_dir, prefix="ad")
    cover_manifest: list[dict] = []
    if cover_rows:
        cover_manifest = _render_rows(
            cover_rows, builder=builder, client=client, template=args.template,
            negative=DEFAULT_NEGATIVE, resolution=resolution,
            base_seed=cover_base, seeds=cover_seeds,
            dest_dir=out_dir / "covers", out_dir=out_dir, prefix="cover")

    # ── Phase 3: curation (ImageScorer triage) ─────────────────────────
    scoring_cfg = cfg.get("scoring", {})
    use_hps = bool(scoring_cfg.get("use_hps_v2", False))
    use_ir = bool(scoring_cfg.get("use_image_reward", False))
    if not args.no_curate:
        print("\n=== Phase 3: curating (ImageScorer) ===", flush=True)
        _curate(manifest, out_dir, content_level=args.tier,
                keep_top=args.keep_top, use_hps_v2=use_hps, use_image_reward=use_ir)
        if cover_manifest:
            _curate(cover_manifest, out_dir, content_level="T1_suggestive",
                    keep_top=1, use_hps_v2=use_hps, use_image_reward=use_ir)
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
        "brief": brief, "tier": args.tier, "template": args.template,
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
