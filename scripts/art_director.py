"""Art-director prompt generator — LLM-direct rich prompts.

A deliberately MINIMAL alternative to the structured SceneFacetGenerator
pipeline: a strong art-director system prompt + rich exemplars + creative
freedom, producing complete, evocative, sellable artistic-nude prompts
directly from a local LLM. It bypasses the entire vocab / canonicalizer /
composer / safety-prefix / realism-tail machinery — the LLM writes ONE
finished prompt, nothing is bolted on around it.

The only enforcement is a light age/solo safety guard (reject + retry).

Calibrated (2026-05-30) to the user's reference set: PHOTOREAL, spanning
three sub-looks that rotate across a series for built-in variety —
golden-hour glamour, soft natural-light beauty, rich fantasy/editorial.
The exemplars are the quality bar; refine them against new references.

Usage:
  python scripts/art_director.py --brief "Mediterranean summer" \
      --tier T3_artnude --count 6
  python scripts/art_director.py --brief "vintage boudoir" --tier T4_explicit \
      --count 6 --temperature 0.9 --out /tmp/ad_prompts.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel, field_validator  # noqa: E402

from src.agents.llm_client import OllamaClient  # noqa: E402

# Default LLM (Ollama tag for cydonia_heretic_24b — recommended for prose).
CYDONIA_TAG = "Fermi/Cydonia-24B-v4.3-heretic-vision:Q4_K_M"

# Target word band for the prose prompt. Chroma's T5 encoder has a 512-token
# ceiling (~350-380 words); the original 110-160 band used only ~30-47% of it.
# Parameterized so the band A/B (110-160 vs 200-300, ImageReward-judged) is a
# CLI flag, not a code edit. `_ACTIVE_WORD_BAND` is set per-run by
# generate_series so the Pydantic validator's lenient floor/ceiling track it.
WORD_BAND_DEFAULT = (110, 160)
_ACTIVE_WORD_BAND = WORD_BAND_DEFAULT

# Inline prompt-quality gate: every generated prompt is scored by
# scripts/audit_prompts.score_prompt; below this, regenerate (keep-best
# fallback after the attempt budget). Calibrated to the rubric (clean
# LLM-direct prompts score ~9-10; 7.5 is a safe floor).
AUDIT_GATE_THRESHOLD_DEFAULT = 7.5


TIER_DIRECTIVES = {
    "T1_suggestive": (
        "T1 — SUGGESTIVE. Fully clothed or lingerie; sensual but no nudity. "
        "Tease through wardrobe, pose and gaze."
    ),
    "T2_implied": (
        "T2 — IMPLIED. Implied nudity only: strategic cover by fabric, shadow, "
        "limbs or props. Nothing explicit shown."
    ),
    "T3_artnude": (
        "T3 — FINE-ART NUDE. Tasteful full nudity (bare breasts, body) in the "
        "fine-art tradition — the body as form and light, sensual and elegant."
    ),
    "T4_explicit": (
        "T4 — EXPLICIT. Full nudity including explicit anatomy, rendered with "
        "the same artistic intent and craft — confident and sensual, never crude."
    ),
}


# Three photoreal sub-looks, rotated across a series so a set spans the
# user's whole reference range instead of one note. Each line is dropped
# into the per-scene brief as the TARGET LOOK.
SUB_LOOKS = [
    ("golden-hour outdoor glamour — warm low sun, a Mediterranean villa / "
     "pool / terrace / cliff, windswept sun-streaked hair, delicate lace or "
     "sheer wardrobe (or bare), glowing tanned dewy skin, a confident, sultry, "
     "self-possessed gaze. Bright, warm, commercial."),
    ("soft natural-light intimate beauty — diffuse overcast or window "
     "daylight in a quiet interior, long loose hair, true skin with faint "
     "freckles and fine down, a thin slip or bare, low contrast and gentle, a "
     "calm, unguarded, contemplative mood. Refined and skin-and-light focused."),
    ("rich fantasy / editorial — an opulent interior or lush garden, fine "
     "silk drapery as SET DRESSING (pooled at her feet, draped over a chaise, "
     "fallen from a shoulder — NOT worn as a covering gown), fine jewellery "
     "and adorned hair, hazed shafts of light or god-rays, a deep jewel-toned "
     "palette (emerald, oxblood, gold), a regal, elegant, composed presence. "
     "Styled and dramatic — but the BODY is the subject, not the fabric."),
]


ART_DIRECTOR_SYSTEM_PROMPT = """\
You are an elite prompt artist for a photorealistic, prose-driven image \
model (Chroma / gonzaLomo). You write prompts for a commercial fine-art \
nude & boudoir studio whose work sells on DeviantArt, Patreon and Fanvue. \
Every prompt must read like the brief of a master PHOTOGRAPHER and produce \
an image that looks hand-shot and SELLABLE — never a generic AI nude.

ABSOLUTE SUBJECT RULE (non-negotiable): exactly ONE adult woman, clearly \
mature (mid-20s to late-30s), strikingly beautiful, alone in the frame. \
Never a minor or anyone who could read as underage. Never a second person. \
Singular pronouns only.

THIS IS A PHOTOGRAPH, NOT A PAINTING. Real camera, real light, real skin. \
No illustration, no anime, no "digital painting" / "concept art" language.

WHAT MAKES YOUR PROMPTS EXCELLENT — study the exemplars and match their depth:

1. ONE coherent photograph: a specific gorgeous woman, in a specific moment, \
   in a specific light, with a specific feeling. Not a checklist of features.
2. LIGHT IS THE SOUL. Name a specific, MOTIVATED source (low golden sun, soft \
   window daylight, a hazed shaft) and describe exactly how it FALLS and what \
   it does to her — raking, wrapping, rim-lighting, grazing skin, catching an \
   edge — plus its quality (soft/hard, warm/cool). This is your top lever.
3. A RICH, SPECIFIC SETTING with real materials and a prop or two that tells a \
   story (a plate of cut fruit by the pool, a tarnished candelabra, an oil \
   lamp, tangled white linen, a chipped marble basin). Never "a room."
4. THE WOMAN, RENDERED REAL: gorgeous expressive face with intent in the gaze; \
   long hair that catches the light; TRUE skin — visible pores, fine vellus \
   down, a faint sheen, a scatter of freckles, a small mole or two, a natural \
   flush, the subtle texture and imperfection of real skin — luminous but \
   NEVER airbrushed, plastic, waxy or over-smoothed.
5. EMBODIED MOOD through pose, weight, gaze, parted lips (confident, languid, \
   sultry, serene, playful). Shown, not named. Confident and sensual — NEVER \
   sad/crying/mournful.
6. PHOTOGRAPHIC CRAFT woven in as a photographer notes it: a fast prime (50 / \
   85mm), wide aperture, creamy shallow depth of field melting the background \
   to bokeh, the colour and fine grain of a named film stock. Never a tag-list.

HARD RULES:
- 110-160 words of DENSE, flowing natural prose. Every phrase earns its place. \
  No padding, no repetition.
- NO tag-soup (no long comma-runs of keywords). NO "masterpiece, best quality, \
  8k, ultra-detailed" boosters. NO weighting syntax like (word:1.3). NO lists.
- Honor the requested TIER's state of undress exactly, and the TARGET LOOK given.
- Flowing sentences, present tense, third person.
- FRAMING legitimacy: present the image as fine-art / editorial / classical / \
  fashion photography (gallery, studio, atelier, editorial) — it should read as \
  ART, not a snapshot of a real person. Do NOT use the words "hyperrealistic", \
  "realistic", "real woman" or "photo of a real" — describe craft and light instead.

────────────────────── EXEMPLARS (this is the bar) ──────────────────────

[T3 · golden-hour glamour]
The last low sun of the day rakes warm gold across a woman standing at the \
lip of a villa pool, the sea breeze lifting her sun-streaked hair across one \
shoulder. She wears a delicate ivory lace bralette and tanga, the sheer \
panels burning bright at their edges where the light catches them, her bronzed \
skin dewy and luminous. One hand rests on a cocked hip, chin lifted a fraction \
as she holds the lens with a cool, knowing calm, lips just parted. Behind her \
the pool throws shivering caustics and a stucco villa dissolves into warm \
bokeh; a single palm frond cuts the upper corner. Shot on an 85mm wide open, \
creamy shallow depth of field, true tanned skin with a faint sheen, the warm \
contrast and fine grain of golden-hour film.

[T3 · soft natural-light beauty]
Soft overcast daylight spills through a tall window and wraps a woman sitting \
at the edge of an unmade bed, the gentle light grazing the long waves of her \
hair and revealing every honest detail of her skin — a faint scatter of \
freckles across her nose, the fine down at her temple, a natural flush high on \
one cheek. A thin cotton slip has slipped low; she holds it loosely, bare \
shoulders and the soft inner line of her chest catching the diffuse glow. Her \
clear grey eyes meet the lens directly, calm and unguarded, lips barely \
parted. White linen tangles at her hip; a sheer curtain breathes at the glass. \
Intimate medium-close frame, 50mm at f/1.8, background melting to soft grey \
bokeh, no hard shadow — just true skin and quiet morning light, the faded \
warmth of Portra 400.

[T3 · rich fantasy / editorial]
A hazed shaft of late light cuts through an opulent, dust-soft bedchamber and \
falls across a woman seated bare at the edge of an ornate gilded bed, a length \
of iridescent oxblood-and-gold silk pooled at her feet where it has slipped \
from her shoulders moments before. Her body is fully nude, olive skin glowing \
luminous against the deep shadow of the room — tiny jewels glint in her dark, \
loosely waved hair and a fine chain traces her collarbone, but no fabric \
covers her. Embroidered damask pillows and a tarnished candelabra sit behind \
her, a thread of smoke curling slow through the beam. She regards the lens \
with serene, regal composure, full lips still. Elegant three-quarter portrait, \
85mm at f/2.2, sumptuous shallow depth of field, a rich jewel-toned palette, \
true skin and the soft sheen of the pooled silk on the floor, warm light and \
fine grain.

──────────────────────────────────────────────────────────────────────────

Write at this level. Return ONLY the prompt text in the requested JSON shape.
"""


def _build_system_prompt(word_band: tuple[int, int] = WORD_BAND_DEFAULT) -> str:
    """The art-director system prompt with the target word band injected.
    Default (110-160) is a no-op replace; the A/B passes (200, 300)."""
    lo, hi = word_band
    if (lo, hi) == (110, 160):
        return ART_DIRECTOR_SYSTEM_PROMPT
    return ART_DIRECTOR_SYSTEM_PROMPT.replace("110-160 words", f"{lo}-{hi} words")


class _PromptOut(BaseModel):
    prompt: str

    @field_validator("prompt")
    @classmethod
    def _check(cls, v: str) -> str:
        text = (v or "").strip()
        words = len(text.split())
        lo, hi = _ACTIVE_WORD_BAND
        floor = max(40, int(lo * 0.6))      # lenient — gate quality via audit, not length
        ceiling = int(hi * 1.4) + 30
        if words < floor:
            raise ValueError(
                f"prompt too short ({words} words) — needs {lo}-{hi} of rich prose"
            )
        if words > ceiling:
            raise ValueError(f"prompt too long ({words} words) — tighten toward {hi}")
        low = text.lower()
        # Age / solo safety guard (non-negotiable).
        banned = (
            "child", "teen", "teenage", "loli", "underage", "minor",
            "young girl", "little girl", "schoolgirl",
            "2girls", "two women", "couple", "her partner", "with him",
            "they ", "group", "threesome", "multiple women",
        )
        hit = [b for b in banned if b in low]
        if hit:
            raise ValueError(f"safety: banned age/multi-subject token(s) {hit}")
        # Photo, not painting (calibrated to refs: chroma = photoreal).
        paint = ("oil painting", "digital painting", "concept art", "illustration",
                 "anime", "painterly brush", "watercolor", "cel shad")
        phit = [p for p in paint if p in low]
        if phit:
            raise ValueError(f"style: non-photographic token(s) {phit}")
        # Render-safety: chroma renders mirrors as warped faces / body
        # doubles. The ONE structural guard worth keeping from the old
        # pipeline — reject + retry so the LLM re-rolls the scene.
        if "mirror" in low:
            raise ValueError(
                "render-risk: 'mirror' present — chroma warps mirror "
                "reflections into double faces. Re-write without a mirror."
            )
        return text


def _signature(prompt: str) -> str:
    """A descriptor of an already-written prompt, fed back into the variety
    guard so the next scene can't reuse the same opening / setting / light.
    2026-06: widened from 24 to 40 words after a strongly-themed brief (PNW
    forest) showed 4/8 prompts sharing the same 'The last/low sun … forest
    clearing where fog …' opener despite the 24-word check."""
    return " ".join(prompt.split()[:40]) + " …"


def _opener(prompt: str) -> str:
    """The first 8 words of a prompt — fed to the variety guard as an
    EXPLICIT banned opener so the LLM cannot reuse the same opening phrase
    structure (a failure mode the signature alone didn't catch)."""
    return " ".join(prompt.split()[:8])


def generate_one(
    client: OllamaClient,
    *,
    brief: str,
    tier: str,
    sub_look: str,
    avoid: list[str],
    banned_openers: list[str],
    model_tag: str,
    temperature: float,
    word_band: tuple[int, int] = WORD_BAND_DEFAULT,
) -> str:
    tier_directive = TIER_DIRECTIVES.get(tier, TIER_DIRECTIVES["T3_artnude"])
    variety = ""
    if avoid or banned_openers:
        parts: list[str] = [
            "\n\nThis is one image in a varied series. Make it VISIBLY "
            "DIFFERENT from the ones already shot — a different opening, "
            "setting, time of day, light direction, pose, wardrobe/props AND "
            "mood."
        ]
        if banned_openers:
            parts.append(
                "\n\nBANNED OPENERS — do NOT begin your prompt with any of "
                "these phrasings or close variants:\n"
                + "\n".join(f"  • {o}" for o in banned_openers)
            )
        if avoid:
            parts.append(
                "\n\nDo NOT echo the setting / light / pose of these prior "
                "prompts:\n" + "\n".join(f"  - {a}" for a in avoid)
            )
        variety = "".join(parts)
    # 2026-06 — tier directive moved AFTER the sub-look (LLMs weight
    # last-mentioned more heavily). Without this, a strongly-themed brief
    # paired with the "rich fantasy / editorial" sub-look's wardrobe
    # language was producing fully-clothed renders at T3 (PNW forest test:
    # gowns won over the bare-body tier directive).
    user_prompt = (
        f"Creative brief: {brief}\n"
        f"TARGET LOOK for THIS image: {sub_look}\n"
        f"TIER — STATE OF UNDRESS (this OVERRIDES any wardrobe language in "
        f"the look above; fabric in the look is SET DRESSING only):\n"
        f"  {tier_directive}\n"
        f"{variety}\n\n"
        'Write ONE excellent photograph-prompt at the level of the exemplars, '
        'in the target look above, honoring the TIER above EXACTLY. The TIER '
        'wins over the look\'s wardrobe descriptors — at T3+, the body is '
        'bare and any silk/gown/lace is set dressing (pooled, draped over a '
        'chaise, fallen) rather than worn. '
        'Return JSON: {"prompt": "<your prompt text>"}'
    )
    result = client.generate_json(
        _build_system_prompt(word_band),
        user_prompt,
        model=model_tag,
        temperature=temperature,
        num_predict=1100,
        schema=_PromptOut,
    )
    return str(result["prompt"]).strip()


def generate_series(
    *,
    brief: str,
    tier: str,
    count: int,
    model_tag: str,
    temperature: float,
    sub_looks: list[str] | None = None,
    word_band: tuple[int, int] = WORD_BAND_DEFAULT,
    audit_gate: bool = True,
    audit_threshold: float = AUDIT_GATE_THRESHOLD_DEFAULT,
    max_attempts: int = 4,
) -> list[dict]:
    """Generate ``count`` prompts. ``sub_looks`` (from the niche selector)
    overrides the default 3; the per-scene look rotates through them.

    Inline quality gate (2026-06): each candidate is scored by
    ``audit_prompts.score_prompt``; below ``audit_threshold`` it regenerates,
    keeping the best-scoring attempt as a fallback so a scene is never dropped
    purely for a soft-quality miss. Pydantic guards (safety/word-band/mirror)
    still hard-reject before scoring."""
    global _ACTIVE_WORD_BAND
    _ACTIVE_WORD_BAND = word_band

    looks = sub_looks or [s for s in SUB_LOOKS]
    score_fn = None
    if audit_gate:
        try:  # lazy — keeps art_director import light + decoupled
            from scripts.audit_prompts import score_prompt as score_fn  # type: ignore
        except Exception as exc:  # noqa: BLE001
            print(f"  (audit gate disabled — score_prompt unavailable: {exc})",
                  file=sys.stderr, flush=True)
            score_fn = None

    client = OllamaClient()
    out: list[dict] = []
    avoid: list[str] = []
    banned_openers: list[str] = []
    for i in range(count):
        sub_look = looks[i % len(looks)]
        look_label = sub_look.split(" — ")[0]
        best: tuple[str, float, list[str]] | None = None  # (prompt, score, issues)
        last_err = None
        for attempt in range(max_attempts):
            try:
                p = generate_one(
                    client,
                    brief=brief,
                    tier=tier,
                    sub_look=sub_look,
                    avoid=avoid,
                    banned_openers=banned_openers,
                    model_tag=model_tag,
                    temperature=temperature,
                    word_band=word_band,
                )
            except Exception as exc:  # noqa: BLE001 — Pydantic/safety reject → retry
                last_err = exc
                print(f"  (scene {i + 1} attempt {attempt + 1} rejected: {exc})",
                      file=sys.stderr, flush=True)
                continue

            score, issues = (score_fn(p, tier) if score_fn else (10.0, []))
            if best is None or score > best[1]:
                best = (p, score, issues)
            if score >= audit_threshold:
                break
            print(f"  (scene {i + 1} attempt {attempt + 1} audit {score:.1f}"
                  f"<{audit_threshold} — regenerating; issues={issues[:2]})",
                  file=sys.stderr, flush=True)

        if best is None:
            print(f"  !! scene {i + 1} failed after {max_attempts} attempts: "
                  f"{last_err}", file=sys.stderr, flush=True)
            continue

        p, score, issues = best
        if score < audit_threshold:
            print(f"  (scene {i + 1} shipping best audit={score:.1f} after "
                  f"{max_attempts} attempts; issues={issues[:3]})",
                  file=sys.stderr, flush=True)
        out.append({"look": look_label, "prompt": p, "audit_score": round(score, 2)})
        avoid.append(_signature(p))
        banned_openers.append(_opener(p))
        print(f"\n[{i + 1}/{count}] {look_label} (audit {score:.1f}, "
              f"{len(p.split())} words)\n{p}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="LLM-direct art-director prompts")
    ap.add_argument("--brief", required=True, help="creative brief / theme")
    ap.add_argument("--tier", default="T3_artnude", choices=list(TIER_DIRECTIVES))
    ap.add_argument("--count", type=int, default=6)
    ap.add_argument("--model-tag", default=CYDONIA_TAG,
                    help="Ollama model tag (default: Cydonia 24B)")
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--word-band", default="110-160",
                    help="target prose word band 'lo-hi' (A/B: try 200-300)")
    ap.add_argument("--no-audit-gate", action="store_true",
                    help="disable the inline audit_prompts quality gate")
    ap.add_argument("--out", default="", help="optional JSON output path")
    args = ap.parse_args()

    try:
        lo_s, hi_s = args.word_band.split("-")
        word_band = (int(lo_s), int(hi_s))
    except ValueError:
        ap.error("--word-band must be 'lo-hi', e.g. 110-160 or 200-300")

    rows = generate_series(
        brief=args.brief,
        tier=args.tier,
        count=args.count,
        model_tag=args.model_tag,
        temperature=args.temperature,
        word_band=word_band,
        audit_gate=not args.no_audit_gate,
    )

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"brief": args.brief, "tier": args.tier,
             "prompts": [r["prompt"] for r in rows], "rows": rows},
            indent=2,
        ))
        print(f"\nSaved {len(rows)} prompts -> {args.out}", flush=True)

    try:
        OllamaClient().unload_all()
    except Exception:  # noqa: BLE001
        pass
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
