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
    ("rich fantasy / editorial — an opulent interior or lush garden, ornate "
     "gown or draped silk slipping away, fine jewellery and adorned hair, "
     "hazed shafts of light or god-rays, a deep jewel-toned palette (emerald, "
     "oxblood, gold), a regal, elegant, composed presence. Styled and dramatic."),
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
   long hair that catches the light; TRUE skin — a faint sheen, fine down, a \
   scatter of freckles, a natural flush — luminous but never plastic or \
   over-smoothed.
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
falls across a woman seated at the edge of an ornate gilded bed, a length of \
iridescent oxblood-and-gold silk drawn loose across her body and slipping from \
one shoulder. Tiny jewels glint in her dark, loosely waved hair and a fine \
chain traces her collarbone; her olive skin glows luminous against the deep \
shadow of the room. Embroidered damask pillows and a tarnished candelabra sit \
behind her, a thread of smoke curling slow through the beam. She regards the \
lens with serene, regal composure, full lips still. Elegant three-quarter \
portrait, 85mm at f/2.2, sumptuous shallow depth of field, a rich jewel-toned \
palette, true skin and the soft sheen of silk, warm light and fine grain.

──────────────────────────────────────────────────────────────────────────

Write at this level. Return ONLY the prompt text in the requested JSON shape.
"""


class _PromptOut(BaseModel):
    prompt: str

    @field_validator("prompt")
    @classmethod
    def _check(cls, v: str) -> str:
        text = (v or "").strip()
        words = len(text.split())
        if words < 70:
            raise ValueError(
                f"prompt too short ({words} words) — needs 110-160 of rich prose"
            )
        if words > 210:
            raise ValueError(f"prompt too long ({words} words) — tighten to ~150")
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
    """A compact descriptor of an already-written prompt, fed back into
    the variety guard so the next scene can't reuse the same opener /
    setting / light."""
    return " ".join(prompt.split()[:24]) + " …"


def generate_one(
    client: OllamaClient,
    *,
    brief: str,
    tier: str,
    sub_look: str,
    avoid: list[str],
    model_tag: str,
    temperature: float,
) -> str:
    tier_directive = TIER_DIRECTIVES.get(tier, TIER_DIRECTIVES["T3_artnude"])
    variety = ""
    if avoid:
        variety = (
            "\n\nThis is one image in a varied series. Make it VISIBLY "
            "DIFFERENT from the ones already shot — a different opening, "
            "setting, time of day, light direction, pose, wardrobe/props AND "
            "mood. Do NOT echo these:\n" + "\n".join(f"  - {a}" for a in avoid)
        )
    user_prompt = (
        f"Creative brief: {brief}\n"
        f"Tier: {tier_directive}\n"
        f"TARGET LOOK for THIS image: {sub_look}\n"
        f"{variety}\n\n"
        'Write ONE excellent photograph-prompt at the level of the exemplars, '
        'in the target look above. '
        'Return JSON: {"prompt": "<your prompt text>"}'
    )
    result = client.generate_json(
        ART_DIRECTOR_SYSTEM_PROMPT,
        user_prompt,
        model=model_tag,
        temperature=temperature,
        num_predict=700,
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
) -> list[dict]:
    client = OllamaClient()
    out: list[dict] = []
    avoid: list[str] = []
    for i in range(count):
        sub_look = SUB_LOOKS[i % len(SUB_LOOKS)]
        look_label = sub_look.split(" — ")[0]
        last_err = None
        for attempt in range(3):
            try:
                p = generate_one(
                    client,
                    brief=brief,
                    tier=tier,
                    sub_look=sub_look,
                    avoid=avoid,
                    model_tag=model_tag,
                    temperature=temperature,
                )
                out.append({"look": look_label, "prompt": p})
                avoid.append(_signature(p))
                print(f"\n[{i + 1}/{count}] {look_label} ({len(p.split())} words)\n{p}",
                      flush=True)
                break
            except Exception as exc:  # noqa: BLE001 — prototype: surface + retry
                last_err = exc
                print(f"  (scene {i + 1} attempt {attempt + 1} rejected: {exc})",
                      file=sys.stderr, flush=True)
        else:
            print(f"  !! scene {i + 1} failed after 3 attempts: {last_err}",
                  file=sys.stderr, flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="LLM-direct art-director prompts")
    ap.add_argument("--brief", required=True, help="creative brief / theme")
    ap.add_argument("--tier", default="T3_artnude", choices=list(TIER_DIRECTIVES))
    ap.add_argument("--count", type=int, default=6)
    ap.add_argument("--model-tag", default=CYDONIA_TAG,
                    help="Ollama model tag (default: Cydonia 24B)")
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--out", default="", help="optional JSON output path")
    args = ap.parse_args()

    rows = generate_series(
        brief=args.brief,
        tier=args.tier,
        count=args.count,
        model_tag=args.model_tag,
        temperature=args.temperature,
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
