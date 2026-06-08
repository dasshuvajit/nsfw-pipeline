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
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

from pydantic import BaseModel, field_validator  # noqa: E402

from src.agents.llm_client import OllamaClient, LLMClientPool  # noqa: E402

# Cydonia (Ollama) — kept as the registry FALLBACK + the painterly-light
# specialist (best optical light-on-form precision in the A/B). Use it via
# `--model-tag $(python -c "import scripts.art_director as a; print(a.CYDONIA_TAG)")`.
CYDONIA_TAG = "Fermi/Cydonia-24B-v4.3-heretic-vision:Q4_K_M"


def _default_llm_tag() -> str:
    """Backend tag of the registry's default_llm (config/llm_models.yaml).
    As of 2026-06-04 that's gemma_4_26b_a4b_heretic (LM Studio) — made the main
    prompt LLM after a blind 4-lens judge panel (wins creativity/sellability/
    render-fidelity) + ~2.7x speed. Falls back to Cydonia if the registry is
    unreadable. NOTE: the Gemma default needs LM Studio running with the model
    loaded; otherwise pass --model-tag for the Ollama/Cydonia path."""
    try:
        from src.memory.llm_registry import LLMRegistryLoader
        reg = LLMRegistryLoader()
        return reg.get_llm(reg.default_llm_id).model_tag
    except Exception:  # noqa: BLE001 — registry unreadable → safe Ollama default
        return CYDONIA_TAG


DEFAULT_LLM_TAG = _default_llm_tag()

# Target word band for the prose prompt. Chroma's T5 encoder has a 512-token
# ceiling (~350-380 words); the original 110-160 band used only ~30-47% of it.
# Parameterized so the band A/B (110-160 vs 200-300, ImageReward-judged) is a
# CLI flag, not a code edit. `_ACTIVE_WORD_BAND` is set per-run by
# generate_series so the Pydantic validator's lenient floor/ceiling track it.
WORD_BAND_DEFAULT = (110, 160)
_ACTIVE_WORD_BAND = WORD_BAND_DEFAULT

# When True, the validator hard-rejects any nudity in a generated prompt.
# Used for PUBLIC SFW covers/thumbnails (DA shopfront ToS: covers must carry
# NO mature content — even a tasteful art-nude silhouette is non-compliant).
# Set per-run by generate_series(require_sfw=True).
_ACTIVE_REQUIRE_SFW = False

# Nudity tokens that disqualify a cover/thumbnail prompt (checked only when
# _ACTIVE_REQUIRE_SFW). Tier directives alone proved insufficient — the LLM
# wrote "wears only stockings" (topless) for a T1 cover.
_NUDITY_TOKENS = (
    "nude", "naked", "topless", "bare breast", "bare chest", "bare-chested",
    "exposed breast", "exposed chest", "areola", "nipple", "bare body",
    "fully bare", "undressed", "unclothed", "bare-skinned",
)

# Mood gate: commercial NSFW sells confident sensuality, not sorrow. The prompt
# validator hard-rejects (reject + re-roll) any prose carrying an unambiguous
# sad-affect word. Matched at the WORD level (token-exact / stem-prefix) so
# 'sad' can't fire inside 'saddle' and 'sob' can't fire inside 'sober'. Quiet
# moods should read introspective / contemplative / serene, never sad.
_SAD_MOOD_EXACT = frozenset({
    "sad", "sadness", "mournful", "melancholic", "melancholy", "sorrow",
    "sorrowful", "crying", "tearful", "forlorn", "woeful", "grief", "doleful",
})
_SAD_MOOD_PREFIX = ("griev", "weep", "despair", "anguish", "mourn")  # grieving, weeping, …

# Implausible-grounding guard: the subject rendered SITTING / KNEELING / LYING /
# FLOATING on water or in mid-air (a body hovering on nothing) — the #1 bad-pose
# failure ("sitting on water"). Tight + high-precision: a pose verb must be
# followed (within 2 words) by on/atop/upon + a water body or "the water's
# surface" — so "kneels AT the water's edge" or "light ON the water" (no pose
# verb) do NOT trip it. Plus floating/hovering on water/air + mid-air.
_WATER_BODY = r"(?:water|lake|river|pond|sea|ocean|pool)"
# A single optional adjective slot ("the CALM water", "the GLASSY lake") — but a
# solid-surface noun consumes the slot and blocks the match ("on the ROCK by the
# water" → no match), keeping precision high.
_ADJ = r"(?:\w+\s+){0,2}"
_IMPLAUSIBLE_GROUNDING_RE = re.compile(
    r"\b(?:sit|sitt|sat|kneel|knelt|kneeling|lie|lying|lay|reclin|perch)\w*\b"
    r"(?:\W+\w+){0,2}?\W+(?:on|atop|upon)\W+(?:the\s+|her\s+)?" + _ADJ
    + rf"(?:{_WATER_BODY}'?s\s+surface|surface\s+of\s+the\s+{_WATER_BODY}|{_WATER_BODY})\b"
    + rf"|\b(?:float|hover)\w*\b(?:\W+\w+){{0,2}}?\W+(?:on|above|over|upon)\W+(?:the\s+)?"
    + _ADJ + rf"(?:{_WATER_BODY}|air)\b"
    + r"|\bmid[\s-]?air\b|\bsuspended\s+in\s+(?:the\s+)?air\b",
    re.IGNORECASE,
)

# Hard SFW instruction appended to PUBLIC cover/teaser prompts.
SFW_COVER_DIRECTIVE = (
    "PUBLIC SHOPFRONT COVER — she must be FULLY CLOTHED in elegant attire that "
    "completely covers the breasts and groin (a dress, gown, robe, blouse, or a "
    "full lingerie set). ABSOLUTELY NO nudity, NO topless, NO bare breasts, NO "
    "implied nude, NO sheer see-through over bare skin. This is a safe-for-work "
    "thumbnail; sensuality comes from pose, wardrobe and gaze only."
)

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

1. ONE coherent photograph: a specific, strikingly beautiful and sexy young \
   adult woman who DOMINATES the frame, in a specific moment, in a specific \
   light, with a specific feeling. She is the magnetic centre of the image — \
   alluring and desirable, not a checklist of features.
2. LIGHT IS THE SOUL — and light MODELS FORM. Name a specific, MOTIVATED source \
   (low golden sun, soft window daylight, a single hazed shaft) and its quality \
   (soft/hard, warm/cool) AND its direction (from camera-left and slightly above, \
   from a low window behind her, raking across from the side). Then describe what \
   it DOES optically: how it sculpts her in three dimensions — where the highlight \
   sits, how it rolls through the soft half-light to the shadow terminator and \
   falls off into shadow, giving the body real volume and weight. Be specific \
   about where it pools and where it leaves dark. Name the precise micro-texture \
   it reveals where it grazes: a catchlight in the eye, a specular sheen sliding \
   along a collarbone or the rise of a hip, the warm translucency at the edge of \
   an ear, grazing light making pores and fine vellus down legible at the \
   terminator, a soft rim separating her from the background. This optical \
   precision — light on form — is your single top lever.
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
   sad/crying/mournful/melancholic/melancholy/sorrowful/wistful/forlorn. For \
   quiet moods use introspective, contemplative, pensive-calm, or serene \
   composure instead — never sadness.
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
- LOOK: a crystal-clear, razor-sharp, high-end GLAMOUR photograph — photoreal, \
  luminous, high-detail, flawless focus. Lean into polished beauty + sensual \
  appeal (editorial / glamour / boudoir photography), not muted gallery restraint. \
  Still describe it through craft + light (lens, key, grain) rather than literally \
  writing "hyperrealistic" / "a real woman" — the realism comes from the rendering.
- NO MIRRORS or reflective surfaces that show the subject (mirror, vanity \
  mirror, reflection in glass/water). The model warps reflections into a \
  second, distorted face — it breaks the image. For dressing-table / boudoir \
  scenes use a vanity, dressing table or console WITHOUT a mirror (perfume \
  bottles, a powder compact, a lamp, jewellery), never a mirror.
- PROPS & ATMOSPHERE must be RENDERABLE and MOTIVATED. Every atmospheric \
  element (smoke, steam, vapour, drifting dust) needs a VISIBLE in-frame \
  source — a lit cigarette held or resting in an ashtray, a candle, a \
  smouldering censer — or leave it out; floating smoke with no source reads \
  as a glitch. Do NOT write an ambiguous "spill / splash / spray of champagne \
  bubbles" for a glass that simply sits there — a coupe HOLDS its champagne, \
  it does not erupt. Avoid thin, hard-to-render focal props (a long cigarette \
  holder, a slim stem, a fine chain) — the model renders them as ambiguous \
  lines, worst of all in silhouette; choose solid, legible props instead.
- NO LEGIBLE TEXT or signage. Keep any signs, neon, screens, labels, posters, \
  branding or tattoos abstract, glowing, distant or blurred — the model renders \
  written words as garbled gibberish that reads as an obvious AI tell on a close \
  look. And NO people, faces or figures shown on any background screen, monitor, \
  poster or photo in the scene — SHE is the only person anywhere in the frame.

────────────── FRAMING & COMPOSITION (you CHOOSE this per image) ──────────────
You decide the ORIENTATION and SHOT TYPE that best serve each scene, and you VARY
them across a series — never default everything to a portrait close-up. The prompt
prose must PHYSICALLY MATCH the framing you choose.

ORIENTATION (aspect):
- portrait (tall 2:3): intimate close-ups; a standing or kneeling full-body where
  height is the story; vertical settings (windows, doorways, tall drapery).
- landscape (wide 3:2): a reclining body along the frame; environmental scenes
  where the setting breathes beside her; horizons, beds, chaises, pools, terraces.
- square (1:1): balanced medium shots; centred, graphic, editorial compositions.

SHOT TYPE — and what each DEMANDS of the prose:
- close_up: the face/gaze fills the frame — describe eyes, lips, skin texture,
  a catchlight, the fall of hair; little or no body. Make the face the subject.
- bust: head to chest/collarbone — face + shoulders + décolletage; a portrait.
- medium: head to waist/hips — the default; pose + expression + upper body.
- full_body: head to FEET, the WHOLE figure in the scene — every limb visible and
  ANATOMICALLY COHERENT (see below). Composition shows the complete pose.
- wide_environmental: subject and setting are co-equal — she sits within a rich
  space (a room, a landscape) that is itself part of the picture; she may be
  smaller in frame, but remains the clear focal point via light and placement.

SUBJECT FOCUS: she DOMINATES every frame — large, close and commanding, filling
the composition so she is unmistakably the subject. Lead the eye straight to her
with light, contrast, scale and shallow depth of field. NEVER a small, distant
figure lost in an empty room or landscape, and never let the background swallow
her — the picture is about HER, the setting is support. If a scene is wide, she
still reads large and central.

ANATOMICAL CLARITY (mandatory whenever the body is visible, especially full_body
at T3/T4): describe a pose the body can actually hold — weight planted on one or
both feet (standing), hips resting on the surface (seated), a continuous natural
spine (reclining); arms and legs in clear, unforced positions; hands resolved
(resting, trailing, in her hair) not hidden-then-mangled. No twisted joints, no
floating or detached torso, no impossible contortion. One coherent body.
HANDS are the #1 render failure — write them to come out clean: at most ONE hand
interacts with a prop, and keep that interaction simple (a hand resting ON a
surface, hip or thigh beats fingers GRIPPING a wreath, a flute or fabric); the
other hand relaxed and clearly placed. AVOID interlaced or tightly clasped hands —
fingers fuse into a soft, indistinct cluster where they overlap. NEVER bury a hand
in deep shadow — it melts to a fingerless blob — and NEVER combine
both-arms-behind-the-head WITH a hand
holding an object, which spawns a phantom third hand. Two hands, two arms, two
legs, every one of them traceable to her single body. In dark, low-key, noir or
neon scenes ESPECIALLY, deliberately place her hands in the pool of light (lit by
the key or a glowing source) so the fingers stay crisp — never let a dark scene
swallow the hands into mushy shadow.

STABLE GROUNDING & SUPPORT (mandatory): she must rest on a believable SOLID
surface that visibly bears her weight — ground, grass, sand, rock, a blanket or
towel, a bed, a chaise, a chair, a ledge, stone steps. NEVER write her sitting,
kneeling, lying or floating ON water, ON the surface of a lake / river / pool /
sea, or in mid-air — it renders as a body hovering on nothing. Near water she is
on a clear bank, rock, dock, towel or shallow edge with the water BESIDE or
BEHIND her, never under her. Choose a stable, weight-bearing, naturally
flattering posture — no precarious balance, no contortion. FEET & TOES: bare feet
are render-fragile, so PREFER poses that keep them tucked under her, angled away,
or out of frame; when a foot IS in frame place it clearly (flat on the surface or
cleanly tucked), exactly five toes per foot, never merged, doubled or overlapping
feet.

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

[T3 · landscape · full_body environmental]  (orientation: landscape, shot_type: full_body)
Late afternoon light pours low across a sun-warmed stone terrace where a woman \
reclines full-length along a weathered chaise, her whole body stretched easy \
through the wide frame — head resting back on one arm, the long line of her spine \
and hip and legs unbroken and relaxed, one knee lifted, the other leg extended, \
bare feet crossed at the ankle. Her nude form is gilded by the raking sun, true \
skin luminous along every edge it catches; a sheer linen throw has slipped to the \
floor beside her. Behind and around her the terrace breathes — a cracked terracotta \
urn, a spill of bougainvillea, the soft blue haze of distant hills — the setting \
co-equal with her, yet she holds the eye through the warm key light falling square \
on her body. She gazes off toward the horizon, lips parted, wholly at ease. Wide \
environmental frame, 35mm at f/4 so the whole body and the terrace stay sharp, the \
warm contrast and fine grain of late-golden-hour film. [Every limb clearly placed, \
weight settled into the chaise, one coherent restful body.]

──────────────────────────────────────────────────────────────────────────

Write at this level. Choose the orientation + shot_type that best serve the scene
and VARY them across the series. Return ONLY the JSON shape requested.
"""


# ── Creative Direction: the tunable, config-driven house style ──────────────
# config/creative_direction.yaml biases the SUBJECT (focus / look / heat /
# quality) as a weighted lean while the niche/scene variety keeps rotating.
# DYNAMIC (edit the YAML, no code change) — injected into the system prompt as
# priorities the LLM applies flexibly, and the look pools are sampled per image
# so a series shows wide variety instead of clones. Absent file -> graceful
# no-op (the prompt engine runs exactly as before).
def _load_creative_direction() -> dict:
    try:
        p = _PROJECT_ROOT / "config" / "creative_direction.yaml"
        return yaml.safe_load(p.read_text()) or {}
    except Exception:  # noqa: BLE001
        return {}


_CREATIVE = _load_creative_direction()


def _creative_system_block() -> str:
    """Assemble the CREATIVE DIRECTION block appended to the system prompt."""
    c = _CREATIVE
    if not c:
        return ""
    L = ["", "──────────── CREATIVE DIRECTION (house style — honor on EVERY image) ────────────"]
    if c.get("subject_focus"):
        L.append(f"SUBJECT DOMINANCE: {c['subject_focus'].strip()}")
    if c.get("age_band") or c.get("appeal"):
        L.append(f"WHO SHE IS: {c.get('age_band','').strip()} — {c.get('appeal','').strip()} "
                 "(ADULT only; the age-safety rules above are never relaxed).")
    if c.get("realism"):
        L.append(f"LOOK & QUALITY: render her as {c['realism'].strip()}.")
    if c.get("content_lean"):
        L.append(f"HEAT: {c['content_lean'].strip()}.")
    L.append("Apply this as the CONSISTENT house style WHILE keeping the scene, setting, "
             "niche, palette and mood VARIED per image — the variety must never be lost.")
    return "\n".join(L)


def _creative_look(index: int) -> str:
    """Per-image subject look (hair + figure) sampled from the look pools so a
    series shows wide variety, not clones. Hair/figure offset so they don't move
    in lockstep. Empty if no pools configured."""
    pools = (_CREATIVE or {}).get("look_pools") or {}
    hair, figure = pools.get("hair") or [], pools.get("figure") or []
    parts = []
    if hair:
        parts.append(hair[index % len(hair)])
    if figure:
        # +3 offset (step 1) so hair and figure don't start in sync yet every
        # value is still visited — full coverage regardless of pool lengths.
        parts.append(figure[(index + 3) % len(figure)])
    return ", ".join(parts)


def _build_system_prompt(word_band: tuple[int, int] = WORD_BAND_DEFAULT) -> str:
    """The art-director system prompt with the target word band + the
    config-driven CREATIVE DIRECTION house style injected. Default word band
    (110-160) is a no-op replace; the A/B passes (200, 300)."""
    lo, hi = word_band
    base = (ART_DIRECTOR_SYSTEM_PROMPT if (lo, hi) == (110, 160)
            else ART_DIRECTOR_SYSTEM_PROMPT.replace("110-160 words", f"{lo}-{hi} words"))
    block = _creative_system_block()
    return f"{base}\n{block}" if block else base


# Per-prompt framing the LLM chooses (Phase 1 — kills the only-portrait
# problem + drives subject/anatomy focus). Tolerant str fields: a value the
# model gets slightly wrong falls back to a safe default rather than failing
# the render. The system prompt + user prompt teach the LLM what each means.
ORIENTATIONS_ALLOWED = ("portrait", "square", "landscape")
SHOT_TYPES_ALLOWED = ("close_up", "bust", "medium", "full_body", "wide_environmental")

# Rotated framing TARGETS — without a per-scene nudge the LLM defaults almost
# everything to portrait/medium. This rotation guarantees a sellable spread
# across all 3 orientations + 5 shot types; the LLM still emits the FINAL choice
# + a rationale and may override when the scene genuinely demands it.
# Subject-FILLING by design: every target keeps her large in frame (the fix for
# "subject too far away / empty picture"). wide_environmental is intentionally
# NOT in the forced rotation — the LLM may still choose it for a scene that
# genuinely needs it, but the default lean is close/medium/full where she
# dominates. Still spreads across all 3 orientations + 4 subject-filling shots.
FRAMING_TARGETS: tuple[tuple[str, str], ...] = (
    ("portrait", "full_body"),          # standing/kneeling, height is the story
    ("portrait", "close_up"),           # face/gaze fills the frame
    ("square", "medium"),               # balanced, graphic, centred
    ("landscape", "full_body"),         # reclining along the frame, body large
    ("portrait", "bust"),               # head-to-chest portrait
    ("square", "bust"),
    ("landscape", "medium"),            # torso fills the wide frame
    ("portrait", "medium"),
)


class _PromptOut(BaseModel):
    prompt: str
    orientation: str = "portrait"      # portrait | square | landscape
    shot_type: str = "medium"          # close_up | bust | medium | full_body | wide_environmental
    framing_rationale: str = ""         # 1-2 sentences: why this orientation+shot fits the scene

    @field_validator("orientation")
    @classmethod
    def _orient(cls, v: str) -> str:
        v = (v or "").strip().lower()
        return v if v in ORIENTATIONS_ALLOWED else "portrait"

    @field_validator("shot_type")
    @classmethod
    def _shot(cls, v: str) -> str:
        v = (v or "").strip().lower().replace("-", "_").replace(" ", "_")
        # tolerate common synonyms the LLM may emit
        synonyms = {"closeup": "close_up", "headshot": "close_up", "portrait": "bust",
                    "fullbody": "full_body", "full": "full_body", "wide": "wide_environmental",
                    "environmental": "wide_environmental", "establishing": "wide_environmental"}
        v = synonyms.get(v, v)
        return v if v in SHOT_TYPES_ALLOWED else "medium"

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
            # multi-subject — precise phrases only (bare "they"/"group" caused
            # false positives on "where they catch the light" / "a group of trees")
            "2girls", "two women", "two figures", "the two of them",
            "both women", "couple", "her partner", "with him",
            "threesome", "multiple women", "multiple figures", "group of women",
            "group of people", "other woman", "another woman",
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
        # Mood gate: confident/sensual, never sad. Word-level so 'sad' can't
        # fire inside 'saddle'. Reject + re-roll toward serene/introspective.
        toks = {w.strip(".,;:!?—–-\"'()[]") for w in low.split()}
        sad = sorted(t for t in toks if t in _SAD_MOOD_EXACT
                     or any(t.startswith(p) for p in _SAD_MOOD_PREFIX))
        if sad:
            raise ValueError(
                f"mood: sad-affect term(s) {sad} — commercial NSFW sells "
                f"confident sensuality. Re-write as introspective / serene / "
                f"contemplative, never sad."
            )
        # Render-safety: chroma renders mirrors as warped faces / body
        # doubles. The ONE structural guard worth keeping from the old
        # pipeline — reject + retry so the LLM re-rolls the scene.
        if "mirror" in low:
            raise ValueError(
                "render-risk: 'mirror' present — chroma warps mirror "
                "reflections into double faces. Re-write without a mirror."
            )
        # Implausible-grounding gate: the subject must rest on a solid,
        # weight-bearing surface — never sitting/kneeling/floating ON water or
        # in mid-air (renders as a body hovering on nothing). Reject + re-roll
        # toward a clear bank/rock/towel with the water beside or behind her.
        if _IMPLAUSIBLE_GROUNDING_RE.search(low):
            raise ValueError(
                "grounding: subject reads as on water / mid-air — re-write so "
                "she rests on a SOLID surface (bank, rock, towel, ground, chair) "
                "with any water beside or behind her, never under her."
            )
        # SFW-cover gate: covers/thumbnails must be fully clothed (DA
        # shopfront ToS). Reject any nudity so the LLM re-rolls clothed.
        if _ACTIVE_REQUIRE_SFW:
            nudity = [t for t in _NUDITY_TOKENS if t in low]
            if nudity:
                raise ValueError(
                    f"SFW-cover: nudity token(s) {nudity} — covers must be "
                    f"FULLY CLOTHED. Re-write with covering attire."
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
    extra_directive: str = "",
    framing_target: tuple[str, str] | None = None,
    look_target: str = "",
) -> dict:
    tier_directive = TIER_DIRECTIVES.get(tier, TIER_DIRECTIVES["T3_artnude"])
    if extra_directive:
        tier_directive = f"{tier_directive}\n{extra_directive}"
    variety = ""
    if avoid or banned_openers:
        parts: list[str] = [
            "\n\nThis is one image in a varied series. Make it VISIBLY "
            "DIFFERENT from the ones already shot AND from any PRIOR series in "
            "this same category listed below — a different opening, setting, "
            "time of day, light direction, pose, wardrobe/props AND mood. We do "
            "NOT want this series to resemble earlier series in the same category."
        ]
        if banned_openers:
            parts.append(
                "\n\nBANNED OPENERS — do NOT begin your prompt with any of these "
                "phrasings or close variants (from this series AND earlier series "
                "in the same category):\n"
                + "\n".join(f"  • {o}" for o in banned_openers)
            )
        if avoid:
            parts.append(
                "\n\nDo NOT echo the setting / light / pose of these prior prompts "
                "(this series AND earlier series in the same category):\n"
                + "\n".join(f"  - {a}" for a in avoid)
            )
        variety = "".join(parts)
    # Framing target — a rotated per-scene nudge so the series spreads across
    # orientations + shot types instead of defaulting to portrait/medium.
    framing_variety = ""
    if framing_target:
        o, s = framing_target
        framing_variety = (
            f"\n\nASSIGNED FRAMING for THIS image: orientation={o}, shot_type={s}. "
            f"COMPOSE THE SCENE TO FIT IT and emit exactly this in your JSON. This "
            f"image's job in the series is to be the {o} {s} shot — write a scene "
            f"that genuinely works in that frame (e.g. a {s} demands you actually "
            f"frame the body that way). Only deviate if {o}/{s} would truly break "
            f"this specific shot, and if so explain why in framing_rationale. A "
            f"sellable series MUST vary — never portrait/medium every time."
        )
    # Per-image subject look (sampled from the creative-direction look pools) so
    # the series shows wide variety rather than the same woman every time.
    look_variety = ""
    if look_target:
        look_variety = (
            f"\n\nSUBJECT LOOK for THIS image (vary her across the series): she has "
            f"{look_target}. She is a striking, sexy young ADULT woman — describe her "
            f"beauty, allure and figure attractively and explicitly within the tier."
        )
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
        f"{variety}{framing_variety}{look_variety}\n\n"
        'Write ONE excellent photograph-prompt at the level of the exemplars, '
        'in the target look above, honoring the TIER above EXACTLY. The TIER '
        'wins over the look\'s wardrobe descriptors — at T3+, the body is '
        'bare and any silk/gown/lace is set dressing (pooled, draped over a '
        'chaise, fallen) rather than worn.\n\n'
        'FRAMING DECISION — choose what best serves THIS scene (see the FRAMING '
        '& COMPOSITION rules in your instructions):\n'
        '  orientation: "portrait" (tall 2:3 — intimate close-ups, standing full-'
        'body) | "landscape" (wide 3:2 — reclining, environmental, two-figure-'
        'wide settings) | "square" (1:1 — balanced medium shots).\n'
        '  shot_type: "close_up" (head/face fills frame) | "bust" (head to chest) '
        '| "medium" (head to waist/hips) | "full_body" (head to feet, ALL limbs '
        'coherent) | "wide_environmental" (subject + setting co-equal).\n'
        '  Make the prompt PHYSICALLY MATCH the chosen framing (a close_up prompt '
        'must describe the face/gaze in detail and crop tight; a full_body prompt '
        'must place the whole body in the scene with clear, correct anatomy).\n\n'
        'Return JSON: {"prompt": "<prompt text>", "orientation": "<portrait|square|'
        'landscape>", "shot_type": "<close_up|bust|medium|full_body|wide_'
        'environmental>", "framing_rationale": "<1-2 sentences>"}'
    )
    result = client.generate_json(
        _build_system_prompt(word_band),
        user_prompt,
        model=model_tag,
        temperature=temperature,
        num_predict=1200,
        schema=_PromptOut,
    )
    return {
        "prompt": str(result["prompt"]).strip(),
        "orientation": str(result.get("orientation", "portrait")),
        "shot_type": str(result.get("shot_type", "medium")),
        "framing_rationale": str(result.get("framing_rationale", "")).strip(),
    }


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
    require_sfw: bool = False,
    extra_directive: str = "",
    client: "object | None" = None,
    seed_avoid: "list[str] | None" = None,
    seed_banned_openers: "list[str] | None" = None,
    run_offset: int = 0,
) -> list[dict]:
    """Generate ``count`` prompts. ``sub_looks`` (from the niche selector)
    overrides the default 3; the per-scene look rotates through them.

    Inline quality gate (2026-06): each candidate is scored by
    ``audit_prompts.score_prompt``; below ``audit_threshold`` it regenerates,
    keeping the best-scoring attempt as a fallback so a scene is never dropped
    purely for a soft-quality miss. Pydantic guards (safety/word-band/mirror)
    still hard-reject before scoring."""
    global _ACTIVE_WORD_BAND, _ACTIVE_REQUIRE_SFW
    _ACTIVE_WORD_BAND = word_band
    _ACTIVE_REQUIRE_SFW = require_sfw

    looks = sub_looks or [s for s in SUB_LOOKS]
    score_fn = None
    if audit_gate:
        try:  # lazy — keeps art_director import light + decoupled
            from scripts.audit_prompts import score_prompt as score_fn  # type: ignore
        except Exception as exc:  # noqa: BLE001
            print(f"  (audit gate disabled — score_prompt unavailable: {exc})",
                  file=sys.stderr, flush=True)
            score_fn = None

    # LLMClientPool routes by the model tag's backend (Ollama / LM Studio / MLX)
    # via the registry, so the default Gemma tag → LM Studio automatically.
    client = client or LLMClientPool()
    out: list[dict] = []
    # Cross-series memory: seed the anti-repetition lists with PRIOR same-niche
    # prompts (openers/signatures, supplied by art_series._load_niche_history) so
    # this series deliberately differs from past ones; they keep growing
    # within-series as before. run_offset rotates the sub-look / framing / look
    # sequences so the SEQUENCE also differs per run (= the niche's prior-run
    # count) — without it, re-running a niche reproduces it verbatim.
    avoid: list[str] = list(seed_avoid or [])
    banned_openers: list[str] = list(seed_banned_openers or [])
    for i in range(count):
        sub_look = looks[(i + run_offset) % len(looks)]
        look_label = sub_look.split(" — ")[0]
        best: tuple[dict, float, list[str]] | None = None  # (candidate, score, issues)
        last_err = None
        for attempt in range(max_attempts):
            try:
                cand = generate_one(
                    client,
                    brief=brief,
                    tier=tier,
                    sub_look=sub_look,
                    avoid=avoid,
                    banned_openers=banned_openers,
                    model_tag=model_tag,
                    temperature=temperature,
                    word_band=word_band,
                    extra_directive=extra_directive,
                    framing_target=FRAMING_TARGETS[(i + run_offset) % len(FRAMING_TARGETS)],
                    look_target=_creative_look(i + run_offset),
                )
            except Exception as exc:  # noqa: BLE001 — Pydantic/safety reject → retry
                last_err = exc
                print(f"  (scene {i + 1} attempt {attempt + 1} rejected: {exc})",
                      file=sys.stderr, flush=True)
                continue

            score, issues = (score_fn(cand["prompt"], tier) if score_fn else (10.0, []))
            if best is None or score > best[1]:
                best = (cand, score, issues)
            if score >= audit_threshold:
                break
            print(f"  (scene {i + 1} attempt {attempt + 1} audit {score:.1f}"
                  f"<{audit_threshold} — regenerating; issues={issues[:2]})",
                  file=sys.stderr, flush=True)

        if best is None:
            print(f"  !! scene {i + 1} failed after {max_attempts} attempts: "
                  f"{last_err}", file=sys.stderr, flush=True)
            continue

        cand, score, issues = best
        if score < audit_threshold:
            print(f"  (scene {i + 1} shipping best audit={score:.1f} after "
                  f"{max_attempts} attempts; issues={issues[:3]})",
                  file=sys.stderr, flush=True)
        ptext = cand["prompt"]
        framing = f"{cand['orientation']}/{cand['shot_type']}"
        out.append({"look": look_label, "prompt": ptext,
                    "orientation": cand["orientation"], "shot_type": cand["shot_type"],
                    "framing_rationale": cand["framing_rationale"],
                    "audit_score": round(score, 2)})
        avoid.append(_signature(ptext))
        banned_openers.append(_opener(ptext))
        print(f"\n[{i + 1}/{count}] {look_label} [{framing}] (audit {score:.1f}, "
              f"{len(ptext.split())} words)\n{ptext}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="LLM-direct art-director prompts")
    ap.add_argument("--brief", required=True, help="creative brief / theme")
    ap.add_argument("--tier", default="T3_artnude", choices=list(TIER_DIRECTIVES))
    ap.add_argument("--count", type=int, default=6)
    ap.add_argument("--model-tag", default=DEFAULT_LLM_TAG,
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
        LLMClientPool().unload_all()  # cascade — frees Ollama + LM Studio + MLX
    except Exception:  # noqa: BLE001
        pass
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
