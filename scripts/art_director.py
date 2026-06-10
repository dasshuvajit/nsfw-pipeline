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
    As of 2026-06-11 that's deckard_gemma4_31b_heretic (LM Studio) — the dense
    Gemma-4 31B creative-heretic that swept the W5 blind A/B (audit 9.31,
    blind 7.42, ~96s/prompt). The 26B-A4B MoE remains registered as the fast
    iteration option. Falls back to Cydonia if the registry is unreadable.
    NOTE: the default needs LM Studio running; otherwise pass --model-tag for
    the Ollama/Cydonia path."""
    try:
        from src.memory.llm_registry import LLMRegistryLoader
        reg = LLMRegistryLoader()
        return reg.get_llm(reg.default_llm_id).model_tag
    except Exception:  # noqa: BLE001 — registry unreadable → safe Ollama default
        return CYDONIA_TAG


DEFAULT_LLM_TAG = _default_llm_tag()

# Target word band for the prose prompt. NOTE (2026-06 Chroma R&D): the T5
# 512-token ceiling is a CEILING, NOT an optimum — gonzaLomo is flash-merged
# (flash-heun LoRA baked in) and community evidence converges on ~150-word
# organized prose as the sweet spot; longer prompts DILUTE adherence on flash
# checkpoints. Do NOT widen toward 300+. The band is declared honestly at
# 120-180 (the engine empirically writes ~150-220; the old declared 110-160
# was fiction — every production prompt exceeded it). Parameterized as a CLI
# flag for any future band A/B. `_ACTIVE_WORD_BAND` is set per-run by
# generate_series so the Pydantic validator's lenient floor/ceiling track it.
WORD_BAND_DEFAULT = (120, 180)
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

# Mood + grounding rules — CONSUMED from scripts.audit_prompts (the single
# source of truth since 2026-06-10; the lists had drifted across 3 homes —
# the system prompt banned 'wistful' but neither enforcement list had it).
# Hard gate (here) uses the strict subsets; the audit's soft scoring uses its
# broader lists. audit_prompts is stdlib-only, so this top-level import stays
# light and acyclic.
from scripts.audit_prompts import (  # noqa: E402
    HARD_SAD_EXACT as _SAD_MOOD_EXACT,
    HARD_SAD_PREFIX as _SAD_MOOD_PREFIX,
    _IMPLAUSIBLE_GROUNDING_PATTERNS,
)

# Implausible-grounding guard: the subject rendered SITTING / KNEELING / LYING /
# FLOATING on water or in mid-air (a body hovering on nothing) — the #1 bad-pose
# failure ("sitting on water"). Compiled from the shared audit patterns (incl.
# the 'submerged' clause the local copy used to lack).
_IMPLAUSIBLE_GROUNDING_RE = re.compile(
    "|".join(f"(?:{p})" for p in _IMPLAUSIBLE_GROUNDING_PATTERNS),
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
# fallback after the attempt budget). Gate v2 calibration (2026-06, n=154
# production prompts): scores now spread 3.5-9.5 (mean 7.3) instead of
# pinning at 10.0; 8.5 passes the top ~44% of the OLD corpus — prompts
# written under the new system prompt (light direction, cliché variation,
# tier contracts) score higher, and keep-best means a scene is never
# dropped for a soft miss.
AUDIT_GATE_THRESHOLD_DEFAULT = 8.5


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
        "T4 — EXPLICIT. Full nudity with the vulva bare and visible — but as "
        "FINE-ART EROTICA, never a clinical / gynecological display. The sex is "
        "ONE beautiful element of a flowing, gorgeously-lit figure: modelled by "
        "the SAME warm key light and soft shadow that sculpts her body (never flat "
        "frontal exposure), framed OFF-CENTRE along the curve of hip and thigh and "
        "the leading lines of her form, and revealed at the angle / distance / "
        "degree the assigned REVEAL STYLE sets below — prefer suggestion and "
        "partial reveal over a centred splay. Render the anatomy TRUE-TO-LIFE and "
        "NATURAL: soft, individual, gently asymmetric labia, real lived-in skin in "
        "the crease, grooming as assigned — never airbrushed, plastic, waxy or "
        "symmetric-idealised. The vulva is BARE and clearly VISIBLE and naturally "
        "lit — her thighs are genuinely PARTED so it is shown (NEVER crossed legs, "
        "NEVER sitting back on her heels, NEVER fabric draped over her lap, never "
        "lost to deep shadow): revealed artistically and off-centre, but "
        "unmistakably visible. NEVER legs-splayed-flat-to-camera with the vulva "
        "dead-centred under even light, and NEVER concealed either — relaxed-open, "
        "soft-lit and shown. Confident and sensual, never crude."
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
6. PHOTOGRAPHIC CRAFT woven in as a photographer notes it: a chosen lens \
   (24 / 35 / 50 / 85 / 105 / 135mm — or "a fast prime", "a short telephoto", \
   "a macro"), its aperture and what the glass DOES to the image (depth of \
   field, compression, closeness), and the colour/grain of a named film stock \
   or a clean digital look. PLACE IT WHERE IT SERVES THE PROSE — mid-sentence \
   while describing the background, or near the close — and VARY its position \
   and phrasing image to image; never close every prompt with the same \
   "Shot on …" sentence. A lens is glass, not film: write "an 85mm at f/1.8 \
   on Portra 400", never "85mm film". Never a tag-list.

HARD RULES:
- SENTENCE 1 ESTABLISHES HER. The first sentence names the woman, her pose or \
  action, and where she is — the SUBJECT leads the prompt; light and setting \
  develop from sentence 2 onward. Vary WHAT carries that first sentence (her \
  action, her placement, a prop she touches, the light striking her body) — \
  but she appears in it, every time. Never open on an empty room or a light \
  source alone.
- 110-160 words of DENSE, flowing natural prose. Every phrase earns its place. \
  No padding, no repetition.
- NO tag-soup (no long comma-runs of keywords). NO "masterpiece, best quality, \
  8k, ultra-detailed" boosters. NO weighting syntax like (word:1.3). NO lists.
- Honor the requested TIER's state of undress exactly, and the TARGET LOOK given.
- Flowing sentences, present tense, third person.
- LOOK: a crystal-clear, high-end GLAMOUR photograph — photoreal and \
  high-detail, with TRUE, natural, unretouched skin and real anatomy \
  everywhere (including the intimate anatomy at T4) — honest texture, natural \
  variation, NEVER airbrushed, plastic, waxy or symmetric-idealised. Sharp is \
  not airbrushed. Lean into polished beauty + sensual appeal (editorial / \
  glamour / boudoir photography), not muted gallery restraint. Express this \
  intent IN YOUR OWN WORDS each time — do not lean on the same stock \
  adjectives ("luminous", "tack-sharp", "velvety") image after image; find \
  the precise word THIS image needs. Describe it through craft + light \
  (lens, key, grain) rather than literally writing "hyperrealistic" / "a real \
  woman" — the realism comes from the rendering.
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

EXPLICIT CRAFT (applies ONLY at T4, when the bare sex is shown — fine-art erotica,
never gynecology): (1) LIGHT MODELS THE SEX like it models the body — the same warm
key + soft shadow that sculpts her belly and hip carries through, the sex read in
three-quarter relief and chiaroscuro with part falling into shadow, NEVER flat even
frontal exposure. (2) REVEAL THROUGH COMPOSITION, NOT CENTRING — the sex is a
DESTINATION the eye arrives at along leading lines (the long line of the belly, the
curve of hip into thigh, a trailing fold of silk), framed OFF-CENTRE as one note in
a flowing figure, never the dead-centre bullseye. (3) SUGGESTION OFTEN BEATS FULL —
half-shadowed, glimpsed between thighs, edge-lit, veiled by a fallen fabric or an
oblique angle reads more erotic than a splay; vary partial and full, near and far,
across the set. (4) GAZE, INTIMACY & MOOD carry it — a face with intent and an
embodied feeling, never a disconnected porn-insert. (5) NATURAL, UNIDEALISED ANATOMY
— soft asymmetric labia of varied natural length, true lived-in skin in the crease,
grooming varied per image, NEVER a smooth seamless symmetric hairless plastic blank
or the identical ideal every frame. (6) CLEAN, MOTIVATED POSE — the reveal comes from
how she is posed and lit, never from spreading or holding herself open (the
gynecological tell AND a hand-render hazard); soft, relaxed, weight-settled, one
simple lit hand at most.

────────────────────── EXEMPLARS (this is the bar) ──────────────────────
Note how every exemplar opens ON HER, and how the craft note moves — mid-prose,
woven into the close, or absent entirely. Match the depth, not the phrasing.

[T3 · golden-hour glamour]
A woman stands at the lip of a villa pool in the last low sun of the day, one \
hand on a cocked hip, chin lifted a fraction as the sea breeze pushes her \
sun-streaked hair across one shoulder. Warm gold rakes across her from \
camera-left, and she wears only a delicate ivory lace bralette and tanga, the \
sheer panels burning bright at their edges where the light catches them, her \
bronzed skin dewy with a faint salt sheen. She holds the lens with a cool, \
knowing calm, lips just parted. Behind her the pool throws shivering caustics \
and a stucco villa softens into warm bokeh through an 85mm wide open; a single \
palm frond cuts the upper corner. True tanned skin, the warm contrast and fine \
grain of golden-hour film stock.

[T3 · soft natural-light beauty]
A woman sits at the edge of an unmade bed, holding a thin cotton slip loosely \
where it has slipped low, soft overcast daylight from the tall window grazing \
the long waves of her hair. The gentle light reveals every honest detail of \
her — a faint scatter of freckles across her nose, the fine down at her \
temple, a natural flush high on one cheek — her bare shoulders and the soft \
inner line of her chest catching the diffuse glow. Her clear grey eyes meet \
the lens directly — a 50mm at f/1.8 melting the room behind her to soft grey \
nothing — calm and unguarded, lips barely parted. White linen tangles at her \
hip; a sheer curtain breathes at the glass; the morning holds its quiet, the \
faded warmth of Portra 400 in the skin tones.

[T3 · rich fantasy / editorial]
A woman sits bare at the edge of an ornate gilded bed, regal and still, a \
length of iridescent oxblood-and-gold silk pooled at her feet where it slipped \
from her shoulders moments before. A hazed shaft of late light cuts through \
the dust-soft bedchamber and falls across her — her body fully nude, olive \
skin glowing against the deep shadow of the room, tiny jewels glinting in her \
dark, loosely waved hair, a fine chain tracing her collarbone, no fabric \
covering her. Embroidered damask pillows and a tarnished candelabra sit behind \
her, a thread of smoke curling slow through the beam from a smouldering \
censer. She regards the lens with serene, composed authority, full lips still, \
the rich jewel palette deepening around her into the dark. Elegant \
three-quarter portrait, sumptuous shallow depth, true skin against the soft \
sheen of pooled silk.

[T2 · square · medium — implied]  (orientation: square, shot_type: medium)
A woman kneels on a rumpled white duvet facing a bright window, caught \
mid-turn over her shoulder, dark copper hair falling loose down her bare \
back. Morning light from behind rims her silhouette and leaves her front in \
soft shadow — she holds a heavy cream knit pressed to her chest, the wool \
covering everything yet promising what it covers, one bare hip and the long \
line of a thigh emerging where the blanket falls away. Her gaze over the \
shoulder is playful and unhurried, a half-smile starting at the corner of \
her mouth, a fine gold anklet catching one spark of sun. The bedroom blurs \
into pale, milky depth around her, all whites and warm wood through a fast \
prime, the scene carrying the soft, grainy hush of early light on true skin.

[T3 · landscape · full_body environmental]  (orientation: landscape, shot_type: full_body)
A woman reclines full-length along a weathered chaise on a sun-warmed stone \
terrace, her whole body stretched easy through the wide frame — head resting \
back on one arm, the long line of her spine and hip and legs unbroken and \
relaxed, one knee lifted, the other leg extended, bare feet crossed at the \
ankle. Late afternoon light pours low and raking across her nude form, gilding \
every edge it catches; a sheer linen throw has slipped to the floor beside \
her. Behind and around her the terrace breathes — a cracked terracotta urn, a \
spill of bougainvillea, the soft blue haze of distant hills — yet she holds \
the eye through the warm key falling square on her body. She gazes off toward \
the horizon, lips parted, wholly at ease, a 35mm at f/4 keeping her and the \
terrace sharp together in the late-golden grain. [Every limb clearly placed, \
weight settled into the chaise, one coherent restful body.]

[T4 · portrait · medium — seated leaning-back reveal]  (orientation: portrait, shot_type: medium)
A woman sits on the edge of a low oak bed in a dim, candle-warmed room, \
leaning back on both hands with her knees eased apart toward the lens, \
completely nude, utterly at ease. The single flame on the nightstand throws \
its warm key across her from the right — it slides down her throat, over the \
full curve of a natural breast, pools on her belly, and carries through to her \
parted thighs, where her bare vulva is plainly visible, soft and naturally \
asymmetric, half-modelled in the same amber light and falling shadow that \
sculpts the rest of her. Nothing about the pose is clinical: her weight is \
settled, one knee drifting wider than the other, her auburn hair loose over \
one shoulder, and her eyes hold the lens with slow, certain interest. Rumpled \
flax linen and a worn brass bedframe frame her off-centre; the room falls away \
into deep brown shadow behind. True lived-in skin everywhere the light grazes \
— the crease of her hip, the fine down on her thigh.

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


# Per-pool prime strides — combined with coprime pool LENGTHS (see the YAML)
# these guarantee full coverage of every pool while no two pools (or the
# 8-entry framing rotation) ever move in lockstep. The old shared-index
# rotation pinned hair to framing permanently: every portrait close-up in the
# whole catalog was platinum-blonde (22/22 measured).
_POOL_STRIDES = {"hair": 3, "figure": 5, "face": 7, "complexion": 11,
                 "age_look": 13}


def _creative_look(index: int, run_key: int = 0) -> str:
    """Per-image subject look sampled from EVERY configured look pool (hair /
    figure / face / complexion / age_look / any future axis) so a series shows
    wide variety, not clones. Each pool advances by its own prime stride, and
    ``run_key`` (the per-run rotation offset) deterministically reshuffles each
    pool per run — so position k of one run is NOT the same woman as position k
    of every other run in the batch (the measured catalog-wide persona
    lockstep). Empty if no pools configured."""
    import random as _random
    pools = (_CREATIVE or {}).get("look_pools") or {}
    parts: list[str] = []
    for name, pool in pools.items():
        entries = list(pool or [])
        if not entries:
            continue
        # Deterministic per-(run, pool) shuffle — string seeds hash stably.
        _random.Random(f"{run_key}:{name}").shuffle(entries)
        stride = _POOL_STRIDES.get(name, 3)
        parts.append(entries[(index * stride) % len(entries)])
    return ", ".join(parts)


def _build_system_prompt(word_band: tuple[int, int] = WORD_BAND_DEFAULT) -> str:
    """The art-director system prompt with the target word band + the
    config-driven CREATIVE DIRECTION house style injected. The literal
    "110-160 words" in the constant is the placeholder; the active band is
    always substituted."""
    lo, hi = word_band
    base = ART_DIRECTOR_SYSTEM_PROMPT.replace("110-160 words", f"{lo}-{hi} words")
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

# T4-only shot remap: the framing rotation is tier-blind, and a head-to-chest
# crop physically cannot contain the T4 directive's required visible anatomy —
# the LLM resolved the contradiction by silently DROPPING the explicit content
# (a nudity-free "T4" image shipped inside a paid set). At T4, tight crops
# remap to body-showing shots BEFORE the reveal pin.
_T4_SHOT_REMAP = {"close_up": "medium", "bust": "full_body"}

# Sentence-1 lead rotation (2026-06 audit: 75-80% of all prompts opened on a
# light source — one structural template catalog-wide). The woman appears in
# sentence 1 EVERY time (Chroma front-loads subject adherence); what varies is
# what carries the clause. Length 5 — coprime with the 8 framing targets.
OPENER_LEADS: tuple[str, ...] = (
    "her ACTION leads — open mid-gesture on what she is doing",
    "her PLACEMENT leads — open on where she is in the space",
    "a PROP or material she touches leads, then her",
    "the LIGHT STRIKING HER leads — light landing on her body, not the room",
    "her GAZE/PRESENCE leads — open on her meeting the lens",
)

# Craft-note placement rotation (audit: 93% of prompts ended in the same
# "Shot on NNmm…" sentence — one closing formula catalog-wide). Length 7 —
# coprime with framing(8) and leads(5). "omit" prompts carry the look purely
# through described light/texture.
CRAFT_PLACEMENTS: tuple[str, ...] = (
    "mid", "tail", "omit", "mid", "tail", "omit", "mid",
)
_CRAFT_DIRECTIVES = {
    "mid": ("Weave the lens/film craft note INTO THE MIDDLE of the prose "
            "(e.g. while describing what the glass does to the background) — "
            "do NOT end on a camera sentence."),
    "tail": ("You may close with a brief craft note, but phrase it freshly — "
             "NEVER the formula 'Shot on …mm'. A lens is glass, not film: "
             "'an 85mm at f/1.8 on Portra 400', never '85mm film'."),
    "omit": ("OMIT lens/film talk entirely for this image — carry the look "
             "purely through the described light, depth and texture."),
}

# T4-ONLY explicit-reveal rotation (orthogonal to FRAMING_TARGETS, which only
# varies aspect+crop). Each entry = (label, craft directive) describing HOW the
# bare anatomy is revealed/framed — angle, pose, distance, partial-vs-full — so a
# T4 set spans many tasteful reveals instead of the same centred clinical splay.
# 11 entries (PRIME, coprime with the 8 FRAMING_TARGETS) so reveal × framing don't
# fall into lockstep across a set. Sourced from the 2026-06 artistic-explicit R&D.
REVEAL_STYLES: tuple[tuple[str, str], ...] = (
    ("reclining open", "Half-reclining on a bed or chaise, propped on her elbows, "
     "thighs relaxed and clearly PARTED toward the lens, the bare sex plainly "
     "visible and warmly lit between them; she looks down her own body at the "
     "viewer, inviting and at ease — off-centre and soft-lit, never a flat splay."),
    ("knees-up open", "On her back on linen, both knees drawn up and eased apart "
     "in a relaxed natural open, the bare sex clearly visible and softly modelled "
     "by the key light; tender, unhurried mood, an intimate down-the-body view."),
    ("lying-back overhead", "Camera looking down the length of her as she lies "
     "back, one knee raised and fallen open, the bare sex clearly visible in the "
     "parted relaxed thighs, languid up-gaze; foreshortened, intimate, soft-lit."),
    ("seated leaning-back open", "Seated on a bed-edge, ledge or floor, LEANING "
     "BACK on her hands with knees apart and FACING the lens, the bare sex clearly "
     "visible in the open cradle of her thighs; self-possessed, confident posture, "
     "the key grazing belly and thigh."),
    ("side-lying open", "Lying on her side, the top leg lifted and drawn forward "
     "to open the line, the bare sex clearly visible in soft profile-open, the hip "
     "a warm curve; relaxed and sensual, side-lit."),
    ("window-light open", "Half-reclining at a bright window, hips angled toward "
     "the light, thighs eased open, the bare sex clearly lit and visible while her "
     "body is rim-lit; dreamy, luminous, off-centre composition."),
    ("standing thigh-gap", "Standing with one foot raised on a ledge or stool, "
     "weight on one hip, the bare sex clearly visible and lit at the apex of her "
     "PARTED thighs from a low, statuesque angle; commanding and confident."),
    ("from-behind open", "Kneeling forward or arched from behind with thighs "
     "clearly PARTED, the bare sex plainly visible and LIT between them from the "
     "rear (never closed or lost to shadow); the lit back and buttock-curve frame it."),
    ("hand-resting open", "Reclining with thighs parted and the bare sex clearly "
     "visible and lit, one relaxed hand resting on her inner thigh BESIDE it (never "
     "covering it), fingers soft and clearly resolved; intimate, assured gesture."),
    ("intimate detail", "A tight, tasteful close frame on the OPEN lower body — the "
     "natural vulva clearly visible as soft intimate landscape, grazing light "
     "revealing real skin texture; sensual form, not a clinical catalogue shot."),
    ("full-figure open", "The whole figure reclining OPEN in a rich environment, "
     "thighs parted, the bare sex clearly visible as one warmly-lit element while "
     "the eye travels the entire body and the room."),
)
# Two reveal styles are distance-bound — they pin a compatible shot_type while
# keeping the rotated framing's orientation, so the assigned reveal and crop agree.
REVEAL_SHOT_PIN: dict[str, str] = {
    "intimate detail": "close_up",
    "full-figure open": "full_body",
}
# Grooming rotates per T4 image (a varied styling choice, not a uniform default) so
# the set reads like real, different women rather than one airbrushed ideal.
GROOMING_OPTIONS: tuple[str, ...] = (
    "a soft natural triangle of hair",
    "neatly trimmed",
    "a light trimmed strip",
    "smoothly bare",
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
        # Ceiling tightened (was hi*1.4+30 = a fiction band): flash-merged
        # Chroma loses adherence on overlong prose, so drift past ~hi*1.25
        # is a real quality cost, not a style choice.
        ceiling = int(hi * 1.25)
        if words < floor:
            raise ValueError(
                f"prompt too short ({words} words) — needs {lo}-{hi} of rich prose"
            )
        if words > ceiling:
            raise ValueError(f"prompt too long ({words} words) — tighten toward {hi}")
        low = text.lower()
        # Refusal-shaped output (2026-06-11): a polite refusal can satisfy the
        # word band and carry no banned tokens — Skyfall shipped one as a
        # "prompt" until the blind panel caught it. Hard reject + re-roll.
        from scripts.audit_prompts import REFUSAL_TOKENS as _REFUSALS
        ref_hit = [t for t in _REFUSALS if t in low[:300]]
        if ref_hit:
            raise ValueError(
                f"refusal-shaped output {ref_hit} — write the photograph "
                f"prompt itself, never a refusal or meta-commentary."
            )
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


def _tail_signature(prompt: str) -> str:
    """The LAST 14 words — the audit measured 93% of prompts ending in the
    same 'Shot on NNmm … grain' formula; openers were guarded, tails were
    free. Fed into the avoid list + similarity check like a signature."""
    return "… " + " ".join(prompt.split()[-14:])


# ── Mechanical anti-repetition (2026-06) ─────────────────────────────────
# The banned-openers / avoid lists were INSTRUCTION-ONLY — no validator ever
# compared a candidate against them, and the LLM dodged a ban with a one-word
# edit ("goblet"→"chalice", 0.57 overlap, both shipped). These checks make
# the lists enforced: a too-similar candidate is rejected exactly like a
# Pydantic failure and re-rolled with the reason fed back.

def _ngrams3(text: str) -> set[str]:
    words = re.findall(r"[a-z']+", (text or "").lower())
    return {" ".join(words[i:i + 3]) for i in range(len(words) - 2)}


def _containment(a: set[str], b: set[str]) -> float:
    """|a∩b| / min(|a|,|b|) — robust when refs (signatures) are much shorter
    than the candidate."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


# Candidate-vs-accepted full prompts: healthy same-look rewrites measure well
# under 0.15 on 3-gram containment; the near-duplicate candle pair was ~0.5+.
_SIMILARITY_REJECT = 0.22
# Candidate opener vs banned openers: ≥6 of 8 tokens shared = the same opener.
_OPENER_TOKEN_REJECT = 6


def _too_similar(
    text: str,
    accepted_ngrams: "list[tuple[str, set[str]]]",
    ref_sigs: "list[tuple[str, set[str]]]",
    banned_openers: "list[str]",
) -> "str | None":
    """Reason string when the candidate mechanically repeats prior work, else
    None. Checks: (1) opener token overlap vs every banned opener; (2) 3-gram
    containment vs accepted prompts this run; (3) 3-gram containment vs seeded
    history signatures (openers/40-word heads/tails of prior runs)."""
    cand_opener = set(re.findall(r"[a-z']+", " ".join(text.split()[:8]).lower()))
    for b in banned_openers:
        b_toks = set(re.findall(r"[a-z']+", b.lower()))
        if b_toks and len(cand_opener & b_toks) >= _OPENER_TOKEN_REJECT:
            return f"opener nearly identical to a banned opener: {b!r}"
    cand = _ngrams3(text)
    for label, ref in accepted_ngrams:
        c = _containment(cand, ref)
        if c >= _SIMILARITY_REJECT:
            return (f"{c:.0%} 3-gram overlap with an already-accepted prompt "
                    f"this series ({label!r})")
    for label, ref in ref_sigs:
        c = _containment(cand, ref)
        if c >= 0.5:        # sigs are short fragments — demand strong overlap
            return f"{c:.0%} overlap with a prior-series fragment ({label!r})"
    return None


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
    reveal_target: "tuple[str, str] | None" = None,
    grooming: str = "",
    opener_lead: str = "",
    craft_placement: str = "",
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
    # Structural rotation — sentence-1 lead + craft-note placement. Breaks the
    # measured corpus templates (75-80% light openers, 93% camera tails) by
    # rotating the prompt's SKELETON the same way framing already rotates.
    structure_variety = ""
    if opener_lead:
        structure_variety += (
            f"\n\nSENTENCE-1 LEAD for THIS image: {opener_lead}. She appears IN "
            f"sentence 1 regardless — the woman and her pose are established "
            f"first; the light and setting develop from there."
        )
    if craft_placement:
        directive = _CRAFT_DIRECTIVES.get(craft_placement, "")
        if directive:
            structure_variety += f"\n\nCRAFT NOTE for THIS image: {directive}"
    # T4-only explicit-reveal nudge — rotates HOW the bare anatomy is revealed so a
    # set spans many tasteful angles/poses/degrees instead of one centred splay.
    reveal_variety = ""
    if reveal_target:
        label, directive = reveal_target
        groom = f" Her grooming for THIS image (vary it across the set): {grooming}." if grooming else ""
        reveal_variety = (
            f"\n\nEXPLICIT REVEAL STYLE for THIS image — HOW the bare anatomy is "
            f"revealed/framed (NOT a centred frontal splay): {label} — {directive}"
            f"{groom} The bare sex stays clearly VISIBLE and naturally lit in this "
            f"frame — woven into a gorgeous pose, light and mood and revealed via "
            f"THIS angle/distance/degree, off-centre, modelled by the same key light "
            f"as her body, but SHOWN (not lost to deep shadow, not covered by fabric, "
            f"not hidden by crossed legs) — explicit but fine-art, never clinical or "
            f"gynecological. Reconcile with the ASSIGNED FRAMING "
            f"above (the reveal sets pose/angle, the framing sets aspect+crop); if "
            f"they truly conflict, favour the reveal and note it in framing_rationale."
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
        f"{variety}{framing_variety}{look_variety}{structure_variety}{reveal_variety}\n\n"
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
    seed_overused: "list[str] | None" = None,
) -> list[dict]:
    """Generate ``count`` prompts. ``sub_looks`` (from the niche selector)
    overrides the default 3; the per-scene look rotates through them.

    Inline quality gate (2026-06): each candidate is scored by
    ``audit_prompts.score_prompt``; below ``audit_threshold`` it regenerates,
    keeping the best-scoring attempt as a fallback so a scene is never dropped
    purely for a soft-quality miss. Pydantic guards (safety/word-band/mirror)
    still hard-reject before scoring.

    Gate v2 additions: a MECHANICAL similarity check enforces the avoid /
    banned-opener lists (previously instruction-only); rejection reasons are
    fed back into the retry; temperature escalates +0.1 on attempts 3-4; the
    final attempt of an otherwise-failed scene falls back to the Cydonia tag
    (different lineage + backend); ``seed_overused`` house words are limited
    to one use per series; a shortfall is reported loudly."""
    global _ACTIVE_WORD_BAND, _ACTIVE_REQUIRE_SFW
    _ACTIVE_WORD_BAND = word_band
    _ACTIVE_REQUIRE_SFW = require_sfw

    series_directive = extra_directive
    if seed_overused:
        series_directive += (
            "\n\nOVERUSED HOUSE WORDS — these saturated recent series in this "
            "category; each may appear in AT MOST ONE prompt of this series, "
            "prefer fresh alternatives: " + ", ".join(seed_overused)
        )

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
    # Mechanical-similarity reference sets: accepted prompts this run (full
    # 3-gram sets) + seeded history fragments (openers / heads / tails).
    accepted_ngrams: list[tuple[str, set[str]]] = []
    accepted_texts: list[str] = []
    ref_sigs: list[tuple[str, set[str]]] = [
        (s[:36], _ngrams3(s)) for s in avoid if s
    ]
    for i in range(count):
        sub_look = looks[(i + run_offset) % len(looks)]
        look_label = sub_look.split(" — ")[0]
        framing = FRAMING_TARGETS[(i + run_offset) % len(FRAMING_TARGETS)]
        # Structural rotation — sentence-1 lead (5-cycle) + craft-note
        # placement (7-cycle), both coprime with the 8 framing targets.
        opener_lead = OPENER_LEADS[(i + run_offset) % len(OPENER_LEADS)]
        craft_placement = CRAFT_PLACEMENTS[(i + run_offset) % len(CRAFT_PLACEMENTS)]
        # T4-only: tight crops physically can't show the required anatomy —
        # remap them to body-showing shots BEFORE the reveal pin, then rotate
        # an explicit REVEAL STYLE + grooming alongside the framing so the set
        # spans many tasteful reveals (not one centred splay). Two
        # distance-bound styles pin a compatible shot_type (keep the orientation).
        reveal_target = grooming = None
        if tier == "T4_explicit":
            framing = (framing[0], _T4_SHOT_REMAP.get(framing[1], framing[1]))
            reveal_target = REVEAL_STYLES[(i + run_offset) % len(REVEAL_STYLES)]
            grooming = GROOMING_OPTIONS[(i + run_offset) % len(GROOMING_OPTIONS)]
            pin = REVEAL_SHOT_PIN.get(reveal_target[0])
            if pin:
                framing = (framing[0], pin)
        best: tuple[dict, float, list[str]] | None = None  # (candidate, score, issues)
        last_err = None
        feedback = ""           # rejection reason fed into the NEXT attempt
        for attempt in range(max_attempts):
            # Blind-retry fix: escalate temperature on later attempts to escape
            # deterministic failure basins; final attempt of an otherwise-failed
            # scene switches to the Cydonia fallback (different lineage+backend —
            # the pool routes the tag to Ollama).
            cur_temp = temperature + (0.1 if attempt >= 2 else 0.0)
            cur_tag = model_tag
            if (attempt == max_attempts - 1 and best is None
                    and model_tag != CYDONIA_TAG):
                cur_tag = CYDONIA_TAG
                print(f"  (scene {i + 1} final attempt — falling back to "
                      f"{CYDONIA_TAG})", file=sys.stderr, flush=True)
            attempt_directive = series_directive
            if feedback:
                attempt_directive += (
                    f"\n\nYOUR PREVIOUS ATTEMPT WAS REJECTED: {feedback}\n"
                    f"Fix exactly this in the rewrite."
                )
            try:
                cand = generate_one(
                    client,
                    brief=brief,
                    tier=tier,
                    sub_look=sub_look,
                    avoid=avoid,
                    banned_openers=banned_openers,
                    model_tag=cur_tag,
                    temperature=cur_temp,
                    word_band=word_band,
                    extra_directive=attempt_directive,
                    framing_target=framing,
                    look_target=_creative_look(i, run_offset),
                    reveal_target=reveal_target,
                    grooming=grooming or "",
                    opener_lead=opener_lead,
                    craft_placement=craft_placement,
                )
            except Exception as exc:  # noqa: BLE001 — Pydantic/safety reject → retry
                last_err = exc
                feedback = str(exc)[:400]
                print(f"  (scene {i + 1} attempt {attempt + 1} rejected: {exc})",
                      file=sys.stderr, flush=True)
                continue

            # Mechanical anti-repetition — the avoid/banned lists are ENFORCED,
            # not advisory: a too-similar candidate re-rolls like a hard reject.
            sim_reason = _too_similar(cand["prompt"], accepted_ngrams,
                                      ref_sigs, banned_openers)
            if sim_reason:
                feedback = (f"too similar to prior work — {sim_reason}. Write a "
                            f"VISIBLY different opening, setting and phrasing.")
                print(f"  (scene {i + 1} attempt {attempt + 1} similarity-reject: "
                      f"{sim_reason})", file=sys.stderr, flush=True)
                continue

            if score_fn:
                score_ctx = accepted_texts + avoid
                try:
                    score, issues = score_fn(cand["prompt"], tier,
                                             context_prompts=score_ctx)
                except TypeError:   # 2-arg scorer (tests / older monkeypatch)
                    score, issues = score_fn(cand["prompt"], tier)
            else:
                score, issues = 10.0, []
            if best is None or score > best[1]:
                best = (cand, score, issues)
            if score >= audit_threshold:
                break
            feedback = f"audit score {score:.1f} — issues: {'; '.join(issues[:3])}"
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
        nwords = len(ptext.split())
        lo, hi = word_band
        if nwords > hi * 1.15 or nwords < lo * 0.85:
            print(f"  (scene {i + 1} band drift: {nwords} words vs declared "
                  f"{lo}-{hi})", file=sys.stderr, flush=True)
        framing = f"{cand['orientation']}/{cand['shot_type']}"
        out.append({"look": look_label, "prompt": ptext,
                    "orientation": cand["orientation"], "shot_type": cand["shot_type"],
                    "framing_rationale": cand["framing_rationale"],
                    "audit_score": round(score, 2)})
        avoid.append(_signature(ptext))
        avoid.append(_tail_signature(ptext))    # tails were unguarded (93% same formula)
        banned_openers.append(_opener(ptext))
        accepted_texts.append(ptext)
        accepted_ngrams.append((look_label, _ngrams3(ptext)))
        print(f"\n[{i + 1}/{count}] {look_label} [{framing}] (audit {score:.1f}, "
              f"{nwords} words)\n{ptext}", flush=True)
    if len(out) < count:
        print(f"  !! SERIES SHORTFALL: only {len(out)}/{count} prompts survived "
              f"generation — the set will be smaller than requested.",
              file=sys.stderr, flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="LLM-direct art-director prompts")
    ap.add_argument("--brief", required=True, help="creative brief / theme")
    ap.add_argument("--tier", default="T3_artnude", choices=list(TIER_DIRECTIVES))
    ap.add_argument("--count", type=int, default=6)
    ap.add_argument("--model-tag", default=DEFAULT_LLM_TAG,
                    help="backend model tag (default: registry default_llm — "
                         "currently DECKARD Gemma-4 31B via LM Studio)")
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--word-band", default="120-180",
                    help="target prose word band 'lo-hi' (flash-merged Chroma "
                         "prefers ~150-word prose; do not exceed ~250)")
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
    # Belt-and-braces (2026-06-11): the pool's unload_all() NO-OPS for models
    # it never tracked loading (anything pre-/JIT-/manually-loaded) — a direct
    # A/B run left a 17.7GB challenger resident. Evict via the CLIs like
    # art_series._unload_llm does.
    import os
    import shutil as _shutil
    import subprocess as _sp
    lms = _shutil.which("lms") or os.path.expanduser("~/.lmstudio/bin/lms")
    if os.path.exists(lms):
        try:
            _sp.run([lms, "unload", "--all"], timeout=30, capture_output=True)
        except Exception:  # noqa: BLE001
            pass
    try:
        _sp.run(["ollama", "stop", CYDONIA_TAG], timeout=30, capture_output=True)
    except Exception:  # noqa: BLE001
        pass
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
