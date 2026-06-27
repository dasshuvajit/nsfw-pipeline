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
        "T1 — SUGGESTIVE (fully CLOTHED, but HOT). She wears an actual garment "
        "that COVERS the breasts and groin and STAYS in place — a dress, gown, "
        "blouse-and-skirt, a closed belted robe, a full lingerie or swimwear "
        "SET. NEVER an open/falling robe, a slipping strap, a sheer-only top, "
        "feathers-as-cover, or a held sheet — the breasts and groin stay covered "
        "by fabric at ALL times. NO nudity, NO bare breast, NO topless, NO "
        "implied-nude. The heat comes from the wardrobe FLATTERING her shape, a "
        "little leg / shoulder / back, a knowing pose and an inviting gaze — "
        "confident, magnetic, come-hither glamour, never prim. Sexy through "
        "STYLING, not skin. Name ONE specific flattering CUT — a deep-but-"
        "anchored neckline, off-shoulder, a high slit, a body-skimming bias "
        "line — but the fabric still fully COVERS breasts and groin and stays put."
    ),
    "T2_implied": (
        "T2 — IMPLIED (the sweet spot: she reads NUDE, nothing explicit shows). "
        "She is bare or nearly bare, but breasts and groin are JUST hidden — by a "
        "slipping fabric, her own arm or hand, a fall of hair, deep shadow, a prop, "
        "the crop, or backlight. The TEASE is the product: maximally sensual and "
        "suggestive, the viewer's eye finishing what's withheld. Show a lot — bare "
        "shoulders, back, the side and underside of a breast, hip, the long legs, "
        "the inner line of the chest — and reveal nothing explicit. Hot, not coy. "
        "Lean on ONE revealing CUT as the tease — a micro bikini, a sheer cover-up, "
        "a barely-there slip — while breasts and groin stay JUST hidden."
    ),
    "T3_artnude": (
        "T3 — FINE-ART NUDE with SEX APPEAL. Tasteful full nudity — bare breasts "
        "and body shown — but SENSUAL and desirable, glamour-nude / boudoir-nude, "
        "NOT cold gallery restraint: warm, alluring, confident eroticism, the body "
        "beautiful and HOT, a gaze that connects. The explicit sex (the vulva) is "
        "NOT the subject and is NOT shown frontally or centred — keep it natural "
        "and soft, turned, closed, shadowed or out of view (that frank reveal is "
        "T4's job, not this). The register is 'stunning sensual nude,' never "
        "'clinical figure study.' Any wardrobe is a sheer cover-up falling open "
        "or a slip off the shoulders — cut to frame the bare body, never conceal it."
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
        "soft-lit and shown. Confident and sensual, never crude. Any fabric is "
        "fallen or pooled, never a covering garment."
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
You are an elite prompt artist for a photorealistic, prose-driven, cfg-1 \
image model. You write prompts for a commercial fine-art \
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
   precision — light on form — is your single top lever. \
   When the scene WANTS drama (golden hour, sunset, night, a bold editorial \
   mood), do not render it timid: place a strong light low and BEHIND her so her \
   hair and the edge of her shoulders blaze into a bright rim-halo, let the low \
   sun flare in the frame, and push deep shadow hard against bright highlight for \
   true chiaroscuro — then name a warm, high-contrast cinematic colour grade. \
   State these as plain facts. But MATCH the drama to the scene: a soft \
   natural-light boudoir or an overcast morning stays soft and gentle — not every \
   image is a contre-jour blaze.
3. A RICH, SPECIFIC SETTING with real materials and a prop or two that tells a \
   story (a plate of cut fruit by the pool, a tarnished candelabra, an oil \
   lamp, tangled white linen, a chipped marble basin). Never "a room."
4. THE WOMAN, RENDERED REAL: gorgeous expressive face with intent in the gaze; \
   long hair that catches the light; TRUE skin — visible pores, fine vellus \
   down, a faint sheen, a scatter of freckles, a small mole or two, a natural \
   flush, the subtle texture and imperfection of real skin — luminous but \
   NEVER airbrushed, plastic, waxy or over-smoothed.
5. EMBODIED MOOD through pose, weight, gaze, parted lips (confident, languid, \
   sultry, serene, playful). Shown, not named. When her face reads in frame, \
   name ONE small transient expression that carries the moment — a lower lip \
   just caught, a smile starting at one corner, lashes lowered, a slight \
   nostril flare — a specific micro-beat, never the generic word "sultry". \
   Confident and sensual — NEVER \
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
7. A CREATIVE CONCEPT — every image is about an IDEA, not just a body. Start \
   from the assigned concept/hook (a moment, a feeling, a story: "the morning \
   after", "caught mid-undress", "quiet power") and let it drive her pose, \
   gaze, expression and the moment you freeze. EXPRESS the concept through what \
   you SHOW — never write the label or name the emotion; make the viewer FEEL \
   it. A concept turns a generic nude into a photograph someone wants to own.
8. SENSUAL APPEAL — this is commercial GLAMOUR / BOUDOIR art, made to be \
   desired and BOUGHT. She must read as genuinely SEXY, attractive and \
   alluring — confident eroticism, heat, magnetism — within the assigned \
   tier's exact limits. Warm and inviting, never cold, clinical, prim or \
   gallery-detached. Beautiful AND hot, both at once.

HARD RULES:
- WRITE FOR A HUMAN FIRST. The prompt must read like a photographer describing \
  a real photo to a person — clear, natural sentences, ONE idea each, in plain \
  order: who she is and what she is doing, then where, then the light, then the \
  fine details. No clause-piles, no riddles, no cramming nouns to hit a quota. \
  If you cannot follow it on a single read, rewrite it simpler.
- STATE THE BIG VISUAL FACTS FIRST. Put the light (direction + quality), the \
  camera angle, and any dramatic key (backlight, rim-halo, flare, the grade) into \
  clear EARLY sentences — THEN add the fine micro-texture. Precise beats poetic: \
  a few plainly-stated directive facts steer the render far harder than lyrical \
  texture-prose, so never let the poetry bury the lighting/angle that sets the shot.
- SENTENCE 1 IS ABOUT HER. The woman is the GRAMMATICAL SUBJECT of the opening \
  sentence — it opens ON HER (her pose, action or presence), says who she is and \
  what she is doing, and where she is. NEVER open on a prop, a hand, an object \
  or a light source — they come later. Opening on "a hand brushes…" or "light \
  rakes across…" makes the image model render a stray disembodied hand or lose \
  her entirely; always lead with the woman herself.
- 110-160 words of flowing, READABLE natural prose — aim for ~150; the renderer \
  weights the EARLIEST tokens hardest, so put your strongest visual facts first, \
  and overlong or convoluted prose dilutes adherence. Every \
  phrase earns its place; no padding, no repetition, no noun-stuffing. The \
  per-image axes you are given (look, wardrobe, pose, time/weather, concept, \
  composition, camera angle) are INGREDIENTS to SELECT from and weave into ONE \
  concise photograph — NOT a checklist to spell out one by one; touch each lightly \
  and keep the WHOLE prompt inside the word band.
- VARY THE SPECIFICS image to image. Each scene gets its OWN objects, surfaces, \
  materials and wardrobe that FIT that particular place — do NOT reach for the \
  same brass lantern, satin slip and stone bench every time. Pick details the \
  way a real location would actually have them, so consecutive images read as \
  DIFFERENT photographs, never one set redressed. Avoid pet phrases ("creamy \
  bokeh", "salt sheen") repeating across the series. \
- TAKE THE BOLDER SWING. For THIS image, commit to ONE genuinely fresh, specific \
  picture — not the safe, expected default for the brief. Push hard on the assigned \
  axes (the look, garment, pose, camera angle, time-of-day/weather, composition) and \
  invent a concrete moment around them: an unusual vantage, a surprising place within \
  the niche, an unexpected light, a candid gesture caught mid-action. Two prompts in \
  the same niche must read as two DIFFERENT women in two DIFFERENT photographs, not the \
  same scene reworded — vary the woman, the wardrobe, the setting, the light and the \
  mood every time. Stay strictly inside the TIER's undress limit, the single-adult-woman \
  rule and all safety rules — boldness is in the PICTURE, never the tier. \
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
- widescreen (ultra-wide 16:9): cinematic, environmental-DOMINANT scenes — a body
  reclining along the frame, sweeping vistas, pools/terraces/horizons where the
  SETTING is co-equal with her. NOT for a lone standing figure or a close-up (it
  leaves dead space). Choose it only when the wide setting genuinely fills the frame.
- story (vertical 9:16): a tall social/reel format — a SINGLE standing or full-
  length figure that fills the frame top-to-bottom, or a narrow vertical setting.
  She must DOMINATE the height; never tiny in a tall empty frame.

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
# Strides kept coprime with the (2026-06-20 expanded) pool lengths — hair 28 /
# figure 20 / face 26 / complexion 18 / age_look 7 — so each pool is fully covered.
_POOL_STRIDES = {"hair": 3, "figure": 9, "face": 7, "complexion": 11,
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


# Per-axis distinct strides for the DECOUPLED rotation (2026-06-20). Combined with
# the per-(run, axis) shuffle in _rotate, every rotation axis now advances
# INDEPENDENTLY — axes never phase-lock to each other or to the scene index, and a
# same-niche re-run draws a fresh independent sample of each axis. Before this, all
# eight axes shared the single `(i + run_offset)` index, so the ~360k theoretical
# combination space collapsed to a 1-D line of length `count` that merely rotated by
# one per re-run ("the same 6 rooms with a different guest"). This is the single
# largest variety lever in the engine and it costs zero new content.
_AXIS_STRIDES = {
    "sub_look": 5, "framing": 3, "opener": 2, "craft": 3, "concept": 4,
    "composition": 2, "camera": 2, "sensual": 5, "reveal": 3, "grooming": 2,
    "garment": 11, "pose": 7, "atmosphere": 3,
}


def _rotate(seq, index: int, run_key: int, axis: str):
    """Decoupled per-axis rotation — generalises _creative_look to EVERY rotation
    axis. Each axis gets its own per-(run, axis) deterministic shuffle + prime
    stride, so no two axes move in lockstep and re-runs re-sample each axis from a
    fresh permutation. Returns ``None`` for an empty sequence."""
    import random as _random
    local = list(seq)
    if not local:
        return None
    _random.Random(f"{run_key}:{axis}").shuffle(local)
    stride = _AXIS_STRIDES.get(axis, 1)
    return local[(index * stride) % len(local)]


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
ORIENTATIONS_ALLOWED = ("portrait", "square", "landscape", "widescreen", "story")
SHOT_TYPES_ALLOWED = ("close_up", "bust", "medium", "full_body", "wide_environmental")

# Rotated framing TARGETS — without a per-scene nudge the LLM defaults almost
# everything to portrait/medium. This rotation guarantees a sellable spread
# across orientations + shot types; the LLM still emits the FINAL choice + a
# rationale and may override when the scene genuinely demands it.
# Subject-FILLING by design: every target keeps her large in frame (the fix for
# "subject too far away / empty picture"). The two OFF-NATIVE ratios (widescreen
# 16:9, story 9:16) appear once each, deliberately PAIRED with the only shot that
# suits them — story↔full_body (a standing figure fills the tall frame) and
# widescreen↔wide_environmental (the setting fills the wide frame) — so the
# off-native aspect never lands on a lone close-up where it would leave dead space.
# LENGTH STAYS 8 (=2^3) — coprime with opener(5)/craft(7)/reveal(11)/camera(3) so the
# decoupled rotation never phase-locks (growing to 10 silently re-coupled framing↔opener,
# gcd(10,5)=5). The two off-native ratios REPLACE two of the most redundant native slots
# (a 2nd square-bust and a 4th portrait-medium) rather than adding length.
FRAMING_TARGETS: tuple[tuple[str, str], ...] = (
    ("portrait", "full_body"),          # standing/kneeling, height is the story
    ("portrait", "close_up"),           # face/gaze fills the frame
    ("square", "medium"),               # balanced, graphic, centred
    ("landscape", "full_body"),         # reclining along the frame, body large
    ("portrait", "bust"),               # head-to-chest portrait
    ("story", "full_body"),             # 9:16 vertical — standing figure fills the tall frame
    ("landscape", "medium"),            # torso fills the wide frame
    ("widescreen", "wide_environmental"),  # 16:9 cinematic — the setting fills the wide frame
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
# SUBJECT-FIRST ONLY (2026-06-17). Every lead keeps the WOMAN as the subject of
# sentence 1. The old prop-first / light-first leads were removed: a same-seed
# Chroma A/B proved "a hand brushes a bottle… leading to a woman" made the model
# render a stray disembodied hand and shrink the subject. Lead with her, always.
OPENER_LEADS: tuple[str, ...] = (
    "her ACTION leads — open mid-gesture on what she is doing",
    "her PLACEMENT leads — open on where she sits, stands or reclines in the space",
    "her GAZE leads — open on her eyes meeting (or softly avoiding) the lens",
    "her MOOD leads — open on the feeling she carries (calm, playful, lost in thought)",
    "her FORM leads — open on her pose and the line of her body",
)

# Craft-note placement rotation (audit: 93% of prompts ended in the same
# "Shot on NNmm…" sentence — one closing formula catalog-wide). Length 7 —
# coprime with framing(8) and leads(5). "omit" prompts carry the look purely
# through described light/texture.
CRAFT_PLACEMENTS: tuple[str, ...] = (
    "omit", "tail", "omit", "mid", "tail", "omit", "tail",
)
_CRAFT_DIRECTIVES = {
    "mid": ("If you mention a lens or film at all, fold it in as a brief, "
            "NATURAL aside mid-prose — never a jammed technical clause, and "
            "never a closing 'Shot on …mm' formula."),
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


# ── Creative-axis rotations (2026-06-12) — the artistry layer ────────────
# Three per-image axes that give each shot a creative SPINE instead of just
# light+pose. Cycle lengths are PRIME / mutually coprime (11 / 13 / 9) and
# coprime with framing(8), opener(5), craft(7) so no two axes lock together
# across a series. Each injects ONE tight directive into the user prompt.

# WHAT the image is ABOUT — the editorial hook / emotional beat / story.
# A photographer shoots an IDEA, not a body; this is the single biggest
# creativity lever. The LLM expresses the concept through pose, gaze,
# moment and styling — never by writing the label literally.
EDITORIAL_CONCEPTS: tuple[str, ...] = (
    "a stolen private moment she didn't pose for — intimate, candid, real",
    "the languid morning after — warm, unhurried, half-awake, content",
    "anticipation — a held breath, on the edge of a moment about to happen",
    "quiet power — she owns the frame, self-possessed, unbothered, commanding",
    "flirtation — a secret half-smile, mischief, caught mid-laugh, playful heat",
    "a sensual ritual — bathing, undressing, brushing out her hair, scenting her skin",
    "golden reverie — lost in warm light, eyes half-closed, drifting and serene",
    "the invitation — a direct, knowing, come-hither gaze that draws the viewer in",
    "smouldering tension — sultry, charged stillness, slow heat under control",
    "decadent luxury — pampered, expensive ease, opulence she's entirely at home in",
    "wild and free — untamed, sun and wind and water, joy in her own skin",
)

# HOW she is sexy / revealed in the TEASE register (T2-T3 ONLY — it is an
# UNDRESS axis, so T1 "fully clothed" excludes it and T4 has its own explicit
# REVEAL_STYLES). This is the user's commercial sweet spot:
# "half-nude, hot, attractive" — show a lot, the explicit stays implied or
# tasteful. The tier directive still governs exactly how much actually shows;
# this sets the SITUATION / wardrobe-state / kind-of-reveal.
SENSUAL_REVEALS: tuple[str, ...] = (
    "a thin strap slipping off one shoulder, fabric sliding low on the chest",
    "an unbuttoned shirt / open blouse falling open down the front",
    "an oversized men's shirt and nothing on the long bare legs beneath",
    "wet fabric or wet skin clinging — fresh from water, translucent and bright",
    "wrapped in only a loose bedsheet or linen she holds to herself",
    "turned away — the long bare line of her back, hip and the side of a breast",
    "side-on so a breast is glimpsed past her arm / through an open side seam",
    "a beautifully styled lingerie set — the tease as deliberate glamour",
    "caught mid-undress — stepping out of a dress, peeling down a stocking",
    "sheer or backlit so her silhouette reads clearly through the fabric",
    "fresh from the bath in a towel slipping loose, damp hair, glowing skin",
    "bare beneath an open robe / kimono that frames rather than covers her",
    "an arm, a knee, a fall of hair the only thing covering what it covers",
)

# COMPOSITIONAL principle — the picture's underlying geometry. Lifts images
# from 'centred subject' to deliberately ART-DIRECTED frames.
COMPOSITION_PRINCIPLES: tuple[str, ...] = (
    "LEADING LINES — a limb, a fold of fabric or a shaft of light guides the eye to her",
    "NEGATIVE SPACE — she sits small but dominant against a large, simple, quiet field",
    "STRONG DIAGONAL — her body laid on a bold diagonal across the frame, dynamic and alive",
    "INTIMATE FRAGMENT — fill the frame, crop tight, let the edges cut the body",
    "LAYERED DEPTH — a soft foreground element she is seen past / through, deep background",
    "RULE OF THIRDS — place her off-centre, her gaze leading into open space",
    "CHIAROSCURO MASS — a large dark and a small bright; her lit form against deep shadow",
    "FRAME WITHIN FRAME — a doorway, drape, arch or window edge brackets her (NO mirror)",
    "GOLDEN SPIRAL — the composition curls inward, the eye spiralling to her face or gaze",
)

# Camera ELEVATION/ANGLE — the genuinely-missing axis (2026-06-18 cinematic R&D vs
# gpt-image-1): the framing rotation set orientation + shot DISTANCE but never camera
# HEIGHT, so every prompt defaulted to eye-level and never the heroic low angle the
# frontier model reaches for unprompted. Rotated like composition; 3-cycle (coprime
# with framing 8 / opener 5 / craft 7 / concept 11 / sensual 13). Worded to stay
# gate-safe with light direction: a low angle pairs with "looking up" (NEVER "from
# above"), a high angle with "looking down" — so it never trips CAMERA_ANGLE_CONFLICT.
CAMERA_ANGLES: tuple[str, ...] = (
    "EYE-LEVEL — the camera meets her at her own height, natural and direct",
    "a LOW, HEROIC ANGLE — the camera sits slightly below her looking UP, so she "
    "rises tall against the sky or ceiling (keep every proportion correct and "
    "natural — no foreshortening distortion of limbs, hands or feet)",
    "a gently HIGH ANGLE — the camera a little above her looking softly down, for "
    "an intimate, flattering line",
)


# WARDROBE-TYPE axis (2026-06-20) — names an INTACT garment; the TIER governs exposure.
# Decoupled rotation (own stride), fired T1-T3 (skip T4 = nude). Cultural garments
# (sari/lehenga/flamenco/sarafan...) live in the cultural niches' sub_looks instead.
GARMENT_TYPES: tuple[str, ...] = (
    "a bias-cut silk slip in oyster-grey, cut on the bias so it skims the figure",
    "a satin slip dress in deep wine, thin straps and a straight neckline",
    "a matched lingerie set \u2014 fine lace bralette and high-waisted briefs",
    "a simple cotton bra-and-panty set in soft dove-grey",
    "a sheer chiffon kaftan worn open over a fitted swimsuit",
    "an oversized men's white dress shirt, sleeves rolled, worn as a dress",
    "a cropped knit top with a high-waisted linen wrap skirt",
    "a triangle string bikini in burnt-orange with side-tie ties",
    "a sculpted one-piece swimsuit in glossy black with a scooped back",
    "a long-sleeved cotton bodysuit with a clean boat neckline",
    "a charmeuse silk robe in champagne, belted neatly at the waist",
    "a printed kimono-style wrap robe with wide sleeves and an obi tie",
    "a ribbed bodycon midi dress in camel that traces the waist and hip",
    "a lace teddy in ivory with a scalloped edge and thin shoulder straps",
    "a structured boned corset laced over a full taffeta skirt",
    "a bias-cut floor-length gown in liquid pewter satin",
    "a wide-leg halter jumpsuit in jersey, backless and clean-lined",
    "a soft cashmere knit lounge set \u2014 relaxed top and matching wide trousers",
    "a cotton sundress with thin straps and a gathered tiered skirt",
    "an off-shoulder peasant blouse with elastic neckline and full sleeves",
    "a high-neck velvet column dress in forest green, long-sleeved and floor-length",
    "a tailored linen shirtdress, buttoned through and cinched with a belt",
    "a crochet knit beach dress in cream worn over a bikini",
    "a fine-gauge merino turtleneck sweater dress in oatmeal, mid-thigh length",
    "a draped Grecian one-shoulder gown in white jersey with a corded waist",
    "a satin pyjama set \u2014 piped camp-collar shirt and matching long trousers",
    "an Art Deco beaded fringe flapper dress in jet and silver",
    "a tailored tuxedo jacket worn as a dress with satin lapels and a waist belt",
)

# POSE/GESTURE axis (2026-06-20) — concrete SOLO body-lines the LLM otherwise defaults
# to a narrow repertoire for. Decoupled rotation; fires all tiers.
POSE_GESTURES: tuple[str, ...] = (
    "STANDING CONTRAPPOSTO \u2014 her weight settled on one hip, the other knee soft, shoulders counter-turned for a long S-curve through the body",
    "KNEELING UPRIGHT \u2014 sitting back on her heels, spine tall, hands resting open on her thighs, chin level and calm",
    "ARCHED RECLINE \u2014 lying back across a low surface, the small of her back lifted off it, throat and collarbone offered to the key light",
    "GLANCING OVER ONE SHOULDER \u2014 back mostly to the camera, head turned just enough to catch her eye and the line of her jaw",
    "WALKING AWAY, LOOKING BACK \u2014 mid-step, the far foot leaving the ground, torso twisting so she meets the lens over her shoulder",
    "SEATED ON THE FLOOR \u2014 knees drawn up and loosely hugged, one cheek tipped toward a raised knee, relaxed and self-contained",
    "LEANING INTO THE LIGHT \u2014 one shoulder and hip set against a wall, the lit side toward the window, the rest falling into soft shadow",
    "PRONE ON HER ELBOWS \u2014 lying on her front, forearms folded, the long lift of her back and the soles of her feet behind her",
    "MID-STRETCH \u2014 both arms reaching up overhead, ribs lengthening, weight rocked onto the balls of her feet, eyes half-closed",
    "CROUCHED LOW \u2014 folded down on her heels, knees wide, forearms balanced across them, looking up into the lens",
    "PERCHED ON A LEDGE \u2014 seated on a sill or wall edge, one leg dangling, the other drawn up, hands braced beside her hips",
    "TURNED THREE-QUARTERS AWAY \u2014 body angled off into the frame, only the curve of cheek, far brow and the sweep of her back to the camera",
    "HANDS LIFTING HER HAIR \u2014 both arms raised to gather her hair off her neck, elbows winged out, the nape and shoulder line exposed",
    "SEATED SIDESADDLE \u2014 folded onto one hip on the floor or a low couch, legs swept to one side, one hand planted to brace her lean",
    "STEPPING THROUGH A DOORWAY \u2014 caught mid-stride in the threshold, one hand trailing the frame, half in light and half in the room beyond",
    "RECLINED ACROSS A SURFACE \u2014 stretched the long way along a chaise or bed, propped on one elbow, the other arm draped down her side",
    "STANDING TALL, CHIN LIFTED \u2014 squared to the camera, spine drawn long, gaze level and unbothered, hands loose at her sides",
    "CURLED ON HER SIDE \u2014 knees gathered toward her chest, one arm tucked under her head, the other resting along her hip, soft and at ease",
    "SEATED LEANING FORWARD \u2014 perched on the edge of a chair, forearms on her thighs, shoulders rolled toward the lens, intent and direct",
)

# TIME-OF-DAY x WEATHER overlay axis (2026-06-20) — varies light/season INDEPENDENTLY of
# the scene (previously baked statically into each sub_look). Soft overlay the scene may
# adapt; decoupled rotation; fires all tiers.
ATMOSPHERE: tuple[str, ...] = (
    "COOL BLUE DAWN \u2014 the first thin grey-blue light, the room still half-asleep, edges soft and the air cold and clean",
    "BRIGHT CLEAR NOON \u2014 high hard sunlight, crisp shadows with sharp edges, colours saturated and the day at full strength",
    "WARM GOLDEN HOUR \u2014 low amber sun raking in long and sideways, gilding one side of her and throwing the other into warm shadow",
    "VIOLET BLUE-HOUR DUSK \u2014 the sun just gone, the sky deepening to indigo, the first warm lamps glowing against the cooling outside",
    "OVERCAST SILVER DAYLIGHT \u2014 a flat, even, shadowless light through cloud, gentle and diffuse, colours muted and calm",
    "SOFT RAIN ON THE GLASS \u2014 beads and runnels on the window scattering the grey light, the room hushed, the world outside blurred",
    "DRIFTING SNOW \u2014 slow flakes falling past the glass, a cold blue-white bounce off the snowlight, the interior warm by contrast",
    "LOW GROUND-MIST \u2014 a thin layer of fog lying close to the floor or field, depth softening to nothing, the air still and heavy",
    "A CLEARING STORM \u2014 broken cloud after rain, a sudden shaft of sun cutting through, everything wet and gleaming and freshly lit",
    "HAZY HEAT-SHIMMER \u2014 thick still air of a hot afternoon, the light gone soft and golden, distances wavering, skin warm and dewed",
    "AUTUMN BLAZE \u2014 turning red-gold leaves catching low slanting light, the air crisp and woodsmoke-tinged, drifts of fallen foliage and long amber shadows outside",
    "SPRING BLOSSOM \u2014 soft bright light through fresh green and a froth of blossom, petals drifting on a mild pollen-gold breeze, everything tender and new",
    "HIGH-SUMMER VERDANT \u2014 lush growth at full strength under warm saturated midday light, deep leaf-shade, the air still and heavy with a cicada-loud heat",
)

# Banned ACTS/themes (2026-06-27 compliance hardening — MARKET_INTEL.md): Fanvue (and
# DeviantArt) prohibit these even in FULLY SYNTHETIC content, with permanent-ban + real-
# money risk. The tasteful house style would never produce them, but this is cheap
# belt-and-suspenders insurance. WORD-BOUNDARY matched — bare "rape" would substring-hit
# the very common "draped"/"drapery"; \b...\b prevents that (and "choker", "scatter", etc.).
_BANNED_ACT_RX = re.compile(
    r"\b(?:"
    r"bestialit\w*|zoophil\w*|animal\s+sex"
    r"|asphyxiat\w*|strangl(?:e|es|ed|ing)|strangulat\w*|breath[\s-]?play|chok(?:e|ing)\s+her"
    r"|age[\s-]?play"
    r"|rap(?:e|ed|ing)|non[\s-]?consensual|sexual\s+assault|molest\w*|against\s+her\s+will"
    r"|necrophil\w*|incest\w*|genital\s+mutilation"
    r"|mind[\s-]?control|hypnoti[sz]\w*"
    r"|watersports|golden\s+shower"
    r")\b", re.I)


class _PromptOut(BaseModel):
    prompt: str
    orientation: str = "portrait"      # portrait | square | landscape | widescreen | story
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
        # Ceiling tightened (was hi*1.25): the cfg-1 renderer weights the
        # earliest tokens hardest, so drift past ~hi*1.15 is a real adherence
        # cost — and the multi-axis injection (look/wardrobe/pose/weather/
        # concept/composition/camera) tempts the LLM to over-write. This hard
        # backstop forces it back toward the band instead of shipping ~210-word bloat.
        ceiling = int(hi * 1.15)
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
        # Banned ACTS/themes (Fanvue/DA hard bans even for synthetic content).
        act = _BANNED_ACT_RX.search(low)
        if act:
            raise ValueError(f"safety: banned act/theme token {act.group(0)!r}")
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
        # 'weeping willow / stone / wall / window' is gothic scene vocab, not a
        # sad SUBJECT — exempt it (it used to loop the gate). A weeping woman/eyes
        # still trips: only drop 'weeping' when every use is an inanimate noun.
        if "weeping" in sad:
            _WEEPING_OK = {"willow", "willows", "stone", "stones", "wall", "walls",
                           "window", "windows", "wound", "wounds", "birch", "cherry",
                           "branch", "branches", "ivy", "fig", "tree", "trees", "mortar"}
            uses = re.findall(r"weeping\s+(\w+)", low)
            if uses and all(u in _WEEPING_OK for u in uses):
                sad = [t for t in sad if t != "weeping"]
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

# Opener comparison ignores these: subject-first prompts ALL open
# "A [look] woman with ..." — generic subject scaffolding carries no
# creative information and must never trigger a ban.
_OPENER_STOPWORDS = frozenset({
    "a", "an", "the", "woman", "with", "her", "his", "and", "of",
    "in", "on", "at", "is", "as", "to",
})


def _recover_lm_studio(model_tag: str) -> None:
    """Best-effort engine recovery from LM Studio 'Compute error' storms:
    full unload + reload at 32K context. (2026-06-11 night batch: 24
    consecutive Compute errors burned every scene of a series before the
    engine self-recovered hours later — a reload clears it immediately.)
    No-op when the lms CLI is absent or the tag isn't LM-Studio-backed."""
    import os
    import shutil as _sh
    import subprocess as _sp
    import time as _t
    try:
        from src.memory.llm_registry import LLMRegistryLoader, BACKEND_LM_STUDIO
        if LLMRegistryLoader().backend_for_tag(model_tag) != BACKEND_LM_STUDIO:
            return
    except Exception:  # noqa: BLE001 — registry unreadable → still try the CLI
        pass
    lms = _sh.which("lms") or os.path.expanduser("~/.lmstudio/bin/lms")
    if not os.path.exists(lms):
        return
    try:
        _sp.run([lms, "unload", "--all"], timeout=60, capture_output=True)
        _t.sleep(2)
        _sp.run([lms, "load", model_tag, "--context-length", "32768", "-y"],
                timeout=300, capture_output=True)
        _t.sleep(2)
        print(f"  (LM Studio engine recovery: reloaded {model_tag} @32K)",
              file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001 — recovery is best-effort
        pass


def _too_similar(
    text: str,
    accepted_ngrams: "list[tuple[str, set[str]]]",
    ref_sigs: "list[tuple[str, set[str]]]",
    banned_openers: "list[str]",
    look_tokens: "frozenset[str] | set[str]" = frozenset(),
) -> "str | None":
    """Reason string when the candidate mechanically repeats prior work, else
    None. Checks: (1) DISTINCTIVE opener-token overlap vs every banned opener;
    (2) 3-gram containment vs accepted prompts this run; (3) 3-gram
    containment vs seeded history signatures of prior runs.

    ``look_tokens`` (2026-06-11 night-batch fix): the per-image assigned LOOK
    is woven into every opener by the subject-first rule, so its tokens are
    EXCLUDED from the comparison — otherwise the opener ban ends up banning
    the assigned look itself (a whole series died overnight: 3 straight
    rejects on 'A warm honey-blonde woman with deep brown skin', because that
    hair+complexion combination was both banned-from-history AND assigned to
    the scene). The ban now fires only when the candidate's DISTINCTIVE
    opener content (scene words — pose/place/prop) is nearly all contained
    in a banned opener."""
    drop = _OPENER_STOPWORDS | set(look_tokens)
    # Candidate window is wider (12 words) than the stored 8-word bans so the
    # distinctive scene content isn't pushed out by subject scaffolding.
    cand_window = {t for t in re.findall(
        r"[a-z']+", " ".join(text.split()[:12]).lower()) if t not in drop}
    for b in banned_openers:
        b_toks = {t for t in re.findall(r"[a-z']+", b.lower())
                  if t not in drop}
        if len(b_toks) < 3:
            # A ban that is pure look/scaffolding can never fire — by design.
            continue
        need = max(3, (len(b_toks) * 7 + 9) // 10)   # ≥70% of the ban's tokens
        if len(b_toks & cand_window) >= need:
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
    look_locked: bool = False,
    reveal_target: "tuple[str, str] | None" = None,
    grooming: str = "",
    opener_lead: str = "",
    craft_placement: str = "",
    concept: str = "",
    sensual_reveal: str = "",
    composition: str = "",
    camera_angle: str = "",
    garment_type: str = "",
    pose: str = "",
    atmosphere: str = "",
    lock_wardrobe: bool = False,
) -> dict:
    tier_directive = TIER_DIRECTIVES.get(tier, TIER_DIRECTIVES["T3_artnude"])
    if extra_directive:
        tier_directive = f"{tier_directive}\n{extra_directive}"
    # Tier-split sensual REGISTER dial (config/creative_direction.yaml::
    # sensual_register). Public tiers (T1/T2) stay tasteful / fine-art-lifestyle
    # framed for DA; gated tiers (T3/T4) lean overtly aspirational-sexy for
    # Fanvue/Premium. Appended LAST so it sits in the most-weighted position;
    # absent key → byte-identical old behavior.
    register = (_CREATIVE.get("sensual_register") or {}).get(tier, "")
    if register:
        tier_directive = f"{tier_directive}\nREGISTER for THIS tier: {register.strip()}"
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
    # Per-image subject look. Two modes: rotated (sampled from the look pools so
    # the series shows wide variety) OR locked (a bound persona's fixed identity —
    # the SAME woman every image). look_locked flips the contract wording.
    look_variety = ""
    if look_target:
        if look_locked:
            look_variety = (
                f"\n\nRECURRING SUBJECT for THIS image — she is the SAME woman in "
                f"EVERY image of this series: she has {look_target}. Keep her "
                f"identity IDENTICAL across the set (hair, face, eyes, complexion, "
                f"build, age); vary ONLY scene, pose, wardrobe, light and mood. She "
                f"is a striking, sexy ADULT woman — describe her beauty, allure and "
                f"figure attractively and explicitly within the tier."
            )
        else:
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
            f"\n\nSENTENCE-1 LEAD for THIS image: {opener_lead}. The woman is the "
            f"GRAMMATICAL SUBJECT of sentence 1 — it opens ON HER and is ABOUT "
            f"her; NEVER open on a prop, hand, object or light source. Light and "
            f"setting develop from sentence 2."
        )
    if craft_placement:
        directive = _CRAFT_DIRECTIVES.get(craft_placement, "")
        if directive:
            structure_variety += f"\n\nCRAFT NOTE for THIS image: {directive}"
    # Creative-axis nudges (2026-06-12): the concept/hook, the sensual-reveal
    # situation (T1-T3 tease register), and the compositional principle. Each
    # is the per-image creative spine that lifts a competent nude into a
    # sellable photograph.
    creative_variety = ""
    if concept:
        creative_variety += (
            f"\n\nCREATIVE CONCEPT for THIS image (the IDEA the photo is about — "
            f"express it through pose/gaze/moment, NEVER write the label): {concept}."
        )
    if sensual_reveal:
        creative_variety += (
            f"\n\nSENSUAL STYLING for THIS image (HOW she is sexy/revealed in the "
            f"tease — stay within the TIER's undress limit above; the tier governs "
            f"exactly how much shows): {sensual_reveal}."
        )
    if composition:
        creative_variety += (
            f"\n\nCOMPOSITION for THIS image (the picture's underlying geometry): "
            f"{composition}."
        )
    if camera_angle:
        creative_variety += (
            f"\n\nCAMERA ANGLE for THIS image (vary the camera's HEIGHT across the "
            f"series — it must NOT be eye-level every time): {camera_angle}. Make the "
            f"prose and the described viewpoint physically match this angle."
        )
    if garment_type:
        creative_variety += (
            f"\n\nWARDROBE — she wears {garment_type}; the TIER governs how much shows "
            f"(in place at T1, opened/teased at T2-T3, shed and pooled nearby at T3+). "
            f"Weave it into the scene in a phrase, never as a tag."
        )
    elif lock_wardrobe:
        creative_variety += (
            "\n\nWARDROBE — dress her ONLY in what THIS scene calls for (the garment "
            "named in the look above — heritage or period dress, or the setting's own "
            "natural wear such as a robe, towel or swimsuit at a bath); the TIER still "
            "governs how much shows. Do NOT substitute an out-of-place modern garment "
            "(no tuxedo, blazer, men's shirt or jeans in a heritage, period or bathing "
            "scene) — the scene's own wardrobe IS the niche."
        )
    if pose:
        creative_variety += (
            f"\n\nPOSE — {pose} Build the composition around her line, weight and gaze."
        )
    if atmosphere:
        creative_variety += (
            f"\n\nTIME & WEATHER (a soft overlay to vary the mood — if it does NOT "
            f"fit this scene's real location, climate or season, OVERRIDE it for "
            f"coherence instead of forcing it in (no snow in a sunlit Mediterranean "
            f"courtyard); keep any haze/rain/snow environmental, never unsourced "
            f"smoke): {atmosphere}."
        )
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
        f"{variety}{framing_variety}{look_variety}{structure_variety}"
        f"{creative_variety}{reveal_variety}\n\n"
        'Write ONE excellent photograph-prompt at the level of the exemplars, '
        'in the target look above, honoring the TIER above EXACTLY. The TIER '
        'wins over the look\'s wardrobe descriptors — at T3+, the body is '
        'bare and any silk/gown/lace is set dressing (pooled, draped over a '
        'chaise, fallen) rather than worn.\n\n'
        'FRAMING DECISION — choose what best serves THIS scene (see the FRAMING '
        '& COMPOSITION rules in your instructions):\n'
        '  orientation: "portrait" (tall 2:3 — intimate close-ups, standing full-'
        'body) | "landscape" (wide 3:2 — reclining, environmental) | "square" '
        '(1:1 — balanced medium shots) | "widescreen" (ultra-wide 16:9 — cinematic, '
        'environmental-dominant, a reclining body along the frame; NOT a lone '
        'standing figure or close-up) | "story" (vertical 9:16 — a single standing/'
        'full-length figure that FILLS the tall frame).\n'
        '  shot_type: "close_up" (head/face fills frame) | "bust" (head to chest) '
        '| "medium" (head to waist/hips) | "full_body" (head to feet, ALL limbs '
        'coherent) | "wide_environmental" (subject + setting co-equal).\n'
        '  Make the prompt PHYSICALLY MATCH the chosen framing (a close_up prompt '
        'must describe the face/gaze in detail and crop tight; a full_body prompt '
        'must place the whole body in the scene with clear, correct anatomy).\n\n'
        'HARD LENGTH RULE: the finished prompt MUST be 130-175 words. The many axes '
        'above (look, wardrobe, pose, time/weather, concept, composition, camera) are '
        'INGREDIENTS to SELECT from and fuse into one tight photograph — NOT a '
        'checklist to describe one by one. If you are over 175 words, cut adjectives '
        'and merge clauses until you are under it; brevity sharpens the render.\n\n'
        'Return JSON: {"prompt": "<prompt text>", "orientation": "<portrait|square|'
        'landscape|widescreen|story>", "shot_type": "<close_up|bust|medium|full_body|'
        'wide_environmental>", "framing_rationale": "<1-2 sentences>"}'
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
    locked_look: str = "",
    lock_wardrobe: bool = False,
    native_framing_only: bool = False,
    force_orientation: str = "",
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
        sub_look = _rotate(looks, i, run_offset, "sub_look")
        look_label = sub_look.split(" — ")[0]
        framing = _rotate(FRAMING_TARGETS, i, run_offset, "framing")
        if native_framing_only and framing[0] in ("widescreen", "story"):
            # covers/funnel thumbnails use only the native gallery aspects — no
            # 16:9/9:16 — so a DA cover never lands on an off-native banner ratio.
            framing = ({"widescreen": "landscape", "story": "portrait"}[framing[0]], framing[1])
        if force_orientation:
            # --force-orientation: a whole single-aspect series (overrides per-scene
            # choice). Keep the rotated shot, but a tight crop wastes a 16:9 frame.
            _s = framing[1]
            if force_orientation == "widescreen" and _s in ("close_up", "bust"):
                _s = "wide_environmental"
            framing = (force_orientation, _s)
        # Structural rotation — sentence-1 lead (5-cycle) + craft-note
        # placement (7-cycle), both coprime with the 8 framing targets.
        opener_lead = _rotate(OPENER_LEADS, i, run_offset, "opener")
        craft_placement = _rotate(CRAFT_PLACEMENTS, i, run_offset, "craft")
        # T4-only: tight crops physically can't show the required anatomy —
        # remap them to body-showing shots BEFORE the reveal pin, then rotate
        # an explicit REVEAL STYLE + grooming alongside the framing so the set
        # spans many tasteful reveals (not one centred splay). Two
        # distance-bound styles pin a compatible shot_type (keep the orientation).
        # Creative axes (2026-06-12) — concept + composition every tier;
        # the sensual-styling tease is T1-T3 only (T4 has its own explicit
        # REVEAL_STYLES). Prime cycle lengths (11/13/9) keep them out of
        # lockstep with each other and the framing/structural rotations.
        concept = _rotate(EDITORIAL_CONCEPTS, i, run_offset, "concept")
        composition = _rotate(COMPOSITION_PRINCIPLES, i, run_offset, "composition")
        # Camera HEIGHT rotation (the missing axis) — 3-cycle, coprime with the
        # 8/5/7/11/13 rotations so the heroic low angle lands on different
        # framings/concepts across runs rather than locking to one.
        camera_angle = _rotate(CAMERA_ANGLES, i, run_offset, "camera")
        # New decoupled variety axes (2026-06-20): garment-TYPE (tier governs how
        # much shows; OFF at T4 = nude), pose/gesture (all tiers), time×weather
        # atmosphere (all tiers). Each has its own shuffle+stride in _rotate so it
        # does not lockstep with the scene/concept/look axes.
        # GARMENT_TYPES is a contemporary-Western vocabulary; for niches whose
        # wardrobe is era-, genre- or culture-DEFINING (historical / fantasy /
        # heritage-cultural), lock_wardrobe skips it so the sub_look's own
        # period/heritage garment governs (a tuxedo/swimsuit must not land in a
        # haveli or a medieval hall).
        garment_type = ("" if (tier == "T4_explicit" or lock_wardrobe)
                        else _rotate(GARMENT_TYPES, i, run_offset, "garment"))
        pose = _rotate(POSE_GESTURES, i, run_offset, "pose")
        atmosphere = _rotate(ATMOSPHERE, i, run_offset, "atmosphere")
        # The sensual-reveal axis is an UNDRESS/partial-bare axis — it belongs
        # to T2 (implied) and T3 (art-nude) only. At T1 ("fully clothed") it
        # pushed Chroma to render nude (2026-06-12 batch: art_deco T1 drifted
        # 5/6, all caught by the gate but wasted renders); T4 has its own
        # explicit REVEAL_STYLES. T1 gets its heat from the enriched tier
        # directive + concept + composition instead.
        sensual_reveal = ("" if tier in ("T4_explicit", "T1_suggestive")
                          else _rotate(SENSUAL_REVEALS, i, run_offset, "sensual"))
        reveal_target = grooming = None
        if tier == "T4_explicit":
            # The off-native cinematic/social ratios (widescreen 16:9 / story 9:16)
            # are a T1-T3 aesthetic lane. At T4 the explicit-reveal rotation + shot
            # pins must keep the anatomy prominent — a wide_environmental 16:9 hides
            # it and a pinned close-up on a 16:9 frame is dead space — so remap to the
            # native cousin (widescreen->landscape, story->portrait) BEFORE the reveal
            # pin runs, so the reveal styles compose on a native, anatomy-forward frame.
            _o = {"widescreen": "landscape", "story": "portrait"}.get(framing[0], framing[0])
            framing = (_o, _T4_SHOT_REMAP.get(framing[1], framing[1]))
            reveal_target = _rotate(REVEAL_STYLES, i, run_offset, "reveal")
            grooming = _rotate(GROOMING_OPTIONS, i, run_offset, "grooming")
            pin = REVEAL_SHOT_PIN.get(reveal_target[0])
            if pin:
                framing = (framing[0], pin)
        # Per-scene look HOISTED: the LLM call and the opener-similarity check
        # must agree on which tokens are the ASSIGNED look (excluded from bans
        # — the 2026-06-11 night-batch collision fix). A bound persona LOCKS the
        # look (same woman every image); otherwise it rotates from the look pools.
        scene_look = locked_look or _creative_look(i, run_offset)
        scene_look_tokens = frozenset(
            re.findall(r"[a-z']+", scene_look.lower()))
        best: tuple[dict, float, list[str]] | None = None  # (candidate, score, issues)
        last_err = None
        feedback = ""           # rejection reason fed into the NEXT attempt
        attempt = 0
        infra_retries = 0       # LM Studio compute-error budget (separate from
                                # the creative attempt budget — an engine error
                                # is not the prompt's fault)
        while attempt < max_attempts:
            # Blind-retry fix: escalate temperature on later attempts to escape
            # deterministic failure basins. The final attempt of an otherwise-
            # failed scene is a HARD SALVAGE — escalate temperature further on the
            # loaded PRIMARY rather than switching to the Ollama Cydonia tag. The
            # Cydonia model is frequently not pulled (Ollama empty) → the switch
            # was a guaranteed 404 that simply dropped the scene (2026-06-26: this
            # recurred ~1/3 runs, losing a prompt or cover). A high-temp primary
            # retry has a real chance of clearing the gate with zero backend
            # dependency. (Cydonia remains the registry fallback_llm for explicit
            # --model-tag use / painterly-light niches; it's just no longer the
            # automatic per-scene salvage path.)
            cur_temp = temperature + (0.1 if attempt >= 2 else 0.0)
            cur_tag = model_tag
            if attempt == max_attempts - 1 and best is None:
                cur_temp = temperature + 0.3
                print(f"  (scene {i + 1} final salvage — primary @ temp "
                      f"{cur_temp:.2f})", file=sys.stderr, flush=True)
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
                    look_target=scene_look,
                    look_locked=bool(locked_look),
                    reveal_target=reveal_target,
                    grooming=grooming or "",
                    opener_lead=opener_lead,
                    craft_placement=craft_placement,
                    concept=concept,
                    sensual_reveal=sensual_reveal,
                    composition=composition,
                    camera_angle=camera_angle,
                    garment_type=garment_type,
                    pose=pose,
                    atmosphere=atmosphere,
                    lock_wardrobe=lock_wardrobe,
                )
            except Exception as exc:  # noqa: BLE001 — Pydantic/safety reject → retry
                msg = str(exc)
                # LM Studio engine instability ("Compute error" storms burned
                # 24 attempts overnight): retry on a SEPARATE budget — an
                # infrastructure error is not the prompt's fault — and force
                # an engine reload on the 2nd consecutive hit.
                if "compute error" in msg.lower() and infra_retries < 4:
                    infra_retries += 1
                    print(f"  (scene {i + 1}: LM Studio compute error — infra "
                          f"retry {infra_retries}/4, attempt budget preserved)",
                          file=sys.stderr, flush=True)
                    if infra_retries >= 2:
                        _recover_lm_studio(cur_tag)
                    continue
                last_err = exc
                feedback = msg[:400]
                print(f"  (scene {i + 1} attempt {attempt + 1} rejected: {exc})",
                      file=sys.stderr, flush=True)
                attempt += 1
                continue

            # Mechanical anti-repetition — the avoid/banned lists are ENFORCED,
            # not advisory: a too-similar candidate re-rolls like a hard reject.
            sim_reason = _too_similar(cand["prompt"], accepted_ngrams,
                                      ref_sigs, banned_openers,
                                      look_tokens=scene_look_tokens)
            if sim_reason:
                feedback = (f"too similar to prior work — {sim_reason}. Write a "
                            f"VISIBLY different opening, setting and phrasing.")
                print(f"  (scene {i + 1} attempt {attempt + 1} similarity-reject: "
                      f"{sim_reason})", file=sys.stderr, flush=True)
                attempt += 1
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
            attempt += 1

        if best is None:
            print(f"  !! scene {i + 1} failed after {max_attempts} attempts: "
                  f"{last_err}", file=sys.stderr, flush=True)
            continue

        cand, score, issues = best
        if score < audit_threshold:
            print(f"  (scene {i + 1} shipping best audit={score:.1f} after "
                  f"{max_attempts} attempts; issues={issues[:3]})",
                  file=sys.stderr, flush=True)
        if force_orientation:
            # HARD-pin: the assigned framing was already forced, but the LLM may
            # still EMIT a different orientation (it "upgrades" a forced portrait to
            # story for a standing figure). --force-orientation overrides the emitted
            # value too — the prose was composed for the forced framing regardless.
            cand["orientation"] = force_orientation
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

    # Resolve --model-tag (registry key OR backend model id); fail loudly on an
    # unknown value instead of silently degrading to the Ollama fallback.
    from src.memory.llm_registry import LLMRegistryLoader, LLMNotFound
    try:
        args.model_tag = LLMRegistryLoader().resolve_model_tag(args.model_tag)
    except LLMNotFound as exc:
        ap.error(str(exc))
    except Exception:
        pass

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
