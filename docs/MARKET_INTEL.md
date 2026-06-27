# MARKET & COMPLIANCE INTEL — DeviantArt funnel + Fanvue revenue
*2026-06-27. Web-sourced (DA ToS/help articles, Fanvue policy, creator-economy reporting) and
adversarially verified — every claim was re-checked by a second pass that deleted fabricated
stats/quotes and downgraded unsourced inference. Confidence is marked per claim. Treat all
earnings/conversion numbers as secondary, single-source estimates, not audited fact. See
docs/COMPETITOR_INTEL.md for per-creator profiles; this file is the policy/monetization/strategy layer.*

## TL;DR — the architecture is already right
The research **validates the pipeline's design against primary sources**: DA = SFW funnel, explicit
gated, Fanvue = revenue, single fictional adult subject, SFW-cover rule, tier-split (public SFW
watermarked / gated clean), watermarking. Nothing structural needs to change. The value is in
**compliance hardening** + **go-to-market execution** + confirming **recurring-persona identity is
the #1 differentiation lever**.

## Platform map (where each tier lives)
- **DeviantArt = discovery/funnel only (NOT primary sales).** [HIGH] Public feed must be SFW; mature
  is gated behind a 3-tier browse filter (Safe/Standard/Mature) — **logged-out + Standard users never
  see Mature**, so any nude teaser is invisible for top-of-funnel acquisition. Lead with **SFW covers**.
  Even SFW can be auto-hidden from logged-out viewers by brand-safety flags + the user "Suppress AI"
  filter (removes labeled AND auto-detected AI) — so **do not rely on DA Browse/Search for cold reach**;
  use DA as a profile/portfolio + link-in-bio to Fanvue, and grow **Watchers** (new posts auto-broadcast
  to them free).
- **Explicit illustrative content on DA is allowed ONLY inside paid 18+ Subscriptions / Premium
  Galleries** [HIGH] — and **pornographic/obscene** material is banned outright "with no exception."
  → Keep DA to **T1/T2 + SFW covers**; route **T3/T4 to Fanvue** (don't monetize explicit ON DA).
- **Fanvue = revenue, and effectively the ONLY viable one** [HIGH]: it explicitly allows fully-synthetic
  AI personas. **OnlyFans/Fansly ban from-scratch AI personas; Fansly bans photoreal AI; Patreon/Ko-fi
  restrict hyperreal/explicit AI.** Concentration risk: revenue depends on Fanvue keeping card rails.
- **Active acquisition: Reddit + X/Twitter** [MED, marketing-folklore confidence] are cited as higher-
  converting than IG/TikTok (which suppress adult-link accounts — always route via an age-gated
  link-in-bio buffer, never paste fanvue.com).

## Compliance MUSTS (do-not-violate)
1. **AI labeling on DA is MANDATORY for anything offered for SALE** [HIGH] (ToS §31); the AI/non-AI
   declaration is a required submit field and DA auto-applies it from metadata + runs a detection model.
   → Keep DA a **free** funnel (label optional but always declare honestly); monetize on Fanvue.
2. **SFW-cover rule, tightened** [HIGH]: covers on the actual paywall surfaces (Premium Galleries /
   subscription tiers) must be **fully SFW — not even suggestive-mature**. Standalone deviation/Premium-
   Download previews may be mature-but-**never explicit**, and only with the Mature label.
3. **Single fictional, non-identifiable subject is the legal shield** [HIGH]: the TAKE IT DOWN Act
   (signed 2025-05-19) + DA's real-person model-release clause + 18 U.S.C. 2257 all bite the moment a
   real person's likeness is used. **Never train on / prompt toward / output a likeness of any real
   identifiable person; never composite real photos/faces/bodies.** Staying 100% synthetic likely keeps
   the operator outside 2257 and the model-release regime. (Fanvue requires operator **KYC** regardless.)
4. **Age is judged by PIXELS, not the prompt** [HIGH, Fanvue]: prose "adult anchors" + banned age tokens
   are necessary but **NOT sufficient** — Fanvue audits the rendered face/body. → keep a conservative
   visual adult margin; the persona must read unambiguously adult.
5. **Banned ACTS apply even to fully-synthetic content** [HIGH, Fanvue]: bestiality, choking/
   asphyxiation, age-play, rape/simulated non-consent, necrophilia, genital mutilation, incest, and
   protected-character sexualization. Enforcement is subjective human review → permanent ban, real money
   at stake. → these belong in the art_director banned-token/theme gate (implemented 2026-06-27, below).
6. **AI disclosure on Fanvue is itself mandatory** [HIGH]: undisclosed AI media is a TOS violation
   regardless of image cleanliness → bake "AI-generated, fictional, 18+" into the Fanvue **bio + every
   caption + the watermark**.

## Monetization (Fanvue) — plan on these
- **Take rate 20%** (creator keeps 80%); the old 85/15 is gone (a "15% first-12-months" promo may show
  at signup — treat as temporary). [HIGH]
- **Payouts: 7–28 day hold, ~10 business days to process; $50 chargeback fee; keep disputes <1.5%.** Plan
  ~2–4 weeks float before cash. [HIGH]
- **Pricing structure** [MED, benchmark]: sub floor $3.99; **price the sub LOW ($4–7) as a tripwire**,
  monetize depth via **PPV image SETS ($8–25) + bundles** (PPV/tips, not subs, are the primary lever).
  Use **Follow-for-Free** + a short (1–3 day) auto-converting free trial; point the DA funnel at the
  free/trial link, not a cold paywall.
- **Conversion is a volume game**: free→paid ~1–5% (unverified OnlyFans-blog heuristic), churn high
  (~30–40%/mo, unverified). → wide top-of-funnel, front-load the best gated set in a new sub's **first
  48h**, steady cadence + DM/re-engagement (the batch pipeline's cheap cadence is the solo-operator edge).
- **Realistic ceiling** [MED]: AI is ~15% of Fanvue GMV; named AI models ~$10–20k/mo (outliers ~$50k).
  Path is consistency + niche ownership, not virality. A verified AI creator can run **up to 15 linked
  Fanvue accounts** (multi-persona scaling under one KYC) — but any active warning freezes expansion.

## Discovery / SEO
- **Max all 30 tags per DA post** [HIGH] — tagged works get ~3× views; tiered set (subject + medium/
  technique + mood/aesthetic + persona/series). The posting_templates already emit tags — ensure they
  fill toward 30.
- **Teasers must be GENUINELY SFW (no Mature flag)** to index + reach logged-out guests. [MED]
- **5-family gallery folders + consistent searchable titles + the "Plate NN/MM" family-serial** scarcity
  framing (honest marketing, never an investment claim). [MED] The pipeline already does Plate titling.

## Strategy — the differentiation moat
- **The #1 commercial risk is visual interchangeability** [MED]: the oversaturated default is the glossy
  hyperreal "same-face beautiful-but-generic young woman." A "pretty AI woman" converts poorly vs
  infinite free equivalents.
- **The moat is recurring-character IDENTITY** [HIGH, repeated across facets]: one memorable face +
  ONE thumbnail-legible signature + cultural/aesthetic rooting, varying wardrobe/scene/mood NOT identity.
  → **This is the strongest argument to prioritize the deferred pixel-level face-lock / recurring-persona
  work for the PAID product** (pin the face; keep the decoupled-axes engine for everything else).
- **Underserved whitespace the library already holds** [MED]: editorial/high-fashion nude, cultural-
  heritage-rooted work (south_asian_editorial / iberian_flamenco / slavic_folk), wellness/atmosphere
  (thermal_bathhouse), surreal. **"Artistic, not clinical" is a real wedge** — fine-art-nude buyers
  reward optical/material craft (named vintage glass, natural-material texture), which is exactly the
  art_director house style. **But for a SINGLE recurring subject, anchor to 1–2 ownable lanes** rather
  than spreading identity across all 28 niches.

## PIPELINE ACTIONS (what this research changes)
- **[DONE 2026-06-27]** Banned-acts/theme gate for Fanvue compliance → added to art_director's banned
  tokens (item 5 above). Tested.
- **[VALIDATED — no change]** SFW-cover rule, tier-split, watermarking, single-subject, DA-funnel/Fanvue-
  revenue split: all policy-aligned. The visual tier-truth QA of public/ + the NudeNet drift gate are the
  right safeguards.
- **[RECOMMENDED — operator]** (a) Prioritize recurring-persona identity-lock for the paid product; anchor
  the persona to 1–2 lanes. (b) Stand up Reddit + X as the active acquisition channels; DA = portfolio +
  link-in-bio. (c) Fanvue: low tripwire sub + PPV sets, AI-disclosure in bio/captions/watermark, complete
  KYC, keep disputes <1.5%. (d) Max 30 tags per DA post. (e) Consider a "visual adult-cue" QA on faces
  (age is judged by pixels) — the existing prose anchors + age tokens are necessary but not sufficient.

## Caveats / gaps to verify firsthand
- All earnings/conversion/churn figures are secondary single-source estimates — **the verify pass
  deleted several fabricated stats and a fabricated Fanvue-enforcement quote**; do not treat any number
  as audited.
- Confirm Fanvue's **current** new-creator promo fee, KYC/age-verification specifics, and the exact
  banned-acts list **on Fanvue's live policy** before relying on them — platform terms change.
- DA's exact handling of photoreal AI nudes (vs illustrated) and current AI-specific gating is not fully
  published — spot-check how much of the SFW funnel actually survives a **logged-out** view.
