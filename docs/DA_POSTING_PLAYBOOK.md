# DeviantArt posting playbook (operator guide)

The pipeline produces a **publish-ready package** (`output/art_series/<ts>/package/<folder>/`)
with a `public/` set, a `gated/` set, `metadata.json`, and a `POSTING_CHECKLIST.md`.
**You upload manually.** This doc is the standing context behind that checklist —
all of it is from the verified 2026 market/policy research (workflow `w9m6qmriy`).

> Why manual upload: automated/bulk posting is the #1 documented ban vector
> (DeviantArt throttles posting velocity; AI-content suspensions happen without
> warning). And never hand an agent your password — it breaches ToS §14, shifts
> all liability to you, and there is no password grant in the API anyway.

## The model (honest economics)
- DeviantArt is a **SFW top-of-funnel**, not a primary earner for this genre.
  Platform-wide ~$23M in 2025 across tens-of-thousands of sellers → the median
  seller earns *low hundreds/year*; the publicized $8–15k earners are
  character-design/adoptables, not nude/boudoir.
- Real recurring revenue lives **off-site on Fanvue** (AI-native, ~80% payout,
  AI-disclosure required). **Patreon bans synthetic AI nudes** (no real model) —
  don't route there. The package's `gated/` set is Fanvue-ready.
- On-DA selling needs a **paid Core membership** (you can't sell on a free
  account); higher Core tiers drop the platform fee (~20% → as low as 2.5%).
  Structure offers as 2–3 Subscription tiers (one ≤ ~$10) + Premium Galleries
  (locked themed sets — a curated series maps perfectly) + Premium Downloads.

## Hard rules (ToS — do not skip)
1. **SFW shopfront.** Every cover / thumbnail / tier-cover MUST be SFW even when
   the unlocked set is explicit. → Covers come only from the package `public/`
   set (the pipeline enforces this; it never routes a T3/T4 image to a cover).
   *(2026-08: SFW covers are now OPT-IN via `--covers`, default 0 — T3/T4
   packages are gated-only; pass `--covers N` when a shopfront cover is needed.)*
2. **AI disclosure.** Apply the **"Created using AI tools"** label on every
   for-sale piece. Do not strip AI metadata to evade the classifier.
3. **Mature label** on every nude/suggestive piece.
4. **No public explicit.** Explicit/T4 goes ONLY to gated Subscriptions /
   Premium Galleries for opted-in 18+ buyers. Public gallery = artistic
   (T1/T2) nudity, mature-labeled.
5. **Avoid the hyperrealism clause.** DA bars depictions reading as a "real
   person / hyperrealistic." Frame as art/render/digital-art; don't use
   "hyperrealistic / realistic / real woman" in titles or tags. The prompt
   engine already biases fine-art/editorial framing — keep it in the metadata.
6. **Synthetic, original adults only.** No real-person likeness / face-swaps /
   deepfakes; subjects clearly fictional 18+. Hard-avoid the named fringe
   categories (incest, non-consent, etc. — termination without notice).
7. **Human cadence.** A few posts/day max — never bulk/scheduled. Keep local
   masters; DA may delete content at its sole discretion.

## Discoverability (where reach actually comes from)
- **Groups > the feed.** Join active pinup / glamour / fine-art-nude / AI-art
  Groups and submit each piece to several — this is the main reach multiplier
  (the DA feed/Suggested is weak). Maintain ≥ weekly cadence + consistent
  branding/thumbnails.
- **Tag every piece with 12–25 of the 30 tags**, layered across 5 axes:
  genre (pinup/boudoir/glamour/fineartnude) · era (1950s/oldhollywood/artdeco) ·
  subject/styling (lingerie/redhead/brunette/elf) · mood (sensual/elegant/moody) ·
  discovery (aiart/digitalart/portrait). The niche library + metadata generator
  pre-fill these per niche.
- **Keyword-rich title + description** (search reads them): put era/aesthetic/
  subject in the title ("Art Deco Boudoir Study — Brunette in Satin").
- **Organize the gallery into tight named folders** by aesthetic (the niche's
  `da_folder`: "1950s Pin-Up", "Old Hollywood", "Fine-Art Figure Study", …) and/
  or subject — the proven top-earner pattern. With a recurring persona, use
  `<Folder> — <Name>` ("Fine-Art Figure Study — Clara").
- **Do NOT plan around Daily Deviation** — AI + adult + premium-locked is
  triple-ineligible. Spend that energy on Groups + Subscriptions + cross-promo
  (link DA from IG/X).

## Per-package workflow
1. Open `package/<folder>/POSTING_CHECKLIST.md` — it has the title/description/
   tags for the public post and (for explicit runs) the gated set.
2. **Public post**: upload `public/` (watermarked SFW teasers), set Mature + the
   AI label, paste title/tags/description, submit to relevant Groups.
3. **Gated set** (explicit runs): route `gated/` to a paid Subscription tier or
   Premium Gallery (Core required). Optionally mirror to Fanvue with the
   AI-disclosure in bio/caption.
4. Keep the whole `output/art_series/<ts>/` as your local master.

## One-time setup (manual, yours to do)
- DeviantArt **Core membership** + Subscription tiers / Premium Galleries +
  payout config (PayPal/Stripe — not Points; purchased Points don't cash out).
- (Optional funnel) **Fanvue** account + AI-disclosure in bio.
- Set your handle in `config/pipeline.yaml::watermark.text` ("@YourDAHandle").
- Before scaling photoreal volume, **post a small test batch** and watch
  enforcement (the hyperrealism clause is the main ambiguity for a photoreal
  pipeline).
