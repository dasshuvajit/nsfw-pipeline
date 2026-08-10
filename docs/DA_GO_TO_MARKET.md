# DeviantArt go-to-market — pricing, shop, categorization, series, upload

> Operator playbook for selling this pipeline's output on DeviantArt. Pairs with
> [DA_POSTING_PLAYBOOK.md](DA_POSTING_PLAYBOOK.md) (policy/ToS/economics) and
> [COMPETITOR_INTEL.md](COMPETITOR_INTEL.md) (13 real DA AI sellers — all prices
> below are anchored to their observed numbers). Built 2026-06-04; **v2
> 2026-06-10** from the market-R&D audit: 5-family gallery architecture
> (replaces one-gallery-per-niche), ratchet pricing, 2-tier subscriptions,
> Mon/Wed/Fri cadence, the hyperrealism-clause risk posture, and the DA→Fanvue
> funnel spec — all benchmarked against DA's own top-seller data (82% of top-100
> sub tiers ≤$10/mo; 68% post tier content weekly; 55% post free ~11×/month).
>
> **Honest framing first:** DA is a SFW top-of-funnel + a low-priced impulse
> marketplace, NOT a high-margin earner for this genre (median seller ~low-
> hundreds/yr). Big cheap catalogs (10K+ items at $1–3) are how the volume sellers
> play it; real recurring revenue is **off-site on Fanvue** for explicit. Treat the
> DA price card as impulse-tier, and build the **funnel** (free SFW → cheap singles
> → family galleries → subscriptions → Fanvue) rather than expecting any one item
> to earn much.

## 0. One-time setup (do this before go-live)
> **Corrected 2026-06-15 — see [DA_OPERATOR_GUIDE.md](DA_OPERATOR_GUIDE.md) for the
> full account/identity/payout layer.** Two fixes to the steps below: (a) **Core is
> NOT required to sell** — a free account sells at a 20% fee; Core only *lowers* it
> (buy it later, once sales clear ~$60/mo). (b) **PayPal/Stripe payout does NOT work
> from Bangladesh** — treat DA as the (largely unbanked) SFW discovery funnel and make
> **Fanvue** the real money rail (Bank Transfer / Payoneer / crypto-USDT).
1. **(Later, not required) Paid Core membership** — Core Pro (yearly, on a 50% promo)
   drops the sub/gallery fee 20% → ~5%; break-even ≈ $60–80/mo. Free account sells now.
2. Payout: attempt **Stripe Express** from DA's Earnings page once (may reject BD);
   the dependable rail is **Fanvue** (see DA_OPERATOR_GUIDE §6). 18+ self-report on DA;
   real KYC happens at the payout processor.
3. Set your handle in `config/pipeline.yaml::watermark.text` ("@YourHandle").
4. Build the **5-family gallery tree** (§3) + write a 1–2 sentence description per
   folder (helps DA search).
5. (Funnel) **Fanvue** account + AI-disclosure in bio for the explicit tier.

## 0b. THE #1 POLICY RISK — the hyperrealism clause (read before posting any T3/T4)
DA's sexual-content policy bars adult deviations that read as **"hyperrealistic"**
or insinuate **a real person** — reviewed case-by-case. A photoreal Chroma pipeline
selling T3/T4 sits directly on that line. Posture (enforcement is visibly lax —
archi444 runs photoreal explicit Rooms at 21.4K watchers — but the clause is the
platform's kill switch and the account is the funnel asset):
- **NEVER** use "photorealistic", "hyperrealistic", "realistic", "real woman" in
  the title/tags/description of any T3/T4 post. Frame as **"fine-art figure
  study"**, "renaissance study", "digital painting", "boudoir art". (The posting
  templates + checklist already enforce this wording rule.)
- Route the most **camera-real explicit** work **Fanvue-primary**; keep DA's T4
  volume modest and **fantasy/period-framed** (Myth & Crown and Nocturne are the
  safest T4 carriers; Atelier B&W next — B&W reads as "art" to moderators).
- **Go-live canary:** for the first 2 weeks run one public fine-art T3 + one gated
  T4 set and watch for moderation action before scaling volume.

## 1. Price card (per product)
Anchored to competitor modal pricing ($1–$12 range; $5–7 modal; subs $5–10/mo) and
DA's own data (82% of top-100 sub tiers ≤$10). Keep DA prices **impulse-low** —
volume + the subscription/Fanvue funnel is the business, not per-image margin.

| Product | What | Tier | Price | Notes |
|---|---|---|---|---|
| **SFW teaser / cover** | 2 public images per package | T1/T2 (clothed) | **Free** | Discovery + Groups reach. Always from the package `public/` set. Watermarked. |
| **Micro-single (Exclusive)** | 1 individual image | T1–T3 | **$2–3** | The volume lane (haselnusskrokant: 10K+ at $1–3). **Resale ON** (default) — the 10% royalty + "Owner" badge builds a collector flywheel. Clean file on purchase. |
| **Hero 4K single (Exclusive)** | 1 hand-picked 4K-finished image | T3/T4 | **$6–8** | Only the very best (`upscale_folder.py` output). Everything else 4K stays sub/Fanvue-only. |
| **Family Premium Gallery** | the growing per-family set | T3 (+T4 where the family carries it) | **$5 launch → ratchet** | ONE gallery per family (§3), NOT per niche. Launch at $5 (~2 packages ≈ 12 imgs); **+$1–2 with each added 6-image set, cap ~$15**. Price rises hit only NEW buyers — early buyers are locked in (verified mechanic) → built-in collector reward, and the gallery gets more valuable as it deepens. |
| **Subscription tier 1 — "The Muse"** | all new T3 sets on release + hi-res + alt takes | T3 | **$5/mo** | The volume tier. Alt takes = the good-but-not-keeper curation culls (free inventory). |
| **Subscription tier 2 — "The Private Vault"** | everything in Muse + ALL T4 + watermark-free 4K + 1-week early access | T4 | **$10/mo** | Watermark-free 4K **is** the paid product (artbyinnovation pattern). |
| **Per-persona Room (Subscription)** | monthly access to a persona's sets | mixed | **$5/mo** | **DEFERRED** until a persona demonstrably recurs and pulls favorites (gate: ≥10 paying subs or 1K watchers). archi444's model — but lunasilverlake's 5 dead tiers show what happens when tiers multiply before audience. |
| **Bulk / DLC pack** | 50–100 images | T2/T3 | **$150–200** | Deferred until a 500+ item back-catalog exists; the micro-Exclusives lane builds it organically. |

> **2026-08:** SFW covers are now OPT-IN via `--covers N` (default 0) — T3/T4
> packages are gated-only unless covers are requested at run time.

Rules: never price a single above ~$8 on DA. 2 subscription tiers at launch, not
more (only 14% of DA's top sellers run 4+ tiers). Push depth + recurring, not ticket.

## 2. Shop architecture (the monetization stack)
```
PUBLIC (free, discovery)        → SFW covers + T1/T2 + fine-art-framed T3, watermarked → Groups
USD Exclusives (impulse lane)   → micro-singles $2-3 (resale ON) + hero 4K singles $6-8
Family Premium Galleries (×5)   → one RATCHET-priced gallery per family ($5 → ~$15 cap)
Subscriptions (×2)              → $5 Muse (T3) / $10 Private Vault (T4 + 4K + early)
Fanvue (off-site revenue)       → $9.99/mo; full T4 + the 4K masters DA never sees
```
The SAME inventory is monetized three times (marketplace arbitrage): impulse buyers
grab singles; collectors buy the family gallery; fans subscribe. Blurred Premium-
Gallery deviations can be submitted to Groups — **free gated advertising** (the
image renders blurred to non-buyers with the price on it).

## 3. Gallery folders — the 5-family architecture (v2)
ONE top-level public folder per aesthetic family + ONE Premium Gallery each.
(The old plan's 20 per-niche galleries fragmented value below the $5 price floor
and was unmaintainable at 1 package/week. Niche `da_folder`s become subfolders.)

| Family folder | Niches mapped | Tier span | Role |
|---|---|---|---|
| **THE ATELIER — Fine-Art Nude** | fine_art_figure_study, monochrome_fine_art | T2–T4 | The premium fine-art lane. B&W positions as art and best dodges the hyperrealism clause. |
| **GILDED GLAMOUR — Vintage & Screen** | old_hollywood_glamour, pinup_1950s, art_deco_boudoir, burlesque_cabaret, film_noir_boudoir | T1–T3 | The SFW-leaning discovery engine. |
| **GOLDEN HOUR — Modern & Natural** | modern_boudoir, bohemian_naturallight, poolside_goldenhour, cottagecore_pastoral | T1–T3 | Broadest-appeal teaser feed. |
| **MYTH & CROWN — History & Legend** | renaissance_baroque, medieval_lady, mythology_goddess, angelic_divine, arabian_nights | T1–**T4** | Carries 4 of 7 T4 niches behind fantasy/period framing — the field's proven moat. |
| **NOCTURNE — Dark Fantasy** | goth_romantic, dark_fantasy_vampire, fantasy_glamour, cyberpunk_pinup | T1–**T4** | The dark-fantasy T4 carrier. |

Plus: **Featured** (rotating best SFW teasers), **Character Rooms** (per persona —
added later, see §1), **Archive** (hidden; culls/tests).
`config/niche_library.yaml` carries a `family:` field per niche mirroring this map.

**Tier placement rules (policy-exact):**
- **PUBLIC free** (Mature-tagged where suggestive; AI-labeled always): all T1/T2,
  the 2 SFW covers per package, and **selected T3 art-nudes ONLY when fine-art-
  framed** (B&W / painterly grade / classical pose — "tasteful nudity" is publicly
  legal with a Mature tag and is the genre's discovery hook).
- **GATED only** (Premium Galleries + Subscriptions, 18+ opt-in): **ALL T4 without
  exception** (public explicit is a ToS violation) + glamour-styled T3 (arousal-
  framed nudity is safer gated).
- **NEVER public anywhere:** covers/thumbnails/tier-covers with any mature content
  — source covers exclusively from the package `public/` set (pipeline-enforced).
- **4K masters: never public.** Private Vault + Fanvue only — watermark-free 4K IS
  the paid product.

## 4. Titling + numbering for collectability ("Plates")
Three layers (the metadata templates emit these):
1. **Family serial** — `Atelier No. 014 — "Marble Light"` (sequential per family;
   gaps in a buyer's collection create completion pressure — the andy-varhall
   pattern). Serial counter lives in `output/art_series/.family_serials`.
2. **Set + Plates** — each package is a named Set; its 6 gated images are
   `Plate I…VI` in descriptions (`"Plate III of Set 14: Marble Light"`) so the set
   reads as one collectible object matching the Premium Gallery drop.
3. **Persona numerals** — `"Clara — Candlelit Reverie III"` once Rooms exist.

Titles stay keyword-rich (era/aesthetic/subject) — the poetic half is the brand,
the keyword half is SEO. **Tags** (12–25, layered): genre · era/theme ·
subject/styling · mood · discovery (aiart/digitalart/portrait). Plus the wording
rule from §0b (never "realistic/photorealistic" on T3/T4).

## 5. Weekly cadence — 1 package/week on a Mon/Wed/Fri rhythm
Calibrated to DA's top-seller baseline (68% post tier content weekly; free content
~11×/month) while staying under the bulk-posting ban vector (**≤4 deviations/day,
never scheduled, human-paced**). ≈10–12 deviations/week total:

- **MON (teaser session):** post the week's 2 SFW covers into the family folder
  (Mature where suggestive + AI label + template title/tags) → submit each to 3–5
  Groups → +1 watchers-only bonus image (free watcher-growth gate).
- **WED (gated session):** add the 6 gated images to the family **Premium
  Gallery** (ratchet the price if this is the family's 3rd+ set) → submit 1–2 of
  them to Groups (they render **blurred** to non-buyers = free gated advertising)
  → post the set to the matching sub tier ($5 gets T3, $10 gets T4+4K) → mirror
  the full set to **Fanvue**.
- **FRI (volume session):** 2–3 micro-Exclusives from backlog + 1 free public
  re-post from an older niche.

**Family rotation:** cycle families week to week so each family's Premium Gallery
gains a new set ~every 5 weeks — every family stays alive, no folder floods.

## 6. DA→Fanvue funnel spec
- **Link placement:** Fanvue link in DA bio + About + the **description of every
  gated/blurred post** ("full series + 4K on Fanvue — link in bio"). Keep links
  OFF the public SFW teasers (those are for Groups reach; off-site links there add
  moderation surface without conversion value).
- **Teaser economics per 8-image package:** DA public gets 2 SFW (25%); Fanvue
  free feed gets 1 cropped/T2 teaser per set (≈1:4 free:paid — standard practice
  and within Fanvue's disclosure rules); Fanvue paid gets all 6 gated + the 4K
  masters DA never sees + persona continuity.
- **Fanvue sub at $9.99/mo** (creators keep 80%, vs DA's 20%-minus-Core fees) with
  a first-month discount. AI disclosure in Fanvue bio AND watermark (pipeline-
  emitted) — prominent disclosure is a Fanvue requirement; hiding AI risks
  termination. (Patreon stays banned for synthetic photoreal nudes — do not use.)
- **Division of labor:** DA's job is watchers + impulse singles; Fanvue's job is
  recurring revenue.

## 7. Sellable-series design (what a series IS)
A series = a **curated collection on ONE coherence axis + ONE theme**:
- **Size:** 8–12 keepers (6 min for "set value"). Maps to one family-gallery drop.
- **Aesthetic lock (mandatory):** one palette + lighting + photographer-ref across
  the series — the pipeline's `aesthetic_lock` does this (27 combos per niche).
- **Variety WITHIN the lock:** orientations + shot types + per-image subject looks
  (hair/figure/face/complexion/age rotate per image since 2026-06) — variety sells
  a set; coherence makes it a collection.
- **Persona (optional 3rd axis):** bind a recurring named character
  (`--persona-name`) → feeds a future Room. Identity is prose-level until
  face-lock ships.
- **Tier:** one tier per series (never mix — `assert_level_purity` + packaging
  enforce this).

## 8. Upload workflow (manual, human cadence)
**Per-series checklist** (the package emits a filled `POSTING_CHECKLIST.md` +
per-image `posting_templates/*.txt`):
1. **QA (15 min):** open `contact_sheet.png`; eyeball keepers for sharpness,
   anatomy, single subject, no artifacts. Cull anything off. **T4 sets: verify
   every image actually carries the tier** (the gate now enforces this at the
   prompt level, but eyes beat regexes).
2. **4K-finish the heroes:** copy the best 4–6 keepers into a folder, run
   `python scripts/upscale_folder.py <folder>` → true-4K clean files.
3. **MON/WED/FRI sessions** per §5 — titles/tags/prices from the posting templates.
4. Keep the whole `output/art_series/<ts>/` as the local master (DA can delete
   content at will). **Never delete manifest.json files** — they are the
   pipeline's cross-run anti-repetition memory.

## 9. What the pipeline emits to make upload trivial
Each packaged run (`output/art_series/<ts>/`) produces:
- `contact_sheet.png` — keeper montage for fast QA.
- `package/<niche>/POSTING_CHECKLIST.md` — per-series guide with the price card.
- `package/<niche>/posting_templates/<image>.txt` — copy-paste-ready per image:
  `TITLE / FOLDER / DESCRIPTION / TAGS / GROUPS / MATURE / AI-LABEL / PRICE`.
- `metadata.json` — niche `da_folder`, family, persona, tags, suggested Groups,
  posting strategy, watermark status.

## 10. Go-live phasing (recommended)
1. **Week 1 — setup + canary.** Core membership, 5 family folders, payouts. Post
   ONE public fine-art T3 (Atelier) + ONE gated T4 set (Myth & Crown) per §0b and
   watch 2 weeks for moderation signals.
2. **Week 2–3 — families live.** Launch all 5 Premium Galleries at $5; start the
   Mon/Wed/Fri rhythm; Groups on every public post.
3. **Week 4+ — subs + Fanvue.** Open the $5/$10 tiers once there are ≥3 sets of
   inventory; open Fanvue with the backlog; begin the micro-Exclusives lane.
4. **Later (audience-gated):** persona Rooms at ≥10 subs/1K watchers; auctions on
   hero 4Ks; bulk packs at 500+ catalog; dual-account split only if the AI/Mature
   double-label measurably strangles public reach.

> The single biggest lever on SALES is consistency + Groups reach + a coherent,
> varied, sharp series — all of which the pipeline produces. Price low, post
> steadily, ratchet the galleries, and let Fanvue carry the explicit recurring.
