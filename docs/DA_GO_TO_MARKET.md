# DeviantArt go-to-market — pricing, shop, categorization, series, upload

> Operator playbook for selling this pipeline's output on DeviantArt. Pairs with
> [DA_POSTING_PLAYBOOK.md](DA_POSTING_PLAYBOOK.md) (policy/ToS/economics) and
> [COMPETITOR_INTEL.md](COMPETITOR_INTEL.md) (13 real DA AI sellers — all prices
> below are anchored to their observed numbers). Built 2026-06-04 from the audit.
>
> **Honest framing first:** DA is a SFW top-of-funnel + a low-priced impulse
> marketplace, NOT a high-margin earner for this genre (median seller ~low-
> hundreds/yr). Big cheap catalogs (10K+ items at $1–3) are how the volume sellers
> play it; real recurring revenue is **off-site on Fanvue** for explicit. Treat the
> DA price card as impulse-tier, and build the **funnel** (free SFW → cheap singles
> → themed sets → per-character subscriptions → Fanvue) rather than expecting any
> one item to earn much.

## 0. One-time setup (do this before go-live)
1. **Paid Core membership** (required to sell USD). Mid-tier ≈ $9.95/mo drops the
   platform fee from 20% → ~10%; top tier → as low as 2.5%. Worth it once you sell.
2. Payout = **PayPal/Stripe** in DA settings (NOT Points — purchased Points don't
   cash out). 18+ / ID verification for mature selling.
3. Set your handle in `config/pipeline.yaml::watermark.text` ("@YourHandle").
4. Build the **gallery folder tree** (§3) + write a 1–2 sentence description per
   folder (helps DA search).
5. (Funnel) **Fanvue** account + AI-disclosure in bio for the explicit tier.

## 1. Price card (per image type / tier)
Anchored to competitor modal pricing ($1–$12 range; $5–7 modal; subs $5–10/mo).
Keep DA prices **impulse-low** — volume + the subscription/Fanvue funnel is the
business, not per-image margin.

| Product | What | Tier | Price | Notes |
|---|---|---|---|---|
| **SFW teaser / cover** | 1–3 public images per series | T1/T2 (clothed) | **Free** | Discovery + Groups reach. Always from the package `public/` set. Watermarked. |
| **Micro-single (Exclusive)** | 1 individual image | T1–T3 | **$2–3** | Impulse buy; the volume driver (haselnusskrokant: 10K+ at $1–3). Clean file on purchase. |
| **Premium image (Exclusive)** | 1 hero / 4K-finished image | T3/T4 | **$5–8** | Your best, 4K-upscaled keepers (`upscale_folder.py`). |
| **Themed set (Premium Gallery)** | 6–12 coherent images | T2/T3 | **$5–8** | 6–8 imgs → $5; 10–12 → $7–8. The curated-series product (matches a packaged run). |
| **Bulk / DLC pack** | 50–100 images | T2/T3 | **$150–200** ($2–3/img) | Optional volume play once you have a deep catalog. |
| **Per-character Room (Subscription)** | monthly access to a persona's set | mixed | **$5/mo** | Recurring. archi444's per-character "Rooms" (21.4K watchers). |
| **Explicit subscription** | T4 unrated tier | T4 | **$10/mo** | DA gated tier AND/OR Fanvue mirror at ~$10/mo. |

Rules: never price a single above ~$8 on DA (the market won't bear it — it's an
impulse platform). Push depth (many cheap items) + recurring subs, not high ticket.

## 2. Shop architecture (the monetization stack)
```
PUBLIC (free, discovery)         → SFW teasers + covers (T1/T2), watermarked, → Groups
USD Exclusives (impulse)         → micro-singles $2-3 (T1-T3) + premium singles $5-8 (T3/T4, clean)
Premium Galleries (sets)         → one per niche, 6-12 imgs, $5-8 (the packaged series)
Character Rooms (subscriptions)  → per-persona $5/mo (recurring; retrospective + new)
Explicit tier (subscription)     → T4 unrated $10/mo on DA + Fanvue mirror
```
The SAME inventory is monetized twice (marketplace arbitrage, per artbyinnovation):
impulse buyers grab singles; collectors buy the set or subscribe to the Room.

## 3. Gallery folders + titling + tags (categorization)
**Folder tree** (mirrors each niche's `da_folder`):
```
SFW Showcase (public)            → Featured Teasers (rotated) + per-niche T1/T2 (3-5 imgs)
Premium Galleries (locked/sold)  → "Fine-Art Figure Study", "Renaissance Portrait",
                                    "Dark Romantic", "Mythic Goddess", "Fine-Art B&W", … (one per niche)
Character Collections            → "Clara's Studio", "Sable's Reverie", … (per persona, as they recur)
Archive (hidden)                 → culls / tests
```
**Titling** (poetic + series numbering, the competitor pattern):
`<Persona — >​<Poetic title> <N>` e.g. `"Clara — Candlelit Reverie 3"`,
`"Renaissance Study — Oxblood & Gold 2"`. Keyword-rich enough for search; numbered
for collectibility. The pipeline now emits these (§6).
**Tags** (12–25 of 30, layered across 5 axes — the metadata generator pre-fills):
genre (fineartnude/boudoir/glamour) · era/theme (renaissance/oldhollywood/vampire) ·
subject/styling (brunette/lace/silk) · mood (sensual/elegant/moody) ·
discovery (aiart/digitalart/portrait).
**SFW-cover HARD rule:** every cover/thumbnail/tier-cover is SFW — sourced only from
the package `public/` set (the pipeline enforces this; never routes T3/T4 to a cover).

## 4. Sellable-series design (what a series IS)
A series = a **curated collection on ONE coherence axis + ONE theme**, not a random
batch. Parameters:
- **Size:** 8–12 keepers (6 min for "set value", 20 max). Maps to a Premium Gallery
  ($5 for 6–8, $7–8 for 10–12). `art_series --count 8 --seeds 2` → curate to ~8–12.
- **Aesthetic lock (mandatory):** one palette + one lighting + one photographer-ref
  held across the whole series — the pipeline's `aesthetic_lock` already does this.
- **Variety WITHIN the lock:** mix orientations + shot types (Phase-1 framing) so a
  set has a close-up, full-bodies, an environmental, across portrait/square/landscape
  — variety sells a set; coherence makes it a collection.
- **Persona (optional 3rd axis):** bind a recurring named character (`--persona-name`)
  → feeds a per-character Room. Identity is prose-level until face-lock ships.
- **Tier:** one tier per series (never mix). T1/T2 → public + cheap; T3/T4 → gated.

## 5. Upload workflow (manual, human cadence)
**Per-series checklist** (the package emits a filled `POSTING_CHECKLIST.md` +
per-image `*.posting_template.txt`, §6):
1. **QA (15 min):** open `contact_sheet.png`; eyeball 8–12 keepers for sharpness,
   anatomy, single subject, no artifacts. Cull anything off.
2. **4K-finish the heroes:** copy your best 4–6 keepers into a folder, run
   `python scripts/upscale_folder.py <folder>` → true-4K clean files.
3. **Public post:** upload `public/` SFW teasers → set **Mature** + **"Created using
   AI tools"** labels → paste the title/description/tags from the posting template →
   submit to 3–5 relevant **Groups** (Groups are the real reach, not the feed).
4. **Gated:** create/extend the niche **Premium Gallery**, add the `gated/` set at the
   §1 price; route T4 to the explicit Subscription / Fanvue.
5. **Cadence:** a few posts/day MAX across 2–3 sessions/week — never bulk/scheduled
   (bulk posting is the #1 ban vector). Keep the whole `output/art_series/<ts>/` as
   your local master (DA can delete content at will).

## 6. What the pipeline emits to make upload trivial
Each packaged run (`output/art_series/<ts>/`) now produces:
- `contact_sheet.png` — montage of keepers for fast QA (already emitted by `_curate`).
- `package/<niche>/POSTING_CHECKLIST.md` — the per-series guide with the price card.
- `package/<niche>/posting_templates/<image>.txt` — **copy-paste-ready** per image:
  `TITLE / FOLDER / DESCRIPTION / TAGS / MATURE / AI-LABEL / PRICE` (price auto-set
  by tier from §1). Public images get the watermark note; gated get the clean-file note.
- `metadata.json` — carries niche `da_folder`, persona, series title/number, tags,
  `da_groups` (suggested Groups per niche), `posting_strategy`, `watermark_status`.

## 7. Go-live phasing (recommended)
1. **Week 1 — setup + 1 test series.** Core membership, folders, payouts. Generate +
   post ONE small SFW-heavy series (e.g. `poolside_goldenhour` T2). Watch 24–72h.
2. **Week 2–3 — 3–4 niches live.** Add Premium Galleries per niche; start a recurring
   persona (Clara/Sable) → first Character Room. Establish 2–3 posts/wk cadence + Groups.
3. **Week 4+ — scale + Fanvue.** Add the explicit tier on DA + a Fanvue mirror; deepen
   the cheap-singles catalog; let the `--auto` niche rotation feed steady fresh content.

> The single biggest lever on SALES is consistency + Groups reach + a coherent,
> varied, sharp series — all of which the pipeline now produces. Price low, post
> steadily, build the subscription/Fanvue funnel.
