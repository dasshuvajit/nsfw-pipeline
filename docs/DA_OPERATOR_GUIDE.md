# DeviantArt Account Setup & Selling — Operator Guide

> From-ZERO setup + selling playbook for a Bangladesh-based operator running the
> 5-family AI-art studio (Atelier · Gilded Glamour · Golden Hour · Myth & Crown ·
> Nocturne; personas Clara/Sable/Margot/Imani/Mei). Built **2026-06-15**, on top of
> [DA_GO_TO_MARKET.md](DA_GO_TO_MARKET.md) (pricing/galleries/funnel) and
> [GO_LIVE_RUNBOOK.md](GO_LIVE_RUNBOOK.md) (Phase 0→3 sequence). This guide is the
> ACCOUNT + IDENTITY + MONEY layer those two docs assume but never spelled out — and
> it **corrects two stale assumptions** in them (see the boxes below).
>
> **Legality/jurisdiction is out of scope** (operator instruction). Everything here is
> "which rails technically work and which buttons to click."

---

## BOTTOM LINE (read this first)

Run a **pseudonymous public brand** (handle = `MaisonLumiere`, the same name everywhere,
baked into the watermark) on top of a **real-identity payout layer** (KYC is unavoidable —
the brand is anonymous to buyers, not to the payment processor). The single hardest
constraint is **payout from Bangladesh**: **all three DeviantArt payout rails
(PayPal / Stripe / BitPay) are problematic-to-dead for a BD recipient**, so treat
**DeviantArt as the unmonetized SFW discovery funnel** and make **Fanvue the real money
rail** (Bank Transfer / Payoneer / crypto-USDT, all of which work from BD). Go live on a
**FREE DeviantArt account** — Core is a fee-reducer, not a gate — and only buy **Core (yearly,
on a 50%-off promo)** once you've proven you can actually withdraw and DA sales clear
~$60–80/mo. Everything else (5 families, $2–3 micro / $6–8 hero / $5–$10 subs, ratchet
Premium Galleries, Mon/Wed/Fri cadence) executes exactly as the existing docs say.

> ### ⚠️ Two corrections to the existing docs
> Both **DA_GO_TO_MARKET §0** and **GO_LIVE_RUNBOOK Phase 0** say *"buy Core (required to
> sell)"* and *"set payout = PayPal/Stripe."* As of 2026-06 both are wrong for this operator:
> 1. **Core is NOT required to sell.** A free account sells at a 20% fee; Core just lowers it.
> 2. **PayPal/Stripe payout does NOT work from Bangladesh.** Plan around it — see §6.

---

## 1. Handle / brand name

**Decision: register `MaisonLumiere` (watermark `@MaisonLumiere`).** "House of light" —
reads as a fine-art photography atelier, names the pipeline's light-on-form craft, and is
**persona-agnostic** so it outlives any one of the 5 personas. It is DA-username-legal
(13 chars) and carries **zero fetish/porn signal** — the deliberate opposite of the
`primalcarnalvore` anti-lesson in COMPETITOR_INTEL.

### Top 3 (in fallback order)
| Rank | Handle | Why | Tagline |
|---|---|---|---|
| **1** | **MaisonLumiere** | atelier + light; ties to all 5 families + the craft | "Fine-art studies in light and shadow." |
| **2** | **AtelierNoir** | the dark-room studio; pairs Atelier + Nocturne + the B&W lane | "The dark-room studio. Light, shadow, form." |
| **3** | **MuseAndMarble** | classical figure-study register; matches the Atelier "Marble Light" titling | "Classical figure studies, reimagined." |

If the top 3 are all taken, next fallback is **NoirBoudoir** ("Vintage boudoir art. Studio
light, slow seduction.").

### Full shortlist (all pass DA's username rules)
- **Group A — Atelier/Gallery:** MaisonLumiere · AtelierNoir · VelvetAtelier · GalerieLumine
- **Group B — Light/Luminous:** ChiaroLume · AuroraVeil · GildedHour · LucentMuse
- **Group C — Muse/Myth:** MuseAndMarble · SableAndMuse · MythAndMuse · OdalisqueStudio
- **Group D — Vintage Glamour:** NoirBoudoir · GildedReverie · SatinAndSepia

**Avoided on purpose:** any porn/fetish word (carnal/lust/naughty/XXX/babes/girls/hot/sexy/
nsfw), "AI" in the handle (keep AI in the *bio*, not the brand), real-name leakage, and using
a single persona name as the master brand.

### How to check availability (one sitting, before you commit the watermark)
**DA username hard rules:** 3–20 chars; letters/numbers/hyphens only (ASCII — no accents, so
spell it **Lumiere**, never *Lumière*); no spaces; can't start/end with a hyphen; not
numeric-only. Case-insensitive (shown CamelCase-friendly).

1. **DA check (authoritative):** open `deviantart.com/MaisonLumiere` — a 404 means likely free;
   an existing profile means taken. Confirm by entering it in the **Join** signup form (DA
   validates uniqueness live; there is no public pre-check API).
2. **All-platforms-at-once:** run the exact handle through a cross-platform checker —
   **Claimbrand** (`claimbrand.com/networks/deviantart-username-availability`) or **Qezir**
   (`qezir.com/check/deviantart`) — to find it free on **DA + Fanvue + Instagram + .com/.art
   in one pass.** Pick the candidate free across all four; cross-platform consistency beats
   getting your #1 on DA and a mangled handle elsewhere.
3. **Reserve all four surfaces the same day:** DA (signup), **Fanvue** (sign up → it assigns a
   random username → rename to your brand in Settings), **Instagram** `@MaisonLumiere`, and the
   **domain** (`maisonlumiere.com` and `.art` if free — even a parked page protects the brand
   and gives a clean DA-bio link).

> **Near-permanent choice.** DA gives only one free username change (then it needs Core/payment)
> and old links break. Check everywhere *before* committing — don't plan to rename later.

### The 5 branding rules
1. **One handle everywhere, identically cased.** Same `@MaisonLumiere` on DA, Fanvue, IG, domain —
   and wire it into `config/pipeline.yaml::watermark.text` (replacing `@YourDAHandle`), then run
   the **no-GPU re-watermark pass** (GO_LIVE_RUNBOOK Phase 0.3) before posting anything.
2. **One tagline, in every bio.** Keep the words *studio / fine-art / figure study / boudoir* in
   it (doubles as the §0b hyperrealism-clause framing); **never** *realistic/photorealistic/AI girls*.
3. **A reusable avatar + banner SYSTEM.** Avatar = a wordmark or one signature **B&W / golden-hour
   SFW crop** (B&W reads as "art" to moderators). Banner = a 3–5 image strip spanning all 5
   families so the profile reads "studio with a roster," not "one model." Same avatar on DA +
   Fanvue + IG. **Source covers ONLY from package `public/`** (SFW, watermarked) — never a mature crop.
4. **AI disclosure IN the bio, on every platform.** DA: the per-post "Created using AI tools"
   label (template-enforced) + an "AI-assisted fine-art" line in the profile. Fanvue: a plain
   "AI-generated art" line (required; hiding it risks termination). This is a positioning asset,
   not a liability. **Patreon stays off the table** (bans synthetic AI nudes).
5. **Bio-link discipline = funnel-correct.** Fanvue link in DA **bio + About + every
   gated/blurred post description only** — keep it OFF public SFW teasers (those are for Groups
   reach; off-site links there only add moderation surface). IG bio links to DA (discovery) and/or
   Fanvue (revenue); reserve a Linktree/own-domain landing under the brand.

---

## 2. Identity & privacy

**Verdict: NEVER expose your real name or personal handle anywhere public.** Pseudonymity is
correct and universal — every one of the 13 real sellers in COMPETITOR_INTEL operates under a
brand handle, not a legal name. DA does **not** force your legal name onto your profile (the
real-name fields are optional — leave them blank).

**But there is a hard floor at the money layer.** The brand is anonymous to the public and to
buyers — **not** to the payment processor, the platform's finance/compliance, or the bank.
Every payout rail (DA's Stripe/BitPay, Fanvue's Payoneer/bank/crypto) runs **KYC on the real
operator**: government ID + selfie, legal name, DOB, address. Fanvue verifies *you*, the real
human, even though your face never appears in the AI content. Accept this — don't pretend full
anonymity exists, and don't burn time chasing an "anonymous payout."

### The model: pseudonymous PUBLIC layer + real-identity PAYOUT layer, hard-separated
| Layer | What it is | Real identity? |
|---|---|---|
| **Public** | Brand handle, avatar, bio, watermark, posts | **No** — pseudonymous, fictional personas |
| **Payout/KYC** | Stripe Express / Fanvue / Payoneer | **Yes** — your real ID, stays with the processor |

### Compartmentalization (do all of these)
- **Dedicated browser profile** (or a separate browser) for the brand — only ever logged into
  the brand Gmail/DA/Fanvue/payout, **never** your personal Google/social in the same profile.
- **Password manager** so brand creds never touch personal ones.
- **Never cross-link, cross-post, or reuse** personal usernames, profile photos, or your
  personal recovery phone between the brand and your real life.
- **If you use a VPN, stay in ONE region.** Flip-flopping IPs/countries during signup or KYC
  triggers review on DA, Stripe, Payoneer, and Fanvue.

> **On DA seller ID:** DA gates mature *viewing/selling* on your **self-reported birth date**
> (set 18+), not a government-ID upload — there is no separate DA seller-ID wall confirmed for
> 2026. The certain ID checkpoint is the **payout processor's KYC** (Stripe Express / Payoneer).
> Plan for that, not a DA ID wall.

---

## 3. Email (this is setup step 1)

**Verdict: a NEW, dedicated Gmail for the brand. Not ProtonMail as primary, and never your
personal Gmail.** For a money-making identity the decisive factor is **deliverability +
reliability** for KYC, 2FA, and payout notices — not message encryption. Gmail is free, never
blocked by DA/Fanvue/Stripe/Payoneer, and avoids the "disposable email" rejections privacy
domains sometimes hit. ProtonMail's "anonymous signup" can still demand a phone on flagged
signups, and its encryption protects you from third parties, **not** from your KYC processor —
so it adds cost, not anonymity. (Optional: a ProtonMail address as a cosmetic public "contact"
inbox — but it's decoration.)

**Create it (do this first):**
1. `gmail.com` → **Create account** → **For my personal use**.
2. Name it for the brand (a brand-y first name is fine; you never publish it).
3. **Recovery:** use the **brand's own number / an alternate**, **NOT your personal recovery
   phone** — keep the compartment clean.
4. **Turn on 2FA** immediately.
5. Use this **single email** for DA + Fanvue + Stripe Express + Payoneer.

---

## 4. Account creation — exact ordered steps

Do these in order. (Selling controls don't appear until late — but you can post free content
the moment the account exists.)

1. **Create the dedicated Gmail** (§3) + 2FA.
2. **Reserve the handle everywhere** (§1) — DA, Fanvue, IG, domain, in one sitting.
3. **Create the DA account:** `deviantart.com/join` → brand Gmail + `MaisonLumiere`.
4. **Set birth date to 18+** in account settings. This is the *only* age gate DA applies to
   mature viewing/selling — there is no government-ID upload step on DA itself.
5. **Wire the handle into the watermark:** edit `config/pipeline.yaml::watermark.text` to
   `@MaisonLumiere`, then run the **no-GPU re-watermark pass** on the packages you'll post
   (GO_LIVE_RUNBOOK Phase 0.3 — Claude can do both).
6. **Fill the profile:** SFW avatar (B&W signature crop or wordmark) + bio (tagline + AI-
   disclosure line + Fanvue link) + About section. Leave the real-name fields **blank**.
7. **Settings → enable "show mature content"** so you can see and manage your own gated work.
8. **Build the 5-family gallery tree** (§7/§8): ATELIER · GILDED GLAMOUR · GOLDEN HOUR ·
   MYTH & CROWN · NOCTURNE, each with a 1–2 sentence description, + **Featured** + a hidden
   **Archive**.
9. **Create the Fanvue account** (§6): sign up → pass **Ondato KYC** (your real passport +
   selfie) → rename from the auto-assigned username to `MaisonLumiere` → add the AI-disclosure
   bio line → set up the payout method (§6).
10. **(Free account = stop here to start posting.)** You can now upload free SFW + Mature-tagged
    content and run the canary. **Selling buttons (Exclusives / Premium Galleries / Subscriptions)
    require Core** — buy it later per §5, only once the §6 payout path is proven.

> **"ID verify" reality check:** the step the existing docs call "18+/ID verification" is two
> separate things — (a) the **self-reported 18+ birthdate** on DA (no upload), and (b) the
> **KYC the payout processor runs** (Stripe Express / Fanvue's Ondato / Payoneer). There is no
> third DA-specific ID wall to clear.

---

## 5. Membership (Core / Pro) — buy LATER, not now

**Decision: go live FREE. Buy Core *Pro*, *yearly*, on a *50%-off promo*, only when you trigger
it.** A free account already sells at a 20% platform fee — Core is a fee-reducer + features
upgrade, not a gate. Paying $100/yr to watch a moderation canary earn nothing is backwards.

### Current tiers (2026-06; re-check the live buy page)
| Tier | ~Yearly (mo-equiv) | Subs / Premium Galleries fee | Exclusives fee |
|---|---|---|---|
| (Free) | $0 | **20%** | **20%** |
| Core+ | $80 (~$6.67/mo) | 12% | 15% |
| **Core Pro** ✅ | **$100 (~$8.33/mo)** | **5%** | **10%** |
| Core Pro+ | $150 (~$12.50/mo) | 2.5% | 5% |
| Core Max | $200 (~$16.67/mo) | 2.5% | 5% (no fee gain over Pro+) |
Commissions: 0% on all Core tiers. The old $4 "Core Basic" selling tier is retired.

### Why Core Pro (not the cheaper Core+)
Core Pro has the **steepest fee drop per dollar** and the **lowest break-even**:
- Subs/galleries 20%→5% saves 15 pts → breaks even at **~$56/mo** of sub+gallery revenue
  ($8.33 ÷ 0.15).
- Exclusives 20%→10% saves 10 pts → breaks even at **~$83/mo** of singles.
- **Blended break-even ≈ $60–80/mo.**

Core+ is the trap: its 12–15% fees barely beat the 20% baseline, so it only saves 8 pts on
subs (break-even ~$83/mo) and 5 pts on exclusives (~$133/mo) — *higher* break-even than Pro
despite costing less. **Core Pro+ / Core Max** only pay back above **~$2,000/yr of subscription
revenue** — far beyond a from-zero operator. The only sane path is **Free → Core Pro → (much
later, if subs scale) Core Pro+.**

### Yearly + 50% promo, not monthly
DA runs a **50%-off (sometimes 60–65%) promo every ~1–4 weeks** — there is almost always a sale,
so **never pay sticker.** The discount reliably hits the **yearly-paid-in-full** plan (the
displayed yearly prices often *are* the promo). Wait days for a banner, then buy yearly — it
locks the low fee for 12 months. (Expect the discount **not** to recur at year-2 renewal;
verify in-cart.)

### Buy trigger
Buy Core Pro the **first month DA sales sustain ~$60+**, **OR** the day you open the $5/$10
subscription tiers (Runbook Week 3–4) — whichever comes first. By that point the fee savings
repay the membership. **Do not buy during the canary.**

> **BD caveat that outranks all of this:** Core only pays off once you can *withdraw*, which
> from Bangladesh you currently **cannot** do cleanly on DA (§6). **Prove the payout path before
> spending a cent on Core.** If DA money is unbankable for you, treat DA as a pure (unpaid)
> discovery funnel and skip Core entirely.

---

## 6. Bangladesh payout — the honest path

This is the binding constraint, and it's where I'm most uncertain — so here is the unvarnished
picture. **Verify each rail yourself before relying on it.**

### DeviantArt: all three native rails are problematic-to-dead for a BD recipient
| DA rail | Bangladesh status |
|---|---|
| **PayPal** | **Cannot receive** in BD (send/shop only — Bangladesh Bank FX rule). Dead. |
| **Stripe (Express)** | BD is **not on Stripe's supported list** (Preview tier at best; can't open a local Stripe account). The research split on whether DA's *Stripe Express* payout-only product reaches BD — treat as **unconfirmed and fragile**. |
| **BitPay (crypto)** | BitPay **explicitly names Bangladesh** among countries it cannot service (OFAC/licensing). Dead as a recipient. |
| Points | Never cash out. Ignore. |
| Payoneer | **DA does not support Payoneer** as a payout method. |

**Net:** there is **no clean, native DA→BD-bank path** in 2026. Two realistic postures:
1. **(Recommended) Treat DeviantArt as the unmonetized SFW discovery funnel** — watchers,
   Groups reach, free covers, every gated post linking to Fanvue. Don't bank on harvesting DA's
   micro-sales. **Try Stripe Express onboarding from DA's Earnings page once** (it's the only DA
   rail with any chance) — if it accepts your BD details, great, that's a bonus; if it rejects
   BD, you've lost nothing because Fanvue is the real rail.
2. **(Advanced/optional, not the plan)** a foreign-jurisdiction PayPal/Stripe held by a fully
   trusted person abroad — real counterparty + account-stability risk, may break the proxy
   platform's terms. **Not vetted here; not the default.**

### Fanvue: the real money rail (this works from BD)
Fanvue is the revenue engine *and* the easier-to-get-paid engine. Set it up as method-1 + fallbacks:
| Fanvue payout | BD status | Notes |
|---|---|---|
| **Bank Transfer** | "available to all locations worldwide" (SWIFT) | ~$20–50 min, weekly 1–4 business days. **Try as method 1.** Conservative BD banks *may* question adult-industry deposits → keep a fallback live. |
| **Payoneer** | **Works in BD** | Withdraw to BD bank in BDT, or via the **Payoneer↔bKash/upay** partnership (~1%). The cleanest cash-out. |
| **Crypto (USDT)** | Works via P2P | ~$50 min; cash out USDT→BDT on Binance/Bybit/Bitget P2P → bKash/Nagad/bank (~120–123 BDT/USDT). The standard BD route if the bank balks. |
Fanvue does **not** offer PayPal. KYC = your real gov ID + selfie (Ondato); AI personas allowed;
up to **15 linked accounts only AFTER your first withdrawal**; you keep **85% for the first 12
months**, then 80/20.

### Decision
- **Fanvue = primary, only-reliably-bankable revenue rail.** Bank Transfer first; Payoneer +
  crypto-USDT configured as fallbacks so one frozen rail never strands you.
- **DeviantArt = funnel.** Attempt Stripe Express once; don't depend on it.
- **Keep a Payoneer account** as your universal BD freelancer hedge (worth the ~$30/yr only once
  you receive ≥~$2,000/yr, else the inactivity fee bites).

> **Honest uncertainty:** BD payout is the weakest-verified part of this guide. Fanvue's exact
> minimums/fees come from third-party aggregators, BD is not on a *public* Fanvue supported-
> countries list, and DA's Stripe-Express-for-BD support is unconfirmed. **The real test is
> running KYC + adding a payout method on each platform yourself.** Do that before building the
> whole funnel on top of it.

---

## 7. Gallery vs Shop vs Subscriptions vs Premium Galleries — what goes where + how to price

DA has one **free Gallery** layer (discovery) and **three paid Shop mechanics** (all require Core).
This mirrors the pipeline's `public/` vs `gated/` split exactly.

| Mechanic | What it is | Your lane | Tier placement |
|---|---|---|---|
| **Free Gallery folders** | discovery layer | 2 SFW covers/package + T1/T2 + **fine-art-framed T3** (B&W/painterly/classical), Mature-tagged + AI-labeled | **public/** only |
| **Exclusives** | per-image one-off sale | **$2–3 micro** lane + **$6–8 hero-4K** lane (resale ON; attach clean hi-res file URL) | singles |
| **Premium Galleries** | buy-once access to a whole growing folder | **ONE ratchet-priced gallery per family** | gated, 18+ |
| **Subscriptions** | recurring tiers | **$5 "Muse"** (T3) / **$10 "Private Vault"** (T4 + watermark-free 4K + early) | gated, 18+ |

**Tier placement (policy-exact, unchanged from §3 of DA_GO_TO_MARKET):**
- **PUBLIC free:** all T1/T2 + 2 SFW covers + fine-art-framed T3 only. Mature-tagged + AI-labeled.
- **GATED only (Premium Galleries + Subs, 18+):** **ALL T4 without exception** + glamour-styled T3.
- **NEVER public:** any mature cover/thumbnail (covers come only from `public/`); **4K masters**
  (Private Vault + Fanvue only — watermark-free 4K *is* the paid product).
- **#1 pre-post rule:** **visual tier-truth QA of `public/`** — Chroma strips clothing on ~20–30%
  of T1/T2 renders; the NudeNet gate quarantines to `_tier_drift/`, but your eyes are final.

**How to set prices (the clicks):**
- **Single (micro/hero):** on the deviation → **"Sell Deviation" → Exclusives** → fixed price
  field (`2`, `3`, `6`, `8`) → optionally enable **"accept offers"** on hero 4Ks → **resale ON**.
- **Family gallery:** Gallery carousel → **Edit → New Gallery → Premium Gallery** → title + price
  → **launch $5**, then **ratchet +$1–2 per added 6-image set, cap ~$15** (price rises hit **only
  new buyers** — early buyers are locked in). Stays hidden until you add the first deviation;
  JPG/PNG/MP4 only.
- **Subscription:** Subscriptions tab → **"Create a Tier"** → name/price/mature/description/cover.

> **⚠️ Subscription prices are PERMANENT once published** (unlike galleries, which you ratchet
> freely). To charge more you must publish a **new** tier and migrate — losing the old tier's
> social proof. **Launch $5 Muse and $10 Private Vault at prices you can live with long-term.**
> (82% of DA's top-100 sub tiers are ≤$10, so $5/$10 is well-anchored.)

---

## 8. Families + personas organization

**Folders:** exactly **5 top-level public Gallery folders** — ATELIER, GILDED GLAMOUR, GOLDEN
HOUR, MYTH & CROWN, NOCTURNE — each with a 1–2 sentence description (helps DA search), plus
**Featured** (rotating best SFW) and a hidden **Archive** (culls/tests). Map niche `da_folder`s
to **subfolders inside the family**, never new top-level folders. Create **one Premium Gallery
per family** (5 total).

**Personas (Clara/Sable/Margot/Imani/Mei) are NOT folders at launch.** Bind a persona as the
**prose/serial identity inside family sets** (the titling layer: `"Clara — Candlelit Reverie III"`,
`Atelier No. 014 — "Marble Light"`).

**When to open a per-persona "Room"** (a dedicated Subscription tier/gallery, the archi444 model):
**only AFTER a persona demonstrably recurs and pulls favorites** — gate = **≥10 paying subs OR
~1K watchers.** `lunasilverlake`'s 5 near-dead tiers are the cautionary case for spinning Rooms
up too early. Seed the initial galleries from the 3 existing Clara sets (re-watermarked) →
GILDED / MYTH&CROWN / GOLDEN HOUR; open Clara's Room only when she clears the gate.

---

## 9. The from-ZERO sequence (today → first upload)

Aligned to GO_LIVE_RUNBOOK's canary. **The canary batch (Atelier fine-art T3 + Myth&Crown T4) is
rendering now — do the account/identity/payout work while it finishes.**

**A. Identity + accounts (while the batch renders)**
1. Create the **dedicated brand Gmail** + 2FA + non-personal recovery (§3).
2. **Reserve `MaisonLumiere`** on DA + Fanvue + IG + domain in one sitting (§1); confirm on DA
   itself before buying the domain.
3. **Create the DA account** (free) → birthdate 18+ → SFW avatar + bio (tagline + AI disclosure +
   Fanvue link) → enable mature filters (§4).
4. **Wire `@MaisonLumiere` into `config/pipeline.yaml::watermark.text`** + run the no-GPU
   **re-watermark pass** (Claude can do both). Do NOT post placeholder-watermarked images.
5. **Build the 5 family folders** (+ Featured + hidden Archive) with descriptions.
6. **Create the Fanvue account** → pass Ondato KYC → rename to `MaisonLumiere` → AI-disclosure
   bio line.

**B. Prove the money rail (the real first canary)**
7. **Set up Fanvue payout:** Bank Transfer (method 1) + Payoneer + crypto-USDT as fallbacks.
   Run KYC end-to-end. **Verify you can actually add a BD payout method before relying on it.**
8. **(Optional)** attempt **Stripe Express** onboarding from DA's Earnings page once — keep DA
   money if it accepts BD, drop it without worry if it rejects.

**C. Content canary (Week 1; then watch 2 weeks — GO_LIVE_RUNBOOK Phase 1)**
9. **QA the rendered batch:** open `contact_sheet.png`; eyeball single-subject, anatomy,
   sharpness; **visual tier-truth QA of `public/`** (the #1 rule).
10. Post **ONE public fine-art T3** (Atelier `fine_art_figure_study`) via its
    `posting_templates/` — Mature tag + AI label; **never** "photorealistic/realistic/real woman"
    in title/tags/description.
11. Post **ONE gated T4** (Myth&Crown `renaissance_baroque`) — gated only (a Premium Gallery or
    the $10 tier once Core is on); never public.
12. **Watch 2 weeks** for removals/moderation/flags/throttling. Clean → scale; flagged → reframe
    harder toward fine-art/fantasy and route camera-real explicit Fanvue-primary.

**D. Scale (Weeks 2–4+, per the existing docs)**
13. **Buy Core Pro (yearly, on a 50% promo)** when DA sales clear ~$60/mo OR you open the sub
    tiers — and only after the payout path is proven (§5/§6).
14. Launch the **5 Premium Galleries at $5** (ratchet to ~$15) + start **Mon/Wed/Fri** cadence.
15. Open **$5 Muse / $10 Private Vault** at ≥3 sets of inventory (prices are permanent) + open
    **Fanvue** with the backlog + start the micro-Exclusives lane.

---

## 10. Things you didn't ask about (gaps & risks)

- **Watermark handle is still `@YourDAHandle`.** Nothing should be posted until step A.4 wires
  `@MaisonLumiere` in and re-watermarks. (GO_LIVE_RUNBOOK flags this; it's easy to forget.)
- **Back up the originals.** `output/art_series/<ts>/` is your only master — **DA can delete
  content at will** and Fanvue/Stripe accounts can be actioned. Keep an off-machine backup. And
  **never delete `manifest.json` files** — they are the pipeline's diversity memory.
- **Record-keeping for AI nudes.** Real-performer 2257 records don't apply to fictional AI
  subjects, but keep your own provenance trail anyway: the generation manifests, prompts, and an
  age-safety note (adult anchors, no minor descriptors) per set — useful if any platform ever
  queries "who is this person." Your subjects are fictional; document that they're fictional.
- **Tax / withdrawal thresholds.** Payoneer charges ~$30/yr inactivity if you receive <~$2,000/yr;
  Fanvue holds funds ~7 days and processes weekly; DA holds all funds in a **7-day pending window**
  before withdrawal. Keep amounts modest and steady — large irregular adult-industry deposits draw
  the most bank scrutiny in BD. (Income-reporting is your own responsibility; out of scope here.)
- **Fanvue-FIRST for explicit, always.** DA's hyperrealism clause is a permanent kill-switch on a
  photoreal NSFW catalog. Keep DA's T4 volume modest and fantasy/period-framed; route camera-real
  explicit to Fanvue. **Do not build income expectations on DA** — it's the funnel; Fanvue is the
  business (median DA seller in this genre earns low-hundreds/yr).
- **Don't trust third-party checkers as final.** Claimbrand/Qezir can lag — always confirm the
  handle on DA itself before buying the matching domain/IG.
- **VPN consistency + one browser profile** — re-stated because IP/country flip-flopping during
  KYC is the most common avoidable cause of payout-account freezes.
- **Trademark sanity check.** Do a quick search that `MaisonLumiere` doesn't collide with an
  existing studio/brand before buying the domain — these are generated suggestions, not cleared marks.

---

## ▶ Immediate next 3 actions (while the batch finishes)

1. **Create the dedicated brand Gmail** (For-my-personal-use, 2FA, non-personal recovery) — §3.
2. **Run `MaisonLumiere` through Claimbrand/Qezir + `deviantart.com/MaisonLumiere`**, confirm it's
   free on DA + Fanvue + IG + domain, and **reserve all four** (fallbacks: AtelierNoir →
   MuseAndMarble → NoirBoudoir) — §1.
3. **Wire `@MaisonLumiere` into `config/pipeline.yaml::watermark.text`** and run the no-GPU
   re-watermark pass so the canary sets post with the real handle, not the placeholder — §4 step 5.
   *(Claude can do this one now — just give the go-ahead.)*
