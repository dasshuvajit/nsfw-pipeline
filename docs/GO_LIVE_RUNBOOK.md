# GO-LIVE RUNBOOK — from packages-on-disk to posting

> The single **actionable sequence** to take this pipeline live. Consolidates the
> strategy in [DA_GO_TO_MARKET.md](DA_GO_TO_MARKET.md) (§0 setup, §0b hyperrealism
> posture, §10 phasing) and the policy in [DA_POSTING_PLAYBOOK.md](DA_POSTING_PLAYBOOK.md)
> into a do-this-then-this checklist. Upload is **manual** by design.
>
> **Current state (2026-06-15):** DeviantArt account **not yet created**; the
> watermark handle is still the placeholder `@YourDAHandle`; the Week-1 canary sets
> are **being generated** (Atelier fine-art T3 + Myth&Crown T4). Inventory on disk:
> 3 Clara persona sets (old-Hollywood T3, renaissance T3, poolside T2 — 5 clean
> frames) + a few older non-persona sets.

## Phase 0 — Account + config (BLOCKER — nothing posts until this is done)
1. **Create the DeviantArt account** + buy **Paid Core membership** (drops the sale
   fee 20% → ~10%, lower at higher tiers). Complete **18+ / ID verification** (required
   to sell mature content).
2. **Set payout** = PayPal/Stripe in DA settings (**NOT** Points — purchased Points
   don't cash out).
3. **Set your real handle** → `config/pipeline.yaml::watermark.text` (currently the
   placeholder `@YourDAHandle`). Then **re-watermark** the packages you intend to post.
   - This is a **cheap re-watermark, NOT a re-render**: the `images/` review frames are
     clean (unwatermarked); `_package` bakes the watermark onto the `public/` copies. A
     re-watermark pass re-applies the new handle to the clean frames — no GPU.
   - *(Claude can do steps 3's config edit + the re-watermark pass once you give the handle.)*
4. **Build the 5-family gallery tree** + a 1–2 sentence description per folder (helps DA
   search): **ATELIER · GILDED GLAMOUR · GOLDEN HOUR · MYTH & CROWN · NOCTURNE**
   (+ **Featured**, + hidden **Archive**). Niche `da_folder`s become subfolders.
5. **(Funnel) Create a Fanvue account** + put the **AI disclosure** in the bio (Fanvue
   requires prominent AI disclosure; hiding it risks termination). Patreon stays **banned**
   for synthetic AI nudes — do not use it.

## Phase 1 — The canary (Week 1; then watch 2 weeks before scaling)
The #1 policy risk is DA's **hyperrealism clause** (§0b): photoreal explicit that reads as a
real person is bannable. So post a **small canary** and watch before volume.
- **Public fine-art T3 → the Atelier `fine_art_figure_study` T3 set** (generating now).
  B&W / fine-art framing reads as "art" to moderators — the safest *public* mature post.
  Post via its `posting_templates/` (Mature tag + AI label). **Never** write
  "photorealistic / realistic / real woman" in the title/tags/description — frame as
  "fine-art figure study."
- **Gated T4 → the `renaissance_baroque` T4 set** (generating now). **ALL T4 is gated**
  (Premium Gallery / $10 sub tier) — never public (public explicit is a ToS violation).
  Old-master period framing is the safest T4 carrier.
- **Watch 2 weeks** for: removals, moderation messages, account flags, reach throttling.
  Clean → scale (Phase 2). Flagged → pull back, reframe harder toward fine-art/fantasy,
  and route camera-real explicit **Fanvue-primary**.

## Phase 2 — Families live (Week 2–3)
- Launch all **5 Premium Galleries at $5** (ratchet **+$1–2 per added 6-image set, cap ~$15**;
  price rises hit only new buyers — early buyers locked in).
- Start the **Mon/Wed/Fri cadence** (§5, ≤4 posts/day, human-paced — never scheduled, to
  avoid the bulk-posting ban vector): **Mon** 2 SFW covers → 3–5 Groups + 1 watchers-only
  bonus · **Wed** 6 gated → family Premium Gallery + 1–2 blurred Group teasers + sub tier +
  Fanvue mirror · **Fri** 2–3 micro-Exclusives ($2–3) + 1 re-post.
- Seed galleries from inventory: the **3 Clara sets** (re-watermarked) feed GILDED / MYTH&CROWN /
  GOLDEN HOUR; once Clara recurs enough (≥10 subs or 1K watchers) open her **Room**.

## Phase 3 — Subscriptions + Fanvue (Week 4+)
- Open **$5 "Muse"** (all T3 sets + hi-res + alt takes) and **$10 "Private Vault"** (T4 +
  watermark-free 4K + 1-week early access) once ≥3 sets of inventory exist.
- Open **Fanvue** with the backlog; begin the **micro-Exclusives** lane ($2–3, resale ON).
- 4K masters: **Private Vault + Fanvue only, never public.**

## Per-set posting mechanics (every package already ships these)
Each `output/art_series/<ts>/package/<niche>/` contains: `contact_sheet.png` (QA),
`POSTING_CHECKLIST.md`, `posting_templates/<image>.txt`
(TITLE / FOLDER / DESCRIPTION / TAGS / GROUPS / MATURE / AI-LABEL / PRICE), `metadata.json`.
- **Always: visual tier-truth QA of `public/`** before posting — the **#1 rule**. The NudeNet
  gate quarantines drift to `_tier_drift/`, but eyes are the final check (Chroma can strip
  clothing on T1/T2). `gated/` may contain nudity; `public/` must not.
- Mature tag + AI label ("Created using AI tools") on **every** post.
- Fanvue link in DA **bio + About + gated/blurred post descriptions only** — keep it off the
  public SFW teasers (those are for Groups reach).

## Known cautions
- **Watermark = placeholder** until Phase 0.3 — do not post placeholder-watermarked images.
- **`poolside_goldenhour` at T2 leaks nude ~50%** (render gate catches it, but ~half the
  renders are wasted + sets shrink). Run poolside at **T1** for public sets (it has no T3).
- **DA is a weak *direct* earner** for this genre (median seller ~low-hundreds/yr) — it's the
  SFW discovery funnel; **recurring revenue is off-site on Fanvue**. Price DA impulse-low,
  build the funnel.

## Division of labor
- **Claude can:** set the watermark handle + re-watermark packages; generate canary/family/
  persona content; verify tier-truth; tune niches (e.g. poolside → T1); draft titles/tags.
- **You (manual):** create DA + Fanvue accounts, buy Core, set payouts, build the galleries,
  post on the human-paced cadence, watch moderation, handle payouts.
