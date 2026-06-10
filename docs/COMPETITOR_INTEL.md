# Competitor / creator intel (DeviantArt AI women-art sellers)

> **Purpose.** A living research file: profiles of DeviantArt creators selling AI
> "beautiful women" / glamour / fine-art-nude content, plus the cross-creator
> patterns and a candidate-implementation backlog for our pipeline. **When a new
> creator link is added, append a new `## Creator —` section using the template
> at the bottom and update the synthesis table + backlog.** Keep it honest: mark
> confirmed vs inferred; we cannot see private sales numbers.
>
> Compiled from read-only public-page research (WebFetch + web search; DeviantArt
> is React/Cloudflare-heavy so individual-artwork URLs render best). Last updated
> 2026-06-04 with **13 creators** (5 in batch 1, 8 in batch 2).

## Honest caveats (read first)
- **Catalog size ≠ revenue.** DA sales counts are private. Our own
  [DA_POSTING_PLAYBOOK](DA_POSTING_PLAYBOOK.md) research found DA is a *weak*
  earner for this genre (median seller ~low-hundreds/yr); real money is usually
  off-site. Treat "big catalog + cheap price" as a *hypothesis to test*, not
  proven income.
- **All 13 are AI.** Validates our approach. Several disclose "Created using AI
  tools" (DA AI-label compliance — we do this too).
- **Two style lanes in the market.** (a) **photoreal glamour/boudoir/fantasy** —
  OUR lane (andy-varhall, archi444, rebaaane, artbyinnovation, haselnusskrokant/
  softember, lunasilverlake, primalcarnalvore, loveoai, mikeyxxxx); (b) **anime/
  waifu adoptables** (mi6okasla, much of zahra-arts) — huge volume but a DIFFERENT
  style our photoreal Chroma pipeline doesn't target. Don't chase the anime lane.
- **Scope flags (do NOT emulate the content):** `mikeyxxxx` includes schoolgirl
  (age-ambiguity — we hard-avoid) + futa (not single-female); `primalcarnalvore`'s
  handle signals fetish but the content is in-scope glamour (a *branding*
  anti-lesson). Selling *mechanics* can still transfer; the content does not.
- **Don't copy anyone's art.** `haselnusskrokant` explicitly forbids AI-training
  on her work. We learn the *selling model + style direction*, never the pixels.
- **Patreon stays off the table** (bans synthetic AI nudes) — consistent with our
  playbook. The off-site funnels these creators use are Ko-fi / Substack / own-site
  / Fanvue / IG, not Patreon.

## Synthesis table

| Creator | Medium | Niche / style | Catalog | Pricing | Monetization | Off-site |
|---|---|---|---|---|---|---|
| **andy-varhall** | AI + Photoshop | Fantasy/historical "beautiful women" (Renaissance, medieval, Arabian Nights, myth) | ~1,860 exclusives / 3.8K devs / 7.8K watchers / 13 yr / Core | $1–100, modal **$5–6** | DA exclusives (volume) | IG only |
| **archi444** | AI photoreal | Glamour / implied-nude **recurring named "synthetic influencer" characters** | 2,358 exclusives / 39K devs / **21.4K watchers** / ~1 yr | excl. $8–12; subs **$5–10/mo** | DA exclusives **+ per-character subscription "Rooms" + "Uncensored Room" ($10/mo explicit)** | none (DA-only) |
| **rebaaane** | AI (DreamUp) | **Gothic-fantasy sensual** (elves/witches), fine-art-nude/glamour; **B&W "Monochrome Muses"** | 2,200 devs / ~490 premium / 1.5K watchers / ~1 yr | ~$2–25/item | DA premium + watchers-only gating | **Ko-fi + PayPal** |
| **artbyinnovation** (Jeff Dupler) | AI (SDXL/Flux/ComfyUI) + PS | Multi: angels (tasteful nude), demons/vampires (explicit), fantasy warriors, "Women Around the World" | ~6,000 female devs / 40K total | excl. $5–10 | DA premium + **own-site membership ($9.95/mo, $99/yr, $39.95 lifetime) + "All Access 4K" $10/mo** | **own site** artbyinnovation.com |
| **haselnusskrokant** | AI (DreamUp) | **Golden-noir Victorian boudoir/glamour**, semi-nude, femme-empowerment | **~10,858 exclusives** / 11.7K devs / 3.5K watchers / 10 mo | **$1–3** (micro) | DA exclusives **+ resale economy** + watchers-only | **Ko-fi + Substack + IG** |
| **softember** | AI | Romantic/ethereal **SFW-leaning** literary fantasy portraiture — *the SFW sister account of haselnusskrokant* | 7,126 excl / 7,200 devs / 1.3K w / 8 mo | $1–7 modal **$2–3** | DA exclusives + resale | IG + TikTok |
| **ai-engyth3cat** | AI | Female fantasy figures (gothic/cyberpunk/warm) + recurring "Sakura"; NSFW gallery | 534 shop / 966 devs / 1.6K w / **2 mo** | $5–20 modal $7 | DA exclusives + premium + **"make first offer" auction** | none |
| **lunasilverlake** | AI | **Moon-goddess gothic-glamour** sensual fantasy "adoptables"; implied not explicit | 1,539 excl / 2,100 devs / 3.9K w / 1 yr | $5–30 modal $7–15 | DA exclusives + 5 sub tiers $1–3 + auction | none |
| **primalcarnalvore** | AI | Glamour/erotic **narrative** portraiture; geographic series (Gobi/Iceland), named chars (misleading handle, NOT vore) | 689 devs / 724 w / 7 yr | $2–5 | DA premium paywall | none |
| **loveoai** | AI (DreamUp) | Fantasy glamour + **fantasy-IP fan-art** (Skyrim/Witcher/Nordic) + NSFW; "4K wallpaper" framing | 230 excl / 372 devs / 1.6K w / 1 yr | $8–10 modal **$10** | DA exclusives + $10/mo sub + tip jar | IG @loveo.ai |
| **mikeyxxxx** ⚠ | AI (undisclosed) | Boudoir/glamour/lingerie **+ fetish** (legs, schoolgirl⚠, lapdancer, futa⚠); explicit-leaning | 5,698 works / 5.6K followers | paywalled | DA premium + **multi-platform** | Patreon(404)/IG/Fanvue/Tumblr |
| **mi6okasla** (anime) | AI (SLA LoRA) | **Anime/waifu adoptables** (not photoreal); NSFW | **24,604 excl** / 29K devs / 2.1K w / 2 yr | **$3** + DLC 200-img/$3 | DA exclusives + $5/mo NSFW sub + **10% resale royalty** | none |
| **zahra-arts** (mixed) | AI (MJ/SD) | Adoptables: clothed fantasy → glamour → gated NSFW (anime + realistic) | **27,218 excl** / 28K devs / **7.9K w** / 2 yr | $5–8; **bulk 100/$200** | DA exclusives + **maturity-tiered** premium galleries $3–5 | none |

## Cross-creator patterns (the takeaways)

1. **AI + high volume + cheap price is the dominant model.** Catalogs of 2K–11K
   items at $1–$12. Impulse pricing; volume is the engine. Our pipeline produces
   *curated small sets* — the opposite. There's a case for a **high-volume cheap-
   exclusives lane** in addition to curated premium sets.
2. **Fantasy/historical framing is the most common style + a moat.** 4/5 lean on
   it (Renaissance, Arabian, myth, elves, witches, angels, demons, Victorian). It
   (a) dodges DA's "hyperrealism / real person" clause, (b) adds narrative
   collectibility, (c) reads as "art." We have `fantasy_glamour` + `goth_romantic`
   — under-built vs the field.
3. **Recurring named CHARACTERS / "synthetic influencers."** archi444's Lina Morel
   / Ginger Love / Eve Noir each get their own subscription Room. Fans attach to a
   *character*, not a single image → recurring revenue + a reason to subscribe. We
   have `persona_pool` (under-used) — this is the strongest under-exploited lever.
4. **Tiered explicitness ladder per subject.** clothed/SFW free → implied/lingerie
   cheap → explicit gated ("Uncensored Room" $10/mo; demon tier). Exactly our
   tier-split (T1/T2 public covers, T3/T4 gated) — validates it; the new idea is
   *same character across all tiers* to climb buyers up the ladder.
5. **Monochrome / B&W as a deliberate fine-art premium** (rebaaane's 208-piece
   Monochrome Muses). Positions work as "fine art," not "content." We have no B&W
   lane.
6. **Narrative / poetic titling + tight themed folders + series numbering**
   ("Velvet Roses and Wicked Girls", "#hotelnight Ginger Love", "Renaissance Lady
   3", "Monochrome Sensuality"). Drives DA search + collectibility. We have
   `da_folder` + a metadata generator; we could add poetic series naming + numbers.
7. **Monetization menu (pick a stack):** DA exclusives (cheap, one-time) · DA
   Subscriptions / Premium Galleries (recurring, per-character or "all-access") ·
   own-site membership · Ko-fi / Substack memberships · watchers-only gating ·
   resale economy · IG/X for discovery. **No Patreon.** Most run 2–3 in parallel.
8. **Diversity sells** (artbyinnovation "Women Around the World" 322; haselnuss
   diversity positioning) — vary ethnicity / skin tone / body type / age across
   outputs.

### Batch-2 reinforcements + new patterns
9. **Recurring named characters reconfirmed — now the #1 repeated signal.** 6/13
   build around named, reusable women (archi444's Lina Morel; ai-engyth3cat's
   Sakura; lunasilverlake's moon-goddess; primalcarnalvore's Beverly/Lena;
   loveoai's Nordic archetypes). Fans collect a *character*. → reinforces **G2**.
10. **"Adoptables" framing + DA's native resale economy.** Selling each AI woman as
    an ownable "adoptable character" with **creator resale royalties** (mi6okasla
    10%) turns buyers into resellers (network effects + "top supporters" status).
    Pairs with named characters. (Our model = curated *sets*, not adoptables — but
    the named-character + ownership framing is the bridge.)
11. **Industrial numbered-series at massive scale.** 24K–27K catalogs with
    sequential SKU names ("SLA ART 5120x2880 #####", "Zahra's AI Arts - #24591",
    "Beach Bar 1-119"). Confirms **G5** (series numbering) and the high-volume lane.
12. **Tiered-MATURITY galleries** (zahra: Kids → Cute → Hot → 18+ → NSFW). Per-folder
    maturity segmentation protects SFW algorithm reach *and* gates adult revenue —
    a more granular public ladder than our binary public/gated split.
13. **Auction + bulk pricing.** "Make first offer" (ai-engyth3cat, lunasilverlake)
    + wholesale bundles (zahra 100-for-$200; mi6okasla 200-image DLC packs).
14. **Dual-account brand segmentation** (softember SFW ↔ haselnusskrokant sensual):
    a clean SFW discovery brand + a separate sensual revenue brand. Maps to our
    SFW-cover/gated split, executed as two accounts.
15. **Fantasy-IP anchoring + "4K wallpaper / decor" framing** (loveoai: Skyrim /
    Witcher / Nordic). Ties output to existing fandoms (built-in audience + search)
    and frames it as aspirational 4K decor, not "porn" — legitimizing + moderation-
    softening. ⚠ Fan-art of game IP carries its own copyright caveat; prefer
    *generic* fantasy archetypes (elf, shieldmaiden, sorceress) over named IP.

## Mapping to our pipeline — what we have vs gaps

**Already have (validated by the field):** AI photoreal pipeline; tier-split
public/gated packaging (= the explicitness ladder); per-niche `da_folder`s + tags
+ keyword-rich metadata; aesthetic-lock signature look; persona support;
watermarking; the [posting playbook](DA_POSTING_PLAYBOOK.md) (SFW shopfront, AI
label, Groups-for-reach, no Patreon).

**Gaps the research surfaces (→ candidate backlog):**
- **G1 — Thin fantasy/historical coverage.** Only `fantasy_glamour` + `goth_romantic`.
  Field leans heavily here. → add Renaissance/baroque, Arabian-Nights/orientalist,
  mythology/goddess (Medusa, Aphrodite), medieval "ladies", witch/sorceress,
  angelic/divine, dark-fantasy/vampire niches.
- **G2 — Recurring named characters under-used.** `persona_pool` exists but
  `--auto` rarely binds one. → lean into recurring personas (a "character" with a
  name, consistent identity across a series/subscription Room). Pairs with the
  planned face-lock (IPAdapter/ReActor) for true identity consistency.
- **G3 — No B&W / monochrome lane.** → a monochrome aesthetic profile / niche
  ("fine-art B&W") as a premium signature.
- **G4 — Single (curated-set) selling motion.** → document/support a high-volume
  cheap-exclusives lane + a per-character subscription-Room motion in the playbook
  (not just curated premium sets).
- **G5 — Plain titling.** → poetic/evocative series titles + series numbering in
  the metadata generator ("<Persona> — <Poetic title> N").
- **G6 — Diversity not explicit.** → make ethnicity/skin-tone/body-type variation
  an explicit axis in niche briefs / variation mode.
- **G7 — Selling-motion ideas (batch 2, playbook-level, manual — not code):**
  adoptables/ownership framing + DA resale-royalty flywheel; tiered-maturity
  galleries (more granular than binary public/gated); auction "make first offer" +
  bulk/wholesale tiers; dual-account SFW↔sensual brand split; "4K wallpaper/decor"
  framing. These are operator decisions for [DA_POSTING_PLAYBOOK](DA_POSTING_PLAYBOOK.md),
  not pipeline code.

> These are **candidates for the implementation-planning step** (not yet built).
> Highest-leverage + lowest-risk first: **G1 (fantasy/historical niches)** and
> **G2 (recurring personas)** — both slot straight into the existing niche library
> + persona system, match the field's proven sweet spot, and G2 is now the single
> most-repeated pattern (6/13 creators build around named characters).

---

## Creator — andy-varhall
- **Medium:** AI, heavily Photoshop-refined (states it openly). Painterly-cinematic, warm saturated palettes, soft atmospheric light. *Not* photoreal studio nude.
- **Niche:** Fantasy/historical "beautiful women" — "Beautiful girl" (2,807 devs!), "Girls of the Middle Ages", "1001 Nights: Arabian Tales for Adults", "Captive Girls" (premium), "Renaissance Lady", "Medusa Gorgon", "Mother Winter", "Fantasy Worlds" (331). Sensual/mature but fantasy-framed, mostly implied — not explicit boudoir.
- **Selling:** ~1,860 USD exclusives, $1–100 modal **$5–6**; watermarked; tight themed folders + **series numbering** ("Renaissance Lady 3"). DA-native (no obvious off-site funnel); 13 yr, 7.8K watchers, Core. IG @andrii_kobyshcha.
- **Steal:** fantasy/historical framing as moat + collectibility; deep themed catalog; series numbering.

## Creator — archi444
- **Medium:** AI photoreal ("synthetic influencer" look; "Created using AI tools"). Pro beauty-photography lighting, warm cinematic grading, 1152×1728 portrait framing.
- **Niche:** Glamour / implied-nude → explicit (gated). Clothed gallery free; lingerie + nude in paid Rooms.
- **Style signature:** **Recurring named characters** — Lina Morel (blonde), Ginger Love (red), Aylin Sora, Eve Noir, Nika Solenko. Hashtag theme collections (#hotelnight, #ethereal).
- **Selling:** 2,358 exclusives ($8–12) / 39K devs / **21.4K watchers** / 118K profile views / ~1 yr / ~200 items/mo. **Per-character subscription "Private Rooms" ($5/mo) + "The Uncensored Room" ($10/mo, explicit).** DA-only, no off-site.
- **Steal:** **recurring-character roster → per-character subscription Rooms**; tiered explicitness ladder (clothed→lingerie→explicit) on the *same* character; hashtag theme waves; high output velocity.

## Creator — rebaaane
- **Medium:** AI (DreamUp; "Created using AI tools", aiart-tagged). Photoreal skin via AI + gothic fantasy.
- **Niche:** Gothic-fantasy sensual portraiture, fine-art-nude/glamour hybrid; elves/witches/sorceresses, smoke/moonlight/darkness. Implied→nude, artistic not pornographic.
- **Style signature:** cool jewel tones (crimson/purple/black) + dramatic low-key + atmosphere; **strong B&W "Monochrome Muses" (208)**; solitary close-up figures.
- **Selling:** 2,200 devs / ~490 premium downloads / 1.5K watchers / ~1 yr / ~6/day. Named collections (Echoes of Elven 205, Entwined Souls 91 watchers-only, Monochrome Muses 208). **Ko-fi + PayPal** off-site. Poetic titling ("Lustful Waves", "Satin Roses").
- **Steal:** gothic-fantasy-sensual fusion; **monochrome as a premium fine-art signature**; watchers-only gating funnel; named themed collections; poetic SEO titling.

## Creator — artbyinnovation (Jeff Dupler)
- **Medium:** AI — **Stable Diffusion SDXL / 3.5 / Flux in ComfyUI, finished in Photoshop** (same stack as us). States "AI-generated, Photoshop Art". 4K (3840×2160) output.
- **Niche:** Multi-genre, gender-parallel lines. Female: angels/divine (tasteful nude), demons/vampires (explicit, gated), fantasy warriors (elves/warlocks), "Beautiful Women Around the World" (322, photoreal global diversity).
- **Selling (dual-channel):** own site **membership $9.95/mo · $99/yr · $39.95 lifetime** (unlimited unwatermarked 4K) **+** DA premium downloads $5–10/item **+** DA "All Access Gallery 4K" $10/mo (1,761 devs). ~6,000 female devs. Watermark-as-funnel (removal = the membership value). Descriptive SEO titles ("Holy Divine Female Angels").
- **Steal:** **own-site membership + marketplace arbitrage** (sell same inventory as subscription AND per-item); **divine/angel framing to justify tasteful nudity** vs demon tier for explicit; 4K-without-watermark as the paid value; global-diversity series.

## Creator — haselnusskrokant
- **Medium:** AI (DreamUp). ⚠️ **Explicitly anti-AI-training** — do not copy her work; model only.
- **Niche:** **Golden-noir Victorian boudoir/glamour**, semi-nude/implied (corsets, lace, latex), femme-empowerment/diversity, narrative voice ("Velvet skin. Heavy lips. Slow art.").
- **Style signature:** warm golden tones + deep noir contrast + Portra/retro film stock; Victorian/period styling; window light; gaze-heavy intimate composition. ("In the style of": warm cinematic boudoir, Art-Deco/Victorian, golden-noir, retro film color, soft unhurried seduction.)
- **Selling:** **~10,858 exclusives at $1–$3 (micro-pricing)** + **resale-enabled** ("buy and resell to join my top supporters") + watchers-only Premium Galleries (Fallen Angels Club, etc.). 11.7K devs / 3.5K watchers / 47.7K views / 10 mo. **Ko-fi + Substack ("sugar-dark stories") + IG.** Poetic/multilingual series naming ("Femmes en bas: Stockings", "Latex After Midnight").
- **Steal:** **$1 micro-pricing + resale/collector economy**; **narrative poetic series + Substack story-pairing**; golden-noir Victorian signature; femme-empowerment positioning.

## Creator — softember  *(= haselnusskrokant's SFW sister account)*
- **Medium:** AI ("Created using AI tools").
- **Niche:** **Romantic / ethereal / SFW-leaning** literary fantasy portraiture — clothed/minimally-exposed, mood over form. Melancholic-dreamy (the only "sad/quiet" brand here — note our pipeline bans sad/crying, but their market tolerates it).
- **Style signature:** pastel/muted warm palette, soft diffused ethereal light, atmospheric haze, liminal settings (balconies, forests), butterflies/flowers; literary titling (Shakespeare — "Juliet on the Balcony", "Petals and Daydreams").
- **Selling:** 7,126 exclusives at **$1–$7 (modal $2–3)**, 50–70% promo discounts, resale; 7.2K devs / 1.3K watchers / 8 mo. IG + TikTok. 29+ thematic folders.
- **Steal:** **dual-account brand split** (this = the SFW discovery brand; haselnusskrokant = the sensual revenue brand); literary/poetic positioning that reads as fine-art, not "AI girls"; deep thematic-folder catalog.

## Creator — ai-engyth3cat
- **Medium:** AI (NSFW-tagged; handle literally "ai-"). ~95% conf.
- **Niche:** Female fantasy figures, mostly nude/sensual with fantasy/cyberpunk overlay; dedicated NSFW gallery (72). Mid-to-explicit. (I eyeballed one earlier this session — "Crimson Allure" — polished sensual AI.)
- **Style signature:** mixed — "Warm Creations" (golden, ~190), "Gothic Vision" (dark, 90), cyberpunk neon, nature/botanical; recurring **"Sakura" character** line (Premium/Exclusive/Standard versions).
- **Selling:** 534 shop items (374 exclusives + 159 premium) / 966 devs / **1.6K watchers in 2 months** (fast) / 48.4K favorites. $5–20 modal $7; **"make the first offer" auction**; resale. DA-only, no off-site.
- **Steal:** recurring character sold in tiered versions (Standard→Premium→Exclusive); auction pricing + sales-history social proof ("Sold for $X"); fast volume velocity; mood-based folder taxonomy.

## Creator — lunasilverlake
- **Medium:** AI ("Created using AI tools"). 100% conf.
- **Niche:** **Moon-goddess gothic-glamour sensual fantasy** sold as "exclusive adoptables"; implied/partial nudity, *not* explicit ("eroticfantasy"/"decolletage" tags). Fantasy-framed sensuality.
- **Style signature:** moonlit blues/silvers/purples + warm red accents; nocturnal/mystical; silver-white hair (moon-goddess archetype); reclining/contemplative; Klimt/Vallejo/surrealist influence. Named collections (SNOWLAND 117, Sylvan Sisters 127, "A deep breath decolletage" 168, gothicart 52, Woman in Red).
- **Selling:** **1,539 exclusive adoptables** / 2,100 devs / 3.9K watchers / 1 yr / ~175/mo. $5–30 modal $7–15; fixed + **"make first offer" auction**; 5 sub tiers $1–3 (near-zero uptake); resale (buyer = "Owner"). DA-only.
- **Steal:** **fantasy "adoptable" framing to de-risk sensual content**; 50–150-piece themed collections (buy the series, not the piece); dual fixed/auction pricing.

## Creator — primalcarnalvore
- **Medium:** AI. **Handle is misleading — ZERO vore/fetish; content is glamour/erotic portraiture.** IN-SCOPE (single female, glamour, lingerie/swimwear), edgier narrative tone.
- **Niche:** Photoreal glamour/erotic **narrative** portraiture — clothed/lingerie/swimwear, suggestive not explicit; transgressive narrative framing ("Lena's Wicked Wedding").
- **Style signature:** moody cinematic (candle/moonlight, jewel tones, warm/cool grade); environment-integrated figures; **geographic/cultural specificity** as a brand (Gobi/Mongolian, Iceland hot-springs); named characters (Beverly 163, Lena) + serialized collections ("72 Vegans" 330).
- **Selling:** small — 689 devs / 724 watchers / 7 yr / ~2–3/wk. DA platform-native paywall ($2–5 premium); no off-site. Quality-first (deprecates weak series).
- **Steal:** **serialized named-character narratives + geographic/cultural niche branding** (own an under-served aesthetic); narrative titles as the buy-reason. **Anti-lesson:** a fetish-signaling handle is a liability — pick a clean brand.

## Creator — loveoai
- **Medium:** AI (DreamUp; ex-oil-painter). 100% conf.
- **Niche:** Fantasy glamour + **fantasy-IP fan-art** (Skyrim/Witcher/Nordic shieldmaidens, elves) + explicit NSFW (bikini/nude/"spreading"); stylized-digital, leans glamour-with-explicit.
- **Style signature:** vibrant full color, soft studio/portrait light, dramatic face shadows; **"4K AI wallpaper" framing** (decor, not porn); fantasy archetype titles ("Nordic Shieldmaiden", "Wood Elf Protector").
- **Selling:** 230 exclusives ($8–10 modal **$10**) / 372 devs / 1.6K watchers / 1 yr / ~1/day. $10/mo sub + $2/mo tip jar. IG @loveo.ai (small).
- **Steal:** **fantasy-IP/fandom anchoring** for built-in audience + searchability (use *generic* archetypes to avoid IP copyright); **"4K wallpaper/decor" framing** to legitimize + soften moderation.

## Creator — mikeyxxxx  ⚠ partly out-of-scope
- **Medium:** AI presented AS photography (no AI disclosure; "model" nomenclature). ~85% AI.
- **Niche:** Boudoir/glamour/lingerie photography style **+ fetish/explicit** — ⚠ schoolgirl (age-ambiguity, we hard-avoid), lapdancer, futa (not single-female). Explicit-leaning.
- **Style signature:** warm saturated color, golden-hour/studio light, curve-emphasis posing; settings (bar/beach/camping/bedroom); heavily **numbered serial scenarios** ("Beach Bar 1-119", "Denim Model 1-78", "Lingerie Model 1-9").
- **Selling:** **5,698 works** / 5.6K followers (low engagement ratio = search-driven). Multi-platform (Patreon 404, IG, Fanvue, Tumblr, Twitter). Premium shop (paywalled).
- **Steal (mechanics only):** **serial-numbered scenario sets** (many variants from 2–3 base setups); **multi-platform spillover** (same art across DA/IG/Fanvue/Tumblr). **Do NOT emulate** the schoolgirl/futa/fetish content — off-scope for our single-adult-female, age-safe pipeline.

## Creator — mi6okasla  *(anime lane — different style from ours)*
- **Medium:** AI — **Stable Diffusion + "SLA" LoRA** (published on Civitai). Photoreal-ish anime/waifu. (I eyeballed one at session start — "PD SLA Art" — anime/semi-real.)
- **Niche:** **Anime/waifu adoptables**, NSFW; idealized feminine beauty. *Not* our photoreal fine-art lane.
- **Style signature:** SLA LoRA "sub-light aesthetic" — soft depth-of-field, anime shading; 5120×2880 (14.7MP); numbered "SLA ART 5120x2880 #####"; fungible (no recurring characters).
- **Selling:** **24,604 exclusives** ($3) / 29K devs / 2.1K watchers / 2 yr / ~40–50/day. **200-image DLC packs ($3)**; $5/mo NSFW sub (7 subs); **resale w/ 10% royalty**; tiered gallery (All/Featured/NSFW/DLC). DA-only.
- **Steal (selling only):** **resale-with-royalty** flywheel; DLC bulk packs; $3 impulse threshold; sequential-SKU inventory at scale. Style is anime — not transferable to our Chroma photoreal pipeline.

## Creator — zahra-arts  *(mixed; partly anime)*
- **Medium:** AI (Midjourney / Stable Diffusion). 100% conf. 5440–5888px.
- **Niche:** Mixed adoptables across a **maturity ladder** — Kids/Cute (SFW) → Hot Girls (glamour/pinup) → 18+/NSFW (464, gated). Anime + realistic. ~60% clothed/SFW public.
- **Style signature:** rich jewel tones, MJ-typical high-gloss; studio/neutral + fantasy; **numbered "Zahra's AI Arts - #24591"**; volume over signature.
- **Selling:** **27,218 exclusives** / 28K devs / **7.9K watchers** / 2 yr / ~38/day. $5–8 modal; **bulk 100-for-$200 ($2/unit)**; maturity-tiered premium galleries $3–5 (only 13 supporters — undermonetized); unwatermarked + commercial-use rights; auction "make first offer"; resale. DA-only.
- **Steal:** **tiered-maturity gallery segmentation** (SFW reach + gated adult revenue in parallel); bulk/wholesale tier; commercial-use-rights to reduce buyer friction.

---

## Template — adding a new creator
When a new link is given, research (read-only) and append:

```
## Creator — <handle>
- **Medium:** AI / photo / 3D + tooling; confidence.
- **Niche:** genre + artistic↔explicit + clothed/implied/nude.
- **Style signature:** palette/B&W, lighting, posing, settings, recurring series/characters.
- **Selling:** products, pricing (range+modal), catalog volume, watchers, tenure,
  on-DA subs/galleries, off-site funnel, watermarking, folder/titling conventions.
- **Steal:** 2–5 transferable ideas (style or selling).
```
Then update the **synthesis table**, the **cross-creator patterns**, and the
**candidate backlog (G1…)** if the new creator surfaces a new gap.
