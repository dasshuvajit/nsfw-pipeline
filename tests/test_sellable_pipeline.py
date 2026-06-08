"""Tests for the sellable-content Phase-1 wiring in scripts/art_series.py:
curation (_curate), tier-split packaging (_package), and the art_director
inline audit gate. The render/LLM calls are mocked — these assert the
orchestration + the safety-critical SFW-cover rule, not model output.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

import scripts.art_series as A
import scripts.art_director as AD
from src.niche.selector import NicheLibrary, build_selection


def _mk(d: Path, rel: str) -> Path:
    p = d / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    return p


@pytest.fixture
def tmp(tmp_path) -> Path:
    return tmp_path


# ── curation (_curate) ─────────────────────────────────────────────

class _FakeScorer:
    """Stand-in for ImageScorer: scores higher for lower seed, and flags the
    seed==99 image as multiple_faces (a hard reject)."""
    def __init__(self, **kwargs):
        pass

    def score_batch(self, images):
        for im in images:
            fp = im["file_path"]
            seed99 = "s99" in fp
            im["quality_score"] = 0.2 if seed99 else (0.9 if "s1." in fp else 0.5)
            im["quality_flags"] = json.dumps(["multiple_faces"] if seed99 else [])
        return images


def test_curate_keeps_top_and_drops_hard_reject(tmp, monkeypatch):
    monkeypatch.setattr("src.scoring.image_scorer.ImageScorer", _FakeScorer)
    man = [{"index": 0, "prompt": "p", "images": [
        {"path": "images/ad01_a_s1.png", "seed": 1},     # best (0.9)
        {"path": "images/ad01_a_s2.png", "seed": 2},     # mid  (0.5)
        {"path": "images/ad01_a_s99.png", "seed": 99},   # flagged multiple_faces
    ]}]
    for im in man[0]["images"]:
        _mk(tmp, im["path"])
    keepers = A._curate(man, tmp, content_level="T3_artnude", keep_top=1,
                        use_hps_v2=False, use_image_reward=False)
    kept = [im for im in man[0]["images"] if im["keeper"]]
    assert len(kept) == 1 and kept[0]["seed"] == 1, "should keep the top-scored"
    # the multiple_faces image is never a keeper
    flagged = next(im for im in man[0]["images"] if im["seed"] == 99)
    assert flagged["keeper"] is False
    assert len(keepers) == 1
    assert (tmp / "keepers").exists()


def test_curate_graceful_when_scoring_unavailable(tmp, monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("weights missing")
    monkeypatch.setattr("src.scoring.image_scorer.ImageScorer", _boom)
    man = [{"index": 0, "prompt": "p", "images": [
        {"path": "images/ad01_a_s1.png", "seed": 1},
        {"path": "images/ad01_a_s2.png", "seed": 2}]}]
    for im in man[0]["images"]:
        _mk(tmp, im["path"])
    keepers = A._curate(man, tmp, content_level="T3_artnude", keep_top=1,
                        use_hps_v2=False, use_image_reward=False)
    # scoring down → keep ALL (never drop a run's output on a curation failure)
    assert all(im["keeper"] for im in man[0]["images"])
    assert len(keepers) == 2


def test_curate_contact_sheet_name_no_clobber(tmp, monkeypatch):
    """The covers curation pass must not overwrite the main keepers'
    contact_sheet.png — each pass writes its own named sheet."""
    monkeypatch.setattr("src.scoring.image_scorer.ImageScorer", _FakeScorer)
    written: list[str] = []

    def _rec(cs, out_path):
        written.append(Path(out_path).name)
        Path(out_path).write_bytes(b"PNG")
    monkeypatch.setattr("src.review.contact_sheet.create_contact_sheet", _rec)

    main = [{"index": 0, "prompt": "p", "images": [
        {"path": "images/ad01_a_s1.png", "seed": 1}]}]
    covers = [{"index": 0, "prompt": "c", "images": [
        {"path": "covers/cover01_a_s1.png", "seed": 1}]}]
    for m in (main, covers):
        for e in m:
            for im in e["images"]:
                _mk(tmp, im["path"])

    A._curate(main, tmp, content_level="T3_artnude", keep_top=1,
              use_hps_v2=False, use_image_reward=False)
    A._curate(covers, tmp, content_level="T1_suggestive", keep_top=1,
              use_hps_v2=False, use_image_reward=False,
              contact_sheet_name="contact_sheet_covers.png")

    assert written == ["contact_sheet.png", "contact_sheet_covers.png"]
    assert (tmp / "contact_sheet.png").exists()
    assert (tmp / "contact_sheet_covers.png").exists()


def test_creative_direction_house_style(monkeypatch):
    """The config-driven creative-direction layer injects the house style into
    the system prompt + samples a VARIED look per image, stays adult-safe, and
    degrades gracefully when the config is absent."""
    block = AD._creative_system_block()
    assert "CREATIVE DIRECTION" in block
    assert "DOMINATE" in block.upper()                 # subject-focus fix
    assert "young adult" in block.lower()              # the look lean
    assert "ADULT only" in block                       # age-safety reinforcement
    assert "VARIED" in block.upper()                   # variety preserved
    # injected into the assembled system prompt
    assert "CREATIVE DIRECTION" in AD._build_system_prompt()
    # per-image look rotation is varied (not clones)
    looks = [AD._creative_look(i) for i in range(6)]
    assert len(set(looks)) >= 5, f"look rotation should vary, got {looks}"
    # graceful: no config -> no block, prompt engine unchanged
    monkeypatch.setattr(AD, "_CREATIVE", {})
    assert AD._creative_system_block() == ""
    assert AD._creative_look(0) == ""
    assert "CREATIVE DIRECTION" not in AD._build_system_prompt()


# ── tier-split packaging (_package) — the SFW-cover HARD rule ───────

def _selection(tier, niche="fine_art_figure_study"):
    return build_selection(NicheLibrary.from_yaml(), 0, tier=tier,
                           force_niche=niche)


_META = {"title": "T", "description": "D", "tags": ["fineartnude", "aiart"],
         "labels": {"ai_tools": A.AI_DISCLOSURE_LABEL, "mature": True}}
_WM_OFF = {"enabled": False, "text": "x", "position": "bottom-right", "opacity": 0.3}


def test_package_explicit_never_puts_explicit_in_public(tmp):
    """HARD rule: for a T4 run, public/ + cover come ONLY from SFW cover
    renders; explicit keepers go to gated/ and NEVER to public/."""
    main = [{"index": 0, "prompt": "explicit", "images": [
        {"path": "images/ad01_a_s1.png", "seed": 1, "keeper": True},
        {"path": "images/ad01_a_s2.png", "seed": 2, "keeper": False}]}]
    covers = [{"index": 0, "prompt": "sfw", "images": [
        {"path": "covers/cover01_a_s9.png", "seed": 9, "keeper": True}]}]
    for e in main + covers:
        for im in e["images"]:
            _mk(tmp, im["path"])
    pkg = A._package(tmp, _selection("T4_explicit"), "T4_explicit",
                     main, covers, _META, _META, _WM_OFF)
    public = [p.name for p in (pkg / "public").glob("*.png")]
    gated = [p.name for p in (pkg / "gated").glob("*.png")]
    meta = json.loads((pkg / "metadata.json").read_text())
    assert public == ["cover01_a_s9.png"]
    assert all(not n.startswith("ad") for n in public), "explicit leaked to public!"
    assert "ad01_a_s1.png" in gated
    assert (meta["cover_image"] or "").startswith("cover")
    assert meta["labels"]["ai_tools"] == A.AI_DISCLOSURE_LABEL
    assert meta["labels"]["mature"] is True
    assert (pkg / "POSTING_CHECKLIST.md").exists()


def test_package_sfw_tier_routes_keepers_to_public(tmp):
    main = [{"index": 0, "prompt": "implied", "images": [
        {"path": "images/ad01_a_s1.png", "seed": 1, "keeper": True}]}]
    for im in main[0]["images"]:
        _mk(tmp, im["path"])
    pkg = A._package(tmp, _selection("T2_implied", "pinup_1950s"), "T2_implied",
                     main, [], _META, _META, _WM_OFF)
    public = [p.name for p in (pkg / "public").glob("*.png")]
    gated = [p.name for p in (pkg / "gated").glob("*.png")]
    assert public == ["ad01_a_s1.png"]   # SFW tier → keepers are public-safe
    assert gated == []                    # nothing gated at T2


def test_posting_checklist_carries_hard_rules(tmp):
    main = [{"index": 0, "prompt": "x", "images": [
        {"path": "images/ad01_a_s1.png", "seed": 1, "keeper": True}]}]
    covers = [{"index": 0, "prompt": "c", "images": [
        {"path": "covers/cover01_a_s9.png", "seed": 9, "keeper": True}]}]
    for e in main + covers:
        for im in e["images"]:
            _mk(tmp, im["path"])
    pkg = A._package(tmp, _selection("T4_explicit"), "T4_explicit",
                     main, covers, _META, _META, _WM_OFF)
    txt = (pkg / "POSTING_CHECKLIST.md").read_text()
    assert "Created using AI tools" in txt
    assert "Mature Content" in txt
    assert "MUST be SFW" in txt
    assert "Groups" in txt          # reach-multiplier guidance
    assert "Fanvue" in txt          # funnel option for the gated set


def test_package_emits_per_image_posting_templates(tmp):
    """Each packaged run emits copy-paste posting templates (TITLE/FOLDER/TAGS/
    GROUPS/MATURE/AI/PRICE) per image so manual DA upload is trivial."""
    main = [{"index": 0, "prompt": "x", "images": [
        {"path": "images/ad01_a_s1.png", "seed": 1, "keeper": True}]}]
    covers = [{"index": 0, "prompt": "c", "images": [
        {"path": "covers/cover01_a_s9.png", "seed": 9, "keeper": True}]}]
    for e in main + covers:
        for im in e["images"]:
            _mk(tmp, im["path"])
    pkg = A._package(tmp, _selection("T3_artnude", "fine_art_figure_study"),
                     "T3_artnude", main, covers, _META, _META, _WM_OFF)
    tdir = pkg / "posting_templates"
    files = sorted(tdir.glob("*.txt"))
    assert len(files) >= 2   # one per public + gated image
    body = files[0].read_text()
    for key in ("TITLE:", "FOLDER:", "TAGS:", "GROUPS:", "MATURE:", "AI-LABEL:", "PRICE:"):
        assert key in body, key
    meta = json.loads((pkg / "metadata.json").read_text())
    assert meta["price_by_tier"] and meta["da_groups"]


# ── art_director inline audit gate ─────────────────────────────────

def test_audit_gate_regenerates_below_threshold(monkeypatch):
    """generate_series should regenerate while the audit score is below
    threshold and keep the first prompt that clears it."""
    calls = {"n": 0}

    def fake_generate_one(client, **kw):
        calls["n"] += 1
        text = "LOW" if calls["n"] == 1 else "HIGH"
        return {"prompt": text, "orientation": "portrait", "shot_type": "medium",
                "framing_rationale": ""}

    def fake_score(text, tier):
        return (5.0, ["LOW_ISSUE"]) if text == "LOW" else (9.0, [])

    monkeypatch.setattr(AD, "generate_one", fake_generate_one)
    monkeypatch.setattr(AD, "OllamaClient", lambda *a, **k: object())
    monkeypatch.setattr("scripts.audit_prompts.score_prompt", fake_score)

    rows = AD.generate_series(brief="b", tier="T3_artnude", count=1,
                              model_tag="m", temperature=0.8,
                              audit_threshold=7.5, max_attempts=4)
    assert calls["n"] == 2, "should have regenerated past the LOW prompt"
    assert rows[0]["prompt"] == "HIGH"
    assert rows[0]["audit_score"] == 9.0


def test_framing_fields_threaded_into_rows(monkeypatch):
    """generate_series carries the LLM's per-prompt orientation/shot_type/
    framing_rationale (Phase 1) into the manifest rows so the renderer can
    pick the aspect ratio per image (kills the only-portrait problem)."""
    seq = iter([
        {"prompt": "p " * 80, "orientation": "landscape", "shot_type": "full_body",
         "framing_rationale": "reclining body suits a wide frame"},
        {"prompt": "q " * 80, "orientation": "square", "shot_type": "close_up",
         "framing_rationale": "graphic centred face"},
    ])
    monkeypatch.setattr(AD, "generate_one", lambda client, **kw: next(seq))
    monkeypatch.setattr(AD, "OllamaClient", lambda *a, **k: object())
    rows = AD.generate_series(brief="b", tier="T3_artnude", count=2,
                              model_tag="m", temperature=0.8, audit_gate=False)
    assert rows[0]["orientation"] == "landscape" and rows[0]["shot_type"] == "full_body"
    assert rows[1]["orientation"] == "square" and rows[1]["shot_type"] == "close_up"
    assert "wide frame" in rows[0]["framing_rationale"]


def test_generate_series_seeds_avoid_and_offsets_rotation(monkeypatch):
    """Cross-series variety wiring: generate_series must (1) seed the
    anti-repetition lists from seed_avoid/seed_banned_openers (so the FIRST
    scene already avoids prior-series prompts) and (2) offset the sub-look /
    framing / look rotation by run_offset (so the SEQUENCE differs per run)."""
    calls: list[dict] = []

    def rec(client, **kw):
        # snapshot the mutable lists at call time (generate_series appends to them)
        calls.append({"avoid": list(kw["avoid"]),
                      "banned_openers": list(kw["banned_openers"]),
                      "sub_look": kw["sub_look"],
                      "framing_target": kw["framing_target"],
                      "look_target": kw["look_target"]})
        return {"prompt": f"scene {len(calls)} " + "word " * 80,
                "orientation": "portrait", "shot_type": "medium",
                "framing_rationale": "r"}

    monkeypatch.setattr(AD, "generate_one", rec)
    looks = ["a — x", "b — y", "c — z", "d — w"]
    rows = AD.generate_series(
        brief="b", tier="T3_artnude", count=3, model_tag="m", temperature=0.8,
        sub_looks=looks, audit_gate=False, client=object(),
        seed_avoid=["PRIOR SIG …"], seed_banned_openers=["prior opener words"],
        run_offset=2,
    )
    assert len(rows) == 3
    # (1) the first scene sees ONLY the seeded history (not empty)
    assert calls[0]["avoid"] == ["PRIOR SIG …"]
    assert calls[0]["banned_openers"] == ["prior opener words"]
    # the seed persists and the lists grow within-series as scenes commit
    assert calls[1]["avoid"][0] == "PRIOR SIG …" and len(calls[1]["avoid"]) == 2
    # (2) run_offset=2 rotates the sub-look sequence: looks[(i+2)%4]
    assert [c["sub_look"] for c in calls] == [looks[2], looks[3], looks[0]]
    # framing + look rotations are offset by the same amount
    assert [c["framing_target"] for c in calls] == \
        [AD.FRAMING_TARGETS[(i + 2) % len(AD.FRAMING_TARGETS)] for i in range(3)]
    assert [c["look_target"] for c in calls] == [AD._creative_look(i + 2) for i in range(3)]


def test_load_niche_history_reads_prior_same_niche_manifests(tmp_path, monkeypatch):
    """art_series._load_niche_history mines prior same-niche manifests for the
    avoid/banned seeds + the prior-run count, ignoring other niches and
    malformed files (no DB — the manifests ARE the prompt store)."""
    base = tmp_path / "output" / "art_series"
    def _mf(name: str, niche: str, prompts: list[str]):
        d = base / name
        d.mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps(
            {"niche": {"id": niche},
             "prompts": [{"prompt": p} for p in prompts]}))
    _mf("20260101_000000", "bohemian_naturallight", ["Warm honey light over X " * 6])
    _mf("20260102_000000", "bohemian_naturallight", ["Cool blue dusk over Y " * 6,
                                                      "Soft grey dawn over Z " * 6])
    _mf("20260103_000000", "goth_romantic", ["Candlelit gothic chamber W " * 6])
    (base / "broken").mkdir(parents=True)
    (base / "broken" / "manifest.json").write_text("{ not json")

    monkeypatch.setattr(A, "ROOT", tmp_path)
    banned, avoid, count = A._load_niche_history("bohemian_naturallight")
    assert count == 2                       # two prior bohemian runs (goth excluded)
    assert len(banned) == 3 and len(avoid) == 3   # 1 + 2 prompts across the 2 runs
    # the seeds are derived via the shared art_director helpers
    assert banned[0] == AD._opener("Warm honey light over X " * 6)
    assert avoid[0] == AD._signature("Warm honey light over X " * 6)
    # a never-run niche → empty seeds + zero offset (graceful)
    assert A._load_niche_history("does_not_exist_niche") == ([], [], 0)
    # recent_series cap limits how many prior runs feed the seeds
    b2, a2, c2 = A._load_niche_history("bohemian_naturallight", recent_series=1)
    assert c2 == 2 and len(b2) == 2         # count = all runs; seeds = newest run only


def test_gen_metadata_routes_by_backend_not_hardcoded_ollama(monkeypatch):
    """Regression: _gen_metadata must drive the LLM via LLMClientPool (which
    routes the tag to its backend — LM Studio / Ollama / MLX), NOT a hardcoded
    OllamaClient. The hardcoded client 404'd whenever model_tag was an LM Studio
    tag (the default Gemma), silently dropping the bespoke set title/description
    back to the niche stub."""
    captured: dict = {}
    import src.agents.metadata_generator as MG

    class _StubGen:
        def __init__(self, client):
            captured["client"] = client

        def generate(self, **kw):
            captured["model"] = kw.get("model")
            return {"title": "T", "description": "D", "tags": ["boho"]}

    monkeypatch.setattr(MG, "MetadataGenerator", _StubGen)
    meta = A._gen_metadata(None, "a boho brief", "T3_artnude", 6,
                           "gemma-4-26b-a4b-it-ultra-uncensored-heretic")
    # the client is the backend-routing pool, not a bare OllamaClient
    assert type(captured["client"]).__name__ == "LLMClientPool"
    assert captured["model"] == "gemma-4-26b-a4b-it-ultra-uncensored-heretic"
    # and the post-processing still merges discovery tags + mandatory labels
    assert "aiart" in meta["tags"]
    assert meta["labels"]["ai_tools"] and meta["labels"]["mature"] is True


def test_grounding_gate_rejects_subject_on_water(monkeypatch):
    """_PromptOut hard-rejects the subject rendered sitting/kneeling/floating ON
    water or in mid-air (the 'sitting on water' failure → a body hovering on
    nothing), but passes a body grounded on a solid bank with water beside her.
    High precision: 'at the water's edge' / 'light on the water' must NOT trip."""
    monkeypatch.setattr(AD, "_ACTIVE_WORD_BAND", (110, 160))
    on_water = ("A low amber sun catches the river reeds as a sun-kissed woman "
                "kneels on the water, her wet skin glistening, gaze soft and "
                "direct, the lake shimmering in a warm golden haze, 35mm film. ") * 2
    grounded = ("A low amber sun catches the river reeds as a sun-kissed woman "
                "kneels on the mossy bank at the water's edge, the river beside "
                "her, her wet skin glistening, gaze soft and direct, 35mm film. ") * 2
    with pytest.raises(Exception):
        AD._PromptOut(prompt=on_water)
    assert AD._PromptOut(prompt=grounded).prompt          # solid bank passes
    # precision: innocent water mentions (no pose verb) are fine
    assert not AD._IMPLAUSIBLE_GROUNDING_RE.search("warm light dances on the water")
    assert not AD._IMPLAUSIBLE_GROUNDING_RE.search("she sits on the rock by the water")
    assert AD._IMPLAUSIBLE_GROUNDING_RE.search("she floats on the calm lake")


def test_audit_flags_implausible_grounding():
    """The audit gate flags on-water / mid-air / submerged grounding with an
    IMPLAUSIBLE_GROUNDING penalty so borderline prompts re-roll; clean grounding
    scores normally."""
    from scripts.audit_prompts import detect_implausible_grounding, score_prompt
    assert detect_implausible_grounding("she kneels on the water")
    assert detect_implausible_grounding("she lies partially submerged in the eddy")
    assert not detect_implausible_grounding("she kneels on the mossy bank at the water's edge")
    bad = ("A bright midday sun fractures through the reeds as a lithe woman "
           "kneels on the water, her wet skin glistening, gaze soft and direct, "
           "the lake shimmering in a warm haze, shot on 35mm film with bokeh. ") * 2
    score, issues = score_prompt(bad, "T3_artnude")
    assert any("IMPLAUSIBLE_GROUNDING" in i for i in issues)


def _fake_render_env(tmp_path):
    """A builder + client that 'render' by copying a tiny real PNG, for
    _render_stage_base anatomy-guard tests."""
    from PIL import Image
    src = tmp_path / "src.png"
    Image.new("RGB", (8, 8)).save(src)

    class _Img:
        type = "output"
        file_path = str(src)

    class _Client:
        def render_single_with_retry(self, wf, timeout=0):
            return [_Img()]

    class _Builder:
        def build_external(self, **kw):
            return {}

    return _Builder(), _Client()


def test_base_anatomy_retry_rerolls_until_clean(tmp_path, monkeypatch):
    """The extra-limb guard rerolls a base render (new seed) when the hand
    detector reports >2 hands, keeps the first clean one, and deletes the
    rejected file."""
    builder, client = _fake_render_env(tmp_path)
    monkeypatch.setattr(A, "_hand_detector", lambda p: "DET")
    counts = iter([3, 2])               # defective → reroll → clean
    monkeypatch.setattr(A, "_count_hands", lambda det, path, conf=0.5: next(counts))
    rows = [{"look": "riverbank scene", "prompt": "p", "orientation": "portrait"}]
    manifest = A._render_stage_base(
        rows, builder=builder, client=client, base_template="t", negative="n",
        resolution=(896, 1152), base_seed=100, seeds=1,
        dest_dir=tmp_path / "base", out_dir=tmp_path, prefix="ad",
        anatomy_retries=2, hand_detector_path="x")
    imgs = manifest[0]["images"]
    assert len(imgs) == 1
    assert imgs[0]["seed"] == 101                       # the reroll seed, not the defective 100
    # only the clean render survives on disk (defective seed-100 file deleted)
    assert sorted(p.name for p in (tmp_path / "base").glob("*.png")) == \
        ["ad01_riverbank_s101.png"]


def test_base_anatomy_retry_keeps_least_bad_when_all_fail(tmp_path, monkeypatch):
    """If every reroll still has an extra limb, keep the render with the FEWEST
    hands (never drop the image — it's flagged for manual culling)."""
    builder, client = _fake_render_env(tmp_path)
    monkeypatch.setattr(A, "_hand_detector", lambda p: "DET")
    counts = iter([4, 3, 5])            # all defective; fewest = 3 (the 2nd, seed 101)
    monkeypatch.setattr(A, "_count_hands", lambda det, path, conf=0.5: next(counts))
    rows = [{"look": "tent scene", "prompt": "p", "orientation": "portrait"}]
    manifest = A._render_stage_base(
        rows, builder=builder, client=client, base_template="t", negative="n",
        resolution=(896, 1152), base_seed=100, seeds=1,
        dest_dir=tmp_path / "base", out_dir=tmp_path, prefix="ad",
        anatomy_retries=2, hand_detector_path="x")
    assert manifest[0]["images"][0]["seed"] == 101     # fewest-hands render kept


def test_base_anatomy_guard_off_single_render(tmp_path, monkeypatch):
    """anatomy_retries=0 → no detector loaded, exactly one render at base_seed
    (back-compat: the guard is opt-out)."""
    builder, client = _fake_render_env(tmp_path)
    loaded = []
    monkeypatch.setattr(A, "_hand_detector", lambda p: loaded.append(p) or "DET")
    rows = [{"look": "loft scene", "prompt": "p", "orientation": "portrait"}]
    manifest = A._render_stage_base(
        rows, builder=builder, client=client, base_template="t", negative="n",
        resolution=(896, 1152), base_seed=100, seeds=1,
        dest_dir=tmp_path / "base", out_dir=tmp_path, prefix="ad",
        anatomy_retries=0, hand_detector_path="x")
    assert manifest[0]["images"][0]["seed"] == 100
    assert loaded == []                                # detector never loaded when off


@pytest.mark.parametrize("tier,expected", [
    ("T4_explicit", "refine_T4.json"),     # explicit main → vagina-detailer variant
    ("T3_artnude", "refine.json"),         # tasteful nude → base refine (tier purity)
    ("T2_implied", "refine.json"),
    ("T1_suggestive", "refine.json"),
])
def test_select_refine_template_is_tier_pure(tier, expected):
    rp = {"refine_template": "templates/chroma/refine.json",
          "refine_template_t4": "templates/chroma/refine_T4.json"}
    assert A._select_refine_template(tier, rp).endswith(expected)


def test_select_refine_template_falls_back_without_t4_key():
    rp = {"refine_template": "templates/chroma/refine.json"}   # no _t4 configured
    assert A._select_refine_template("T4_explicit", rp).endswith("refine.json")


def test_genital_detailer_detection_drives_tier_purity_guard():
    """Content-based (not filename) tier-purity signal: refine_T4 has a vagina
    detailer, the base refine does not — so the staged guard aborts a sub-T4
    render only when a T4 template leaks in (e.g. via --refine-template)."""
    wd = Path("config/comfyui_workflows")
    assert A._template_has_genital_detailer(wd, "templates/chroma/refine_T4.json") is True
    assert A._template_has_genital_detailer(wd, "templates/chroma/refine.json") is False
    assert A._template_has_genital_detailer(wd, None) is False              # no template
    assert A._template_has_genital_detailer(wd, "templates/chroma/nope.json") is False  # missing


def test_promptout_framing_validators_tolerant():
    P = "A warm shaft of light falls across a woman in a quiet room. " * 8
    # synonyms coerce, junk falls back, omission defaults
    assert AD._PromptOut(prompt=P, orientation="LANDSCAPE ", shot_type="closeup").shot_type == "close_up"
    bad = AD._PromptOut(prompt=P, orientation="weird", shot_type="nonsense")
    assert (bad.orientation, bad.shot_type) == ("portrait", "medium")
    assert AD._PromptOut(prompt=P).orientation == "portrait"


def test_sfw_cover_gate_rejects_nudity(monkeypatch):
    """require_sfw=True must hard-reject any nudity in a cover prompt
    (DA SFW-shopfront ToS) so the LLM re-rolls clothed. The verification
    e2e caught a T1 cover drifting to a topless art-nude silhouette."""
    nude = ("She stands bare and topless against black, her exposed breasts "
            "catching the rim light, nude form sculpted in shadow. " * 4)
    clothed = ("She stands in an elegant floor-length silk gown that drapes "
               "her fully, a confident hand on one hip, warm light across the "
               "fabric and her serene face, fine-art editorial framing, 85mm. " * 3)
    # validator runs under the active SFW flag
    monkeypatch.setattr(AD, "_ACTIVE_REQUIRE_SFW", True)
    with pytest.raises(Exception):
        AD._PromptOut(prompt=nude)
    # a fully-clothed prompt passes
    assert AD._PromptOut(prompt=clothed).prompt
    # and with the flag off, nudity is allowed (normal T3/T4 path)
    monkeypatch.setattr(AD, "_ACTIVE_REQUIRE_SFW", False)
    assert AD._PromptOut(prompt=nude).prompt


def test_mood_gate_rejects_sad_affect():
    """The mood gate hard-rejects unambiguous sad-affect words (commercial
    NSFW sells confident sensuality, not sorrow) at the WORD level — 'sad'
    must not fire inside 'saddle', nor a stem inside 'sober'. Caught a Vermeer
    prompt that drifted to 'melancholic composure'."""
    base = "A warm shaft of light falls across a confident woman in a quiet room. " * 6
    for sadword in ("melancholic", "a deep sorrow", "mournful and grieving",
                    "softly weeping", "a forlorn, tearful"):
        with pytest.raises(Exception):
            AD._PromptOut(prompt=base + sadword + " she gazes away.")
    # serene / introspective prose passes
    assert AD._PromptOut(prompt=base + "serene and introspective, she gazes ahead.").prompt
    # look-alikes must NOT false-positive (word-level matching)
    assert AD._PromptOut(prompt=base + "seated on a worn leather saddle, sober and composed.").prompt


def test_audit_gate_keeps_best_when_all_below(monkeypatch):
    """If no attempt clears the threshold, ship the best-scoring one
    (never drop the scene over a soft-quality miss)."""
    scores = iter([4.0, 6.5, 5.0, 3.0])

    monkeypatch.setattr(AD, "generate_one", lambda client, **kw: {
        "prompt": "P", "orientation": "portrait", "shot_type": "medium",
        "framing_rationale": ""})
    monkeypatch.setattr(AD, "OllamaClient", lambda *a, **k: object())
    monkeypatch.setattr("scripts.audit_prompts.score_prompt",
                        lambda t, tier: (next(scores), []))

    rows = AD.generate_series(brief="b", tier="T3_artnude", count=1,
                              model_tag="m", temperature=0.8,
                              audit_threshold=7.5, max_attempts=4)
    assert len(rows) == 1
    assert rows[0]["audit_score"] == 6.5    # the best of the 4 attempts
