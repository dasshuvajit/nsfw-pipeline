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
    # subject-dominance + photoreal LOOK live ONLY in the system prompt since
    # the 2026-06-10 dedup (the YAML used to duplicate them verbatim and the
    # LLM received both copies); the YAML keeps the tunable knobs.
    assert "DOMINATES" in AD.ART_DIRECTOR_SYSTEM_PROMPT
    assert "DOMINATE" not in block.upper()             # dedup held
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


def test_package_persona_numbers_title_and_metadata(tmp, monkeypatch):
    """A bound persona stamps a body-of-work serial into metadata and numbers the
    posting-template title ('Clara — <title> No. 01'). Serial files are redirected
    to tmp so the test is deterministic and doesn't touch real run-state."""
    monkeypatch.setattr(A, "PERSONA_SERIALS_FILE", tmp / ".persona_serials")
    monkeypatch.setattr(A, "FAMILY_SERIALS_FILE", tmp / ".family_serials")
    sel = build_selection(NicheLibrary.from_yaml(), 0, tier="T3_artnude",
                          force_niche="old_hollywood_glamour",
                          persona=True, persona_name="Clara")
    main = [{"index": 0, "prompt": "x", "images": [
        {"path": "images/ad01_a_s1.png", "seed": 1, "keeper": True}]}]
    covers = [{"index": 0, "prompt": "c", "images": [
        {"path": "covers/cover01_a_s9.png", "seed": 9, "keeper": True}]}]
    for e in main + covers:
        for im in e["images"]:
            _mk(tmp, im["path"])
    pkg = A._package(tmp, sel, "T3_artnude", main, covers, _META, _META, _WM_OFF)
    meta = json.loads((pkg / "metadata.json").read_text())
    assert meta["persona"] == "Clara"
    assert meta["persona_serial"] == 1          # first appearance
    titles = " ".join(f.read_text() for f in (pkg / "posting_templates").glob("*.txt"))
    assert "Clara — T No. 01" in titles         # numbered persona body-of-work title


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
        n = len(calls)
        filler = " ".join(chr(110 + n % 12) + chr(97 + j % 26) + chr(97 + (j // 26) % 26)
                          for j in range(80))
        return {"prompt": f"scene {filler}",
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
    # (each accept adds a head signature AND a tail signature since 2026-06)
    assert calls[1]["avoid"][0] == "PRIOR SIG …" and len(calls[1]["avoid"]) == 3
    # (2) run_offset=2 rotates the sub-look sequence: looks[(i+2)%4]
    assert [c["sub_look"] for c in calls] == [looks[2], looks[3], looks[0]]
    # framing + look rotations are offset by the same amount
    assert [c["framing_target"] for c in calls] == \
        [AD.FRAMING_TARGETS[(i + 2) % len(AD.FRAMING_TARGETS)] for i in range(3)]
    assert [c["look_target"] for c in calls] == [AD._creative_look(i, 2) for i in range(3)]


def test_generate_series_locked_look_overrides_rotation(monkeypatch):
    """A bound persona (locked_look) makes EVERY image the same woman: look_target
    is the locked string for all scenes and look_locked is True — while sub_look /
    framing STILL rotate by run_offset (scene/pose vary, identity does not)."""
    calls: list[dict] = []

    def rec(client, **kw):
        calls.append({"sub_look": kw["sub_look"],
                      "framing_target": kw["framing_target"],
                      "look_target": kw["look_target"],
                      "look_locked": kw.get("look_locked")})
        n = len(calls)
        filler = " ".join(chr(110 + n % 12) + chr(97 + j % 26) + chr(97 + (j // 26) % 26)
                          for j in range(80))
        return {"prompt": f"scene {filler}",
                "orientation": "portrait", "shot_type": "medium",
                "framing_rationale": "r"}

    monkeypatch.setattr(AD, "generate_one", rec)
    looks = ["a — x", "b — y", "c — z", "d — w"]
    rows = AD.generate_series(
        brief="b", tier="T3_artnude", count=3, model_tag="m", temperature=0.8,
        sub_looks=looks, audit_gate=False, client=object(),
        run_offset=2, locked_look="LOCKED-CLARA-LOOK",
    )
    assert len(rows) == 3
    # identity LOCKED — same look every image, flagged locked
    assert [c["look_target"] for c in calls] == ["LOCKED-CLARA-LOOK"] * 3
    assert all(c["look_locked"] is True for c in calls)
    # scene/pose STILL vary — sub_look + framing rotate by run_offset as usual
    assert [c["sub_look"] for c in calls] == [looks[2], looks[3], looks[0]]
    assert [c["framing_target"] for c in calls] == \
        [AD.FRAMING_TARGETS[(i + 2) % len(AD.FRAMING_TARGETS)] for i in range(3)]


def test_next_persona_serial_increments_and_persists(tmp_path, monkeypatch):
    """Per-persona appearance serial: 1-based, persists, independent per persona,
    case-insensitive key (matches select_persona's lookup)."""
    monkeypatch.setattr(A, "PERSONA_SERIALS_FILE", tmp_path / ".persona_serials")
    assert A._next_persona_serial("Clara") == 1
    assert A._next_persona_serial("Clara") == 2
    assert A._next_persona_serial("Sable") == 1          # independent counter
    assert A._next_persona_serial("clara") == 3          # case-insensitive
    # persisted to disk
    data = json.loads((tmp_path / ".persona_serials").read_text())
    assert data == {"clara": 3, "sable": 1}


def test_comfyui_free_posts_unload_and_free(monkeypatch):
    """Pre-flight ComfyUI free: POSTs to /free with unload_models + free_memory so
    each series releases ComfyUI's ~32GB before the Phase-1 LLM load (OOM fix)."""
    captured = {}

    class _Resp:
        status_code = 200

    def _fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return _Resp()

    import requests
    monkeypatch.setattr(requests, "post", _fake_post)
    assert A._comfyui_free("http://127.0.0.1:8188/") is True
    assert captured["url"] == "http://127.0.0.1:8188/free"   # trailing slash stripped
    assert captured["json"] == {"unload_models": True, "free_memory": True}


def test_comfyui_free_never_raises(monkeypatch):
    """A free failure must NOT abort the run — best-effort, returns False."""
    import requests

    def _boom(*a, **k):
        raise ConnectionError("ComfyUI dropped")

    monkeypatch.setattr(requests, "post", _boom)
    assert A._comfyui_free("http://127.0.0.1:8188") is False


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
    banned, avoid, count, overused = A._load_niche_history("bohemian_naturallight")
    assert count == 2                       # two prior bohemian runs (goth excluded)
    # 3 prompts across the 2 runs: 1 opener each; head + TAIL signature each
    assert len(banned) == 3 and len(avoid) == 6
    # the seeds are derived via the shared art_director helpers
    assert banned[0] == AD._opener("Warm honey light over X " * 6)
    assert avoid[0] == AD._signature("Warm honey light over X " * 6)
    assert avoid[1] == AD._tail_signature("Warm honey light over X " * 6)
    # a never-run niche → empty seeds + zero offset (graceful)
    assert A._load_niche_history("does_not_exist_niche") == ([], [], 0, [])
    # recent_series cap limits how many prior runs feed the seeds
    b2, a2, c2, _ = A._load_niche_history("bohemian_naturallight", recent_series=1)
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


@pytest.mark.parametrize("tier,main_ext,cover_ext", [
    ("T4_explicit", "refine_T4.json", "refine.json"),   # covers stay SFW even on a T4 run
    ("T3_artnude", "refine.json", "refine.json"),
    ("T2_implied", "refine.json", "refine.json"),
    ("T1_suggestive", "refine.json", "refine.json"),
])
def test_refine_templates_for_keeps_covers_sfw(tier, main_ext, cover_ext):
    """The whole staged routing decision: MAIN follows the tier, COVERS ALWAYS use
    the base refine — so a public/teaser image is never genital-detailed, even when
    the main set is T4 explicit."""
    rp = {"refine_template": "templates/chroma/refine.json",
          "refine_template_t4": "templates/chroma/refine_T4.json"}
    main, cover = A._refine_templates_for(tier, rp)
    assert main.endswith(main_ext)
    assert cover.endswith(cover_ext)
    # the cover template never carries a genital detailer, at any tier
    assert A._template_has_genital_detailer(Path("config/comfyui_workflows"), cover) is False


def test_tier_purity_guard_aborts_on_refine_template_override_leak():
    """The staged guard: a --refine-template override that injects the T4 template
    into a sub-T4 tier is caught (would abort); the same template at T4 is allowed;
    the normal base refine at any tier passes."""
    wd = Path("config/comfyui_workflows")
    leaked = {"refine_template": "templates/chroma/refine_T4.json",   # CLI override leak
              "refine_template_t4": "templates/chroma/refine_T4.json"}
    main_t3, _ = A._refine_templates_for("T3_artnude", leaked)
    assert A._violates_tier_purity("T3_artnude", main_t3, wd) is True       # ABORT
    main_t4, _ = A._refine_templates_for("T4_explicit", leaked)
    assert A._violates_tier_purity("T4_explicit", main_t4, wd) is False     # allowed at T4
    # the normal (un-overridden) base refine never violates purity, at any tier
    normal = {"refine_template": "templates/chroma/refine.json",
              "refine_template_t4": "templates/chroma/refine_T4.json"}
    for tier in ("T1_suggestive", "T3_artnude", "T4_explicit"):
        m, _ = A._refine_templates_for(tier, normal)
        assert A._violates_tier_purity(tier, m, wd) is False


def test_genital_detailer_detection_drives_tier_purity_guard():
    """Content-based (not filename) tier-purity signal: refine_T4 has a vagina
    detailer, the base refine does not — so the staged guard aborts a sub-T4
    render only when a T4 template leaks in (e.g. via --refine-template)."""
    wd = Path("config/comfyui_workflows")
    assert A._template_has_genital_detailer(wd, "templates/chroma/refine_T4.json") is True
    assert A._template_has_genital_detailer(wd, "templates/chroma/refine.json") is False
    assert A._template_has_genital_detailer(wd, None) is False              # no template
    assert A._template_has_genital_detailer(wd, "templates/chroma/nope.json") is False  # missing


def test_t4_explicit_reveal_rotation_only_at_t4(monkeypatch):
    """T4_explicit gets a rotated EXPLICIT REVEAL STYLE + grooming per image (so
    the set spans many tasteful reveals, not one centred splay); T3 and below get
    NONE (no vulva shown → nothing to rotate). Distance-bound styles pin shot_type."""
    seen = []

    def rec(client, **kw):
        seen.append({"reveal": kw.get("reveal_target"), "grooming": kw.get("grooming"),
                     "framing": kw.get("framing_target")})
        n = len(seen)
        # distinct ALPHA-ONLY filler per call — the similarity rejector
        # (2026-06) refuses candidates that 3-gram-match an accepted prompt,
        # and its tokenizer drops digits
        filler = " ".join(chr(97 + n % 26) + chr(97 + j % 26) + chr(97 + (j // 26) % 26)
                          for j in range(80))
        return {"prompt": filler, "orientation": "portrait",
                "shot_type": "medium", "framing_rationale": "r"}

    monkeypatch.setattr(AD, "generate_one", rec)
    AD.generate_series(brief="b", tier="T4_explicit", count=6, model_tag="m",
                       temperature=0.8, audit_gate=False, client=object())
    reveals = [s["reveal"] for s in seen]
    assert all(r is not None for r in reveals)                       # every T4 image
    assert [r[0] for r in reveals] == [AD.REVEAL_STYLES[i][0] for i in range(6)]  # rotates
    assert all(s["grooming"] for s in seen)                          # grooming assigned
    for s in seen:                                                   # pins honoured
        pin = AD.REVEAL_SHOT_PIN.get(s["reveal"][0])
        if pin:
            assert s["framing"][1] == pin
    seen.clear()
    AD.generate_series(brief="b", tier="T3_artnude", count=4, model_tag="m",
                       temperature=0.8, audit_gate=False, client=object())
    assert all(s["reveal"] is None and not s["grooming"] for s in seen)  # T3 untouched


def test_generate_one_weaves_reveal_style_into_prompt():
    """generate_one injects the assigned REVEAL STYLE + grooming into the user
    prompt (the per-image artistic-explicit nudge)."""
    captured = {}

    class _Client:
        def generate_json(self, system, user, **kw):
            captured["user"] = user
            return {"prompt": "p " * 80, "orientation": "portrait",
                    "shot_type": "medium", "framing_rationale": "r"}

    AD.generate_one(_Client(), brief="b", tier="T4_explicit", sub_look="x — y",
                    avoid=[], banned_openers=[], model_tag="m", temperature=0.8,
                    reveal_target=("from-behind arch", "From behind and above ..."),
                    grooming="neatly trimmed")
    u = captured["user"]
    assert "EXPLICIT REVEAL STYLE" in u and "from-behind arch" in u
    assert "neatly trimmed" in u and "never clinical or gynecological" in u


def test_t4_directive_and_reveal_data_are_artistic():
    d = AD.TIER_DIRECTIVES["T4_explicit"].lower()
    assert "fine-art" in d and "off-centre" in d and "natural" in d
    assert "clinical" in d or "gynecolog" in d                       # explicitly forbids it
    import math
    assert len(AD.REVEAL_STYLES) == 11                               # prime …
    assert math.gcd(len(AD.REVEAL_STYLES), len(AD.FRAMING_TARGETS)) == 1  # … coprime: no lockstep
    for label, st in AD.REVEAL_SHOT_PIN.items():
        assert any(label == r[0] for r in AD.REVEAL_STYLES)
        assert st in {"close_up", "bust", "medium", "full_body", "wide_environmental"}


def test_mirror_dangling_ignores_normal_prose():
    """The MIRROR_DANGLING audit check must NOT flag normal LLM prose — `before
    her,` / `in a hand,` / `before,` are legitimate ("the stones before her,",
    "a goblet in a hand,", "moments before,"); only ungrammatical mirror-strip
    leftovers trip it."""
    from scripts.audit_prompts import detect_mirror_dangling, score_prompt
    assert detect_mirror_dangling("her hands rest on the mossy stones before her, fingers clean") == []
    assert detect_mirror_dangling("a silver goblet in a hand, raised to the candlelight") == []
    assert detect_mirror_dangling("moments before, the room was dark and still") == []
    # genuinely ungrammatical mirror-strip artifacts are still caught
    assert detect_mirror_dangling("she holds the at arm's length, gazing") != []
    # end-to-end: a clean prompt with "before her," is no longer docked
    clean = ("Dappled light falls across a woman kneeling on the mossy bank, her "
             "hands resting on the smooth stones before her, her gaze serene and "
             "untamed, shot on 35mm film with a warm dreamy bokeh. ") * 2
    score, issues = score_prompt(clean, "T3_artnude")
    assert not any("MIRROR_DANGLING" in i for i in issues)


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


def test_opener_ban_ignores_assigned_look_tokens():
    """2026-06-11 night-batch regression: subject-first prompts all open
    'A [look] woman with ...', so a banned opener that matches via the
    ASSIGNED look's tokens must NOT fire — only distinctive scene content
    (pose/place/prop) counts."""
    look = "warm honey-blonde hair, deep brown skin"
    look_toks = frozenset(
        __import__("re").findall(r"[a-z']+", look.lower()))
    banned = ["A warm honey-blonde woman with deep brown skin"]
    # Same look, DIFFERENT scene content -> must pass
    cand = ("A warm honey-blonde woman with deep brown skin kneels at the "
            "marble fountain, trailing one hand in the spray of water.")
    assert AD._too_similar(cand, [], [], banned, look_tokens=look_toks) is None
    # Same DISTINCTIVE scene words as a banned opener -> must still reject
    banned2 = ["A woman stands at the lip of a villa pool in golden light"]
    cand2 = ("A woman stands at the lip of the villa pool, "
             "golden light raking across her shoulders and hair.")
    assert AD._too_similar(cand2, [], [], banned2) is not None


def test_compute_error_uses_infra_budget_not_attempts(monkeypatch):
    """2026-06-11 regression: LM Studio 'Compute error' storms must NOT
    consume the creative attempt budget — they retry on a separate infra
    budget (with engine recovery) and the scene still succeeds."""
    calls = {"n": 0}

    def flaky(client, **kw):
        calls["n"] += 1
        if calls["n"] <= 3:   # three straight engine errors...
            raise RuntimeError(
                'LM Studio returned HTTP 400: {"error":"Compute error."}')
        filler = " ".join(chr(97 + j % 26) + chr(98 + j % 25) for j in range(80))
        return {"prompt": filler, "orientation": "portrait",
                "shot_type": "medium", "framing_rationale": "r"}

    recovered = []
    monkeypatch.setattr(AD, "generate_one", flaky)
    monkeypatch.setattr(AD, "_recover_lm_studio", lambda tag: recovered.append(tag))
    rows = AD.generate_series(brief="b", tier="T3_artnude", count=1,
                              model_tag="m", temperature=0.8,
                              audit_gate=False, client=object())
    assert len(rows) == 1, "scene must survive an engine-error storm"
    assert calls["n"] == 4          # 3 infra retries + 1 success
    assert recovered, "engine recovery should fire on the 2nd consecutive error"


def test_creative_axes_assigned_and_injected(monkeypatch):
    """2026-06-12 creativity upgrade: every image gets a rotating editorial
    CONCEPT + COMPOSITION (all tiers) and a SENSUAL STYLING reveal (T1-T3 only;
    T4 uses its own explicit REVEAL_STYLES), each injected into the prompt."""
    seen = []

    def rec(client, **kw):
        seen.append({"concept": kw.get("concept"),
                     "sensual_reveal": kw.get("sensual_reveal"),
                     "composition": kw.get("composition")})
        n = len(seen)
        filler = " ".join(chr(97 + n % 26) + chr(98 + j % 25) + chr(99 + (j // 25) % 24)
                          for j in range(80))
        return {"prompt": filler, "orientation": "portrait",
                "shot_type": "medium", "framing_rationale": "r"}

    monkeypatch.setattr(AD, "generate_one", rec)
    AD.generate_series(brief="b", tier="T3_artnude", count=4, model_tag="m",
                       temperature=0.8, audit_gate=False, client=object())
    assert all(s["concept"] in AD.EDITORIAL_CONCEPTS for s in seen)
    assert all(s["composition"] in AD.COMPOSITION_PRINCIPLES for s in seen)
    assert all(s["sensual_reveal"] in AD.SENSUAL_REVEALS for s in seen)  # T1-T3
    assert len({s["concept"] for s in seen}) == 4, "concept must vary per image"

    seen.clear()
    AD.generate_series(brief="b", tier="T4_explicit", count=3, model_tag="m",
                       temperature=0.8, audit_gate=False, client=object())
    assert all(s["sensual_reveal"] == "" for s in seen)  # T4 excluded
    assert all(s["concept"] in AD.EDITORIAL_CONCEPTS for s in seen)

    seen.clear()  # T1 is "fully clothed" — the undress reveal axis is excluded
    AD.generate_series(brief="b", tier="T1_suggestive", count=3, model_tag="m",
                       temperature=0.8, audit_gate=False, client=object())
    assert all(s["sensual_reveal"] == "" for s in seen)  # T1 excluded
    assert all(s["concept"] in AD.EDITORIAL_CONCEPTS for s in seen)  # concept stays
    assert all(s["composition"] in AD.COMPOSITION_PRINCIPLES for s in seen)


def test_generate_one_injects_creative_axes():
    """generate_one weaves the concept / sensual-styling / composition
    directives into the user prompt."""
    captured = {}

    class _Client:
        def generate_json(self, system, user, **kw):
            captured["user"] = user
            return {"prompt": "p " * 80, "orientation": "portrait",
                    "shot_type": "medium", "framing_rationale": "r"}

    AD.generate_one(_Client(), brief="b", tier="T2_implied", sub_look="x — y",
                    avoid=[], banned_openers=[], model_tag="m", temperature=0.8,
                    concept="quiet power — she owns the frame",
                    sensual_reveal="a thin strap slipping off one shoulder",
                    composition="LEADING LINES — a fold guides the eye to her")
    u = captured["user"]
    assert "CREATIVE CONCEPT" in u and "quiet power" in u
    assert "SENSUAL STYLING" in u and "slipping off one shoulder" in u
    assert "COMPOSITION" in u and "LEADING LINES" in u
