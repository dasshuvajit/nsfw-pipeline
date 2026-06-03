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
