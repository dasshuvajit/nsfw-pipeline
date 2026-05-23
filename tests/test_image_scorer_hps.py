"""Phase G — HPS v2 + ImageReward scoring tests.

Mocked end-to-end: the heavyweight CLIP / HPS / ImageReward backbones
are stubbed out so the tests run on any machine without ~3GB of
weights cached. We only exercise the *composition*, *threading*, and
*graceful-fallback* logic — not the predictor backends themselves
(those are upstream packages).

Coverage:

  * ``_image_reward_norm`` / ``_hps_v2_norm`` — sigmoid + clamp shapes.
  * ``CompositeWeights`` — Phase-G defaults sum to 1.0; legacy round-trip.
  * ``ImageScorer`` constructor — flag → predictor wiring + weights pick.
  * ``_compose`` — weight redistribution when Phase-G signals are None.
  * ``score()`` — prompt threading, missing prompt → both signals None,
    flag emission for low_hps_v2 / low_image_reward.
  * Graceful degradation — predictor raises ScorerModelError once, gets
    disabled for the rest of the run, no second exception.
  * ``score_batch`` — reads ``prompt_text`` from each img dict, writes
    ``hps_v2_score`` / ``image_reward_score`` keys.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from src.scoring.image_scorer import (
    CompositeWeights,
    ImageScorer,
    ScorerModelError,
    _hps_v2_norm,
    _image_reward_norm,
    _LEGACY_WEIGHTS,
    _PHASE_G_WEIGHTS,
)


# ── Pure helpers ────────────────────────────────────────────────────


class TestNormalizers:
    def test_image_reward_norm_zero_is_half(self):
        assert _image_reward_norm(0.0) == pytest.approx(0.5, abs=1e-6)

    def test_image_reward_norm_positive_above_half(self):
        assert _image_reward_norm(2.0) > 0.7

    def test_image_reward_norm_negative_below_half(self):
        assert _image_reward_norm(-2.0) < 0.3

    def test_image_reward_norm_bounded(self):
        # Sigmoid is bounded in (0, 1) for any finite input.
        for raw in [-100.0, -10.0, 0.0, 10.0, 100.0]:
            v = _image_reward_norm(raw)
            assert 0.0 <= v <= 1.0

    def test_image_reward_norm_monotonic(self):
        # Higher reward → higher normalized score.
        assert _image_reward_norm(-1.0) < _image_reward_norm(0.0) < _image_reward_norm(1.0)

    def test_hps_v2_norm_passthrough_in_range(self):
        assert _hps_v2_norm(0.5) == 0.5
        assert _hps_v2_norm(0.0) == 0.0
        assert _hps_v2_norm(1.0) == 1.0

    def test_hps_v2_norm_clamps_negative(self):
        assert _hps_v2_norm(-0.3) == 0.0

    def test_hps_v2_norm_clamps_above_one(self):
        # Defensive clamp: future HPS variants may return logits.
        assert _hps_v2_norm(1.5) == 1.0


# ── CompositeWeights ────────────────────────────────────────────────


class TestCompositeWeights:
    def test_phase_g_defaults_sum_to_one(self):
        w = CompositeWeights()
        total = w.hps_v2 + w.image_reward + w.aesthetic + w.face + w.blur + w.resolution
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_legacy_zeros_phase_g_signals(self):
        w = CompositeWeights.legacy()
        assert w.hps_v2 == 0.0
        assert w.image_reward == 0.0

    def test_legacy_sums_to_one(self):
        w = CompositeWeights.legacy()
        total = w.hps_v2 + w.image_reward + w.aesthetic + w.face + w.blur + w.resolution
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_phase_g_module_constants_match_classes(self):
        # _LEGACY_WEIGHTS / _PHASE_G_WEIGHTS are how ImageScorer picks
        # without the caller specifying weights — guard the wiring.
        assert _LEGACY_WEIGHTS == CompositeWeights.legacy()
        assert _PHASE_G_WEIGHTS == CompositeWeights()


# ── ImageScorer constructor ─────────────────────────────────────────


class TestScorerConstruction:
    def test_default_off_picks_legacy_weights(self):
        s = ImageScorer()
        assert s.use_hps_v2 is False
        assert s.use_image_reward is False
        assert s.weights == _LEGACY_WEIGHTS
        assert s.hps_v2_model is None
        assert s.image_reward_model is None

    def test_only_hps_picks_phase_g_weights(self):
        s = ImageScorer(use_hps_v2=True)
        assert s.weights == _PHASE_G_WEIGHTS
        assert s.hps_v2_model is not None
        assert s.image_reward_model is None

    def test_only_image_reward_picks_phase_g_weights(self):
        s = ImageScorer(use_image_reward=True)
        assert s.weights == _PHASE_G_WEIGHTS
        assert s.hps_v2_model is None
        assert s.image_reward_model is not None

    def test_both_on_constructs_both(self):
        s = ImageScorer(use_hps_v2=True, use_image_reward=True)
        assert s.weights == _PHASE_G_WEIGHTS
        assert s.hps_v2_model is not None
        assert s.image_reward_model is not None

    def test_explicit_weights_override_flag_default(self):
        custom = CompositeWeights(
            hps_v2=0.5, image_reward=0.5,
            aesthetic=0.0, face=0.0, blur=0.0, resolution=0.0,
        )
        s = ImageScorer(use_hps_v2=True, weights=custom)
        assert s.weights == custom


# ── _compose weight redistribution ──────────────────────────────────


class TestComposeRedistribution:
    """When a Phase-G signal is None its weight is dropped from the
    denominator. The composite stays in [0, 1] regardless of which
    signals are active.
    """

    def test_legacy_only_signals(self):
        s = ImageScorer()
        # Legacy weights: aesthetic=0.40 blur=0.25 face=0.25 res=0.10.
        # Perfect inputs → composite = 1.0.
        c = s._compose(
            aesthetic=10.0, blur=500.0, face_conf=1.0, res_ok=True,
            hps_v2=None, image_reward=None,
        )
        assert c == pytest.approx(1.0, abs=1e-6)

    def test_phase_g_perfect_signals_compose_to_one(self):
        s = ImageScorer(use_hps_v2=True, use_image_reward=True)
        c = s._compose(
            aesthetic=10.0, blur=500.0, face_conf=1.0, res_ok=True,
            hps_v2=1.0, image_reward=10.0,  # 10.0 sigmoid → ~0.99
        )
        assert 0.99 <= c <= 1.0

    def test_phase_g_with_none_hps_redistributes(self):
        s = ImageScorer(use_hps_v2=True, use_image_reward=True)
        # When hps_v2 is None, its 0.30 weight drops out of the denominator
        # rather than contributing 0 to the numerator. Same numerically-
        # perfect inputs should still produce a high composite.
        c = s._compose(
            aesthetic=10.0, blur=500.0, face_conf=1.0, res_ok=True,
            hps_v2=None, image_reward=10.0,
        )
        # Without redistribution: 0.20+0.10+0.10+0.05+0.25*0.99 ≈ 0.6975
        # With redistribution (denominator 0.70): ~0.998.
        assert c > 0.99

    def test_phase_g_with_both_none_falls_back_to_legacy_shape(self):
        s = ImageScorer(use_hps_v2=True, use_image_reward=True)
        # Both Phase-G signals None → only the four legacy slots are
        # active; composite = used / (0.20+0.10+0.10+0.05) = used / 0.45.
        c = s._compose(
            aesthetic=10.0, blur=500.0, face_conf=1.0, res_ok=True,
            hps_v2=None, image_reward=None,
        )
        assert c == pytest.approx(1.0, abs=1e-6)

    def test_zero_signals_still_in_range(self):
        s = ImageScorer(use_hps_v2=True, use_image_reward=True)
        c = s._compose(
            aesthetic=0.0, blur=0.0, face_conf=0.0, res_ok=False,
            hps_v2=0.0, image_reward=-10.0,
        )
        assert 0.0 <= c <= 1.0


# ── score() with mocked predictors ──────────────────────────────────


@pytest.fixture
def fake_image(tmp_path: Path) -> Path:
    """Write a small RGB PNG so cv2.imread + PIL.open both succeed."""
    arr = np.full((1024, 768, 3), 200, dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    p = tmp_path / "fake.png"
    img.save(p)
    return p


def _stub_aesthetic(_self, _pil):
    return 7.5  # > 4.5 threshold; never triggers low_aesthetic flag


def _stub_face(_self, _bgr):
    class _Face:
        det_score = 0.92
    return [_Face()]


class TestScoreWithMockedPredictors:
    def test_score_without_prompt_skips_phase_g(self, fake_image: Path):
        """No prompt → both Phase-G signals None even when flags are on."""
        with (
            patch(
                "src.scoring.image_scorer.AestheticPredictor.predict",
                _stub_aesthetic,
            ),
            patch(
                "src.scoring.image_scorer._FaceAnalyzerWrapper.detect",
                _stub_face,
            ),
        ):
            s = ImageScorer(use_hps_v2=True, use_image_reward=True)
            result = s.score(fake_image)  # no prompt
        assert result["hps_v2"] is None
        assert result["image_reward"] is None
        assert "low_hps_v2" not in result["flags"]
        assert "low_image_reward" not in result["flags"]
        # composite is still produced — uses legacy slot redistribution.
        assert 0.0 <= result["composite"] <= 1.0

    def test_score_threads_prompt_to_both_predictors(self, fake_image: Path):
        captured: dict[str, Any] = {}

        def stub_hps(_self, _pil, prompt: str):
            captured["hps_prompt"] = prompt
            return 0.85

        def stub_ir(_self, _pil, prompt: str):
            captured["ir_prompt"] = prompt
            return 0.5

        with (
            patch(
                "src.scoring.image_scorer.AestheticPredictor.predict",
                _stub_aesthetic,
            ),
            patch(
                "src.scoring.image_scorer._FaceAnalyzerWrapper.detect",
                _stub_face,
            ),
            patch("src.scoring.image_scorer.HPSv2Predictor.predict", stub_hps),
            patch("src.scoring.image_scorer.ImageRewardPredictor.predict", stub_ir),
        ):
            s = ImageScorer(use_hps_v2=True, use_image_reward=True)
            r = s.score(fake_image, prompt="a woman in a red dress")
        assert captured["hps_prompt"] == "a woman in a red dress"
        assert captured["ir_prompt"] == "a woman in a red dress"
        assert r["hps_v2"] == pytest.approx(0.85, abs=1e-3)
        assert r["image_reward"] == pytest.approx(0.5, abs=1e-3)

    def test_low_hps_v2_flag_emitted(self, fake_image: Path):
        with (
            patch(
                "src.scoring.image_scorer.AestheticPredictor.predict",
                _stub_aesthetic,
            ),
            patch(
                "src.scoring.image_scorer._FaceAnalyzerWrapper.detect",
                _stub_face,
            ),
            patch(
                "src.scoring.image_scorer.HPSv2Predictor.predict",
                lambda _s, _p, _t: 0.10,  # below 0.20 threshold
            ),
        ):
            s = ImageScorer(use_hps_v2=True)
            r = s.score(fake_image, prompt="x")
        assert "low_hps_v2" in r["flags"]

    def test_low_image_reward_flag_emitted(self, fake_image: Path):
        with (
            patch(
                "src.scoring.image_scorer.AestheticPredictor.predict",
                _stub_aesthetic,
            ),
            patch(
                "src.scoring.image_scorer._FaceAnalyzerWrapper.detect",
                _stub_face,
            ),
            patch(
                "src.scoring.image_scorer.ImageRewardPredictor.predict",
                lambda _s, _p, _t: -2.0,  # below -1.5 threshold
            ),
        ):
            s = ImageScorer(use_image_reward=True)
            r = s.score(fake_image, prompt="x")
        assert "low_image_reward" in r["flags"]


# ── NSFW-mode behavior ──────────────────────────────────────────────


def _stub_face_none(_self, _bgr):
    """No face detected — simulates an intentional close-up anatomy crop."""
    return []


def _stub_aesthetic_nsfw(_self, _pil):
    """LAION-CLIP underrates nude work; 5.4 is typical for a clean T4 shot."""
    return 5.4


class TestNsfwMode:
    """T3/T4 scoring path: blur/aesthetic floors relax, no-face is neutral."""

    def test_no_face_drops_face_from_composite_at_t4(self, fake_image: Path):
        """A close-up anatomy crop (no face) shouldn't be penalized at T4."""
        with (
            patch(
                "src.scoring.image_scorer.AestheticPredictor.predict",
                _stub_aesthetic_nsfw,
            ),
            patch(
                "src.scoring.image_scorer._FaceAnalyzerWrapper.detect",
                _stub_face_none,
            ),
        ):
            s = ImageScorer()
            sfw = s.score(fake_image)  # no content_level → SFW path
            nsfw = s.score(fake_image, content_level="T4_explicit")
        # SFW path: face_conf=0 drags composite down hard.
        # NSFW path: face slot is skipped entirely; composite jumps.
        assert nsfw["composite"] > sfw["composite"] + 0.15
        assert "no_face" in sfw["flags"]
        assert "no_face" not in nsfw["flags"]

    def test_nsfw_blur_floor_relaxed(self, fake_image: Path):
        """Chroma soft cinematic skin (blur var ~40) shouldn't flag at T4."""
        # Real Laplacian variance from fake_image is computed inside score().
        # The flat 200-fill image has var ~0, so we instead verify behavior
        # via the threshold module constants.
        from src.scoring.image_scorer import (
            _BLUR_FLAG_THRESHOLD,
            _NSFW_BLUR_FLAG_THRESHOLD,
        )
        assert _NSFW_BLUR_FLAG_THRESHOLD < _BLUR_FLAG_THRESHOLD
        # 40 should pass the NSFW floor but fail the SFW one.
        assert 40 > _NSFW_BLUR_FLAG_THRESHOLD and 40 < _BLUR_FLAG_THRESHOLD

    def test_nsfw_aesthetic_remap_widens_top_end(self, fake_image: Path):
        """LAION aesthetic 5.4 should map ~0.6 in NSFW mode, ~0.54 in SFW."""
        with (
            patch(
                "src.scoring.image_scorer.AestheticPredictor.predict",
                _stub_aesthetic_nsfw,
            ),
            patch(
                "src.scoring.image_scorer._FaceAnalyzerWrapper.detect",
                _stub_face,
            ),
        ):
            s = ImageScorer()
            sfw = s.score(fake_image)
            nsfw = s.score(fake_image, content_level="T4_explicit")
        # NSFW remap (aes - 1.5) / 6.5 widens the SFW [0, 10] range.
        # Same aesthetic input → higher composite in NSFW mode.
        assert nsfw["composite"] > sfw["composite"]

    def test_t1_t2_use_sfw_path(self, fake_image: Path):
        """Tiers below T3 must NOT trigger NSFW mode."""
        with (
            patch(
                "src.scoring.image_scorer.AestheticPredictor.predict",
                _stub_aesthetic_nsfw,
            ),
            patch(
                "src.scoring.image_scorer._FaceAnalyzerWrapper.detect",
                _stub_face_none,
            ),
        ):
            s = ImageScorer()
            t1 = s.score(fake_image, content_level="T1_suggestive")
            t2 = s.score(fake_image, content_level="T2_implied")
            sfw = s.score(fake_image)  # baseline
        # T1/T2 keep the SFW no-face penalty.
        assert "no_face" in t1["flags"]
        assert "no_face" in t2["flags"]
        assert t1["composite"] == sfw["composite"]
        assert t2["composite"] == sfw["composite"]

    def test_caller_supplied_weights_not_overridden_in_nsfw_mode(
        self, fake_image: Path,
    ):
        """Explicit ``weights=`` at construction wins even in NSFW mode."""
        custom = CompositeWeights(
            hps_v2=0.0, image_reward=0.0,
            aesthetic=1.0, face=0.0, blur=0.0, resolution=0.0,
        )
        with (
            patch(
                "src.scoring.image_scorer.AestheticPredictor.predict",
                _stub_aesthetic_nsfw,
            ),
            patch(
                "src.scoring.image_scorer._FaceAnalyzerWrapper.detect",
                _stub_face,
            ),
        ):
            s = ImageScorer(weights=custom)
            r = s.score(fake_image, content_level="T4_explicit")
        # With pure-aesthetic weights and NSFW remap: (5.4 - 1.5) / 6.5 ≈ 0.6
        assert r["composite"] == pytest.approx(0.6, abs=0.01)

    def test_multiple_faces_tolerates_bokeh_ghost(self, fake_image: Path):
        """At T3/T4 a second face (bokeh element) shouldn't flag."""
        def _two_faces(_self, _bgr):
            class _F:
                def __init__(self, score):
                    self.det_score = score
            return [_F(0.92), _F(0.55)]
        with (
            patch(
                "src.scoring.image_scorer.AestheticPredictor.predict",
                _stub_aesthetic_nsfw,
            ),
            patch(
                "src.scoring.image_scorer._FaceAnalyzerWrapper.detect",
                _two_faces,
            ),
        ):
            s = ImageScorer()
            sfw = s.score(fake_image)
            nsfw = s.score(fake_image, content_level="T4_explicit")
        assert "multiple_faces" in sfw["flags"]
        assert "multiple_faces" not in nsfw["flags"]


# ── Graceful degradation ────────────────────────────────────────────


class TestGracefulDegradation:
    def test_hps_v2_load_failure_disables_for_run(
        self, fake_image: Path, caplog: pytest.LogCaptureFixture,
    ):
        """ScorerModelError on first predict → flag set, no second call."""
        call_count = {"n": 0}

        def stub_hps_fails(_self, _pil, _prompt):
            call_count["n"] += 1
            raise ScorerModelError("hpsv2 not installed")

        with (
            patch(
                "src.scoring.image_scorer.AestheticPredictor.predict",
                _stub_aesthetic,
            ),
            patch(
                "src.scoring.image_scorer._FaceAnalyzerWrapper.detect",
                _stub_face,
            ),
            patch(
                "src.scoring.image_scorer.HPSv2Predictor.predict",
                stub_hps_fails,
            ),
        ):
            s = ImageScorer(use_hps_v2=True)
            r1 = s.score(fake_image, prompt="x")
            r2 = s.score(fake_image, prompt="x")
        assert r1["hps_v2"] is None
        assert r2["hps_v2"] is None
        assert call_count["n"] == 1  # disabled after first failure
        assert s._hps_v2_disabled is True

    def test_image_reward_runtime_error_logged_not_disabled(
        self, fake_image: Path,
    ):
        """A non-ScorerModelError runtime failure logs but doesn't poison
        the rest of the run — could be a transient torch glitch.
        """
        call_count = {"n": 0}

        def stub_ir_runtime_err(_self, _pil, _prompt):
            call_count["n"] += 1
            raise RuntimeError("transient torch failure")

        with (
            patch(
                "src.scoring.image_scorer.AestheticPredictor.predict",
                _stub_aesthetic,
            ),
            patch(
                "src.scoring.image_scorer._FaceAnalyzerWrapper.detect",
                _stub_face,
            ),
            patch(
                "src.scoring.image_scorer.ImageRewardPredictor.predict",
                stub_ir_runtime_err,
            ),
        ):
            s = ImageScorer(use_image_reward=True)
            r1 = s.score(fake_image, prompt="x")
            r2 = s.score(fake_image, prompt="x")
        assert r1["image_reward"] is None
        assert r2["image_reward"] is None
        # Runtime errors keep the predictor enabled for retry.
        assert call_count["n"] == 2
        assert s._image_reward_disabled is False


# ── score_batch threading ───────────────────────────────────────────


class TestScoreBatch:
    def test_batch_threads_prompt_text_per_image(self, fake_image: Path):
        captured: list[str | None] = []

        def stub_hps(_self, _pil, prompt: str):
            captured.append(prompt)
            return 0.7

        with (
            patch(
                "src.scoring.image_scorer.AestheticPredictor.predict",
                _stub_aesthetic,
            ),
            patch(
                "src.scoring.image_scorer._FaceAnalyzerWrapper.detect",
                _stub_face,
            ),
            patch("src.scoring.image_scorer.HPSv2Predictor.predict", stub_hps),
        ):
            s = ImageScorer(use_hps_v2=True)
            batch = [
                {"file_path": str(fake_image), "prompt_text": "scene one"},
                {"file_path": str(fake_image), "prompt_text": "scene two"},
            ]
            s.score_batch(batch)
        assert captured == ["scene one", "scene two"]
        assert all(img["hps_v2_score"] == pytest.approx(0.7, abs=1e-3) for img in batch)
        assert all(img["image_reward_score"] is None for img in batch)

    def test_batch_writes_phase_g_keys_when_disabled(self, fake_image: Path):
        """Even when Phase-G is off, the keys exist (set to None) so
        downstream SQL INSERT doesn't have to special-case the schema."""
        with (
            patch(
                "src.scoring.image_scorer.AestheticPredictor.predict",
                _stub_aesthetic,
            ),
            patch(
                "src.scoring.image_scorer._FaceAnalyzerWrapper.detect",
                _stub_face,
            ),
        ):
            s = ImageScorer()  # both flags off
            batch = [{"file_path": str(fake_image)}]
            s.score_batch(batch)
        assert batch[0]["hps_v2_score"] is None
        assert batch[0]["image_reward_score"] is None
        # Legacy fields are still written.
        assert "quality_score" in batch[0]
        assert "aesthetic_score" in batch[0]

    def test_batch_handles_missing_prompt_text(self, fake_image: Path):
        """Image dict without ``prompt_text`` → Phase-G signals None even
        with both flags on; no crash."""
        with (
            patch(
                "src.scoring.image_scorer.AestheticPredictor.predict",
                _stub_aesthetic,
            ),
            patch(
                "src.scoring.image_scorer._FaceAnalyzerWrapper.detect",
                _stub_face,
            ),
        ):
            s = ImageScorer(use_hps_v2=True, use_image_reward=True)
            batch = [{"file_path": str(fake_image)}]  # no prompt_text
            s.score_batch(batch)
        assert batch[0]["hps_v2_score"] is None
        assert batch[0]["image_reward_score"] is None
