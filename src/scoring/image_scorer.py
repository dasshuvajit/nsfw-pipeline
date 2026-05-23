"""Concrete image quality scorer — ARCHITECTURE.md Section 15.

Six signals + a resolution check, combined into one composite score:

    hps_v2           HPS v2 — human preference (CLIP-H + classifier head)
    image_reward     BAAI ImageReward — prompt-image alignment + aesthetics
    aesthetic        LAION aesthetic-predictor v2 (CLIP ViT-L/14 + MLP)
    blur             OpenCV Laplacian variance
    face_confidence  insightface buffalo_l detector `det_score`
    resolution       h >= 1024 AND w >= 768  (binary pass/fail)

    composite = 0.30 * hps_v2_norm
              + 0.25 * image_reward_norm
              + 0.20 * aesthetic/10
              + 0.10 * face_confidence
              + 0.10 * min(blur, 500)/500
              + 0.05 * (1.0 if resolution_ok else 0.0)

Phase G adds HPS v2 + ImageReward — both prompt-conditioned human
preference models. LAION aesthetic correlates only ~0.3 with curated
NSFW fine-art taste; HPS v2 (~0.5) and ImageReward (~0.55) close that
gap. Both models stay opt-in via ``scoring.use_hps_v2`` and
``scoring.use_image_reward`` config flags — when off, the composite
falls back to the legacy four-signal formula so users without the
~1GB of new model weights aren't broken.

Flags raised on low-quality outputs so downstream filters can reject:

    low_aesthetic    aesthetic < 4.5
    blurry           blur    < 80
    no_face          no face detected
    multiple_faces   >1 faces detected (character consistency killer)
    low_hps_v2       hps_v2 < 0.20  (only when HPS v2 enabled)
    low_image_reward image_reward < -1.5  (only when ImageReward enabled)

Design notes:

1. **Lazy model loading.** All backbones (CLIP+MLP, insightface, HPS v2,
   ImageReward) are multi-hundred-MB, slow to initialize, and only needed
   when we actually score something. Importing this module or constructing
   an ``ImageScorer`` is cheap; the first ``score()`` call is where the
   cost lands. This lets tests and the rest of the pipeline import the
   class without paying for it.

2. **One-time load, process-lifetime cache.** After the first score, the
   models stay in memory for the rest of the process. Scoring N images in
   a row pays the init cost once.

3. **MPS vs CPU.** On Mac M4 Pro we prefer MPS for CLIP; insightface uses
   its own CPU onnxruntime path (the official buffalo_l bundle has no
   CoreML variant). HPS v2 and ImageReward both use torch and inherit
   the same MPS/CPU pick.

4. **Prompt-conditioned scorers.** HPS v2 and ImageReward both score
   image-vs-prompt alignment, not the image alone — they need the
   positive prompt at score time. ``score(path, prompt=...)`` accepts
   it explicitly; ``score_batch`` reads ``prompt_text`` from each img
   dict. When the prompt is missing, those two signals are skipped and
   the composite falls back to the legacy formula for that one image.

5. **Graceful degradation.** When a flag is on but the predictor fails
   to load (e.g. weights not downloaded yet), we log once at WARNING and
   silently fall back to the legacy formula for the rest of the run.
   The pipeline doesn't crash mid-batch over a missing scorer.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Default weights path: models/aesthetic/sac+logos+ava1-l14-linearMSE.pth
# (LAION improved-aesthetic-predictor v2, trained on SAC+LOGOs+AVA1, MSE loss).
# This is downloaded once during setup — see PROJECT_GUIDE.md.
_DEFAULT_AESTHETIC_WEIGHTS = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "aesthetic"
    / "sac+logos+ava1-l14-linearMSE.pth"
)

# Composite score weights — Phase G default. ARCHITECTURE.md Section 15
# legacy weights (0.40/0.25/0.25/0.10 across aesthetic/blur/face/res) are
# used as fallback when ``use_hps_v2`` / ``use_image_reward`` are both
# off, preserving back-compat for users without the new model weights.
@dataclass(frozen=True)
class CompositeWeights:
    hps_v2: float = 0.30
    image_reward: float = 0.25
    aesthetic: float = 0.20
    face: float = 0.10
    blur: float = 0.10
    resolution: float = 0.05

    @classmethod
    def legacy(cls) -> "CompositeWeights":
        """Pre-Phase-G four-signal composite (HPS / ImageReward zeroed)."""
        return cls(
            hps_v2=0.0, image_reward=0.0,
            aesthetic=0.40, face=0.25, blur=0.25, resolution=0.10,
        )

    @classmethod
    def nsfw_legacy(cls) -> "CompositeWeights":
        """T3/T4 weighting when HPS v2 + ImageReward are disabled.

        Drops face + blur weight versus :meth:`legacy` because the
        SFW-trained scorers misjudge intentional NSFW composition:
        close-up anatomy crops have no face (face → 0) and Chroma's
        soft cinematic skin texture pushes Laplacian variance under
        the SFW blur floor. Pushes the released share onto aesthetic +
        resolution, which still correlate with finished quality at
        higher tiers.
        """
        return cls(
            hps_v2=0.0, image_reward=0.0,
            aesthetic=0.55, face=0.15, blur=0.15, resolution=0.15,
        )

    @classmethod
    def nsfw(cls) -> "CompositeWeights":
        """T3/T4 weighting when HPS v2 + ImageReward are enabled.

        Same rebalancing principle as :meth:`nsfw_legacy` but with the
        Phase-G predictors absorbing most of the released face/blur
        share — HPS v2 + ImageReward both handle nudity gracefully.
        """
        return cls(
            hps_v2=0.35, image_reward=0.30,
            aesthetic=0.20, face=0.05, blur=0.05, resolution=0.05,
        )


_LEGACY_WEIGHTS = CompositeWeights.legacy()
_PHASE_G_WEIGHTS = CompositeWeights()
_NSFW_LEGACY_WEIGHTS = CompositeWeights.nsfw_legacy()
_NSFW_PHASE_G_WEIGHTS = CompositeWeights.nsfw()
_NSFW_LEVELS = frozenset({"T3_artnude", "T4_explicit"})


# Normalization / threshold constants — also from Section 15.
_BLUR_CLAMP = 500.0           # Laplacian var is normalized by min(v, 500) / 500
_BLUR_FLAG_THRESHOLD = 80.0   # below this → "blurry" flag
_AESTHETIC_FLAG_THRESHOLD = 4.5
_HPS_V2_FLAG_THRESHOLD = 0.20      # below this → "low_hps_v2"
_IMAGE_REWARD_FLAG_THRESHOLD = -1.5  # below this → "low_image_reward"
_RES_MIN_H = 1024
_RES_MIN_W = 768

# T3/T4 thresholds — SFW-trained models (LAION CLIP aesthetic +
# Laplacian-variance blur) systematically under-rate finished NSFW
# work. Soft cinematic skin texture from Chroma sits at 30-60 var,
# below the SFW floor of 80. LAION CLIP aesthetic for nude content
# tends to land 0.5-1.0 points below equivalent clothed work because
# LAION-2B is filtered. These two floors release the pressure.
_NSFW_BLUR_FLAG_THRESHOLD = 30.0
_NSFW_AESTHETIC_FLAG_THRESHOLD = 3.5

# ImageReward typical range is [-2.4, +1.0]. Map to [0, 1] via a sigmoid
# centered at 0 with a divisor of 2 so a score of 0.0 → 0.5, +2 → ~0.88,
# -2 → ~0.12. Smooth and bounded.
def _image_reward_norm(raw: float) -> float:
    import math
    return 1.0 / (1.0 + math.exp(-raw / 2.0))


# HPS v2 raw scores are already preference probabilities in [0, 1] but
# we still clamp defensively in case a future model returns logits.
def _hps_v2_norm(raw: float) -> float:
    return max(0.0, min(1.0, float(raw)))


# ----- exceptions -----------------------------------------------------------


class ScorerError(Exception):
    """Anything that prevents scoring from producing a result."""


class ScorerModelError(ScorerError):
    """A backing model (CLIP, MLP weights, insightface) failed to load."""


# ----- aesthetic predictor --------------------------------------------------


class AestheticPredictor:
    """LAION aesthetic-predictor v2: CLIP ViT-L/14 image embedding → MLP → 0–10 score.

    Architecture of the MLP head matches
    https://github.com/christophschuhmann/improved-aesthetic-predictor verbatim,
    so the `sac+logos+ava1-l14-linearMSE.pth` checkpoint loads with
    ``strict=True``.
    """

    _EMBED_DIM = 768  # CLIP ViT-L/14 image feature dim

    def __init__(
        self,
        weights_path: Path | str = _DEFAULT_AESTHETIC_WEIGHTS,
        device: str | None = None,
    ) -> None:
        self.weights_path = Path(weights_path).expanduser()
        self.device = device  # None → auto-pick at load time
        self._model = None          # CLIP image encoder
        self._preprocess = None     # torchvision transform chain
        self._head = None           # MLP head

    def load(self) -> None:
        """Load CLIP ViT-L/14 + the MLP head. Idempotent."""
        if self._model is not None:
            return

        try:
            import torch
            import open_clip
        except ImportError as exc:
            raise ScorerModelError(
                "ImageScorer requires torch + open_clip_torch. Install with:\n"
                "    pip install torch open-clip-torch\n"
                "(See requirements.txt Image/Scoring section.)"
            ) from exc

        if not self.weights_path.exists():
            raise ScorerModelError(
                f"Aesthetic MLP weights not found at {self.weights_path}.\n"
                f"Download sac+logos+ava1-l14-linearMSE.pth from "
                f"https://github.com/christophschuhmann/improved-aesthetic-predictor "
                f"and save it to models/aesthetic/. See PROJECT_GUIDE.md."
            )

        device = self.device or self._pick_device(torch)
        self.device = device

        logger.info("Loading CLIP ViT-L/14 (open_clip) on %s ...", device)
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        model = model.to(device).eval()

        logger.info("Loading aesthetic MLP head from %s", self.weights_path)
        head = _build_aesthetic_mlp(self._EMBED_DIM)
        state = torch.load(self.weights_path, map_location="cpu", weights_only=True)
        # LAION's improved-aesthetic-predictor saves the head as a
        # pl.LightningModule with `self.layers = nn.Sequential(...)`, so
        # keys look like "layers.0.weight". We use a flat nn.Sequential
        # for simplicity — strip the "layers." prefix so strict load works.
        state = {k.removeprefix("layers."): v for k, v in state.items()}
        head.load_state_dict(state)
        head = head.to(device).eval()

        self._model = model
        self._preprocess = preprocess
        self._head = head

    def predict(self, pil_img: Image.Image) -> float:
        """Return the aesthetic score (typically in roughly 1.0–9.0)."""
        self.load()
        import torch
        import torch.nn.functional as F

        img_rgb = pil_img.convert("RGB")
        tensor = self._preprocess(img_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self._model.encode_image(tensor)
            features = F.normalize(features, dim=-1)
            # The LAION v2 head was trained on float32 CLIP features.
            score = self._head(features.float()).squeeze()
        return float(score.item())

    @staticmethod
    def _pick_device(torch_mod: Any) -> str:
        if torch_mod.cuda.is_available():
            return "cuda"
        if getattr(torch_mod.backends, "mps", None) and torch_mod.backends.mps.is_available():
            return "mps"
        return "cpu"


def _build_aesthetic_mlp(input_dim: int):
    """Re-construct the LAION improved-aesthetic-predictor v2 MLP head.

    Shape matches the pretrained checkpoint exactly:
        Linear(768, 1024) → Dropout → Linear(1024, 128) → Dropout
        → Linear(128, 64) → Dropout → Linear(64, 16) → Linear(16, 1)
    """
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(input_dim, 1024),
        nn.Dropout(0.2),
        nn.Linear(1024, 128),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.Dropout(0.1),
        nn.Linear(64, 16),
        nn.Linear(16, 1),
    )


# ----- HPS v2 predictor -----------------------------------------------------


class HPSv2Predictor:
    """Wraps the ``hpsv2`` package for human-preference scoring.

    HPS v2 is a CLIP-H/14 backbone + a small classifier head fine-tuned
    on the HPDv2 dataset (~430k human preference pairs). It returns a
    preference probability in [0, 1] for an (image, prompt) pair.

    First call downloads the HPS v2 weights (~1GB) into the package's
    cache dir. Subsequent calls reuse them.

    Parameters
    ----------
    hps_version : str
        Which HPS variant to use. ``"v2.1"`` is the public default.
    """

    def __init__(self, hps_version: str = "v2.1") -> None:
        self.hps_version = hps_version
        self._loaded = False

    def load(self) -> None:
        """Verify the ``hpsv2`` package imports. Idempotent.

        The package itself lazy-loads weights on first ``score()`` call,
        so this is a quick import-check rather than a full model load.
        """
        if self._loaded:
            return
        try:
            import hpsv2  # noqa: F401
        except ImportError as exc:
            raise ScorerModelError(
                "ImageScorer with HPS v2 requires the hpsv2 package. "
                "Install with:\n"
                "    pip install hpsv2\n"
                "First call downloads ~1GB of CLIP-H weights to the "
                "package cache. See PROJECT_GUIDE.md → Scoring."
            ) from exc
        self._loaded = True

    def predict(self, pil_img: Image.Image, prompt: str) -> float:
        """Return the HPS v2 preference probability in [0, 1].

        Both the image and prompt are required — HPS v2 scores the
        alignment between them, not the image alone. Empty prompt
        raises ScorerError.
        """
        if not prompt:
            raise ScorerError("HPSv2Predictor.predict requires a non-empty prompt")
        self.load()
        import hpsv2

        # hpsv2.score accepts a PIL image, file path, or list of either,
        # plus the prompt and version. With a single image it returns a
        # single-element list.
        result = hpsv2.score(pil_img.convert("RGB"), prompt, hps_version=self.hps_version)
        if isinstance(result, list):
            return float(result[0]) if result else 0.0
        return float(result)


# ----- ImageReward predictor ------------------------------------------------


class ImageRewardPredictor:
    """Wraps the BAAI ImageReward reward model.

    ImageReward is a CLIP ViT-H/14 backbone + reward head trained on
    137k human preference comparisons. Returns a scalar reward — typical
    range [-2.4, +1.0]. Higher = better human-judged quality. Like HPS
    v2 it's prompt-conditioned.

    Parameters
    ----------
    model_name : str
        Which ImageReward variant. ``"ImageReward-v1.0"`` is the default
        public release.
    device : str | None
        Torch device override. ``None`` → MPS on Mac, CUDA when available,
        CPU fallback (matches AestheticPredictor).
    """

    def __init__(
        self,
        model_name: str = "ImageReward-v1.0",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import ImageReward as RM
            import torch
        except ImportError as exc:
            raise ScorerModelError(
                "ImageScorer with ImageReward requires image-reward + torch. "
                "Install with:\n"
                "    pip install image-reward\n"
                "First call downloads ~1.5GB of CLIP-H + reward head "
                "weights to ~/.cache/ImageReward. See PROJECT_GUIDE.md "
                "→ Scoring."
            ) from exc

        device = self.device or AestheticPredictor._pick_device(torch)
        self.device = device
        logger.info("Loading ImageReward %s on %s ...", self.model_name, device)
        self._model = RM.load(self.model_name, device=device)

    def predict(self, pil_img: Image.Image, prompt: str) -> float:
        """Return the ImageReward score (typical range [-2.4, +1.0])."""
        if not prompt:
            raise ScorerError(
                "ImageRewardPredictor.predict requires a non-empty prompt"
            )
        self.load()
        # The package's score() accepts (prompt, image) where image can
        # be a PIL.Image or a file path. We convert to RGB defensively.
        return float(self._model.score(prompt, pil_img.convert("RGB")))


# ----- face analyzer --------------------------------------------------------


class _FaceAnalyzerWrapper:
    """Lazy-loading wrapper around insightface FaceAnalysis (buffalo_l).

    On first use, insightface downloads buffalo_l (~300 MB) into
    ~/.insightface/models/. Subsequent runs are instant.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
        providers: Iterable[str] | None = None,
    ) -> None:
        self.model_name = model_name
        self.det_size = det_size
        self.providers = list(providers) if providers else ["CPUExecutionProvider"]
        self._app = None

    def load(self) -> None:
        if self._app is not None:
            return
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise ScorerModelError(
                "ImageScorer requires insightface + onnxruntime. Install with:\n"
                "    pip install insightface onnxruntime\n"
                "(See requirements.txt Image/Scoring section.)"
            ) from exc

        logger.info(
            "Loading insightface %s (first run will download ~300MB to "
            "~/.insightface/models/)",
            self.model_name,
        )
        app = FaceAnalysis(name=self.model_name, providers=self.providers)
        # ctx_id=-1 forces CPU; insightface has no MPS/CoreML path today.
        app.prepare(ctx_id=-1, det_size=self.det_size)
        self._app = app

    def detect(self, bgr_image: np.ndarray) -> list[Any]:
        self.load()
        return list(self._app.get(bgr_image))


# ----- result type ----------------------------------------------------------


@dataclass
class ScoreResult:
    composite: float
    aesthetic: float
    blur: float
    face_confidence: float
    flags: list[str]
    # Phase G — None when the predictor is disabled or the prompt was
    # missing for that one image. Distinct from 0.0 (which is a real
    # very-low score).
    hps_v2: float | None = None
    image_reward: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "composite": self.composite,
            "aesthetic": self.aesthetic,
            "blur": self.blur,
            "face_confidence": self.face_confidence,
            "hps_v2": self.hps_v2,
            "image_reward": self.image_reward,
            "flags": list(self.flags),
        }


# ----- main scorer ----------------------------------------------------------


class ImageScorer:
    """Composite image quality scorer.

    The constructor is cheap (no model loading). The first call to
    ``score()`` loads the enabled backbones; subsequent calls reuse them.

    Parameters
    ----------
    aesthetic_weights : Path | str
        Path to the LAION aesthetic-predictor v2 MLP head weights.
    device : str | None
        Torch device override. ``None`` → auto-pick MPS / CUDA / CPU.
    use_hps_v2 : bool
        Enable HPS v2 scoring. Default ``False`` so users without the
        ``hpsv2`` package or its weights aren't broken.
    use_image_reward : bool
        Enable ImageReward scoring. Default ``False`` for the same reason.
    weights : CompositeWeights | None
        Override the composite weighting. ``None`` → ``CompositeWeights()``
        when either Phase-G predictor is on, otherwise the legacy weights.
    """

    def __init__(
        self,
        aesthetic_weights: Path | str = _DEFAULT_AESTHETIC_WEIGHTS,
        device: str | None = None,
        *,
        use_hps_v2: bool = False,
        use_image_reward: bool = False,
        weights: CompositeWeights | None = None,
    ) -> None:
        self.aesthetic_model = AestheticPredictor(
            weights_path=aesthetic_weights, device=device
        )
        self.face_analyzer = _FaceAnalyzerWrapper()
        self.use_hps_v2 = use_hps_v2
        self.use_image_reward = use_image_reward
        self.hps_v2_model = HPSv2Predictor() if use_hps_v2 else None
        self.image_reward_model = (
            ImageRewardPredictor(device=device) if use_image_reward else None
        )
        if weights is not None:
            self.weights = weights
            self._weights_overridden = True
        elif use_hps_v2 or use_image_reward:
            self.weights = _PHASE_G_WEIGHTS
            self._weights_overridden = False
        else:
            self.weights = _LEGACY_WEIGHTS
            self._weights_overridden = False
        # Once a Phase-G predictor fails to load we stop trying it for
        # the rest of the run — log once, fall back to the legacy
        # composite for that signal. Reset by re-instantiating the scorer.
        self._hps_v2_disabled = False
        self._image_reward_disabled = False

    def load(self) -> None:
        """Eagerly load all enabled backbones. Useful for benchmarks."""
        self.aesthetic_model.load()
        self.face_analyzer.load()
        if self.hps_v2_model is not None and not self._hps_v2_disabled:
            self.hps_v2_model.load()
        if self.image_reward_model is not None and not self._image_reward_disabled:
            self.image_reward_model.load()

    def score(
        self,
        image_path: str | Path,
        *,
        prompt: str | None = None,
        content_level: str | None = None,
    ) -> dict[str, Any]:
        """Score one image. Returns a dict; rounded numeric values.

        Keys: ``composite``, ``aesthetic``, ``blur``, ``face_confidence``,
        ``hps_v2`` (None when disabled / no prompt), ``image_reward``
        (same), ``flags``.

        ``prompt`` is required for HPS v2 + ImageReward — both score
        image-vs-prompt alignment. When omitted those signals are skipped
        and the composite falls back to the legacy weighting for that
        image (no error; just degraded scoring).

        ``content_level`` switches the scorer into NSFW-tolerant mode
        when it is ``T3_artnude`` or ``T4_explicit``: blur + aesthetic
        flag floors drop, the no-face penalty becomes neutral
        (intentional close-up anatomy is valid composition, not a
        defect), and an extra face is tolerated (bokeh ghosts). Weights
        rebalance to :meth:`CompositeWeights.nsfw_legacy` (or
        :meth:`CompositeWeights.nsfw` when Phase-G predictors are on)
        unless the caller passed explicit weights at construction time.
        """
        nsfw_mode = content_level in _NSFW_LEVELS
        path = Path(image_path).expanduser()
        if not path.exists():
            raise ScorerError(f"Image not found: {path}")

        # OpenCV reads BGR from disk; insightface expects BGR; PIL wants its
        # own load for CLIP preprocessing. Two reads are cheap next to
        # inference cost.
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise ScorerError(
                f"OpenCV could not decode {path}. Corrupt file or unsupported "
                f"format?"
            )
        try:
            pil_img = Image.open(path)
        except Exception as exc:  # noqa: BLE001 — PIL raises a zoo
            raise ScorerError(f"PIL could not open {path}: {exc}") from exc

        # 1. Aesthetic (CLIP embedding → MLP, ~0–10)
        aesthetic = self.aesthetic_model.predict(pil_img)

        # 2. Blur (Laplacian variance; higher = sharper)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 3. Face detection (list of Face objects; each has .det_score in [0, 1])
        faces = self.face_analyzer.detect(bgr)
        face_conf = float(faces[0].det_score) if faces else 0.0

        # 4. Resolution: the spec mandates h >= 1024 AND w >= 768 (portrait-ish).
        h, w = bgr.shape[:2]
        res_ok = h >= _RES_MIN_H and w >= _RES_MIN_W

        # 5. HPS v2 + ImageReward — prompt-conditioned, opt-in. Both
        # silently skipped when the prompt is empty (caller didn't pass
        # one) or when the predictor was disabled at runtime.
        hps_v2 = self._maybe_hps_v2(pil_img, prompt)
        image_reward = self._maybe_image_reward(pil_img, prompt)

        composite = self._compose(
            aesthetic=aesthetic,
            blur=blur,
            face_conf=face_conf,
            res_ok=res_ok,
            hps_v2=hps_v2,
            image_reward=image_reward,
            nsfw_mode=nsfw_mode,
            no_face=not faces,
        )

        flags: list[str] = []
        aesthetic_floor = (
            _NSFW_AESTHETIC_FLAG_THRESHOLD if nsfw_mode else _AESTHETIC_FLAG_THRESHOLD
        )
        blur_floor = (
            _NSFW_BLUR_FLAG_THRESHOLD if nsfw_mode else _BLUR_FLAG_THRESHOLD
        )
        max_faces = 2 if nsfw_mode else 1
        if aesthetic < aesthetic_floor:
            flags.append("low_aesthetic")
        if blur < blur_floor:
            flags.append("blurry")
        if not faces:
            # T3/T4: a close-up anatomy crop with no face is a valid
            # composition. Skip the flag — _compose() also drops face
            # from the composite when no_face=True in nsfw_mode.
            if not nsfw_mode:
                flags.append("no_face")
        elif len(faces) > max_faces:
            flags.append("multiple_faces")
        if hps_v2 is not None and hps_v2 < _HPS_V2_FLAG_THRESHOLD:
            flags.append("low_hps_v2")
        if image_reward is not None and image_reward < _IMAGE_REWARD_FLAG_THRESHOLD:
            flags.append("low_image_reward")

        return {
            "composite": round(composite, 3),
            "aesthetic": round(aesthetic, 2),
            "blur": round(blur, 1),
            "face_confidence": round(face_conf, 3),
            "hps_v2": round(hps_v2, 4) if hps_v2 is not None else None,
            "image_reward": round(image_reward, 4) if image_reward is not None else None,
            "flags": flags,
        }

    def _maybe_hps_v2(self, pil_img: Image.Image, prompt: str | None) -> float | None:
        """Return the HPS v2 score, or ``None`` when skipped."""
        if self.hps_v2_model is None or self._hps_v2_disabled or not prompt:
            return None
        try:
            return float(self.hps_v2_model.predict(pil_img, prompt))
        except ScorerModelError as exc:
            logger.warning(
                "HPS v2 disabled for the rest of this run: %s", exc,
            )
            self._hps_v2_disabled = True
            return None
        except Exception as exc:  # noqa: BLE001 — runtime errors from torch
            logger.warning("HPS v2 score failed: %s", exc)
            return None

    def _maybe_image_reward(
        self, pil_img: Image.Image, prompt: str | None,
    ) -> float | None:
        """Return the ImageReward score, or ``None`` when skipped."""
        if self.image_reward_model is None or self._image_reward_disabled or not prompt:
            return None
        try:
            return float(self.image_reward_model.predict(pil_img, prompt))
        except ScorerModelError as exc:
            logger.warning(
                "ImageReward disabled for the rest of this run: %s", exc,
            )
            self._image_reward_disabled = True
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("ImageReward score failed: %s", exc)
            return None

    def _compose(
        self,
        *,
        aesthetic: float,
        blur: float,
        face_conf: float,
        res_ok: bool,
        hps_v2: float | None,
        image_reward: float | None,
        nsfw_mode: bool = False,
        no_face: bool = False,
    ) -> float:
        """Apply ``self.weights`` to the per-signal normalized scores.

        When a Phase-G signal is None, its weight is redistributed
        proportionally across the four legacy signals so the composite
        stays in roughly the same scale instead of dropping due to a
        zeroed slot.

        In ``nsfw_mode`` the weight set switches to the NSFW preset
        (unless the caller passed explicit weights at construction
        time) and ``no_face`` causes the face slot to be excluded from
        the composite entirely rather than scoring a hard zero — close-
        up anatomy crops shouldn't be punished for the absent face.
        """
        if nsfw_mode and not self._weights_overridden:
            w = (
                _NSFW_PHASE_G_WEIGHTS
                if (hps_v2 is not None or image_reward is not None)
                else _NSFW_LEGACY_WEIGHTS
            )
        else:
            w = self.weights

        # Per-signal normalized values in [0, 1]. In NSFW mode we
        # recalibrate the aesthetic axis: LAION-2B trained the CLIP
        # aesthetic predictor on SFW data, so finished nude work
        # systematically lands ~5-6 on the [0, 10] scale where SFW
        # equivalents score ~7-8. Mapping (aesthetic - 1.5) / 6.5
        # widens the effective NSFW range so the LAION bias doesn't
        # drag a commercially-strong T4 close-up below the cutoff.
        if nsfw_mode:
            a_norm = max(0.0, min(1.0, (aesthetic - 1.5) / 6.5))
        else:
            a_norm = max(0.0, min(1.0, aesthetic / 10.0))
        b_norm = min(blur, _BLUR_CLAMP) / _BLUR_CLAMP
        f_norm = max(0.0, min(1.0, face_conf))
        r_norm = 1.0 if res_ok else 0.0

        skip_face = nsfw_mode and no_face

        active_weight = w.aesthetic + w.blur + w.resolution
        used = (
            a_norm * w.aesthetic
            + b_norm * w.blur
            + r_norm * w.resolution
        )
        if not skip_face:
            used += f_norm * w.face
            active_weight += w.face
        if hps_v2 is not None:
            used += _hps_v2_norm(hps_v2) * w.hps_v2
            active_weight += w.hps_v2
        if image_reward is not None:
            used += _image_reward_norm(image_reward) * w.image_reward
            active_weight += w.image_reward
        if active_weight <= 0:
            return 0.0
        # Normalize so the composite stays in [0, 1] regardless of which
        # signals were actually used. Without this, disabling HPS+IR
        # would cap the composite at ~0.45 even on great images.
        return used / active_weight

    def score_batch(self, images: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Score a list of image dicts, mutating each one with score fields.

        Each dict must have ``file_path``; ``prompt_text`` is read when
        present and threaded to HPS v2 / ImageReward. Writes back:
        ``quality_score`` (composite), ``aesthetic_score``, ``blur_score``,
        ``face_confidence``, ``hps_v2_score`` (Phase G), ``image_reward_score``
        (Phase G), ``quality_flags`` (JSON string). Returns the same list
        for call-site chaining.

        Errors on a single image don't poison the whole batch: we log,
        record the error in ``quality_flags`` as ``scorer_error``, and move on.
        """
        for img in images:
            path = img.get("file_path")
            if not path:
                raise ScorerError(
                    "score_batch: image dict missing 'file_path' key: "
                    f"{list(img.keys())}"
                )
            prompt = img.get("prompt_text")
            try:
                scores = self.score(
                    path, prompt=prompt,
                    content_level=img.get("content_level"),
                )
            except ScorerError as exc:
                logger.warning("Skipping %s: %s", path, exc)
                img["quality_score"] = 0.0
                img["aesthetic_score"] = 0.0
                img["blur_score"] = 0.0
                img["face_confidence"] = 0.0
                img["hps_v2_score"] = None
                img["image_reward_score"] = None
                img["quality_flags"] = json.dumps(["scorer_error"])
                continue
            img["quality_score"] = scores["composite"]
            img["aesthetic_score"] = scores["aesthetic"]
            img["blur_score"] = scores["blur"]
            img["face_confidence"] = scores["face_confidence"]
            img["hps_v2_score"] = scores["hps_v2"]
            img["image_reward_score"] = scores["image_reward"]
            img["quality_flags"] = json.dumps(scores["flags"])
        return images
