"""Deterministic cinematic post-grade for rendered frames.

The finishing pass that adds the filmic "look" Z-Image (and Chroma) under-render:
a warm high-contrast colour grade, a gentle highlight bloom/halation, a soft
vignette and fine grain. Pure CPU (Pillow + numpy), sub-second per frame, no GPU.

Runs in ``art_series._package()`` AFTER render and BEFORE watermarking, on the
public AND gated sets. For true-4K it must run AFTER the USDU upscale (in
``upscale_folder``), never before — the diffusion upscaler would smear baked-in
glow/grain.

It AMPLIFIES light that already exists; it cannot manufacture directional light
or composition — so it pairs with the cinematic prompting levers, it does not
replace them. Blended at strength<1 so it never over-cooks into an Instagram
filter (2026-06-18 Z-Image-vs-gpt-image-1 R&D).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from PIL.PngImagePlugin import PngInfo

# Fixed seed → the grain pattern is deterministic (reproducible renders + tests).
_GRAIN_SEED = 0x5A1707

# Split-tone (highlight RGB mult, shadow RGB mult) per tone. The finish leans to match
# the niche mood: warm glamour/golden-hour, cool noir/night, neutral restrained fine-art.
_SPLIT_TONES = {
    "warm":    (np.array([1.0, 0.85, 0.55], dtype=np.float32),   # amber highlights
                np.array([0.55, 0.75, 1.0], dtype=np.float32)),  # teal shadows
    "cool":    (np.array([0.80, 0.90, 1.0], dtype=np.float32),   # cool highlights
                np.array([0.45, 0.62, 1.0], dtype=np.float32)),  # deep-blue shadows
    "neutral": (np.array([1.0, 0.97, 0.92], dtype=np.float32),   # barely warm
                np.array([0.92, 0.96, 1.0], dtype=np.float32)),  # barely cool
}
# Highlight-bloom halation tint per tone (warm halation reads golden; cool reads icy).
_BLOOM_TINTS = {
    "warm":    np.array([1.0, 0.7, 0.4], dtype=np.float32),
    "cool":    np.array([0.6, 0.8, 1.0], dtype=np.float32),
    "neutral": np.array([0.9, 0.9, 0.95], dtype=np.float32),
}


class Grader:
    """Apply a deterministic cinematic grade to an image file in place or to a copy."""

    def __init__(self, cfg: "dict | None" = None) -> None:
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.strength = _clamp(cfg.get("strength", 0.6), 0.0, 1.0)
        self.warmth = float(cfg.get("warmth", 0.06))            # split-tone amount
        # tone = which DIRECTION the split-tone leans, so the FINISH matches the niche
        # mood instead of one global warm look (2026-06-27 per-family grade): "warm" =
        # amber highlights / teal shadows (glamour, golden-hour); "cool" = cool
        # highlights / deep-blue shadows (noir/night nocturne); "neutral" = minimal tint
        # (restrained fine-art atelier). Bloom halation follows the tone.
        self.tone = str(cfg.get("tone", "warm")).lower()
        self.contrast = float(cfg.get("contrast", 0.12))        # filmic S-curve strength
        self.shadow_lift = float(cfg.get("shadow_lift", 0.02))  # gentle filmic shadow lift
        self.bloom = float(cfg.get("bloom", 0.18))              # highlight halation amount
        self.bloom_threshold = _clamp(cfg.get("bloom_threshold", 0.72), 0.0, 0.99)
        self.vignette = float(cfg.get("vignette", 0.18))        # corner darkening
        self.grain = float(cfg.get("grain", 0.015))             # film-grain sigma

    # ── public API ────────────────────────────────────────────────────
    def apply(self, src: "str | Path", out: "str | Path") -> None:
        """Grade ``src`` and write to ``out`` (may equal ``src``). PNG text chunks
        (render parameters) are preserved."""
        src, out = Path(src), Path(out)
        with Image.open(src) as im:
            text = dict(getattr(im, "text", {}) or {})
            rgb = im.convert("RGB")
            a = np.asarray(rgb, dtype=np.float32) / 255.0
        graded = self._grade(a)
        blended = np.clip(graded * self.strength + a * (1.0 - self.strength), 0.0, 1.0)
        result = Image.fromarray((blended * 255.0 + 0.5).astype(np.uint8), "RGB")
        self._save(result, out, text)

    def apply_monochrome(self, src: "str | Path", out: "str | Path") -> None:
        """True fine-art black & white: FULL luminance desaturation + a neutral
        filmic contrast S-curve, vignette and grain — the warm split-tone and warm
        bloom are skipped so the output stays pure grey. Z-Image renders muted
        COLOUR even when the prompt says 'monochrome'; this guarantees grayscale
        for niches flagged ``grayscale``. PNG text chunks are preserved."""
        src, out = Path(src), Path(out)
        with Image.open(src) as im:
            text = dict(getattr(im, "text", {}) or {})
            a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        g = np.repeat(_luma(a), 3, axis=2)                       # full desaturation → grey
        g = g + self.shadow_lift * (1.0 - g)                    # filmic shadow lift
        g = np.clip(0.5 + (1.0 + self.contrast) * (g - 0.5), 0.0, 1.0)  # contrast S-curve
        if self.vignette > 0:                                   # neutral (mask is per-pixel)
            g = g * self._vignette_mask(g.shape)
        if self.grain > 0:                                      # neutral (same noise all 3ch)
            g = np.clip(g + self._grain(g.shape), 0.0, 1.0)
        result = Image.fromarray((g * 255.0 + 0.5).astype(np.uint8), "RGB")
        self._save(result, out, text)

    @staticmethod
    def _save(result: Image.Image, out: Path, text: dict) -> None:
        if text:
            info = PngInfo()
            for k, v in text.items():
                try:
                    info.add_text(str(k), str(v))
                except Exception:  # noqa: BLE001 — never lose the image over a bad chunk
                    pass
            result.save(out, pnginfo=info)
        else:
            result.save(out)

    # ── grade stages ──────────────────────────────────────────────────
    def _grade(self, a: np.ndarray) -> np.ndarray:
        g = a.copy()
        # 1. filmic shadow lift + contrast S-curve around mid-grey
        g = g + self.shadow_lift * (1.0 - g)
        g = np.clip(0.5 + (1.0 + self.contrast) * (g - 0.5), 0.0, 1.0)
        # 2. split-tone — direction set by self.tone so the finish matches the mood
        luma = _luma(g)
        hi, lo = _SPLIT_TONES.get(self.tone, _SPLIT_TONES["warm"])
        split = hi * luma + lo * (1.0 - luma)
        g = np.clip(g * (1.0 + self.warmth * (split - 1.0)), 0.0, 1.0)
        # 3. highlight bloom / warm halation (screen-blended)
        if self.bloom > 0:
            g = self._bloom(g, luma)
        # 4. vignette
        if self.vignette > 0:
            g = g * self._vignette_mask(g.shape)
        # 5. fine grain
        if self.grain > 0:
            g = np.clip(g + self._grain(g.shape), 0.0, 1.0)
        return g

    def _bloom(self, g: np.ndarray, luma: np.ndarray) -> np.ndarray:
        denom = max(1e-3, 1.0 - self.bloom_threshold)
        mask = np.clip((luma - self.bloom_threshold) / denom, 0.0, 1.0)
        bright = (g * mask * 255.0).astype(np.uint8)
        radius = max(2, int(min(g.shape[0], g.shape[1]) * 0.012))
        blur = np.asarray(
            Image.fromarray(bright).filter(ImageFilter.GaussianBlur(radius)),
            dtype=np.float32,
        ) / 255.0
        blur = blur * _BLOOM_TINTS.get(self.tone, _BLOOM_TINTS["warm"])  # halation follows tone
        return np.clip(1.0 - (1.0 - g) * (1.0 - self.bloom * blur), 0.0, 1.0)

    def _vignette_mask(self, shape: tuple) -> np.ndarray:
        h, w = shape[0], shape[1]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        d = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2) / np.sqrt(2.0)
        return (1.0 - self.vignette * np.clip(d, 0.0, 1.0) ** 2)[..., None]

    def _grain(self, shape: tuple) -> np.ndarray:
        rng = np.random.default_rng(_GRAIN_SEED)
        return rng.normal(0.0, self.grain, size=(shape[0], shape[1], 1)).astype(np.float32)


def _luma(a: np.ndarray) -> np.ndarray:
    return (0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2])[..., None]


def _clamp(v, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))
