"""Tests for the post-grade Grader (src/postprocess/grader.py)."""
from __future__ import annotations

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from src.postprocess.grader import Grader


def _save_color(path, text=None):
    a = np.zeros((48, 48, 3), dtype=np.uint8)
    a[:, :, 0] = 210; a[:, :, 1] = 90; a[:, :, 2] = 25   # saturated orange
    info = None
    if text:
        info = PngInfo()
        for k, v in text.items():
            info.add_text(k, v)
    Image.fromarray(a, "RGB").save(path, pnginfo=info)


def _saturation(path):
    im = np.asarray(Image.open(path).convert("RGB")).astype(int)
    return float(np.mean(np.max(im, 2) - np.min(im, 2)))


def test_apply_monochrome_produces_true_grayscale(tmp_path):
    """2026-06-26: monochrome_fine_art rendered muted COLOUR despite 'monochrome'
    prompts; apply_monochrome deterministically desaturates to true B&W (R==G==B)."""
    p = tmp_path / "c.png"
    _save_color(p)
    assert _saturation(p) > 100          # genuinely colourful to start
    Grader({}).apply_monochrome(p, p)
    assert _saturation(p) <= 1.0, "output is not true grayscale"


def test_apply_monochrome_preserves_png_text_chunks(tmp_path):
    """Render parameters (PNG text) must survive the desaturation."""
    p = tmp_path / "c.png"
    _save_color(p, text={"seed": "12345", "prompt": "a test prompt"})
    Grader({}).apply_monochrome(p, p)
    with Image.open(p) as im:
        assert (im.text or {}).get("seed") == "12345"


def test_color_grade_stays_colour(tmp_path):
    """Sanity: the normal colour grade does NOT desaturate (only apply_monochrome does)."""
    p = tmp_path / "c.png"
    _save_color(p)
    Grader({"strength": 0.6}).apply(p, p)
    assert _saturation(p) > 20, "colour grade should keep colour"


def _mean_r_minus_b(path):
    im = np.asarray(Image.open(path).convert("RGB")).astype(float)
    return im[..., 0].mean() - im[..., 2].mean()


def test_grade_tone_leans_the_finish(tmp_path):
    """2026-06-27 per-family grade: tone steers the split-tone so the finish matches
    mood — warm leans amber (R>B), cool leans blue (B>R), neutral stays ~balanced."""
    import numpy as np
    a = np.tile(np.linspace(40, 210, 48).astype(np.uint8)[:, None, None], (1, 48, 3))
    leans = {}
    for tone in ("warm", "cool", "neutral"):
        p = tmp_path / f"{tone}.png"
        Image.fromarray(a, "RGB").save(p)
        Grader({"strength": 1.0, "warmth": 0.2, "tone": tone}).apply(p, p)
        leans[tone] = _mean_r_minus_b(p)
    assert leans["warm"] > leans["neutral"] > leans["cool"], leans
    assert leans["cool"] < 0 < leans["warm"], leans   # cool is blue, warm is amber
