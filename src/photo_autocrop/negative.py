"""Convert a cropped colour-negative frame to a positive.

The crop passed in is the image area only, with the pink film base
already excluded by the detection inset. We invert the raw pixels and
neutralise the orange/pink dye mask by sampling the darkest quantile in
each channel of the inverted image (those darkest values correspond to
the film-base colour bleeding into the image area) and treating that as
the true black point per channel.

Downstream ``auto_tone`` still runs on the result, which handles the
final per-channel stretch to the 99.5 percentile.
"""
from __future__ import annotations

import numpy as np


def negative_to_positive(bgr: np.ndarray) -> np.ndarray:
    """Invert a colour negative and remove the orange-mask cast.

    Steps:
    1. Invert BGR (positive = 255 - negative).
    2. Per channel, find the 1st percentile in the inverted image. This
       is the film-base colour projected into positive space. Subtract
       it as the channel black point, rescale to 0..255.

    The result is a colour-balanced positive with headroom preserved
    (99th percentile isn't clipped). ``auto_tone`` should be applied
    after this to bring midtones/highlights in line.
    """
    if bgr.dtype != np.uint8:
        return bgr
    inv = (255 - bgr).astype(np.float32)
    out = np.empty_like(inv)
    for c in range(inv.shape[-1]):
        channel = inv[..., c]
        black = float(np.percentile(channel, 1.0))
        span = 255.0 - black
        if span < 1.0:
            out[..., c] = channel
            continue
        rescaled = (channel - black) * (255.0 / span)
        out[..., c] = rescaled
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def gray_world_balance(bgr: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Scale each channel toward a shared mean.

    Removes the uniform-midtone colour cast that per-channel percentile
    stretching (``auto_tone``) leaves behind on scanned negatives. Meant
    to run after ``auto_tone`` so the extremes are already snapped.
    """
    if bgr.dtype != np.uint8:
        return bgr
    strength = float(max(0.0, min(1.0, strength)))
    if strength == 0.0:
        return bgr
    f = bgr.astype(np.float32)
    means = np.array([f[..., c].mean() for c in range(3)], dtype=np.float32)
    target = float(means.mean())
    out = np.empty_like(f)
    for c in range(3):
        m = means[c]
        if m < 1.0:
            out[..., c] = f[..., c]
            continue
        scale = 1.0 + (target / m - 1.0) * strength
        out[..., c] = f[..., c] * scale
    return np.clip(out, 0.0, 255.0).astype(np.uint8)
