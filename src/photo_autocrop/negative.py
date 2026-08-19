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
