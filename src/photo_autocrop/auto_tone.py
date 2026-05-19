"""Photoshop-style Auto Tone for scanned photos.

Stretches each colour channel so its 0.5–99.5 percentile maps to 0–255.
This removes the dull-haze and age-yellowing that scans of old prints
almost always have, while ignoring outliers (specular highlights, dust
spots, scratches) at the extremes.
"""
from __future__ import annotations

import cv2
import numpy as np


def pull_highlights(bgr: np.ndarray, amount: float) -> np.ndarray:
    """Soft highlight rolloff. ``amount`` ∈ [0, 1].

    Values below mid-grey (128) stay as-is. Values above 128 are scaled
    toward the pivot — at ``amount = 1.0`` the upper half is halved
    (so 255 → ~191). Lets the user rescue blown-out faces without
    affecting shadows or midtones.
    """
    if amount <= 0:
        return bgr
    amount = float(min(1.0, max(0.0, amount)))
    pivot = 128.0
    compress = 1.0 - amount * 0.5
    f = bgr.astype(np.float32)
    above = np.maximum(f - pivot, 0.0)
    out = np.where(f < pivot, f, pivot + above * compress)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def auto_tone(
    bgr: np.ndarray,
    low_pct: float = 0.5,
    high_pct: float = 99.5,
) -> np.ndarray:
    """Per-channel histogram stretch — equivalent to Photoshop's Auto Tone."""
    if bgr.dtype != np.uint8:
        return bgr
    out = np.empty_like(bgr)
    flat = bgr.reshape(-1, bgr.shape[-1])
    for c in range(bgr.shape[-1]):
        channel = flat[:, c]
        lo = float(np.percentile(channel, low_pct))
        hi = float(np.percentile(channel, high_pct))
        if hi - lo < 1.0:
            out[..., c] = bgr[..., c]
            continue
        scale = 255.0 / (hi - lo)
        scaled = (bgr[..., c].astype(np.float32) - lo) * scale
        out[..., c] = np.clip(scaled, 0, 255).astype(np.uint8)
    return out
