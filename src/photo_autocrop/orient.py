from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np


ORIENT_MAX_EDGE = 600  # Haar is slow at full res; downscale for the vote.
MIN_FACE_AREA_FRAC = 0.005  # smallest accepted face vs. image area
HAAR_FRONTAL = "haarcascade_frontalface_default.xml"
HAAR_PROFILE = "haarcascade_profileface.xml"


@lru_cache(maxsize=1)
def _frontal_cascade() -> cv2.CascadeClassifier | None:
    path = cv2.data.haarcascades + HAAR_FRONTAL
    c = cv2.CascadeClassifier(path)
    return None if c.empty() else c


@lru_cache(maxsize=1)
def _profile_cascade() -> cv2.CascadeClassifier | None:
    path = cv2.data.haarcascades + HAAR_PROFILE
    c = cv2.CascadeClassifier(path)
    return None if c.empty() else c


def _score_orientation(gray: np.ndarray, min_face_area: float) -> float:
    """Sum of face areas in this gray image (frontal + profile detectors).
    Profiles are worth half because they fire on both upright and 180°.
    """
    score = 0.0
    frontal = _frontal_cascade()
    if frontal is not None:
        faces = frontal.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=5, minSize=(30, 30)
        )
        for (_, _, w, h) in faces:
            area = float(w * h)
            if area >= min_face_area:
                score += area
    profile = _profile_cascade()
    if profile is not None:
        faces = profile.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=5, minSize=(30, 30)
        )
        for (_, _, w, h) in faces:
            area = float(w * h)
            if area >= min_face_area:
                score += area * 0.5
    return score


def detect_upright_quarter_turns(bgr: np.ndarray) -> int:
    """Return the number of 90° CW rotations needed to make faces upright.

    0 means already upright (or no signal). 1 = rotate CW once, etc.

    Uses Haar frontal+profile cascades, which only fire on upright faces.
    We test all four 90° rotations and pick the one with the largest
    total face area. Returns 0 when no orientation produces a confident
    detection — better to leave the auto-straighten output alone than to
    rotate based on noise.
    """
    if bgr is None or bgr.size == 0:
        return 0

    h, w = bgr.shape[:2]
    long_edge = max(h, w)
    if long_edge > ORIENT_MAX_EDGE:
        scale = ORIENT_MAX_EDGE / long_edge
        small = cv2.resize(
            bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    else:
        small = bgr

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    sh, sw = gray.shape
    min_face_area = MIN_FACE_AREA_FRAC * sh * sw

    best_k = 0
    best_score = 0.0
    for k in range(4):
        # np.rot90 with negative k rotates CW; k=1 -> CW once.
        rotated = np.ascontiguousarray(np.rot90(gray, -k)) if k else gray
        s = _score_orientation(rotated, min_face_area)
        if s > best_score:
            best_score = s
            best_k = k

    return best_k if best_score > 0 else 0
