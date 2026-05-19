from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .detect import Detection


APPROVED_THRESHOLD = 80
NEEDS_REVIEW_THRESHOLD = 50


@dataclass
class ScoreResult:
    total: int
    bucket: str
    signals: dict[str, int] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


def _area_fraction_score(area_frac: float) -> int:
    if 0.08 <= area_frac <= 0.70:
        return 20
    if 0.04 <= area_frac < 0.08 or 0.70 < area_frac <= 0.85:
        return 10
    return 0


def _aspect_ratio_score(longer: float, shorter: float) -> int:
    if shorter <= 0:
        return 0
    ratio = longer / shorter
    if 1.2 <= ratio <= 2.0:
        return 15
    if 1.0 <= ratio < 1.2 or 2.0 < ratio <= 2.5:
        return 8
    return 0


def _solidity_score(contour: np.ndarray) -> int:
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return 0
    solidity = area / hull_area
    if solidity > 0.92:
        return 20
    if solidity > 0.85:
        return 10
    return 0


def _rotation_score(angle: float) -> int:
    mag = abs(angle)
    if mag < 30:
        return 10
    if mag < 40:
        return 5
    return 0


def _l_separation_score(
    small_bgr: np.ndarray, mat_mask: np.ndarray, contour: np.ndarray
) -> int:
    """Mean L (lightness) inside the contour vs. outside on the mat.

    A real photo usually differs in lightness from the mat. If they match,
    we may be looking at a same-tone shadow blob.
    """
    fg_mask = np.zeros(small_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(fg_mask, [contour], -1, color=255, thickness=cv2.FILLED)
    lab = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    inside = L[fg_mask > 0]
    outside = L[(mat_mask > 0) & (fg_mask == 0)]
    if inside.size < 50 or outside.size < 50:
        return 0
    diff = abs(float(inside.mean()) - float(outside.mean()))
    if diff > 25:
        return 15
    if diff > 12:
        return 8
    return 0


def _edge_straightness_score(contour: np.ndarray, cv_rect) -> int:
    """How well the contour points hug the minAreaRect's four sides."""
    box = cv2.boxPoints(cv_rect)
    pts = contour.reshape(-1, 2).astype(np.float32)
    if pts.shape[0] < 8:
        return 0
    residuals = []
    for i in range(4):
        a = box[i]
        b = box[(i + 1) % 4]
        edge = b - a
        edge_len = np.linalg.norm(edge)
        if edge_len < 1.0:
            continue
        normal = np.array([-edge[1], edge[0]]) / edge_len
        d = np.abs((pts - a) @ normal)
        t = (pts - a) @ (edge / edge_len)
        mask = (t >= 0) & (t <= edge_len)
        if mask.sum() < 4:
            continue
        residuals.append(float(d[mask].mean()))
    if not residuals:
        return 0
    mean_residual = float(np.mean(residuals))
    if mean_residual < 3.0:
        return 15
    if mean_residual < 6.0:
        return 8
    return 0


def _frame_border_score(detection: Detection) -> int:
    if "largest_touches_frame" in detection.notes:
        return 0
    return 5


def score_detection(detection: Detection, small_bgr: np.ndarray) -> ScoreResult:
    result = ScoreResult(total=0, bucket="rejected")
    if not detection.found or detection.contour_small is None or detection.rect_small is None:
        result.reasons.append("no_detection")
        return result

    contour = detection.contour_small
    rect = detection.rect_small
    h, w = detection.small_shape
    area_frac = cv2.contourArea(contour) / float(h * w)
    longer = max(rect.w, rect.h)
    shorter = min(rect.w, rect.h)

    cv_rect = ((rect.cx, rect.cy), (rect.w, rect.h), rect.angle)

    signals = {
        "area_fraction": _area_fraction_score(area_frac),
        "aspect_ratio": _aspect_ratio_score(longer, shorter),
        "solidity": _solidity_score(contour),
        "rotation": _rotation_score(rect.angle),
        "l_separation": _l_separation_score(small_bgr, detection.mat_mask, contour),
        "edge_straightness": _edge_straightness_score(contour, cv_rect),
        "frame_border": _frame_border_score(detection),
    }
    total = int(sum(signals.values()))

    if total >= APPROVED_THRESHOLD:
        bucket = "approved"
    elif total >= NEEDS_REVIEW_THRESHOLD:
        bucket = "needs-review"
    else:
        bucket = "rejected"

    result.total = total
    result.bucket = bucket
    result.signals = signals
    if signals["solidity"] < 20:
        result.reasons.append("low_solidity")
    if signals["aspect_ratio"] == 0:
        result.reasons.append("unusual_aspect")
    if signals["l_separation"] == 0:
        result.reasons.append("weak_contrast_vs_mat")
    return result
