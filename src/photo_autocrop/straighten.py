from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .detect import Detection, Rect


# Fraction of the detected rect to keep — 0.975 means the saved crop is
# 2.5% smaller on each axis than the detected edge. This compensates for
# detection slop along the mat boundary so the output doesn't include a
# sliver of grey mat at the photo's edge.
CROP_SHRINK = 0.975
# Slides have a softer photo→mount transition (anti-aliased film edge),
# so we bite further inside to avoid a thin dark border in the output.
CROP_SHRINK_SLIDE = 0.945
# Negatives: the detected rect encloses the entire film area including
# the sprocket-hole strip on the two long edges (35mm sprockets run
# along the frame's long edges, so along the short axis of the rect
# they eat significantly into the crop). Shrink the short axis much
# more than the long axis.
CROP_SHRINK_NEGATIVE_LONG = 0.965
CROP_SHRINK_NEGATIVE_SHORT = 0.80


def shrink_for_method(method: str) -> float | tuple[float, float]:
    """Return the crop shrink for a detection method. Most methods use
    a single uniform shrink factor; negatives use asymmetric (long,
    short) because the sprocket strip only eats into the short axis.
    """
    if method == "slide":
        return CROP_SHRINK_SLIDE
    if method == "negative":
        return (CROP_SHRINK_NEGATIVE_LONG, CROP_SHRINK_NEGATIVE_SHORT)
    return CROP_SHRINK


@dataclass
class CropBox:
    cx: float
    cy: float
    w: float
    h: float

    def to_dict(self) -> dict:
        return {"cx": self.cx, "cy": self.cy, "w": self.w, "h": self.h}


@dataclass
class StraightenResult:
    cropped: np.ndarray
    rotation_deg: float
    crop_box: CropBox


def apply_quarter_turns(bgr: np.ndarray, k: int) -> np.ndarray:
    """Rotate by k * 90° clockwise. k is taken mod 4."""
    k = int(k) % 4
    if k == 0:
        return bgr
    if k == 1:
        return cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    if k == 2:
        return cv2.rotate(bgr, cv2.ROTATE_180)
    return cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)


def _scale_rect(rect: Rect, scale: float, shrink) -> Rect:
    """Scale rect from small-frame to full-frame coords, optionally
    shrinking. ``shrink`` is either a single float (applied to both
    axes) or a (long, short) tuple applied along the rect's own long
    and short dimensions.
    """
    if isinstance(shrink, tuple):
        shrink_long, shrink_short = shrink
        if rect.w >= rect.h:
            sw, sh = shrink_long, shrink_short
        else:
            sw, sh = shrink_short, shrink_long
    else:
        sw = sh = shrink
    w_small = max(1.0, rect.w * sw)
    h_small = max(1.0, rect.h * sh)
    return Rect(
        cx=rect.cx / scale,
        cy=rect.cy / scale,
        w=w_small / scale,
        h=h_small / scale,
        angle=rect.angle,
    )


def _rotate_around_image_center(
    bgr: np.ndarray, angle_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    fh, fw = bgr.shape[:2]
    center = (fw / 2.0, fh / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        bgr,
        M,
        (fw, fh),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, M


def _apply_affine_to_point(M: np.ndarray, x: float, y: float) -> tuple[float, float]:
    nx = M[0, 0] * x + M[0, 1] * y + M[0, 2]
    ny = M[1, 0] * x + M[1, 1] * y + M[1, 2]
    return float(nx), float(ny)


def _crop_to_box(bgr: np.ndarray, box: CropBox) -> np.ndarray | None:
    fh, fw = bgr.shape[:2]
    half_w = box.w / 2.0
    half_h = box.h / 2.0
    x1 = max(0, int(round(box.cx - half_w)))
    y1 = max(0, int(round(box.cy - half_h)))
    x2 = min(fw, int(round(box.cx + half_w)))
    y2 = min(fh, int(round(box.cy + half_h)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return bgr[y1:y2, x1:x2].copy()


def straighten_and_crop(
    bgr: np.ndarray,
    detection: Detection,
    shrink=CROP_SHRINK,
) -> StraightenResult | None:
    if not detection.found or detection.rect_small is None:
        return None
    rect_full = _scale_rect(detection.rect_small, detection.scale, shrink)
    rotated, M = _rotate_around_image_center(bgr, rect_full.angle)
    cx_new, cy_new = _apply_affine_to_point(M, rect_full.cx, rect_full.cy)
    rotated_box = CropBox(cx=cx_new, cy=cy_new, w=rect_full.w, h=rect_full.h)
    cropped = _crop_to_box(rotated, rotated_box)
    if cropped is None:
        return None
    # Manifest stores the crop box in ORIGINAL (un-rotated) image coords,
    # so the editor can render it in the same space as detected_corners.
    # `apply_user_edit` converts back to rotated-frame coords at execution.
    box_original = CropBox(
        cx=rect_full.cx, cy=rect_full.cy, w=rect_full.w, h=rect_full.h
    )
    return StraightenResult(cropped=cropped, rotation_deg=rect_full.angle, crop_box=box_original)


def apply_user_edit(
    bgr: np.ndarray,
    rotation_deg: float,
    crop_box: CropBox,
    upright_qt: int = 0,
) -> np.ndarray | None:
    """Apply the user's saved crop_box, treating cx/cy as ORIGINAL image
    coords. We rotate the image first, then map the un-rotated center
    through the rotation to find where to crop.
    """
    rotated, M = _rotate_around_image_center(bgr, rotation_deg)
    cx_rot, cy_rot = _apply_affine_to_point(M, crop_box.cx, crop_box.cy)
    rotated_box = CropBox(cx=cx_rot, cy=cy_rot, w=crop_box.w, h=crop_box.h)
    cropped = _crop_to_box(rotated, rotated_box)
    if cropped is None:
        return None
    return apply_quarter_turns(cropped, upright_qt)
