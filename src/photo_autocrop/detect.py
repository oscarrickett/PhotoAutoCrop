from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


DETECT_MAX_EDGE = 800  # GrabCut is O(pixels); smaller is faster
GRABCUT_ITERATIONS = 4

MIN_AREA_FRACTION = 0.03
MAX_AREA_FRACTION = 0.97
FRAME_EDGE_MARGIN_PX = 6


@dataclass
class Rect:
    cx: float
    cy: float
    w: float
    h: float
    angle: float

    def as_cv(self) -> tuple[tuple[float, float], tuple[float, float], float]:
        return ((self.cx, self.cy), (self.w, self.h), self.angle)


@dataclass
class Detection:
    rect_small: Rect | None
    contour_small: np.ndarray | None
    scale: float
    small_shape: tuple[int, int]
    full_shape: tuple[int, int]
    mat_mask: np.ndarray | None
    photo_mask: np.ndarray | None
    notes: list[str]

    @property
    def found(self) -> bool:
        return self.rect_small is not None


def _resize_for_detection(bgr: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = bgr.shape[:2]
    long_edge = max(h, w)
    if long_edge <= DETECT_MAX_EDGE:
        return bgr, 1.0
    scale = DETECT_MAX_EDGE / long_edge
    new_size = (int(round(w * scale)), int(round(h * scale)))
    return cv2.resize(bgr, new_size, interpolation=cv2.INTER_AREA), scale


def _mat_mask(small_bgr: np.ndarray) -> np.ndarray:
    """Used only by scoring."""
    lab = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.int16) - 128
    b = lab[:, :, 2].astype(np.int16) - 128
    chroma = np.sqrt(a * a + b * b).astype(np.uint8)
    L = lab[:, :, 0]
    return (((chroma < 10) & (L > 40) & (L < 220)).astype(np.uint8)) * 255


def _photo_mask_auto(small_bgr: np.ndarray) -> np.ndarray:
    """Default GrabCut: 4% margin rect, runs purely on RGB statistics."""
    h, w = small_bgr.shape[:2]
    margin_x = int(w * 0.04)
    margin_y = int(h * 0.04)
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(small_bgr, mask, rect, bgd, fgd, GRABCUT_ITERATIONS, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return np.zeros((h, w), np.uint8)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    return fg


def _photo_mask_small(small_bgr: np.ndarray) -> np.ndarray:
    """For photos that are small relative to the mat. Seeds GrabCut with
    mat-hint = probable-background so the algorithm doesn't drag the mat
    into the foreground.
    """
    h, w = small_bgr.shape[:2]
    lab = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.int16) - 128
    b = lab[:, :, 2].astype(np.int16) - 128
    chroma = np.sqrt(a * a + b * b).astype(np.uint8)
    L = lab[:, :, 0]
    mat_hint = (chroma < 15) & (L > 45) & (L < 210)

    mask = np.full((h, w), cv2.GC_PR_FGD, np.uint8)
    mask[mat_hint] = cv2.GC_PR_BGD
    edge_px = max(3, int(min(h, w) * 0.012))
    mask[:edge_px, :] = cv2.GC_BGD
    mask[-edge_px:, :] = cv2.GC_BGD
    mask[:, :edge_px] = cv2.GC_BGD
    mask[:, -edge_px:] = cv2.GC_BGD

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(small_bgr, mask, None, bgd, fgd, GRABCUT_ITERATIONS, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return np.zeros((h, w), np.uint8)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    return fg


def _photo_mask_loose(small_bgr: np.ndarray) -> np.ndarray:
    """Looser GrabCut: minimal margin, more iterations. Useful when the
    default crop is slightly inside the true photo edges.
    """
    h, w = small_bgr.shape[:2]
    margin_x = int(w * 0.015)
    margin_y = int(h * 0.015)
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(small_bgr, mask, rect, bgd, fgd, GRABCUT_ITERATIONS + 2, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return np.zeros((h, w), np.uint8)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    return fg


def _photo_mask_edges(small_bgr: np.ndarray) -> np.ndarray:
    """Edge-based fallback: find the largest closed region in a Canny edge
    map. Useful when GrabCut's GMM gets confused but the photo has a
    visually crisp boundary.
    """
    h, w = small_bgr.shape[:2]
    gray = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, d=9, sigmaColor=80, sigmaSpace=80)
    med = float(np.median(blur))
    edges = cv2.Canny(blur, int(max(0, 0.5 * med)), int(min(255, 1.4 * med)), L2gradient=True)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    # Fill the edge map's interior so the photo region becomes a solid blob.
    flood = edges.copy()
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    filled = cv2.bitwise_not(flood) | edges
    filled = cv2.morphologyEx(
        filled, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    )
    filled = cv2.morphologyEx(
        filled, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    )
    return filled


METHODS = ("auto", "small", "loose", "edges")


def _photo_mask(small_bgr: np.ndarray, method: str = "auto") -> np.ndarray:
    if method == "small":
        return _photo_mask_small(small_bgr)
    if method == "loose":
        return _photo_mask_loose(small_bgr)
    if method == "edges":
        return _photo_mask_edges(small_bgr)
    return _photo_mask_auto(small_bgr)


def _touches_frame(contour: np.ndarray, shape: tuple[int, int], margin: int) -> bool:
    h, w = shape
    xs = contour[:, 0, 0]
    ys = contour[:, 0, 1]
    return bool(
        (xs.min() <= margin)
        or (ys.min() <= margin)
        or (xs.max() >= w - 1 - margin)
        or (ys.max() >= h - 1 - margin)
    )


def _select_contour(
    photo_mask: np.ndarray, small_shape: tuple[int, int]
) -> tuple[np.ndarray | None, list[str]]:
    notes: list[str] = []
    contours, _ = cv2.findContours(photo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        notes.append("no_contours")
        return None, notes
    h, w = small_shape
    frame_area = h * w
    min_area = MIN_AREA_FRACTION * frame_area
    max_area = MAX_AREA_FRACTION * frame_area
    ranked = sorted(contours, key=cv2.contourArea, reverse=True)
    saw_frame_toucher = False
    for c in ranked:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        if _touches_frame(c, small_shape, FRAME_EDGE_MARGIN_PX):
            saw_frame_toucher = True
            continue
        return c, notes
    if saw_frame_toucher:
        notes.append("only_frame_touchers")
    else:
        notes.append("no_qualifying_contour")
    return None, notes


def _normalize_rect_angle(cv_rect) -> Rect:
    (cx, cy), (w, h), angle = cv_rect
    if w < h:
        w, h = h, w
        angle = angle + 90.0
    while angle > 45:
        angle -= 90.0
    while angle < -45:
        angle += 90.0
    return Rect(cx=cx, cy=cy, w=w, h=h, angle=angle)


def _refine_rotation(small_bgr: np.ndarray, photo_mask: np.ndarray, fallback: float) -> float:
    """Replace the morphology-smoothed angle with one derived from real
    edges along the photo's border.

    Build a thin band along the boundary of the GrabCut foreground, mask
    Canny edges to that band, then take the weighted-median orientation
    of long Hough line segments. The boundary band excludes photo-interior
    content (cabinets, faces, etc.) so only the actual photo edges vote.
    """
    eroded = cv2.erode(
        photo_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    )
    band = cv2.subtract(photo_mask, eroded)
    if band.sum() == 0:
        return fallback

    gray = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.2)
    canny = cv2.Canny(blur, 40, 130, L2gradient=True)
    masked = cv2.bitwise_and(canny, canny, mask=band)

    h, w = small_bgr.shape[:2]
    min_len = max(40, int(0.20 * min(h, w)))
    lines = cv2.HoughLinesP(
        masked,
        rho=1,
        theta=np.pi / 720,
        threshold=40,
        minLineLength=min_len,
        maxLineGap=20,
    )
    if lines is None or len(lines) < 2:
        return fallback

    angles: list[float] = []
    weights: list[float] = []
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        if x2 == x1:
            theta = 90.0
        else:
            theta = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        # Fold to [-45, 45] so horizontal and vertical edges contribute to
        # the same angle (we want the rotation that makes either set
        # axis-aligned).
        while theta > 45:
            theta -= 90.0
        while theta < -45:
            theta += 90.0
        length = float(np.hypot(x2 - x1, y2 - y1))
        angles.append(theta)
        weights.append(length)

    a = np.array(angles)
    wts = np.array(weights)
    order = np.argsort(a)
    cum = np.cumsum(wts[order])
    if cum[-1] <= 0:
        return fallback
    idx = int(np.searchsorted(cum, cum[-1] / 2))
    idx = min(idx, len(order) - 1)
    refined = float(a[order[idx]])
    # Sanity: if refined is wildly different from fallback, distrust it.
    if abs(refined - fallback) > 20:
        return fallback
    return refined


def detect(bgr: np.ndarray, method: str = "auto") -> Detection:
    full_shape = (bgr.shape[0], bgr.shape[1])
    small, scale = _resize_for_detection(bgr)
    small_shape = (small.shape[0], small.shape[1])
    mat = _mat_mask(small)
    photo = _photo_mask(small, method=method)
    contour, notes = _select_contour(photo, small_shape)
    if contour is None:
        return Detection(
            rect_small=None, contour_small=None, scale=scale,
            small_shape=small_shape, full_shape=full_shape,
            mat_mask=mat, photo_mask=photo, notes=notes,
        )
    rect = _normalize_rect_angle(cv2.minAreaRect(contour))
    refined_angle = _refine_rotation(small, photo, rect.angle)
    rect = Rect(cx=rect.cx, cy=rect.cy, w=rect.w, h=rect.h, angle=refined_angle)
    return Detection(
        rect_small=rect, contour_small=contour, scale=scale,
        small_shape=small_shape, full_shape=full_shape,
        mat_mask=mat, photo_mask=photo, notes=notes,
    )
