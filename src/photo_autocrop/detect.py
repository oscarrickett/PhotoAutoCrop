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
    # True when the rect already matches the target crop exactly (no
    # need to apply the method's shrink). Set by negative sprocket
    # detection where we know the image bounds precisely.
    precise_crop: bool = False

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


def _photo_mask_negative(small_bgr: np.ndarray) -> np.ndarray:
    """For scanned colour negatives: pink film rectangle on the black
    scanner bed. Luminance alone fails because dark interior scenes
    approach the bed's brightness. Instead, threshold on the LAB
    a-channel — the film base's orange-pink mask sits well into the
    positive-a (red) region, while the scanner bed is neutral. Then
    close aggressively so neutral patches inside the image (blue sky,
    grey walls) get filled back in as part of the film blob.
    """
    h, w = small_bgr.shape[:2]
    lab = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    a = lab[:, :, 1].astype(np.int16) - 128
    # Pink cast AND not pure-black bed. a > 6 catches the film base
    # everywhere the orange mask is visible; L > 25 excludes the bed
    # without cutting into dim image interior.
    mask = (((a > 6) & (L > 25)).astype(np.uint8)) * 255
    if mask.sum() // 255 < 0.05 * h * w:
        return np.zeros((h, w), np.uint8)
    # Modest close: fill neutral interior patches (blue sky → cyan in
    # the raw negative → drops out of the a-channel mask) without
    # dilating the blob past the film's true boundary. Larger kernels
    # bleed the mask out to the scan edges when the bed border is thin.
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    )
    return mask


def _photo_mask_slide(small_bgr: np.ndarray) -> np.ndarray:
    """For scanned slides: bright image rectangle in a dark mount surround.

    1. Estimate the mount luminance level from the 15th-percentile of L
       (the mount typically covers >15% of the slide).
    2. Mark every pixel brighter than mount+margin as candidate photo.
    3. Close small holes (interior dark scene content), then OPEN with
       an 11px ellipse — this removes the thin (1-5 px) scanner
       reflection ring that fooled GrabCut while leaving the large
       photo blob intact.
    4. Returns an empty mask if mount level is too bright (not a slide),
       so the caller falls through to another method.
    """
    h, w = small_bgr.shape[:2]
    lab = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    mount_level = float(np.percentile(L, 15))
    # Mount must actually be dark — bail on light backgrounds (grey mat).
    if mount_level > 50:
        return np.zeros((h, w), np.uint8)
    # Threshold well above mount level so we skip the soft transition
    # band at the photo's true edge (anti-aliased film/scanner boundary
    # that the user sees as a "faint line" around the real photo).
    above = (L > mount_level + 20).astype(np.uint8) * 255
    # Need a meaningful amount of "above-mount" pixels.
    if above.sum() // 255 < 0.05 * h * w:
        return np.zeros((h, w), np.uint8)
    # Fill small interior holes (dark photo content).
    above = cv2.morphologyEx(
        above, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    )
    # Kill the thin scanner-reflection ring (3-5 px wide) without
    # eroding the photo blob too much.
    above = cv2.morphologyEx(
        above, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    )
    return above


METHODS = ("auto", "small", "loose", "edges", "slide", "negative")


def _photo_mask(small_bgr: np.ndarray, method: str = "auto") -> np.ndarray:
    if method == "small":
        return _photo_mask_small(small_bgr)
    if method == "loose":
        return _photo_mask_loose(small_bgr)
    if method == "edges":
        return _photo_mask_edges(small_bgr)
    if method == "slide":
        return _photo_mask_slide(small_bgr)
    if method == "negative":
        return _photo_mask_negative(small_bgr)
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
    photo_mask: np.ndarray,
    small_shape: tuple[int, int],
    allow_frame_touchers: bool = False,
    max_area_fraction: float = MAX_AREA_FRACTION,
) -> tuple[np.ndarray | None, list[str]]:
    notes: list[str] = []
    contours, _ = cv2.findContours(photo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        notes.append("no_contours")
        return None, notes
    h, w = small_shape
    frame_area = h * w
    min_area = MIN_AREA_FRACTION * frame_area
    max_area = max_area_fraction * frame_area
    ranked = sorted(contours, key=cv2.contourArea, reverse=True)
    saw_frame_toucher = False
    for c in ranked:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        if _touches_frame(c, small_shape, FRAME_EDGE_MARGIN_PX):
            saw_frame_toucher = True
            if allow_frame_touchers:
                return c, notes
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

    # OpenCV 4.x returns lines as shape (N, 1, 4); OpenCV 5.x drops the
    # middle dim to (N, 4). Flatten so ln unpacks the same way either way.
    lines = np.asarray(lines).reshape(-1, 4)
    angles: list[float] = []
    weights: list[float] = []
    for ln in lines:
        x1, y1, x2, y2 = ln
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


def _refine_inside_sprocket_rect(
    small_bgr: np.ndarray, outer: Rect
) -> tuple[Rect | None, list[str]]:
    """Second-pass detection: crop the sprocket-detected film area,
    invert to positive, and locate the image's inner boundary by
    projection. If a clear inner rect is found and it's meaningfully
    smaller than the outer, return it; otherwise return None so the
    caller keeps the sprocket rect.

    Uses column/row projection on the a-channel of the inverted crop
    to detect the transition from the residual film-base border
    (dark and cyan-tinted after inversion) to the image content
    (variable but generally more chromatic).
    """
    from .negative import negative_to_positive
    h, w = small_bgr.shape[:2]
    # Extract axis-aligned crop around the outer rect.
    half_w = outer.w / 2.0; half_h = outer.h / 2.0
    x1 = int(max(0, round(outer.cx - half_w)))
    y1 = int(max(0, round(outer.cy - half_h)))
    x2 = int(min(w, round(outer.cx + half_w)))
    y2 = int(min(h, round(outer.cy + half_h)))
    if x2 - x1 < 40 or y2 - y1 < 40:
        return None, ["inner_crop_too_small"]
    crop = small_bgr[y1:y2, x1:x2]
    positive = negative_to_positive(crop)
    ph, pw = positive.shape[:2]
    # Residual film base after inversion is a low-saturation dark
    # cyan strip; image content is more chromatic. Threshold on
    # LAB chroma to separate image from border, then take the largest
    # contour and its axis-aligned bounding rect.
    lab = cv2.cvtColor(positive, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.int16) - 128
    b = lab[:, :, 2].astype(np.int16) - 128
    chroma = np.sqrt(a * a + b * b).astype(np.uint8)
    # Also flag "meaningfully bright" pixels to catch scene highlights
    # that might be near-neutral (whites, greys) — those are image too.
    L = lab[:, :, 0]
    fg = ((chroma > 12) | (L > 90)).astype(np.uint8) * 255
    fg = cv2.morphologyEx(
        fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    )
    fg = cv2.morphologyEx(
        fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    )
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, ["inner_no_contour"]
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 0.3 * ph * pw:
        return None, ["inner_contour_too_small"]
    x, y, bw, bh = cv2.boundingRect(largest)

    # Only trim if the refinement actually shrinks the rect. Reject
    # tiny gains (<2% each side) and absurd shrinks (>25% each side).
    inset_top = y; inset_bot = ph - (y + bh)
    inset_left = x; inset_right = pw - (x + bw)
    max_edge = min(ph, pw)
    if max(inset_top, inset_bot, inset_left, inset_right) < 0.02 * max_edge:
        return None, ["inner_edges_already_tight"]
    if min(inset_top, inset_bot, inset_left, inset_right) > 0.25 * max_edge:
        return None, ["inner_edges_shrank_too_much"]

    new_w = float(bw); new_h = float(bh)
    new_cx = x1 + x + bw / 2.0
    new_cy = y1 + y + bh / 2.0
    return Rect(
        cx=new_cx, cy=new_cy,
        w=max(new_w, new_h), h=min(new_w, new_h),
        angle=outer.angle,
    ), ["inner_edges_refined"]


def _detect_from_sprockets(small_bgr: np.ndarray) -> tuple[Rect | None, list[str]]:
    """Find sprocket-hole strips on a scanned negative via 1D
    projection and return the image Rect between them.

    Approach: the film has two bright strips running along its long
    edges (the sprocket rows), separated by a darker image band. In
    the row-mean projection this appears as two prominent peaks
    flanking a broad valley. Same in reverse for the col-mean when
    the film is scanned in portrait orientation. Projection is much
    more robust than per-hole blob detection because it averages away
    per-hole brightness variation.
    """
    h, w = small_bgr.shape[:2]
    gray = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    def _find_strip_peaks(proj: np.ndarray, length: int) -> tuple[int, int, int, int] | None:
        """Locate top/bottom sprocket-strip peaks. Returns (top_peak,
        top_inner_edge, bottom_inner_edge, bottom_peak) or None.
        Inner edges are where the projection drops below the midpoint
        between peak brightness and interior brightness — that's the
        transition from sprocket strip to image band.
        """
        # Search windows: peaks live in the outer ~15% of the film's
        # long dimension. Bed is even further out (very dark), so a
        # generous window is fine — we take argmax within it.
        window = max(20, length // 6)
        top_peak = int(np.argmax(proj[:window]))
        bot_peak = int(length - window + np.argmax(proj[length - window:]))
        top_val = float(proj[top_peak])
        bot_val = float(proj[bot_peak])
        # Interior brightness = median of the mid 60%.
        interior = float(np.median(proj[length // 5 : 4 * length // 5]))
        # Peaks must be substantially brighter than interior.
        if top_val - interior < 30 or bot_val - interior < 30:
            return None
        # Bail if peaks are absurdly close (< 40% of length apart —
        # can't be film + image + film).
        if bot_peak - top_peak < 0.4 * length:
            return None
        # Inner edge: first row after each peak where the projection
        # falls at least halfway back to interior.
        top_mid = (top_val + interior) / 2.0
        bot_mid = (bot_val + interior) / 2.0
        top_inner = top_peak
        for i in range(top_peak, length):
            if proj[i] <= top_mid:
                top_inner = i
                break
        bot_inner = bot_peak
        for i in range(bot_peak, -1, -1):
            if proj[i] <= bot_mid:
                bot_inner = i
                break
        if bot_inner <= top_inner + 20:
            return None
        return top_peak, top_inner, bot_inner, bot_peak

    row_proj = gray.mean(axis=1)
    col_proj = gray.mean(axis=0)
    row_result = _find_strip_peaks(row_proj, h)
    col_result = _find_strip_peaks(col_proj, w)

    # Pick landscape (sprockets on top/bottom = row peaks) if row peaks
    # are stronger, else portrait. Prefer landscape when both work
    # and film width > film height in the found rect.
    def strength(proj: np.ndarray, result: tuple[int, int, int, int]) -> float:
        if result is None:
            return 0.0
        tp, _, _, bp = result
        interior_mid = float(np.median(proj[len(proj) // 5 : 4 * len(proj) // 5]))
        return (float(proj[tp]) + float(proj[bp])) / 2.0 - interior_mid

    row_strength = strength(row_proj, row_result)
    col_strength = strength(col_proj, col_result)

    if row_result is None and col_result is None:
        return None, ["no_sprocket_projection_peaks"]

    landscape = row_strength >= col_strength

    if landscape and row_result is not None:
        _, top_inner, bot_inner, _ = row_result
        image_top = float(top_inner)
        image_bottom = float(bot_inner)
        # Left/right image edges: use col projection but look for the
        # transition from bed (very dark) to film (bright), not
        # sprocket peaks. The film's L is stable across the whole
        # width until it drops off into the black bed.
        bed_level = float(np.percentile(col_proj, 10))
        film_level = float(np.median(col_proj))
        thresh = (bed_level + film_level) / 2.0
        left = 0
        for i in range(len(col_proj)):
            if col_proj[i] > thresh:
                left = i
                break
        right = len(col_proj) - 1
        for i in range(len(col_proj) - 1, -1, -1):
            if col_proj[i] > thresh:
                right = i
                break
        image_left = float(left)
        image_right = float(right)
    elif col_result is not None:
        _, left_inner, right_inner, _ = col_result
        image_left = float(left_inner)
        image_right = float(right_inner)
        bed_level = float(np.percentile(row_proj, 10))
        film_level = float(np.median(row_proj))
        thresh = (bed_level + film_level) / 2.0
        top = 0
        for i in range(len(row_proj)):
            if row_proj[i] > thresh:
                top = i
                break
        bot = len(row_proj) - 1
        for i in range(len(row_proj) - 1, -1, -1):
            if row_proj[i] > thresh:
                bot = i
                break
        image_top = float(top)
        image_bottom = float(bot)
    else:
        return None, ["no_sprocket_projection_peaks"]

    rw = image_right - image_left
    rh = image_bottom - image_top
    if rw <= 20 or rh <= 20:
        return None, ["sprocket_rect_too_small"]

    cx = (image_left + image_right) / 2.0
    cy = (image_top + image_bottom) / 2.0
    # Angle is left as 0 here — the caller refines it via the film-mask
    # boundary (Hough lines), which is much more robust than trying to
    # fit through discrete sprocket-hole peak positions.
    return Rect(cx=cx, cy=cy, w=max(rw, rh), h=min(rw, rh), angle=0.0), []


def detect(bgr: np.ndarray, method: str = "auto") -> Detection:
    full_shape = (bgr.shape[0], bgr.shape[1])
    small, scale = _resize_for_detection(bgr)
    small_shape = (small.shape[0], small.shape[1])
    mat = _mat_mask(small)
    photo = _photo_mask(small, method=method)

    if method == "negative":
        sprocket_rect, sprocket_notes = _detect_from_sprockets(small)
        if sprocket_rect is not None:
            # Second pass: within the sprocket-detected film area, run
            # inner-edge detection on the inverted (positive) crop to
            # find the actual image boundary and shed any residual film
            # base still hugging the crop.
            refined_rect, refine_notes = _refine_inside_sprocket_rect(
                small, sprocket_rect
            )
            if refined_rect is not None:
                sprocket_rect = refined_rect
                sprocket_notes = list(sprocket_notes) + refine_notes
            # Refine rotation from the film-mask boundary — the per-
            # column peak-fit inside the sprocket strip is noisy
            # because sprocket holes are discrete features. The mask
            # boundary is a continuous edge that Hough handles well.
            mask_angle = _refine_rotation(small, photo, sprocket_rect.angle)
            # Projection gives axis-aligned bounds; cv2.boxPoints treats
            # (w, h) as the tilted rect's OWN dimensions. Convert bounds
            # → true dims so the drawn corners match the film.
            th = float(np.radians(mask_angle))
            c = float(np.cos(th)); s = float(abs(np.sin(th)))
            det = c * c - s * s
            if abs(det) > 0.001:
                true_w = (sprocket_rect.w * c - sprocket_rect.h * s) / det
                true_h = (sprocket_rect.h * c - sprocket_rect.w * s) / det
            else:
                true_w, true_h = sprocket_rect.w, sprocket_rect.h
            refined = Rect(
                cx=sprocket_rect.cx, cy=sprocket_rect.cy,
                w=max(10.0, true_w), h=max(10.0, true_h),
                angle=mask_angle,
            )
            return Detection(
                rect_small=refined, contour_small=None, scale=scale,
                small_shape=small_shape, full_shape=full_shape,
                mat_mask=mat, photo_mask=photo,
                notes=["sprockets_detected", *sprocket_notes],
                precise_crop=True,
            )
    # Negatives fill almost the entire scan with just a thin scanner-bed
    # border — the standard MAX_AREA_FRACTION=0.97 rejects them. Slide
    # mounts also tend to be tight-cropped by the scanner. Both need the
    # ceiling lifted and frame-touchers permitted.
    contour, notes = _select_contour(
        photo,
        small_shape,
        allow_frame_touchers=method in ("slide", "negative"),
        max_area_fraction=0.998 if method == "negative" else MAX_AREA_FRACTION,
    )
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
