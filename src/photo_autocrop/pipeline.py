from __future__ import annotations

import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .detect import detect
from .io_utils import (
    SUPPORTED_EXTENSIONS,
    list_input_images,
    load_image,
    save_jpeg,
    save_thumbnail,
)
from .auto_tone import auto_tone
from .negative import gray_world_balance, negative_to_positive
from .orient import detect_upright_quarter_turns
from .schemas import CropBoxModel, Manifest, ManifestEntry
from .score import score_detection
from .straighten import apply_quarter_turns, shrink_for_method, straighten_and_crop


BUCKETS = ("approved", "needs-review", "rejected")  # manifest decision enum
MANIFEST_NAME = "_manifest.json"
CACHE_DIRNAME = ".cache"
THUMBS_DIRNAME = "thumbs"
PREVIEW_DIRNAME = "cropped_preview"
RAW_PREVIEW_DIRNAME = "cropped_preview_raw"
REFERENCE_DIRNAME = "reference"
CROPPED_DIRNAME = "cropped"
# Subfolders from the previous layout that we ignore as inputs but otherwise
# leave alone. Users can delete them manually if they no longer want them.
_LEGACY_OUTPUT_DIRS = ("approved", "needs-review", "rejected")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _positive(raw_bgr: np.ndarray, method: str | None) -> np.ndarray:
    """Convert negatives to positives; pass everything else through."""
    if method == "negative":
        return negative_to_positive(raw_bgr)
    return raw_bgr


def ensure_layout(input_folder: Path) -> None:
    """Create the hidden cache subfolders. The user-visible `cropped/`
    folder is created lazily, on the first approve. The legacy
    approved/needs-review/rejected subfolders are NOT created — the
    manifest is the source of truth for decision state.
    """
    (input_folder / CACHE_DIRNAME / THUMBS_DIRNAME).mkdir(parents=True, exist_ok=True)
    (input_folder / CACHE_DIRNAME / PREVIEW_DIRNAME).mkdir(parents=True, exist_ok=True)
    (input_folder / CACHE_DIRNAME / RAW_PREVIEW_DIRNAME).mkdir(parents=True, exist_ok=True)


def preview_path_for(input_folder: Path, filename: str) -> Path:
    """Auto-toned cropped preview. Used by the list view and the editor."""
    return input_folder / CACHE_DIRNAME / PREVIEW_DIRNAME / f"{Path(filename).stem}.jpg"


def raw_preview_path_for(input_folder: Path, filename: str) -> Path:
    """Raw (no Auto Tone) cropped preview, kept alongside the toned one so
    the UI can flip between them instantly without re-rendering.
    """
    return input_folder / CACHE_DIRNAME / RAW_PREVIEW_DIRNAME / f"{Path(filename).stem}.jpg"


def cropped_path_for(input_folder: Path, filename: str) -> Path:
    """Final delivered crop. Exists only when the entry's decision is 'approved'."""
    return input_folder / CROPPED_DIRNAME / f"{Path(filename).stem}.jpg"


def remove_cropped_if_present(input_folder: Path, filename: str) -> None:
    p = cropped_path_for(input_folder, filename)
    if p.exists():
        try:
            p.unlink()
        except OSError:
            pass


def manifest_path(input_folder: Path) -> Path:
    return input_folder / MANIFEST_NAME


def load_manifest(input_folder: Path) -> Manifest:
    p = manifest_path(input_folder)
    if not p.exists():
        now = _now_iso()
        return Manifest(created_at=now, updated_at=now, entries=[])
    raw = json.loads(p.read_text(encoding="utf-8"))
    return Manifest.model_validate(raw)


def save_manifest(input_folder: Path, manifest: Manifest) -> None:
    manifest.updated_at = _now_iso()
    p = manifest_path(input_folder)
    p.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )


def _output_filename(src: Path) -> str:
    return f"{src.stem}.jpg"


def _is_under(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _is_legacy_reference_file(p: Path) -> bool:
    """Legacy: pre-existing 'NAME copy.jpg' files sitting next to inputs.
    New layout puts these in the reference/ subfolder, but we still skip
    them as inputs if anyone hasn't moved them.
    """
    stem = p.stem
    return stem.endswith(" copy") or stem.endswith("-copy") or stem.endswith("_copy")


def reference_for(input_folder: Path, input_filename: str) -> Path | None:
    """Find the manual-crop reference for an input filename.

    Preferred location: ``input_folder/reference/<stem>.<ext>`` matching the
    input stem case-insensitively. Falls back to legacy 'NAME copy.jpg'
    siblings next to the input file.
    """
    target_stem = Path(input_filename).stem
    ref_dir = input_folder / REFERENCE_DIRNAME
    if ref_dir.is_dir():
        for p in ref_dir.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() not in (".jpg", ".jpeg"):
                continue
            if p.stem.lower() == target_stem.lower():
                return p
    for suffix in (" copy", "-copy", "_copy"):
        for ext in (".jpg", ".JPG", ".jpeg", ".JPEG"):
            cand = input_folder / f"{target_stem}{suffix}{ext}"
            if cand.exists():
                return cand
    return None


def discover_inputs(input_folder: Path) -> list[Path]:
    """JPGs in the input folder that are actual raw inputs.

    Excludes the cropped output folder, any leftover legacy bucket
    subfolders, the cache, and reference / pre-cropped sibling files
    (filenames ending in ' copy').
    """
    excluded = [
        input_folder / d
        for d in (CROPPED_DIRNAME, *_LEGACY_OUTPUT_DIRS, CACHE_DIRNAME, REFERENCE_DIRNAME)
    ]
    out: list[Path] = []
    for p in sorted(input_folder.iterdir()):
        if not p.is_file():
            continue
        if p.suffix not in SUPPORTED_EXTENSIONS:
            continue
        # macOS metadata sidecars ("._filename.jpg") sit next to originals
        # on OneDrive / external drives and aren't valid JPEGs.
        if p.name.startswith("._"):
            continue
        if any(_is_under(p, e) for e in excluded):
            continue
        if _is_legacy_reference_file(p):
            continue
        out.append(p)
    return out


def _process_one(args: tuple[str, str]) -> dict:
    """Worker. Returns a serialisable dict suitable for ManifestEntry."""
    if len(args) == 2:
        src_str, input_folder_str = args
        method = "auto"
    else:
        src_str, input_folder_str, method = args
    src = Path(src_str)
    input_folder = Path(input_folder_str)
    timestamp = _now_iso()
    # Detection results change → the cached "original with green guide"
    # PNGs are stale (their cache key is per-corner-coords, so a new file
    # would be written anyway, but pruning keeps the cache from growing
    # without bound across reprocess runs).
    _prune_with_guide_cache(input_folder, src.name)
    try:
        loaded = load_image(src)
    except Exception as exc:
        remove_cropped_if_present(input_folder, src.name)
        return {
            "filename": src.name,
            "decision": "rejected",
            "score": 0,
            "signals": {},
            "rotation_deg": None,
            "crop_box": None,
            "detection_method": method,
            "notes": [],
            "error": f"load_failed: {exc}",
            "edited_by_user": False,
            "timestamp": timestamp,
        }

    detection = detect(loaded.bgr, method=method)

    detected_corners = None
    if detection.found and detection.rect_small is not None:
        # Compute the 4 corners of the detected rect in ORIGINAL full-res
        # image coords, in clockwise order. This is what the editor draws
        # as a guide overlay (in image-pixel space, before any rotation).
        box_small = cv2.boxPoints(detection.rect_small.as_cv())
        scale = detection.scale if detection.scale else 1.0
        box_full = box_small / scale
        # Sort clockwise from top-left for stable display.
        center = box_full.mean(axis=0)
        angles = np.arctan2(box_full[:, 1] - center[1], box_full[:, 0] - center[0])
        order = np.argsort(angles)
        sorted_pts = box_full[order]
        detected_corners = [{"x": float(pt[0]), "y": float(pt[1])} for pt in sorted_pts]

    if not detection.found:
        # No auto detection. Write the original uncropped to the preview cache
        # so the list view has something to show, and pre-fill a centred
        # 80% crop_box as a starting point. Nothing goes into cropped/ yet.
        preview = preview_path_for(input_folder, src.name)
        save_jpeg(preview, loaded.bgr, icc_profile=loaded.icc_profile)
        thumb_path = input_folder / CACHE_DIRNAME / THUMBS_DIRNAME / _output_filename(src)
        save_thumbnail(thumb_path, loaded.bgr, max_edge=320)
        remove_cropped_if_present(input_folder, src.name)
        fh, fw = loaded.bgr.shape[:2]
        default_box = {"cx": fw / 2.0, "cy": fh / 2.0, "w": fw * 0.8, "h": fh * 0.8}
        return {
            "filename": src.name,
            "decision": "needs-review",
            "score": 0,
            "signals": {},
            "rotation_deg": 0.0,
            "crop_box": default_box,
            "detection_method": method,
            "notes": [*detection.notes, "no_auto_detection"],
            "error": None,
            "edited_by_user": False,
            "timestamp": timestamp,
        }

    # Re-resize to get the small bgr for L-separation score; cheap.
    h, w = loaded.bgr.shape[:2]
    long_edge = max(h, w)
    scale = detection.scale
    if scale < 1.0:
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        small_bgr = cv2.resize(loaded.bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        small_bgr = loaded.bgr

    score = score_detection(detection, small_bgr)
    # Sprocket-based negative detection returns the exact image bounds
    # already, but shave a symmetrical 2% (1% each side) so residual
    # film-base slivers don't sneak into the final. Other paths return
    # a shrink-target rect handled by the method's own shrink.
    shrink = 0.98 if getattr(detection, "precise_crop", False) else shrink_for_method(method)
    result = straighten_and_crop(loaded.bgr, detection, shrink=shrink)
    if result is None:
        remove_cropped_if_present(input_folder, src.name)
        return {
            "filename": src.name,
            "decision": "rejected",
            "score": score.total,
            "signals": score.signals,
            "rotation_deg": None,
            "crop_box": None,
            "detected_corners": detected_corners,
            "detection_method": method,
            "notes": [*detection.notes, *score.reasons, "crop_failed"],
            "error": None,
            "edited_by_user": False,
            "timestamp": timestamp,
        }

    # No automatic approvals — every photo lands in needs-review for the
    # user to inspect and approve manually. The confidence score is still
    # computed and surfaced in the manifest / list view as a hint.
    # Precise-crop detections (e.g. sprocket-based negatives) synthesise
    # a rect without a contour, so the scorer returns 0; treat them as
    # needs-review regardless.
    if getattr(detection, "precise_crop", False):
        bucket = "needs-review"
    else:
        bucket = "rejected" if score.bucket == "rejected" else "needs-review"
    out_name = _output_filename(src)
    # Face-based upright auto-rotation is disabled — it misfires too
    # often (flipping correctly-oriented photos 180°). The user can hit
    # the 90° buttons in the editor when a photo actually needs it.
    upright_qt = 0
    # Generate BOTH the auto-toned version (the default the UI shows) and
    # a raw version. Caching both means flipping the global Auto Tone
    # toggle reloads instantly without re-rendering any image.
    raw_cropped = result.cropped
    if method == "negative":
        raw_cropped = negative_to_positive(raw_cropped)
    final_cropped = auto_tone(raw_cropped)
    if method == "negative":
        final_cropped = gray_world_balance(final_cropped)
    # Always write to the preview cache (used by the list view + editor).
    preview = preview_path_for(input_folder, src.name)
    save_jpeg(preview, final_cropped, icc_profile=loaded.icc_profile)
    save_jpeg(raw_preview_path_for(input_folder, src.name), raw_cropped, icc_profile=loaded.icc_profile)
    save_thumbnail(input_folder / CACHE_DIRNAME / THUMBS_DIRNAME / out_name, final_cropped, max_edge=320)

    # Nothing goes into the user-visible cropped/ folder until the user
    # explicitly approves the photo in the review UI.
    remove_cropped_if_present(input_folder, src.name)

    return {
        "filename": src.name,
        "decision": bucket,
        "score": score.total,
        "signals": score.signals,
        "rotation_deg": result.rotation_deg,
        "crop_box": result.crop_box.to_dict(),
        "upright_rotation_qt": int(upright_qt),
        "detected_corners": detected_corners,
        "detection_method": method,
        "notes": [*detection.notes, *score.reasons],
        "error": None,
        "edited_by_user": False,
        "timestamp": timestamp,
    }


def cluster_by_size(
    entries: list[ManifestEntry], tolerance: float = 0.07
) -> list[dict]:
    """Cluster manifest entries by their crop_box dimensions (long × short
    edge). Returns a list of clusters, each with `median_long`,
    `median_short`, and `members` (filenames).
    """
    sized = []
    for e in entries:
        if e.crop_box is None:
            continue
        long_edge = max(e.crop_box.w, e.crop_box.h)
        short_edge = min(e.crop_box.w, e.crop_box.h)
        sized.append((e.filename, long_edge, short_edge))
    sized.sort(key=lambda x: -x[1])  # largest first

    clusters: list[dict] = []
    for fn, lo, sh in sized:
        matched = False
        for c in clusters:
            cl = c["median_long"]
            cs = c["median_short"]
            if abs(lo - cl) / cl < tolerance and abs(sh - cs) / cs < tolerance:
                c["members"].append(fn)
                c["_longs"].append(lo)
                c["_shorts"].append(sh)
                c["median_long"] = float(np.median(c["_longs"]))
                c["median_short"] = float(np.median(c["_shorts"]))
                matched = True
                break
        if not matched:
            clusters.append({
                "median_long": lo,
                "median_short": sh,
                "members": [fn],
                "_longs": [lo],
                "_shorts": [sh],
            })
    for c in clusters:
        c.pop("_longs", None)
        c.pop("_shorts", None)
    return clusters


def apply_size_to_entry(
    input_folder: Path, filename: str, new_long: float, new_short: float
) -> ManifestEntry:
    """Replace the entry's crop dimensions with (new_long, new_short),
    keeping center + rotation, then re-render the output JPEG.
    """
    input_folder = input_folder.resolve()
    manifest = load_manifest(input_folder)
    entry = next((e for e in manifest.entries if e.filename == filename), None)
    if entry is None or entry.crop_box is None or entry.rotation_deg is None:
        raise ValueError(f"entry has no crop_box: {filename}")
    src = input_folder / filename
    if not src.exists():
        raise FileNotFoundError(f"input not found: {src}")

    # Preserve portrait vs landscape orientation of the existing crop.
    cur_w, cur_h = entry.crop_box.w, entry.crop_box.h
    if cur_w >= cur_h:
        w, h = new_long, new_short
    else:
        w, h = new_short, new_long

    from .straighten import CropBox, apply_user_edit
    loaded = load_image(src)
    box = CropBox(cx=entry.crop_box.cx, cy=entry.crop_box.cy, w=w, h=h)
    cropped = apply_user_edit(
        loaded.bgr, entry.rotation_deg, box, upright_qt=entry.upright_rotation_qt
    )
    if cropped is None or cropped.size == 0:
        raise ValueError("crop produced empty image")
    cropped = _positive(cropped, entry.detection_method)
    cropped = auto_tone(cropped)

    save_jpeg(preview_path_for(input_folder, filename), cropped, icc_profile=loaded.icc_profile)
    save_thumbnail(input_folder / CACHE_DIRNAME / THUMBS_DIRNAME / f"{Path(filename).stem}.jpg", cropped, max_edge=320)
    if entry.decision == "approved":
        save_jpeg(cropped_path_for(input_folder, filename), cropped, icc_profile=loaded.icc_profile)
    else:
        remove_cropped_if_present(input_folder, filename)

    entry.crop_box.w = w
    entry.crop_box.h = h
    entry.timestamp = _now_iso()
    save_manifest(input_folder, manifest)
    return entry


def unify_slide_crops(input_folder: Path, progress_callback=None) -> int:
    return _unify_scanner_crops(input_folder, "slide", progress_callback)


def unify_negative_crops(input_folder: Path, progress_callback=None) -> int:
    return _unify_scanner_crops(input_folder, "negative", progress_callback)


def _unify_scanner_crops(
    input_folder: Path, method: str, progress_callback=None
) -> int:
    """Snap every entry from the given fixed-scanner method to the median
    rect across all such entries. Slides and negatives both come off the
    same scanner at a consistent physical position, so per-photo
    detection jitter is noise — averaging it out gives identical crops
    across the whole batch.

    Re-renders every affected entry's preview/thumb/cropped output.
    Returns the number of entries unified.
    """
    from .straighten import CropBox, apply_user_edit

    input_folder = input_folder.resolve()
    manifest = load_manifest(input_folder)
    candidates = [
        e for e in manifest.entries
        if e.detection_method == method
        and e.crop_box is not None
        and e.rotation_deg is not None
    ]
    if len(candidates) < 2:
        return 0

    # Normalise each rect to (long, short) so portrait/landscape entries
    # vote into the same median bucket. Median is taken on long/short
    # instead of w/h to keep portrait slides from dragging w down.
    longs = [max(e.crop_box.w, e.crop_box.h) for e in candidates]
    shorts = [min(e.crop_box.w, e.crop_box.h) for e in candidates]
    cxs = [e.crop_box.cx for e in candidates]
    cys = [e.crop_box.cy for e in candidates]
    angles = [e.rotation_deg for e in candidates]
    med_long = float(np.median(longs))
    med_short = float(np.median(shorts))
    med_cx = float(np.median(cxs))
    med_cy = float(np.median(cys))
    med_angle = float(np.median(angles))

    unified = 0
    total = len(candidates)
    for idx, entry in enumerate(candidates, start=1):
        if progress_callback is not None:
            try:
                progress_callback(idx, total, entry)
            except Exception:
                pass
        # Preserve each entry's portrait/landscape orientation.
        is_portrait = entry.crop_box.h > entry.crop_box.w
        w = med_short if is_portrait else med_long
        h = med_long if is_portrait else med_short
        src = input_folder / entry.filename
        if not src.exists():
            continue
        loaded = load_image(src)
        box = CropBox(cx=med_cx, cy=med_cy, w=w, h=h)
        cropped = apply_user_edit(
            loaded.bgr, med_angle, box, upright_qt=entry.upright_rotation_qt
        )
        if cropped is None or cropped.size == 0:
            continue
        raw_cropped = _positive(cropped, method)
        toned_cropped = auto_tone(raw_cropped)
        save_jpeg(preview_path_for(input_folder, entry.filename), toned_cropped, icc_profile=loaded.icc_profile)
        save_jpeg(raw_preview_path_for(input_folder, entry.filename), raw_cropped, icc_profile=loaded.icc_profile)
        save_thumbnail(
            input_folder / CACHE_DIRNAME / THUMBS_DIRNAME / f"{Path(entry.filename).stem}.jpg",
            toned_cropped if entry.auto_tone else raw_cropped,
            max_edge=320,
        )
        if entry.decision == "approved":
            visible = toned_cropped if entry.auto_tone else raw_cropped
            save_jpeg(cropped_path_for(input_folder, entry.filename), visible, icc_profile=loaded.icc_profile)
        entry.crop_box = CropBoxModel(cx=med_cx, cy=med_cy, w=w, h=h)
        entry.rotation_deg = med_angle
        # Keep the green detected-edge guide in sync with the unified
        # crop_box; otherwise the editor draws the original per-entry
        # detection which no longer matches the crop.
        cv_rect = ((med_cx, med_cy), (w, h), med_angle)
        box_pts = cv2.boxPoints(cv_rect)
        centre = box_pts.mean(axis=0)
        angles_pt = np.arctan2(box_pts[:, 1] - centre[1], box_pts[:, 0] - centre[0])
        order = np.argsort(angles_pt)
        sorted_pts = box_pts[order]
        entry.detected_corners = [
            {"x": float(p[0]), "y": float(p[1])} for p in sorted_pts
        ]
        entry.timestamp = _now_iso()
        _prune_with_guide_cache(input_folder, entry.filename)
        unified += 1

    save_manifest(input_folder, manifest)
    return unified


def _prune_with_guide_cache(input_folder: Path, filename: str) -> None:
    """Delete cached 'original-with-guide' images for a filename. They are
    keyed by detected-corner coordinates, so any pre-existing entries are
    stale once the corners change. The server regenerates on next request.
    """
    cache_dir = input_folder / CACHE_DIRNAME / "with_guide"
    if not cache_dir.is_dir():
        return
    stem = Path(filename).stem
    for p in cache_dir.glob(f"{stem}_*.jpg"):
        try:
            p.unlink()
        except OSError:
            pass


def redetect_and_match_aspect(
    input_folder: Path,
    filename: str,
    target_long: float,
    target_short: float,
) -> ManifestEntry:
    """Re-run edge detection on the input, then build a crop rectangle
    that matches the target aspect ratio (target_long/target_short).

    The detected centre, angle, and long-edge length are kept — only the
    short edge is replaced with ``detected_long / target_aspect`` so the
    rectangle hugs whichever edge the detector was most confident about
    while snapping the proportions to the picked size type.

    If detection fails on the new method, falls back to plain size
    snapping at the existing crop centre.
    """
    from .straighten import CropBox, apply_user_edit, _scale_rect

    input_folder = input_folder.resolve()
    if target_long <= 0 or target_short <= 0:
        raise ValueError("target dimensions must be positive")
    if target_long < target_short:
        target_long, target_short = target_short, target_long
    target_aspect = target_long / target_short

    manifest = load_manifest(input_folder)
    entry = next((e for e in manifest.entries if e.filename == filename), None)
    if entry is None:
        raise ValueError(f"unknown entry: {filename}")
    src = input_folder / filename
    if not src.exists():
        raise FileNotFoundError(f"input not found: {src}")

    method = entry.detection_method or "auto"
    loaded = load_image(src)
    detection = detect(loaded.bgr, method=method)
    if not detection.found or detection.rect_small is None:
        # Detection lost the photo on this method — fall back to a plain
        # aspect-corrected resize at the existing crop centre. Use the
        # target dims directly so the user still gets the requested ratio.
        return apply_size_to_entry(input_folder, filename, target_long, target_short)

    rect_full = _scale_rect(detection.rect_small, detection.scale, shrink_for_method(method))
    detected_long = max(rect_full.w, rect_full.h)
    new_long = detected_long
    new_short = new_long / target_aspect

    box = CropBox(cx=rect_full.cx, cy=rect_full.cy, w=new_long, h=new_short)
    cropped = apply_user_edit(
        loaded.bgr, rect_full.angle, box, upright_qt=entry.upright_rotation_qt
    )
    if cropped is None or cropped.size == 0:
        raise ValueError("crop produced empty image")
    cropped = _positive(cropped, entry.detection_method)
    cropped = auto_tone(cropped)

    save_jpeg(preview_path_for(input_folder, filename), cropped, icc_profile=loaded.icc_profile)
    save_thumbnail(
        input_folder / CACHE_DIRNAME / THUMBS_DIRNAME / f"{Path(filename).stem}.jpg",
        cropped,
        max_edge=320,
    )
    if entry.decision == "approved":
        save_jpeg(cropped_path_for(input_folder, filename), cropped, icc_profile=loaded.icc_profile)

    # Recompute the green guide's corners from the FINAL aspect-constrained
    # crop rectangle, not the raw detection. Otherwise the outline keeps
    # showing the detected mask's proportions even after we snap to 4×6.
    cv_rect = ((rect_full.cx, rect_full.cy), (new_long, new_short), rect_full.angle)
    box_pts = cv2.boxPoints(cv_rect)
    centre = box_pts.mean(axis=0)
    angles = np.arctan2(box_pts[:, 1] - centre[1], box_pts[:, 0] - centre[0])
    order = np.argsort(angles)
    sorted_pts = box_pts[order]
    entry.detected_corners = [
        {"x": float(pt[0]), "y": float(pt[1])} for pt in sorted_pts
    ]

    entry.rotation_deg = rect_full.angle
    entry.crop_box = CropBoxModel(
        cx=rect_full.cx, cy=rect_full.cy, w=new_long, h=new_short
    )
    entry.edited_by_user = True
    entry.timestamp = _now_iso()
    _prune_with_guide_cache(input_folder, filename)
    save_manifest(input_folder, manifest)
    return entry


def reprocess_one(input_folder: Path, filename: str, method: str = "auto") -> ManifestEntry:
    """Re-run detection on a single input file using the specified method
    and update the manifest entry in place. Returns the new entry.
    """
    input_folder = input_folder.resolve()
    ensure_layout(input_folder)
    src = input_folder / filename
    if not src.exists():
        raise FileNotFoundError(f"Input not found: {src}")
    result = _process_one((str(src), str(input_folder), method))
    entry = ManifestEntry.model_validate(result)
    manifest = load_manifest(input_folder)
    found = False
    for i, e in enumerate(manifest.entries):
        if e.filename == filename:
            # Preserve user-edit flag only if the user explicitly edited.
            entry.edited_by_user = False
            manifest.entries[i] = entry
            found = True
            break
    if not found:
        manifest.entries.append(entry)
    _prune_with_guide_cache(input_folder, filename)
    save_manifest(input_folder, manifest)
    return entry


def run_batch(
    input_folder: Path,
    workers: int | None = None,
    progress_callback=None,
    only_new: bool = False,
    default_method: str = "auto",
    force_method: str | None = None,
) -> Manifest:
    input_folder = input_folder.resolve()
    if not input_folder.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    ensure_layout(input_folder)
    manifest = load_manifest(input_folder)
    existing: dict[str, ManifestEntry] = {e.filename: e for e in manifest.entries}

    images = discover_inputs(input_folder)
    tasks: list[tuple[str, str, str]] = []
    skipped: list[ManifestEntry] = []
    for img in images:
        prev = existing.get(img.name)
        if prev is not None and (prev.edited_by_user or only_new):
            # `only_new` (used by Reprocess) keeps every already-processed
            # entry; the deterministic detector would only reproduce the
            # same result. `edited_by_user` is always preserved.
            skipped.append(prev)
            continue
        if force_method is not None:
            method = force_method
        elif prev is not None:
            method = prev.detection_method
        else:
            method = default_method
        tasks.append((str(img), str(input_folder), method))

    new_entries: list[ManifestEntry] = []
    if not tasks:
        manifest.entries = skipped
        save_manifest(input_folder, manifest)
        return manifest

    cpu = workers or min(8, max(1, (os.cpu_count() or 2) - 1))

    # Incremental persistence: every PERSIST_EVERY completed photos we
    # write the manifest so the in-browser list can poll it and grow
    # while detection keeps running on the rest of the batch.
    PERSIST_EVERY = 5

    def _flush() -> None:
        by_name: dict[str, ManifestEntry] = {e.filename: e for e in skipped}
        for e in new_entries:
            by_name[e.filename] = e
        manifest.entries = sorted(by_name.values(), key=lambda e: e.filename)
        save_manifest(input_folder, manifest)

    if cpu == 1 or len(tasks) == 1:
        for i, t in enumerate(tasks):
            entry_dict = _process_one(t)
            entry = ManifestEntry.model_validate(entry_dict)
            new_entries.append(entry)
            if progress_callback:
                progress_callback(i + 1, len(tasks), entry)
            if (i + 1) % PERSIST_EVERY == 0:
                _flush()
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=cpu, mp_context=ctx) as ex:
            futures = {ex.submit(_process_one, t): t for t in tasks}
            for i, fut in enumerate(as_completed(futures)):
                entry_dict = fut.result()
                entry = ManifestEntry.model_validate(entry_dict)
                new_entries.append(entry)
                if progress_callback:
                    progress_callback(i + 1, len(tasks), entry)
                if (i + 1) % PERSIST_EVERY == 0:
                    _flush()

    _flush()
    return manifest
