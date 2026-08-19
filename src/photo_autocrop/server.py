from __future__ import annotations

import io
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .io_utils import encode_jpeg_bytes, load_image, save_jpeg, save_thumbnail
from .pipeline import (
    CACHE_DIRNAME,
    CROPPED_DIRNAME,
    THUMBS_DIRNAME,
    BUCKETS,
    apply_size_to_entry,
    cluster_by_size,
    cropped_path_for,
    discover_inputs,
    ensure_layout,
    load_manifest,
    manifest_path,
    preview_path_for,
    raw_preview_path_for,
    redetect_and_match_aspect,
    reference_for,
    remove_cropped_if_present,
    reprocess_one,
    save_manifest,
    run_batch,
    unify_slide_crops,
)
from .schemas import (
    CropBoxModel,
    Manifest,
    ManifestEntry,
    RedetectRequest,
    RotateUprightRequest,
    SaveEditRequest,
    SnapToSizeRequest,
)
from .straighten import CropBox, apply_user_edit


def _resolve_web_dir() -> Path:
    """Where the bundled HTML/JS/CSS lives at runtime.

    Inside a PyInstaller --onefile bundle the data files are extracted
    to ``sys._MEIPASS`` (a temp folder). In dev they sit next to the
    package source.
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass) / "web"
    return Path(__file__).resolve().parent.parent.parent / "web"


WEB_DIR = _resolve_web_dir()

app = FastAPI(title="photo-autocrop")

_state: dict = {"input_folder": None}
# Session-scoped UI preferences. Auto Tone toggle starts on.
_settings: dict = {"auto_tone": True}
_progress: dict = {
    "running": False, "total": 0, "done": 0,
    "current": "", "error": None,
    "started_at": None, "finished_at": None, "eta_seconds": None,
    "phase": "idle", "phase_label": "",
}
_progress_lock = threading.Lock()


def set_input_folder(folder: Path) -> None:
    folder = folder.resolve()
    ensure_layout(folder)
    _state["input_folder"] = folder


def _folder() -> Path:
    f = _state.get("input_folder")
    if f is None:
        raise HTTPException(status_code=400, detail="no folder selected")
    return f


@app.get("/api/state")
def get_state() -> JSONResponse:
    from . import __version__
    folder = _state.get("input_folder")
    return JSONResponse({
        "folder": str(folder) if folder else None,
        "processing": _progress["running"],
        "version": __version__,
    })


@app.get("/api/settings")
def get_settings() -> JSONResponse:
    return JSONResponse(dict(_settings))


@app.post("/api/settings")
def set_settings(req: dict) -> JSONResponse:
    if "auto_tone" in req:
        _settings["auto_tone"] = bool(req["auto_tone"])
    return JSONResponse(dict(_settings))


@app.get("/api/progress")
def get_progress() -> JSONResponse:
    with _progress_lock:
        snap = dict(_progress)
    # ETA computed from server-side wall clock so it stays correct across
    # client polls. Only meaningful once at least one image has finished.
    started = snap.get("started_at")
    if started and snap.get("running") and snap["done"] > 0 and snap["total"] > snap["done"]:
        try:
            t0 = datetime.fromisoformat(started)
            elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
            per = elapsed / snap["done"]
            snap["eta_seconds"] = int(per * (snap["total"] - snap["done"]))
        except Exception:
            snap["eta_seconds"] = None
    else:
        snap["eta_seconds"] = None
    return JSONResponse(snap)


def _bg_pipeline(folder: Path, default_method: str = "auto") -> None:
    def cb(i: int, total: int, entry) -> None:
        with _progress_lock:
            _progress["done"] = i
            _progress["total"] = total
            _progress["current"] = entry.filename
    try:
        manifest = load_manifest(folder)
        existing_names = {e.filename for e in manifest.entries}
        edited_names = {e.filename for e in manifest.entries if e.edited_by_user}
        files = discover_inputs(folder)
        if default_method == "auto":
            # Default open: detection is deterministic, so only process
            # files that don't already have a manifest entry.
            to_process = [f for f in files if f.name not in existing_names]
            with _progress_lock:
                _progress["total"] = len(to_process)
            if to_process:
                run_batch(folder, progress_callback=cb, only_new=True)
        else:
            # User explicitly picked a method (e.g. "slide"). Re-run
            # detection on every non-user-edited file with that method —
            # otherwise existing manifest entries stay stuck on whatever
            # method they were first processed with.
            to_process = [f for f in files if f.name not in edited_names]
            with _progress_lock:
                _progress["total"] = len(to_process)
            if to_process:
                run_batch(
                    folder,
                    progress_callback=cb,
                    only_new=False,
                    default_method=default_method,
                    force_method=default_method,
                )
            if default_method == "slide":
                def unify_cb(i: int, total: int, entry) -> None:
                    with _progress_lock:
                        _progress["done"] = i
                        _progress["total"] = total
                        _progress["current"] = entry.filename
                with _progress_lock:
                    _progress["phase"] = "unifying"
                    _progress["phase_label"] = "Aligning slide crops"
                    _progress["done"] = 0
                    _progress["total"] = 0
                    _progress["current"] = ""
                    _progress["started_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                unify_slide_crops(folder, progress_callback=unify_cb)
            # Negatives: sprocket detection is precise per-image, so
            # skip the median-alignment pass — it was smoothing away
            # real, correct variation across frames.
    except Exception as exc:
        with _progress_lock:
            _progress["error"] = str(exc)
    finally:
        with _progress_lock:
            _progress["running"] = False
            _progress["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            _progress["phase"] = "idle"
            _progress["phase_label"] = ""


VALID_METHODS = {"auto", "small", "loose", "edges", "slide", "negative"}


@app.post("/api/open-folder")
def open_folder(req: dict) -> JSONResponse:
    path_str = (req or {}).get("path", "").strip()
    if not path_str:
        raise HTTPException(status_code=400, detail="path is required")
    folder = Path(path_str).expanduser()
    if not folder.exists():
        raise HTTPException(status_code=404, detail=f"folder not found: {folder}")
    if not folder.is_dir():
        raise HTTPException(status_code=400, detail=f"not a folder: {folder}")
    method = (req or {}).get("method", "auto") or "auto"
    if method not in VALID_METHODS:
        raise HTTPException(status_code=400, detail=f"unknown method: {method}")
    set_input_folder(folder)
    with _progress_lock:
        if _progress["running"]:
            raise HTTPException(status_code=409, detail="another batch is already running")
        _progress.update({
            "running": True, "total": 0, "done": 0,
            "current": "", "error": None,
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "finished_at": None, "eta_seconds": None,
            "phase": "processing", "phase_label": "",
        })
    threading.Thread(target=_bg_pipeline, args=(folder, method), daemon=True).start()
    return JSONResponse({"ok": True, "folder": str(folder), "method": method})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _input_path(filename: str) -> Path:
    safe = Path(filename).name
    return _folder() / safe


def _entry_for(filename: str) -> ManifestEntry | None:
    manifest = load_manifest(_folder())
    return next((e for e in manifest.entries if e.filename == filename), None)


def _auto_tone_on(filename: str) -> bool:
    """Read the per-photo Auto Tone preference. Defaults to True."""
    e = _entry_for(filename)
    if e is None:
        return True
    return bool(getattr(e, "auto_tone", True))


def _find_existing_output(filename: str) -> Path | None:
    """Return the best available rendered crop for this filename, picking
    the toned or raw cache based on the entry's per-photo `auto_tone`.
    Approved files in `cropped/` are kept in sync with the entry's
    setting (see `set_entry_auto_tone`), so we always prefer that copy.
    """
    folder = _folder()
    auto_on = _auto_tone_on(filename)
    final = cropped_path_for(folder, filename)
    if final.exists():
        return final
    if auto_on:
        preview = preview_path_for(folder, filename)
        if preview.exists():
            return preview
    else:
        raw = raw_preview_path_for(folder, filename)
        if raw.exists():
            return raw
        # Fall back to the toned preview if the raw cache is missing
        # (legacy entries from before per-photo caching).
        preview = preview_path_for(folder, filename)
        if preview.exists():
            return preview
    return None


def _regenerate_preview(folder: Path, entry) -> bool:
    """Rebuild the toned preview cache (and cropped/ if approved) from
    the cached raw crop plus the entry's auto_tone setting.
    """
    raw_path = raw_preview_path_for(folder, entry.filename)
    if not raw_path.exists():
        return False
    loaded = load_image(raw_path)
    bgr = loaded.bgr
    from .auto_tone import auto_tone as _auto_tone
    if getattr(entry, "auto_tone", True):
        bgr = _auto_tone(bgr)
    save_jpeg(preview_path_for(folder, entry.filename), bgr, icc_profile=loaded.icc_profile)
    save_thumbnail(
        folder / CACHE_DIRNAME / THUMBS_DIRNAME / f"{Path(entry.filename).stem}.jpg",
        bgr, max_edge=320,
    )
    if entry.decision == "approved":
        save_jpeg(cropped_path_for(folder, entry.filename), bgr, icc_profile=loaded.icc_profile)
    return True


@app.post("/api/auto-tone/{filename}")
def set_entry_auto_tone(filename: str, on: bool = True) -> JSONResponse:
    """Toggle Auto Tone for a single photo. Regenerates the preview (and
    cropped/, if approved) so the saved file stays in sync.
    """
    folder = _folder()
    manifest = load_manifest(folder)
    entry = next((e for e in manifest.entries if e.filename == filename), None)
    if entry is None:
        raise HTTPException(status_code=404, detail="entry not found")
    entry.auto_tone = bool(on)
    entry.timestamp = _now()
    _regenerate_preview(folder, entry)
    save_manifest(folder, manifest)
    return JSONResponse({"ok": True, "entry": entry.model_dump(mode="json")})




@app.get("/api/manifest")
def get_manifest() -> JSONResponse:
    m = load_manifest(_folder())
    return JSONResponse(m.model_dump(mode="json"))


@app.get("/api/image/{filename}")
def get_original(filename: str) -> FileResponse:
    p = _input_path(filename)
    if not p.exists():
        raise HTTPException(status_code=404, detail="original not found")
    positive = _positive_original_path(filename)
    if positive is not None:
        return FileResponse(positive, media_type="image/jpeg")
    return FileResponse(p, media_type="image/jpeg")


def _positive_original_path(filename: str) -> Path | None:
    """For negative-method entries, return a disk-cached path to the
    original scan inverted to positive. First call generates and caches
    the file; subsequent calls hit the cache. Returns None for
    non-negative entries so the raw JPG is served as-is.
    """
    entry = _entry_for(filename)
    if entry is None or entry.detection_method != "negative":
        return None
    folder = _folder()
    src = _input_path(filename)
    cache_dir = folder / CACHE_DIRNAME / "positive_original"
    cache_path = cache_dir / f"{Path(filename).stem}.jpg"
    if cache_path.exists() and cache_path.stat().st_mtime >= src.stat().st_mtime:
        return cache_path
    from .negative import negative_to_positive
    from .auto_tone import auto_tone as _auto_tone
    loaded = load_image(src)
    positive = negative_to_positive(loaded.bgr)
    positive = _auto_tone(positive)
    cache_dir.mkdir(parents=True, exist_ok=True)
    save_jpeg(cache_path, positive, icc_profile=loaded.icc_profile)
    return cache_path


@app.get("/api/output/{filename}")
def get_output(filename: str) -> FileResponse:
    p = _find_existing_output(filename)
    if p is None:
        raise HTTPException(status_code=404, detail="output not found")
    return FileResponse(p, media_type="image/jpeg")


@app.get("/api/reference/{filename}")
def get_reference(filename: str) -> FileResponse:
    ref = reference_for(_folder(), filename)
    if ref is None:
        raise HTTPException(status_code=404, detail="no reference for this file")
    return FileResponse(ref, media_type="image/jpeg")


@app.get("/api/has-reference/{filename}")
def has_reference(filename: str) -> JSONResponse:
    ref = reference_for(_folder(), filename)
    return JSONResponse({"has_reference": ref is not None})


@app.get("/api/original-thumb/{filename}")
def get_original_thumb(filename: str, max_edge: int = 600) -> Response:
    """Thumbnail of the original (uncropped) input image."""
    folder = _folder()
    cache_path = folder / CACHE_DIRNAME / "originals" / f"{Path(filename).stem}_{max_edge}.jpg"
    if cache_path.exists():
        return FileResponse(cache_path, media_type="image/jpeg")
    src = _input_path(filename)
    if not src.exists():
        raise HTTPException(status_code=404, detail="original not found")
    bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=500, detail="cv2.imread failed")
    save_thumbnail(cache_path, bgr, max_edge=max_edge)
    return FileResponse(cache_path, media_type="image/jpeg")


@app.get("/api/original-with-guide/{filename}")
def get_original_with_guide(filename: str, max_edge: int = 600) -> Response:
    """Original input with the detected photo edges drawn as a green
    polygon. Cached per (filename, manifest-corners) — invalidated when
    the manifest changes the corners.
    """
    folder = _folder()
    src = _input_path(filename)
    if not src.exists():
        raise HTTPException(status_code=404, detail="original not found")

    manifest = load_manifest(folder)
    entry = next((e for e in manifest.entries if e.filename == filename), None)
    corners = entry.detected_corners if entry else None

    cache_dir = folder / CACHE_DIRNAME / "with_guide"
    # Cache key includes a coordinate hash so we re-render when the
    # detected corners change (e.g. after the user picks "small" method).
    key = "none" if not corners else "_".join(f"{int(p.x)}-{int(p.y)}" for p in corners)
    cache_path = cache_dir / f"{Path(filename).stem}_{max_edge}_{key[:60]}.jpg"
    if cache_path.exists():
        return FileResponse(cache_path, media_type="image/jpeg")

    # For negatives, render the guide over the inverted positive so the
    # editor thumbnail and the "with guide" strip in the list view both
    # show something recognisable instead of a pink negative.
    positive_src = _positive_original_path(filename)
    read_from = str(positive_src) if positive_src is not None else str(src)
    bgr = cv2.imread(read_from, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=500, detail="cv2.imread failed")

    h, w = bgr.shape[:2]
    scale = max_edge / max(h, w)
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0

    if corners and len(corners) == 4:
        pts = np.array([[c.x * scale, c.y * scale] for c in corners], dtype=np.int32)
        # Dashed-look polygon: draw each segment as short dashes.
        dash_color = (102, 204, 46)  # BGR green
        for i in range(4):
            a = pts[i]
            b = pts[(i + 1) % 4]
            seg_len = float(np.linalg.norm(b - a))
            if seg_len < 1:
                continue
            direction = (b - a) / seg_len
            dash = 14.0
            gap = 8.0
            t = 0.0
            while t < seg_len:
                p1 = a + direction * t
                p2 = a + direction * min(t + dash, seg_len)
                cv2.line(bgr, tuple(p1.astype(int)), tuple(p2.astype(int)), dash_color, 3, cv2.LINE_AA)
                t += dash + gap

    cache_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(cache_path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return FileResponse(cache_path, media_type="image/jpeg")


@app.get("/api/thumb/{filename}")
def get_thumb(filename: str) -> Response:
    stem = Path(filename).stem
    p = _folder() / CACHE_DIRNAME / THUMBS_DIRNAME / f"{stem}.jpg"
    if not p.exists():
        out = _find_existing_output(filename) or _input_path(filename)
        if not out.exists():
            raise HTTPException(status_code=404, detail="no source for thumb")
        bgr = cv2.imread(str(out), cv2.IMREAD_COLOR)
        if bgr is None:
            raise HTTPException(status_code=500, detail="cv2.imread failed")
        save_thumbnail(p, bgr, max_edge=320)
    return FileResponse(p, media_type="image/jpeg")


@app.post("/api/save-edit")
def save_edit(req: SaveEditRequest) -> JSONResponse:
    folder = _folder()
    original = _input_path(req.filename)
    if not original.exists():
        raise HTTPException(status_code=404, detail="original not found")

    manifest = load_manifest(folder)
    existing = next((e for e in manifest.entries if e.filename == req.filename), None)
    if req.upright_rotation_qt is not None:
        upright_qt = int(req.upright_rotation_qt) % 4
    elif existing is not None:
        upright_qt = int(existing.upright_rotation_qt) % 4
    else:
        upright_qt = 0

    loaded = load_image(original)
    box = CropBox(cx=req.crop_box.cx, cy=req.crop_box.cy, w=req.crop_box.w, h=req.crop_box.h)
    raw_cropped = apply_user_edit(loaded.bgr, req.rotation_deg, box, upright_qt=upright_qt)
    if raw_cropped is None or raw_cropped.size == 0:
        raise HTTPException(status_code=400, detail="crop produced empty image")
    if existing is not None and existing.detection_method == "negative":
        from .negative import negative_to_positive
        raw_cropped = negative_to_positive(raw_cropped)
    from .auto_tone import auto_tone as _auto_tone
    toned_cropped = _auto_tone(raw_cropped)
    # Per-photo Auto Tone — preserves the previous choice; defaults to on
    # for brand-new entries.
    auto_tone_on = bool(existing.auto_tone) if existing is not None else True
    # Keep both caches in sync so toggling Auto Tone after save stays instant.
    save_jpeg(preview_path_for(folder, req.filename), toned_cropped, icc_profile=loaded.icc_profile)
    save_jpeg(raw_preview_path_for(folder, req.filename), raw_cropped, icc_profile=loaded.icc_profile)
    # Thumbnail follows the entry's setting.
    visible = toned_cropped if auto_tone_on else raw_cropped
    stem = Path(req.filename).stem
    save_thumbnail(folder / CACHE_DIRNAME / THUMBS_DIRNAME / f"{stem}.jpg", visible, max_edge=320)
    # The user-visible cropped/ folder only holds approved files.
    if req.decision == "approved":
        save_jpeg(cropped_path_for(folder, req.filename), visible, icc_profile=loaded.icc_profile)
    else:
        remove_cropped_if_present(folder, req.filename)

    if existing is not None:
        existing.decision = req.decision
        existing.rotation_deg = req.rotation_deg
        existing.crop_box = req.crop_box
        existing.upright_rotation_qt = upright_qt
        existing.edited_by_user = True
        existing.timestamp = _now()
    else:
        manifest.entries.append(
            ManifestEntry(
                filename=req.filename,
                decision=req.decision,
                score=0,
                signals={},
                rotation_deg=req.rotation_deg,
                crop_box=req.crop_box,
                upright_rotation_qt=upright_qt,
                notes=[],
                edited_by_user=True,
                timestamp=_now(),
            )
        )
    save_manifest(folder, manifest)
    return JSONResponse({"ok": True, "decision": req.decision})


@app.post("/api/decision/{filename}")
def set_decision(filename: str, decision: str) -> JSONResponse:
    if decision not in BUCKETS:
        raise HTTPException(status_code=400, detail=f"bad decision: {decision}")
    folder = _folder()
    preview = preview_path_for(folder, filename)
    final = cropped_path_for(folder, filename)
    if decision == "approved":
        if not preview.exists():
            raise HTTPException(status_code=404, detail="no preview to approve")
        final.parent.mkdir(parents=True, exist_ok=True)
        # Copy preview bytes; preview is already encoded JPEG with the right
        # crop + ICC. Reading + re-writing would be lossy.
        import shutil
        shutil.copyfile(preview, final)
    else:
        remove_cropped_if_present(folder, filename)

    manifest = load_manifest(folder)
    for e in manifest.entries:
        if e.filename == filename:
            e.decision = decision
            e.timestamp = _now()
            break
    save_manifest(folder, manifest)
    return JSONResponse({"ok": True, "decision": decision})


@app.post("/api/save-all")
def save_all() -> JSONResponse:
    """Mark every needs-review entry as approved in one shot. Copies the
    matching preview (toned or raw, per the photo's Auto Tone setting)
    into `cropped/`. No re-rendering — purely a file copy + manifest update.
    """
    folder = _folder()
    manifest = load_manifest(folder)
    import shutil
    saved: list[str] = []
    for entry in manifest.entries:
        if entry.decision != "needs-review":
            continue
        src = (
            preview_path_for(folder, entry.filename)
            if getattr(entry, "auto_tone", True)
            else raw_preview_path_for(folder, entry.filename)
        )
        if not src.exists():
            # No preview yet (e.g. detection never produced an image) —
            # leave the entry alone.
            continue
        target = cropped_path_for(folder, entry.filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        entry.decision = "approved"
        entry.edited_by_user = True
        entry.timestamp = _now()
        saved.append(entry.filename)
    save_manifest(folder, manifest)
    return JSONResponse({"ok": True, "saved": len(saved), "filenames": saved})


@app.post("/api/reprocess")
def reprocess(method: str = "auto") -> JSONResponse:
    """Re-run detection.

    method="auto" (default): only picks up JPGs added since the manifest
    was last written — re-running the deterministic detector on
    already-processed files would produce identical results.

    Any other method (e.g. "slide"): re-runs detection on EVERY file in
    the folder with that method, including ones the user has previously
    touched. The user is explicitly opting into a different detection
    mode for the whole batch, which overrides any prior per-photo
    edits/saves.
    """
    folder = _folder()
    if method not in VALID_METHODS:
        raise HTTPException(status_code=400, detail=f"unknown method: {method}")
    manifest = load_manifest(folder)
    existing_names = {e.filename for e in manifest.entries}
    images = discover_inputs(folder)
    if method == "auto":
        to_process = [img for img in images if img.name not in existing_names]
        if not to_process:
            return JSONResponse({"ok": True, "processed": 0, "filenames": []})
        run_batch(folder, only_new=True)
    else:
        # Clear edited_by_user so run_batch doesn't skip these files.
        # The user just asked for an explicit method override — that
        # supersedes any previous "I clicked save on this" flag.
        for entry in manifest.entries:
            entry.edited_by_user = False
        save_manifest(folder, manifest)
        to_process = list(images)
        if not to_process:
            return JSONResponse({"ok": True, "processed": 0, "filenames": []})
        run_batch(
            folder,
            only_new=False,
            default_method=method,
            force_method=method,
        )
        # Slides come off the scanner at a fixed position — snap every
        # entry to the median rect so the whole batch is identically
        # framed instead of jittering per-photo.
        unified = 0
        if method == "slide":
            unified = unify_slide_crops(folder)
        return JSONResponse({
            "ok": True,
            "processed": len(to_process),
            "filenames": [f.name for f in to_process],
            "method": method,
            "unified": unified,
        })
    return JSONResponse({
        "ok": True,
        "processed": len(to_process),
        "filenames": [f.name for f in to_process],
        "method": method,
    })


@app.post("/api/redetect")
def redetect(req: RedetectRequest) -> JSONResponse:
    entry = reprocess_one(_folder(), req.filename, req.method)
    return JSONResponse({"ok": True, "entry": entry.model_dump(mode="json"), "method": req.method})


@app.post("/api/rotate-upright")
def rotate_upright(req: RotateUprightRequest) -> JSONResponse:
    """Bump upright_rotation_qt by ``delta`` (mod 4) and re-render the
    preview/thumbnail/cropped output. Used by the list-view ↺/↻ buttons
    so the user can fix orientation without entering the editor.

    Marks the entry as edited_by_user so subsequent reprocess runs don't
    revert the manual choice.
    """
    folder = _folder()
    original = _input_path(req.filename)
    if not original.exists():
        raise HTTPException(status_code=404, detail="original not found")

    manifest = load_manifest(folder)
    entry = next((e for e in manifest.entries if e.filename == req.filename), None)
    if entry is None or entry.crop_box is None or entry.rotation_deg is None:
        raise HTTPException(
            status_code=400,
            detail="no crop yet — run auto-detection or use the editor first",
        )

    new_qt = ((int(entry.upright_rotation_qt) + int(req.delta)) % 4 + 4) % 4
    loaded = load_image(original)
    box = CropBox(
        cx=entry.crop_box.cx, cy=entry.crop_box.cy,
        w=entry.crop_box.w, h=entry.crop_box.h,
    )
    cropped = apply_user_edit(loaded.bgr, entry.rotation_deg, box, upright_qt=new_qt)
    if cropped is None or cropped.size == 0:
        raise HTTPException(status_code=400, detail="crop produced empty image")
    if entry.detection_method == "negative":
        from .negative import negative_to_positive
        cropped = negative_to_positive(cropped)
    from .auto_tone import auto_tone as _auto_tone
    raw_cropped = cropped
    toned_cropped = _auto_tone(raw_cropped)
    auto_tone_on = bool(entry.auto_tone)
    save_jpeg(preview_path_for(folder, req.filename), toned_cropped, icc_profile=loaded.icc_profile)
    save_jpeg(raw_preview_path_for(folder, req.filename), raw_cropped, icc_profile=loaded.icc_profile)
    visible = toned_cropped if auto_tone_on else raw_cropped
    stem = Path(req.filename).stem
    save_thumbnail(folder / CACHE_DIRNAME / THUMBS_DIRNAME / f"{stem}.jpg", visible, max_edge=320)
    if entry.decision == "approved":
        save_jpeg(cropped_path_for(folder, req.filename), visible, icc_profile=loaded.icc_profile)

    entry.upright_rotation_qt = new_qt
    entry.edited_by_user = True
    entry.timestamp = _now()
    save_manifest(folder, manifest)
    return JSONResponse({"ok": True, "entry": entry.model_dump(mode="json")})


@app.post("/api/snap-to-size")
def snap_to_size(req: SnapToSizeRequest) -> JSONResponse:
    """Re-detect the photo's edges, then constrain the crop to match the
    aspect ratio implied by (long_edge, short_edge). The detected centre,
    angle, and long-edge length are kept; the short edge is recomputed
    from the target ratio so the rectangle hugs whichever edge detection
    was most confident about. Falls back to a plain centre-preserving
    resize if detection no longer finds the photo.
    """
    folder = _folder()
    if req.long_edge <= 0 or req.short_edge <= 0:
        raise HTTPException(status_code=400, detail="dimensions must be positive")
    try:
        entry = redetect_and_match_aspect(folder, req.filename, req.long_edge, req.short_edge)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse({"ok": True, "entry": entry.model_dump(mode="json")})


@app.get("/api/size-clusters")
def size_clusters() -> JSONResponse:
    manifest = load_manifest(_folder())
    clusters = cluster_by_size(manifest.entries)
    return JSONResponse({"clusters": clusters})


@app.post("/api/apply-size-clusters")
def apply_size_clusters(req: dict) -> JSONResponse:
    """Body: { "clusters": [ {"median_long": float, "median_short": float, "members": [filenames...]}, ... ] }
    Applies each cluster's size to its members and re-renders.
    """
    clusters = (req or {}).get("clusters") or []
    updated_filenames: list[str] = []
    errors: list[str] = []
    for c in clusters:
        ml = float(c.get("median_long", 0))
        ms = float(c.get("median_short", 0))
        members = c.get("members") or []
        if ml <= 0 or ms <= 0 or not members:
            continue
        for fn in members:
            try:
                apply_size_to_entry(_folder(), fn, ml, ms)
                updated_filenames.append(fn)
            except Exception as exc:
                errors.append(f"{fn}: {exc}")
    return JSONResponse({"ok": True, "updated": updated_filenames, "errors": errors})


@app.post("/api/shutdown")
def shutdown() -> JSONResponse:
    """Quit PhotoAutoCrop from the browser. Schedules a process exit on a
    short delay so this response flushes back to the client first.
    """
    import os as _os
    import threading as _threading

    def _kill() -> None:
        _os._exit(0)

    _threading.Timer(0.3, _kill).start()
    return JSONResponse({"ok": True, "message": "shutting down"})


if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
