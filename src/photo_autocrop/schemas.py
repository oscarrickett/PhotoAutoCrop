from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


Decision = Literal["approved", "needs-review", "rejected", "no-detection"]


class CropBoxModel(BaseModel):
    cx: float
    cy: float
    w: float
    h: float


class Point(BaseModel):
    x: float
    y: float


class ManifestEntry(BaseModel):
    filename: str
    decision: Decision
    score: int = 0
    signals: dict[str, int] = {}
    rotation_deg: Optional[float] = None
    crop_box: Optional[CropBoxModel] = None
    # Quarter-turns (90° CW each) applied AFTER cropping to make faces
    # upright. 0 = leave the cropped image as-is. Computed via face
    # detection during processing; stays sticky across re-runs so manual
    # overrides aren't lost.
    upright_rotation_qt: int = 0
    # Per-photo Auto Tone toggle. On by default; user can disable for
    # individual photos when the correction overshoots (e.g. a sunset
    # that should stay warm).
    auto_tone: bool = True
    # Highlight rolloff strength, 0-100. 0 = leave highlights alone.
    # Useful for rescuing blown-out faces in over-exposed scans.
    highlight_pull: int = 0
    # Four corners of the detected photo in original-image pixel coords
    # (clockwise starting top-left after sorting). Surfaced to the editor
    # as a guide overlay so the user can see what the algorithm thinks the
    # photo's edges are.
    detected_corners: Optional[list[Point]] = None
    # Detection algorithm the user has elected for this image. Sticks
    # across re-runs so a "this is a small photo" choice keeps applying.
    detection_method: Literal["auto", "small", "loose", "edges", "slide", "negative"] = "auto"
    notes: list[str] = []
    error: Optional[str] = None
    edited_by_user: bool = False
    timestamp: str


class Manifest(BaseModel):
    version: int = 1
    created_at: str
    updated_at: str
    entries: list[ManifestEntry] = []


class SaveEditRequest(BaseModel):
    filename: str
    rotation_deg: float
    crop_box: CropBoxModel
    decision: Decision = "approved"
    # If omitted, the server keeps whatever upright_rotation_qt the entry
    # already had (so saving an edit doesn't accidentally reset the
    # auto-detected orientation).
    upright_rotation_qt: Optional[int] = None


class RedetectRequest(BaseModel):
    filename: str
    method: Literal["auto", "small", "loose", "edges", "slide", "negative"] = "auto"


class RotateUprightRequest(BaseModel):
    filename: str
    # +1 = 90° CW, -1 = 90° CCW, ±2 = 180°. Anything mod 4.
    delta: int


class SnapToSizeRequest(BaseModel):
    filename: str
    # Long and short edge in input-image pixels. Portrait vs landscape
    # orientation of the existing crop is preserved by apply_size_to_entry.
    long_edge: float
    short_edge: float
