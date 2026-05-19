from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".JPG", ".JPEG"}


@dataclass
class LoadedImage:
    bgr: np.ndarray
    icc_profile: bytes | None
    exif_bytes: bytes | None


def list_input_images(folder: Path) -> list[Path]:
    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix in SUPPORTED_EXTENSIONS
    )


def load_image(path: Path) -> LoadedImage:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        icc = im.info.get("icc_profile")
        exif = im.info.get("exif")
        rgb = im.convert("RGB")
        arr = np.array(rgb)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return LoadedImage(bgr=bgr, icc_profile=icc, exif_bytes=exif)


def save_jpeg(
    path: Path,
    bgr: np.ndarray,
    quality: int = 95,
    icc_profile: bytes | None = None,
) -> None:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    save_kwargs: dict = {
        "format": "JPEG",
        "quality": quality,
        "optimize": True,
        "subsampling": "4:4:4",
    }
    if icc_profile:
        save_kwargs["icc_profile"] = icc_profile
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, **save_kwargs)


def save_thumbnail(path: Path, bgr: np.ndarray, max_edge: int = 256, quality: int = 80) -> None:
    h, w = bgr.shape[:2]
    scale = max_edge / max(h, w)
    if scale < 1.0:
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        bgr = cv2.resize(bgr, new_size, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, format="JPEG", quality=quality, optimize=True)


def encode_jpeg_bytes(bgr: np.ndarray, quality: int = 85) -> bytes:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()
