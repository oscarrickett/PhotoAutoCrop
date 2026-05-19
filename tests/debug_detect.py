"""Dump masks + a contour overlay for one sample image to diagnose detection."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from photo_autocrop.detect import detect, _resize_for_detection, _photo_mask
from photo_autocrop.io_utils import load_image
from photo_autocrop.straighten import straighten_and_crop


def main(path: str) -> None:
    out_dir = Path(__file__).parent / "_debug_out"
    out_dir.mkdir(exist_ok=True)
    stem = Path(path).stem.replace(" ", "_")

    loaded = load_image(Path(path))
    small, scale = _resize_for_detection(loaded.bgr)
    cv2.imwrite(str(out_dir / f"{stem}_00_small.jpg"), small)

    photo = _photo_mask(small)
    cv2.imwrite(str(out_dir / f"{stem}_01_photo_mask.jpg"), photo)

    detection = detect(loaded.bgr)
    print(f"found={detection.found} notes={detection.notes}")
    if detection.rect_small is not None:
        r = detection.rect_small
        print(f"rect_small cx={r.cx:.1f} cy={r.cy:.1f} w={r.w:.1f} h={r.h:.1f} angle={r.angle:.2f}")

    overlay = small.copy()
    if detection.contour_small is not None:
        cv2.drawContours(overlay, [detection.contour_small], -1, (0, 255, 0), 2)
    if detection.rect_small is not None:
        box = cv2.boxPoints(detection.rect_small.as_cv()).astype(int)
        cv2.drawContours(overlay, [box], 0, (0, 0, 255), 3)
    cv2.imwrite(str(out_dir / f"{stem}_03_overlay.jpg"), overlay)

    result = straighten_and_crop(loaded.bgr, detection)
    if result is not None:
        cv2.imwrite(str(out_dir / f"{stem}_04_cropped.jpg"), result.cropped)
        print(f"cropped shape={result.cropped.shape} rotation={result.rotation_deg:.2f}")
    else:
        print("cropped: <none>")
    print(f"Wrote diagnostics to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1])
