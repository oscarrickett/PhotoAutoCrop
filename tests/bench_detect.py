"""Benchmark detection against ground-truth references.

For each input file in TEST_FOLDER, run detect+straighten and compare
the auto-cropped output dimensions to the manual reference dimensions.
Reports a per-image table and overall hit rate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from photo_autocrop.detect import detect
from photo_autocrop.io_utils import load_image
from photo_autocrop.pipeline import discover_inputs, reference_for
from photo_autocrop.straighten import straighten_and_crop


def main(folder: str) -> None:
    folder_p = Path(folder)
    inputs = discover_inputs(folder_p)
    debug_dir = Path(__file__).parent / "_bench_out"
    debug_dir.mkdir(exist_ok=True)
    print(f"   {'file':<32} {'detected (LxS)':<18} {'reference (LxS)':<18} {'%diff L':<8} {'%diff S':<8}")
    n_total = 0
    n_within = 0  # within 15% of reference dimensions
    for p in inputs:
        n_total += 1
        ref = reference_for(folder_p, p.name)
        ref_w = ref_h = None
        if ref is not None:
            r_img = cv2.imread(str(ref))
            if r_img is not None:
                ref_h, ref_w = r_img.shape[:2]
        loaded = load_image(p)
        det = detect(loaded.bgr)
        if det.found:
            result = straighten_and_crop(loaded.bgr, det)
            if result is not None:
                dh, dw = result.cropped.shape[:2]
                # Save side-by-side overlay for visual diff
                stem = p.stem.replace(" ", "_")
                cv2.imwrite(str(debug_dir / f"{stem}_auto.jpg"), result.cropped)
            else:
                dh = dw = 0
        else:
            dw = dh = 0

        if ref_w and dw:
            # Compare long/short edges so portrait-vs-landscape orientation
            # doesn't penalise an otherwise-correct detection.
            d_long, d_short = max(dw, dh), min(dw, dh)
            r_long, r_short = max(ref_w, ref_h), min(ref_w, ref_h)
            pdl = abs(d_long - r_long) / r_long * 100
            pds = abs(d_short - r_short) / r_short * 100
            within = pdl < 15 and pds < 15
            if within:
                n_within += 1
            mark = "OK" if within else "  "
            print(f"{mark} {p.name:<32} {f'{d_long}x{d_short}':<18} {f'{r_long}x{r_short}':<18} {pdl:<8.1f} {pds:<8.1f}")
        else:
            print(f"   {p.name:<32} {'(no detect)':<18} {f'{ref_w}x{ref_h}' if ref_w else '(no ref)':<18}")
    print()
    print(f"Within 15% of reference: {n_within}/{n_total}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\or\OneDrive - solidicon.com\Desktop\Test Files")
