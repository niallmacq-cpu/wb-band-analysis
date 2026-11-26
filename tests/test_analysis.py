"""Accuracy test of the analysis core (no GUI) against synthetic blots with known answers.

Runs the same steps as a region split in the app (detect, marker filter, row
alignment, ROI expansion, weak-candidate filter), measures every band with
each background method, pairs lanes, and compares the normalised
target/reference ratios with ground_truth.csv.

Run:  python tests/test_analysis.py      (exit code 1 if any check fails)
"""
import csv
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import tifffile

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
sys.path.insert(0, str(HERE.parent))
warnings.filterwarnings("ignore", category=FutureWarning)

from wb_analysis import (  # noqa: E402
    align_rois_to_region_signal, detect_bands_in_region, expand_band_rois,
    filter_bands_below_mean_signal, filter_marker_lanes, measure_all,
    normalize_target_to_reference_by_lane,
)

TOLERANCE = 0.05  # largest allowed relative error per lane, after normalising to control
# (x1, y1, x2, y2) regions around each protein row; the ladder lane is left out.
REGIONS = {"reference": (150, 370, 1070, 470), "target": (150, 250, 1070, 350)}
FILES = {"reference": "channel_reference.tif", "target": "channel_target.tif"}


def split_region(gray, region, polarity="light"):
    """The same detection pipeline the GUI runs after a region drag."""

    found, actual = detect_bands_in_region(gray, region, polarity=polarity)
    found, _marker = filter_marker_lanes(found, "auto")
    found = align_rois_to_region_signal(gray, found, region, polarity=actual)
    found = expand_band_rois(found, (0, 0, gray.shape[1], gray.shape[0]), padding=6, gap=0)
    found, _threshold = filter_bands_below_mean_signal(gray, found, polarity=actual, region=region)
    return found


def main() -> int:
    if not (DATA / FILES["reference"]).exists():  # synthetic blots are generated, not committed
        subprocess.run([sys.executable, "make_blot.py"], cwd=DATA, check=True)
    truth = np.array([float(r["normalised_to_control"]) for r in csv.DictReader(open(DATA / "ground_truth.csv"))])
    images = {key: tifffile.imread(DATA / name).astype(np.float32) for key, name in FILES.items()}

    failures = 0
    rois = {}
    uid = 1
    for key, gray in images.items():
        rois[key] = split_region(gray, REGIONS[key])
        # Unique IDs across both images, as in the app.
        for roi in rois[key]:
            roi.uid = uid
            uid += 1
        ok = len(rois[key]) == len(truth)
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'} {key}: {len(rois[key])} bands found (expected {len(truth)})")
    if failures:
        return 1

    for method in ("profile", "above_below", "ring"):
        measured = {key: measure_all(images[key], rois[key], "light", background_mode=method) for key in images}
        pairs = normalize_target_to_reference_by_lane(measured["target"], measured["reference"])
        ratios = np.array([p.target_over_reference for p in sorted(pairs, key=lambda p: p.lane)], dtype=float)
        normalised = ratios / ratios[:3].mean()  # lanes 1-3 are the controls
        error = np.abs(normalised / truth - 1)
        ok = len(ratios) == len(truth) and float(error.max()) < TOLERANCE
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'} {method:<12} max error {error.max() * 100:5.2f}%  "
              f"normalised {np.round(normalised, 3).tolist()}")
    print("PASSED" if not failures else f"{failures} check(s) FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
