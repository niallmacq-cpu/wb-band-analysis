"""Synthetic western blot generator with exact ground truth.

Ground truth = integrated band signal (sum of band-only pixel values, in raw
16-bit counts) BEFORE background, noise and saturation are added. Software
that does correct background subtraction should recover these values
(relative to each other) on the 'clean' image exactly and on the 'realistic'
image approximately.
"""
import numpy as np, tifffile, csv
from PIL import Image
from scipy.ndimage import gaussian_filter

rng = np.random.default_rng(42)

W, H = 1100, 700
N_LANES = 10
LANE_W = 70
LANE_PITCH = 100
X0 = 100  # centre of lane 1

# Lane design: ladder + 3 groups x 3 replicates
# 'load' = relative protein loaded (deliberately uneven so normalisation matters)
# 'fold' = true target expression relative to control (per unit protein)
lanes = [
    dict(name="Ladder",  group="Ladder", load=0.0,  fold=0.0),
    dict(name="Ctrl-1",  group="Control", load=1.00, fold=1.0),
    dict(name="Ctrl-2",  group="Control", load=0.85, fold=1.0),
    dict(name="Ctrl-3",  group="Control", load=1.15, fold=1.0),
    dict(name="TrtA-1",  group="TreatA",  load=0.90, fold=2.0),
    dict(name="TrtA-2",  group="TreatA",  load=1.10, fold=2.0),
    dict(name="TrtA-3",  group="TreatA",  load=0.75, fold=2.0),
    dict(name="TrtB-1",  group="TreatB",  load=1.20, fold=0.5),
    dict(name="TrtB-2",  group="TreatB",  load=0.95, fold=0.5),
    dict(name="TrtB-3",  group="TreatB",  load=1.05, fold=0.5),
]

# Band positions (y, px) — target ~50 kDa, reference (GAPDH-like) ~37 kDa
Y_TARGET, Y_REF = 300, 420
TARGET_UNIT = 2.4e7   # integrated counts per unit (load*fold)
REF_UNIT = 3.6e7      # integrated counts per unit load
LADDER = [(110, 250), (180, 150), (255, 100), (300, 75), (420, 50), (500, 37), (580, 25)]
# (y position, kDa label) — ladder positions; markers at 75 and 37 intentionally
# sit on the same rows as the target/reference bands

yy, xx = np.mgrid[0:H, 0:W].astype(float)


def band_shape(xc, yc, width, sigma_y, smile=0.0, skew=0.0):
    """Unit-integral band: flat-top lane profile x gaussian in y, with smile."""
    half = width / 2
    edge = 4.0
    px = 0.5 * (np.tanh((xx - (xc - half)) / edge) - np.tanh((xx - (xc + half)) / edge))
    # smile: band curves down at lane edges
    ycen = yc + smile * ((xx - xc) / half) ** 2
    # slightly asymmetric (tailing towards bottom)
    s = np.where(yy > ycen, sigma_y * (1 + skew), sigma_y)
    py = np.exp(-0.5 * ((yy - ycen) / s) ** 2)
    b = px * py
    return b / b.sum()


def build():
    target_img = np.zeros((H, W))
    ref_img = np.zeros((H, W))
    ladder_img = np.zeros((H, W))
    rows = []
    for i, ln in enumerate(lanes):
        xc = X0 + i * LANE_PITCH + rng.normal(0, 2)       # small lane jitter
        smile = 2.0 + 3.0 * abs(i - 4.5) / 4.5            # stronger smile at edges
        dy = rng.normal(0, 1.5)
        if ln["group"] == "Ladder":
            for y, kda in LADDER:
                ladder_img += 6e6 * band_shape(xc, y + dy, LANE_W * 0.9, 3.5, smile)
            continue
        t_true = TARGET_UNIT * ln["load"] * ln["fold"]
        r_true = REF_UNIT * ln["load"]
        target_img += t_true * band_shape(xc, Y_TARGET + dy, LANE_W, 7 + 2 * ln["load"], smile, 0.3)
        ref_img += r_true * band_shape(xc, Y_REF + dy, LANE_W, 6 + 2 * ln["load"], smile, 0.3)
        rows.append(dict(lane=i + 1, name=ln["name"], group=ln["group"],
                         load=ln["load"], true_fold=ln["fold"],
                         target_integrated=round(t_true), reference_integrated=round(r_true),
                         target_over_ref=t_true / r_true, x_centre_px=round(xc, 1)))
    # normalise to control mean
    ctrl = np.mean([r["target_over_ref"] for r in rows if r["group"] == "Control"])
    for r in rows:
        r["normalised_to_control"] = r["target_over_ref"] / ctrl
    return target_img, ref_img, ladder_img, rows


def background():
    """Uneven membrane background: gradient + vignetting + blotchy patches."""
    bg = 900 + 500 * (xx / W) + 300 * (yy / H)
    bg += 400 * np.exp(-(((xx - W * 0.7) / 300) ** 2 + ((yy - H * 0.3) / 200) ** 2))
    blotch = gaussian_filter(rng.normal(0, 1, (H, W)), 25)
    bg += 250 * blotch / blotch.std()
    # vertical streak per lane (smear) — mild
    for i in range(N_LANES):
        xc = X0 + i * LANE_PITCH
        bg += 120 * np.exp(-0.5 * ((xx - xc) / 25) ** 2) * (yy / H)
    # a couple of specks/dust
    for _ in range(25):
        cx, cy = rng.uniform(0, W), rng.uniform(0, H)
        bg += rng.uniform(800, 4000) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * rng.uniform(1, 2.5) ** 2))
    return bg


def realistic(signal, bg, gain=1.0, saturate=65535):
    img = signal + bg
    img = rng.poisson(np.clip(img * gain, 0, None)) / gain   # shot noise
    img = img + rng.normal(0, 60, img.shape)                   # read noise
    return np.clip(img, 0, saturate)


def save(arr, stem):
    a16 = np.clip(arr, 0, 65535).astype(np.uint16)
    tifffile.imwrite(f"{stem}.tif", a16)                          # bright bands on dark (raw)
    tifffile.imwrite(f"{stem}_inverted.tif", (65535 - a16).astype(np.uint16))  # dark bands on light
    # 8-bit preview, inverted, auto-contrast
    lo, hi = np.percentile(arr, [0.5, 99.8])
    p = np.clip((arr - lo) / (hi - lo), 0, 1) ** 0.7
    Image.fromarray((255 * (1 - p)).astype(np.uint8)).save(f"{stem}_preview.png")


if __name__ == "__main__":
    t, r, lad, rows = build()
    sig = t + r + lad
    bg = background()

    save(sig, "blot_clean")                      # no background, no noise
    save(realistic(sig, bg), "blot_realistic")   # background + noise
    # Saturated version: same blot, 3x exposure, clipped at 16-bit max
    save(realistic(3 * sig, 3 * bg), "blot_saturated")
    # Separate channels, useful for 2-colour (fluorescent-style) software
    save(realistic(t + lad, bg), "channel_target")
    save(realistic(r + lad, bg), "channel_reference")

    # Report saturation in the overexposed image
    sat = realistic(3 * sig, 3 * bg)
    with open("ground_truth.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + ["peak_target_clean", "peak_ref_clean"])
        w.writeheader()
        for row in rows:
            x = int(round(row["x_centre_px"]))
            row["peak_target_clean"] = round(t[Y_TARGET - 30:Y_TARGET + 30, x - 35:x + 35].max())
            row["peak_ref_clean"] = round(r[Y_REF - 30:Y_REF + 30, x - 35:x + 35].max())
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    print("saturated pixels in blot_saturated:", int((sat >= 65535).sum()))
    for row in rows:
        print(row["name"], row["target_integrated"], row["reference_integrated"],
              round(row["normalised_to_control"], 3), row["peak_target_clean"], row["peak_ref_clean"])
