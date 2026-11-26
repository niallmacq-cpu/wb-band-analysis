"""Same bands as make_blot.py (identical ground truth), on strongly variable backgrounds."""
import numpy as np, csv, tifffile
from scipy.ndimage import gaussian_filter
import make_blot as mb   # seeded rng at import -> build() gives identical bands

t, r, lad, rows = mb.build()
sig = t + r + lad
xx, yy, H, W = mb.xx, mb.yy, mb.H, mb.W
rng = np.random.default_rng(7)


def variable_bg(level):
    bg = np.full((H, W), 1500.0)
    # steep diagonal gradient (left-top dark -> right-bottom bright)
    bg += 6000 * level * (0.6 * xx / W + 0.4 * yy / H)
    # large high-background cloud sitting over the target bands of lanes 6-8
    bg += 9000 * level * np.exp(-(((xx - 700) / 160) ** 2 + ((yy - 290) / 110) ** 2))
    # darker hole over reference band of lanes 2-3 (over-stripped area)
    bg -= 1200 * level * np.exp(-(((xx - 250) / 120) ** 2 + ((yy - 430) / 80) ** 2))
    # horizontal smear between target and reference rows (non-specific band)
    bg += 2500 * level * np.exp(-0.5 * ((yy - 360) / 18) ** 2) * (0.5 + 0.5 * np.sin(xx / 140))
    # per-lane vertical streaks of random strength, increasing down the lane
    for i in range(mb.N_LANES):
        xc = mb.X0 + i * mb.LANE_PITCH
        amp = rng.uniform(0, 3500) * level
        bg += amp * np.exp(-0.5 * ((xx - xc) / 30) ** 2) * (yy / H) ** 1.5
    # large-scale mottling
    m = gaussian_filter(rng.normal(0, 1, (H, W)), 40)
    bg += 1500 * level * m / m.std()
    # fine mottling
    m = gaussian_filter(rng.normal(0, 1, (H, W)), 8)
    bg += 400 * level * m / m.std()
    # dust / speckles
    for _ in range(int(40 * level) + 10):
        cx, cy = rng.uniform(0, W), rng.uniform(0, H)
        bg += rng.uniform(2000, 15000) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * rng.uniform(1, 3) ** 2))
    return np.clip(bg, 200, None)


out = []
for name, level in [("mild", 0.4), ("moderate", 1.0), ("severe", 2.0)]:
    bg = variable_bg(level)
    img = mb.realistic(sig, bg, saturate=65535)
    mb.save(img, f"varbg_{name}")
    tifffile.imwrite(f"varbg_{name}_background_only.tif", np.clip(bg, 0, 65535).astype(np.uint16))
    for row in rows:
        x = int(round(row["x_centre_px"]))
        for prot, y, tru, img_ in [("target", mb.Y_TARGET, row["target_integrated"], t),
                                   ("reference", mb.Y_REF, row["reference_integrated"], r)]:
            box = (slice(y - 40, y + 40), slice(x - 40, x + 40))
            bgbox = bg[box]
            raw_sum = img[box].sum()
            out.append(dict(level=name, lane=row["lane"], name=row["name"], protein=prot,
                            true_integrated=tru,
                            true_bg_mean_under_band=round(bgbox.mean(), 1),
                            bg_top_minus_bottom=round(bg[y - 40, x] - bg[y + 39, x], 1),
                            bg_left_minus_right=round(bg[y, x - 40] - bg[y, x + 39], 1),
                            bg_fraction_of_box_signal=round(bgbox.sum() / raw_sum, 3)))
    print(name, "saturated px:", int((img >= 65535).sum()),
          "bg range:", int(bg.min()), int(bg.max()))

with open("varbg_background_truth.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader(); w.writerows(out)
