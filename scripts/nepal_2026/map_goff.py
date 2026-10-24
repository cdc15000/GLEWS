"""Map NISAR GOFF offsets for one pair around the source, with outlines.

Usage: python scripts/nepal_2026/map_goff.py [GOFF_DIR] [REF] [SEC] [LAYER]

Shows whether the A1 signal sits inside the source outline or leaks in
from the glacier east of it. Overlays the ICIMOD polygon and, if present,
the post-event scar (output/acceptance/scar_mask_10m.npy, from posthoc_s2.py).
"""
import sys, json, yaml, numpy as np
from pathlib import Path
from matplotlib.path import Path as MP
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).parent))
from glews.nisar import read_goff_pairs
from utm import fwd, inv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

a = sys.argv[1:] + [None] * 4
GOFF = Path(a[0] or ROOT / "data/nisar/nepal_2026/goff_t98_crop")
REF, SEC, LAYER = a[1] or "20260726", a[2] or "20260819", a[3] or "layer2"
site = yaml.safe_load(open(ROOT / "config/nepal_2026.yaml"))["site"]
poly = np.array([fwd(la, lo) for la, lo in site["known_source"]["polygon"]])
cx, cy = fwd(site["latitude"], site["longitude"])
probe = read_goff_pairs(GOFF, layer=LAYER, min_snr=0, track=98)[0]
X, Y = np.meshgrid(probe["x_coords"], probe["y_coords"])
src = MP(poly).contains_points(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
d = np.hypot(X - cx, Y - cy)
ring = (d > 2000) & (d < 5000) & ~src
p = [q for q in read_goff_pairs(GOFF, layer=LAYER, min_snr=5, track=98, reference_mask=ring)
     if (q["ref_date"], q["sec_date"]) == (REF, SEC)][0]
box = (np.abs(X - cx) < 3000) & (np.abs(Y - cy) < 3000)
rr, cc = np.where(box.any(1))[0], np.where(box.any(0))[0]
sl = (slice(rr[0], rr[-1] + 1), slice(cc[0], cc[-1] + 1))
ext = [X[sl][0, 0] - 40, X[sl][0, -1] + 40, Y[sl][-1, 0] - 40, Y[sl][0, 0] + 40]
scar_path = ROOT / "output/acceptance/scar_mask_10m.npy"
fig, axs = plt.subplots(1, 2, figsize=(11, 5))
for ax, key, lab in ((axs[0], "range_offset", "slant range (m, + = away)"), (axs[1], "along_track_offset", "along-track (m)")):
    v = p[key][sl]; lim = 1.5           # fixed: ridge-line blunders reach tens of metres
    im = ax.imshow(v, extent=ext, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.plot(*np.vstack([poly, poly[:1]]).T, c="k", lw=1)
    if scar_path.exists():
        s = np.load(scar_path)
        z = np.load(ROOT / "data/sentinel2/nepal_2026/scar/2025-10-21.npz")["B08_transform"]
        SX, SY = np.meshgrid(z[0] + 10 * (np.arange(s.shape[1]) + .5), z[3] - 10 * (np.arange(s.shape[0]) + .5))
        ax.contour(SX, SY, s, [0.5], colors="lime", linewidths=1)
    ax.set_xlim(ext[:2]); ax.set_ylim(ext[2:])
    ax.set_title(f"GOFF {REF}→{SEC} {LAYER} SNR≥5: {lab}", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
out = ROOT / f"output/acceptance/goff_map_{REF}_{SEC}_{LAYER}.png"; fig.savefig(out, dpi=110)
# Where is the peak? Median range offset in the polygon vs. a 600 m band east of it
east = (~src) & (X > poly[:, 0].max() - 200) & (X < poly[:, 0].max() + 600) & (Y > poly[:, 1].min()) & (Y < poly[:, 1].max())
print(json.dumps({"inside_polygon_m": round(float(np.nanmedian(p["range_offset"][src])), 3),
                  "east_band_m": round(float(np.nanmedian(p["range_offset"][east])), 3),
                  "noise_sigma_m": [round(x, 3) for x in p["noise_sigma"]], "figure": str(out)}))
