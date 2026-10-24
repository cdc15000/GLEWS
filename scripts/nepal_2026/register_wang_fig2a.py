"""Georeference Wang et al. (2026) Fig. 2a and trace their main-glacier and stable boxes.

Usage (from the repository root):
    python scripts/nepal_2026/register_wang_fig2a.py FIG2AB_PNG [OUT_YAML]

FIG2AB_PNG is the top strip of Fig. 2 (insets a and b plus the legend with
the 0-2 km scale bar), rendered from the PDF of Wang et al. (2026),
Landslides 23:3489, doi:10.1007/s10346-026-02842-6 (not redistributed).

Inset (a) is placed north-up with:
- scale from the 2 km scale bar, cross-checked against the size of the
  blue measurement triangle;
- position from the blue triangle, whose map coordinates come from Fig. 1b
  (site.reference_areas[0] in config/nepal_2026.yaml).

It then traces the black "Measurement Area (Glacier)" box and the purple
"Measurement Area (Stable Area)" box.
"""
import sys, yaml, numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from utm import fwd, inv

im = np.asarray(Image.open(sys.argv[1]).convert("RGB")).astype(int)
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "output/acceptance/wang_fig2a_boxes.yaml"
H, W = im.shape[:2]
R, G, B = im[..., 0], im[..., 1], im[..., 2]

# Scale bar: the long black bar in the legend's lower right (0-1 km black, 1-2 km white)
legend = slice(int(.75 * H), int(.85 * H))
blk = (R < 60) & (G < 60) & (B < 60)
rows = np.where(blk[legend].sum(axis=1) > W * 0.08)[0]
bar_row = int(.75 * H) + rows[len(rows) // 2]
xs = np.where(blk[bar_row])[0]
xs = xs[xs > W * 0.5]
# outline spans 0-2 km: first to last dark pixel on the bar row
km2_px = xs.max() - xs.min()
m_per_px_bar = 2000.0 / km2_px

# Inset (a): left map frame (dark border)
frame_cols = np.where(blk[: int(.6 * H)].sum(axis=0) > 0.5 * .6 * H)[0]
a_left, a_right = frame_cols[frame_cols < W * .45].min(), frame_cols[frame_cols < W * .45].max()
frame_rows = np.where(blk[:, a_left:a_right].sum(axis=1) > 0.6 * (a_right - a_left))[0]
a_top, a_bot = frame_rows.min(), frame_rows[frame_rows < .7 * H].max()
inset = im[a_top:a_bot, a_left:a_right]
r, g, b = inset[..., 0], inset[..., 1], inset[..., 2]

blue = (b > 170) & (r < 90) & (g > 60) & (g < 170)
lab, n = ndi.label(ndi.binary_dilation(blue, iterations=1))
tri = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
ty, tx = np.nonzero(tri)
site = yaml.safe_load(open(ROOT / "config/nepal_2026.yaml"))["site"]
T = np.array([fwd(a, c) for a, c in site["reference_areas"][0]["polygon"]])
tri_h_m = T[:, 1].max() - T[:, 1].min()
m_per_px_tri = tri_h_m / (ty.max() - ty.min())
print(f"scale: bar {m_per_px_bar:.2f} m/px, triangle height {m_per_px_tri:.2f} m/px")
s = m_per_px_bar
# Anchor: bounding-box centre of the triangle (figure) <-> of the georeferenced triangle
fx, fy = (tx.min() + tx.max()) / 2, (ty.min() + ty.max()) / 2
ux, uy = (T[:, 0].min() + T[:, 0].max()) / 2, (T[:, 1].min() + T[:, 1].max()) / 2


def to_utm(px, py):
    return ux + (px - fx) * s, uy - (py - fy) * s


def box(mask):
    """Rotated-rectangle corners of the largest outline in mask (PCA of its pixels)."""
    # All outline pixels of the box (its four sides may not join into one component)
    lab, n = ndi.label(ndi.binary_dilation(mask, iterations=3))
    k = np.argmax(np.bincount(lab.ravel())[1:]) + 1
    yy, xx = np.nonzero(mask & (lab == k))
    P = np.c_[xx, yy].astype(float); c = P.mean(axis=0)
    u, sv, vt = np.linalg.svd(P - c, full_matrices=False)
    q = (P - c) @ vt.T
    lo, hi = q.min(axis=0), q.max(axis=0)
    corners = [c + vt.T @ np.array([a, b_]) for a, b_ in ((lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1]))]
    return corners


# Black box near the middle of the inset (exclude the frame)
hh, ww = inset.shape[:2]
black = (r < 50) & (g < 50) & (b < 50)
black[: int(.2 * hh)] = False; black[int(.8 * hh):] = False
black[:, : int(.5 * ww)] = False; black[:, int(.9 * ww):] = False
purple = (r > 130) & (b > 170) & (g < 90)
res = {"source": "Wang et al. 2026, Landslides 23:3489, Fig. 2a (doi:10.1007/s10346-026-02842-6)",
       "registration": {"metres_per_fig_px_scale_bar": round(float(m_per_px_bar), 3),
                        "metres_per_fig_px_triangle": round(float(m_per_px_tri), 3),
                        "anchor": "blue measurement triangle from Fig. 1b"}}
for name, mask in (("main_glacier_box", black), ("stable_box", purple)):
    cs = box(mask)
    pts = [to_utm(x, y) for x, y in cs]
    lat, lon = inv(np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
    L = [np.hypot(*(np.array(pts[i]) - np.array(pts[(i + 1) % 4]))) for i in range(4)]
    res[name] = {"utm_45n": [[round(float(a), 1), round(float(b_), 1)] for a, b_ in pts],
                 "latlon": [[round(float(a), 5), round(float(b_), 5)] for a, b_ in zip(lat, lon)],
                 "sides_m": [round(float(x)) for x in L]}
    print(f"{name}: centre {lat.mean():.4f}N {lon.mean():.4f}E, sides {[round(float(x)) for x in L]} m")
OUT.write_text(yaml.safe_dump(res, sort_keys=False))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
z = np.load(ROOT / "data/sentinel2/nepal_2026/scar/2026-08-12_rgb.npz"); t = z["B04_transform"]
rgb = np.dstack([z["B04"], z["B03"], z["B02"]]).astype(float)
rgb = np.clip(rgb / np.percentile(rgb[400:700, 400:700], 99.5), 0, 1) ** 0.8
fig, axs = plt.subplots(1, 2, figsize=(12, 5.6))
axs[0].imshow(inset.astype(np.uint8)); axs[0].set_title("Wang et al. Fig. 2a (input)", fontsize=9)
ax = axs[1]
ax.imshow(rgb, extent=[t[0], t[0] + 11000, t[3] - 11000, t[3]])
x0, y0 = to_utm(0, 0); x1, y1 = to_utm(ww, hh)
ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
ax.plot(*np.vstack([T, T[:1]]).T, c="cyan", lw=1, label="measurement area (Fig. 1b)")
for name, col in (("main_glacier_box", "k"), ("stable_box", "m")):
    P = np.array(res[name]["utm_45n"]); ax.plot(*np.vstack([P, P[:1]]).T, c=col, lw=1.4, label=name.replace("_", " "))
ic = np.array([fwd(a, c) for a, c in site["known_source"]["polygon"]])
ax.plot(*np.vstack([ic, ic[:1]]).T, c="yellow", lw=1, ls="--", label="ICIMOD outline")
ax.legend(fontsize=7); ax.set_title("Placed on our 12 Aug 2026 Sentinel-2 image", fontsize=9)
fig.tight_layout(); fig.savefig(OUT.with_suffix(".png"), dpi=100)
print("wrote", OUT)
