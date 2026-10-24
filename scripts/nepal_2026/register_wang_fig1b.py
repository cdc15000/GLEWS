"""Georeference Wang et al. (2026) Fig. 1b and digitise their measurement area.

Usage (from the repository root):
    python scripts/nepal_2026/register_wang_fig1b.py FIG1B_PNG [OUT_YAML]

FIG1B_PNG is panel (b) of Fig. 1 of Wang et al. (2026), Landslides 23:3489,
doi:10.1007/s10346-026-02842-6, rendered from the PDF (Sentinel-2,
12 Aug 2026). The paper itself is not redistributed. The panel is matched
to our own 12 Aug 2026 Sentinel-2 crop (same date; orbit 119) by
normalised cross-correlation over scale and position, north-up, with the
red/blue overlays, labels and arrow masked out. The blue "measurement area"
and red "collapse area" outlines are then traced from their colours and
written as UTM (EPSG:32645) and lat/lon vertices.
"""
import sys, yaml, numpy as np
from pathlib import Path
from PIL import Image
from scipy.signal import fftconvolve
from scipy import ndimage as ndi
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from utm import inv

fig = np.asarray(Image.open(sys.argv[1]).convert("RGB")).astype(float)
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "output/acceptance/wang_fig1b_outlines.yaml"
# Map frame of panel (b) inside the crop: find the black border
dark = fig.sum(axis=2) < 120
rows = np.where(dark.mean(axis=1) > 0.6)[0]; cols = np.where(dark.mean(axis=0) > 0.6)[0]
top = rows[rows < fig.shape[0] * 0.2].max() + 3; bot = rows[(rows > fig.shape[0] * 0.5) & (rows < fig.shape[0] * 0.8)].min() - 3
left = cols[cols < fig.shape[1] * 0.2].max() + 3 if (cols < fig.shape[1] * 0.2).any() else 3
right = cols[cols > fig.shape[1] * 0.8].min() - 3
panel = fig[top:bot, left:right]
R, G, Bc = panel[..., 0], panel[..., 1], panel[..., 2]
red = (R > 180) & (G < 80) & (Bc < 80)
blue_line = (Bc > 180) & (R < 80) & (G < 140)
blue_fill = (Bc - R > 25) & (G - R > 10) & (Bc > 120)
white = (R > 225) & (G > 225) & (Bc > 225)
overlay = ndi.binary_dilation(red | blue_line | blue_fill | white, iterations=4)
# Mask the label/north-arrow corners
h, w = panel.shape[:2]
overlay[: int(.11 * h), :] = True
overlay[int(.8 * h):, int(.85 * w):] = True
overlay[int(.53 * h):int(.80 * h), int(.62 * w):int(.86 * w)] = True   # "Bedrock" arrow and label
gray_fig = panel.mean(axis=2)

z = np.load(ROOT / "data/sentinel2/nepal_2026/scar/2026-08-12_rgb.npz")
img = np.dstack([z["B04"], z["B03"], z["B02"]]).astype(float).mean(axis=2)
t = z["B04_transform"]


def ncc(image, tmpl, mask):
    """Masked NCC of tmpl (valid where mask) at every offset in image ('valid' mode)."""
    m = (~mask).astype(float)
    tm = tmpl * m
    n = m.sum()
    tmean = tm.sum() / n
    tz = (tmpl - tmean) * m
    tnorm = np.sqrt((tz ** 2).sum())
    f = lambda a, k: fftconvolve(a, k[::-1, ::-1], mode="valid")
    s1 = f(image, m); s2 = f(image ** 2, m)
    num = f(image, tz)
    var = np.maximum(s2 - s1 ** 2 / n, 1e-9)
    return num / (np.sqrt(var) * tnorm)


best = None
for scale in np.arange(3.0, 6.01, 0.05):              # figure pixels per 10 m S2 pixel
    th, tw = int(h / scale), int(w / scale)
    tmpl = np.asarray(Image.fromarray(gray_fig.astype(np.uint8)).resize((tw, th), Image.BILINEAR), float)
    msk = np.asarray(Image.fromarray(overlay.astype(np.uint8) * 255).resize((tw, th), Image.NEAREST)) > 0
    c = ncc(img, tmpl, msk)
    k = np.unravel_index(np.argmax(c), c.shape)
    if best is None or c[k] > best[0]:
        best = (float(c[k]), scale, k, (th, tw))
score, scale, (r0, c0), (th, tw) = best
print(f"best NCC {score:.3f} at scale {scale:.2f} fig px per S2 px ({10 / scale:.2f} m per fig px), "
      f"offset row {r0} col {c0}; panel {tw * 10 / 1000:.2f} x {th * 10 / 1000:.2f} km")


def fig_to_utm(py, px):
    """Panel pixel (row, col) -> UTM, via the best scale and offset."""
    srow = r0 + (py + 0.5) / scale - 0.5; scol = c0 + (px + 0.5) / scale - 0.5
    return t[0] + t[1] * (scol + 0.5), t[3] + t[5] * (srow + 0.5)


def outline(mask):
    """Ordered polygon around a filled mask (convex for the triangle, boundary trace otherwise)."""
    lab, n = ndi.label(mask)
    big = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    big = ndi.binary_fill_holes(big)
    from matplotlib import pyplot as plt
    cs = plt.contour(big.astype(float), [0.5]); plt.close()
    seg = max(cs.allsegs[0], key=len)          # (x=col, y=row)
    step = max(1, len(seg) // 40)
    return seg[::step]


tri = outline(blue_fill | blue_line)
col_poly = ndi.binary_erosion(ndi.binary_fill_holes(ndi.binary_dilation(red, iterations=6)), iterations=6)
coll = outline(col_poly)
res = {"source": "Wang et al. 2026, Landslides 23:3489, Fig. 1b (doi:10.1007/s10346-026-02842-6)",
       "registration": {"ncc": round(score, 3), "metres_per_fig_px": round(float(10 / scale), 3),
                        "reference": str(z["scene"])}}
for name, seg in (("measurement_area", tri),):   # their red collapse line runs under the blue edge, so it does not trace closed
    pts = [fig_to_utm(y, x) for x, y in seg]
    lat, lon = inv(np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
    res[name] = {"utm_45n": [[round(float(a), 1), round(float(b), 1)] for a, b in pts],
                 "latlon": [[round(float(a), 5), round(float(b), 5)] for a, b in zip(lat, lon)]}
    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    area = 0.5 * abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))
    res[name]["area_km2"] = round(float(area) / 1e6, 3)
    print(f"{name}: {len(pts)} vertices, area {area / 1e6:.3f} km2, centroid {lat.mean():.4f}N {lon.mean():.4f}E")
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(yaml.safe_dump(res, sort_keys=False))
print("wrote", OUT)

# Check figure: our 12 Aug 2026 image with the traced outlines and the ICIMOD polygon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utm import fwd
site = yaml.safe_load(open(ROOT / "config/nepal_2026.yaml"))["site"]
icimod = np.array([fwd(a, b) for a, b in site["known_source"]["polygon"]])
rgb = np.dstack([z["B04"], z["B03"], z["B02"]]).astype(float)
rgb = np.clip(rgb / np.percentile(rgb[400:700, 400:700], 99.5), 0, 1) ** 0.8
ext = [t[0], t[0] + t[1] * rgb.shape[1], t[3] + t[5] * rgb.shape[0], t[3]]
fig2, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(panel.astype(np.uint8)); axs[0].set_title("Wang et al. Fig. 1b (input panel)", fontsize=9)
ax = axs[1]
ax.imshow(rgb, extent=ext)
for name, colr in (("measurement_area", "cyan"),):
    P = np.array(res[name]["utm_45n"]); ax.plot(*np.vstack([P, P[:1]]).T, c=colr, lw=1.2, label=f"Wang {name.replace('_', ' ')}")
ax.plot(*np.vstack([icimod, icimod[:1]]).T, c="yellow", lw=1, ls="--", label="ICIMOD (config)")
x0, y0 = fig_to_utm(0, 0); x1, y1 = fig_to_utm(h, w)
ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.legend(fontsize=7, loc="lower left")
ax.set_title(f"Our S2 12 Aug 2026, registered extent (NCC {score:.2f})", fontsize=9)
fig2.tight_layout(); fig2.savefig(OUT.with_suffix(".png"), dpi=100)
