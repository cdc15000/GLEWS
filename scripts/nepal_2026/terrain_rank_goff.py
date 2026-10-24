"""Phase 1b (docs/acceptance_phase1b.md): rank the GOFF source signal against terrain-matched patches.

Usage (from the repository root):
    python scripts/nepal_2026/terrain_rank_goff.py [GOFF_DIR]

Settings are those pre-registered in docs/acceptance_phase1b.md. Writes
output/acceptance/B_terrain_rank.json and a map of the terrain-like mask.
"""
import sys, json, yaml, numpy as np
from pathlib import Path
from matplotlib.path import Path as MP
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).parent))
from glews.nisar import read_goff_pairs
from glews.optical import GDAL_HTTP_ENV
from utm import fwd

GOFF = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/nisar/nepal_2026/goff_t98_crop"
LAYER, MIN_SNR, PRIMARY = "layer2", 5.0, ("20260726", "20260819")
SLOPE_TOL, ASPECT_TOL, TERRAIN_FRAC, VALID_FRAC = 7.5, 30.0, 0.7, 0.5
N_CTRL, MAX_DRAWS, MIN_CTRL, PCTL = 500, 200_000, 30, 95.0
DEM = ("https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N28_00_E085_00_DEM/"
       "Copernicus_DSM_COG_10_N28_00_E085_00_DEM.tif")
OUT = ROOT / "output/acceptance"; OUT.mkdir(parents=True, exist_ok=True)

site = yaml.safe_load(open(ROOT / "config/nepal_2026.yaml"))["site"]
poly = np.array([fwd(la, lo) for la, lo in site["known_source"]["polygon"]])
cx, cy = fwd(site["latitude"], site["longitude"])

probe = read_goff_pairs(GOFF, layer=LAYER, min_snr=0, track=98)[0]
x, y = probe["x_coords"], probe["y_coords"]
X, Y = np.meshgrid(x, y)
src = MP(poly).contains_points(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
d = np.hypot(X - cx, Y - cy)
ring = (d > 2000) & (d < 5000) & ~src

# ---- Terrain on the GOFF grid --------------------------------------------------
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.transform import from_origin
post = float(x[1] - x[0])
assert y[1] < y[0], "expects north-up grid"
with rasterio.Env(**GDAL_HTTP_ENV), rasterio.open(DEM) as ds:
    tr = from_origin(x[0] - post / 2, y[0] + post / 2, post, post)
    with WarpedVRT(ds, crs="EPSG:32645", transform=tr, width=len(x), height=len(y),
                   resampling=Resampling.average) as v:
        z = v.read(1).astype(float)
g_row, g_col = np.gradient(z, post)
dzde, dzdn = g_col, -g_row                       # rows increase southward
slope = np.degrees(np.arctan(np.hypot(dzde, dzdn)))
aspect = np.degrees(np.arctan2(-dzde, -dzdn)) % 360   # downslope azimuth
s0 = float(np.median(slope[src]))
a_rad = np.radians(aspect[src])
a0 = float(np.degrees(np.arctan2(np.sin(a_rad).mean(), np.cos(a_rad).mean())) % 360)
dasp = np.abs((aspect - a0 + 180) % 360 - 180)
terrain = (np.abs(slope - s0) <= SLOPE_TOL) & (dasp <= ASPECT_TOL)
print(f"source: {src.sum()} px, slope {s0:.1f} deg, aspect {a0:.0f} deg; "
      f"terrain-like pixels outside source: {(terrain & ~src).sum()} of {(~src).size - src.sum()}")

# Candidate translations: fixed by terrain and overlap only (independent of the pair)
r, c = np.nonzero(src)
H, W = src.shape
rng = np.random.default_rng(0)
cands = []
seen = set()
for _ in range(MAX_DRAWS):
    dr = int(rng.integers(-r.min(), H - r.max())); dc = int(rng.integers(-c.min(), W - c.max()))
    if (dr, dc) in seen:
        continue
    seen.add((dr, dc))
    rr, cc = r + dr, c + dc
    if src[rr, cc].any() or terrain[rr, cc].mean() < TERRAIN_FRAC:
        continue
    cands.append((rr, cc))
print(f"terrain-matched translations available: {len(cands)} (from {len(seen)} distinct draws)")

pairs = read_goff_pairs(GOFF, layer=LAYER, min_snr=MIN_SNR, track=98, reference_mask=ring)
rows = []
for p in pairs:
    f = p["range_offset"]
    sig = float(np.nanmedian(f[src]))
    meds = []
    for rr, cc in cands:
        v = f[rr, cc]
        if np.isfinite(v).mean() >= VALID_FRAC:
            meds.append(float(np.nanmedian(v)))
        if len(meds) == N_CTRL:
            break
    meds = np.array(meds)
    rank = float(100 * np.mean(meds < sig)) if len(meds) else float("nan")
    rows.append(dict(ref=p["ref_date"], sec=p["sec_date"], days=p["days"], source_m=round(sig, 3),
                     n_controls=len(meds), rank_pct=round(rank, 1),
                     controls_median_m=round(float(np.median(meds)), 3) if len(meds) else None,
                     controls_p95_m=round(float(np.percentile(meds, 95)), 3) if len(meds) else None))
    # POST HOC (not pre-registered; added because no matched patch exists):
    # do scattered terrain-like pixels away from the source show the same offset?
    far = terrain & ~src & (np.hypot(X - cx, Y - cy) > 1000) & np.isfinite(f)
    near_ok = src & np.isfinite(f)
    tl = f[far]
    rows[-1].update(posthoc_terrain_px=int(far.sum()),
                    posthoc_terrain_median_m=round(float(np.median(tl)), 3) if tl.size else None,
                    posthoc_noise_sigma_m=round(float(p["noise_sigma"][0]), 3),
                    posthoc_source_terrain_frac=round(float(terrain[src].mean()), 2))
    q = rows[-1]
    ctrl = (f"controls median {q['controls_median_m']:+.3f} p95 {q['controls_p95_m']:+.3f} rank {q['rank_pct']:5.1f}%"
            if q["n_controls"] else "no matched controls")
    tmed = q["posthoc_terrain_median_m"]
    print(f"{q['ref']}-{q['sec']}  source {q['source_m']:+.3f} m  {ctrl}  n={q['n_controls']} | "
          f"post hoc: terrain-like px >1 km away (n={q['posthoc_terrain_px']}) median "
          f"{'n/a' if tmed is None else f'{tmed:+.3f}'} m, sigma {q['posthoc_noise_sigma_m']:.3f}")

prim = next(q for q in rows if (q["ref"], q["sec"]) == PRIMARY)
if prim["n_controls"] < MIN_CTRL:
    verdict = "insufficient controls"
else:
    verdict = "PASS" if prim["rank_pct"] >= PCTL else "FAIL"
n_high = sum(1 for q in rows if q["n_controls"] >= MIN_CTRL and q["rank_pct"] >= PCTL)
res = dict(criteria="docs/acceptance_phase1b.md",
           terrain=dict(source_slope_deg=round(s0, 1), source_aspect_deg=round(a0, 0),
                        slope_tol=SLOPE_TOL, aspect_tol=ASPECT_TOL,
                        terrain_like_px_outside_source=int((terrain & ~src).sum()),
                        matched_translations=len(cands)),
           B1=dict(pair=PRIMARY, verdict=verdict, **prim),
           B2=dict(pairs_ranked_ge_95=n_high, n_pairs=len(rows), rows=rows),
           posthoc_note="posthoc_* fields were added after B1 returned no matched controls; not pre-registered")
(OUT / "B_terrain_rank.json").write_text(json.dumps(res, indent=1))
print(f"\nB1 {verdict}: rank {prim['rank_pct']}% of {prim['n_controls']} matched controls; "
      f"source pixels terrain-like: {terrain[src].mean():.0%}")
print(f"B2: source ranks >= {PCTL:g}% in {n_high} of {len(rows)} pairs")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 5.4))
ext = [x[0] - post / 2, x[-1] + post / 2, y[-1] - post / 2, y[0] + post / 2]
ax.imshow(np.where(terrain, 1.0, np.nan), extent=ext, cmap="Greens", vmin=0, vmax=1.5)
ax.contour(X, Y, src, [0.5], colors="k", linewidths=1)
cover = np.zeros_like(src, float)
for rr, cc in cands[:N_CTRL]:
    cover[rr, cc] += 1
ax.contour(X, Y, cover > 0, [0.5], colors="tab:orange", linewidths=0.6)
ax.set_title(f"Terrain like the source (slope {s0:.0f}±{SLOPE_TOL:g}°, aspect {a0:.0f}±{ASPECT_TOL:g}°)\n"
             "green; source black; area covered by matched controls orange", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "B_terrain_mask.png", dpi=110)
print("wrote", OUT / "B_terrain_rank.json")
