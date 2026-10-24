"""DEVELOPMENT diagnosis of Phase 1e false alarms (Nepal is the development set).

Usage (from the repository root):
    python -u scripts/nepal_2026/diag_false_alarms.py [GOFF_DIR]

Not a test: no pass/fail. Describes what the false alarms are, to choose
Phase 2 filters that are then tested on independent control sites.

- E1: flags per cutoff split by sub-detector (acceleration vs step-change).
- E2: each final false-alarm cluster (and the target) tagged with terrain
  (GLO-30 slope, aspect, elevation), surface (snow/ice fraction from the
  21 Oct 2025 Sentinel-2 SCL), GOFF SNR, distance from the source, and
  direction consistency: per alarm pixel and pair, the slope-parallel
  azimuth from range + along-track offsets vs the downslope aspect.
Writes output/acceptance/diag_false_alarms.json.
"""
import sys, json, logging, tempfile, yaml, numpy as np
from pathlib import Path
from matplotlib.path import Path as MP
from scipy import ndimage as ndi
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).parent))
from glews.nisar import load_nisar_goff_stack, read_goff_pairs
from glews.timeseries import compute_acceleration_map
from glews.detect import detect_anomalies
from glews.optical import GDAL_HTTP_ENV
from glews.fusion import los_enu
from utm import fwd, inv

logging.basicConfig(level=logging.ERROR)
GOFF = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/nisar/nepal_2026/goff_t98_crop"
OUT = ROOT / "output/acceptance"
cfg = yaml.safe_load(open(ROOT / "config/nepal_2026.yaml")); site = cfg["site"]
cx, cy = fwd(site["latitude"], site["longitude"])
icimod = np.array([fwd(a, b) for a, b in site["known_source"]["polygon"]])
outlet = np.array([fwd(a, b) for a, b in site["reference_areas"][0]["polygon"]])
files = sorted(GOFF.glob("NISAR_*GOFF*.h5"), key=lambda p: p.name.split("_")[13])
probe = read_goff_pairs(GOFF, layer="layer2", min_snr=0, track=98)[0]
x, y = probe["x_coords"], probe["y_coords"]
X, Y = np.meshgrid(x, y)
inside = lambda P: MP(P).contains_points(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
src = inside(icimod); target = src | inside(outlet)
ring = (np.hypot(X - cx, Y - cy) > 2000) & (np.hypot(X - cx, Y - cy) < 5000) & ~src
LAT, LON = inv(X, Y)
pairs = read_goff_pairs(GOFF, layer="layer2", min_snr=5, track=98, reference_mask=ring)
raw = read_goff_pairs(GOFF, layer="layer2", min_snr=0, track=98)
snr = np.array([p["snr"] for p in raw])

# ---- E1 by sub-detector ------------------------------------------------------------
print("E1 flags by sub-detector (tag) per cutoff:")
e1 = []
for cd in [p["sec_date"] for p in pairs][5:]:
    with tempfile.TemporaryDirectory() as td:
        for f in files:
            if f.name.split("_")[13][:8] <= cd:
                (Path(td) / f.name).symlink_to(f.resolve())
        ts = load_nisar_goff_stack(td, track=98, reference_mask=ring)
    am = compute_acceleration_map(ts.dates, ts.displacement, window_size_days=cfg["detect"]["acceleration"]["window_size_days"],
                                  step_days=cfg["detect"]["acceleration"]["step_days"], n_harmonics=cfg["detect"].get("n_harmonics", 2))
    fl = detect_anomalies(am, LAT, LON, cfg, dates=ts.dates, displacement=ts.displacement)
    tags = {}
    for f in fl:
        t = f.detection_details.get("tag", "acceleration")
        hit = bool(len(f.pixel_indices) and target[f.pixel_indices[:, 0], f.pixel_indices[:, 1]].any())
        key = f"{t}{'_hit' if hit else ''}"
        tags[key] = tags.get(key, 0) + 1
    e1.append(dict(cutoff=cd, tags=tags, valid_px=int(np.isfinite(ts.displacement[-1]).sum())))
    print(f"  {cd}: {tags}  (pixels with a full series: {e1[-1]['valid_px']})", flush=True)

# ---- Terrain, surface, LOS -----------------------------------------------------------
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.transform import from_origin
DEM = "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N28_00_E085_00_DEM/Copernicus_DSM_COG_10_N28_00_E085_00_DEM.tif"
with rasterio.Env(**GDAL_HTTP_ENV), rasterio.open(DEM) as ds:
    with WarpedVRT(ds, crs="EPSG:32645", transform=from_origin(x[0] - 40, y[0] + 40, 80, 80),
                   width=len(x), height=len(y), resampling=Resampling.average) as v:
        z = v.read(1).astype(float)
g_row, g_col = np.gradient(z, 80.0)
gx, gy = g_col, -g_row
slope = np.degrees(np.arctan(np.hypot(gx, gy)))
aspect = np.degrees(np.arctan2(-gx, -gy)) % 360
sc = np.load(ROOT / "data/sentinel2/nepal_2026/scar/2025-10-21.npz")
scl, st = sc["SCL"], sc["SCL_transform"]
ci = ((X - st[0]) / st[1]).astype(int); ri = ((Y - st[3]) / st[5]).astype(int)
okc = (ci >= 0) & (ri >= 0) & (ci < scl.shape[1]) & (ri < scl.shape[0])
snow = np.zeros(X.shape, bool); snow[okc] = scl[ri[okc], ci[okc]] == 11
L = json.loads((OUT / "nisar_los_t98.json").read_text()); i5 = L["heights"].index(5000.0)
lx, ly, lz = los_enu(L["losUnitVectorX"][i5], L["losUnitVectorY"][i5]); ax_, ay_ = L["alongTrackUnitVectorX"][i5], L["alongTrackUnitVectorY"][i5]

# Per-pixel slope-parallel inversion (2x2 solve per pixel) -> azimuth difference to aspect
a11 = -(lx + lz * gx); a12 = -(ly + lz * gy); det = a11 * ay_ - a12 * ax_
RG = np.array([p["range_offset"] for p in pairs]); AT = np.array([p["along_track_offset"] for p in pairs])
E = (RG * ay_ - a12 * AT) / det; N = (a11 * AT - ax_ * RG) / det
az = np.degrees(np.arctan2(E, N)) % 360
dazi = np.abs((az - aspect + 180) % 360 - 180)

# ---- E2 reconstruction (as specified in Phase 1e) ---------------------------------------
Rr = np.array([ndi.generic_filter(p["range_offset"] * 12.0 / p["days"], np.nanmedian, size=3, mode="nearest") for p in pairs])
S = np.array([1.4826 * np.median(np.abs(r[ring & np.isfinite(r)] - np.median(r[ring & np.isfinite(r)]))) for r in Rr])
nb = np.isfinite(Rr[:4]).sum(axis=0)
b = np.where(nb >= 3, np.nanmedian(Rr[:4], axis=0), np.nan)
sb = 1.2533 * np.median(S[:4]) / np.sqrt(np.maximum(nb, 1))
Z = np.abs(Rr - b) / np.sqrt(S[:, None, None] ** 2 + sb ** 2)
alarm_any = np.zeros(X.shape, bool); alarm_k = np.zeros(Z.shape, bool)
for k in range(5, len(pairs)):
    a = (np.nan_to_num(Z[max(4, k - 3):k + 1], nan=0) >= 3).sum(axis=0) >= 3
    lab, n = ndi.label(a, structure=np.ones((3, 3)))
    sizes = ndi.sum(a, lab, range(1, n + 1))
    keep = np.isin(lab, [i + 1 for i, s_ in enumerate(sizes) if s_ >= 4])
    alarm_any |= keep; alarm_k[k] = keep
lab, n = ndi.label(alarm_any, structure=np.ones((3, 3)))
rows = []
for i in range(1, n + 1):
    m = lab == i
    is_target = bool((m & target).any())
    sel = alarm_k & m[None]
    rows.append(dict(id=i, target=is_target, px=int(m.sum()), km2=round(m.sum() * 0.0064, 3),
                     dist_km=round(float(np.hypot(X[m] - cx, Y[m] - cy).mean() / 1000), 1),
                     elev_m=round(float(np.median(z[m]))), slope_deg=round(float(np.median(slope[m])), 1),
                     snow_ice_frac=round(float(snow[m].mean()), 2), median_snr=round(float(np.nanmedian(snr[:, m])), 1),
                     frac_masked=round(float(np.mean(~np.isfinite(RG[:, m]))), 2),
                     dir_dev_deg=round(float(np.nanmedian(dazi[sel & (slope >= 10)[None]])), 0) if (sel & (slope >= 10)[None]).any() else None,
                     dir_within_45=round(float(np.nanmean(dazi[sel & (slope >= 10)[None]] <= 45)), 2) if (sel & (slope >= 10)[None]).any() else None,
                     first_cutoff=pairs[int(np.argmax(alarm_k[:, m].any(axis=1)))]["sec_date"]))
rows.sort(key=lambda r: (not r["target"], -r["px"]))
print(f"\nE2 clusters (union over cutoffs): {n}")
print(" id  target   km2  dist  elev  slope snow/ice  SNR  masked  dir dev  within45  first")
for r in rows:
    print(f"{r['id']:3d}  {str(r['target']):6s} {r['km2']:5.2f} {r['dist_km']:5.1f} {r['elev_m']:5d} {r['slope_deg']:5.1f}   {r['snow_ice_frac']:4.2f}  "
          f"{r['median_snr']:5.1f}  {r['frac_masked']:4.2f}   {str(r['dir_dev_deg']):>5s}   {str(r['dir_within_45']):>5s}   {r['first_cutoff']}")
fa = [r for r in rows if not r["target"]]
summ = {k: round(float(np.median([r[k] for r in fa if r[k] is not None])), 2) for k in ("slope_deg", "snow_ice_frac", "median_snr", "frac_masked", "dir_dev_deg", "dir_within_45", "elev_m")}
tg = [r for r in rows if r["target"]]
print("\nfalse alarms, median:", summ)
print("target:", {k: tg[0][k] for k in summ} if tg else None)
(OUT / "diag_false_alarms.json").write_text(json.dumps(dict(note="DEVELOPMENT diagnosis on Nepal; no pass/fail", E1=e1, E2_clusters=rows, false_alarm_medians=summ), indent=1, default=float))
