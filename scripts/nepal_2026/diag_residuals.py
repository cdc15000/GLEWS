"""DEVELOPMENT diagnosis: what is still flagged once offset blunders are rejected? (no pass/fail)

Usage (from the repository root):
    python -u scripts/nepal_2026/diag_residuals.py

Runs the best Phase 2b gated E2 (sign-consistent window, M = 1.0 m / 12 d,
K = 2) on Nepal, Ganesh East and Jugal Himal, with blunders now rejected
in the loader (read_goff_pairs max_rate = 5/12 m per day, ~150 m/yr).
Each residual false-alarm cluster is described by terrain (GLO-30),
surface (Sentinel-2 SCL snow/ice, autumn 2025), rate history by season,
and direction (slope-parallel azimuth vs downslope aspect). Writes
output/acceptance/diag_residuals.json and diag_residuals.png (Sentinel-2
true colour with the clusters outlined, and each cluster's rate series).
"""
import sys, json, logging, yaml, numpy as np
from pathlib import Path
from matplotlib.path import Path as MP
from scipy import ndimage as ndi
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).parent))
from glews.nisar import read_goff_pairs
from glews.optical import GDAL_HTTP_ENV, search_sentinel2, read_window
from glews.fusion import los_enu
from utm import fwd

logging.basicConfig(level=logging.ERROR)
OUT = ROOT / "output/acceptance"
S2_CACHE = ROOT / "data/sentinel2/residuals"; S2_CACHE.mkdir(parents=True, exist_ok=True)
MAX_RATE, M, K = 5.0 / 12, 1.0, 2.0
cfg = yaml.safe_load(open(ROOT / "config/nepal_2026.yaml")); site = cfg["site"]
icimod_ll = site["known_source"]["polygon"]; outlet_ll = site["reference_areas"][0]["polygon"]
SITES = [
    dict(name="nepal", goff=ROOT / "data/nisar/nepal_2026/goff_t98_crop", centre=(site["latitude"], site["longitude"]), target=True),
    dict(name="ganesh_east", goff=ROOT / "data/nisar/control_ganesh_east/goff_t98_crop", centre=(28.36, 85.22), target=False),
    dict(name="jugal", goff=ROOT / "data/nisar/control_jugal/goff_t98_crop", centre=(28.13, 85.86), target=False),
]
DEM = ("https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N28_00_E085_00_DEM/"
       "Copernicus_DSM_COG_10_N28_00_E085_00_DEM.tif")
SEASONS = [("Nov-Jan", "20251128", "20260115"), ("Jan-Mar", "20260115", "20260328"),
           ("Apr-May", "20260328", "20260527"), ("Jun-Aug", "20260527", "20260820")]
L = json.loads((OUT / "nisar_los_t98.json").read_text()); i5 = L["heights"].index(5000.0)
lx, ly, lz = los_enu(L["losUnitVectorX"][i5], L["losUnitVectorY"][i5])
ax_, ay_ = L["alongTrackUnitVectorX"][i5], L["alongTrackUnitVectorY"][i5]


def dem_on_grid(x, y):
    import rasterio
    from rasterio.vrt import WarpedVRT
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    post = float(x[1] - x[0])
    with rasterio.Env(**GDAL_HTTP_ENV), rasterio.open(DEM) as ds:
        with WarpedVRT(ds, crs="EPSG:32645", transform=from_origin(x[0] - post / 2, y[0] + post / 2, post, post),
                       width=len(x), height=len(y), resampling=Resampling.average) as v:
            return v.read(1).astype(float)


def s2_rgb(name, lat, lon, bounds):
    """Clear autumn-2025 true colour + SCL over the crop (cached)."""
    f = S2_CACHE / f"{name}.npz"
    if f.exists():
        return dict(np.load(f))
    for s in sorted(search_sentinel2(lat, lon, "2025-10-10", "2025-12-15", max_cloud=15), key=lambda s: s.cloud_cover):
        scl, t = read_window(s.hrefs["scl"], bounds)
        if np.mean(scl == 0) > 0.02 or np.mean(np.isin(scl, (3, 8, 9, 10))) > 0.05:
            continue
        rgb = np.dstack([read_window(s.hrefs[b], bounds)[0] for b in ("red", "green", "blue")])
        out = dict(rgb=rgb, scl=scl, scl_transform=np.array(t), scene=s.id)
        np.savez_compressed(f, **out)
        return out
    raise RuntimeError(f"no clear Sentinel-2 scene for {name}")


def gated(Rr, b, Z, b_floor):
    s = np.where(np.abs(b) >= b_floor, np.sign(b), np.sign(Rr))
    ok = (np.nan_to_num(Z, nan=0) >= 3) & (np.nan_to_num(np.abs(Rr), nan=0) >= M)
    ok &= np.nan_to_num(s * Rr, nan=-np.inf) >= K * np.maximum(np.nan_to_num(np.abs(b), nan=np.inf), b_floor)
    sgn = np.sign(np.nan_to_num(Rr, nan=0))
    alarms = np.zeros(Rr.shape, bool)
    for k in range(5, len(Rr)):
        w = slice(max(4, k - 3), k + 1)
        a = ok[w].sum(axis=0) >= 3
        ref = np.sign(np.nansum(np.where(ok[w], Rr[w], 0), axis=0))
        a &= ~((sgn[w] != 0) & (sgn[w] != ref)).any(axis=0)
        lab, n = ndi.label(a, structure=np.ones((3, 3)))
        sz = ndi.sum(a, lab, range(1, n + 1))
        alarms[k] = np.isin(lab, [i + 1 for i, v in enumerate(sz) if v >= 4])
    return alarms


report, panels = {}, []
for st in SITES:
    probe = read_goff_pairs(st["goff"], layer="layer2", min_snr=0, track=98)[0]
    x, y = probe["x_coords"], probe["y_coords"]
    X, Y = np.meshgrid(x, y)
    cx, cy = fwd(*st["centre"])
    inside = lambda ll: MP(np.array([fwd(a, b) for a, b in ll])).contains_points(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
    src = inside(icimod_ll) if st["target"] else np.zeros(X.shape, bool)
    target = (src | inside(outlet_ll)) if st["target"] else src
    d = np.hypot(X - cx, Y - cy)
    ring = (d > 2000) & (d < 5000) & ~src
    pairs = read_goff_pairs(st["goff"], layer="layer2", min_snr=5, track=98, reference_mask=ring, max_rate=MAX_RATE)
    dates = [p["sec_date"] for p in pairs]
    RG = np.array([p["range_offset"] for p in pairs]); AT = np.array([p["along_track_offset"] for p in pairs])
    days = np.array([p["days"] for p in pairs])
    Rr = np.array([ndi.generic_filter(p["range_offset"] * 12.0 / p["days"], np.nanmedian, size=3, mode="nearest") for p in pairs])
    S = np.array([1.4826 * np.median(np.abs(r[ring & np.isfinite(r)] - np.median(r[ring & np.isfinite(r)]))) for r in Rr])
    nb = np.isfinite(Rr[:4]).sum(axis=0)
    b = np.where(nb >= 3, np.nanmedian(Rr[:4], axis=0), np.nan)
    sb = 1.2533 * np.median(S[:4]) / np.sqrt(np.maximum(nb, 1))
    Z = np.abs(Rr - b) / np.sqrt(S[:, None, None] ** 2 + sb ** 2)
    alarms = gated(Rr, b, Z, float(np.median(S[:4])))
    union = alarms.any(axis=0)

    z = dem_on_grid(x, y)
    g_row, g_col = np.gradient(z, float(x[1] - x[0])); gx, gy = g_col, -g_row
    slope = np.degrees(np.arctan(np.hypot(gx, gy))); aspect = np.degrees(np.arctan2(-gx, -gy)) % 360
    a11 = -(lx + lz * gx); a12 = -(ly + lz * gy); det = a11 * ay_ - a12 * ax_
    E = (RG * ay_ - a12 * AT) / det; N = (a11 * AT - ax_ * RG) / det
    dazi = np.abs((np.degrees(np.arctan2(E, N)) % 360 - aspect + 180) % 360 - 180)
    bounds = (x[0] - 40, y[-1] - 40, x[-1] + 40, y[0] + 40)
    s2 = s2_rgb(st["name"], *st["centre"], bounds)
    t = s2["scl_transform"]
    ci = ((X - t[0]) / t[1]).astype(int).clip(0, s2["scl"].shape[1] - 1)
    ri = ((Y - t[3]) / t[5]).astype(int).clip(0, s2["scl"].shape[0] - 1)
    snow = s2["scl"][ri, ci] == 11

    lab, n = ndi.label(union, structure=np.ones((3, 3)))
    rows = []
    for i in range(1, n + 1):
        m = lab == i
        sel = alarms & m[None]
        med = np.nanmedian(np.where(m[None], Rr, np.nan).reshape(len(Rr), -1), axis=1)
        rate_by = {nm: round(float(np.nanmean([med[k] for k, dt in enumerate(dates) if a < dt <= e2])) * 365 / 12, 1)
                   for nm, a, e2 in SEASONS}
        dsel = dazi[sel & (slope >= 10)[None]]
        rows.append(dict(id=i, target=bool((m & target).any()), px=int(m.sum()), km2=round(m.sum() * 0.0064, 3),
                         elev_m=int(np.median(z[m])), slope_deg=round(float(np.median(slope[m])), 1),
                         aspect_deg=int(np.median(aspect[m])), snow_ice_frac=round(float(snow[m].mean()), 2),
                         rate_m_per_yr=rate_by, n_alarm_cutoffs=int(alarms[:, m].any(axis=1).sum()),
                         first=dates[int(np.argmax(alarms[:, m].any(axis=1)))],
                         dir_within_45=round(float(np.mean(dsel <= 45)), 2) if dsel.size else None,
                         series=[None if not np.isfinite(v) else round(float(v), 3) for v in med]))
    rows.sort(key=lambda r: (not r["target"], -r["px"]))
    n_fa = sum(not r["target"] for r in rows)
    nbl = [p["n_blunders"] for p in pairs]
    print(f"\n== {st['name']}: {n_fa} false-alarm clusters; blunders rejected per pair median {int(np.median(nbl))} "
          f"of {X.size} px (max {max(nbl)}); S2 {s2['scene'] if 'scene' in s2 else ''}")
    print(" id tgt    km2  elev slope aspect snow/ice  cutoffs first     dir<45  rate m/yr slant (Nov-Jan | Jan-Mar | Apr-May | Jun-Aug)")
    for r in rows:
        rb = r["rate_m_per_yr"]
        print(f"{r['id']:3d} {'T' if r['target'] else '-'}  {r['km2']:6.3f} {r['elev_m']:5d} {r['slope_deg']:5.1f} {r['aspect_deg']:6d}"
              f"   {r['snow_ice_frac']:5.2f}  {r['n_alarm_cutoffs']:6d}  {r['first']}  {str(r['dir_within_45']):>5s}   "
              f"{rb['Nov-Jan']:6.1f} {rb['Jan-Mar']:6.1f} {rb['Apr-May']:6.1f} {rb['Jun-Aug']:6.1f}")
    report[st["name"]] = dict(n_false_alarm_clusters=n_fa, blunders_per_pair=nbl, s2_scene=str(s2.get("scene", "")), clusters=rows)
    panels.append((st["name"], s2, lab, rows, bounds, dates))

(OUT / "diag_residuals.json").write_text(json.dumps(dict(note="DEVELOPMENT diagnosis; no pass/fail",
                                                         settings=dict(max_rate_m_per_day=MAX_RATE, M=M, K=K, consistent=True),
                                                         sites=report), indent=1))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 3, figsize=(16, 10))
for j, (nm, s2, lab, rows, bounds, dates) in enumerate(panels):
    ax = axs[0, j]
    rgb = s2["rgb"].astype(float); rgb = np.clip(rgb / np.percentile(rgb, 99.5), 0, 1) ** 0.8
    ax.imshow(rgb, extent=[bounds[0], bounds[2], bounds[1], bounds[3]])
    ext = [bounds[0], bounds[2], bounds[1], bounds[3]]
    ax.contour(np.flipud(lab > 0), levels=[0.5], colors="red", linewidths=0.8, extent=ext, origin="lower")
    for r in rows:
        yy, xx = np.nonzero(lab == r["id"])
        px = bounds[0] + (xx.mean() + 0.5) * 80; py = bounds[3] - (yy.mean() + 0.5) * 80
        ax.text(px, py, str(r["id"]), color="cyan" if r["target"] else "yellow", fontsize=7, weight="bold")
    ax.set_title(f"{nm}: residual flags (red), autumn-2025 S2", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    ax = axs[1, j]
    t = np.arange(len(dates))
    for r in rows[:8]:
        ax.plot(t, np.array([np.nan if v is None else v for v in r["series"]]) * 365 / 12,
                lw=2.5 if r["target"] else 1, label=f"{'T ' if r['target'] else ''}#{r['id']}")
    ax.set_xticks(t[::3]); ax.set_xticklabels([d[2:] for d in dates[::3]], fontsize=6)
    ax.set_ylabel("cluster median rate, m/yr slant range", fontsize=8); ax.legend(fontsize=6); ax.axhline(0, c="k", lw=0.5)
fig.tight_layout(); fig.savefig(OUT / "diag_residuals.png", dpi=90)
print("wrote", OUT / "diag_residuals.json")
