"""Phase 1d (docs/acceptance_phase1d.md): empirical noise model (D1) and main-glacier control (D2).

Usage (from the repository root):
    python -u scripts/nepal_2026/acceptance_phase1d.py [S2_DIR] [GOFF_DIR]

Implements D1 and D2 as pre-registered (committed before this script was
run). D2 runs first (seconds); D1 prints a progress line per pair. Writes
output/acceptance/D_phase1d.json.
"""
import sys, json, time, yaml, numpy as np
from datetime import date
from pathlib import Path
from matplotlib.path import Path as MP
from scipy import ndimage as ndi
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).parent))
from glews.optical import highpass, offset_points
from glews.nisar import read_goff_pairs
from utm import fwd

S2 = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/sentinel2/nepal_2026"
GOFF = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data/nisar/nepal_2026/goff_t98_crop"
OUT = ROOT / "output/acceptance"
PIX, HP, WINDOW, STEP = 10.0, 8.0, 16, 2
MIN_PEAK, STABLE_PEAK, DOY_TOL, MAX_BAD = 0.2, 0.6, 15, 0.30
N_COPIES, MIN_COPIES, COPY_STABLE_FRAC = 100, 30, 0.5
site = yaml.safe_load(open(ROOT / "config/nepal_2026.yaml"))["site"]
areas = site["reference_areas"]
area_ll, main_ll = np.array(areas[0]["polygon"]), np.array(areas[1]["polygon"])
area_utm = np.array([fwd(a, b) for a, b in area_ll]); main_utm = np.array([fwd(a, b) for a, b in main_ll])
icimod = np.array([fwd(a, b) for a, b in site["known_source"]["polygon"]])
cx, cy = fwd(site["latitude"], site["longitude"])
result = {"criteria": "docs/acceptance_phase1d.md"}

# ============================== D2 (GOFF) ======================================
probe = read_goff_pairs(GOFF, layer="layer2", min_snr=0, track=98)[0]
GX, GY = np.meshgrid(probe["x_coords"], probe["y_coords"])
inside = lambda P, X, Y: MP(P).contains_points(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
g_area, g_main, g_src = inside(area_utm, GX, GY), inside(main_utm, GX, GY), inside(icimod, GX, GY)
gd = np.hypot(GX - cx, GY - cy)
g_ring = (gd > 2000) & (gd < 5000) & ~g_src
gp = read_goff_pairs(GOFF, layer="layer2", min_snr=5, track=98, reference_mask=g_ring)
series = {"area": [], "main": []}
for p in gp:
    for k, m in (("area", g_area), ("main", g_main)):
        v = p["range_offset"][m]
        series[k].append(dict(ref=p["ref_date"], sec=p["sec_date"], days=p["days"], valid=int(np.isfinite(v).sum()),
                              pixels=int(m.sum()), range_m=float(np.nanmedian(v)) if np.isfinite(v).any() else None))


def period_rate(rows, a, b):
    q = [x for x in rows if a <= x["ref"] and x["sec"] <= b and x["range_m"] is not None]
    n = sum(x["days"] for x in q)
    return (sum(x["range_m"] for x in q) / n * 365.25) if n else float("nan")


periods = [("20251128", "20260115"), ("20260115", "20260328"), ("20260328", "20260527"),
           ("20260527", "20260726"), ("20260726", "20260819")]
rates = {k: {f"{a}-{b}": round(period_rate(series[k], a, b), 1) for a, b in periods} for k in series}
early, late = ("20251128", "20260115"), ("20260527", "20260819")
R = {k: period_rate(series[k], *late) / period_rate(series[k], *early) for k in series}
# slope-parallel conversion for the main glacier (GLO-30 over the box; LOS at 5000 m)
L = json.loads((OUT / "nisar_los_t98.json").read_text())
import rasterio
from rasterio.windows import from_bounds
from glews.optical import GDAL_HTTP_ENV
from glews.fusion import los_enu
los = los_enu(L["losUnitVectorX"][L["heights"].index(5000.0)], L["losUnitVectorY"][L["heights"].index(5000.0)])
DEM = ("https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N28_00_E085_00_DEM/"
       "Copernicus_DSM_COG_10_N28_00_E085_00_DEM.tif")
with rasterio.Env(**GDAL_HTTP_ENV), rasterio.open(DEM) as ds:
    w = from_bounds(85.51, 28.28, 85.55, 28.30, transform=ds.transform).round_offsets().round_lengths()
    z = ds.read(1, window=w).astype(float); tw = ds.window_transform(w)
lat = tw.f + tw.e * (np.arange(z.shape[0]) + .5); lon = tw.c + tw.a * (np.arange(z.shape[1]) + .5)
LO, LA = np.meshgrid(lon, lat)
dm = MP(main_ll[:, ::-1]).contains_points(np.c_[LO.ravel(), LA.ravel()]).reshape(LA.shape)
gr, gc = np.gradient(z)
gx = float(np.median((gc / (tw.a * 111320 * np.cos(np.radians(28.29))))[dm]))
gy = float(np.median((-gr / (abs(tw.e) * 111320))[dm]))
az = np.arctan2(-gx, -gy)                                   # downslope azimuth
dvec = np.array([np.sin(az), np.cos(az)])
sens = -float((los[0] + los[2] * gx) * dvec[0] + (los[1] + los[2] * gy) * dvec[1])   # range per unit horizontal downslope
main_rate_all = period_rate(series["main"], "20251128", "20260819")
a_r = [x["range_m"] / x["days"] for x in series["area"]]; m_r = [x["range_m"] / x["days"] for x in series["main"]]
ok = [i for i in range(len(a_r)) if a_r[i] is not None and m_r[i] is not None]
corr = float(np.corrcoef([a_r[i] for i in ok], [m_r[i] for i in ok])[0, 1])
d2_pass = bool(R["main"] <= 1.5 and R["area"] >= 2 * R["main"])
result["D2"] = dict(series=series, period_rates_m_yr=rates, R_area=round(R["area"], 2), R_main=round(R["main"], 2),
                    main_terrain=dict(slope_deg=round(float(np.degrees(np.arctan(np.hypot(gx, gy)))), 1),
                                      downslope_az_deg=round(float(np.degrees(az) % 360)), range_per_horizontal=round(sens, 2)),
                    main_rate_range_m_yr=round(main_rate_all, 1),
                    main_rate_horizontal_m_yr=round(main_rate_all / sens, 1) if abs(sens) > 0.1 else None,
                    wang_main_m_yr=42.2, series_correlation=round(corr, 2), verdict="PASS" if d2_pass else "FAIL")
print("D2 GOFF slant-range rates (m/yr) by period:")
for k in ("area", "main"):
    print(f"  {k:5s} ({series[k][0]['pixels']} px): " + "  ".join(f"{p}: {v}" for p, v in rates[k].items()))
print(f"  R_area {R['area']:.2f}, R_main {R['main']:.2f}; main glacier overall {main_rate_all:.1f} m/yr range "
      f"(sens {sens:.2f}) ~ {result['D2']['main_rate_horizontal_m_yr']} m/yr horizontal vs Wang 42.2; "
      f"per-pair correlation {corr:.2f} -> D2 {result['D2']['verdict']}", flush=True)
(OUT / "D_phase1d.json").write_text(json.dumps(result, indent=1, default=float))

# ============================== D1 (Sentinel-2) =================================
index = json.loads((S2 / "index.json").read_text())
by_date = {}
for k, v in sorted(index.items()):
    s = v["source"]
    if (S2 / "b08" / f"{k}.npz").exists() and s["cloud"] + s["cloud_shadow"] <= MAX_BAD and s["nodata"] == 0:
        by_date.setdefault(v["date"], k)
DATES = sorted(by_date)
z0 = np.load(S2 / "b08" / f"{by_date[DATES[0]]}.npz"); t = z0["transform"]; H, W = z0["b08"].shape
X, Y = np.meshgrid(t[0] + PIX * (np.arange(W) + .5), t[3] - PIX * (np.arange(H) + .5))
in_area = inside(area_utm, X, Y)
stable = (np.hypot(X - cx, Y - cy) > 1500) & (np.hypot(X - cx, Y - cy) < 5000) & ~ndi.binary_dilation(inside(icimod, X, Y), iterations=30)
rr, cc = np.mgrid[0:H:STEP, 0:W:STEP]
a_r0, a_c0 = rr[in_area[rr, cc]], cc[in_area[rr, cc]]
r8, c8 = np.mgrid[0:H:8, 0:W:8]
ring_r, ring_c = r8[stable[r8, c8]], c8[stable[r8, c8]]
rng = np.random.default_rng(0)
copies = []
for _ in range(200000):
    if len(copies) == N_COPIES:
        break
    dr, dc = int(rng.integers(-H, H)) // STEP * STEP, int(rng.integers(-W, W)) // STEP * STEP
    r_, c_ = a_r0 + dr, a_c0 + dc
    if r_.min() < WINDOW or c_.min() < WINDOW or r_.max() >= H - WINDOW or c_.max() >= W - WINDOW:
        continue
    if stable[r_, c_].all():
        copies.append((r_, c_))
n_a, n_ring = len(a_r0), len(ring_r)
ALL_R = np.concatenate([a_r0, ring_r] + [r_ for r_, _ in copies])
ALL_C = np.concatenate([a_c0, ring_c] + [c_ for _, c_ in copies])
print(f"\nD1: {n_a} area windows, {n_ring} ring windows, {len(copies)} stable copies of the area", flush=True)
_cache = {}


def img(d):
    if d not in _cache:
        _cache[d] = highpass(np.load(S2 / "b08" / f"{by_date[d]}.npz")["b08"].astype(np.float32), HP)
    return _cache[d]


def doy(d): return date.fromisoformat(d).timetuple().tm_yday
def ndays(a, b): return (date.fromisoformat(b) - date.fromisoformat(a)).days


def measure(a, b):
    dx, dy, pk = offset_points(img(a), img(b), ALL_R, ALL_C, window=WINDOW)
    ring = slice(n_a, n_a + n_ring)
    st = np.isfinite(dx[ring]) & (pk[ring] >= STABLE_PEAK)
    if st.sum() < 30:
        return None
    e = (dx - np.median(dx[ring][st])) * PIX
    n = -(dy - np.median(dy[ring][st])) * PIX
    good = np.isfinite(e) & (pk >= MIN_PEAK)
    yr = ndays(a, b) / 365.25

    def stat(sl):
        g = good[sl]
        return (np.median(e[sl][g]), np.median(n[sl][g])) if g.sum() >= 5 else None
    s_area = stat(slice(0, n_a))
    if s_area is None:
        return None
    meds = []
    off = n_a + n_ring
    for i in range(len(copies)):
        sl = slice(off + i * n_a, off + (i + 1) * n_a)
        if np.mean(pk[sl] >= STABLE_PEAK) >= COPY_STABLE_FRAC:
            sv = stat(sl)
            if sv is not None:
                meds.append(sv)
    if len(meds) < MIN_COPIES:
        return dict(ref=a, sec=b, copies=len(meds))
    M = np.array(meds)
    se = 1.4826 * np.median(np.abs(M[:, 0] - np.median(M[:, 0])))
    sn = 1.4826 * np.median(np.abs(M[:, 1] - np.median(M[:, 1])))
    speed = float(np.hypot(*s_area))
    rank = float(100 * np.mean(np.hypot(M[:, 0], M[:, 1]) < speed))
    return dict(ref=a, sec=b, days=ndays(a, b), copies=len(meds), speed_m_yr=round(speed / yr, 2),
                azimuth_deg=round(float(np.degrees(np.arctan2(*s_area)) % 360)),
                sigma_patch_m_yr=round(float(np.hypot(se, sn)) / yr, 2), rank_pct=round(rank, 1))


spring = [d for d in DATES if d[5:7] in ("03", "04", "05")]
jobs = [("C1", y, a, b) for y in range(2018, 2025) for a in spring if a.startswith(str(y))
        for b in spring if b.startswith(str(y + 1)) and abs(doy(a) - doy(b)) <= DOY_TOL]
jobs += [("C2", 2026, a, b) for a in DATES if "2026-05-01" <= a <= "2026-06-15"
         for b in DATES if "2026-08-01" <= b <= "2026-08-25"]
rows, t0 = [], time.time()
for i, (kind, y, a, b) in enumerate(jobs, 1):
    m = measure(a, b)
    if m:
        m["kind"] = kind; m["year"] = y; rows.append(m)
    el = time.time() - t0
    bar = "█" * int(30 * i / len(jobs)) + "░" * (30 - int(30 * i / len(jobs)))
    msg = (f"{m['speed_m_yr']:6.2f} m/yr ±{m['sigma_patch_m_yr']:5.2f} rank {m['rank_pct']:5.1f}%"
           if m and "speed_m_yr" in m else "skipped")
    print(f"[{bar}] {i:3d}/{len(jobs)} {el / 60:5.1f} min, ~{el / i * (len(jobs) - i) / 60:4.1f} left  {kind} {a}->{b}  {msg}", flush=True)

annual = []
for y in range(2018, 2025):
    rs = [m for m in rows if m["kind"] == "C1" and m["year"] == y and "speed_m_yr" in m]
    if rs:
        annual.append(dict(years=f"{y}->{y + 1}", n=len(rs), speed_m_yr=round(float(np.median([m["speed_m_yr"] for m in rs])), 2),
                           sigma_m_yr=round(float(np.median([m["sigma_patch_m_yr"] for m in rs])), 2),
                           rank_pct=round(float(np.median([m["rank_pct"] for m in rs])), 1)))
v1 = float(np.median([q["speed_m_yr"] for q in annual])); s1 = float(np.median([q["sigma_m_yr"] for q in annual]))
c2 = [m for m in rows if m["kind"] == "C2" and "speed_m_yr" in m]
v2 = float(np.median([m["speed_m_yr"] for m in c2])) if c2 else float("nan")
s2 = float(np.median([m["sigma_patch_m_yr"] for m in c2])) if c2 else float("nan")
comb = float(np.hypot(s1, s2))
d1a = bool(3.7 <= v1 <= 14.8 and v1 >= 2 * s1)
d1b = bool(c2) and bool(v2 >= 2 * v1 and (v2 - v1) >= 3 * comb)
result["D1"] = dict(pairs=rows, annual=annual, v1_m_yr=round(v1, 2), sigma1_m_yr=round(s1, 2),
                    v2_m_yr=round(v2, 2), sigma2_m_yr=round(s2, 2), diff_sigma=round((v2 - v1) / comb, 2),
                    D1a="PASS" if d1a else "FAIL", D1b=("PASS" if d1b else "FAIL") if c2 else "not testable",
                    c2_rank_pct=[m["rank_pct"] for m in c2])
print("\nD1a annual (median over combinations):")
for q in annual:
    print(f"  {q['years']}  n={q['n']:3d}  {q['speed_m_yr']:6.2f} m/yr  sigma_patch {q['sigma_m_yr']:5.2f}  rank {q['rank_pct']:5.1f}%")
print(f"D1a: v1 {v1:.2f} m/yr, sigma {s1:.2f} -> {result['D1']['D1a']}")
print(f"D1b: v2 {v2:.2f} m/yr, sigma {s2:.2f}; (v2-v1)/combined sigma {(v2 - v1) / comb:.2f} -> {result['D1']['D1b']}")
print(f"     2026 ranks among stable copies: {[m['rank_pct'] for m in c2]}")
(OUT / "D_phase1d.json").write_text(json.dumps(result, indent=1, default=float))
print("wrote", OUT / "D_phase1d.json")
