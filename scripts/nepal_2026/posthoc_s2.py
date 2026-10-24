"""POST-HOC Sentinel-2 re-test after the pre-registered A2/A3 failed.

Usage (from the repository root):
    python scripts/nepal_2026/posthoc_s2.py [S2_DIR]

Changes from the pre-registered acceptance_s2.py, each motivated by a
diagnosed failure, not by the result:

- Source = the failure scar mapped independently of motion: pixels whose
  NIR reflectance changed by more than 2000 DN (either sign) between
  21 Oct 2025 and 21 Oct 2026 (same orbit, sun 48.8 vs 48.9 deg), the
  connected component(s) overlapping the ICIMOD polygon, holes filled.
- Stable reference = windows 1.5-5 km from the source with correlation
  >= 0.6 in that pair (rejects featureless snow), scar excluded.
- Only same-season pairs (spring, Mar-May; partner within +/-15 days of
  day-of-year); every combination per year pair, summarised by the median.
- Distinctiveness = rank of the scar's median speed among random patches
  of the same shape (500) elsewhere in the crop with valid data, so
  ordinary glaciers moving at similar speed count against it.

Writes output/acceptance/posthoc_s2.json and figures.
"""
import sys, json, itertools, yaml, numpy as np
from datetime import date
from pathlib import Path
from matplotlib.path import Path as MP
from scipy import ndimage as ndi
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).parent))
from glews.optical import highpass, offset_field
from utm import fwd

S2 = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/sentinel2/nepal_2026"
OUT = ROOT / "output/acceptance"
WINDOW, STEP, HP, PIX = 32, 8, 8.0, 10.0
STABLE_PEAK, MIN_PEAK, DOY_TOL, CROP = 0.6, 0.2, 15, slice(250, 850)   # central 6 x 6 km
rng = np.random.default_rng(0)

site = yaml.safe_load(open(ROOT / "config/nepal_2026.yaml"))["site"]
poly = np.array([fwd(la, lo) for la, lo in site["known_source"]["polygon"]])
cx, cy = fwd(site["latitude"], site["longitude"])

# ---- Scar outline (independent of motion) ---------------------------------
pre = np.load(S2 / "scar/2025-10-21.npz"); post = np.load(S2 / "scar/2026-10-21.npz")
t = pre["B08_transform"]
X, Y = np.meshgrid(t[0] + PIX * (np.arange(1100) + .5), t[3] - PIX * (np.arange(1100) + .5))
inpoly = MP(poly).contains_points(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
dn = post["B08"].astype(float) - pre["B08"].astype(float)
chg = ndi.binary_opening(np.abs(dn) > 2000, iterations=1)
lab, _ = ndi.label(chg)
ids = [i for i in np.unique(lab[inpoly & chg]) if i > 0 and (lab == i).sum() > 200]
scar = ndi.binary_fill_holes(np.isin(lab, ids))
inter = (scar & inpoly).sum()
scar_stats = dict(area_km2=round(scar.sum() * 1e-4, 3), polygon_km2=round(inpoly.sum() * 1e-4, 3),
                  iou=round(inter / (scar | inpoly).sum(), 2), scar_inside_polygon=round(inter / scar.sum(), 2),
                  polygon_covered=round(inter / inpoly.sum(), 2))
print("scar vs ICIMOD polygon:", scar_stats)

# ---- Scenes ----------------------------------------------------------------
index = json.loads((S2 / "index.json").read_text())
by_date = {}
for k, v in sorted(index.items()):
    if (S2 / "b08" / f"{k}.npz").exists():
        by_date.setdefault(v["date"], k)
DATES = sorted(by_date)
spring = [d for d in DATES if d[5:7] in ("03", "04", "05")]
_cache = {}


def img(d):
    if d not in _cache:
        _cache[d] = highpass(np.load(S2 / "b08" / f"{by_date[d]}.npz")["b08"].astype(np.float32), HP)[CROP, CROP]
    return _cache[d]


def doy(d): return date.fromisoformat(d).timetuple().tm_yday
def ndays(a, b): return (date.fromisoformat(b) - date.fromisoformat(a)).days


f0 = offset_field(img(spring[0]), img(spring[0]), WINDOW, STEP, passes=1)
WR, WC = CROP.start + f0.rows, CROP.start + f0.cols
wscar = scar[np.ix_(WR, WC)]
WX, WY = X[np.ix_(WR, WC)], Y[np.ix_(WR, WC)]
dist = np.hypot(WX - cx, WY - cy)
ring = (dist > 1500) & (dist < 5000) & ~ndi.binary_dilation(wscar, iterations=3)
print(f"scar windows {wscar.sum()}, ring windows {ring.sum()}")


def speed_map(a, b):
    f = offset_field(img(a), img(b), WINDOW, STEP, passes=2)
    good = np.isfinite(f.dx) & (f.peak >= MIN_PEAK)
    st = ring & good & (f.peak >= STABLE_PEAK)
    if st.sum() < 30:
        return None
    e = (f.dx - np.median(f.dx[st])) * PIX
    n = -(f.dy - np.median(f.dy[st])) * PIX
    e[~good] = np.nan; n[~good] = np.nan
    yr = ndays(a, b) / 365.25
    sig = float(np.hypot(*(1.4826 * np.median(np.abs(v[st] - np.median(v[st]))) for v in (e, n)))) / yr
    return e / yr, n / yr, sig, int(st.sum())


def patch_rank(sp, mask):
    r, c = np.nonzero(mask); H, W = sp.shape
    sig = np.nanmedian(sp[mask]); meds = []
    for _ in range(20000):
        if len(meds) == 500: break
        dr = rng.integers(-r.min(), H - r.max()); dc = rng.integers(-c.min(), W - c.max())
        rr, cc = r + dr, c + dc
        if mask[rr, cc].any(): continue
        v = sp[rr, cc]
        if np.isfinite(v).mean() >= 0.5: meds.append(np.nanmedian(v))
    return float(100 * np.mean(np.array(meds) < sig)), len(meds)


rows, maps = [], {}
for y in range(2018, 2026):
    combos = [(a, b) for a in spring if a.startswith(str(y))
              for b in spring if b.startswith(str(y + 1)) and abs(doy(a) - doy(b)) <= DOY_TOL]
    if not combos and y == 2021:      # 2022 is sparse in the archive: bridge 2021 -> 2023
        combos = [(a, b) for a in spring if a.startswith("2021")
                  for b in spring if b.startswith("2023") and abs(doy(a) - doy(b)) <= DOY_TOL]
    for a, b in combos:
        m = speed_map(a, b)
        if m is None:
            continue
        e, n, sig, nst = m
        sp = np.hypot(e, n)
        ok = wscar & np.isfinite(sp)
        if ok.sum() < 0.5 * wscar.sum():
            continue
        me, mn = np.median(e[ok]), np.median(n[ok])
        rank, npatch = patch_rank(sp, wscar)
        rows.append(dict(ref=a, sec=b, days=ndays(a, b), sigma_m_yr=round(sig, 2), stable_windows=nst,
                         scar_valid=int(ok.sum()), speed_m_yr=round(float(np.hypot(me, mn)), 2),
                         median_speed_m_yr=round(float(np.nanmedian(sp[ok])), 2),
                         azimuth_deg=round(float(np.degrees(np.arctan2(me, mn)) % 360), 0),
                         patch_rank_pct=round(rank, 1)))
        maps[(a, b)] = (e, n)
        r = rows[-1]
        print(f"{a} -> {b}  {r['speed_m_yr']:6.2f} m/yr (az {r['azimuth_deg']:3.0f})  median|v| {r['median_speed_m_yr']:6.2f}"
              f"  sigma {r['sigma_m_yr']:5.2f}  {r['speed_m_yr'] / r['sigma_m_yr']:5.1f} sigma  rank {r['patch_rank_pct']:5.1f}%"
              f"  scar {r['scar_valid']}/{wscar.sum()}  stable {nst}", flush=True)

# Per year-pair summary (median over combinations)
summary = {}
for r in rows:
    key = f"{r['ref'][:4]}->{r['sec'][:4]}"
    summary.setdefault(key, []).append(r)
print("\nyear pair    n   speed m/yr   sigma   n_sigma   rank%")
table = []
for k, rs in sorted(summary.items()):
    v = float(np.median([r["speed_m_yr"] for r in rs])); s = float(np.median([r["sigma_m_yr"] for r in rs]))
    rk = float(np.median([r["patch_rank_pct"] for r in rs]))
    table.append(dict(years=k, n=len(rs), speed_m_yr=round(v, 2), sigma_m_yr=round(s, 2), n_sigma=round(v / s, 1),
                      patch_rank_pct=round(rk, 1)))
    print(f"{k}  {len(rs):3d}   {v:8.2f}   {s:6.2f}   {v / s:6.1f}   {rk:6.1f}")

# Rate change: 2025->2026 vs earlier year pairs (A3 analogue, same season)
late = [t_ for t_ in table if t_["years"] == "2025->2026"]
early = [t_ for t_ in table if t_["years"] in ("2023->2024", "2024->2025")]
rate = None
if late and early:
    v1 = late[0]["speed_m_yr"]; v0 = float(np.median([t_["speed_m_yr"] for t_ in early]))
    s1 = late[0]["sigma_m_yr"]; s0 = float(np.sqrt(np.mean([t_["sigma_m_yr"] ** 2 for t_ in early])))
    rate = dict(v_2025_2026=v1, v_2023_2025=round(v0, 2), ratio=round(v1 / v0, 2) if v0 else None,
                diff_sigma=round((v1 - v0) / np.hypot(s1, s0), 2))
    print("\nrate change:", rate)
cum = sum(t_["speed_m_yr"] for t_ in table)   # one year per pair (2021->2023 counted as its rate x 2 below)
cum += sum(t_["speed_m_yr"] for t_ in table if t_["years"] == "2021->2023")
print(f"cumulative 2018 -> 2026 (sum of annual rates): {cum:.1f} m")

res = dict(note="POST HOC - see docstring; pre-registered results are in A1_goff.json and A2_A3_s2.json",
           settings=dict(window_px=WINDOW, step_px=STEP, stable_peak=STABLE_PEAK, min_peak=MIN_PEAK,
                         doy_tolerance=DOY_TOL, scar_threshold_dn=2000),
           scar=scar_stats, pairs=rows, by_year=table, rate_change=rate, cumulative_m=round(cum, 1))
(OUT / "posthoc_s2.json").write_text(json.dumps(res, indent=1, default=float))
np.save(OUT / "scar_mask_10m.npy", scar)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
keys = [k for k in maps if k[0].startswith("2024")][:1] + [k for k in maps if k[0].startswith("2025")][:1]
fig, axs = plt.subplots(1, len(keys), figsize=(6 * len(keys), 5.4), squeeze=False)
ext = [WX[0, 0], WX[0, -1], WY[-1, 0], WY[0, 0]]
for ax, k in zip(axs[0], keys):
    e, n = maps[k]
    im = ax.imshow(np.hypot(e, n), extent=ext, cmap="magma", vmin=0, vmax=25)
    ax.contour(WX, WY, wscar, [0.5], colors="lime", linewidths=1)
    ax.plot(*np.vstack([poly, poly[:1]]).T, c="yellow", lw=0.8)
    s = 3
    ax.quiver(WX[::s, ::s], WY[::s, ::s], e[::s, ::s], n[::s, ::s], color="white", width=0.002, scale=400)
    ax.set_title(f"{k[0]} → {k[1]} speed (m/yr); scar green, ICIMOD yellow", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout(); fig.savefig(OUT / "posthoc_s2_speed.png", dpi=110)
print("wrote", OUT / "posthoc_s2.json")
