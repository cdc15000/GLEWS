"""Phase 1c (docs/acceptance_phase1c.md): GLEWS over Wang et al.'s measurement area.

Usage (from the repository root):
    python scripts/nepal_2026/acceptance_phase1c.py [S2_DIR] [GOFF_DIR]

Implements C1-C4 exactly as pre-registered in docs/acceptance_phase1c.md
(committed before this script was run). Writes
output/acceptance/C_phase1c.json and C_phase1c.png.
"""
import sys, json, yaml, numpy as np
from datetime import date
from pathlib import Path
from matplotlib.path import Path as MP
from scipy import ndimage as ndi
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).parent))
from glews.optical import highpass, offset_points, GDAL_HTTP_ENV
from glews.nisar import read_goff_pairs
from glews.fusion import los_enu
from utm import fwd, inv

S2 = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/sentinel2/nepal_2026"
GOFF = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data/nisar/nepal_2026/goff_t98_crop"
OUT = ROOT / "output/acceptance"
PIX, HP = 10.0, 8.0
MIN_PEAK, STABLE_PEAK, MIN_WIN, DOY_TOL, MAX_BAD = 0.2, 0.6, 5, 15, 0.30
WINDOWS = {"primary": (16, 2), "secondary": (32, 4)}
WANG_PRE, WANG_2026 = 7.4, 55.0

site = yaml.safe_load(open(ROOT / "config/nepal_2026.yaml"))["site"]
tri_ll = np.array(site["reference_areas"][0]["polygon"])
tri = np.array([fwd(a, b) for a, b in tri_ll])
icimod = np.array([fwd(a, b) for a, b in site["known_source"]["polygon"]])
cx, cy = fwd(site["latitude"], site["longitude"])

# ---- Scenes (corrected selection: cloud + cloud shadow only) ----------------------
index = json.loads((S2 / "index.json").read_text())
by_date = {}
for k, v in sorted(index.items()):
    s = v["source"]
    if (S2 / "b08" / f"{k}.npz").exists() and s["cloud"] + s["cloud_shadow"] <= MAX_BAD and s["nodata"] == 0:
        by_date.setdefault(v["date"], k)
DATES = sorted(by_date)
z0 = np.load(S2 / "b08" / f"{by_date[DATES[0]]}.npz")
t = z0["transform"]; H, W = z0["b08"].shape
X, Y = np.meshgrid(t[0] + PIX * (np.arange(W) + .5), t[3] - PIX * (np.arange(H) + .5))
in_tri = MP(tri).contains_points(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
in_ici = MP(icimod).contains_points(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
excl = ndi.binary_dilation(in_ici, iterations=30)          # 300 m
dist = np.hypot(X - cx, Y - cy)
ring_px = (dist > 1500) & (dist < 5000) & ~excl
_cache = {}


def img(d):
    if d not in _cache:
        _cache[d] = highpass(np.load(S2 / "b08" / f"{by_date[d]}.npz")["b08"].astype(np.float32), HP)
    return _cache[d]


def doy(d): return date.fromisoformat(d).timetuple().tm_yday
def ndays(a, b): return (date.fromisoformat(b) - date.fromisoformat(a)).days


def points(window, step):
    rr, cc = np.mgrid[0:H:step, 0:W:step]
    tri_pts = in_tri[rr, cc]
    r8, c8 = np.mgrid[0:H:8, 0:W:8]
    ring_pts = ring_px[r8, c8]
    return (np.r_[rr[tri_pts], r8[ring_pts]], np.r_[cc[tri_pts], c8[ring_pts]], int(tri_pts.sum()))


PTS = {k: points(*v) for k, v in WINDOWS.items()}
for k, (r, c, n) in PTS.items():
    print(f"{k} {WINDOWS[k][0]} px windows: {n} area windows, {len(r) - n} stable-ring candidates")


def measure(a, b, which):
    window, _ = WINDOWS[which]
    r, c, n = PTS[which]
    dx, dy, pk = offset_points(img(a), img(b), r, c, window=window)
    ring = np.zeros(len(r), bool); ring[n:] = True
    good = np.isfinite(dx) & (pk >= MIN_PEAK)
    st = ring & good & (pk >= STABLE_PEAK)
    if st.sum() < 30:
        return None
    e = (dx - np.median(dx[st])) * PIX
    nn = -(dy - np.median(dy[st])) * PIX
    sx = 1.4826 * np.median(np.abs(e[st] - np.median(e[st])))
    sy = 1.4826 * np.median(np.abs(nn[st] - np.median(nn[st])))
    area = ~ring & good
    yr = ndays(a, b) / 365.25
    res = dict(ref=a, sec=b, days=ndays(a, b), area_windows=int(area.sum()), stable_windows=int(st.sum()),
               sigma_m_yr=round(float(np.hypot(sx, sy)) / yr, 2))
    if area.sum() >= MIN_WIN:
        me, mn = float(np.median(e[area])), float(np.median(nn[area]))
        res.update(east_m_yr=round(me / yr, 2), north_m_yr=round(mn / yr, 2),
                   speed_m_yr=round(float(np.hypot(me, mn)) / yr, 2),
                   azimuth_deg=round(float(np.degrees(np.arctan2(me, mn)) % 360), 0))
    return res


spring = [d for d in DATES if d[5:7] in ("03", "04", "05")]
result = {"criteria": "docs/acceptance_phase1c.md", "n_scenes": len(DATES),
          "scenes_2026": [d for d in DATES if d >= "2026-01-01"]}
for which in ("primary", "secondary"):
    print(f"\n=== {which}: {WINDOWS[which][0]} px windows ===")
    # ---- C1 ----
    annual = []
    for y in range(2018, 2025):
        combos = [(a, b) for a in spring if a.startswith(str(y))
                  for b in spring if b.startswith(str(y + 1)) and abs(doy(a) - doy(b)) <= DOY_TOL]
        rs = [m for m in (measure(a, b, which) for a, b in combos) if m and "speed_m_yr" in m]
        if rs:
            annual.append(dict(years=f"{y}->{y + 1}", n=len(rs),
                               speed_m_yr=round(float(np.median([m["speed_m_yr"] for m in rs])), 2),
                               sigma_m_yr=round(float(np.median([m["sigma_m_yr"] for m in rs])), 2),
                               azimuth_deg=round(float(np.median([m["azimuth_deg"] for m in rs])), 0)))
            q = annual[-1]
            print(f"  {q['years']}  n={q['n']:3d}  {q['speed_m_yr']:6.2f} m/yr  sigma {q['sigma_m_yr']:5.2f}  az {q['azimuth_deg']:.0f}")
    v1 = float(np.median([q["speed_m_yr"] for q in annual])) if annual else float("nan")
    s1 = float(np.median([q["sigma_m_yr"] for q in annual])) if annual else float("nan")
    c1 = dict(annual=annual, rate_m_yr=round(v1, 2), sigma_m_yr=round(s1, 2), wang_m_yr=WANG_PRE,
              passed=bool(annual) and 3.7 <= v1 <= 14.8 and v1 >= 2 * s1)
    print(f"  C1: {v1:.2f} m/yr (sigma {s1:.2f}); Wang 7.4 -> {'PASS' if c1['passed'] else 'FAIL'}")
    # ---- C2 ----
    refs = [d for d in DATES if "2026-05-01" <= d <= "2026-06-15"]
    secs = [d for d in DATES if "2026-08-01" <= d <= "2026-08-25"]
    c2_pairs = [m for m in (measure(a, b, which) for a in refs for b in secs) if m]
    for m in c2_pairs:
        print(f"  2026 {m['ref']} -> {m['sec']} ({m['days']} d): "
              + (f"{m['speed_m_yr']:6.2f} m/yr az {m['azimuth_deg']:.0f}, sigma {m['sigma_m_yr']:.2f}, "
                 f"{m['area_windows']} windows" if "speed_m_yr" in m else f"only {m['area_windows']} valid windows"))
    ok2 = [m for m in c2_pairs if "speed_m_yr" in m]
    if ok2:
        v2 = float(np.median([m["speed_m_yr"] for m in ok2])); s2 = float(np.median([m["sigma_m_yr"] for m in ok2]))
        comb = float(np.hypot(s1, s2))
        c2 = dict(pairs=c2_pairs, rate_m_yr=round(v2, 2), sigma_m_yr=round(s2, 2), wang_m_yr=WANG_2026,
                  ratio=round(v2 / v1, 2) if v1 > 0 else None, diff_sigma=round((v2 - v1) / comb, 2),
                  verdict="PASS" if (v2 >= 2 * v1 and (v2 - v1) >= 3 * comb) else "FAIL")
    else:
        c2 = dict(pairs=c2_pairs, verdict="not testable")
    print(f"  C2: {c2.get('rate_m_yr')} m/yr vs C1 {v1:.2f} (ratio {c2.get('ratio')}, {c2.get('diff_sigma')} sigma) -> {c2['verdict']}")
    result[which] = dict(C1=c1, C2=c2)

# ---- GOFF over the area (C3, C4) ------------------------------------------------
probe = read_goff_pairs(GOFF, layer="layer2", min_snr=0, track=98)[0]
GX, GY = np.meshgrid(probe["x_coords"], probe["y_coords"])
g_tri = MP(tri).contains_points(np.c_[GX.ravel(), GY.ravel()]).reshape(GX.shape)
g_src = MP(icimod).contains_points(np.c_[GX.ravel(), GY.ravel()]).reshape(GX.shape)
gd = np.hypot(GX - cx, GY - cy)
g_ring = (gd > 2000) & (gd < 5000) & ~g_src
gp = read_goff_pairs(GOFF, layer="layer2", min_snr=5, track=98, reference_mask=g_ring)
c4 = []
for p in gp:
    v = p["range_offset"][g_tri]
    c4.append(dict(ref=p["ref_date"], sec=p["sec_date"], days=p["days"], pixels=int(g_tri.sum()),
                   valid=int(np.isfinite(v).sum()),
                   range_m=round(float(np.nanmedian(v)), 3) if np.isfinite(v).sum() else None,
                   sigma_m=round(float(p["noise_sigma"][0]), 3)))
print(f"\nC4 GOFF over the area ({int(g_tri.sum())} px of 80 m):")
for q in c4:
    print(f"  {q['ref']}-{q['sec']}  {q['valid']}/{q['pixels']} valid  "
          + (f"{q['range_m']:+.3f} m ({q['range_m'] / q['days'] * 12:+.3f} per 12 d)" if q["range_m"] is not None else "n/a"))
result["C4"] = c4

# ---- C3 -----------------------------------------------------------------------------
L = json.loads((OUT / "nisar_los_t98.json").read_text())
i5 = L["heights"].index(5000.0)
los = los_enu(L["losUnitVectorX"][i5], L["losUnitVectorY"][i5])
import rasterio
from rasterio.windows import from_bounds
DEM = ("https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N28_00_E085_00_DEM/"
       "Copernicus_DSM_COG_10_N28_00_E085_00_DEM.tif")
with rasterio.Env(**GDAL_HTTP_ENV), rasterio.open(DEM) as ds:
    w = from_bounds(85.51, 28.28, 85.54, 28.30, transform=ds.transform).round_offsets().round_lengths()
    zdem = ds.read(1, window=w).astype(float); tw = ds.window_transform(w)
lat = tw.f + tw.e * (np.arange(zdem.shape[0]) + .5); lon = tw.c + tw.a * (np.arange(zdem.shape[1]) + .5)
LO, LA = np.meshgrid(lon, lat)
dm = MP(tri_ll[:, ::-1]).contains_points(np.c_[LO.ravel(), LA.ravel()]).reshape(LA.shape)
gr, gc = np.gradient(zdem)
gxe = float(np.median((gc / (tw.a * 111320 * np.cos(np.radians(28.29))))[dm]))
gyn = float(np.median((-gr / (abs(tw.e) * 111320))[dm]))
proj = np.array([los[0] + los[2] * gxe, los[1] + los[2] * gyn])     # range = -(proj . [e, n])
print(f"\nArea terrain: dz/dE {gxe:.3f}, dz/dN {gyn:.3f} (slope {np.degrees(np.arctan(np.hypot(gxe, gyn))):.1f} deg, "
      f"downslope az {np.degrees(np.arctan2(-gxe, -gyn)) % 360:.0f})")
gdates = sorted({p["ref_date"] for p in gp} | {p["sec_date"] for p in gp})
gdo = {d: date(int(d[:4]), int(d[4:6]), int(d[6:])) for d in gdates}
c3 = []
for m in result["primary"]["C2"]["pairs"]:
    if "speed_m_yr" not in m:
        continue
    a, b = date.fromisoformat(m["ref"]), date.fromisoformat(m["sec"])
    ga = min(gdates, key=lambda d: abs((gdo[d] - a).days)); gb = min(gdates, key=lambda d: abs((gdo[d] - b).days))
    chain = [q for q in c4 if q["ref"] >= ga and q["sec"] <= gb]
    if not chain or any(q["range_m"] is None for q in chain):
        c3.append(dict(s2=(m["ref"], m["sec"]), goff=(ga, gb), verdict="not testable")); continue
    gdays = (gdo[gb] - gdo[ga]).days
    g_rate = sum(q["range_m"] for q in chain) / gdays * 365.25
    g_sig = float(np.sqrt(sum(q["sigma_m"] ** 2 for q in chain))) / gdays * 365.25
    s2_range = -(proj[0] * m["east_m_yr"] + proj[1] * m["north_m_yr"])
    s2_sig = m["sigma_m_yr"] / np.sqrt(2) * float(np.linalg.norm(proj))
    comb = float(np.hypot(g_sig, s2_sig))
    agree = bool(np.sign(g_rate) == np.sign(s2_range) and abs(g_rate - s2_range) <= 2 * comb)
    c3.append(dict(s2=(m["ref"], m["sec"]), goff=(ga, gb), goff_range_m_yr=round(g_rate, 2), goff_sigma_m_yr=round(g_sig, 2),
                   s2_range_m_yr=round(float(s2_range), 2), s2_sigma_m_yr=round(float(s2_sig), 2),
                   n_sigma=round(abs(g_rate - s2_range) / comb, 2), agree=agree))
    print(f"C3 S2 {m['ref']}->{m['sec']} vs GOFF {ga}->{gb}: GOFF {g_rate:+.2f}±{g_sig:.2f} m/yr, "
          f"S2 projected {s2_range:+.2f}±{s2_sig:.2f} m/yr -> {'agree' if agree else 'disagree'}")
ok3 = [q for q in c3 if "agree" in q]
if ok3:
    g_med = float(np.median([q["goff_range_m_yr"] for q in ok3])); s_med = float(np.median([q["s2_range_m_yr"] for q in ok3]))
    comb = float(np.hypot(np.median([q["goff_sigma_m_yr"] for q in ok3]), np.median([q["s2_sigma_m_yr"] for q in ok3])))
    v3 = "PASS" if (np.sign(g_med) == np.sign(s_med) and abs(g_med - s_med) <= 2 * comb) else "FAIL"
else:
    g_med = s_med = None; v3 = "not testable"
result["C3"] = dict(pairs=c3, goff_median_m_yr=g_med, s2_median_m_yr=s_med, verdict=v3,
                    terrain=dict(dz_dE=round(gxe, 3), dz_dN=round(gyn, 3)))
print(f"C3: {v3}")
(OUT / "C_phase1c.json").write_text(json.dumps(result, indent=1, default=float))

# ---- Figure ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 4.2))
for which, col in (("primary", "C0"), ("secondary", "C1")):
    an = result[which]["C1"]["annual"]
    xs = [date(int(q["years"][:4]), 9, 1) for q in an]
    ax.errorbar(xs, [q["speed_m_yr"] for q in an], yerr=[q["sigma_m_yr"] for q in an], fmt="o", color=col,
                label=f"S2 spring year pairs, {WINDOWS[which][0]} px")
    c2p = [m for m in result[which]["C2"].get("pairs", []) if "speed_m_yr" in m]
    if c2p:
        ax.errorbar([date(2026, 7, 1)], [np.median([m["speed_m_yr"] for m in c2p])],
                    yerr=[np.median([m["sigma_m_yr"] for m in c2p])], fmt="s", color=col, ms=8)
ax.hlines(WANG_PRE, date(2018, 3, 1), date(2025, 12, 31), colors="k", linestyles="--", lw=1, label="Wang et al. 7.4 m/yr")
ax.hlines(WANG_2026, date(2026, 1, 1), date(2026, 8, 26), colors="r", linestyles="--", lw=1, label="Wang et al. 55 m/yr (2026)")
ax.set_ylabel("horizontal speed (m/yr)"); ax.legend(fontsize=7, loc="upper left")
ax.set_title("GLEWS Sentinel-2 offsets over Wang et al.'s measurement area", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "C_phase1c.png", dpi=120)
print("wrote", OUT / "C_phase1c.json")
