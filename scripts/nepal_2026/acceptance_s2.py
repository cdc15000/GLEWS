"""Phase 1 acceptance tests A2 and A3 (docs/acceptance_phase1.md): Sentinel-2 offsets.

Usage (from the repository root):
    python scripts/nepal_2026/acceptance_s2.py [S2_DIR]

Reads the crops written by fetch_s2_crops.py (relative orbit 119 only).
Matching settings, fixed before any result was seen: band 8, high-pass
sigma 8 px, 32 px (320 m) windows every 4 px (40 m), two passes,
correlation >= 0.2, offsets referenced to the stable ring (2-5 km from the
source centre, source excluded). Signal = magnitude of the median offset
vector over source windows; sigma = sqrt(sx^2 + sy^2) from the ring.

Pairs are matched by season (day of year within +/-30 days) so sun
geometry is similar; every pair lists both sun elevations and the source
SCL fractions. Writes output/acceptance/A2_A3_s2.json and a figure.
"""
import sys, json, yaml, numpy as np
from datetime import date
from pathlib import Path
from matplotlib.path import Path as MP
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).parent))
from glews.optical import highpass, offset_field, reference_offsets
from utm import fwd

S2 = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/sentinel2/nepal_2026"
WINDOW, STEP, MIN_PEAK, HP_SIGMA, PIX = 32, 4, 0.2, 8.0, 10.0
K_SIGMA = 5.0
DOY_TOL = 30
OUT = ROOT / "output/acceptance"; OUT.mkdir(parents=True, exist_ok=True)

site = yaml.safe_load(open(ROOT / "config/nepal_2026.yaml"))["site"]
poly = np.array([fwd(la, lo) for la, lo in site["known_source"]["polygon"]])
cx, cy = fwd(site["latitude"], site["longitude"])
index = json.loads((S2 / "index.json").read_text())
scenes = sorted((v["date"], k) for k, v in index.items()
                if (S2 / "b08" / f"{k}.npz").exists())
# One scene per date (the first listed if a date has two)
by_date = {}
for d, k in scenes:
    by_date.setdefault(d, k)
DATES = sorted(by_date)
_cache = {}


def load(d):
    if d not in _cache:
        z = np.load(S2 / "b08" / f"{by_date[d]}.npz")
        _cache[d] = (highpass(z["b08"].astype(np.float32), HP_SIGMA), tuple(z["transform"]))
    return _cache[d]


def grids(field, transform):
    x0, dx, _, y0, _, dy = transform
    X, Y = np.meshgrid(x0 + dx * (field.cols + 0.5), y0 + dy * (field.rows + 0.5))
    src = MP(poly).contains_points(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
    dist = np.hypot(X - cx, Y - cy)
    return src, (dist > 2000) & (dist < 5000) & ~src


def doy(d):
    return date.fromisoformat(d).timetuple().tm_yday


def days(a, b):
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def pair(a, b, keep_maps=False):
    (ia, ta), (ib, tb) = load(a), load(b)
    assert ta == tb, "crops must share a grid"
    f = offset_field(ia, ib, window=WINDOW, step=STEP, passes=2)
    src, ring = grids(f, ta)
    ex, ny_, sx, sy = reference_offsets(f, ring, min_peak=MIN_PEAK)
    east, north = ex * PIX, -ny_ * PIX          # rows increase southward
    ok = np.isfinite(east) & src
    va, vb = index[by_date[a]], index[by_date[b]]
    r = dict(ref=a, sec=b, days=days(a, b), sun_elev=(va["sun_elevation"], vb["sun_elevation"]),
             scl_source=(va["source"], vb["source"]), source_windows=int(src.sum()),
             source_windows_valid=int(ok.sum()), sigma_m=round(float(np.hypot(sx, sy) * PIX), 3))
    if ok.sum() >= 5:
        me, mn = float(np.median(east[ok])), float(np.median(north[ok]))
        r.update(east_m=round(me, 2), north_m=round(mn, 2), signal_m=round(float(np.hypot(me, mn)), 2),
                 azimuth_deg=round(float(np.degrees(np.arctan2(me, mn)) % 360), 1))
        r["n_sigma"] = round(r["signal_m"] / r["sigma_m"], 2) if r["sigma_m"] > 0 else None
        r["rate_m_yr"] = round(r["signal_m"] / r["days"] * 365.25, 2)
    if keep_maps:
        r["_maps"] = (east, north, src, ring, f, ta)
    return r


def seasonal_partner(a, year):
    """Scene in ``year`` closest in day of year to ``a`` (within DOY_TOL)."""
    c = [d for d in DATES if d.startswith(str(year)) and abs(doy(d) - doy(a)) <= DOY_TOL]
    return min(c, key=lambda d: abs(doy(d) - doy(a))) if c else None


def fmt(r):
    if "signal_m" not in r:
        return f"{r['ref']} -> {r['sec']} ({r['days']:4d} d)  too few valid source windows ({r['source_windows_valid']})"
    return (f"{r['ref']} -> {r['sec']} ({r['days']:4d} d)  E {r['east_m']:7.2f}  N {r['north_m']:7.2f}  "
            f"|d| {r['signal_m']:6.2f} m  sigma {r['sigma_m']:5.2f}  {r['n_sigma']:6.1f} sigma  "
            f"{r['rate_m_yr']:6.2f} m/yr  az {r['azimuth_deg']:5.1f}  sun {r['sun_elev'][0]:.0f}/{r['sun_elev'][1]:.0f}  "
            f"valid {r['source_windows_valid']}/{r['source_windows']}")


result = {"settings": dict(window_px=WINDOW, step_px=STEP, min_peak=MIN_PEAK, highpass_sigma_px=HP_SIGMA,
                           orbit=119, doy_tolerance=DOY_TOL, n_scenes=len(DATES))}
print(f"{len(DATES)} usable scenes, {DATES[0]} .. {DATES[-1]}")

# ---- A2.1: year-apart autumn pairs (as pre-registered) --------------------
autumn = [d for d in DATES if d[5:7] in ("09", "10", "11")]
a2_autumn = []
for d in autumn:
    p = seasonal_partner(d, int(d[:4]) + 1)
    if p and not any(r["ref"][:4] == d[:4] for r in a2_autumn):
        a2_autumn.append(pair(d, p))
print("\nA2.1 year-apart autumn pairs (pre-registered):")
for r in a2_autumn: print("  " + fmt(r))
n_det = sum(1 for r in a2_autumn if r.get("n_sigma") and r["n_sigma"] >= K_SIGMA)
a21 = dict(pairs=a2_autumn, n_pairs=len(a2_autumn), n_detected=n_det,
           passed=bool(a2_autumn) and n_det >= len(a2_autumn) / 2)

# ---- Spring year-apart chain (deviation: autumn scenes are too sparse) ----
spring = [d for d in DATES if d[5:7] in ("03", "04", "05")]
chain, cur = [], None
start = min((d for d in spring if d < "2019"), default=spring[0])
cur = start
while True:
    nxt = None
    for gap in (1, 2):
        nxt = seasonal_partner(cur, int(cur[:4]) + gap)
        if nxt:
            break
    if not nxt or nxt > "2026-08-25":
        break
    chain.append(pair(cur, nxt, keep_maps=True)); cur = nxt
print("\nSpring chain (deviation from pre-registration, used for A2.1-bis, A2.2, A2.3):")
for r in chain: print("  " + fmt(r))
n_det_s = sum(1 for r in chain if r.get("n_sigma") and r["n_sigma"] >= K_SIGMA)

# Cumulative (Eulerian sum of per-window offsets along the chain)
E = sum(np.nan_to_num(r["_maps"][0]) for r in chain)
N = sum(np.nan_to_num(r["_maps"][1]) for r in chain)
valid_all = np.logical_and.reduce([np.isfinite(r["_maps"][0]) for r in chain])
src = chain[0]["_maps"][2]
sig_cum = float(np.sqrt(sum(r["sigma_m"] ** 2 for r in chain)))
mag = np.hypot(E, N)
cum_signal = float(np.hypot(np.median(E[src & valid_all]), np.median(N[src & valid_all]))) if (src & valid_all).sum() >= 5 else float("nan")
frac_moving = float(np.mean(mag[src & valid_all] >= K_SIGMA * sig_cum)) if (src & valid_all).any() else 0.0
max_cum = float(np.nanpercentile(mag[src & valid_all], 90)) if (src & valid_all).any() else float("nan")
print(f"\nCumulative {chain[0]['ref']} -> {chain[-1]['sec']}: median-vector {cum_signal:.1f} m, "
      f"90th percentile {max_cum:.1f} m, sigma {sig_cum:.2f} m, {frac_moving:.0%} of source windows >= {K_SIGMA:g} sigma "
      f"({int((src & valid_all).sum())} windows valid in every pair)")
a2 = {
    "A2.1_autumn": a21,
    "A2.1_spring_deviation": dict(n_pairs=len(chain), n_detected=n_det_s, passed=bool(chain) and n_det_s >= len(chain) / 2),
    "A2.2": dict(start=chain[0]["ref"] if chain else None, end=chain[-1]["sec"] if chain else None,
                 cumulative_median_vector_m=round(cum_signal, 2), cumulative_p90_m=round(max_cum, 2),
                 sigma_m=round(sig_cum, 2), passed=bool(cum_signal >= 50)),
    "A2.3": dict(fraction_source_windows_moving=round(frac_moving, 3), passed=frac_moving >= 0.5),
    "chain": [{k: v for k, v in r.items() if k != "_maps"} for r in chain],
}
a2["passed"] = a21["passed"] and a2["A2.2"]["passed"] and a2["A2.3"]["passed"]
result["A2"] = a2

# ---- A3: rate change from January 2026 -------------------------------------
in26 = [d for d in DATES if "2026-01-01" <= d <= "2026-08-25"]
first26, last26 = in26[0], in26[-1]
p26 = pair(first26, last26)
base_pairs = []
for d in [x for x in DATES if "2024-01-01" <= x <= "2024-12-31"]:
    p = seasonal_partner(d, 2025)
    if p and p <= "2025-12-31" and not any(abs(doy(b["ref"]) - doy(d)) < 45 for b in base_pairs):
        base_pairs.append(pair(d, p))
# Control (not pre-registered): the same within-year season window in 2024 and 2025
controls = []
for y in (2024, 2025):
    a = seasonal_partner(first26, y); b = seasonal_partner(last26, y)
    if a and b and a < b:
        controls.append(pair(a, b))
print("\nA3 2026 pair:\n  " + fmt(p26))
print("A3 baseline year-apart pairs within 2024-2025:")
for r in base_pairs: print("  " + fmt(r))
print("Control (not pre-registered): same season window in earlier years:")
for r in controls: print("  " + fmt(r))
good_base = [r for r in base_pairs if "rate_m_yr" in r]
if "rate_m_yr" in p26 and good_base:
    v26, s26 = p26["rate_m_yr"], p26["sigma_m"] / p26["days"] * 365.25
    vb = float(np.median([r["rate_m_yr"] for r in good_base]))
    sb = float(np.sqrt(np.mean([(r["sigma_m"] / r["days"] * 365.25) ** 2 for r in good_base])))
    comb = float(np.hypot(s26, sb))
    a3 = dict(v2026_m_yr=v26, sigma2026_m_yr=round(s26, 2), vbase_m_yr=round(vb, 2), sigmabase_m_yr=round(sb, 2),
              ratio=round(v26 / vb, 2) if vb > 0 else None, diff_sigma=round((v26 - vb) / comb, 2),
              passed=bool(vb > 0 and v26 >= 2 * vb and (v26 - vb) >= 3 * comb))
else:
    a3 = dict(passed=False, note="no measurable 2026 or baseline pair")
a3.update(pair_2026={k: v for k, v in p26.items() if k != "_maps"}, baseline_pairs=base_pairs, controls=controls)
result["A3"] = a3
print("\nA3:", {k: v for k, v in a3.items() if k not in ("pair_2026", "baseline_pairs", "controls")})
print("A2 passed:", a2["passed"], {k: a2[k]["passed"] for k in ("A2.1_autumn", "A2.1_spring_deviation", "A2.2", "A2.3")})

(OUT / "A2_A3_s2.json").write_text(json.dumps(result, indent=1, default=float))

# Figure: cumulative displacement over the crop with the source outline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
f, t = chain[0]["_maps"][4], chain[0]["_maps"][5]
x0, dx, _, y0, _, dy = t
ext = [x0 + dx * f.cols[0], x0 + dx * f.cols[-1], y0 + dy * f.rows[-1], y0 + dy * f.rows[0]]
fig, ax = plt.subplots(figsize=(6.5, 5.5))
show = np.where(valid_all, mag, np.nan)
im = ax.imshow(show, extent=ext, cmap="magma", vmin=0, vmax=np.nanpercentile(show, 99.5))
ax.plot(*np.vstack([poly, poly[:1]]).T, color="cyan", lw=1)
s = 6
X, Y = np.meshgrid(x0 + dx * f.cols, y0 + dy * f.rows)
ax.quiver(X[::s, ::s], Y[::s, ::s], np.where(valid_all, E, np.nan)[::s, ::s], np.where(valid_all, N, np.nan)[::s, ::s],
          color="white", scale=None, width=0.002)
plt.colorbar(im, ax=ax, label="cumulative horizontal displacement (m)")
ax.set_title(f"Sentinel-2 R119 offsets, {chain[0]['ref']} to {chain[-1]['sec']}", fontsize=9)
ax.set_xlabel("UTM 45N easting (m)"); ax.set_ylabel("northing (m)")
fig.tight_layout(); fig.savefig(OUT / "s2_cumulative.png", dpi=130)
print("wrote", OUT / "A2_A3_s2.json", OUT / "s2_cumulative.png")
