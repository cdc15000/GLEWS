"""DEVELOPMENT sweep of magnitude and ratio gates on E2 (no pass/fail).

Usage (from the repository root):
    python -u scripts/nepal_2026/dev_gates.py

Development set: Nepal plus the two Phase 2 controls (Ganesh East, Jugal
Himal), all now seen. Adds two gates to the Phase 1e/2 rate-change
detector E2 and sweeps their thresholds:

- magnitude: |r_k| >= M (m per 12 days, slant range, smoothed rate);
- ratio (direction-aware): s * r_k >= K * max(|b|, b_floor), where b is the
  Nov-Jan baseline rate, s = sign(b) when |b| >= b_floor, else sign(r_k),
  and b_floor = median ring noise over the baseline pairs.

An alarm pixel needs z >= 3 and both gates in >= 3 of the last 4 pairs.
Two blunder vetoes are also swept (added after the first sweep showed the
residual false alarms are mostly implausible rates, > 300 m/yr, with
frequent sign flips): CAP, a rate above which the window is rejected, and
sign consistency across the window.
Reports, per (M, K): false-alarm flags per site, the Nepal persistent hit
and lead, and the size of the hitting group. Also prints the per-pixel
smoothed rates inside the target, to show the margin left by M.
Writes output/acceptance/dev_gates.json. Fresh controls are the only test.
"""
import sys, json, logging, yaml, numpy as np
from datetime import date
from pathlib import Path
from matplotlib.path import Path as MP
from scipy import ndimage as ndi
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).parent))
from glews.nisar import read_goff_pairs
from utm import fwd

logging.basicConfig(level=logging.ERROR)
OUT = ROOT / "output/acceptance"
EVENT, LATEST_OK, LIMIT = date(2026, 8, 26), "20260726", 2
cfg = yaml.safe_load(open(ROOT / "config/nepal_2026.yaml")); site = cfg["site"]
icimod_ll = site["known_source"]["polygon"]; outlet_ll = site["reference_areas"][0]["polygon"]
SITES = [
    dict(name="nepal", goff=ROOT / "data/nisar/nepal_2026/goff_t98_crop", centre=(site["latitude"], site["longitude"]), target=True),
    dict(name="ganesh_east", goff=ROOT / "data/nisar/control_ganesh_east/goff_t98_crop", centre=(28.36, 85.22), target=False),
    dict(name="jugal", goff=ROOT / "data/nisar/control_jugal/goff_t98_crop", centre=(28.13, 85.86), target=False),
]
CAP_GRID = [np.inf, 10.0, 5.0, 3.0]
M_GRID = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
K_GRID = [0.0, 1.5, 2.0, 2.5, 3.0, 4.0]


def persistent_first(hits):
    for i, h in enumerate(hits):
        if h and sum(1 for x in hits[i:] if not x) <= 1:
            return i
    return None


def prepare(s):
    """Smoothed rates, z-scores and baseline for one site (identical to Phase 2's E2)."""
    pairs = read_goff_pairs(s["goff"], layer="layer2", min_snr=5, track=98, reference_mask=None)
    x, y = pairs[0]["x_coords"], pairs[0]["y_coords"]
    X, Y = np.meshgrid(x, y)
    cx, cy = fwd(*s["centre"])
    inside = lambda ll: MP(np.array([fwd(a, b) for a, b in ll])).contains_points(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
    src = inside(icimod_ll) if s["target"] else np.zeros(X.shape, bool)
    target = (src | inside(outlet_ll)) if s["target"] else src
    outlet = inside(outlet_ll) if s["target"] else src
    d = np.hypot(X - cx, Y - cy)
    ring = (d > 2000) & (d < 5000) & ~src
    pairs = read_goff_pairs(s["goff"], layer="layer2", min_snr=5, track=98, reference_mask=ring)
    Rr = np.array([ndi.generic_filter(p["range_offset"] * 12.0 / p["days"], np.nanmedian, size=3, mode="nearest") for p in pairs])
    S = np.array([1.4826 * np.median(np.abs(r[ring & np.isfinite(r)] - np.median(r[ring & np.isfinite(r)]))) for r in Rr])
    nb = np.isfinite(Rr[:4]).sum(axis=0)
    b = np.where(nb >= 3, np.nanmedian(Rr[:4], axis=0), np.nan)
    sb = 1.2533 * np.median(S[:4]) / np.sqrt(np.maximum(nb, 1))
    Z = np.abs(Rr - b) / np.sqrt(S[:, None, None] ** 2 + sb ** 2)
    return dict(Rr=Rr, S=S, b=b, Z=Z, target=target, outlet=outlet, dates=[p["sec_date"] for p in pairs],
                area_km2=X.size * (x[1] - x[0]) ** 2 / 1e6, b_floor=float(np.median(S[:4])))


def run(P, M, K, CAP=np.inf, consistent=False):
    """CAP: rates above it (m / 12 d) are treated as matching blunders and veto the window.
    consistent: every valid rate in the window must have the alarm sign."""
    Rr, b, Z, target = P["Rr"], P["b"], P["Z"], P["target"]
    s = np.where(np.abs(b) >= P["b_floor"], np.sign(b), np.sign(Rr))
    ok = (np.nan_to_num(Z, nan=0) >= 3) & (np.nan_to_num(np.abs(Rr), nan=0) >= M)
    if K > 0:
        ok &= np.nan_to_num(s * Rr, nan=-np.inf) >= K * np.maximum(np.nan_to_num(np.abs(b), nan=np.inf), P["b_floor"])
    blunder = np.nan_to_num(np.abs(Rr), nan=0) > CAP
    sgn = np.sign(np.nan_to_num(Rr, nan=0))
    fa_union, hits, sizes_hit = np.zeros(target.shape, bool), [], []
    for k in range(5, len(Rr)):
        w = slice(max(4, k - 3), k + 1)
        alarm = (ok[w].sum(axis=0) >= 3) & ~blunder[w].any(axis=0)
        if consistent:
            ref = np.sign(np.nansum(np.where(ok[w], Rr[w], 0), axis=0))
            alarm &= ~((sgn[w] != 0) & (sgn[w] != ref)).any(axis=0)
        lab, n = ndi.label(alarm, structure=np.ones((3, 3)))
        sizes = ndi.sum(alarm, lab, range(1, n + 1))
        hit, hs = False, 0
        for i in [i + 1 for i, s_ in enumerate(sizes) if s_ >= 4]:
            m = lab == i
            if (m & target).any():
                hit, hs = True, max(hs, int(m.sum()))
            else:
                fa_union |= m
        hits.append(hit); sizes_hit.append(hs)
    n_fa = int(ndi.label(fa_union, structure=np.ones((3, 3)))[1])
    i0 = persistent_first(hits)
    first = P["dates"][5 + i0] if i0 is not None else None
    return dict(fa=n_fa, first=first, hit_sizes=sizes_hit,
                lead=(EVENT - date(int(first[:4]), int(first[4:6]), int(first[6:]))).days if first else None)


prep = {}
for s in SITES:
    print(f"preparing {s['name']} ...", flush=True)
    prep[s["name"]] = prepare(s)
N = prep["nepal"]
print("\nNepal per-pixel smoothed rate (m / 12 d, slant range) inside the outlet / whole target; baseline noise floor "
      f"{N['b_floor']:.3f}")
for k, dt in enumerate(N["dates"]):
    o = N["Rr"][k][N["outlet"]]; t = N["Rr"][k][N["target"]]
    print(f"  {dt}: outlet median {np.nanmedian(o):5.2f} max {np.nanmax(o):5.2f} | target p90 {np.nanpercentile(t, 90):5.2f} "
          f"max {np.nanmax(t):5.2f} | ring sigma {N['S'][k]:.2f}", flush=True)
print(f"  outlet baseline b: median {np.nanmedian(N['b'][N['outlet']]):.2f}")

rows = []
print("\n  CAP cons    M     K | FA nepal ganesh_e jugal | Nepal first  lead | hit group px (per cutoff)")
for CAP, cons, K, M in [(c, q, k, m) for c in CAP_GRID for q in (False, True) for k in K_GRID for m in M_GRID]:
    if (CAP, cons) != (np.inf, False) and (K > 2.0 or M not in (0.0, 0.3, 0.5, 0.8, 1.0, 1.2)):
        continue
    if True:
        r = {n: run(P, M, K, CAP, cons) for n, P in prep.items()}
        ok = r["ganesh_east"]["fa"] <= LIMIT and r["jugal"]["fa"] <= LIMIT and r["nepal"]["first"] and r["nepal"]["first"] <= LATEST_OK
        rows.append(dict(CAP=None if np.isinf(CAP) else CAP, consistent=cons, M=M, K=K, fa={n: v["fa"] for n, v in r.items()}, nepal_first=r["nepal"]["first"],
                         nepal_lead=r["nepal"]["lead"], hit_sizes=r["nepal"]["hit_sizes"], meets_dev=bool(ok)))
        print(f"{CAP:5.0f} {str(cons)[0]:>4s} {M:5.2f} {K:5.1f} | {r['nepal']['fa']:8d} {r['ganesh_east']['fa']:8d} {r['jugal']['fa']:5d} | "
              f"{str(r['nepal']['first']):9s} {str(r['nepal']['lead']):>5s} | {r['nepal']['hit_sizes']}{'  <== meets' if ok else ''}",
              flush=True)
(OUT / "dev_gates.json").write_text(json.dumps(dict(note="DEVELOPMENT sweep on Nepal + Ganesh East + Jugal; no pass/fail",
                                                    b_floor={n: P["b_floor"] for n, P in prep.items()}, rows=rows), indent=1))
print("wrote", OUT / "dev_gates.json")
