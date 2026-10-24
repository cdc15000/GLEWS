"""DEVELOPMENT sweep for E4, a causal pre-melt acceleration detector (no pass/fail).

Usage (from the repository root):
    python -u scripts/nepal_2026/dev_premelt_detector.py

Development set: Nepal, Ganesh East, Jugal Himal (all seen). Loader blunder
rejection on (max_rate 5/12 m per day). Per pixel, rates r_k are the
Phase 1e/2 smoothed slant-range rates (m / 12 d, 3x3 nan-median) and b the
Nov-Jan baseline (first 4 pairs, >= 3 valid). Only pixels with a moving
baseline, |b| >= the ring noise floor, can alarm.

At cutoff k the pre-melt window is the pairs with secondary date in
(15 Jan, min(cutoff, 27 May)], needing >= MIN_PAIRS valid. With pm its
median and s = sign(b):

    ratio = s * pm / |b| >= R0     and     increase = (s * pm - |b|) * 365/12 >= I0 m/yr

A flag is a connected group (8-connectivity) of >= MIN_PX alarm pixels.
After 27 May the window stops growing, so alarms are frozen. Reports
false-alarm flags (union over cutoffs, as in Phase 2) per site and the
Nepal persistent hit and lead, for each (R0, I0, MIN_PX).
Writes output/acceptance/dev_premelt_detector.json.
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
MAX_RATE, MIN_PAIRS, PM_START, PM_END = 5.0 / 12, 3, "20260115", "20260527"
cfg = yaml.safe_load(open(ROOT / "config/nepal_2026.yaml")); site = cfg["site"]
icimod_ll = site["known_source"]["polygon"]; outlet_ll = site["reference_areas"][0]["polygon"]
SITES = [("nepal", "data/nisar/nepal_2026/goff_t98_crop", (site["latitude"], site["longitude"]), True),
         ("ganesh_east", "data/nisar/control_ganesh_east/goff_t98_crop", (28.36, 85.22), False),
         ("jugal", "data/nisar/control_jugal/goff_t98_crop", (28.13, 85.86), False)]
R_GRID = [1.3, 1.5, 1.7, 1.8, 2.0]
I_GRID = [6.0, 8.0, 10.0, 12.0, 14.0]
PX_GRID = [3, 4, 6]


def persistent_first(hits):
    for i, h in enumerate(hits):
        if h and sum(1 for x in hits[i:] if not x) <= 1:
            return i
    return None


def prepare(name, g, centre, has_target):
    pairs = read_goff_pairs(ROOT / g, layer="layer2", min_snr=0, track=98)
    X, Y = np.meshgrid(pairs[0]["x_coords"], pairs[0]["y_coords"])
    cx, cy = fwd(*centre)
    inside = lambda ll: MP(np.array([fwd(a, b) for a, b in ll])).contains_points(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
    src = inside(icimod_ll) if has_target else np.zeros(X.shape, bool)
    target = (src | inside(outlet_ll)) if has_target else src
    d = np.hypot(X - cx, Y - cy); ring = (d > 2000) & (d < 5000) & ~src
    pairs = read_goff_pairs(ROOT / g, layer="layer2", min_snr=5, track=98, reference_mask=ring, max_rate=MAX_RATE)
    dates = [p["sec_date"] for p in pairs]
    Rr = np.array([ndi.generic_filter(p["range_offset"] * 12.0 / p["days"], np.nanmedian, size=3, mode="nearest") for p in pairs])
    S = np.array([1.4826 * np.median(np.abs(r[ring & np.isfinite(r)] - np.median(r[ring & np.isfinite(r)]))) for r in Rr])
    b = np.where(np.isfinite(Rr[:4]).sum(0) >= 3, np.nanmedian(Rr[:4], axis=0), np.nan)
    floor = float(np.median(S[:4]))
    moving = np.isfinite(b) & (np.abs(b) >= floor)
    s = np.sign(np.nan_to_num(b))
    ratio, incr = [], []
    for k in range(5, len(pairs)):
        sel = [i for i, dt in enumerate(dates[:k + 1]) if PM_START < dt <= PM_END]
        if not sel:
            ratio.append(np.full(X.shape, np.nan)); incr.append(np.full(X.shape, np.nan)); continue
        n = np.isfinite(Rr[sel]).sum(0)
        pm = np.where(n >= MIN_PAIRS, np.nanmedian(Rr[sel], axis=0), np.nan)
        ratio.append(np.where(moving, s * pm / np.abs(b), np.nan))
        incr.append(np.where(moving, (s * pm - np.abs(b)) * 365 / 12, np.nan))
    return dict(ratio=np.array(ratio), incr=np.array(incr), target=target, cutoffs=dates[5:], has_target=has_target,
                area_km2=X.size * 0.0064)


def run(P, R0, I0, MIN_PX):
    fa_union, hits = np.zeros(P["target"].shape, bool), []
    for k in range(len(P["cutoffs"])):
        alarm = (np.nan_to_num(P["ratio"][k], nan=0) >= R0) & (np.nan_to_num(P["incr"][k], nan=-1e9) >= I0)
        lab, n = ndi.label(alarm, structure=np.ones((3, 3)))
        sz = ndi.sum(alarm, lab, range(1, n + 1))
        hit = False
        for i in [i + 1 for i, v in enumerate(sz) if v >= MIN_PX]:
            m = lab == i
            if (m & P["target"]).any():
                hit = True
            else:
                fa_union |= m
        hits.append(hit)
    i0 = persistent_first(hits) if P["has_target"] else None
    first = P["cutoffs"][i0] if i0 is not None else None
    return dict(fa=int(ndi.label(fa_union, structure=np.ones((3, 3)))[1]), first=first,
                lead=(EVENT - date(int(first[:4]), int(first[4:6]), int(first[6:]))).days if first else None)


if __name__ == "__main__":
    prep = {}
    for s_ in SITES:
        print(f"preparing {s_[0]} ...", flush=True)
        prep[s_[0]] = prepare(*s_)
    rows = []
    print("\n  R0    I0  px | FA nepal ganesh_e jugal | Nepal first  lead")
    for MIN_PX in PX_GRID:
        for R0 in R_GRID:
            for I0 in I_GRID:
                r = {n: run(P, R0, I0, MIN_PX) for n, P in prep.items()}
                ok = (r["ganesh_east"]["fa"] <= LIMIT and r["jugal"]["fa"] <= LIMIT
                      and r["nepal"]["first"] is not None and r["nepal"]["first"] <= LATEST_OK)
                rows.append(dict(R0=R0, I0=I0, min_px=MIN_PX, fa={n: v["fa"] for n, v in r.items()},
                                 nepal_first=r["nepal"]["first"], nepal_lead=r["nepal"]["lead"], meets_dev=bool(ok)))
                print(f"{R0:4.1f} {I0:5.1f} {MIN_PX:3d} | {r['nepal']['fa']:8d} {r['ganesh_east']['fa']:8d} {r['jugal']['fa']:5d} | "
                      f"{str(r['nepal']['first']):9s} {str(r['nepal']['lead']):>5s}{'  <== meets' if ok else ''}", flush=True)
    (OUT / "dev_premelt_detector.json").write_text(json.dumps(dict(note="DEVELOPMENT sweep; no pass/fail", rows=rows), indent=1))
    print("wrote", OUT / "dev_premelt_detector.json")
