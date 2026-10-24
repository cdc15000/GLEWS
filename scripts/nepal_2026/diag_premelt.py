"""DEVELOPMENT check: is the Nepal outlet's pre-melt acceleration unusual among ALL moving pixels? (no pass/fail)

Usage (from the repository root):
    python -u scripts/nepal_2026/diag_premelt.py

diag_residuals.py compared the target only with clusters a gate had
already picked, which selects summer-triggered glaciers. Here every pixel
with a moving Nov-Jan baseline (|b| >= the ring noise) at each site is
used, with loader blunder rejection (max_rate 5/12 m/day). Pre-melt ratio
= s * median rate (15 Jan - 27 May) / |b|, s = sign(b); increase = the
same difference in m/yr. Reports where the Nepal outlet (Wang et al.
measurement area) and target pixels rank against each control's
distribution, for all moving pixels and for snow/ice (Dec 2025 SCL).
Writes output/acceptance/diag_premelt.json.
"""
import sys, json, logging, yaml, numpy as np
from pathlib import Path
from matplotlib.path import Path as MP
from scipy import ndimage as ndi
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).parent))
from glews.nisar import read_goff_pairs
from utm import fwd

logging.basicConfig(level=logging.ERROR)
OUT = ROOT / "output/acceptance"; S2 = ROOT / "data/sentinel2/residuals"
cfg = yaml.safe_load(open(ROOT / "config/nepal_2026.yaml")); site = cfg["site"]
icimod_ll = site["known_source"]["polygon"]; outlet_ll = site["reference_areas"][0]["polygon"]
SITES = [("nepal", "data/nisar/nepal_2026/goff_t98_crop", (site["latitude"], site["longitude"])),
         ("ganesh_east", "data/nisar/control_ganesh_east/goff_t98_crop", (28.36, 85.22)),
         ("jugal", "data/nisar/control_jugal/goff_t98_crop", (28.13, 85.86))]
res = {}
for name, g, centre in SITES:
    pairs = read_goff_pairs(ROOT / g, layer="layer2", min_snr=0, track=98)
    X, Y = np.meshgrid(pairs[0]["x_coords"], pairs[0]["y_coords"])
    cx, cy = fwd(*centre)
    inside = lambda ll: MP(np.array([fwd(a, b) for a, b in ll])).contains_points(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
    src = inside(icimod_ll) if name == "nepal" else np.zeros(X.shape, bool)
    outlet = inside(outlet_ll) if name == "nepal" else src
    d = np.hypot(X - cx, Y - cy); ring = (d > 2000) & (d < 5000) & ~src
    pairs = read_goff_pairs(ROOT / g, layer="layer2", min_snr=5, track=98, reference_mask=ring, max_rate=5.0 / 12)
    dates = np.array([p["sec_date"] for p in pairs])
    Rr = np.array([ndi.generic_filter(p["range_offset"] * 12.0 / p["days"], np.nanmedian, size=3, mode="nearest") for p in pairs])
    S = np.array([1.4826 * np.median(np.abs(r[ring & np.isfinite(r)] - np.median(r[ring & np.isfinite(r)]))) for r in Rr])
    b = np.where(np.isfinite(Rr[:4]).sum(0) >= 3, np.nanmedian(Rr[:4], axis=0), np.nan)
    pm_sel = (dates > "20260115") & (dates <= "20260527")
    pm = np.where(np.isfinite(Rr[pm_sel]).sum(0) >= 5, np.nanmedian(Rr[pm_sel], axis=0), np.nan)
    floor = float(np.median(S[:4]))
    moving = np.isfinite(b) & np.isfinite(pm) & (np.abs(b) >= floor)
    s = np.sign(b)
    ratio = np.where(moving, s * pm / np.abs(b), np.nan)
    incr = np.where(moving, (s * pm - np.abs(b)) * 365 / 12, np.nan)
    sc = np.load(S2 / f"{name}.npz"); t = sc["scl_transform"]
    ci = ((X - t[0]) / t[1]).astype(int).clip(0, sc["scl"].shape[1] - 1)
    ri = ((Y - t[3]) / t[5]).astype(int).clip(0, sc["scl"].shape[0] - 1)
    snow = sc["scl"][ri, ci] == 11
    res[name] = dict(ratio=ratio, incr=incr, moving=moving, snow=snow, outlet=outlet, src=src, floor=floor)
    print(f"{name}: moving pixels {moving.sum()} (snow-covered {(moving & snow).sum()}), noise floor {floor * 365 / 12:.1f} m/yr", flush=True)

N = res["nepal"]
probes = {"outlet (median)": N["outlet"] & N["moving"], "ICIMOD outline (median)": N["src"] & N["moving"]}
out = {}
print("\nPercentile of the Nepal probe within each control's moving-pixel distribution (100 = most extreme)")
for pname, pm_ in probes.items():
    if not pm_.any():
        print(f"  {pname}: no moving pixels"); continue
    vr, vi = float(np.nanmedian(N["ratio"][pm_])), float(np.nanmedian(N["incr"][pm_]))
    line = {}
    for cname in ("ganesh_east", "jugal", "nepal"):
        C = res[cname]
        for sub, msk in (("all", C["moving"]), ("snow", C["moving"] & C["snow"])):
            if cname == "nepal":
                msk = msk & ~N["src"] & ~N["outlet"]
            r_, i_ = C["ratio"][msk], C["incr"][msk]
            line[f"{cname}/{sub}"] = dict(n=int(msk.sum()), ratio_pct=round(float(np.mean(r_ < vr) * 100), 1),
                                          incr_pct=round(float(np.mean(i_ < vi) * 100), 1),
                                          n_exceeding_both=int(np.sum((r_ >= vr) & (i_ >= vi))))
    out[pname] = dict(ratio=round(vr, 2), increase_m_per_yr=round(vi, 1), n_px=int(pm_.sum()), vs=line)
    print(f"  {pname}: {pm_.sum()} px, ratio {vr:.2f}, increase {vi:+.1f} m/yr")
    for k, v in line.items():
        print(f"     vs {k:18s} (n={v['n']:5d}): ratio pct {v['ratio_pct']:5.1f}, increase pct {v['incr_pct']:5.1f}, "
              f"pixels exceeding both {v['n_exceeding_both']}")
(OUT / "diag_premelt.json").write_text(json.dumps(dict(note="DEVELOPMENT; no pass/fail", probes=out), indent=1))
