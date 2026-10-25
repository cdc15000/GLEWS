"""DEVELOPMENT go/no-go checks for a Sentinel-1 multi-year baseline (no pass/fail).

Usage (from the repository root):
    python -u scripts/nepal_2026/dev_s1_checks.py

Seen sites only (Nepal, Ganesh East, Jugal Himal, Annapurna Sanctuary,
Langtang East). Manaslu West, Lapchi and Annapurna IV north are held out:
neither their GOFF nor their ITS_LIVE data are read here.

1. Persistence: are E4's false-alarm flags (frozen setting 1.5 / 12 / 6)
   still alarms when the pre-melt window closes (27 May cutoff)?
2. Resolution: ITS_LIVE Sentinel-1 (autoRIFT, 120 m) window-median noise
   over stable ground (landice == 0) vs the flagged pixels' winter speed;
   chip sizes at the outlet.
3. History: per year 2015-2022, the pre-melt ratio
   |median(vx, vy) 15 Jan - 27 May| / |median(vx, vy) 28 Nov - 15 Jan|
   (pairs dated by mid-date, >= 2 valid pairs per window) for each E4
   flag and for the outlet.
Writes output/acceptance/dev_s1_checks.json.
"""
import sys, json, logging, numpy as np, pandas as pd, xarray as xr
from pathlib import Path
from scipy import ndimage as ndi
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).parent))
from glews.nisar import read_goff_pairs
from rasterio.warp import transform as rtransform
from matplotlib.path import Path as MP
from utm import fwd
import dev_premelt_detector as D

logging.basicConfig(level=logging.ERROR)
R0, I0, MIN_PX, CLOSE = 1.5, 12.0, 6, "20260527"
CUBE = "https://its-live-data.s3.amazonaws.com/datacubes/v2-updated-october2024/N20E080/ITS_LIVE_vel_EPSG{epsg}_G0120_X{x}_Y3150000.zarr"
SITES = D.SITES + [("annapurna_sanctuary", "data/nisar/control_annapurna_sanctuary/goff_t98_crop", (28.55, 83.90), False),
                   ("langtang_east", "data/nisar/control_langtang_east/goff_t98_crop", (28.25, 85.72), False)]
CUBES = {"32645": CUBE.format(epsg=32645, x=350000), "32644": CUBE.format(epsg=32644, x=750000)}
YEARS = range(2015, 2023)


def e4_flags(P):
    """Flags (union over cutoffs) and whether each is still alarmed at the window-closing cutoff."""
    alarms = [(np.nan_to_num(P["ratio"][k], nan=0) >= R0) & (np.nan_to_num(P["incr"][k], nan=-1e9) >= I0)
              for k in range(len(P["cutoffs"]))]
    union = np.zeros(P["target"].shape, bool); tgt = np.zeros_like(union)
    for a in alarms:
        lab, n = ndi.label(a, structure=np.ones((3, 3))); sz = ndi.sum(a, lab, range(1, n + 1))
        for i in [i + 1 for i, v in enumerate(sz) if v >= MIN_PX]:
            m = lab == i
            (tgt if (m & P["target"]).any() else union)[:] |= m
    k_close = P["cutoffs"].index(CLOSE)
    out = []
    for name, u in (("fa", union), ("target", tgt)):
        lab, n = ndi.label(u, structure=np.ones((3, 3)))
        for i in range(1, n + 1):
            m = lab == i
            out.append(dict(kind=name, mask=m, px=int(m.sum()),
                            alarmed_px_at_close=int((alarms[k_close] & m).sum()),
                            cutoffs_alarmed=int(sum((a & m).sum() >= max(1, m.sum() // 3) for a in alarms))))
    return out


def s1_cube(epsg, X, Y):
    """ITS_LIVE Sentinel-1 vx, vy over the crop (120 m), with mid-dates and chip sizes."""
    ds = xr.open_zarr(CUBES[epsg], consolidated=True)
    sat = ds.satellite_img1.values.astype(str)
    idx = np.nonzero(np.char.startswith(sat, "1"))[0]
    if epsg == "32645":
        xs, ys = X, Y
    else:
        xs, ys = (np.array(a).reshape(X.shape) for a in rtransform("EPSG:32645", f"EPSG:{epsg}", X.ravel(), Y.ravel()))
    sub = ds.isel(mid_date=idx).sel(x=slice(xs.min() - 240, xs.max() + 240), y=slice(ys.max() + 240, ys.min() - 240))
    vx, vy = sub.vx.load().values, sub.vy.load().values
    mid = pd.to_datetime(sub.mid_date.values)
    ci = np.clip(np.round((xs - sub.x.values[0]) / 120).astype(int), 0, len(sub.x) - 1)
    ri = np.clip(np.round((sub.y.values[0] - ys) / 120).astype(int), 0, len(sub.y) - 1)
    land = sub.landice.load().values if "landice" in sub else None
    chips = (sub.chip_size_height.isel(mid_date=slice(None)).load().values if "chip_size_height" in sub else None)
    return dict(vx=vx, vy=vy, mid=mid, ri=ri, ci=ci, landice=land, chips=chips, chips_w=sub.chip_size_width.load().values)


def window_speed(C, a, b, r, c):
    sel = (C["mid"] > a) & (C["mid"] <= b)
    vx, vy = C["vx"][sel][:, r, c], C["vy"][sel][:, r, c]
    n = np.isfinite(vx).sum(0)
    return np.where(n >= 2, np.hypot(np.nanmedian(vx, axis=0), np.nanmedian(vy, axis=0)), np.nan), n


report = {}
for name, g, centre, has_t in SITES:
    print(f"\n== {name}", flush=True)
    P = D.prepare(name, g, centre, has_t)
    flags = e4_flags(P)
    probe = read_goff_pairs(ROOT / g, layer="layer2", min_snr=0, track=98)[0]
    X, Y = np.meshgrid(probe["x_coords"], probe["y_coords"])
    epsg = "32644" if centre[1] < 84.0 else "32645"
    C = s1_cube(epsg, X, Y)
    outlet = None
    if has_t:
        outlet = MP(np.array([fwd(a, b) for a, b in D.outlet_ll])).contains_points(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
        flags.append(dict(kind="outlet", mask=outlet, px=int(outlet.sum()), alarmed_px_at_close=None, cutoffs_alarmed=None))
    # 2. noise over stable ground (cube landice == 0) at the crop's ITS_LIVE pixels
    rr, cc = np.unique(np.c_[C["ri"].ravel(), C["ci"].ravel()], axis=0).T
    stable = (C["landice"][rr, cc] == 0) if C["landice"] is not None else np.ones(rr.size, bool)
    noise = {}
    for yr in YEARS:
        for wn, a, b in (("base", f"{yr - 1}-11-28", f"{yr}-01-15"), ("pm", f"{yr}-01-15", f"{yr}-05-27")):
            v, n = window_speed(C, pd.Timestamp(a), pd.Timestamp(b), rr[stable], cc[stable])
            if np.isfinite(v).sum() > 20:
                noise[f"{yr}_{wn}"] = round(float(np.nanmedian(v)), 1)
    print(f"  stable ITS_LIVE pixels {stable.sum()}; window-median stable-ground speed (m/yr, median over px): {noise}")
    rows = []
    for f in flags:
        r, c = C["ri"][f["mask"]], C["ci"][f["mask"]]
        hist = {}
        for yr in YEARS:
            vb, nb = window_speed(C, pd.Timestamp(f"{yr - 1}-11-28"), pd.Timestamp(f"{yr}-01-15"), r, c)
            vp, npm = window_speed(C, pd.Timestamp(f"{yr}-01-15"), pd.Timestamp(f"{yr}-05-27"), r, c)
            if np.isfinite(vb).sum() and np.isfinite(vp).sum():
                hist[yr] = dict(base=round(float(np.nanmedian(vb)), 1), pm=round(float(np.nanmedian(vp)), 1),
                                ratio=round(float(np.nanmedian(vp) / np.nanmedian(vb)), 2), n_pairs=(int(np.median(nb)), int(np.median(npm))))
        ch = None
        if f["kind"] == "outlet" and C["chips"] is not None:
            ch = (int(np.nanmedian(C["chips"][:, r, c])), int(np.nanmedian(C["chips_w"][:, r, c])))
        row = {k: v for k, v in f.items() if k != "mask"}
        row.update(hist=hist, chip_px=ch)
        rows.append(row)
        rs = " ".join(f"{y}:{h['ratio']}({h['base']:.0f})" for y, h in hist.items())
        print(f"  {f['kind']:6s} {f['px']:3d}px close-alarmed {str(f['alarmed_px_at_close']):>4s} cutoffs {str(f['cutoffs_alarmed']):>4s} "
              f"| hist ratio(base m/yr): {rs}" + (f" | chip (m) {ch}" if ch else ""), flush=True)
    report[name] = dict(stable_noise=noise, flags=rows)
(D.OUT / "dev_s1_checks.json").write_text(json.dumps(dict(note="DEVELOPMENT; no pass/fail; held-out sites untouched", sites=report),
                                                    indent=1, default=str))
print("wrote", D.OUT / "dev_s1_checks.json")
