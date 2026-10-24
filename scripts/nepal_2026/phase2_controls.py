"""Phase 2 (docs/acceptance_phase2.md): detectors E1, E2, E1f, E2f on Nepal and two control sites.

Usage (from the repository root):
    python -u scripts/nepal_2026/phase2_controls.py

Runs the four pre-registered detectors on each site's GOFF crops,
replaying pair by pair, and applies the pass rule. Writes
output/acceptance/phase2.json and phase2.png.
"""
import sys, json, logging, tempfile, yaml, numpy as np
from datetime import date
from pathlib import Path
from matplotlib.path import Path as MP
from scipy import ndimage as ndi
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).parent))
from glews.nisar import load_nisar_goff_stack, read_goff_pairs
from glews.timeseries import compute_acceleration_map
from glews.detect import detect_anomalies
from glews.optical import GDAL_HTTP_ENV
from utm import fwd, inv

logging.basicConfig(level=logging.ERROR)
OUT = ROOT / "output/acceptance"
EVENT, LATEST_OK = date(2026, 8, 26), "20260726"
MAX_SLOPE, MIN_VALID_FRAC, LIMIT_PER_100KM2 = 45.0, 0.70, 2.0
cfg = yaml.safe_load(open(ROOT / "config/nepal_2026.yaml"))
site = cfg["site"]
icimod_ll = site["known_source"]["polygon"]; outlet_ll = site["reference_areas"][0]["polygon"]
SITES = [
    dict(name="nepal", role="development", goff=ROOT / "data/nisar/nepal_2026/goff_t98_crop",
         centre=(site["latitude"], site["longitude"]), target=True),
    dict(name="ganesh_east", role="control", goff=ROOT / "data/nisar/control_ganesh_east/goff_t98_crop",
         centre=(28.36, 85.22), target=False),
    dict(name="jugal", role="control", goff=ROOT / "data/nisar/control_jugal/goff_t98_crop",
         centre=(28.13, 85.86), target=False),
]
DEM = ("https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N28_00_E085_00_DEM/"
       "Copernicus_DSM_COG_10_N28_00_E085_00_DEM.tif")


def slope_on_grid(x, y):
    import rasterio
    from rasterio.vrt import WarpedVRT
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    post = float(x[1] - x[0])
    with rasterio.Env(**GDAL_HTTP_ENV), rasterio.open(DEM) as ds:
        with WarpedVRT(ds, crs="EPSG:32645", transform=from_origin(x[0] - post / 2, y[0] + post / 2, post, post),
                       width=len(x), height=len(y), resampling=Resampling.average) as v:
            z = v.read(1).astype(float)
    g_row, g_col = np.gradient(z, post)
    return np.degrees(np.arctan(np.hypot(g_row, g_col)))


def persistent_first(hits):
    for i, h in enumerate(hits):
        if h and sum(1 for x in hits[i:] if not x) <= 1:
            return i
    return None


def run_site(s):
    goff = s["goff"]
    files = sorted(goff.glob("NISAR_*GOFF*.h5"), key=lambda p: p.name.split("_")[13])
    probe = read_goff_pairs(goff, layer="layer2", min_snr=0, track=98)[0]
    x, y = probe["x_coords"], probe["y_coords"]
    X, Y = np.meshgrid(x, y)
    cx, cy = fwd(*s["centre"])
    inside = lambda ll: MP(np.array([fwd(a, b) for a, b in ll])).contains_points(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
    if s["target"]:
        src = inside(icimod_ll); target = src | inside(outlet_ll)
    else:
        src = np.zeros(X.shape, bool); target = src
    d = np.hypot(X - cx, Y - cy)
    ring = (d > 2000) & (d < 5000) & ~src
    LAT, LON = inv(X, Y)
    slope = slope_on_grid(x, y)
    area_km2 = X.size * (x[1] - x[0]) ** 2 / 1e6
    pairs = read_goff_pairs(goff, layer="layer2", min_snr=5, track=98, reference_mask=ring)
    valid = np.array([np.isfinite(p["range_offset"]) for p in pairs])
    cuts = [p["sec_date"] for p in pairs][5:]
    print(f"\n== {s['name']} ({s['role']}): {len(pairs)} pairs, {X.shape} grid, {area_km2:.0f} km2, "
          f"slope > {MAX_SLOPE:g}: {np.mean(slope > MAX_SLOPE):.0%}", flush=True)

    def mask_at(k):          # k = number of pairs available
        return (slope <= MAX_SLOPE) & (valid[:k].mean(axis=0) >= MIN_VALID_FRAC)

    res = {}
    # ---------------- E1 / E1f ----------------
    for name, filt in (("E1", False), ("E1f", True)):
        rows, fa_union = [], np.zeros(X.shape, bool)
        for k, cd in enumerate(cuts, 6):
            with tempfile.TemporaryDirectory() as td:
                for f in files:
                    if f.name.split("_")[13][:8] <= cd:
                        (Path(td) / f.name).symlink_to(f.resolve())
                ts = load_nisar_goff_stack(td, track=98, reference_mask=ring)
            disp = ts.displacement.copy()
            if filt:
                disp[:, ~mask_at(k)] = np.nan
            am = compute_acceleration_map(ts.dates, disp, window_size_days=cfg["detect"]["acceleration"]["window_size_days"],
                                          step_days=cfg["detect"]["acceleration"]["step_days"],
                                          n_harmonics=cfg["detect"].get("n_harmonics", 2))
            fl = sorted(detect_anomalies(am, LAT, LON, cfg, dates=ts.dates, displacement=disp), key=lambda f: -f.score)
            hit_rank = None
            for i, f in enumerate(fl):
                pix = np.zeros(X.shape, bool)
                if len(f.pixel_indices):
                    pix[f.pixel_indices[:, 0], f.pixel_indices[:, 1]] = True
                if (pix & target).any():
                    hit_rank = hit_rank or i + 1
                else:
                    fa_union |= pix
            rows.append(dict(cutoff=cd, n_flags=len(fl), hit_rank=hit_rank))
            print(f"  {name} {cd}: {len(fl):3d} flags, hit rank {hit_rank}", flush=True)
        res[name] = (rows, fa_union)
    # ---------------- E2 / E2f ----------------
    Rr = np.array([ndi.generic_filter(p["range_offset"] * 12.0 / p["days"], np.nanmedian, size=3, mode="nearest") for p in pairs])
    S = np.array([1.4826 * np.median(np.abs(r[ring & np.isfinite(r)] - np.median(r[ring & np.isfinite(r)]))) for r in Rr])
    nb = np.isfinite(Rr[:4]).sum(axis=0)
    b = np.where(nb >= 3, np.nanmedian(Rr[:4], axis=0), np.nan)
    sb = 1.2533 * np.median(S[:4]) / np.sqrt(np.maximum(nb, 1))
    Z = np.abs(Rr - b) / np.sqrt(S[:, None, None] ** 2 + sb ** 2)
    for name, filt in (("E2", False), ("E2f", True)):
        rows, fa_union = [], np.zeros(X.shape, bool)
        for k in range(5, len(pairs)):
            alarm = (np.nan_to_num(Z[max(4, k - 3):k + 1], nan=0) >= 3).sum(axis=0) >= 3
            if filt:
                alarm &= mask_at(k + 1)
            lab, n = ndi.label(alarm, structure=np.ones((3, 3)))
            sizes = ndi.sum(alarm, lab, range(1, n + 1))
            flags = [i + 1 for i, s_ in enumerate(sizes) if s_ >= 4]
            hit = False
            for i in flags:
                m = lab == i
                if (m & target).any():
                    hit = True
                else:
                    fa_union |= m
            rows.append(dict(cutoff=pairs[k]["sec_date"], n_flags=len(flags), hit_rank=1 if hit else None))
        print(f"  {name}: flags per cutoff {[r['n_flags'] for r in rows]}", flush=True)
        res[name] = (rows, fa_union)

    out = {}
    for name, (rows, fa_union) in res.items():
        n_fa = int(ndi.label(fa_union, structure=np.ones((3, 3)))[1])
        i0 = persistent_first([r["hit_rank"] is not None for r in rows]) if s["target"] else None
        first = rows[i0]["cutoff"] if i0 is not None else None
        out[name] = dict(cutoffs=rows, false_alarm_flags=n_fa, false_alarm_km2=round(float(fa_union.sum()) * 0.0064, 2),
                         rate_per_100km2=round(n_fa / area_km2 * 100, 2),
                         first_persistent_hit=first,
                         lead_days=(EVENT - date(int(first[:4]), int(first[4:6]), int(first[6:]))).days if first else None)
        print(f"  {name}: false-alarm flags {n_fa} ({out[name]['rate_per_100km2']}/100 km2, {out[name]['false_alarm_km2']} km2)"
              + (f", persistent hit {first} (lead {out[name]['lead_days']} d)" if s["target"] else ""), flush=True)
    return dict(area_km2=round(area_km2, 1), detectors=out, fa_maps={k: v[1] for k, v in res.items()},
                extent=[x[0] - 40, x[-1] + 40, y[-1] - 40, y[0] + 40])


results = {s["name"]: run_site(s) for s in SITES}
verdicts = {}
for det in ("E1", "E2", "E1f", "E2f"):
    nep = results["nepal"]["detectors"][det]
    hit_ok = bool(nep["first_persistent_hit"] and nep["first_persistent_hit"] <= LATEST_OK)
    ctrl = {n: results[n]["detectors"][det]["rate_per_100km2"] for n in ("ganesh_east", "jugal")}
    ctrl_ok = all(results[n]["detectors"][det]["false_alarm_flags"] <= int(LIMIT_PER_100KM2 * results[n]["area_km2"] / 100)
                  for n in ctrl)
    verdicts[det] = dict(nepal_hit_lead_ok=hit_ok, nepal_lead_days=nep["lead_days"], nepal_rate_per_100km2=nep["rate_per_100km2"],
                         control_rates_per_100km2=ctrl, controls_ok=ctrl_ok, verdict="PASS" if hit_ok and ctrl_ok else "FAIL")
print("\nVerdicts:")
for det, v in verdicts.items():
    print(f"  {det}: Nepal lead {v['nepal_lead_days']} d (in-sample for filtered), Nepal FA {v['nepal_rate_per_100km2']}/100 km2; "
          f"controls {v['control_rates_per_100km2']} -> {v['verdict']}")
(OUT / "phase2.json").write_text(json.dumps(dict(criteria="docs/acceptance_phase2.md", verdicts=verdicts,
                                                 sites={k: {kk: vv for kk, vv in v.items() if kk != "fa_maps"} for k, v in results.items()}),
                                            indent=1, default=float))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axs = plt.subplots(3, 4, figsize=(14, 10.5))
for i, (sn, r) in enumerate(results.items()):
    for j, det in enumerate(("E1", "E1f", "E2", "E2f")):
        ax = axs[i, j]
        ax.imshow(np.where(r["fa_maps"][det], 1.0, np.nan), extent=r["extent"], cmap="autumn", vmin=0, vmax=1)
        ax.set_facecolor("0.92"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{sn} {det}: {r['detectors'][det]['false_alarm_flags']} FA flags", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "phase2.png", dpi=90)
print("wrote", OUT / "phase2.json")
