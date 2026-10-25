"""DEVELOPMENT: E4 restricted to failure-relevant terrain (reach to real exposure). No pass/fail.

Usage (from the repository root):
    python -u scripts/nepal_2026/dev_exposure.py

Seen sites only. Held-out sites (Manaslu West, Lapchi, Annapurna IV north)
are not read.

For each site:
- Domain: the crop centre +/- 46 km at 90 m (GLO-30 mosaic, EPSG:32645).
- Assets: OpenStreetMap as of 2026-08-01 (before the event): settlements,
  roads, power plants and generators, dams.
- Reach: glews.exposure.reach_exposure from every crop cell, D8 on the
  filled DEM, buffer 250 m. Travel-angle floor H/L in {0.10, 0.15, 0.20}.
- Relevant mask on the NISAR 80 m grid: the crop pixels that reach exposure.
- E4 (frozen 1.5 / 12 / 6) false alarms counted two ways:
  (a) alarm pixels masked to relevant terrain before grouping;
  (b) all flags, kept if any of their pixels is relevant.
Reports the relevant share of the crop, the false alarms, whether the
Nepal hit survives, and the assets the Nepal target reaches.
Writes output/acceptance/dev_exposure.json.
"""
import sys, json, logging, numpy as np
from pathlib import Path
from scipy import ndimage as ndi
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).parent))
from glews.nisar import read_goff_pairs
from glews.exposure import fetch_osm_assets, rasterize_assets, reach_exposure, fill_depressions, glo30_mosaic
from rasterio.warp import transform as rtransform, transform_bounds
from utm import fwd
import dev_premelt_detector as D

logging.basicConfig(level=logging.ERROR)
R0, I0, MIN_PX = 1.5, 12.0, 6
POST, HALF_DOMAIN, BUFFER, OSM_DATE = 90.0, 46000.0, 250.0, "2026-08-01T00:00:00Z"
HL = (0.10, 0.15, 0.20)
CACHE = ROOT / "data/exposure"
SITES = D.SITES + [("annapurna_sanctuary", "data/nisar/control_annapurna_sanctuary/goff_t98_crop", (28.55, 83.90), False),
                   ("langtang_east", "data/nisar/control_langtang_east/goff_t98_crop", (28.25, 85.72), False)]


def site_reach(name, centre, X, Y):
    """Relevant masks (per H/L) on the NISAR grid, plus the asset grid and DEM pieces."""
    cx, cy = fwd(*centre)
    x0, y0 = round((cx - HALF_DOMAIN) / POST) * POST, round((cy + HALF_DOMAIN) / POST) * POST
    n = int(2 * HALF_DOMAIN / POST)
    z = glo30_mosaic(x0, y0, POST, (n, n), cache=CACHE / f"{name}_dem90.npy")
    w, s, e, nn = transform_bounds("EPSG:32645", "EPSG:4326", x0, y0 - n * POST, x0 + n * POST, y0)
    assets = fetch_osm_assets(s, w, nn, e, date=OSM_DATE, cache=CACHE / f"{name}_osm_20260801.json")
    xy = [tuple(np.array(v) for v in rtransform("EPSG:4326", "EPSG:32645", a["lon"], a["lat"])) for a in assets]
    exposed = rasterize_assets(xy, x0, y0, POST, (n, n))
    ri = np.clip(((y0 - Y) / POST).astype(int), 0, n - 1); ci = np.clip(((X - x0) / POST).astype(int), 0, n - 1)
    src = np.zeros((n, n), bool); src[ri, ci] = True
    zf = fill_depressions(z)
    masks = {hl: reach_exposure(z, POST, exposed, src, min_hl=hl, buffer_m=BUFFER, zf=zf)[ri, ci] for hl in HL}
    kinds = {}
    for a in assets:
        kinds[a["kind"]] = kinds.get(a["kind"], 0) + 1
    return masks, dict(assets=kinds, exposed_cells=int(exposed.sum()), dem_nodata=float(np.mean(~np.isfinite(z))))


def e4_counts(P, relevant):
    fa_a, fa_b, hit_a, hit_b = np.zeros(relevant.shape, bool), np.zeros(relevant.shape, bool), [], []
    for k in range(len(P["cutoffs"])):
        a = (np.nan_to_num(P["ratio"][k], nan=0) >= R0) & (np.nan_to_num(P["incr"][k], nan=-1e9) >= I0)
        for alarm, fa, hits, mode in ((a & relevant, fa_a, hit_a, "a"), (a, fa_b, hit_b, "b")):
            lab, n = ndi.label(alarm, structure=np.ones((3, 3))); sz = ndi.sum(alarm, lab, range(1, n + 1))
            h = False
            for i in [i + 1 for i, v in enumerate(sz) if v >= MIN_PX]:
                m = lab == i
                if mode == "b" and not (m & relevant).any():
                    continue
                if (m & P["target"]).any():
                    h = True
                else:
                    fa |= m
            hits.append(h)
    cnt = lambda u: int(ndi.label(u, structure=np.ones((3, 3)))[1])
    first = lambda hs: (P["cutoffs"][D.persistent_first(hs)] if D.persistent_first(hs) is not None else None)
    return dict(fa_masked=cnt(fa_a), fa_any_px=cnt(fa_b),
                nepal_first_masked=first(hit_a) if P["has_target"] else None,
                nepal_first_any_px=first(hit_b) if P["has_target"] else None)


report = {}
for name, g, centre, has_t in SITES:
    print(f"\n== {name}", flush=True)
    probe = read_goff_pairs(ROOT / g, layer="layer2", min_snr=0, track=98)[0]
    X, Y = np.meshgrid(probe["x_coords"], probe["y_coords"])
    masks, info = site_reach(name, centre, X, Y)
    P = D.prepare(name, g, centre, has_t)
    print(f"  OSM assets {info['assets']}, exposed cells {info['exposed_cells']}, DEM nodata {info['dem_nodata']:.3f}", flush=True)
    rows = {}
    for hl, m in masks.items():
        r = e4_counts(P, m)
        r.update(relevant_share=round(float(m.mean()), 3),
                 target_relevant_share=round(float(m[P["target"]].mean()), 2) if has_t else None)
        rows[str(hl)] = r
        print(f"  H/L >= {hl:.2f}: relevant {r['relevant_share']:.0%} of crop"
              + (f" (target {r['target_relevant_share']:.0%})" if has_t else "")
              + f" | E4 FA masked {r['fa_masked']}, any-pixel {r['fa_any_px']}"
              + (f" | Nepal hit masked {r['nepal_first_masked']}, any-pixel {r['nepal_first_any_px']}" if has_t else ""), flush=True)
    report[name] = dict(info=info, by_hl=rows)
(D.OUT / "dev_exposure.json").write_text(json.dumps(dict(note="DEVELOPMENT; no pass/fail; held-out sites untouched",
                                                      osm_date=OSM_DATE, buffer_m=BUFFER, sites=report), indent=1))
print("wrote", D.OUT / "dev_exposure.json")
