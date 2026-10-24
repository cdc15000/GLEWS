"""Terrain, surface and coverage statistics for candidate control sites (no GOFF values read).

Usage (from the repository root):
    python -u scripts/nepal_2026/select_controls.py

For each candidate 12 x 12 km crop (EPSG:32645), reports:
- GLO-30 elevation p10/50/90, median slope, share of slope > 45 deg
  (tiles mosaicked, so crops off N28_E085 are handled; nodata share reported);
- Sentinel-2 SCL snow/ice share (clear Oct-Dec 2025 scene, cached);
- whether the crop (with a 1 km margin) lies inside the track-98 frame-16
  footprint, and the number of pre-event GOFF date pairs ASF lists for
  the centre (metadata only);
- distance to the Nepal source and the two Phase 2 controls.
Writes output/acceptance/control_candidates.json. Used to choose the
Phase 3 fresh controls before any of their GOFF data are extracted.
"""
import sys, json, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).parent))
from glews.optical import GDAL_HTTP_ENV, search_sentinel2, read_window
from matplotlib.path import Path as MP
from utm import fwd

OUT = ROOT / "output/acceptance"; S2 = ROOT / "data/sentinel2/residuals"; S2.mkdir(parents=True, exist_ok=True)
HALF, POST = 6000.0, 80.0
FRAME = [(86.60648, 27.97942), (85.94647, 30.10547), (83.37345, 29.46177), (84.11993, 27.36049)]
SEEN = {"nepal": (28.288, 85.524), "ganesh_east": (28.36, 85.22), "jugal": (28.13, 85.86)}
CANDIDATES = {
    "annapurna_sanctuary": (28.55, 83.90),
    "annapurna_ii_south": (28.50, 84.15),
    "annapurna_iv_north": (28.60, 84.08),
    "manaslu_west": (28.58, 84.48),
    "himlung": (28.76, 84.32),
    "langtang_east": (28.25, 85.72),
    "lapchi": (28.00, 86.18),
}
TILE = ("https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM/"
        "Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM.tif")


def dem(lat, lon):
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_origin
    cx, cy = fwd(lat, lon)
    n = int(2 * HALF / POST)
    dst = np.full((n, n), np.nan)
    tr = from_origin(cx - HALF, cy + HALF, POST, POST)
    with rasterio.Env(**GDAL_HTTP_ENV):
        for la in range(int(np.floor(lat - 0.1)), int(np.floor(lat + 0.1)) + 1):
            for lo in range(int(np.floor(lon - 0.1)), int(np.floor(lon + 0.1)) + 1):
                with rasterio.open(TILE.format(lat=la, lon=lo)) as ds:
                    part = np.full((n, n), np.nan)
                    reproject(rasterio.band(ds, 1), part, dst_transform=tr, dst_crs="EPSG:32645",
                              resampling=Resampling.average, dst_nodata=np.nan)
                    dst = np.where(np.isfinite(part), part, dst)
    return dst, tr


def snow_share(name, lat, lon):
    f = S2 / f"{name}.npz"
    if not f.exists():
        cx, cy = fwd(lat, lon); b = (cx - HALF, cy - HALF, cx + HALF, cy + HALF)
        for s in sorted(search_sentinel2(lat, lon, "2025-10-10", "2025-12-31", max_cloud=20), key=lambda s: s.cloud_cover):
            if s.epsg != 32645:
                continue
            scl, t = read_window(s.hrefs["scl"], b)
            if np.mean(scl == 0) > 0.02 or np.mean(np.isin(scl, (3, 8, 9, 10))) > 0.05:
                continue
            rgb = np.dstack([read_window(s.hrefs[k], b)[0] for k in ("red", "green", "blue")])
            np.savez_compressed(f, rgb=rgb, scl=scl, scl_transform=np.array(t), scene=s.id)
            break
        else:
            return None, None
    z = np.load(f)
    return float(np.mean(z["scl"] == 11)), str(z["scene"])


def goff_pairs(lat, lon):
    import asf_search as asf
    res = asf.search(platform=asf.PLATFORM.NISAR, intersectsWith=f"POINT({lon} {lat})",
                     start="2025-10-01", end="2026-08-27", processingLevel="GOFF")
    pairs = set()
    for x in res:
        p = x.properties
        if p["pathNumber"] != 98:
            continue
        d = [t[:8] for t in p["fileID"].split("_") if len(t) >= 15 and t[0] == "2" and "T" in t]
        if d[2] <= "20260826":
            pairs.add((d[0], d[2], p.get("frameNumber")))
    return pairs


def km(a, b):
    (x1, y1), (x2, y2) = fwd(*a), fwd(*b)
    return np.hypot(x1 - x2, y1 - y2) / 1000


from utm import inv
frame = MP(np.array(FRAME))
rows = {}
ref_pairs = goff_pairs(*SEEN["nepal"])
for name, (lat, lon) in {**{"nepal (dev)": SEEN["nepal"]}, **CANDIDATES}.items():
    z, tr = dem(lat, lon)
    g_row, g_col = np.gradient(z, POST)
    slope = np.degrees(np.arctan(np.hypot(g_row, g_col)))
    cx, cy = fwd(lat, lon); m = HALF + 1000
    corners = [inv(cx + sx * m, cy + sy * m) for sx in (-1, 1) for sy in (-1, 1)]
    in_frame = bool(frame.contains_points([(lo, la) for la, lo in corners]).all())
    pairs = goff_pairs(lat, lon) if not name.startswith("nepal") else ref_pairs
    snow, scene = snow_share(name.split(" ")[0], lat, lon)
    rows[name] = dict(centre=(lat, lon), elev_p10_50_90=[int(v) for v in np.nanpercentile(z, [10, 50, 90])],
                      median_slope=round(float(np.nanmedian(slope)), 1),
                      share_gt45=round(float(np.nanmean(slope[np.isfinite(slope)] > 45)), 2),
                      dem_nodata=round(float(np.mean(~np.isfinite(z))), 3),
                      snow_ice_share=None if snow is None else round(snow, 2), s2_scene=scene,
                      in_frame=in_frame, n_pairs=len({(a, b) for a, b, _ in pairs}),
                      same_pairs_as_nepal=({(a, b) for a, b, _ in pairs} == {(a, b) for a, b, _ in ref_pairs}),
                      frames=sorted({f for *_, f in pairs if f is not None}),
                      km_to_seen={k: round(km((lat, lon), v), 1) for k, v in SEEN.items()})
    r = rows[name]
    print(f"{name:22s} elev {r['elev_p10_50_90']}  slope {r['median_slope']:4.1f}  >45 {r['share_gt45']:.2f}  "
          f"snow {r['snow_ice_share']}  in_frame {r['in_frame']}  pairs {r['n_pairs']} same {r['same_pairs_as_nepal']} "
          f"frames {r['frames']}  dist {r['km_to_seen']}  nodata {r['dem_nodata']}", flush=True)
(OUT / "control_candidates.json").write_text(json.dumps(rows, indent=1))
