"""Fetch Sentinel-2 L2A crops around the source for optical offset tracking.

Usage (from the repository root):
    python scripts/nepal_2026/fetch_s2_crops.py [OUT_DIR] [ORBIT]

Two passes, both cached (re-running only fetches what is missing):
1. For every scene of relative orbit ORBIT (default 119, which covers the
   whole of tile 45RUM) from 2017 to the day before the event, read the
   20 m scene classification (SCL) over the crop and record cloud, shadow,
   snow and nodata fractions over the source polygon and the stable ring
   in OUT_DIR/index.json.
2. For scenes whose source is <= 30 % cloud or shadow, with no nodata,
   read the 10 m near-infrared band (B08) over the crop into
   OUT_DIR/b08/<scene>.npz.

The crop is 11 x 11 km centred on the source, in the tile's CRS
(EPSG:32645, the same as the NISAR grids).
"""
import sys, json, time, yaml, numpy as np
from pathlib import Path
from matplotlib.path import Path as MP
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).parent))
from glews.optical import search_sentinel2, read_window, scl_fractions
from utm import fwd

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/sentinel2/nepal_2026"
ORBIT = int(sys.argv[2]) if len(sys.argv) > 2 else 119
HALF = 5500.0
MAX_BAD = 0.30
site = yaml.safe_load(open(ROOT / "config/nepal_2026.yaml"))["site"]
cx, cy = fwd(site["latitude"], site["longitude"])
cx, cy = round(cx / 20) * 20, round(cy / 20) * 20          # align to the 20 m SCL grid
BOUNDS = (cx - HALF, cy - HALF, cx + HALF, cy + HALF)
poly = np.array([fwd(la, lo) for la, lo in site["known_source"]["polygon"]])


def masks(shape, transform):
    x0, dx, _, y0, _, dy = transform
    xs = x0 + dx * (np.arange(shape[1]) + 0.5); ys = y0 + dy * (np.arange(shape[0]) + 0.5)
    X, Y = np.meshgrid(xs, ys)
    src = MP(poly).contains_points(np.c_[X.ravel(), Y.ravel()]).reshape(shape)
    d = np.hypot(X - cx, Y - cy)
    return src, (d > 2000) & (d < 5000) & ~src


(OUT / "b08").mkdir(parents=True, exist_ok=True)
idx_path = OUT / "index.json"
index = json.loads(idx_path.read_text()) if idx_path.exists() else {}
scenes = search_sentinel2(site["latitude"], site["longitude"], "2017-01-01", "2026-08-25",
                          max_cloud=80, relative_orbit=ORBIT)
print(f"{len(scenes)} scenes on R{ORBIT:03d}; {len(index)} already classified", flush=True)

for sc in scenes:
    if sc.id in index:
        continue
    for attempt in range(3):
        try:
            scl, t = read_window(sc.hrefs["scl"], BOUNDS)
            src, ring = masks(scl.shape, t)
            index[sc.id] = dict(date=sc.date, datetime=sc.datetime, orbit=sc.relative_orbit,
                                tile_cloud=sc.cloud_cover, sun_elevation=round(sc.sun_elevation, 2),
                                sun_azimuth=round(sc.sun_azimuth, 2), epsg=sc.epsg,
                                source=scl_fractions(scl, src), ring=scl_fractions(scl, ring),
                                b08=sc.hrefs.get("nir"))
            break
        except Exception as e:
            print("retry", sc.id, type(e).__name__, str(e)[:80], flush=True); time.sleep(3)
    if len(index) % 20 == 0:
        idx_path.write_text(json.dumps(index, indent=1))
idx_path.write_text(json.dumps(index, indent=1))

good = {k: v for k, v in index.items()
        if v["source"]["cloud"] + v["source"]["shadow"] <= MAX_BAD and v["source"]["nodata"] == 0}
print(f"{len(good)} of {len(index)} scenes have a usable source (cloud+shadow <= {MAX_BAD:.0%})", flush=True)
for k, v in sorted(good.items(), key=lambda kv: kv[1]["datetime"]):
    dst = OUT / "b08" / f"{k}.npz"
    if dst.exists():
        continue
    for attempt in range(3):
        try:
            t0 = time.time()
            arr, t = read_window(v["b08"], BOUNDS)
            np.savez_compressed(dst, b08=arr, transform=np.array(t), bounds=np.array(BOUNDS))
            print(k, arr.shape, f"{time.time() - t0:.1f}s", flush=True)
            break
        except Exception as e:
            print("retry", k, type(e).__name__, str(e)[:80], flush=True); time.sleep(3)
