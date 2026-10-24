"""Fetch matched pre- and post-event Sentinel-2 scenes to map the failure scar.

Usage (from the repository root):
    python scripts/nepal_2026/fetch_s2_scar.py [OUT_DIR]

Reads 10 m bands B02, B03, B04, B08 and 20 m B11 and SCL over the same
11 x 11 km crop as fetch_s2_crops.py, for one pre-event and one
post-event relative-orbit-119 scene in the same season (21 Oct 2025 and
21 Oct 2026, near-identical sun elevation), into OUT_DIR/scar/<date>.npz.
"""
import sys, yaml, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).parent))
from glews.optical import search_sentinel2, read_window
from utm import fwd

OUT = (Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/sentinel2/nepal_2026") / "scar"
OUT.mkdir(parents=True, exist_ok=True)
site = yaml.safe_load(open(ROOT / "config/nepal_2026.yaml"))["site"]
cx, cy = fwd(site["latitude"], site["longitude"])
cx, cy = round(cx / 20) * 20, round(cy / 20) * 20
BOUNDS = (cx - 5500, cy - 5500, cx + 5500, cy + 5500)
BANDS = {"blue": "B02", "green": "B03", "red": "B04", "nir": "B08", "swir16": "B11", "scl": "SCL"}
for day in ("2025-10-21", "2026-10-21"):
    sc = [s for s in search_sentinel2(site["latitude"], site["longitude"], day, day, max_cloud=100,
                                      relative_orbit=119)]
    if not sc:
        print("no scene", day); continue
    s = sc[0]
    arrs = {}
    for asset, name in BANDS.items():
        arrs[name], t = read_window(s.hrefs[asset], BOUNDS)
        arrs[name + "_transform"] = np.array(t)
    np.savez_compressed(OUT / f"{day}.npz", scene=s.id, sun_elevation=s.sun_elevation, **arrs)
    print(day, s.id, f"sun {s.sun_elevation:.1f}", {k: v.shape for k, v in arrs.items() if not k.endswith("transform")})
