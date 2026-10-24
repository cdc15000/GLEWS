"""Crop NISAR track-98 GOFF (pixel offset) products around the source via HTTP range reads.

Usage (from the repository root):
    python scripts/nepal_2026/extract_goff_crops.py OUT_DIR [LAT LON]

LAT LON default to the Gyirong–Rasuwa source; pass another centre to crop a
control site from the same products (Phase 2).

Same pair selection as extract_gunw_crops.py: pre-event pairs only (the
pair spanning the 26 Aug collapse is dropped), one product per date pair.
Writes small HDF5 files with the GOFF group layout, all three offset
layers (32/64/128-pixel matching windows), so glews.nisar can read them.
"""
import sys, time, asf_search as asf, requests, fsspec, h5py, numpy as np
from pathlib import Path
from utm import fwd

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
LAT, LON = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (28.288, 85.524)
HALF = 6000.0
cx, cy = fwd(LAT, LON)
FIELDS = ("slantRangeOffset", "alongTrackOffset", "slantRangeOffsetVariance",
          "alongTrackOffsetVariance", "correlationSurfacePeak", "snr")
res = asf.search(platform=asf.PLATFORM.NISAR, intersectsWith=f"POINT({LON} {LAT})",
                 start="2025-10-01", end="2026-08-27", processingLevel="GOFF")
pairs = {}
for x in res:
    p = x.properties
    if p["pathNumber"] != 98:
        continue
    d = [t[:8] for t in p["fileID"].split("_") if len(t) >= 15 and t[0] == "2" and "T" in t]
    if d[2] > "20260826":
        continue
    if (d[0], d[2]) not in pairs or p["fileID"] > pairs[(d[0], d[2])]["fileID"]:
        pairs[(d[0], d[2])] = p
print(len(pairs), "unique pre-event pairs")
s = requests.Session(); fs = fsspec.filesystem("http")
base = "science/LSAR/GOFF/grids/frequencyA/pixelOffsets"
for (ref, sec), p in sorted(pairs.items()):
    dst = OUT / (p["fileID"] + ".h5")
    if dst.exists():
        continue
    for attempt in range(3):
        try:
            t = time.time()
            r = s.get(p["url"], stream=True); final = r.url; r.close()
            with fs.open(final, block_size=2**20, cache_type="blockcache") as fo, h5py.File(fo, "r") as f:
                pol = [k for k in f[base] if len(k) == 2][0]
                x = f[f"{base}/xCoordinates"][:] if f"{base}/xCoordinates" in f else f[f"{base}/{pol}/layer1/xCoordinates"][:]
                y = f[f"{base}/yCoordinates"][:] if f"{base}/yCoordinates" in f else f[f"{base}/{pol}/layer1/yCoordinates"][:]
                ci = np.where(abs(x - cx) <= HALF)[0]; ri = np.where(abs(y - cy) <= HALF)[0]
                sl = (slice(ri[0], ri[-1] + 1), slice(ci[0], ci[-1] + 1))
                layers = {lay: {k: f[f"{base}/{pol}/{lay}/{k}"][sl] for k in FIELDS}
                          for lay in ("layer1", "layer2", "layer3")}
            tmp = dst.with_suffix(".part")
            with h5py.File(tmp, "w") as o:
                for lay, arrs in layers.items():
                    g = o.create_group(f"{base}/{pol}/{lay}")
                    for k, v in arrs.items():
                        g.create_dataset(k, data=v, compression="gzip")
                    g["xCoordinates"] = x[ci]; g["yCoordinates"] = y[ri]
                o.attrs["source_granule"] = p["fileID"]
            tmp.rename(dst)
            print(ref, sec, layers["layer1"]["snr"].shape, f"{time.time() - t:.0f}s", flush=True)
            break
        except Exception as e:
            print("retry", ref, sec, type(e).__name__, str(e)[:80], flush=True)
