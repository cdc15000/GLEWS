"""Crop NISAR track-98 GUNW products around the Langtang Lirung source via HTTP range reads.
Writes small HDF5 files with the GUNW group layout GLEWS's nisar.py expects."""
import sys, time, asf_search as asf, requests, fsspec, h5py, numpy as np
from pathlib import Path
from utm import fwd
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
LAT, LON, HALF = 28.288, 85.524, 6000.0   # 12 km box
cx, cy = fwd(LAT, LON)
res = asf.search(platform=asf.PLATFORM.NISAR, intersectsWith=f"POINT({LON} {LAT})",
                 start="2025-10-01", end="2026-08-27", processingLevel="GUNW")
pairs = {}
for x in res:
    p = x.properties
    if p["pathNumber"] != 98: continue
    d = [t[:8] for t in p["fileID"].split("_") if len(t) >= 15 and t[0] == "2" and "T" in t]
    ref, sec = d[0], d[2]
    if sec > "20260826": print("skip co-event", ref, sec); continue
    key = (ref, sec)
    if key not in pairs or p["fileID"] > pairs[key]["fileID"]: pairs[key] = p
print(len(pairs), "unique pre-event pairs")
s = requests.Session(); fs = fsspec.filesystem("http")
base = "science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram"
for (ref, sec), p in sorted(pairs.items()):
    dst = OUT / (p["fileID"] + ".h5")
    if dst.exists(): continue
    for attempt in range(3):
        try:
            t = time.time()
            r = s.get(p["url"], stream=True); final = r.url; r.close()
            with fs.open(final, block_size=2**20, cache_type="blockcache") as fo, h5py.File(fo, "r") as f:
                pol = [k for k in f[base] if len(k) == 2][0]; g = f[f"{base}/{pol}"]
                x = g["xCoordinates"][:]; y = g["yCoordinates"][:]
                ci = np.where((x >= cx-HALF) & (x <= cx+HALF))[0]; ri = np.where((y >= cy-HALF) & (y <= cy+HALF))[0]
                sl = (slice(ri[0], ri[-1]+1), slice(ci[0], ci[-1]+1))
                layers = {k: g[k][sl] for k in ("unwrappedPhase", "coherenceMagnitude", "connectedComponents", "ionospherePhaseScreen")}
                epsg = int(g["projection"].attrs["epsg_code"])
            tmp = dst.with_suffix(".part")
            with h5py.File(tmp, "w") as o:
                og = o.create_group(f"{base}/{pol}")
                for k, v in layers.items(): og.create_dataset(k, data=v, compression="gzip")
                og["xCoordinates"] = x[ci]; og["yCoordinates"] = y[ri]
                o.attrs["epsg_code"] = epsg; o.attrs["source_granule"] = p["fileID"]
                o.attrs["note"] = "Crop of NISAR GUNW around Langtang Lirung source (GLEWS re-analysis)"
            tmp.rename(dst)
            print(ref, sec, layers["unwrappedPhase"].shape, "coh=%.2f" % np.nanmean(layers["coherenceMagnitude"]), "%.0fs" % (time.time()-t), flush=True)
            break
        except Exception as e:
            print("retry", ref, sec, type(e).__name__, str(e)[:80], flush=True)
