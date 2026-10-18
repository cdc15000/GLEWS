"""Unwrap the 20 m NISAR GUNW wrapped-interferogram crops with SNAPHU.

Usage (from the repository root):
    python scripts/nepal_2026/unwrap_20m_snaphu.py WRAPPED_DIR OUT_DIR

Reads the .npz crops written by extract_wrapped_crops.py and writes one
HDF5 file per pair in the GUNW group layout that glews.nisar's loader
expects (unwrappedPhase, coherenceMagnitude, connectedComponents, x/y
coordinates), so run_glews.py can run the standard pipeline at 20 m.

Requires the `snaphu` package (snaphu-py, which bundles SNAPHU 2.0).

Effective looks: the product metadata gives 6 azimuth x 5 range looks for
the wrapped interferogram. Range (40 MHz bandwidth, 3.12 m spacing) and
azimuth (1263 Hz bandwidth, 1520 Hz line rate) are each oversampled ~1.2x,
so the effective number of looks is ~30 / 1.44 ~= 20.

No ionospheric or tropospheric correction is applied to the 20 m phase.
"""
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import snaphu

NLOOKS = 20.0
COST = "defo"   # deformation cost model: suited to localized ground motion
BASE = "science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HH"


def main(wrapped_dir: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(wrapped_dir).glob("*.npz"), key=lambda p: p.name.split("_")[11])
    for fp in files:
        dst = out / (fp.stem + ".h5")
        if dst.exists():
            continue
        d = np.load(fp)
        igram = d["ifg"].astype(np.complex64)
        corr = np.clip(d["coh"].astype(np.float32), 0, 1)
        valid = np.isfinite(igram) & (igram != 0) & (d["mask"] != 0) & (d["mask"] != 255)
        t = time.time()
        unw, cc = snaphu.unwrap(igram, corr, nlooks=NLOOKS, cost=COST, init="mcf", mask=valid)
        tmp = dst.with_suffix(".part")
        with h5py.File(tmp, "w") as o:
            g = o.create_group(BASE)
            g.create_dataset("unwrappedPhase", data=unw.astype(np.float32), compression="gzip")
            g.create_dataset("coherenceMagnitude", data=corr, compression="gzip")
            g.create_dataset("connectedComponents", data=cc.astype(np.uint16), compression="gzip")
            g["xCoordinates"] = d["x"]
            g["yCoordinates"] = d["y"]
            o.attrs["unwrapper"] = f"SNAPHU via snaphu-py {snaphu.__version__}"
            o.attrs["cost"] = COST
            o.attrs["nlooks"] = NLOOKS
            o.attrs["source_crop"] = fp.name
        tmp.rename(dst)
        n = fp.name.split("_")
        frac0 = float(np.mean(cc == 0))
        print(f"{n[11][:8]}-{n[13][:8]} components={int(cc.max())} "
              f"unassigned={frac0:.0%} {time.time() - t:.1f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
