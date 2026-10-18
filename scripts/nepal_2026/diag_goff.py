"""Per-pair GOFF offsets over the failed slope, referenced to stable ground.

Usage: python scripts/nepal_2026/diag_goff.py GOFF_CROP_DIR [LAYER] [MIN_SNR]
"""
import sys, glob, h5py, yaml, numpy as np
from pathlib import Path
from matplotlib.path import Path as MP
from utm import inv
ROOT = Path(__file__).resolve().parents[2]
d_in = sys.argv[1]; layer = sys.argv[2] if len(sys.argv) > 2 else "layer2"
min_snr = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
B = f"science/LSAR/GOFF/grids/frequencyA/pixelOffsets/HH/{layer}"
fs = sorted(glob.glob(d_in + "/*.h5"), key=lambda p: Path(p).name.split("_")[11])
with h5py.File(fs[0]) as f:
    X, Y = np.meshgrid(f[B + "/xCoordinates"][:], f[B + "/yCoordinates"][:])
LA, LO = inv(X, Y)
poly = np.array(yaml.safe_load(open(ROOT / "config/nepal_2026.yaml"))["site"]["known_source"]["polygon"])
src = MP(poly[:, ::-1]).contains_points(np.c_[LO.ravel(), LA.ravel()]).reshape(LA.shape)
dist = np.hypot(LA - 28.288, (LO - 85.524) * 0.88) * 111.2
ring = (dist > 2) & (dist < 5)
print(f"{layer}, SNR>={min_snr}: source px {src.sum()}, ring px {ring.sum()}")
print("pair                days  src px ok   range m (m/yr)     along m (m/yr)    |v| m/yr   ring MAD range/along m")
for fp in fs:
    with h5py.File(fp) as f:
        rg = f[B + "/slantRangeOffset"][:]; az = f[B + "/alongTrackOffset"][:]; snr = f[B + "/snr"][:]
    n = Path(fp).name.split("_"); r0, s0 = n[11][:8], n[13][:8]
    days = (np.datetime64(f"{s0[:4]}-{s0[4:6]}-{s0[6:]}") - np.datetime64(f"{r0[:4]}-{r0[4:6]}-{r0[6:]}")).astype(int)
    ok = (snr >= min_snr) & np.isfinite(rg) & np.isfinite(az) & (rg != 0)
    rr, ra = np.median(rg[ring & ok]), np.median(az[ring & ok])
    mad_r = np.median(abs(rg[ring & ok] - rr)); mad_a = np.median(abs(az[ring & ok] - ra))
    m = src & ok
    if m.sum() < 5:
        print(f"{r0}-{s0}  {days:3d}  {m.sum():3d}/{src.sum()}  too few good pixels"); continue
    dr, da = np.median(rg[m]) - rr, np.median(az[m]) - ra
    k = 365.25 / days
    print(f"{r0}-{s0}  {days:3d}  {m.sum():3d}/{src.sum()}   {dr:6.2f} ({dr*k:6.1f})    {da:6.2f} ({da*k:6.1f})    {np.hypot(dr,da)*k:6.1f}     {mad_r:.2f}/{mad_a:.2f}")
