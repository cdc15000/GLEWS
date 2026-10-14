"""Re-run GLEWS Tier 0 over the Langtang Lirung source using NISAR track-98 GUNW crops."""
import sys, json, logging, yaml, numpy as np
from pathlib import Path
ROOT=str(Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT+"/src")
from glews.nisar import load_nisar_gunw_stack
from glews.timeseries import compute_acceleration_map
from glews.detect import detect_anomalies
from glews.report import generate_report
from utm import inv
logging.basicConfig(level=logging.WARNING)
cfg=yaml.safe_load(open(f"{ROOT}/config/nepal_2026.yaml"))
min_coh=float(sys.argv[1]); out=sys.argv[2]
ts=load_nisar_gunw_stack(f"{ROOT}/data/nisar/nepal_2026/gunw_t98_crop", min_coherence=min_coh, track=98)
LAT,LON=inv(ts.longitude, ts.latitude)        # loader returns UTM x/y in these fields
src=(LAT>=28.284)&(LAT<=28.293)&(LON>=85.519)&(LON<=85.530)
d=np.hypot(LAT-28.288,(LON-85.524)*np.cos(np.radians(28.288)))*111.2
ring=(d>2)&(d<5)&np.isfinite(ts.displacement[-1])
# Spatial reference: subtract stable-ring median at each date (sequential chain => equivalent to per-ifg referencing)
ts.displacement -= np.nanmedian(ts.displacement[:, ring], axis=1)[:, None, None]
valid=np.isfinite(ts.displacement[-1])
print(f"min_coh={min_coh}: {len(ts.date_strings)} dates {ts.date_strings[0]}..{ts.date_strings[-1]}; valid px {valid.sum()}/{valid.size}; source px valid {(valid&src).sum()}/{src.sum()}; ring px {ring.sum()}")
if (valid&src).any():
    cs=np.nanmedian(ts.displacement[:, valid&src], axis=1)*100
    print("  source median cumulative LOS (cm):", " ".join(f"{v:.1f}" for v in cs))
det=cfg["detect"]
am=compute_acceleration_map(ts.dates, ts.displacement, window_size_days=det["acceleration"]["window_size_days"], step_days=det["acceleration"]["step_days"], n_harmonics=det.get("n_harmonics",2))
flags=detect_anomalies(am, LAT, LON, cfg, dates=ts.dates, displacement=ts.displacement)
print(f"  {len(flags)} flags")
ev=np.datetime64("2026-08-26")
for f in sorted(flags, key=lambda f:-f.score)[:15]:
    dist=np.hypot(f.center_lat-28.288,(f.center_lon-85.524)*0.88)*111.2
    first=getattr(f,"first_detection_iso",None) or getattr(f,"onset_iso",None)
    print(f"  flag {f.flag_id:3d} score={f.score:.2f} z={f.peak_zscore:.1f} area={f.area_m2:,.0f} at {f.center_lat:.4f},{f.center_lon:.4f} ({dist:.1f} km from source) {first or ''}")
if out!="-" and flags:
    cfg["site"]["data_source"]="nisar-gunw-t98"
    print("  report:", generate_report(flags=flags, assessments=None, accel_map=am, latitude=LAT, longitude=LON, config=cfg, output_dir=out))
