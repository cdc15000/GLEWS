"""Re-run GLEWS Tier 0 over the Langtang Lirung source using NISAR track-98 GUNW crops."""
import sys, json, logging, yaml, numpy as np
from pathlib import Path
ROOT=str(Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT+"/src")
from glews.nisar import load_nisar_gunw_stack
from glews.timeseries import compute_acceleration_map
from glews.detect import detect_anomalies
from glews.report import generate_report
from glews.quality import assess_flag_quality, render_interferogram_strips
import glob, h5py
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

def iso(d): return f"{d[:4]}-{d[4:6]}-{d[6:8]}"

# Per-pair coherence and unwrapping reliability on the same grid (files sorted as the loader sorts them)
GUNW=f"{ROOT}/data/nisar/nepal_2026/gunw_t98_crop"; B="science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HH"
coh_s, unrel_s, phase80, pair_dates = [], [], [], []
for fp in sorted(Path(GUNW).glob("NISAR_*GUNW*.h5")):
    with h5py.File(fp) as h:
        coh_s.append(h[B+"/coherenceMagnitude"][:]); unrel_s.append(h[B+"/connectedComponents"][:] == 0)
        phase80.append(np.angle(np.exp(1j*h[B+"/unwrappedPhase"][:])))
    parts=[t[:8] for t in fp.stem.split("_") if len(t)>=15 and t[0]=="2" and "T" in t]
    pair_dates.append((iso(parts[0]), iso(parts[2])))
assess_flag_quality(flags, np.array(coh_s), np.array(unrel_s), pair_dates)
grades={}
for f in flags: grades[f.detection_details["quality"]["reliability"]]=grades.get(f.detection_details["quality"]["reliability"],0)+1
print("  reliability:", grades)

if out!="-" and flags:
    # Wrapped-phase strips: 20 m crops near the source, 80 m re-wrapped phase elsewhere
    tiles20=[]
    for fp in sorted(glob.glob(f"{ROOT}/data/nisar/nepal_2026/gunw_t98_wrapped20m/*.npz"), key=lambda p: Path(p).name.split("_")[11]):
        z=np.load(fp); X,Y=np.meshgrid(z["x"],z["y"]); la,lo=inv(X,Y); n=Path(fp).name.split("_")
        tiles20.append(dict(ref=iso(n[11]), sec=iso(n[13]), phase=np.angle(z["ifg"]), lat=la, lon=lo))
    tiles80=[dict(ref=r, sec=s_, phase=ph, lat=LAT, lon=LON) for (r,s_),ph in zip(pair_dates, phase80)]
    near=[f for f in flags if tiles20 and np.hypot(f.center_lat-28.288,(f.center_lon-85.524)*0.88)*111.2 < 1.8]
    far=[f for f in flags if f not in near]
    n_strips=render_interferogram_strips(near, tiles20, out)+render_interferogram_strips(far, tiles80, out)
    print(f"  interferogram strips: {n_strips} ({len(near)} at 20 m)")
    cfg["site"]["data_source"]="nisar-gunw-t98"
    cfg["site"]["provenance"]={
        "sensor": "NISAR", "product": "L2 GUNW", "band": "L",
        "track": 98, "direction": "ascending", "frame": 16,
        "date_start": pair_dates[0][0], "date_end": pair_dates[-1][1],
        "n_pairs": len(pair_dates),
        "excluded": ["19 Aug-31 Aug pair (spans the collapse)", "duplicate product listings",
                     "track 48 (no Jan-Jun acquisitions)"],
        "coherence_mask": min_coh,
        "reference": "median of a stable ring 2-5 km from the source, per date",
        "posting_m": 80,
        "extent": [float(LAT.min()), float(LON.min()), float(LAT.max()), float(LON.max())],
        "script": "scripts/nepal_2026/run_glews.py",
    }
    print("  report:", generate_report(flags=flags, assessments=None, accel_map=am, latitude=LAT, longitude=LON, config=cfg, output_dir=out))
