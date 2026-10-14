from pathlib import Path as _P; ROOT = str(_P(__file__).resolve().parents[2])
import sys, glob, h5py, numpy as np, yaml, logging, collections
sys.path.insert(0,ROOT+"/src"); logging.disable(logging.WARNING)
import warnings; warnings.filterwarnings("ignore")
from glews.nisar import load_nisar_gunw_stack
from glews.timeseries import compute_acceleration_map
from glews.detect import detect_anomalies
from datetime import date
from utm import inv
R=ROOT; cfg=yaml.safe_load(open(R+"/config/nepal_2026.yaml"))
ts=load_nisar_gunw_stack(R+"/data/nisar/nepal_2026/gunw_t98_crop",min_coherence=0.3,track=98)
LAT,LON=inv(ts.longitude,ts.latitude)
d=np.hypot(LAT-28.288,(LON-85.524)*0.88)*111.2; ring=(d>2)&(d<5)&np.isfinite(ts.displacement[-1])
ts.displacement-=np.nanmedian(ts.displacement[:,ring],axis=1)[:,None,None]
det=cfg["detect"]; am=compute_acceleration_map(ts.dates,ts.displacement,window_size_days=60,step_days=12,n_harmonics=2)
flags=sorted(detect_anomalies(am,LAT,LON,cfg,dates=ts.dates,displacement=ts.displacement),key=lambda f:-f.score)
# connected components: fraction of pairs where pixel cc==0
cc0=np.zeros(LAT.shape)
for fp in glob.glob(R+"/data/nisar/nepal_2026/gunw_t98_crop/*.h5"):
    with h5py.File(fp) as f: cc0+=(f["science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HH/connectedComponents"][:]==0)
cc0/=21
inpoly=lambda la,lo: 28.282<=la<=28.297 and 85.518<=lo<=85.533
months=collections.Counter()
fl_cc=[]
for rank,f in enumerate(flags,1):
    dist=np.hypot(f.center_lat-28.288,(f.center_lon-85.524)*0.88)*111.2
    pi=f.pixel_indices; frac=np.mean(cc0[pi[:,0],pi[:,1]]>0.5); fl_cc.append(frac)
    wd=date.fromordinal(int(f.window_date)); months[wd.strftime("%Y-%m")]+=1
    pts=[(LAT[r,c],LON[r,c]) for r,c in pi]
    if dist<1.0 or any(inpoly(*p) for p in pts):
        print(f"rank {rank}/{len(flags)} flag {f.flag_id} score {f.score:.2f} at {f.center_lat:.4f},{f.center_lon:.4f} dist {dist:.2f} km, px in Fig9 box {sum(inpoly(*p) for p in pts)}/{len(pts)}, window {wd}, cc0-majority px {frac:.0%}")
print("flag window_date by month:", dict(sorted(months.items())))
print("flags whose pixels are mostly cc=0 in >50% of pairs: median frac {:.0f}%".format(100*np.median(fl_cc)))
print("pixels cc0 in >50% pairs: crop {:.0f}%, Fig9 box {:.0f}%".format(100*np.mean(cc0>0.5),100*np.mean(cc0[(LAT>=28.284)&(LAT<=28.293)&(LON>=85.519)&(LON<=85.530)]>0.5)))
