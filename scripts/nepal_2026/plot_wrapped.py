import sys
from pathlib import Path as _P; ROOT = str(_P(__file__).resolve().parents[2])
import glob, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from utm import inv
fs=sorted(glob.glob(ROOT+"/data/nisar/nepal_2026/gunw_t98_wrapped20m/*.npz"), key=lambda p:p.split("/")[-1].split("_")[11])
fig,ax=plt.subplots(6,7,figsize=(21,18))
for i,fp in enumerate(fs):
    d=np.load(fp); z=d["ifg"]; X,Y=np.meshgrid(d["x"],d["y"]); LA,LO=inv(X,Y)
    zs=uniform_filter(z.real,3)+1j*uniform_filter(z.imag,3)  # light multilook for display
    n=fp.split("/")[-1].split("_"); t=f"{n[11][:8]}-{n[13][4:8]}"
    ext=[LO.min(),LO.max(),LA.min(),LA.max()]
    r,c=divmod(i,7); r*=2
    a=ax[r,c]; a.imshow(np.angle(zs),cmap="hsv",extent=ext,origin="upper" if d["y"][0]>d["y"][-1] else "lower",interpolation="nearest"); a.set_title(t,fontsize=9)
    b=ax[r+1,c]; b.imshow(d["coh"],cmap="gray",vmin=0,vmax=1,extent=ext,origin="upper" if d["y"][0]>d["y"][-1] else "lower")
    for q in (a,b):
        q.plot([85.519,85.530,85.530,85.519,85.519],[28.284,28.284,28.293,28.293,28.284],"k-",lw=1)
        q.plot([85.519,85.534,85.534,85.519,85.519],[28.282,28.282,28.297,28.297,28.282],"w--",lw=.6)
        q.set_xlim(85.505,85.545); q.set_ylim(28.270,28.305); q.tick_params(labelsize=6)
plt.tight_layout(); plt.savefig(sys.argv[1] if len(sys.argv)>1 else "wrapped_grid.png",dpi=70)
