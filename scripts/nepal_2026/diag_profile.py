import sys
from pathlib import Path as _P; ROOT = str(_P(__file__).resolve().parents[2])
import glob, json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from utm import fwd, inv
k=0.2384/(4*np.pi)
fs=sorted(glob.glob(ROOT+"/data/nisar/nepal_2026/gunw_t98_wrapped20m/*.npz"), key=lambda p:p.split("/")[-1].split("_")[11])
# Profile along inferred collapse direction (~306° flow / NW) through the deformation core
A=fwd(28.2945,85.5145); B=fwd(28.2815,85.5335)   # ~2.3 km, NW -> SE
L=np.hypot(B[0]-A[0],B[1]-A[1]); s=np.arange(0,L,10.0)
ux,uy=(B[0]-A[0])/L,(B[1]-A[1])/L; px,py=-uy,ux
out=[]; fig,ax=plt.subplots(1,2,figsize=(14,5))
for fp in fs:
    d=np.load(fp); z=d["ifg"]; x=d["x"]; y=d["y"]; dx=x[1]-x[0]; dy=y[1]-y[0]
    acc=0
    for w in np.arange(-60,61,20.0):           # 120 m wide swath, complex-averaged
        X=A[0]+ux*s+px*w; Y=A[1]+uy*s+py*w
        c=(X-x[0])/dx; r=(Y-y[0])/dy
        acc=acc+map_coordinates(z.real,[r,c],order=1)+1j*map_coordinates(z.imag,[r,c],order=1)
    q=abs(acc)/7/np.median(abs(z))
    ph=np.unwrap(np.angle(acc)); los=-ph*k*100
    la,lo=inv(A[0]+ux*s,A[1]+uy*s); core=(la>=28.284)&(la<=28.293)&(lo>=85.519)&(lo<=85.530)
    ends=np.r_[los[:20],los[-20:]].mean() if True else 0
    rel=los-ends
    n=fp.split("/")[-1].split("_"); t=f"{n[11][:8]}-{n[13][:8]}"
    peak=rel[core][np.argmax(abs(rel[core]))]
    out.append(dict(pair=t, peak_rel_cm=float(peak), range_core_cm=float(np.ptp(los[core]))))
    if n[11][:8]>="20260608": ax[0].plot(s,rel,label=t)
    print(f"{t}: core peak rel LOS {peak:6.1f} cm, core range {np.ptp(los[core]):5.1f} cm")
ax[0].axvspan(s[core][0],s[core][-1],color="0.9"); ax[0].legend(fontsize=7); ax[0].set_xlabel("m along NW→SE profile"); ax[0].set_ylabel("LOS cm rel. ends")
d=np.load(fs[-1]); X,Y=np.meshgrid(d["x"],d["y"]); LA,LO=inv(X,Y)
ax[1].imshow(np.angle(d["ifg"]),cmap="twilight_shifted",extent=[LO.min(),LO.max(),LA.min(),LA.max()],origin="upper" if d["y"][0]>d["y"][-1] else "lower")
la,lo=inv(A[0]+ux*s,A[1]+uy*s); ax[1].plot(lo,la,"k-"); ax[1].set_xlim(85.508,85.540); ax[1].set_ylim(28.278,28.300)
plt.tight_layout(); plt.savefig(sys.argv[1] if len(sys.argv)>1 else "profile.png",dpi=75); json.dump(out,open("profile.json","w"),indent=1)
