import h5py, numpy as np, glob, sys
from utm import inv
base="science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HH"
files=sorted(glob.glob(sys.argv[1]+"/*.h5"), key=lambda p: p.split("_")[11])
with h5py.File(files[0]) as f: x=f[base+"/xCoordinates"][:]; y=f[base+"/yCoordinates"][:]
X,Y=np.meshgrid(x,y); LAT,LON=inv(X,Y)
src=(LAT>=28.284)&(LAT<=28.293)&(LON>=85.519)&(LON<=85.530)      # failed rock slope core (Fig 9)
bed=(LAT>=28.282)&(LAT<=28.287)&(LON>=85.529)&(LON<=85.533)      # intact bedrock (Fig 9, green)
d=np.hypot(LAT-28.288,(LON-85.524)*np.cos(np.radians(28.288)))*111.2
ring=(d>2)&(d<5)
print("pixels: source",src.sum(),"bedrock",bed.sum(),"ring",ring.sum())
lam=0.2384; cum=0
for fp in files:
    with h5py.File(fp) as f:
        u=f[base+"/unwrappedPhase"][:]; c=f[base+"/coherenceMagnitude"][:]; cc=f[base+"/connectedComponents"][:]
    p=fp.split("/")[-1].split("_"); ref,sec=p[11][:8],p[13][:8]
    dl=-u*lam/(4*np.pi)
    good=ring&(c>0.5)&(u!=0)
    refv=np.nanmedian(dl[good]) if good.sum()>20 else np.nan
    s=dl[src&(u!=0)]-refv
    ccs=np.unique(cc[src]); ccr=np.bincount(cc[good].astype(int)).argmax() if good.sum() else -1
    lo=np.nanpercentile(s,5) if s.size else np.nan; hi=np.nanpercentile(s,95) if s.size else np.nan
    print(f"{ref}-{sec} coh src={np.nanmean(c[src]):.2f} ring={np.nanmean(c[ring]):.2f} nref={good.sum():4d} "
          f"cc_src={ccs[:4]} cc_ref={ccr} src_med={np.nanmedian(s)*100:6.1f}cm p5={lo*100:6.1f} p95={hi*100:6.1f} bed_med={(np.nanmedian(dl[bed&(u!=0)])-refv)*100:6.1f}")
