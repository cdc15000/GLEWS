import sys, time, glob, asf_search as asf, requests, fsspec, h5py, numpy as np
from pathlib import Path
from utm import fwd
OUT=Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
cx,cy=fwd(28.288,85.524); HALF=3000.0
names={Path(p).stem for p in glob.glob(sys.argv[2]+"/*.h5")}
res=[x for x in asf.search(platform=asf.PLATFORM.NISAR, intersectsWith="POINT(85.524 28.288)", start="2025-10-01", end="2026-08-27", processingLevel="GUNW") if x.properties["fileID"] in names]
s=requests.Session(); fs=fsspec.filesystem("http"); W="science/LSAR/GUNW/grids/frequencyA/wrappedInterferogram"
for x in sorted(res,key=lambda r:r.properties["fileID"].split("_")[11]):
    p=x.properties; dst=OUT/(p["fileID"]+".npz")
    if dst.exists(): continue
    for attempt in range(3):
        try:
            t=time.time(); q=s.get(p["url"],stream=True); u=q.url; q.close()
            with fs.open(u, block_size=2**20, cache_type="blockcache") as fo, h5py.File(fo) as f:
                xs=f[W+"/xCoordinates"][:] if W+"/xCoordinates" in f else f[W+"/HH/xCoordinates"][:]
                ys=f[W+"/yCoordinates"][:] if W+"/yCoordinates" in f else f[W+"/HH/yCoordinates"][:]
                ci=np.where(abs(xs-cx)<=HALF)[0]; ri=np.where(abs(ys-cy)<=HALF)[0]
                sl=(slice(ri[0],ri[-1]+1),slice(ci[0],ci[-1]+1))
                ifg=f[W+"/HH/wrappedInterferogram"][sl]; coh=f[W+"/HH/coherenceMagnitude"][sl]; m=f[W+"/mask"][sl]
            np.savez_compressed(dst, ifg=ifg, coh=coh, mask=m, x=xs[ci], y=ys[ri])
            print(p["fileID"].split("_")[11][:8], p["fileID"].split("_")[13][:8], ifg.shape, "%.0fs"%(time.time()-t), flush=True); break
        except Exception as e: print("retry", type(e).__name__, str(e)[:80], flush=True)
