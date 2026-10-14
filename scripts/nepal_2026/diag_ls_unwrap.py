import sys
from pathlib import Path as _P; ROOT = str(_P(__file__).resolve().parents[2])
import glob, json, numpy as np
from scipy.fft import dctn, idctn
from scipy.ndimage import uniform_filter
from utm import inv
lam=0.2384; k=lam/(4*np.pi)
def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi
def ls_unwrap(psi):
    # Ghiglia & Romero (1994) unweighted least-squares unwrapping via DCT
    dx=wrap(np.diff(psi,axis=1)); dy=wrap(np.diff(psi,axis=0))
    rho=np.zeros_like(psi)
    rho[:,:-1]+=dx; rho[:,1:]-=dx; rho[:-1,:]+=dy; rho[1:,:]-=dy
    M,N=psi.shape; i=np.arange(M)[:,None]; j=np.arange(N)[None,:]
    den=2*(np.cos(np.pi*i/M)+np.cos(np.pi*j/N)-2); den[0,0]=1
    phi=idctn(dctn(-rho,norm="ortho")/den,norm="ortho"); phi[0,0]=0
    return phi
fs=sorted(glob.glob(ROOT+"/data/nisar/nepal_2026/gunw_t98_wrapped20m/*.npz"), key=lambda p:p.split("/")[-1].split("_")[11])
rows=[]; cum=0.0
for fp in fs:
    d=np.load(fp); z=d["ifg"]; X,Y=np.meshgrid(d["x"],d["y"]); LA,LO=inv(X,Y)
    z=uniform_filter(z.real,3)+1j*uniform_filter(z.imag,3)
    n=fp.split("/")[-1].split("_"); ref,sec=n[11][:8],n[13][:8]
    src=(LA>=28.284)&(LA<=28.293)&(LO>=85.519)&(LO<=85.530)
    dist=np.hypot(LA-28.288,(LO-85.524)*0.88)*111.2; ring=(dist>1.5)&(dist<2.8)
    zn=z/np.maximum(abs(z),1e-9); qual=abs(uniform_filter(zn.real,5)+1j*uniform_filter(zn.imag,5))
    # wrapped: coherent phase of high-quality source pixels relative to ring
    sq=src&(qual>0.5); rq=ring&(qual>0.5)
    dphi=np.angle(np.sum(zn[sq])*np.conj(np.sum(zn[rq]))) if sq.sum()>10 else np.nan
    u=ls_unwrap(np.angle(z))*-k
    s=np.nanmedian(u[sq])-np.nanmedian(u[rq]) if sq.sum()>10 else np.nan
    # spread: source p5 relative to ring (largest motion)
    p=np.nanpercentile(u[sq],5)-np.nanmedian(u[rq]) if sq.sum()>10 else np.nan
    cum+= s if np.isfinite(s) else 0
    days=(np.datetime64(f"{sec[:4]}-{sec[4:6]}-{sec[6:]}")-np.datetime64(f"{ref[:4]}-{ref[4:6]}-{ref[6:]}")).astype(int)
    rows.append(dict(ref=ref,sec=sec,days=int(days),src_good_px=int(sq.sum()),ring_good_px=int(rq.sum()),wrapped_dphi_cm=float(-dphi*k*100),ls_median_cm=float(s*100),ls_p5_cm=float(p*100),cum_cm=float(cum*100)))
    print(f"{ref}-{sec} {days:2d}d good src px {sq.sum():4d}/{src.sum()} wrapped Δ={-dphi*k*100:6.1f} cm  LS median={s*100:6.1f}  LS p5={p*100:6.1f}  cum={cum*100:6.1f}")
json.dump(rows,open("hires_series.json","w"),indent=1)
