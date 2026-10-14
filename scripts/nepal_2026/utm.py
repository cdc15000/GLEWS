"""WGS84 transverse Mercator (Krüger n-series), sub-mm accurate within a zone."""
import numpy as np
a=6378137.0; f=1/298.257223563; k0=0.9996; FE=500000.0
n=f/(2-f); A=a/(1+n)*(1+n**2/4+n**4/64)
al=[n/2-2*n**2/3+5*n**3/16, 13*n**2/48-3*n**3/5, 61*n**3/240]
be=[n/2-2*n**2/3+37*n**3/96, n**2/48+n**3/15, 17*n**3/480]
de=[2*n-2*n**2/3-2*n**3, 7*n**2/3-8*n**3/5, 56*n**3/15]
def fwd(lat,lon,lon0=87.0):
    phi=np.radians(lat); lam=np.radians(lon-lon0)
    e=np.sqrt(f*(2-f)); t=np.sinh(np.arctanh(np.sin(phi))-e*np.arctanh(e*np.sin(phi)))
    xi=np.arctan2(t,np.cos(lam)); eta=np.arctanh(np.sin(lam)/np.sqrt(1+t*t))
    E=eta+sum(al[j]*np.cos(2*(j+1)*xi)*np.sinh(2*(j+1)*eta) for j in range(3))
    N=xi+sum(al[j]*np.sin(2*(j+1)*xi)*np.cosh(2*(j+1)*eta) for j in range(3))
    return FE+k0*A*E, k0*A*N
def inv(x,y,lon0=87.0):
    xi=y/(k0*A); eta=(x-FE)/(k0*A)
    xp=xi-sum(be[j]*np.sin(2*(j+1)*xi)*np.cosh(2*(j+1)*eta) for j in range(3))
    ep=eta-sum(be[j]*np.cos(2*(j+1)*xi)*np.sinh(2*(j+1)*eta) for j in range(3))
    chi=np.arcsin(np.sin(xp)/np.cosh(ep))
    phi=chi+sum(de[j]*np.sin(2*(j+1)*chi) for j in range(3))
    return np.degrees(phi), lon0+np.degrees(np.arctan2(np.sinh(ep),np.cos(xp)))
if __name__=="__main__":
    x,y=fwd(28.288,85.524); print(x,y, inv(x,y))
