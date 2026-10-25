"""
WGS84 transverse Mercator (Krüger n-series), sub-mm accurate within a zone.

The same maths as ``scripts/nepal_2026/utm.py``, which placed the 2026
crops, rings and polygons, so detector results reproduce exactly. The
central meridian is a parameter; ``utm_lon0`` gives it for an EPSG:326zz
(UTM north) code.
"""
from __future__ import annotations

import numpy as np

_a = 6378137.0
_f = 1 / 298.257223563
_k0 = 0.9996
_FE = 500000.0
_n = _f / (2 - _f)
_A = _a / (1 + _n) * (1 + _n ** 2 / 4 + _n ** 4 / 64)
_al = [_n / 2 - 2 * _n ** 2 / 3 + 5 * _n ** 3 / 16, 13 * _n ** 2 / 48 - 3 * _n ** 3 / 5, 61 * _n ** 3 / 240]
_be = [_n / 2 - 2 * _n ** 2 / 3 + 37 * _n ** 3 / 96, _n ** 2 / 48 + _n ** 3 / 15, 17 * _n ** 3 / 480]
_de = [2 * _n - 2 * _n ** 2 / 3 - 2 * _n ** 3, 7 * _n ** 2 / 3 - 8 * _n ** 3 / 5, 56 * _n ** 3 / 15]


def utm_lon0(crs: str) -> float:
    """Central meridian of a UTM-north CRS such as "EPSG:32645" (zone 45 -> 87.0)."""
    code = int(str(crs).upper().replace("EPSG:", ""))
    if not 32601 <= code <= 32660:
        raise ValueError(f"not a UTM-north EPSG code: {crs}")
    return -183.0 + 6.0 * (code - 32600)


def utm_crs(lon: float) -> str:
    """UTM-north EPSG code of the zone containing ``lon``."""
    return f"EPSG:{32600 + int((lon + 180) // 6) + 1}"


def fwd(lat, lon, lon0: float = 87.0):
    """(lat, lon) in degrees -> (easting, northing) in metres."""
    phi = np.radians(lat)
    lam = np.radians(np.asarray(lon) - lon0)
    e = np.sqrt(_f * (2 - _f))
    t = np.sinh(np.arctanh(np.sin(phi)) - e * np.arctanh(e * np.sin(phi)))
    xi = np.arctan2(t, np.cos(lam))
    eta = np.arctanh(np.sin(lam) / np.sqrt(1 + t * t))
    E = eta + sum(_al[j] * np.cos(2 * (j + 1) * xi) * np.sinh(2 * (j + 1) * eta) for j in range(3))
    N = xi + sum(_al[j] * np.sin(2 * (j + 1) * xi) * np.cosh(2 * (j + 1) * eta) for j in range(3))
    return _FE + _k0 * _A * E, _k0 * _A * N


def inv(x, y, lon0: float = 87.0):
    """(easting, northing) in metres -> (lat, lon) in degrees."""
    xi = np.asarray(y) / (_k0 * _A)
    eta = (np.asarray(x) - _FE) / (_k0 * _A)
    xp = xi - sum(_be[j] * np.sin(2 * (j + 1) * xi) * np.cosh(2 * (j + 1) * eta) for j in range(3))
    ep = eta - sum(_be[j] * np.cos(2 * (j + 1) * xi) * np.sinh(2 * (j + 1) * eta) for j in range(3))
    chi = np.arcsin(np.sin(xp) / np.cosh(ep))
    phi = chi + sum(_de[j] * np.sin(2 * (j + 1) * chi) for j in range(3))
    return np.degrees(phi), lon0 + np.degrees(np.arctan2(np.sinh(ep), np.cos(xp)))
