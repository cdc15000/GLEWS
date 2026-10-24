"""
Cross-sensor geometry for comparing radar and optical offsets.

Radar offset tracking (NISAR GOFF) measures motion along the slant range
(line of sight) and along-track directions; optical offset tracking
(Sentinel-2) measures east and north motion and is blind to vertical
motion. To compare them, either project optical motion into radar
geometry, or invert the two radar components into east/north/up motion
under the assumption that the ground moves parallel to the local slope.

Conventions:
- ``los`` is the east-north-up unit vector from the target to the sensor
  (as in NISAR ``losUnitVectorX/Y``; up = sqrt(1 - x^2 - y^2)).
- A positive slant-range offset is an increase in range, i.e. motion away
  from the sensor: range = -(d · los).
- ``along_track`` is the east-north unit vector of the platform heading
  (NISAR ``alongTrackUnitVectorX/Y``); along-track offset = d · along_track.
- ``grad`` = (dz/dE, dz/dN) of the terrain; slope-parallel motion has
  up = grad · (east, north).
"""

from __future__ import annotations

import numpy as np


def los_enu(los_x: float, los_y: float) -> np.ndarray:
    """Complete a target-to-sensor LOS unit vector with its up component."""
    up = np.sqrt(max(0.0, 1.0 - los_x ** 2 - los_y ** 2))
    return np.array([los_x, los_y, up])


def project_to_radar(east, north, up, los: np.ndarray, along_track: tuple[float, float]):
    """Slant-range and along-track offsets that motion (east, north, up) would produce."""
    d = np.stack(np.broadcast_arrays(east, north, up), axis=-1)
    rng = -(d @ los)
    at = np.asarray(east) * along_track[0] + np.asarray(north) * along_track[1]
    return rng, at


def slope_parallel_from_radar(range_offset, along_track_offset, los: np.ndarray,
                              along_track: tuple[float, float], grad: tuple[float, float]):
    """
    Invert slant-range and along-track offsets to (east, north, up) motion,
    assuming motion parallel to a plane with gradient ``grad``.

    Solves  range = -(e*lx + n*ly + u*lz),  along = e*ax + n*ay,
            u = gx*e + gy*n.
    """
    gx, gy = grad
    lx, ly, lz = los
    ax, ay = along_track
    A = np.array([[-(lx + lz * gx), -(ly + lz * gy)],
                  [ax, ay]])
    b = np.stack(np.broadcast_arrays(range_offset, along_track_offset), axis=-1)
    en = np.linalg.solve(A, b[..., None])[..., 0] if b.ndim > 1 else np.linalg.solve(A, b)
    e, n = en[..., 0], en[..., 1]
    return e, n, gx * e + gy * n


def agreement(a: float, sigma_a: float, b: float, sigma_b: float, k: float = 2.0) -> dict:
    """Whether two estimates agree in sign and within ``k`` combined sigma."""
    comb = float(np.hypot(sigma_a, sigma_b))
    diff = float(a - b)
    return {"difference": round(diff, 3), "combined_sigma": round(comb, 3),
            "n_sigma": round(abs(diff) / comb, 2) if comb > 0 else None,
            "same_sign": bool(np.sign(a) == np.sign(b)),
            "agree": bool(np.sign(a) == np.sign(b) and abs(diff) <= k * comb)}
