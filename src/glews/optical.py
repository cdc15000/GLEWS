"""
Optical pixel-offset tracking (Sentinel-2) for metre-scale slope motion.

Radar phase saturates over fast-moving or decorrelated slopes (a few cm per
pair at L-band). Offset tracking between optical images measures horizontal
motion of metres to hundreds of metres, which is how the 2026 Langtang
Lirung precursor was found (Wang et al. 2026, COSI-Corr on Sentinel-2 and
Landsat). This module provides the pieces GLEWS needs to do the same:

- ``search_sentinel2``: query the public Earth Search STAC catalogue
  (Sentinel-2 L2A, collection 1, improved geolocation) for one point.
- ``read_window``: read a small window of a cloud-optimised GeoTIFF over
  HTTP, without downloading the scene.
- ``scl_fractions``: cloud / shadow / snow fractions from the L2A scene
  classification over a mask.
- ``offset_field``: dense offsets between two co-registered images by
  frequency-domain cross-correlation with upsampled-DFT sub-pixel
  refinement (Guizar-Sicairos et al. 2008), in the same family as
  COSI-Corr's frequency correlator.

Pair only images from the same relative orbit: with kilometres of relief,
orthorectification parallax between orbits is larger than the signal.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

logger = logging.getLogger(__name__)

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1/search"
COLLECTION = "sentinel-2-c1-l2a"

# Sentinel-2 L2A scene classification (SCL) classes
SCL_NODATA = (0,)
SCL_CLOUD = (8, 9, 10)          # medium, high probability, thin cirrus
SCL_SHADOW = (2, 3)             # dark area / topographic shadow, cloud shadow
SCL_DARK = (2,)                 # "dark area pixels": also catches dark rock and debris
SCL_CLOUD_SHADOW = (3,)
SCL_SNOW = (11,)

_ORBIT_RE = re.compile(r"_R(\d{3})_")

# GDAL settings for windowed COG reads over HTTP
GDAL_HTTP_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "AWS_NO_SIGN_REQUEST": "YES",
}


@dataclass
class Scene:
    """One Sentinel-2 L2A scene from the STAC catalogue."""
    id: str
    date: str                   # YYYY-MM-DD
    datetime: str
    relative_orbit: int | None
    cloud_cover: float          # whole tile, %
    sun_elevation: float
    sun_azimuth: float
    epsg: int | None
    hrefs: dict                 # asset name -> URL


def search_sentinel2(
    lat: float,
    lon: float,
    start: str,
    end: str,
    max_cloud: float = 60.0,
    relative_orbit: int | None = None,
    collection: str = COLLECTION,
    session=None,
) -> list[Scene]:
    """Scenes covering (lat, lon) between ISO dates ``start`` and ``end``, oldest first."""
    import requests

    s = session or requests.Session()
    body = {
        "collections": [collection],
        "intersects": {"type": "Point", "coordinates": [lon, lat]},
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "limit": 100,
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
    }
    scenes: dict[str, Scene] = {}
    for _ in range(100):
        r = s.post(EARTH_SEARCH_URL, json=body, timeout=90)
        r.raise_for_status()
        j = r.json()
        for f in j.get("features", []):
            sc = _scene_from_feature(f)
            if relative_orbit is None or sc.relative_orbit == relative_orbit:
                scenes[sc.id] = sc
        nxt = [link for link in j.get("links", []) if link.get("rel") == "next"]
        if not nxt or "body" not in nxt[0]:
            break
        body = nxt[0]["body"]
    return sorted(scenes.values(), key=lambda sc: sc.datetime)


def _scene_from_feature(f: dict) -> Scene:
    p = f["properties"]
    m = _ORBIT_RE.search(p.get("s2:product_uri", ""))
    return Scene(
        id=f["id"],
        date=p["datetime"][:10],
        datetime=p["datetime"],
        relative_orbit=int(m.group(1)) if m else None,
        cloud_cover=float(p.get("eo:cloud_cover", np.nan)),
        sun_elevation=float(p.get("view:sun_elevation", np.nan)),
        sun_azimuth=float(p.get("view:sun_azimuth", np.nan)),
        epsg=p.get("proj:epsg"),
        hrefs={k: a["href"] for k, a in f.get("assets", {}).items() if a.get("href", "").endswith(".tif")},
    )


def read_window(href: str, bounds: tuple[float, float, float, float]) -> tuple[np.ndarray, tuple]:
    """
    Read ``bounds`` = (xmin, ymin, xmax, ymax), in the raster's CRS, from a COG.

    Returns the array and its affine transform as a 6-tuple
    (x0, dx, 0, y0, 0, -dy) in GDAL order.
    """
    import rasterio
    from rasterio.windows import from_bounds

    with rasterio.Env(**GDAL_HTTP_ENV), rasterio.open(href) as ds:
        win = from_bounds(*bounds, transform=ds.transform).round_offsets().round_lengths()
        arr = ds.read(1, window=win, boundless=True, fill_value=0)
        t = ds.window_transform(win)
    return arr, (t.c, t.a, t.b, t.f, t.d, t.e)


def scl_fractions(scl: np.ndarray, mask: np.ndarray) -> dict:
    """
    Fractions of ``mask`` pixels in each SCL group.

    ``shadow`` is dark area plus cloud shadow. Dark area (class 2) is
    reported separately as ``dark`` because it also flags sunlit dark rock
    and debris; ``cloud_shadow`` (class 3) is the part caused by cloud.
    """
    v = scl[mask]
    n = max(v.size, 1)
    return {
        "cloud": round(float(np.isin(v, SCL_CLOUD).sum() / n), 3),
        "shadow": round(float(np.isin(v, SCL_SHADOW).sum() / n), 3),
        "dark": round(float(np.isin(v, SCL_DARK).sum() / n), 3),
        "cloud_shadow": round(float(np.isin(v, SCL_CLOUD_SHADOW).sum() / n), 3),
        "snow": round(float(np.isin(v, SCL_SNOW).sum() / n), 3),
        "nodata": round(float(np.isin(v, SCL_NODATA).sum() / n), 3),
    }


def highpass(img: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    """Remove illumination gradients: image minus its Gaussian blur, scaled to unit variance."""
    from scipy.ndimage import gaussian_filter

    a = img.astype(np.float32)
    a = a - gaussian_filter(a, sigma)
    sd = np.nanstd(a)
    return a / sd if sd > 0 else a


@dataclass
class OffsetField:
    """Dense offsets of the secondary image relative to the reference.

    ``dx`` is positive to the east (increasing column), ``dy`` positive to
    the south (increasing row), both in pixels. ``peak`` is the normalised
    cross-correlation coefficient at the match (0-1), a quality measure. ``rows`` /
    ``cols`` are the centre pixel of each window in the input images.
    """
    dx: np.ndarray
    dy: np.ndarray
    peak: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    window: int
    step: int


def offset_field(
    ref: np.ndarray,
    sec: np.ndarray,
    window: int = 32,
    step: int = 8,
    upsample: int = 20,
    batch: int = 4000,
    passes: int = 2,
) -> OffsetField:
    """
    Dense cross-correlation offsets between two co-registered images.

    Each ``window`` x ``window`` chip (Hann-tapered) is matched by FFT
    cross-correlation; the integer peak is refined to
    1/``upsample`` pixel with a local upsampled DFT. Tapering both chips
    biases large shifts toward zero, so later passes re-extract the
    secondary chip at the rounded shift from the previous pass and add the
    residual. Shifts up to about ±window/2 pixels are measurable.
    """
    if ref.shape != sec.shape:
        raise ValueError("ref and sec must have the same shape")
    ref = np.asarray(ref, np.float32)
    sec = np.asarray(sec, np.float32)
    H, W = ref.shape
    wa = sliding_window_view(ref, (window, window))
    wb = sliding_window_view(sec, (window, window))
    r_tl = np.arange(0, H - window + 1, step)
    c_tl = np.arange(0, W - window + 1, step)
    ny, nx = len(r_tl), len(c_tl)
    R_TL, C_TL = (g.ravel() for g in np.meshgrid(r_tl, c_tl, indexing="ij"))
    taper = np.outer(np.hanning(window), np.hanning(window)).astype(np.float32)

    shift_r = np.zeros(ny * nx)
    shift_c = np.zeros(ny * nx)
    dx = np.full(ny * nx, np.nan, np.float32)
    dy = np.full_like(dx, np.nan)
    peak = np.zeros_like(dx)
    for _ in range(max(1, passes)):
        ir = np.clip(R_TL + np.round(shift_r).astype(int), 0, H - window)
        ic = np.clip(C_TL + np.round(shift_c).astype(int), 0, W - window)
        base_r, base_c = ir - R_TL, ic - C_TL
        for s in range(0, ny * nx, batch):
            sl = slice(s, s + batch)
            sr, sc, pk, ok = _correlate(wa[R_TL[sl], C_TL[sl]], wb[ir[sl], ic[sl]], taper, upsample)
            dy[sl] = np.where(ok, base_r[sl] + sr, np.nan)
            dx[sl] = np.where(ok, base_c[sl] + sc, np.nan)
            peak[sl] = np.where(ok, pk, 0)
        shift_r = np.nan_to_num(dy.astype(np.float64))
        shift_c = np.nan_to_num(dx.astype(np.float64))
    return OffsetField(dx.reshape(ny, nx), dy.reshape(ny, nx), peak.reshape(ny, nx),
                       r_tl + window // 2, c_tl + window // 2, window, step)


def offset_points(
    ref: np.ndarray,
    sec: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    window: int = 16,
    upsample: int = 20,
    batch: int = 4000,
    passes: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Offsets (dx east, dy south, in pixels) and correlation peak at chosen
    window centres (``rows``, ``cols``), with the same matching as
    ``offset_field``. Centres whose window would leave the image are NaN.
    """
    if ref.shape != sec.shape:
        raise ValueError("ref and sec must have the same shape")
    ref = np.asarray(ref, np.float32)
    sec = np.asarray(sec, np.float32)
    H, W = ref.shape
    wa = sliding_window_view(ref, (window, window))
    wb = sliding_window_view(sec, (window, window))
    R_TL = np.asarray(rows, int) - window // 2
    C_TL = np.asarray(cols, int) - window // 2
    inside = (R_TL >= 0) & (C_TL >= 0) & (R_TL <= H - window) & (C_TL <= W - window)
    R_TL, C_TL = np.clip(R_TL, 0, H - window), np.clip(C_TL, 0, W - window)
    taper = np.outer(np.hanning(window), np.hanning(window)).astype(np.float32)
    n = len(R_TL)
    dx = np.full(n, np.nan, np.float32); dy = np.full_like(dx, np.nan); peak = np.zeros_like(dx)
    shift_r = np.zeros(n); shift_c = np.zeros(n)
    for _ in range(max(1, passes)):
        ir = np.clip(R_TL + np.round(shift_r).astype(int), 0, H - window)
        ic = np.clip(C_TL + np.round(shift_c).astype(int), 0, W - window)
        for s in range(0, n, batch):
            sl = slice(s, s + batch)
            sr, sc, pk, ok = _correlate(wa[R_TL[sl], C_TL[sl]], wb[ir[sl], ic[sl]], taper, upsample)
            dy[sl] = np.where(ok, ir[sl] - R_TL[sl] + sr, np.nan)
            dx[sl] = np.where(ok, ic[sl] - C_TL[sl] + sc, np.nan)
            peak[sl] = np.where(ok, pk, 0)
        shift_r = np.nan_to_num(dy.astype(np.float64)); shift_c = np.nan_to_num(dx.astype(np.float64))
    dx[~inside] = np.nan; dy[~inside] = np.nan; peak[~inside] = 0
    return dx, dy, peak


def _correlate(ca: np.ndarray, cb: np.ndarray, taper: np.ndarray, upsample: int):
    """Sub-pixel shift of each chip in ``cb`` relative to ``ca`` ([B, n, n] stacks)."""
    n = ca.shape[-1]
    ca = (ca - ca.mean(axis=(1, 2), keepdims=True)) * taper
    cb = (cb - cb.mean(axis=(1, 2), keepdims=True)) * taper
    # Plain (unwhitened) cross-correlation: whitening the cross-power
    # spectrum pixel-locks band-limited imagery toward integer shifts.
    R = np.fft.fft2(cb) * np.conj(np.fft.fft2(ca))
    energy = np.sqrt((ca ** 2).sum(axis=(1, 2)) * (cb ** 2).sum(axis=(1, 2)))
    ok = energy > 1e-6
    corr = np.fft.ifft2(R).real.reshape(len(ca), -1)
    k = corr.argmax(axis=1)
    pk = corr[np.arange(len(k)), k] / np.maximum(energy, 1e-12)
    r0, c0 = np.divmod(k, n)
    # Signed integer shift (FFT wrap-around)
    r0 = np.where(r0 > n // 2, r0 - n, r0).astype(np.float64)
    c0 = np.where(c0 > n // 2, c0 - n, c0).astype(np.float64)
    sr, sc = _upsampled_refine(R, r0, c0, upsample)
    return sr, sc, pk, ok


def _upsampled_refine(R: np.ndarray, r0: np.ndarray, c0: np.ndarray, up: int) -> tuple[np.ndarray, np.ndarray]:
    """Refine integer peaks to 1/up pixel with a 1.5-pixel upsampled DFT around each."""
    n = R.shape[-1]
    size = int(np.ceil(1.5 * up))
    half = size // 2
    freqs = np.fft.ifftshift(np.arange(n) - n // 2).astype(np.float64)       # [n]
    u = (np.arange(size) - half) / up                                         # [size]
    # Sample positions around each peak: r0 + u, c0 + u
    pr = r0[:, None] + u[None, :]                                             # [B, size]
    pc = c0[:, None] + u[None, :]
    ker_r = np.exp(2j * np.pi * pr[:, :, None] * freqs[None, None, :] / n)    # [B, size, n]
    ker_c = np.exp(2j * np.pi * pc[:, :, None] * freqs[None, None, :] / n)
    cc = np.einsum("bun,bnm,bvm->buv", ker_r, R, ker_c, optimize=True).real   # [B, size, size]
    k = cc.reshape(cc.shape[0], -1).argmax(axis=1)
    i, j = np.divmod(k, size)
    return r0 + u[i], c0 + u[j]


def reference_offsets(field: OffsetField, stable: np.ndarray, min_peak: float = 0.0):
    """
    Subtract the median offset over stable windows (a bulk co-registration
    residual) and return (dx, dy, sigma_x, sigma_y), sigma = 1.4826 × MAD
    of the referenced stable offsets. ``stable`` is a bool mask on the
    field's window grid. Windows with ``peak < min_peak`` become NaN.
    """
    good = np.isfinite(field.dx) & (field.peak >= min_peak)
    dx = np.where(good, field.dx, np.nan)
    dy = np.where(good, field.dy, np.nan)
    st = stable & good
    if st.sum() < 10:
        raise ValueError(f"only {st.sum()} stable windows pass the quality mask")
    dx = dx - np.median(dx[st])
    dy = dy - np.median(dy[st])

    def sig(v):
        return float(1.4826 * np.median(np.abs(v - np.median(v))))
    return dx, dy, sig(dx[st]), sig(dy[st])
