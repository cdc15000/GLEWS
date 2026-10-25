"""
Crop NISAR GOFF (pixel offset) products around a site via HTTP range reads.

Generalises scripts/nepal_2026/extract_goff_crops.py to any track, frame
and UTM zone for the 2027 prospective test (docs/prereg_2027_prospective.md):

- crops are taken on the product's native map grid, never resampled; the
  grid's EPSG code is read from the product and written into the crop;
- one product per (reference, secondary) date pair, the latest fileID;
- the track is chosen by ``choose_track``: the footprint must hold the crop
  plus 1 km, then the most pairs with secondary date in the baseline
  window, ties to ascending.

Output files keep the GOFF group layout (all three offset layers) so
``glews.nisar.read_goff_pairs`` reads them unchanged.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from .tm import fwd, inv, utm_lon0

logger = logging.getLogger(__name__)

BASE = "science/LSAR/GOFF/grids/frequencyA/pixelOffsets"
FIELDS = ("slantRangeOffset", "alongTrackOffset", "slantRangeOffsetVariance",
          "alongTrackOffsetVariance", "correlationSurfacePeak", "snr")
LAYERS = ("layer1", "layer2", "layer3")
HALF_M = 6000.0


def pair_dates(file_id: str) -> tuple[str, str]:
    """(reference, secondary) YYYYMMDD from a GOFF fileID."""
    d = [t[:8] for t in file_id.split("_") if len(t) >= 15 and t[0] == "2" and "T" in t]
    return d[0], d[2]


def unique_pairs(props: list[dict]) -> dict:
    """One product per (reference, secondary) pair: the latest fileID."""
    out = {}
    for p in props:
        k = pair_dates(p["fileID"])
        if k not in out or p["fileID"] > out[k]["fileID"]:
            out[k] = p
    return out


def _contains(ring_lonlat, lon, lat) -> bool:
    from matplotlib.path import Path as MP
    return bool(MP(np.asarray(ring_lonlat)).contains_point((lon, lat)))


def crop_corners_lonlat(lat: float, lon: float, half_m: float, lon0: float) -> list[tuple[float, float]]:
    """(lon, lat) corners of a square of half-width ``half_m`` around the site in a zone."""
    cx, cy = fwd(lat, lon, lon0)
    out = []
    for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        la, lo = inv(cx + dx * half_m, cy + dy * half_m, lon0)
        out.append((float(lo), float(la)))
    return out


def choose_track(results: list[dict], lat: float, lon: float, base_start: str, base_end: str,
                 margin_m: float = 1000.0) -> tuple | None:
    """
    (path, frame, direction) per the pre-registered rule, or None.

    ``results`` are dicts with ``properties`` (pathNumber, frameNumber,
    flightDirection, fileID) and ``footprint``, a list of (lon, lat)
    vertices. The footprint must contain every corner of the crop plus
    ``margin_m``; then most pairs with secondary date in
    [base_start, base_end]; ties to ascending, then lowest path.
    """
    from .tm import utm_crs
    corners = crop_corners_lonlat(lat, lon, HALF_M + margin_m, utm_lon0(utm_crs(lon)))
    groups: dict[tuple, list[dict]] = {}
    for r in results:
        p = r["properties"]
        if not all(_contains(r["footprint"], lo, la) for lo, la in corners):
            continue
        groups.setdefault((p["pathNumber"], p["frameNumber"], p["flightDirection"]), []).append(p)
    best, best_key = None, None
    for k, ps in groups.items():
        n = sum(1 for (_, sec) in unique_pairs(ps) if base_start <= sec <= base_end)
        key = (n, k[2].upper().startswith("ASC"), -k[0])
        if n > 0 and (best_key is None or key > best_key):
            best, best_key = k, key
    return best


def crop_product(f, lat: float, lon: float, half_m: float = HALF_M) -> dict:
    """Crop an open GOFF HDF5 product on its native grid around (lat, lon)."""
    pol = [k for k in f[BASE] if len(k) == 2][0]
    epsg = int(f[f"{BASE}/projection"][()])
    x = f[f"{BASE}/xCoordinates"][:] if f"{BASE}/xCoordinates" in f else f[f"{BASE}/{pol}/layer1/xCoordinates"][:]
    y = f[f"{BASE}/yCoordinates"][:] if f"{BASE}/yCoordinates" in f else f[f"{BASE}/{pol}/layer1/yCoordinates"][:]
    cx, cy = fwd(lat, lon, utm_lon0(f"EPSG:{epsg}"))
    ci = np.where(abs(x - cx) <= half_m)[0]
    ri = np.where(abs(y - cy) <= half_m)[0]
    if ci.size == 0 or ri.size == 0:
        raise ValueError(f"site ({lat}, {lon}) is outside the product grid")
    sl = (slice(ri[0], ri[-1] + 1), slice(ci[0], ci[-1] + 1))
    layers = {lay: {k: f[f"{BASE}/{pol}/{lay}/{k}"][sl] for k in FIELDS} for lay in LAYERS}
    return dict(pol=pol, epsg=epsg, x=x[ci], y=y[ri], layers=layers)


def write_crop(dst: str | Path, crop: dict, source_granule: str) -> Path:
    """Write a crop in the GOFF layout (atomically, via a .part file)."""
    import h5py
    dst = Path(dst)
    tmp = dst.with_suffix(".part")
    with h5py.File(tmp, "w") as o:
        for lay, arrs in crop["layers"].items():
            g = o.create_group(f"{BASE}/{crop['pol']}/{lay}")
            for k, v in arrs.items():
                g.create_dataset(k, data=v, compression="gzip")
            g["xCoordinates"] = crop["x"]
            g["yCoordinates"] = crop["y"]
        o[f"{BASE}/projection"] = np.int64(crop["epsg"])
        o.attrs["source_granule"] = source_granule
        o.attrs["epsg_code"] = crop["epsg"]
    tmp.rename(dst)
    return dst


def crop_crs(goff_dir: str | Path) -> str:
    """EPSG code of the crops in a directory; EPSG:32645 for 2026 crops written without one."""
    import h5py
    files = sorted(Path(goff_dir).glob("NISAR_*GOFF*.h5"))
    codes = set()
    for fp in files:
        with h5py.File(fp, "r") as f:
            codes.add(int(f.attrs.get("epsg_code", 32645)))
    if len(codes) != 1:
        raise ValueError(f"crops in {goff_dir} have mixed or no CRS: {codes}")
    return f"EPSG:{codes.pop()}"


def search(lat: float, lon: float, start: str, end: str) -> list[dict]:
    """ASF GOFF listing at a point: dicts with ``properties``, ``footprint`` and ``url``."""
    import asf_search as asf
    res = asf.search(platform=asf.PLATFORM.NISAR, processingLevel="GOFF",
                     intersectsWith=f"POINT({lon} {lat})", start=start, end=end)
    out = []
    for r in res:
        g = r.geometry
        ring = g["coordinates"][0] if g["type"] == "Polygon" else g["coordinates"][0][0]
        out.append(dict(properties=r.properties, footprint=[tuple(v[:2]) for v in ring], url=r.properties["url"]))
    return out


def extract(lat: float, lon: float, out_dir: str | Path, results: list[dict], track: tuple,
            sec_before: str | None = None, retries: int = 3) -> list[Path]:
    """Crop every unique pair on ``track`` (path, frame, direction), secondary date < ``sec_before``."""
    import fsspec
    import h5py
    import requests
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ps = [r["properties"] for r in results
          if (r["properties"]["pathNumber"], r["properties"]["frameNumber"], r["properties"]["flightDirection"]) == track]
    pairs = {k: p for k, p in unique_pairs(ps).items() if sec_before is None or k[1] < sec_before}
    session, fs, written = requests.Session(), fsspec.filesystem("http"), []
    for (ref, sec), p in sorted(pairs.items()):
        dst = out_dir / (p["fileID"] + ".h5")
        if dst.exists():
            written.append(dst)
            continue
        for attempt in range(retries):
            try:
                t = time.time()
                r = session.get(p["url"], stream=True)
                final = r.url
                r.close()
                with fs.open(final, block_size=2 ** 20, cache_type="blockcache") as fo, h5py.File(fo, "r") as f:
                    crop = crop_product(f, lat, lon)
                written.append(write_crop(dst, crop, p["fileID"]))
                logger.info("%s %s %s %.0fs", ref, sec, crop["layers"]["layer1"]["snr"].shape, time.time() - t)
                break
            except Exception as e:  # transient HTTP errors
                logger.warning("retry %s %s: %s %s", ref, sec, type(e).__name__, str(e)[:80])
    return written
