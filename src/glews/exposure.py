"""
Real downstream exposure: which source pixels can send a mass flow onto
people or infrastructure?

- Assets come from OpenStreetMap through the Overpass API (settlements,
  roads, power plants and generators, dams), optionally as they stood on
  a past date, so exposure is judged from before an event.
- Reach is a flow-path test on a DEM: from each source cell, follow D8
  steepest descent on the depression-filled surface. The path reaches an
  asset if it passes within ``buffer_m`` of one while the travel angle
  still allows it: H / L >= ``min_hl``, where H is the drop from the source
  and L the horizontal path length so far (Fahrböschung / angle of reach).

This replaces synthetic settlements for any claim about who is exposed.
"""
from __future__ import annotations

import heapq
import json
import logging
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

logger = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = "GLEWS-research/0.1 (glacier hazard early-warning prototype)"
ASSET_QUERY = (
    'node["place"~"^(city|town|village|hamlet|isolated_dwelling)$"]{bb};'
    'way["highway"~"^(trunk|primary|secondary|tertiary|unclassified|track)$"]{bb};'
    'nwr["power"~"^(plant|generator)$"]{bb};'
    'nwr["waterway"="dam"]{bb};'
)
GLO30_TILE = ("https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM/"
              "Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM.tif")
# D8 neighbour offsets (row, col) and their horizontal step lengths in cells
_D8 = np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)])
_D8_LEN = np.hypot(_D8[:, 0], _D8[:, 1])


def fetch_osm_assets(
    south: float, west: float, north: float, east: float,
    date: str | None = None,
    cache: str | Path | None = None,
    timeout: int = 180,
) -> list[dict]:
    """
    Assets in a lat/lon box as ``{"kind", "name", "lon": [...], "lat": [...]}``
    (one vertex for a node, the full line or outline for a way).

    ``date`` (ISO, e.g. "2026-08-01T00:00:00Z") queries OSM as it stood
    then. ``cache`` is a JSON file read if present, written otherwise.
    """
    if cache is not None and Path(cache).exists():
        return json.loads(Path(cache).read_text())
    import requests
    bb = f"({south},{west},{north},{east})"
    head = f'[out:json][timeout:{timeout}]' + (f'[date:"{date}"]' if date else "") + ";"
    query = head + "(" + ASSET_QUERY.format(bb=bb) + ");out geom tags;"
    r = requests.post(OVERPASS_URL, data={"data": query}, headers={"User-Agent": USER_AGENT}, timeout=timeout + 30)
    r.raise_for_status()
    assets = parse_overpass(r.json())
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        Path(cache).write_text(json.dumps(assets))
    return assets


def parse_overpass(payload: dict) -> list[dict]:
    """Overpass JSON (``out geom tags``) -> asset dicts."""
    out = []
    for e in payload.get("elements", []):
        t = e.get("tags", {})
        if "place" in t:
            kind = "settlement"
        elif "highway" in t:
            kind = "road"
        elif "power" in t:
            kind = "power"
        elif t.get("waterway") == "dam":
            kind = "dam"
        else:
            continue
        if "lon" in e:
            lon, lat = [e["lon"]], [e["lat"]]
        elif "geometry" in e:
            lon = [p["lon"] for p in e["geometry"] if p]
            lat = [p["lat"] for p in e["geometry"] if p]
        elif "center" in e:
            lon, lat = [e["center"]["lon"]], [e["center"]["lat"]]
        else:
            continue
        out.append(dict(kind=kind, name=t.get("name:en") or t.get("name"), lon=lon, lat=lat))
    return out


def rasterize_assets(xy_assets: list[tuple[np.ndarray, np.ndarray]], x0: float, y0: float, post: float,
                     shape: tuple[int, int]) -> np.ndarray:
    """
    Boolean grid of cells touched by assets given in grid CRS coordinates.

    ``x0``, ``y0`` are the upper-left corner; rows run south. Lines are
    densified to half a cell so no cell along a road is skipped.
    """
    grid = np.zeros(shape, bool)
    for xs, ys in xy_assets:
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        if xs.size > 1:
            seg = np.hypot(np.diff(xs), np.diff(ys))
            n = np.maximum(1, np.ceil(seg / (post / 2)).astype(int))
            xs = np.concatenate([np.linspace(xs[i], xs[i + 1], n[i], endpoint=False) for i in range(len(n))] + [xs[-1:]])
            ys = np.concatenate([np.linspace(ys[i], ys[i + 1], n[i], endpoint=False) for i in range(len(n))] + [ys[-1:]])
        c = np.floor((xs - x0) / post).astype(int)
        r = np.floor((y0 - ys) / post).astype(int)
        ok = (r >= 0) & (c >= 0) & (r < shape[0]) & (c < shape[1])
        grid[r[ok], c[ok]] = True
    return grid


def fill_depressions(z: np.ndarray, epsilon: float = 1e-3) -> np.ndarray:
    """Priority-flood fill (Barnes et al. 2014) so every cell drains to the edge.

    ``epsilon`` adds a small gradient across flats so D8 can route them.
    NaN cells are treated as outlets.
    """
    z = np.asarray(z, float)
    out = z.copy()
    nr, nc = z.shape
    done = np.zeros(z.shape, bool)
    heap = []
    edge = np.zeros(z.shape, bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    nan = ~np.isfinite(z)
    edge |= ndi.binary_dilation(nan, structure=np.ones((3, 3))) & ~nan
    done |= nan
    for r, c in zip(*np.nonzero(edge & ~nan)):
        heapq.heappush(heap, (out[r, c], r, c))
        done[r, c] = True
    while heap:
        h, r, c = heapq.heappop(heap)
        for dr, dc in _D8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < nr and 0 <= cc < nc and not done[rr, cc]:
                done[rr, cc] = True
                if out[rr, cc] <= h:
                    out[rr, cc] = h + epsilon
                heapq.heappush(heap, (out[rr, cc], rr, cc))
    return out


def d8_receivers(zf: np.ndarray) -> np.ndarray:
    """Index (into the flattened grid) of each cell's steepest-descent neighbour; -1 at outlets."""
    nr, nc = zf.shape
    pad = np.pad(zf, 1, constant_values=np.inf)
    drops = np.stack([(zf - pad[1 + dr:1 + dr + nr, 1 + dc:1 + dc + nc]) / L
                      for (dr, dc), L in zip(_D8, _D8_LEN)])
    drops = np.where(np.isfinite(drops), drops, -np.inf)
    k = np.argmax(drops, axis=0)
    best = np.take_along_axis(drops, k[None], 0)[0]
    rows, cols = np.indices(zf.shape)
    rr, cc = rows + _D8[k, 0], cols + _D8[k, 1]
    rec = rr * nc + cc
    return np.where(best > 0, rec, -1).ravel()


def reach_exposure(
    z: np.ndarray,
    post: float,
    exposed: np.ndarray,
    sources: np.ndarray,
    min_hl: float = 0.1,
    buffer_m: float = 250.0,
    zf: np.ndarray | None = None,
) -> np.ndarray:
    """
    For each source cell, can a flow following D8 steepest descent reach an
    exposed cell (within ``buffer_m``) while H / L >= ``min_hl``?

    ``z`` is the DEM (metres) on a square grid of spacing ``post``;
    ``exposed`` and ``sources`` are boolean grids of the same shape. Returns
    a boolean grid, True at sources that reach exposure. ``zf`` (the
    depression-filled DEM) can be passed to reuse it.
    """
    if zf is None:
        zf = fill_depressions(z)
    rec = d8_receivers(zf)
    nc = z.shape[1]
    near = ndi.binary_dilation(exposed, iterations=max(0, int(round(buffer_m / post)))) if buffer_m > 0 else exposed
    near = near.ravel()
    zr = zf.ravel()
    idx = np.flatnonzero(sources)
    z0 = zr[idx]
    cur = idx.copy()
    L = np.zeros(idx.size)
    hit = near[cur].copy()
    alive = ~hit
    while alive.any():
        nxt = rec[cur[alive]]
        stop = nxt < 0
        a = np.flatnonzero(alive)
        alive[a[stop]] = False
        a, nxt = a[~stop], nxt[~stop]
        step = np.hypot(cur[a] // nc - nxt // nc, cur[a] % nc - nxt % nc) * post
        L[a] += step
        cur[a] = nxt
        ok = (z0[a] - zr[nxt]) >= min_hl * L[a]
        alive[a[~ok]] = False
        h = ok & near[nxt]
        hit[a[h]] = True
        alive[a[h]] = False
    out = np.zeros(z.shape, bool)
    out.ravel()[idx] = hit
    return out


def glo30_mosaic(x0: float, y0: float, post: float, shape: tuple[int, int], crs: str = "EPSG:32645",
                 cache: str | Path | None = None) -> np.ndarray:
    """Copernicus GLO-30 averaged onto a regular grid (upper-left ``x0, y0``), mosaicking every tile it needs."""
    if cache is not None and Path(cache).exists():
        return np.load(cache)
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import reproject, Resampling, transform_bounds
    from .optical import GDAL_HTTP_ENV
    tr = from_origin(x0, y0, post, post)
    w, s, e, n = transform_bounds(crs, "EPSG:4326", x0, y0 - shape[0] * post, x0 + shape[1] * post, y0)
    dst = np.full(shape, np.nan)
    with rasterio.Env(**GDAL_HTTP_ENV):
        for la in range(int(np.floor(s)), int(np.floor(n)) + 1):
            for lo in range(int(np.floor(w)), int(np.floor(e)) + 1):
                try:
                    ds = rasterio.open(GLO30_TILE.format(lat=la, lon=lo))
                except Exception as ex:          # ocean / missing tile
                    logger.warning("GLO-30 tile N%02d E%03d unavailable: %s", la, lo, ex)
                    continue
                with ds:
                    part = np.full(shape, np.nan)
                    reproject(rasterio.band(ds, 1), part, dst_transform=tr, dst_crs=crs,
                              resampling=Resampling.average, dst_nodata=np.nan)
                    dst = np.where(np.isfinite(part), part, dst)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, dst)
    return dst
