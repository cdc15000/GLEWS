"""
NISAR data loading module.

Reads NISAR L2 GUNW (unwrapped interferograms) and GOFF (offset fields)
HDF5 products and converts them into formats compatible with the GLEWS
detection pipeline.

NISAR L-band (24 cm wavelength) maintains coherence on glaciated surfaces
where Sentinel-1 C-band (5.6 cm) decorrelates.

GOFF products provide pixel offset tracking — amplitude cross-correlation
that measures large displacements where phase-based InSAR loses coherence.

Per-pair handling (both products):

- **Reliability mask.** GUNW pixels with ``connectedComponents == 0``
  (unwrapping unreliable) or the 65535 fill value are dropped for that
  pair. GOFF pixels below a minimum SNR are dropped for that pair.
- **Referencing.** Each connected component of an unwrapped interferogram
  carries its own unknown 2π multiple, so a pair can only be referenced
  within one component. With ``reference_mask`` given, each pair is
  referenced to the median over the reference pixels of the component
  that holds most of them; pixels in other components are dropped for that
  pair. GOFF pairs are referenced to the median offset over the reference
  pixels.
- **Gaps.** In a sequential network (each pair spans one interval), a
  pixel dropped from one pair has no measurement for that interval.
  ``gap_policy="break"`` (default) leaves every later epoch NaN for that
  pixel; ``"bridge"`` fills the missing interval from the mean rate of the
  nearest valid intervals either side and records where it did so. No
  policy inserts zero motion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)

# NISAR L-band wavelength (m)
WAVELENGTH_M = 0.2384
# connectedComponents fill value in L2 GUNW
CC_FILL = 65535
# Fewest reference pixels a pair needs to be referenced
MIN_REFERENCE_PIXELS = 10
GAP_POLICIES = ("break", "bridge")


@dataclass
class NISARTimeseries:
    """Displacement time series from NISAR interferograms or offsets."""
    dates: np.ndarray
    date_strings: list[str]
    displacement: np.ndarray    # [n_dates, n_rows, n_cols] meters
    velocity: np.ndarray        # [n_rows, n_cols] m/yr
    coherence: np.ndarray       # [n_rows, n_cols]
    latitude: np.ndarray        # [n_rows, n_cols] (may be projected coords)
    longitude: np.ndarray       # [n_rows, n_cols]
    source: str                 # "GUNW" or "GOFF"
    metadata: dict
    # GOFF only: along-track displacement, same shape as ``displacement``
    displacement_along_track: np.ndarray | None = None


def load_nisar_gunw_stack(
    gunw_dir: str | Path,
    min_coherence: float = 0.3,
    track: int | None = None,
    crop_to_site: tuple[float, float, float] | None = None,
    reference_mask: np.ndarray | None = None,
    mask_unreliable: bool = True,
    correct_ionosphere: bool = False,
    gap_policy: str = "break",
) -> NISARTimeseries:
    """
    Load NISAR GUNW products and invert to displacement time series.

    Parameters
    ----------
    gunw_dir : path
        Directory containing NISAR GUNW .h5 files.
    min_coherence : float
        Pixels whose mean coherence over all pairs is below this are dropped.
    track : int or None
        If specified, only load products from this track number.
    crop_to_site : (easting, northing, radius_m) or None
        If specified, crop to this area to reduce memory usage.
    reference_mask : bool array [rows, cols] or None
        Stable reference pixels on the (cropped) grid. Each pair is
        referenced to their median within one connected component; see the
        module docstring. None leaves pairs unreferenced.
    mask_unreliable : bool
        Drop pixels with ``connectedComponents`` 0 or fill, per pair.
    correct_ionosphere : bool
        Subtract ``ionospherePhaseScreen`` where the product has it.
    gap_policy : "break" or "bridge"
        How a pixel missing from one pair of a sequential network is
        carried forward; see the module docstring.
    """
    if gap_policy not in GAP_POLICIES:
        raise ValueError(f"gap_policy must be one of {GAP_POLICIES}")
    gunw_dir = Path(gunw_dir)
    gunw_files = sorted(gunw_dir.glob("NISAR_*GUNW*.h5"))

    if not gunw_files:
        raise FileNotFoundError(f"No NISAR GUNW files in {gunw_dir}")

    # Filter by track if specified
    if track is not None:
        gunw_files = [f for f in gunw_files if f"_{track:03d}_" in f.name]

    logger.info("Loading %d NISAR GUNW products", len(gunw_files))

    # Load all interferograms
    ifg_data = []
    for f in gunw_files:
        try:
            data = _read_gunw(f, crop_to_site)
            if data is not None:
                ifg_data.append(data)
                logger.info("  %s → %s: %s, coh=%.2f",
                           data["ref_date"], data["sec_date"],
                           data["unwrapped_phase"].shape,
                           np.nanmean(data["coherence"]))
        except Exception as e:
            logger.warning("  Failed %s: %s", f.name[:50], e)

    if not ifg_data:
        raise ValueError("No valid GUNW products loaded")

    n_rows, n_cols = ifg_data[0]["unwrapped_phase"].shape
    if reference_mask is not None and reference_mask.shape != (n_rows, n_cols):
        raise ValueError(f"reference_mask shape {reference_mask.shape} != grid {(n_rows, n_cols)}")

    phase_stack = np.zeros((len(ifg_data), n_rows, n_cols), dtype=np.float32)
    coh_stack = np.zeros_like(phase_stack)
    pair_info = []
    for k, d in enumerate(ifg_data):
        phase, info = _prepare_gunw_pair(d, reference_mask, mask_unreliable, correct_ionosphere)
        phase_stack[k] = phase
        coh_stack[k] = d["coherence"]
        pair_info.append({"ref": d["ref_date"], "sec": d["sec_date"], **info})

    # Phase to LOS displacement
    disp_stack = -phase_stack * WAVELENGTH_M / (4 * np.pi)

    # Mean coherence for masking
    mean_coh = np.nanmean(coh_stack, axis=0)
    valid_mask = mean_coh >= min_coherence
    logger.info("Valid pixels: %d of %d (%.1f%%, coh≥%.2f)",
               valid_mask.sum(), valid_mask.size, 100 * valid_mask.mean(), min_coherence)

    pairs = [(d["ref_date"], d["sec_date"]) for d in ifg_data]
    all_dates, displacement, inv_info = _invert_network(pairs, disp_stack, valid_mask, gap_policy)

    # Build coordinate grids
    lon_grid, lat_grid = np.meshgrid(ifg_data[0]["x_coords"], ifg_data[0]["y_coords"])

    dates_ord = np.array([datetime.strptime(d, "%Y%m%d").toordinal() for d in all_dates])
    velocity = _compute_velocity(dates_ord, displacement)

    return NISARTimeseries(
        dates=dates_ord,
        date_strings=all_dates,
        displacement=displacement,
        velocity=velocity,
        coherence=mean_coh,
        latitude=lat_grid,
        longitude=lon_grid,
        source="GUNW",
        metadata={"wavelength_m": WAVELENGTH_M, "n_interferograms": len(ifg_data),
                  "sensor": "NISAR", "band": "L", "track": track,
                  "referenced": reference_mask is not None,
                  "mask_unreliable": mask_unreliable,
                  "ionosphere_corrected": correct_ionosphere,
                  "pairs": pair_info, **inv_info},
    )


def read_goff_pairs(
    goff_dir: str | Path,
    layer: str = "layer2",
    min_snr: float = 5.0,
    track: int | None = None,
    crop_to_site: tuple[float, float, float] | None = None,
    reference_mask: np.ndarray | None = None,
    max_rate: float | None = None,
) -> list[dict]:
    """
    Read GOFF products as referenced, SNR-masked offset pairs.

    ``max_rate`` (metres per day, off by default) masks offset blunders:
    pixels whose referenced offset magnitude, hypot(range, along-track),
    divided by the pair length exceeds it are set to NaN in both components
    (``n_blunders`` counts them). Mismatched correlation windows give
    implausible, sign-flipping offsets that survive the SNR mask
    (docs/dev_phase2b.md).

    Returns one dict per pair, sorted by reference date, with ``ref_date``,
    ``sec_date``, ``days``, ``range_offset`` and ``along_track_offset``
    (metres, NaN where masked), ``snr``, ``x_coords`` / ``y_coords``, and
    with ``reference_mask``: ``reference_offset`` (range, along-track
    medians removed) and ``noise_sigma`` (1.4826 × MAD of the referenced
    offsets over the reference pixels, range and along-track).
    """
    goff_dir = Path(goff_dir)
    goff_files = sorted(goff_dir.glob("NISAR_*GOFF*.h5"))
    if not goff_files:
        raise FileNotFoundError(f"No NISAR GOFF files in {goff_dir}")
    if track is not None:
        goff_files = [f for f in goff_files if f"_{track:03d}_" in f.name]

    pairs = []
    for f in goff_files:
        try:
            d = _read_goff(f, crop_to_site, layer=layer)
        except Exception as e:
            logger.warning("  Failed %s: %s", f.name[:50], e)
            continue
        if d is None:
            continue
        ok = np.isfinite(d["range_offset"]) & np.isfinite(d["along_track_offset"])
        if d["snr"] is not None:
            ok &= d["snr"] >= min_snr
        rg = np.where(ok, d["range_offset"], np.nan)
        az = np.where(ok, d["along_track_offset"], np.nan)
        d.update(range_offset=rg, along_track_offset=az, layer=layer, min_snr=min_snr,
                 days=_days_between(d["ref_date"], d["sec_date"]))
        if reference_mask is not None:
            if reference_mask.shape != rg.shape:
                raise ValueError(f"reference_mask shape {reference_mask.shape} != grid {rg.shape}")
            ref = reference_mask & ok
            if ref.sum() < MIN_REFERENCE_PIXELS:
                logger.warning("  %s→%s: %d reference pixels pass SNR; pair dropped",
                               d["ref_date"], d["sec_date"], ref.sum())
                continue
            r0, a0 = float(np.median(rg[ref])), float(np.median(az[ref]))
            d["range_offset"] = rg - r0
            d["along_track_offset"] = az - a0
            d["reference_offset"] = (r0, a0)
        if max_rate is not None:
            with np.errstate(invalid="ignore"):
                bad = np.hypot(d["range_offset"], d["along_track_offset"]) / d["days"] > max_rate
            d["range_offset"] = np.where(bad, np.nan, d["range_offset"])
            d["along_track_offset"] = np.where(bad, np.nan, d["along_track_offset"])
            d["n_blunders"] = int(bad.sum())
        d["max_rate"] = max_rate
        if reference_mask is not None:
            d["noise_sigma"] = (_robust_sigma(d["range_offset"][ref]),
                                _robust_sigma(d["along_track_offset"][ref]))
        pairs.append(d)
    pairs.sort(key=lambda d: (d["ref_date"], d["sec_date"]))
    return pairs


def load_nisar_goff_stack(
    goff_dir: str | Path,
    layer: str = "layer2",
    min_snr: float = 5.0,
    track: int | None = None,
    crop_to_site: tuple[float, float, float] | None = None,
    reference_mask: np.ndarray | None = None,
    gap_policy: str = "break",
    max_rate: float | None = None,
) -> NISARTimeseries:
    """
    Load NISAR GOFF (offset field) products as a displacement time series.

    ``displacement`` is the slant-range component (positive = away from the
    sensor per NISAR convention); ``displacement_along_track`` holds the
    along-track component. ``coherence`` is the fraction of pairs in which
    each pixel passed the SNR mask (and ``max_rate``). See ``read_goff_pairs``
    for the masking, blunder rejection and referencing, and the module docstring for ``gap_policy``.
    """
    if gap_policy not in GAP_POLICIES:
        raise ValueError(f"gap_policy must be one of {GAP_POLICIES}")
    offsets = read_goff_pairs(goff_dir, layer=layer, min_snr=min_snr, track=track,
                              crop_to_site=crop_to_site, reference_mask=reference_mask,
                              max_rate=max_rate)
    if not offsets:
        raise ValueError("No valid GOFF products loaded")
    logger.info("Loaded %d NISAR GOFF pairs (%s, SNR≥%g)", len(offsets), layer, min_snr)

    rg = np.array([d["range_offset"] for d in offsets], dtype=np.float32)
    az = np.array([d["along_track_offset"] for d in offsets], dtype=np.float32)
    valid_frac = np.mean(np.isfinite(rg), axis=0)
    pixel_mask = valid_frac > 0
    pairs = [(d["ref_date"], d["sec_date"]) for d in offsets]
    all_dates, disp_rg, inv_info = _invert_network(pairs, rg, pixel_mask, gap_policy)
    _, disp_az, _ = _invert_network(pairs, az, pixel_mask, gap_policy)

    lon_grid, lat_grid = np.meshgrid(offsets[0]["x_coords"], offsets[0]["y_coords"])
    dates_ord = np.array([datetime.strptime(d, "%Y%m%d").toordinal() for d in all_dates])
    return NISARTimeseries(
        dates=dates_ord,
        date_strings=all_dates,
        displacement=disp_rg,
        velocity=_compute_velocity(dates_ord, disp_rg),
        coherence=valid_frac.astype(np.float32),
        latitude=lat_grid,
        longitude=lon_grid,
        source="GOFF",
        metadata={"n_offsets": len(offsets), "sensor": "NISAR",
                  "method": "pixel_offset_tracking", "layer": layer, "min_snr": min_snr,
                  "max_rate": max_rate,
                  "track": track, "referenced": reference_mask is not None,
                  "pairs": [{"ref": d["ref_date"], "sec": d["sec_date"],
                             "noise_sigma": d.get("noise_sigma")} for d in offsets],
                  **inv_info},
        displacement_along_track=disp_az,
    )


def _prepare_gunw_pair(
    d: dict,
    reference_mask: np.ndarray | None,
    mask_unreliable: bool,
    correct_ionosphere: bool,
) -> tuple[np.ndarray, dict]:
    """Mask, correct and reference one unwrapped interferogram."""
    phase = d["unwrapped_phase"].astype(np.float32).copy()
    if correct_ionosphere and d.get("ionosphere") is not None:
        phase -= d["ionosphere"]
    invalid = ~np.isfinite(phase)
    cc = d.get("connected_components")
    if mask_unreliable and cc is not None:
        invalid |= (cc == 0) | (cc == CC_FILL)

    info = {"reference_component": None, "reference_pixels": 0}
    if reference_mask is not None:
        ref = reference_mask & ~invalid
        if cc is not None and ref.any():
            comps, counts = np.unique(cc[ref], return_counts=True)
            keep = (comps != 0) & (comps != CC_FILL)
            if keep.any():
                comp = comps[keep][np.argmax(counts[keep])]
                invalid |= cc != comp
                ref &= cc == comp
                info["reference_component"] = int(comp)
        info["reference_pixels"] = int(ref.sum())
        if ref.sum() < MIN_REFERENCE_PIXELS:
            logger.warning("  %s→%s: %d usable reference pixels; pair masked",
                           d["ref_date"], d["sec_date"], ref.sum())
            invalid[:] = True
        else:
            phase -= np.median(phase[ref])
    phase[invalid] = np.nan
    info["valid_fraction"] = round(float(np.mean(~invalid)), 4)
    return phase, info


def _invert_network(
    pairs: list[tuple[str, str]],
    stack: np.ndarray,
    pixel_mask: np.ndarray,
    gap_policy: str,
) -> tuple[list[str], np.ndarray, dict]:
    """
    Turn pairwise displacements into a time series (first date = 0).

    ``stack`` is [n_pairs, rows, cols] with NaN where a pair is masked.
    A sequential network is chained directly, honouring ``gap_policy``.
    Any other network is solved by least squares, only for pixels valid in
    every pair (others are left NaN, and a warning is logged).
    """
    all_dates = sorted({r for r, _ in pairs} | {s for _, s in pairs})
    idx = {d: i for i, d in enumerate(all_dates)}
    n_dates = len(all_dates)
    _, n_rows, n_cols = stack.shape
    G = np.zeros((len(pairs), n_dates - 1))
    for k, (r, s) in enumerate(pairs):
        G[k, idx[r]:idx[s]] = 1.0

    displacement = np.full((n_dates, n_rows, n_cols), np.nan, dtype=np.float32)
    displacement[0] = np.where(pixel_mask, 0.0, np.nan)
    info: dict = {}

    sequential = (G.sum(axis=1) == 1).all() and (G.sum(axis=0) == 1).all()
    if sequential:
        order = np.argmax(G, axis=1)               # interval index of each pair
        inc = np.empty((n_dates - 1, n_rows, n_cols), dtype=np.float32)
        inc[order] = stack
        inc[:, ~pixel_mask] = np.nan
        missing = ~np.isfinite(inc)
        if gap_policy == "bridge":
            days = np.diff([datetime.strptime(d, "%Y%m%d").toordinal() for d in all_dates])
            inc, bridged = _bridge_gaps(inc, days.astype(np.float32))
            info["bridged_fraction"] = round(float(bridged[:, pixel_mask].mean()), 4) if pixel_mask.any() else 0.0
            cum = np.cumsum(inc, axis=0)               # NaN only where no rate either side
        else:
            broken = np.cumsum(missing, axis=0) > 0
            cum = np.cumsum(np.where(missing, 0.0, inc), axis=0)
            cum[broken] = np.nan
        displacement[1:] = cum
        info["network"] = "sequential"
    else:
        full = pixel_mask & np.isfinite(stack).all(axis=0)
        n_drop = int(pixel_mask.sum() - full.sum())
        if n_drop:
            logger.warning("Non-sequential network: %d pixels masked in some pair are left NaN", n_drop)
        if full.any():
            inc = np.linalg.lstsq(G, stack[:, full], rcond=None)[0]
            displacement[1:, full] = np.cumsum(inc, axis=0)
        info["network"] = "least_squares"
    info["gap_policy"] = gap_policy
    info["missing_interval_fraction"] = (
        round(float((~np.isfinite(stack[:, pixel_mask])).mean()), 4) if pixel_mask.any() else 1.0)
    return all_dates, displacement, info


def _bridge_gaps(inc: np.ndarray, days: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fill NaN intervals with the mean rate of the nearest valid intervals either side."""
    n = inc.shape[0]
    rate = inc / days[:, None, None]
    finite = np.isfinite(rate)
    pos = np.arange(n)[:, None, None]
    # Index of the last valid interval at or before each position, and the next at or after
    prev = np.maximum.accumulate(np.where(finite, pos, -1), axis=0)
    nxt = np.minimum.accumulate(np.where(finite, pos, n)[::-1], axis=0)[::-1]
    cols = np.indices(inc.shape[1:])
    r_prev = np.where(prev >= 0, rate[np.clip(prev, 0, n - 1), cols[0], cols[1]], np.nan)
    r_next = np.where(nxt < n, rate[np.clip(nxt, 0, n - 1), cols[0], cols[1]], np.nan)
    with np.errstate(invalid="ignore"):
        fill = np.nanmean(np.stack([r_prev, r_next]), axis=0) * days[:, None, None]
    bridged = ~finite & np.isfinite(fill)
    return np.where(finite, inc, fill).astype(np.float32), bridged


def _robust_sigma(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def _days_between(a: str, b: str) -> int:
    return (datetime.strptime(b, "%Y%m%d") - datetime.strptime(a, "%Y%m%d")).days


def _pair_dates(filepath: Path) -> tuple[str, str] | None:
    """Reference and secondary dates from a NISAR product file name."""
    # Pattern: ..._YYYYMMDDT..._YYYYMMDDT..._YYYYMMDDT..._YYYYMMDDT..._
    parts = filepath.stem.split("_")
    date_parts = [p[:8] for p in parts if len(p) >= 15 and p[0] == "2" and "T" in p]
    # First two are ref start/end, second two are sec start/end
    if len(date_parts) >= 4:
        return date_parts[0], date_parts[2]
    if len(date_parts) >= 2:
        return date_parts[0], date_parts[1]
    return None


def _crop_index(x_coords, y_coords, crop, min_size):
    cx, cy, radius = crop
    col_mask = (x_coords >= cx - radius) & (x_coords <= cx + radius)
    row_mask = (y_coords >= cy - radius) & (y_coords <= cy + radius)
    if np.sum(col_mask) < min_size or np.sum(row_mask) < min_size:
        return None
    return row_mask, col_mask


def _find_pol(group) -> str | None:
    for key in group.keys():
        if len(key) == 2 and key.isalpha():
            return key
    return None


def _read_gunw(filepath: Path, crop: tuple | None = None) -> dict | None:
    """Read a single NISAR GUNW HDF5 file."""
    with h5py.File(filepath, "r") as f:
        base = "science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram"
        pol = _find_pol(f[base])
        if pol is None:
            return None
        g = f[f"{base}/{pol}"]
        x_coords = g["xCoordinates"][:]
        y_coords = g["yCoordinates"][:]
        sel = (slice(None), slice(None))
        if crop is not None:
            masks = _crop_index(x_coords, y_coords, crop, 10)
            if masks is None:
                return None
            sel = np.ix_(*masks)
            x_coords, y_coords = x_coords[masks[1]], y_coords[masks[0]]

        def read(name):
            return g[name][:][sel] if name in g else None

        # Float layers use NaN as _FillValue; 0 rad is a valid phase.
        unw = read("unwrappedPhase")
        coh = read("coherenceMagnitude")
        cc = read("connectedComponents")
        iono = read("ionospherePhaseScreen")

    dates = _pair_dates(filepath)
    if dates is None:
        return None
    return {
        "ref_date": dates[0],
        "sec_date": dates[1],
        "unwrapped_phase": unw.astype(np.float32),
        "coherence": coh.astype(np.float32),
        "connected_components": cc,
        "ionosphere": None if iono is None else iono.astype(np.float32),
        "x_coords": x_coords,
        "y_coords": y_coords,
    }


def _read_goff(filepath: Path, crop: tuple | None = None, layer: str = "layer2") -> dict | None:
    """Read one layer of a NISAR GOFF HDF5 file (offsets in metres, NaN fill)."""
    with h5py.File(filepath, "r") as f:
        base = "science/LSAR/GOFF/grids/frequencyA/pixelOffsets"
        pol = _find_pol(f[base])
        if pol is None:
            return None
        g = f[f"{base}/{pol}/{layer}"]
        for name in ("slantRangeOffset", "alongTrackOffset"):
            units = g[name].attrs.get("units")
            if units is not None and (units.decode() if isinstance(units, bytes) else units) != "meters":
                raise ValueError(f"{name} units are {units!r}, expected meters")
        # Each layer may carry its own grid; fall back to the shared one
        x_coords = g["xCoordinates"][:] if "xCoordinates" in g else f[f"{base}/xCoordinates"][:]
        y_coords = g["yCoordinates"][:] if "yCoordinates" in g else f[f"{base}/yCoordinates"][:]
        sel = (slice(None), slice(None))
        if crop is not None:
            masks = _crop_index(x_coords, y_coords, crop, 5)
            if masks is None:
                return None
            sel = np.ix_(*masks)
            x_coords, y_coords = x_coords[masks[1]], y_coords[masks[0]]
        rg = g["slantRangeOffset"][:][sel].astype(np.float32)
        az = g["alongTrackOffset"][:][sel].astype(np.float32)
        snr = g["snr"][:][sel].astype(np.float32) if "snr" in g else None

    dates = _pair_dates(filepath)
    if dates is None:
        return None
    return {
        "ref_date": dates[0],
        "sec_date": dates[1],
        "range_offset": rg,
        "along_track_offset": az,
        "snr": snr,
        "x_coords": x_coords,
        "y_coords": y_coords,
    }


def _compute_velocity(dates: np.ndarray, displacement: np.ndarray) -> np.ndarray:
    """Compute linear velocity from time series via vectorized regression."""
    n_dates, n_rows, n_cols = displacement.shape
    t = (dates - dates[0]).astype(np.float64) / 365.25

    velocity = np.full((n_rows, n_cols), np.nan, dtype=np.float32)

    # Vectorize: reshape to [n_dates, n_pixels]
    d_flat = displacement.reshape(n_dates, -1)
    valid_count = np.sum(np.isfinite(d_flat), axis=0)
    enough = valid_count >= 3

    if np.any(enough):
        for px in np.where(enough)[0]:
            d = d_flat[:, px]
            valid = np.isfinite(d)
            coeffs = np.polyfit(t[valid], d[valid], 1)
            r, c = divmod(px, n_cols)
            velocity[r, c] = coeffs[0]

    return velocity
