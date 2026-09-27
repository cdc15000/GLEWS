"""
NISAR data loading module.

Reads NISAR L2 GUNW (unwrapped interferograms) and GOFF (offset fields)
HDF5 products and converts them into formats compatible with the GEWS
detection pipeline.

NISAR L-band (24 cm wavelength) maintains coherence on glaciated surfaces
where Sentinel-1 C-band (5.6 cm) decorrelates.

GOFF products provide pixel offset tracking — amplitude cross-correlation
that measures large displacements where phase-based InSAR loses coherence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)


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


def load_nisar_gunw_stack(
    gunw_dir: str | Path,
    min_coherence: float = 0.3,
    track: int | None = None,
    crop_to_site: tuple[float, float, float] | None = None,
) -> NISARTimeseries:
    """
    Load NISAR GUNW products and invert to displacement time series.

    Parameters
    ----------
    gunw_dir : path
        Directory containing NISAR GUNW .h5 files.
    min_coherence : float
        Minimum coherence threshold.
    track : int or None
        If specified, only load products from this track number.
    crop_to_site : (easting, northing, radius_m) or None
        If specified, crop to this area to reduce memory usage.
    """
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

    # Collect unique dates
    all_dates = sorted({d["ref_date"] for d in ifg_data} | {d["sec_date"] for d in ifg_data})
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    n_dates = len(all_dates)
    n_ifg = len(ifg_data)

    # Use first interferogram's grid
    n_rows, n_cols = ifg_data[0]["unwrapped_phase"].shape
    x_coords = ifg_data[0]["x_coords"]
    y_coords = ifg_data[0]["y_coords"]

    logger.info("SBAS: %d ifg → %d dates (%s to %s), grid %d×%d",
               n_ifg, n_dates, all_dates[0], all_dates[-1], n_rows, n_cols)

    # NISAR L-band wavelength
    wavelength = 0.2384  # meters

    # Build design matrix and data stack
    G = np.zeros((n_ifg, n_dates - 1))
    phase_stack = np.zeros((n_ifg, n_rows, n_cols), dtype=np.float32)
    coh_stack = np.zeros((n_ifg, n_rows, n_cols), dtype=np.float32)

    for k, d in enumerate(ifg_data):
        ref_idx = date_to_idx[d["ref_date"]]
        sec_idx = date_to_idx[d["sec_date"]]
        for j in range(ref_idx, sec_idx):
            G[k, j] = 1.0
        phase_stack[k] = d["unwrapped_phase"]
        coh_stack[k] = d["coherence"]

    # Phase to LOS displacement
    disp_stack = -phase_stack * wavelength / (4 * np.pi)

    # Mean coherence for masking
    mean_coh = np.nanmean(coh_stack, axis=0)
    valid_mask = mean_coh >= min_coherence
    n_valid = np.sum(valid_mask)

    logger.info("Valid pixels: %d of %d (%.1f%%, coh≥%.2f)",
               n_valid, valid_mask.size, 100*n_valid/valid_mask.size, min_coherence)

    # SBAS inversion (vectorized)
    displacement = np.full((n_dates, n_rows, n_cols), np.nan, dtype=np.float32)
    displacement[0, :, :] = 0  # reference date

    if n_valid > 0:
        d_flat = disp_stack[:, valid_mask]  # [n_ifg, n_valid]
        w_flat = coh_stack[:, valid_mask] ** 2
        valid_idx = np.argwhere(valid_mask)

        # Use lstsq which handles rank-deficient systems (temporal gaps)
        # Vectorized: solve for all pixels at once
        # G @ inc = d, solve for inc via weighted least squares
        W_diag = np.mean(w_flat, axis=1)  # average weight per ifg
        W_sqrt = np.sqrt(np.maximum(W_diag, 1e-10))
        Gw = G * W_sqrt[:, None]  # weighted design matrix

        logger.info("  Inverting %d pixels (lstsq, rank G=%d/%d)...",
                    n_valid, np.linalg.matrix_rank(G), n_dates - 1)

        # Batch solve: weight each observation row
        dw = d_flat * W_sqrt[:, None]
        inc_all, residuals, rank, sv = np.linalg.lstsq(Gw, dw, rcond=None)
        # inc_all: [n_dates-1, n_valid]

        cum_disp = np.cumsum(inc_all, axis=0)  # [n_dates-1, n_valid]
        for px in range(n_valid):
            r, c = valid_idx[px]
            displacement[1:, r, c] = cum_disp[:, px]

        logger.info("  Inversion complete (rank=%d)", rank)

    # Build coordinate grids
    lon_grid, lat_grid = np.meshgrid(x_coords, y_coords)

    # Velocity
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
        metadata={"wavelength_m": wavelength, "n_interferograms": n_ifg,
                  "sensor": "NISAR", "band": "L", "track": track},
    )


def load_nisar_goff_stack(
    goff_dir: str | Path,
    min_correlation: float = 0.1,
    track: int | None = None,
    crop_to_site: tuple[float, float, float] | None = None,
) -> NISARTimeseries:
    """Load NISAR GOFF (offset field) products."""
    goff_dir = Path(goff_dir)
    goff_files = sorted(goff_dir.glob("NISAR_*GOFF*.h5"))

    if not goff_files:
        raise FileNotFoundError(f"No NISAR GOFF files in {goff_dir}")

    if track is not None:
        goff_files = [f for f in goff_files if f"_{track:03d}_" in f.name]

    logger.info("Loading %d NISAR GOFF products", len(goff_files))

    offsets = []
    for f in goff_files:
        try:
            data = _read_goff(f, crop_to_site)
            if data is not None:
                offsets.append(data)
                logger.info("  %s → %s: %s",
                           data["ref_date"], data["sec_date"],
                           data["range_offset"].shape)
        except Exception as e:
            logger.warning("  Failed %s: %s", f.name[:50], e)

    if not offsets:
        raise ValueError("No valid GOFF products loaded")

    all_dates = sorted({d["ref_date"] for d in offsets} | {d["sec_date"] for d in offsets})
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    n_dates = len(all_dates)
    n_off = len(offsets)

    n_rows, n_cols = offsets[0]["range_offset"].shape
    x_coords = offsets[0]["x_coords"]
    y_coords = offsets[0]["y_coords"]

    G = np.zeros((n_off, n_dates - 1))
    disp_stack = np.zeros((n_off, n_rows, n_cols), dtype=np.float32)
    corr_stack = np.zeros((n_off, n_rows, n_cols), dtype=np.float32)

    for k, d in enumerate(offsets):
        ref_idx = date_to_idx[d["ref_date"]]
        sec_idx = date_to_idx[d["sec_date"]]
        for j in range(ref_idx, sec_idx):
            G[k, j] = 1.0
        disp_stack[k] = d["range_offset"]
        corr_stack[k] = d["correlation"]

    mean_corr = np.nanmean(corr_stack, axis=0)
    valid_mask = mean_corr >= min_correlation
    n_valid = np.sum(valid_mask)

    displacement = np.full((n_dates, n_rows, n_cols), np.nan, dtype=np.float32)
    displacement[0, :, :] = 0

    if n_valid > 0:
        valid_idx = np.argwhere(valid_mask)
        d_flat = disp_stack[:, valid_mask]
        w_flat = corr_stack[:, valid_mask] ** 2

        W_diag = np.mean(w_flat, axis=1)
        W_sqrt = np.sqrt(np.maximum(W_diag, 1e-10))
        Gw = G * W_sqrt[:, None]
        dw = d_flat * W_sqrt[:, None]

        logger.info("  Inverting %d pixels...", n_valid)
        inc_all = np.linalg.lstsq(Gw, dw, rcond=None)[0]
        cum_disp = np.cumsum(inc_all, axis=0)
        for px in range(n_valid):
            r, c = valid_idx[px]
            displacement[1:, r, c] = cum_disp[:, px]

    lon_grid, lat_grid = np.meshgrid(x_coords, y_coords)
    dates_ord = np.array([datetime.strptime(d, "%Y%m%d").toordinal() for d in all_dates])
    velocity = _compute_velocity(dates_ord, displacement)

    return NISARTimeseries(
        dates=dates_ord,
        date_strings=all_dates,
        displacement=displacement,
        velocity=velocity,
        coherence=mean_corr,
        latitude=lat_grid,
        longitude=lon_grid,
        source="GOFF",
        metadata={"n_offsets": n_off, "sensor": "NISAR", "method": "pixel_offset_tracking"},
    )


def _read_gunw(filepath: Path, crop: tuple | None = None) -> dict | None:
    """Read a single NISAR GUNW HDF5 file."""
    with h5py.File(filepath, "r") as f:
        base = "science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram"

        # Find polarization (HH, HV, etc.)
        pol_group = f[base]
        pol = None
        for key in pol_group.keys():
            if len(key) == 2 and key.isalpha():
                pol = key
                break

        if pol is None:
            return None

        unw = f[f"{base}/{pol}/unwrappedPhase"][:]
        coh = f[f"{base}/{pol}/coherenceMagnitude"][:]
        x_coords = f[f"{base}/{pol}/xCoordinates"][:]
        y_coords = f[f"{base}/{pol}/yCoordinates"][:]

        # Crop if requested (easting, northing, radius_m)
        if crop is not None:
            cx, cy, radius = crop
            col_mask = (x_coords >= cx - radius) & (x_coords <= cx + radius)
            row_mask = (y_coords >= cy - radius) & (y_coords <= cy + radius)
            if np.sum(col_mask) < 10 or np.sum(row_mask) < 10:
                return None
            unw = unw[np.ix_(row_mask, col_mask)]
            coh = coh[np.ix_(row_mask, col_mask)]
            x_coords = x_coords[col_mask]
            y_coords = y_coords[row_mask]

        # Replace fill values
        unw = np.where(unw == 0, np.nan, unw)

        # Extract dates from filename
        # Pattern: ..._YYYYMMDDT..._YYYYMMDDT..._YYYYMMDDT..._YYYYMMDDT..._
        parts = filepath.stem.split("_")
        date_parts = [p[:8] for p in parts if len(p) >= 15 and p[0] == "2" and "T" in p]
        # First two are ref start/end, second two are sec start/end
        if len(date_parts) >= 4:
            ref_date = date_parts[0]
            sec_date = date_parts[2]
        elif len(date_parts) >= 2:
            ref_date = date_parts[0]
            sec_date = date_parts[1]
        else:
            return None

        return {
            "ref_date": ref_date,
            "sec_date": sec_date,
            "unwrapped_phase": unw.astype(np.float32),
            "coherence": coh.astype(np.float32),
            "x_coords": x_coords,
            "y_coords": y_coords,
        }


def _read_goff(filepath: Path, crop: tuple | None = None) -> dict | None:
    """Read a single NISAR GOFF HDF5 file."""
    with h5py.File(filepath, "r") as f:
        base = "science/LSAR/GOFF/grids/frequencyA/pixelOffsets"

        # Find polarization
        pol_group = f[base]
        pol = None
        for key in pol_group.keys():
            if len(key) == 2 and key.isalpha():
                pol = key
                break
        if pol is None:
            return None

        # Use layer1 (finest resolution)
        layer = f"{base}/{pol}/layer1"
        range_off = f[f"{layer}/slantRangeOffset"][:]
        corr = f[f"{layer}/correlationSurfacePeak"][:]
        x_coords = f[f"{layer}/xCoordinates"][:]
        y_coords = f[f"{layer}/yCoordinates"][:]

        # Crop
        if crop is not None:
            cx, cy, radius = crop
            col_mask = (x_coords >= cx - radius) & (x_coords <= cx + radius)
            row_mask = (y_coords >= cy - radius) & (y_coords <= cy + radius)
            if np.sum(col_mask) < 5 or np.sum(row_mask) < 5:
                return None
            range_off = range_off[np.ix_(row_mask, col_mask)]
            corr = corr[np.ix_(row_mask, col_mask)]
            x_coords = x_coords[col_mask]
            y_coords = y_coords[row_mask]

        range_off = np.where(range_off == 0, np.nan, range_off)

        parts = filepath.stem.split("_")
        date_parts = [p[:8] for p in parts if len(p) >= 15 and p[0] == "2" and "T" in p]
        if len(date_parts) >= 4:
            ref_date = date_parts[0]
            sec_date = date_parts[2]
        elif len(date_parts) >= 2:
            ref_date = date_parts[0]
            sec_date = date_parts[1]
        else:
            return None

        return {
            "ref_date": ref_date,
            "sec_date": sec_date,
            "range_offset": range_off.astype(np.float32),
            "correlation": corr.astype(np.float32),
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
