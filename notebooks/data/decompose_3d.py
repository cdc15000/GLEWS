#!/usr/bin/env python3
"""
3D displacement decomposition from ascending + descending NISAR GOFF data.

Combines slant-range and along-track offsets from two look directions
to solve for east, north, and vertical displacement components at the
Nepal 2026 collapse site.

Theory:
  For each SAR pass, the measured offsets relate to 3D displacement via:
    slant_range_offset = u_e·d_e + u_n·d_n + u_u·d_u
    along_track_offset = a_e·d_e + a_n·d_n

  With ascending + descending, we have 4 equations for 3 unknowns → least squares.
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from glews.nisar import load_nisar_goff_stack

# Site parameters
SITE_E, SITE_N = 392042.0, 3119847.0
COLLAPSE_DATE = date(2026, 8, 26)
CROP_RADIUS = 1500.0  # meters

# SAR geometry parameters for NISAR at ~28°N
# (approximate mid-swath values for this latitude)
INCIDENCE_ANGLE_DEG = 37.0  # typical NISAR L-band mid-swath
ASC_HEADING_DEG = 348.0     # ascending orbit heading (CW from north)
DESC_HEADING_DEG = 192.0    # descending orbit heading

theta = np.radians(INCIDENCE_ANGLE_DEG)


def los_unit_vector(heading_deg: float, left_looking: bool = True) -> np.ndarray:
    """
    Line-of-sight unit vector in ENU (east, north, up) coordinates.
    Points from ground toward satellite.
    """
    look_az = np.radians(heading_deg - 90 if left_looking else heading_deg + 90)
    return np.array([
        np.sin(theta) * np.sin(look_az),  # east
        np.sin(theta) * np.cos(look_az),  # north
        np.cos(theta),                     # up
    ])


def azimuth_unit_vector(heading_deg: float) -> np.ndarray:
    """Along-track unit vector in ENU (only east and north components)."""
    h = np.radians(heading_deg)
    return np.array([
        np.sin(h),   # east
        np.cos(h),   # north
        0.0,         # no vertical sensitivity
    ])


def build_design_matrix() -> np.ndarray:
    """4×3 design matrix mapping [d_e, d_n, d_u] to [LOS_asc, LOS_desc, AZI_asc, AZI_desc]."""
    los_asc = los_unit_vector(ASC_HEADING_DEG)
    los_desc = los_unit_vector(DESC_HEADING_DEG)
    azi_asc = azimuth_unit_vector(ASC_HEADING_DEG)
    azi_desc = azimuth_unit_vector(DESC_HEADING_DEG)
    return np.vstack([los_asc, los_desc, azi_asc, azi_desc])


def extract_site_timeseries(data_dir: str, track: int):
    """Load GOFF stack and extract site-patch mean displacement + along-track."""
    import h5py

    ts = load_nisar_goff_stack(
        data_dir, track=track,
        crop_to_site=(SITE_E, SITE_N, CROP_RADIUS),
        min_correlation=0.05,
    )

    # Find nearest pixel
    dist2 = (ts.longitude - SITE_E)**2 + (ts.latitude - SITE_N)**2
    ri, ci = np.unravel_index(np.argmin(dist2), dist2.shape)

    # Patch mean slant-range displacement (already in ts.displacement from SBAS)
    hw = 2
    patch = ts.displacement[:, max(0,ri-hw):ri+hw+1, max(0,ci-hw):ci+hw+1]
    range_disp = np.nanmean(patch.reshape(len(ts.dates), -1), axis=1)

    # Also need along-track offsets — extract from raw files and do SBAS
    # The GOFF loader uses slantRangeOffset; we need to also extract alongTrackOffset
    along_track_disp = _extract_along_track_sbas(
        data_dir, track, ts, ri, ci, hw
    )

    return ts.dates, ts.date_strings, range_disp, along_track_disp


def _extract_along_track_sbas(data_dir, track, ts, ri, ci, hw):
    """Extract along-track offsets from GOFF HDF5 files and do SBAS inversion."""
    import h5py
    from pathlib import Path

    goff_dir = Path(data_dir)
    files = sorted(goff_dir.glob("NISAR_L2_PR_GOFF_*.h5"))

    # Filter by track
    track_files = []
    for f in files:
        parts = f.name.split("_")
        try:
            t = int(parts[5])
            if t == track:
                track_files.append(f)
        except (IndexError, ValueError):
            pass

    if not track_files:
        return np.full(len(ts.dates), np.nan)

    # Parse date pairs from filenames (same as _read_goff)
    date_pairs = []
    for f in track_files:
        parts = f.name.split("_")
        d1 = parts[11][:8]  # reference date
        d2 = parts[13][:8]  # secondary date
        date_pairs.append((d1, d2, f))

    # Build SBAS system for along-track
    all_dates = sorted(set(d for d1, d2, _ in date_pairs for d in [d1, d2]))
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    n_dates = len(all_dates)
    n_pairs = len(date_pairs)

    # Extract along-track offsets for each pair at the site patch
    at_path = "science/LSAR/GOFF/grids/frequencyA/pixelOffsets/HH/layer1/alongTrackOffset"

    offsets = []
    A = np.zeros((n_pairs, n_dates))

    for k, (d1, d2, fpath) in enumerate(date_pairs):
        try:
            with h5py.File(fpath, "r") as hf:
                # Read coordinates to find crop indices
                x = hf["science/LSAR/GOFF/grids/frequencyA/pixelOffsets/HH/layer1/xCoordinates"][:]
                y = hf["science/LSAR/GOFF/grids/frequencyA/pixelOffsets/HH/layer1/yCoordinates"][:]

                # Crop same as ts
                x_mask = (x >= SITE_E - CROP_RADIUS) & (x <= SITE_E + CROP_RADIUS)
                y_mask = (y >= SITE_N - CROP_RADIUS) & (y <= SITE_N + CROP_RADIUS)

                if not np.any(x_mask) or not np.any(y_mask):
                    offsets.append(np.nan)
                    continue

                xi = np.where(x_mask)[0]
                yi = np.where(y_mask)[0]
                x0, x1 = xi[0], xi[-1] + 1
                y0, y1 = yi[0], yi[-1] + 1

                at = hf[at_path][y0:y1, x0:x1]
                corr = hf["science/LSAR/GOFF/grids/frequencyA/pixelOffsets/HH/layer1/correlationSurfacePeak"][y0:y1, x0:x1]

                at = np.where(corr > 0.05, at, np.nan)

                # Patch mean at same relative position as ri, ci
                # Use the center of the cropped array
                r_mid = at.shape[0] // 2
                c_mid = at.shape[1] // 2
                patch = at[max(0,r_mid-hw):r_mid+hw+1, max(0,c_mid-hw):c_mid+hw+1]
                val = float(np.nanmean(patch))
                offsets.append(val)

                A[k, date_to_idx[d1]] = -1
                A[k, date_to_idx[d2]] = 1
        except Exception as e:
            print(f"  Warning: along-track extraction failed for {fpath.name}: {e}")
            offsets.append(np.nan)

    offsets = np.array(offsets)
    valid = np.isfinite(offsets)

    if valid.sum() < 2:
        return np.full(len(ts.dates), np.nan)

    # Pin first date to zero, solve
    A_valid = A[valid, 1:]
    b_valid = offsets[valid]

    disp_at, _, _, _ = np.linalg.lstsq(A_valid, b_valid, rcond=None)
    disp_at = np.concatenate([[0.0], disp_at])  # prepend reference = 0

    # Map to ts.dates order
    result = np.full(len(ts.dates), np.nan)
    ts_date_set = {d: i for i, d in enumerate(ts.date_strings)}
    for d, val in zip(all_dates, disp_at):
        if d in ts_date_set:
            result[ts_date_set[d]] = val

    return result


def main():
    DATA_DIR = Path(__file__).resolve().parent
    ASC_DIR = DATA_DIR / "nisar" / "goff"
    DESC_DIR = DATA_DIR / "nisar" / "goff_desc"

    print("Loading ascending (track 098)...")
    asc_dates, asc_dstr, asc_range, asc_azimuth = extract_site_timeseries(str(ASC_DIR), 98)

    print("Loading descending (track 048)...")
    desc_dates, desc_dstr, desc_range, desc_azimuth = extract_site_timeseries(str(DESC_DIR), 48)

    print(f"\nAscending dates: {asc_dstr}")
    print(f"Descending dates: {desc_dstr}")

    # Print raw measurements
    print("\n=== Ascending (Track 098) ===")
    print(f"{'Date':<12} {'Range (mm)':>12} {'Azimuth (mm)':>14}")
    for d, r, a in zip(asc_dstr, asc_range*1000, asc_azimuth*1000):
        print(f"  {d:<12} {r:>+12.1f} {a:>+14.1f}")

    print("\n=== Descending (Track 048) ===")
    print(f"{'Date':<12} {'Range (mm)':>12} {'Azimuth (mm)':>14}")
    for d, r, a in zip(desc_dstr, desc_range*1000, desc_azimuth*1000):
        print(f"  {d:<12} {r:>+12.1f} {a:>+14.1f}")

    # Find common dates for decomposition
    common_dates = sorted(set(asc_dstr) & set(desc_dstr))
    print(f"\nCommon dates: {common_dates}")

    if len(common_dates) < 2:
        print("Not enough common dates for temporal decomposition.")
        print("Using nearest-date interpolation instead...")

    # Build design matrix
    G = build_design_matrix()
    print(f"\nDesign matrix G (4×3):")
    print(f"  {'':>15} {'East':>8} {'North':>8} {'Up':>8}")
    labels = ["LOS_asc", "LOS_desc", "AZI_asc", "AZI_desc"]
    for label, row in zip(labels, G):
        print(f"  {label:>15} {row[0]:>+8.3f} {row[1]:>+8.3f} {row[2]:>+8.3f}")

    # For decomposition, use the LAST available date from each track
    # (closest to collapse) and solve for 3D displacement
    results = {}

    # Use the latest shared period
    # Ascending last date, descending last date
    asc_last_idx = len(asc_dstr) - 1
    desc_last_idx = len(desc_dstr) - 1

    # Total displacement at end of series
    obs = np.array([
        asc_range[asc_last_idx],
        desc_range[desc_last_idx],
        asc_azimuth[asc_last_idx],
        desc_azimuth[desc_last_idx],
    ])

    print(f"\nFinal observations (m):")
    print(f"  Ascending  range:   {obs[0]:+.4f}")
    print(f"  Descending range:   {obs[1]:+.4f}")
    print(f"  Ascending  azimuth: {obs[2]:+.4f}")
    print(f"  Descending azimuth: {obs[3]:+.4f}")

    # Solve: G·d = obs  →  d = (G^T G)^(-1) G^T obs
    d_3d, residuals, rank, sv = np.linalg.lstsq(G, obs, rcond=None)

    print(f"\n╔══════════════════════════════════════════╗")
    print(f"║   3D DISPLACEMENT AT COLLAPSE SITE       ║")
    print(f"╠══════════════════════════════════════════╣")
    print(f"║  East:     {d_3d[0]*1000:>+10.0f} mm ({d_3d[0]:>+.3f} m)  ║")
    print(f"║  North:    {d_3d[1]*1000:>+10.0f} mm ({d_3d[1]:>+.3f} m)  ║")
    print(f"║  Vertical: {d_3d[2]*1000:>+10.0f} mm ({d_3d[2]:>+.3f} m)  ║")
    print(f"╚══════════════════════════════════════════╝")

    magnitude = np.sqrt(d_3d[0]**2 + d_3d[1]**2 + d_3d[2]**2)
    horizontal = np.sqrt(d_3d[0]**2 + d_3d[1]**2)
    azimuth_deg = np.degrees(np.arctan2(d_3d[0], d_3d[1]))

    print(f"\n  3D magnitude:   {magnitude*1000:.0f} mm ({magnitude:.3f} m)")
    print(f"  Horizontal:     {horizontal*1000:.0f} mm  (azimuth {azimuth_deg:.0f}°)")
    print(f"  H/V ratio:      {horizontal/abs(d_3d[2]):.2f}" if d_3d[2] != 0 else "")

    residual_rms = float(np.sqrt(np.mean(residuals))) if len(residuals) > 0 else 0
    print(f"  Residual RMS:   {residual_rms*1000:.1f} mm")

    # Temporal decomposition: solve for each common date
    print(f"\n=== Temporal 3D Decomposition ===")
    decomposed = []

    # For each ascending date, find the nearest descending date
    asc_ords = np.array([datetime.strptime(d, "%Y%m%d").toordinal() for d in asc_dstr])
    desc_ords = np.array([datetime.strptime(d, "%Y%m%d").toordinal() for d in desc_dstr])

    for i, (d_asc, ord_asc) in enumerate(zip(asc_dstr, asc_ords)):
        # Find nearest descending date within 12 days
        dt = np.abs(desc_ords - ord_asc)
        j = np.argmin(dt)
        if dt[j] > 15:
            continue

        obs_t = np.array([
            asc_range[i],
            desc_range[j],
            asc_azimuth[i],
            desc_azimuth[j],
        ])

        if not np.all(np.isfinite(obs_t)):
            continue

        d3, res, _, _ = np.linalg.lstsq(G, obs_t, rcond=None)

        print(f"  {d_asc} (desc≈{desc_dstr[j]}): "
              f"E={d3[0]*1000:>+8.0f} N={d3[1]*1000:>+8.0f} U={d3[2]*1000:>+8.0f} mm")

        decomposed.append({
            "asc_date": d_asc,
            "desc_date": desc_dstr[j],
            "east_mm": round(float(d3[0] * 1000), 1),
            "north_mm": round(float(d3[1] * 1000), 1),
            "up_mm": round(float(d3[2] * 1000), 1),
            "magnitude_mm": round(float(np.sqrt(d3[0]**2 + d3[1]**2 + d3[2]**2) * 1000), 1),
        })

    # Save results
    output = {
        "site": {"easting": SITE_E, "northing": SITE_N, "epsg": 32645},
        "collapse_date": COLLAPSE_DATE.isoformat(),
        "generated": datetime.now().isoformat(),
        "geometry": {
            "incidence_angle_deg": INCIDENCE_ANGLE_DEG,
            "ascending_heading_deg": ASC_HEADING_DEG,
            "descending_heading_deg": DESC_HEADING_DEG,
            "ascending_track": 98,
            "descending_track": 48,
        },
        "final_3d_displacement": {
            "asc_date": asc_dstr[-1],
            "desc_date": desc_dstr[-1],
            "east_m": round(float(d_3d[0]), 4),
            "north_m": round(float(d_3d[1]), 4),
            "vertical_m": round(float(d_3d[2]), 4),
            "magnitude_m": round(float(magnitude), 4),
            "horizontal_m": round(float(horizontal), 4),
            "azimuth_deg": round(float(azimuth_deg), 1),
        },
        "temporal_decomposition": decomposed,
        "ascending_range_series_mm": {d: round(float(v*1000), 1) for d, v in zip(asc_dstr, asc_range)},
        "descending_range_series_mm": {d: round(float(v*1000), 1) for d, v in zip(desc_dstr, desc_range)},
        "ascending_azimuth_series_mm": {d: round(float(v*1000), 1) for d, v in zip(asc_dstr, asc_azimuth)},
        "descending_azimuth_series_mm": {d: round(float(v*1000), 1) for d, v in zip(desc_dstr, desc_azimuth)},
    }

    out_path = DATA_DIR / "decomposition_3d.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
