#!/usr/bin/env python3
"""
Weisshorn (Randa) hanging glacier analysis using NISAR GOFF data.

This is a LIVE FORECASTING test — the Weisshorn hanging glacier is
currently under surveillance (as of Sep 2026) with repeated partial
ice break-offs. Unlike the Nepal 2026 case, this is NOT retrospective.

Tracks:
  - T138 descending (4 pairs: Jul 5 – Sep 3, 2026)
  - T72 ascending (4 pairs: Jul 1 – Aug 30, 2026)

Site: 46.1064°N, 7.7169°E, elevation 3800-4500 m
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from glews.nisar import load_nisar_goff_stack
from glews.timeseries import compute_acceleration_map
from glews.detect import detect_anomalies

DATA_DIR = Path(__file__).resolve().parent
T138_DIR = DATA_DIR / "nisar" / "weisshorn" / "t138_d"
T72_DIR = DATA_DIR / "nisar" / "weisshorn" / "t72_a"

# Weisshorn hanging glacier coordinates (WGS84 → approximate UTM 32N)
SITE_LAT = 46.1064
SITE_LON = 7.7169
CROP_RADIUS = 2000.0  # meters

# Known ice break-off date
BREAKOFF_DATE = date(2026, 8, 24)


def analyze_track(label: str, data_dir: Path, track: int, direction: str):
    """Run full GOFF pipeline for one track."""
    if not data_dir.exists() or not list(data_dir.glob("*.h5")):
        print(f"[{label}] No data in {data_dir}")
        return None

    print(f"\n{'='*60}")
    print(f"  {label} — Track {track} {direction}")
    print(f"{'='*60}")

    # Load and invert GOFF stack
    # Need to determine site coords in the product's CRS
    # GOFF products are geocoded — check if UTM or lat/lon
    import h5py
    sample = list(data_dir.glob("NISAR_L2_PR_GOFF_*.h5"))[0]
    with h5py.File(sample, "r") as hf:
        proj_path = "science/LSAR/GOFF/grids/frequencyA/pixelOffsets/HH/layer1/projection"
        if proj_path in hf:
            epsg = int(hf[proj_path][()])
            print(f"  Product EPSG: {epsg}")
        else:
            epsg = 32632  # assume UTM 32N for Switzerland

        x = hf["science/LSAR/GOFF/grids/frequencyA/pixelOffsets/HH/layer1/xCoordinates"][:]
        y = hf["science/LSAR/GOFF/grids/frequencyA/pixelOffsets/HH/layer1/yCoordinates"][:]
        print(f"  Grid extent: x=[{x[0]:.0f}, {x[-1]:.0f}], y=[{y[0]:.0f}, {y[-1]:.0f}]")

    # Convert site lat/lon to UTM if needed
    if epsg >= 32600:
        # UTM projection — convert lat/lon to easting/northing
        # Simple formula for UTM zone 32N
        from math import radians, sin, cos, tan, sqrt
        lat_r = radians(SITE_LAT)
        lon_r = radians(SITE_LON)
        # UTM zone 32 central meridian = 9°E
        zone = (epsg - 32600) if epsg < 32700 else (epsg - 32700)
        cm = radians(zone * 6 - 183)

        # Simplified UTM conversion (accurate to ~1m for Switzerland)
        a = 6378137.0
        f = 1 / 298.257223563
        e2 = 2 * f - f * f
        N_val = a / sqrt(1 - e2 * sin(lat_r)**2)
        T = tan(lat_r)**2
        C = e2 / (1 - e2) * cos(lat_r)**2
        A_val = cos(lat_r) * (lon_r - cm)
        M = a * ((1 - e2/4 - 3*e2**2/64) * lat_r
                 - (3*e2/8 + 3*e2**2/32) * sin(2*lat_r)
                 + (15*e2**2/256) * sin(4*lat_r))

        site_e = 500000 + 0.9996 * N_val * (A_val + (1-T+C)*A_val**3/6)
        site_n = 0.9996 * (M + N_val * tan(lat_r) * (A_val**2/2 + (5-T+9*C+4*C**2)*A_val**4/24))
    else:
        site_e, site_n = SITE_LON, SITE_LAT

    print(f"  Site coordinates in CRS: E={site_e:.0f}, N={site_n:.0f}")

    # Check if site is within grid
    if site_e < x[0] - CROP_RADIUS or site_e > x[-1] + CROP_RADIUS:
        print(f"  WARNING: Site easting {site_e:.0f} outside grid range [{x[0]:.0f}, {x[-1]:.0f}]")
    if site_n < y[-1] - CROP_RADIUS or site_n > y[0] + CROP_RADIUS:
        print(f"  WARNING: Site northing {site_n:.0f} outside grid range [{y[-1]:.0f}, {y[0]:.0f}]")

    try:
        ts = load_nisar_goff_stack(
            str(data_dir), track=track,
            crop_to_site=(site_e, site_n, CROP_RADIUS),
            min_correlation=0.05,
        )
    except Exception as e:
        print(f"  GOFF stack loading failed: {e}")
        return None

    print(f"  Dates: {ts.date_strings}")
    print(f"  Grid: {ts.displacement.shape[1]} × {ts.displacement.shape[2]} pixels")
    print(f"  Valid pixels: {np.sum(np.isfinite(ts.velocity)):,}")

    # Site displacement time series
    dist2 = (ts.longitude - site_e)**2 + (ts.latitude - site_n)**2
    ri, ci = np.unravel_index(np.argmin(dist2), dist2.shape)
    site_dist = float(np.sqrt(dist2[ri, ci]))
    print(f"  Nearest pixel to site: ({ri},{ci}), distance={site_dist:.0f} m")

    hw = 2
    patch = ts.displacement[:, max(0,ri-hw):ri+hw+1, max(0,ci-hw):ci+hw+1]
    disp = np.nanmean(patch.reshape(len(ts.dates), -1), axis=1)

    print(f"\n  Displacement at glacier site:")
    for d, v in zip(ts.date_strings, disp * 1000):
        print(f"    {d}: {v:>+10.1f} mm")

    total_disp = float(np.nanmax(np.abs(disp))) * 1000
    print(f"\n  Maximum displacement: {total_disp:.0f} mm")

    # Velocity map
    vel = ts.velocity
    vel_valid = vel[np.isfinite(vel)]
    if len(vel_valid) > 0:
        print(f"  Velocity range: [{np.nanmin(vel_valid)*1000:.0f}, {np.nanmax(vel_valid)*1000:.0f}] mm/yr")
        print(f"  Velocity std:   {np.nanstd(vel_valid)*1000:.0f} mm/yr")

    # Run anomaly detection
    print(f"\n  Running anomaly detection...")

    # Load config
    import yaml
    config_path = Path(__file__).resolve().parents[2] / "config" / "weisshorn_2026.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    det_cfg = cfg["detect"]
    try:
        accel_map = compute_acceleration_map(
            ts.dates,
            ts.displacement,
            window_size_days=det_cfg["acceleration"]["window_size_days"],
            step_days=det_cfg["acceleration"]["step_days"],
            n_harmonics=det_cfg.get("n_harmonics", 2),
        )

        flags = detect_anomalies(accel_map, ts.latitude, ts.longitude, cfg)
        print(f"  Detected {len(flags)} anomaly flags")

        for f_obj in flags[:10]:
            dist_to_site = np.sqrt((f_obj.center_lat - site_n)**2 + (f_obj.center_lon - site_e)**2)
            print(f"    Flag {f_obj.flag_id}: z={f_obj.peak_zscore:.1f}, "
                  f"area={f_obj.area_m2:,.0f} m², "
                  f"dist_to_glacier={dist_to_site:.0f} m")
    except Exception as e:
        print(f"  Anomaly detection failed: {e}")
        flags = []

    return {
        "track": track,
        "direction": direction,
        "label": label,
        "dates": ts.date_strings,
        "n_pixels_valid": int(np.sum(np.isfinite(ts.velocity))),
        "site_displacement_mm": {d: round(float(v*1000), 1) for d, v in zip(ts.date_strings, disp)},
        "max_displacement_mm": round(total_disp, 1),
        "n_flags": len(flags),
        "flags": [
            {
                "flag_id": f.flag_id,
                "peak_zscore": round(float(f.peak_zscore), 2),
                "area_m2": round(float(f.area_m2), 0),
                "center": [float(f.center_lat), float(f.center_lon)],
            }
            for f in flags[:20]
        ],
    }


def main():
    results = {
        "site": {
            "name": "Weisshorn East Hanging Glacier (Randa)",
            "lat": SITE_LAT,
            "lon": SITE_LON,
            "breakoff_date": BREAKOFF_DATE.isoformat(),
        },
        "generated": datetime.now().isoformat(),
        "tracks": {},
    }

    # Analyze each track
    for label, data_dir, track, direction in [
        ("T138 Descending", T138_DIR, 138, "descending"),
        ("T72 Ascending", T72_DIR, 72, "ascending"),
    ]:
        result = analyze_track(label, data_dir, track, direction)
        if result:
            results["tracks"][label] = result

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")

    total_flags = sum(r.get("n_flags", 0) for r in results["tracks"].values())
    print(f"  Total anomaly flags: {total_flags}")

    for label, r in results["tracks"].items():
        print(f"  {label}: {r['n_flags']} flags, max disp = {r['max_displacement_mm']:.0f} mm")

    if total_flags > 0:
        print(f"\n  ✅ GLEWS DETECTS ANOMALOUS MOTION AT WEISSHORN")
        print(f"     This validates the pipeline on a second, independent site.")
    else:
        print(f"\n  ⚠️  No flags detected. Possible reasons:")
        print(f"     - Motion too small for GOFF resolution (~1-3 m)")
        print(f"     - Too few epochs for robust SBAS inversion")
        print(f"     - Hanging glacier below min_cluster_pixels threshold")

    out_path = DATA_DIR / "weisshorn_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
