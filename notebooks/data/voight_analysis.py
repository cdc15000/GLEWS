#!/usr/bin/env python3
"""
Fit Voight's inverse-velocity failure law to NISAR GUNW and GOFF
displacement time series at the Nepal 2026 collapse site (track 98).

Site: easting=392042, northing=3119847 (EPSG:32645, UTM 45N)
Collapse date: 2026-08-26
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from glews.nisar import load_nisar_gunw_stack, load_nisar_goff_stack, NISARTimeseries
from glews.timeseries import fit_voight

DATA_DIR = Path(__file__).resolve().parent
GUNW_DIR = DATA_DIR / "nisar" / "gunw"
GOFF_DIR = DATA_DIR / "nisar" / "goff"
OUT_JSON = DATA_DIR / "voight_results.json"

SITE_E = 392042.0
SITE_N = 3119847.0
COLLAPSE_DATE = date(2026, 8, 26)
COLLAPSE_ORDINAL = COLLAPSE_DATE.toordinal()

TRACK = 98
CROP_RADIUS_M = 1500.0  # generous crop for site search


def ordinal_to_str(o: float) -> str:
    o_int = int(round(o))
    try:
        return date.fromordinal(o_int).isoformat()
    except Exception:
        return str(o)


def nearest_pixel(ts: NISARTimeseries, easting: float, northing: float) -> tuple[int, int, float]:
    """Find the pixel nearest the site coordinates. Returns (row, col, distance_m)."""
    dist2 = (ts.longitude - easting) ** 2 + (ts.latitude - northing) ** 2
    dist2 = np.where(np.isfinite(ts.coherence), dist2, np.inf)
    idx = np.unravel_index(np.argmin(dist2), dist2.shape)
    row, col = idx
    dist = float(np.sqrt(dist2[row, col]))
    return row, col, dist


def patch_mean_displacement(
    ts: NISARTimeseries, row: int, col: int, half_win: int = 2
) -> np.ndarray:
    """Average displacement over a small patch centered at (row, col), per date."""
    n_dates, n_rows, n_cols = ts.displacement.shape
    r0, r1 = max(0, row - half_win), min(n_rows, row + half_win + 1)
    c0, c1 = max(0, col - half_win), min(n_cols, col + half_win + 1)
    patch = ts.displacement[:, r0:r1, c0:c1]
    with np.errstate(invalid="ignore"):
        return np.nanmean(patch.reshape(n_dates, -1), axis=1)


def sliding_window_velocity(
    dates: np.ndarray, displacement: np.ndarray, min_pts: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute velocity between consecutive valid epochs (classic Fukuzono/Voight
    inverse-velocity setup: v_i = (d_{i+1} - d_i) / (t_{i+1} - t_i), assigned
    to the midpoint time). Returns (window_center_dates, velocity_m_per_yr).
    """
    valid = np.isfinite(displacement)
    t = dates[valid].astype(float)
    d = displacement[valid]
    if len(t) < min_pts + 1:
        return np.array([]), np.array([])

    centers = []
    vels = []
    for i in range(len(t) - 1):
        dt_days = t[i + 1] - t[i]
        if dt_days <= 0:
            continue
        dd = d[i + 1] - d[i]
        v = dd / dt_days * 365.25  # m/yr
        centers.append((t[i] + t[i + 1]) / 2.0)
        vels.append(v)
    return np.array(centers), np.array(vels)


def analyze_source(
    label: str, ts: NISARTimeseries, focus_start: str | None = "20260701"
) -> dict:
    row, col, dist_m = nearest_pixel(ts, SITE_E, SITE_N)
    disp = patch_mean_displacement(ts, row, col, half_win=2)

    date_strings = ts.date_strings
    dates = ts.dates

    n_valid = int(np.sum(np.isfinite(disp)))
    print(f"[{label}] nearest pixel ({row},{col}) dist={dist_m:.0f} m, "
          f"{n_valid}/{len(disp)} valid dates")

    total_disp = float(np.nanmax(disp) - np.nanmin(disp)) if n_valid > 0 else float("nan")
    print(f"[{label}] total displacement range at site patch: {total_disp:.3f} m")

    centers, vels = sliding_window_velocity(dates, disp)
    abs_vels = np.abs(vels)

    # Use magnitude of velocity (direction sign depends on LOS geometry / offset sign)
    result_all = fit_voight(centers, abs_vels, min_points=5) if len(centers) else None

    # Focus window: acceleration phase (Jul-Aug 2026) if requested
    result_focus = None
    focus_centers = focus_vels = np.array([])
    if focus_start is not None and len(centers):
        t_focus_start = datetime.strptime(focus_start, "%Y%m%d").toordinal()
        mask = centers >= t_focus_start
        focus_centers = centers[mask]
        focus_vels = abs_vels[mask]
        if len(focus_centers) >= 3:
            result_focus = fit_voight(focus_centers, focus_vels, min_points=3)

    def fmt_result(r):
        if r is None:
            return None
        return {
            "predicted_failure_date": ordinal_to_str(r["predicted_failure_date"]),
            "predicted_failure_ordinal": float(r["predicted_failure_date"]),
            "r_squared": float(r["r_squared"]),
            "inverse_velocity_slope": float(r["inverse_velocity_slope"]),
            "days_until_failure_from_last_obs": float(r["days_until_failure"]),
        }

    out = {
        "source": label,
        "track": TRACK,
        "pixel_row": int(row),
        "pixel_col": int(col),
        "distance_to_site_m": dist_m,
        "n_dates_total": int(len(disp)),
        "n_dates_valid": n_valid,
        "date_strings": date_strings,
        "total_displacement_range_m": total_disp,
        "velocity_series": {
            "dates": [ordinal_to_str(c) for c in centers],
            "dates_ordinal": centers.tolist(),
            "velocity_m_per_yr": vels.tolist(),
            "abs_velocity_m_per_yr": abs_vels.tolist(),
        },
        "voight_fit_full_series": fmt_result(result_all),
        "voight_fit_focus_window": fmt_result(result_focus),
        "focus_window_start": focus_start,
        "focus_window_n_points": int(len(focus_centers)),
    }

    for tag, r in (("full-series", result_all), ("focus-window", result_focus)):
        if r is not None:
            pred = ordinal_to_str(r["predicted_failure_date"])
            err_days = r["predicted_failure_date"] - COLLAPSE_ORDINAL
            print(f"[{label}] Voight fit ({tag}): predicted failure {pred}, "
                  f"R²={r['r_squared']:.3f}, error vs actual collapse = {err_days:+.1f} days")
        else:
            print(f"[{label}] Voight fit ({tag}): no valid fit")

    return out


def main():
    results = {
        "site": {"easting": SITE_E, "northing": SITE_N, "epsg": 32645},
        "collapse_date": COLLAPSE_DATE.isoformat(),
        "track": TRACK,
        "generated": datetime.now().isoformat(),
        "sources": {},
    }

    crop = (SITE_E, SITE_N, CROP_RADIUS_M)

    # --- GOFF (primary: larger, clearer signal) ---
    try:
        goff_ts = load_nisar_goff_stack(GOFF_DIR, track=TRACK, crop_to_site=crop, min_correlation=0.05)
        results["sources"]["GOFF"] = analyze_source("GOFF", goff_ts)
    except Exception as e:
        print(f"GOFF analysis failed: {e}")
        results["sources"]["GOFF"] = {"error": str(e)}

    # --- GUNW (secondary) ---
    try:
        gunw_ts = load_nisar_gunw_stack(GUNW_DIR, track=TRACK, crop_to_site=crop, min_coherence=0.2)
        results["sources"]["GUNW"] = analyze_source("GUNW", gunw_ts)
    except Exception as e:
        print(f"GUNW analysis failed: {e}")
        results["sources"]["GUNW"] = {"error": str(e)}

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults written to {OUT_JSON}")


if __name__ == "__main__":
    main()
