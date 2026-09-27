#!/usr/bin/env python3
"""
Aru Glacier twin collapses (2016) — synthetic retrospective analysis.

This is a SYNTHETIC RETROSPECTIVE validation: NISAR was not operational
in 2016, so no real InSAR displacement data exist for this event.
Instead, we construct synthetic displacement time series based on
published literature values for pre-collapse velocity changes, then
run the GLEWS detection pipeline to evaluate whether the system would
have detected the precursory acceleration in time.

Events:
  - Aru-1 collapse: 17 July 2016 — ~68 Mm3 ice avalanche, 9 killed
  - Aru-2 collapse: 21 September 2016 — ~83 Mm3 from adjacent glacier

Location: 34.0 N, 82.2 E, Aru Range, Rutog County, western Tibet

Key references:
  - Kaab et al. 2018 (Nature Geoscience): velocity increased from
    ~1 m/yr to ~20 m/yr over ~2 months before first collapse; low-angle
    (~15 deg) bed; subglacial meltwater pressurization on fine-grained
    till, amplified by permafrost degradation.
  - Tian et al. 2017: surging behavior detected in satellite imagery.
  - Gilbert et al. 2018: thermomechanical modeling of Aru collapses.

Limitations:
  - No real NISAR data for 2016 — all displacement values are synthetic.
  - Synthetic time series use idealized noise and seasonal models; real
    InSAR data would have spatially correlated atmospheric noise,
    coherence loss, and viewing geometry effects.
  - The 12-day repeat used here mirrors NISAR's design; Landsat/radar
    data used in actual studies had different cadences.
  - Results demonstrate detection capability on the documented signal
    magnitude, not an independent validation.
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from glews.timeseries import (
    AccelerationMap,
    compute_acceleration_map,
    bocpd_changepoints,
    detect_step_changes,
    fit_voight,
)
from glews.detect import detect_anomalies, AnomalyFlag
from glews.cascade import (
    assess_cascade_risk,
    estimate_runout,
    estimate_exposure,
    generate_exposure_summary,
)

DATA_DIR = Path(__file__).resolve().parent
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "aru_2016.yaml"

# Aru glacier coordinates
SITE_LAT = 34.00
SITE_LON = 82.20

# Event dates
COLLAPSE_1 = date(2016, 7, 17)   # Aru-1: ~68 Mm3
COLLAPSE_2 = date(2016, 9, 21)   # Aru-2: ~83 Mm3

# Published velocity values (Kaab et al. 2018)
BACKGROUND_VELOCITY_M_YR = 1.0     # steady-state creep
PEAK_VELOCITY_M_YR = 20.0          # just before collapse
ACCELERATION_ONSET_DAYS = 60       # ~2 months before first collapse

# Grid dimensions for synthetic data (small tile around glacier)
N_ROWS = 40
N_COLS = 40
PIXEL_SIZE_DEG = 0.001  # ~100 m at 34 N


def load_config() -> dict:
    """Load site configuration."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def generate_synthetic_dates(
    start: date, end: date, repeat_days: int = 12,
) -> np.ndarray:
    """
    Generate acquisition dates at NISAR-like 12-day repeat cadence.

    Returns ordinal-day array.
    """
    dates = []
    current = start
    while current <= end:
        dates.append(current.toordinal())
        current += timedelta(days=repeat_days)
    return np.array(dates, dtype=float)


def generate_synthetic_displacement(
    dates: np.ndarray,
    collapse_date: date,
    n_rows: int = N_ROWS,
    n_cols: int = N_COLS,
) -> np.ndarray:
    """
    Generate synthetic LOS displacement matching Aru pre-collapse behavior.

    The displacement model has four components:
    1. Background creep at ~1 m/yr (steady-state)
    2. Seasonal signal (annual + semi-annual harmonics, ~5 mm amplitude)
    3. Pre-collapse acceleration from ~1 m/yr to ~20 m/yr over 2 months
    4. Spatially correlated noise (~2 mm RMS per epoch)

    The acceleration follows an exponential ramp starting ~60 days before
    the collapse, reaching peak velocity just before failure. This matches
    the Kaab et al. 2018 observations from Landsat and SAR data.

    The "glacier" occupies a central region of the grid (rows 10-30,
    cols 10-30); surrounding pixels show only atmospheric noise.

    Parameters
    ----------
    dates : np.ndarray
        Acquisition dates as ordinal days.
    collapse_date : date
        Date of the collapse event.
    n_rows, n_cols : int
        Grid dimensions.

    Returns
    -------
    np.ndarray
        Synthetic displacement, shape [n_dates, n_rows, n_cols], in meters.
    """
    n_dates = len(dates)
    collapse_ord = collapse_date.toordinal()
    rng = np.random.RandomState(20160717)  # reproducible

    displacement = np.zeros((n_dates, n_rows, n_cols))

    # Define glacier mask: central 20x20 pixels
    glacier_mask = np.zeros((n_rows, n_cols), dtype=bool)
    glacier_mask[10:30, 10:30] = True

    # Spatial weight: Gaussian tapering from glacier center
    row_center, col_center = 20, 20
    rows_all, cols_all = np.where(glacier_mask)
    dist_from_center = np.sqrt(
        (rows_all - row_center) ** 2 + (cols_all - col_center) ** 2
    )
    spatial_weight = np.exp(-dist_from_center ** 2 / (2 * 8.0 ** 2))

    # Time relative to first date
    t0 = dates[0]
    t_years = (dates - t0) / 365.25

    for i in range(n_dates):
        t_yr = t_years[i]
        current_ord = dates[i]

        # 1. Background creep (m/yr * yr = meters)
        background_disp = BACKGROUND_VELOCITY_M_YR * t_yr

        # 2. Seasonal component (~5 mm amplitude annual, ~2 mm semi-annual)
        seasonal = (
            0.005 * np.sin(2 * np.pi * t_yr)
            + 0.002 * np.sin(4 * np.pi * t_yr)
        )

        # 3. Pre-collapse acceleration
        days_before_collapse = collapse_ord - current_ord
        accel_disp = 0.0
        k = np.log(PEAK_VELOCITY_M_YR / BACKGROUND_VELOCITY_M_YR)
        duration_yr = ACCELERATION_ONSET_DAYS / 365.25

        if 0 <= days_before_collapse <= ACCELERATION_ONSET_DAYS:
            # Fraction of acceleration period elapsed
            frac = 1.0 - days_before_collapse / ACCELERATION_ONSET_DAYS
            # Cumulative displacement from acceleration onset:
            # integral of v0*exp(k*s) ds from 0 to frac, times duration
            accel_disp = (
                BACKGROUND_VELOCITY_M_YR * duration_yr / k
                * (np.exp(k * frac) - 1.0)
            )
        elif days_before_collapse < 0:
            # Post-collapse: clamp at peak displacement
            accel_disp = (
                BACKGROUND_VELOCITY_M_YR * duration_yr / k
                * (np.exp(k) - 1.0)
            )

        # Apply to glacier pixels with spatial weighting
        glacier_signal = background_disp + seasonal + accel_disp
        displacement[i, glacier_mask] = glacier_signal * spatial_weight

        # 4. Noise: spatially correlated atmospheric + measurement
        atmo = gaussian_filter(rng.randn(n_rows, n_cols) * 0.003, sigma=5)
        meas = rng.randn(n_rows, n_cols) * 0.001
        displacement[i] += atmo + meas

    return displacement


def generate_coordinate_grids() -> tuple[np.ndarray, np.ndarray]:
    """Generate lat/lon grids for the synthetic tile."""
    lat = np.linspace(
        SITE_LAT - N_ROWS / 2 * PIXEL_SIZE_DEG,
        SITE_LAT + N_ROWS / 2 * PIXEL_SIZE_DEG,
        N_ROWS,
    )
    lon = np.linspace(
        SITE_LON - N_COLS / 2 * PIXEL_SIZE_DEG,
        SITE_LON + N_COLS / 2 * PIXEL_SIZE_DEG,
        N_COLS,
    )
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    return lat_grid, lon_grid


def run_acceleration_detection(
    dates: np.ndarray,
    displacement: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    config: dict,
) -> tuple[AccelerationMap, list[AnomalyFlag]]:
    """Run the full acceleration + anomaly detection pipeline."""
    det_cfg = config["detect"]

    print("  Computing acceleration map...")
    accel_map = compute_acceleration_map(
        dates,
        displacement,
        window_size_days=det_cfg["acceleration"]["window_size_days"],
        step_days=det_cfg["acceleration"]["step_days"],
        n_harmonics=det_cfg.get("n_harmonics", 2),
    )

    print(f"  Acceleration map: {accel_map.acceleration_zscore.shape[0]} windows")
    max_z = np.nanmax(np.abs(accel_map.acceleration_zscore))
    print(f"  Max |z-score|: {max_z:.1f}")

    print("  Running anomaly detection...")
    flags = detect_anomalies(
        accel_map, lat_grid, lon_grid, config,
        dates=dates, displacement=displacement,
    )
    print(f"  Detected {len(flags)} anomaly flags")

    return accel_map, flags


def run_bocpd_on_glacier_pixel(
    dates: np.ndarray,
    displacement: np.ndarray,
) -> dict:
    """
    Run BOCPD on the central glacier pixel to detect the acceleration onset.

    Returns a dict with changepoint indices and probabilities.
    """
    # Central glacier pixel
    pixel_disp = displacement[:, 20, 20]

    print("  Running BOCPD on central glacier pixel...")
    result = bocpd_changepoints(
        dates,
        pixel_disp,
        hazard_rate=1 / 50,
        threshold=0.25,
    )

    cp_dates = []
    for idx in result.changepoint_indices:
        if idx < len(dates):
            d = date.fromordinal(int(dates[idx]))
            cp_dates.append(d.isoformat())
            print(f"    Changepoint at {d.isoformat()} "
                  f"(prob={result.changepoint_probabilities[idx]:.3f})")

    return {
        "changepoint_dates": cp_dates,
        "changepoint_indices": result.changepoint_indices.tolist(),
        "max_probability": float(np.max(result.changepoint_probabilities)),
    }


def run_voight_analysis(
    dates: np.ndarray,
    displacement: np.ndarray,
    collapse_date: date,
) -> dict | None:
    """
    Run Voight's inverse-velocity analysis on the central glacier pixel.

    Returns dict with predicted failure date and R-squared, or None.
    """
    pixel_disp = displacement[:, 20, 20]

    # Compute velocity from displacement differences
    dt = np.diff(dates)
    dt[dt == 0] = 1.0
    velocity = np.abs(np.diff(pixel_disp) / dt) * 365.25  # m/yr
    vel_dates = dates[1:]

    # Use only pre-collapse data
    collapse_ord = collapse_date.toordinal()
    pre_mask = vel_dates < collapse_ord
    if np.sum(pre_mask) < 5:
        return None

    print("  Running Voight inverse-velocity analysis...")
    result = fit_voight(
        vel_dates[pre_mask],
        velocity[pre_mask],
        min_points=5,
    )

    if result is not None:
        pred_date = date.fromordinal(int(result["predicted_failure_date"]))
        actual_date = collapse_date
        error_days = (pred_date - actual_date).days
        print(f"    Predicted failure: {pred_date.isoformat()}")
        print(f"    Actual collapse:   {actual_date.isoformat()}")
        print(f"    Error: {error_days:+d} days")
        print(f"    R-squared: {result['r_squared']:.3f}")

        return {
            "predicted_failure_date": pred_date.isoformat(),
            "actual_collapse_date": actual_date.isoformat(),
            "prediction_error_days": error_days,
            "r_squared": round(result["r_squared"], 4),
            "inverse_velocity_slope": round(result["inverse_velocity_slope"], 6),
        }
    else:
        print("    Voight fit failed (insufficient acceleration or poor fit)")
        return None


def run_cascade_assessment(
    flags: list[AnomalyFlag],
    config: dict,
) -> list[dict]:
    """Run cascade risk assessment on detected flags."""
    print("\n  Running cascade risk assessment...")
    assessments = []

    for flag in flags[:5]:  # top 5 flags
        assessment = assess_cascade_risk(flag, config=config)

        print(f"    Flag {flag.flag_id}: risk={assessment.risk_level.value}, "
              f"volume={assessment.estimated_volume_m3:,.0f} m3, "
              f"runout={assessment.runout_distance_m / 1000:.1f} km, "
              f"pop={assessment.population_exposed}")

        assessments.append({
            "flag_id": flag.flag_id,
            "risk_level": assessment.risk_level.value,
            "estimated_volume_m3": round(assessment.estimated_volume_m3),
            "volume_sufficient": assessment.volume_sufficient,
            "slope_angle_deg": round(assessment.slope_angle_deg, 1),
            "elevation_m": round(assessment.elevation_m),
            "runout_distance_km": round(assessment.runout_distance_m / 1000, 1),
            "runout_reaches_river": assessment.runout_reaches_river,
            "angle_of_reach_deg": round(assessment.angle_of_reach_deg, 1),
            "population_exposed": assessment.population_exposed,
            "dam_potential": assessment.dam_potential,
            "cascade_score": round(assessment.cascade_score, 3),
        })

    return assessments


def estimate_advance_warning(
    flags: list[AnomalyFlag],
    bocpd_result: dict,
    collapse_date: date,
    dates: np.ndarray,
) -> dict:
    """
    Estimate the advance warning time from various detection methods.

    Returns a dict with days of warning from each method.
    """
    collapse_ord = collapse_date.toordinal()
    result = {}

    # From acceleration z-score flags
    if flags:
        earliest_flag_date = None
        for f in flags:
            flag_ord = f.window_date
            if flag_ord < collapse_ord:
                flag_date = date.fromordinal(int(flag_ord))
                if earliest_flag_date is None or flag_date < earliest_flag_date:
                    earliest_flag_date = flag_date

        if earliest_flag_date is not None:
            warning_days = (collapse_date - earliest_flag_date).days
            result["acceleration_zscore"] = {
                "first_detection_date": earliest_flag_date.isoformat(),
                "advance_warning_days": warning_days,
            }

    # From BOCPD
    if bocpd_result["changepoint_dates"]:
        for cp_date_str in bocpd_result["changepoint_dates"]:
            cp_date = date.fromisoformat(cp_date_str)
            if cp_date < collapse_date:
                warning_days = (collapse_date - cp_date).days
                result["bocpd"] = {
                    "first_detection_date": cp_date_str,
                    "advance_warning_days": warning_days,
                }
                break

    return result


def analyze_collapse(
    label: str,
    collapse_date: date,
    volume_mm3: float,
    config: dict,
    dates: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
) -> dict:
    """Full analysis pipeline for one collapse event."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  Collapse date: {collapse_date.isoformat()}")
    print(f"  Volume: ~{volume_mm3:.0f} Mm3")
    print(f"{'=' * 60}")

    # Generate synthetic displacement for this collapse
    print("\n  Generating synthetic displacement time series...")
    displacement = generate_synthetic_displacement(dates, collapse_date)
    print(f"  Grid: {displacement.shape[1]} x {displacement.shape[2]} pixels, "
          f"{displacement.shape[0]} epochs")

    # Central glacier pixel displacement summary
    center_disp = displacement[:, 20, 20]
    print(f"\n  Central pixel displacement range: "
          f"[{np.min(center_disp) * 1000:.1f}, "
          f"{np.max(center_disp) * 1000:.1f}] mm")

    # Print displacement at key dates
    print(f"  Displacement at selected dates:")
    collapse_ord = collapse_date.toordinal()
    for i, d_ord in enumerate(dates):
        d = date.fromordinal(int(d_ord))
        days_to_collapse = collapse_ord - int(d_ord)
        if (days_to_collapse in [120, 90, 60, 30, 12, 0]
                or abs(days_to_collapse) <= 1):
            print(f"    {d.isoformat()} (T-{days_to_collapse:>3d} days): "
                  f"{center_disp[i] * 1000:>+10.1f} mm")

    # 1. Acceleration detection
    accel_map, flags = run_acceleration_detection(
        dates, displacement, lat_grid, lon_grid, config,
    )

    # 2. BOCPD changepoint detection
    bocpd_result = run_bocpd_on_glacier_pixel(dates, displacement)

    # 3. Voight analysis
    voight_result = run_voight_analysis(dates, displacement, collapse_date)

    # 4. Advance warning estimate
    warning = estimate_advance_warning(flags, bocpd_result, collapse_date, dates)

    # 5. Cascade assessment
    cascade_results = run_cascade_assessment(flags, config)

    # Compile results
    result = {
        "collapse_date": collapse_date.isoformat(),
        "volume_mm3": volume_mm3,
        "volume_m3": volume_mm3 * 1e6,
        "n_epochs": len(dates),
        "date_range": [
            date.fromordinal(int(dates[0])).isoformat(),
            date.fromordinal(int(dates[-1])).isoformat(),
        ],
        "detection_results": {
            "acceleration_zscore": {
                "n_flags": len(flags),
                "max_zscore": round(float(np.nanmax(np.abs(
                    accel_map.acceleration_zscore
                ))), 2) if flags else 0.0,
                "flags": [
                    {
                        "flag_id": f.flag_id,
                        "score": round(f.score, 3),
                        "peak_zscore": round(f.peak_zscore, 2),
                        "area_m2": round(f.area_m2),
                        "detection_tag": f.detection_details.get("tag", "unknown"),
                        "window_date": date.fromordinal(
                            int(f.window_date)
                        ).isoformat() if np.isfinite(f.window_date) else None,
                    }
                    for f in flags[:10]
                ],
            },
            "bocpd": bocpd_result,
            "voight": voight_result,
        },
        "advance_warning": warning,
        "cascade_assessment": cascade_results,
    }

    # Print summary
    print(f"\n  --- Detection Summary ---")
    print(f"  Acceleration flags: {len(flags)}")
    if warning.get("acceleration_zscore"):
        w = warning["acceleration_zscore"]
        print(f"  Earliest acceleration flag: {w['first_detection_date']} "
              f"({w['advance_warning_days']} days before collapse)")
    if warning.get("bocpd"):
        w = warning["bocpd"]
        print(f"  BOCPD changepoint: {w['first_detection_date']} "
              f"({w['advance_warning_days']} days before collapse)")
    if voight_result:
        print(f"  Voight prediction: {voight_result['predicted_failure_date']} "
              f"(error: {voight_result['prediction_error_days']:+d} days)")

    return result


def main():
    config = load_config()
    lat_grid, lon_grid = generate_coordinate_grids()

    # Observation window: Jan 2016 to Nov 2016 (covers both collapses)
    start_date = date(2016, 1, 1)
    end_date = date(2016, 11, 30)
    dates = generate_synthetic_dates(start_date, end_date, repeat_days=12)

    print(f"Aru Glacier Twin Collapses -- Synthetic Retrospective Analysis")
    print(f"=" * 60)
    print(f"IMPORTANT: This analysis uses SYNTHETIC data, not real NISAR")
    print(f"observations. NISAR was not operational in 2016. Synthetic")
    print(f"displacement is constructed from published literature values")
    print(f"(Kaab et al. 2018, Tian et al. 2017).")
    print(f"=" * 60)
    print(f"\nObservation period: {start_date} to {end_date}")
    print(f"Acquisition cadence: 12 days (NISAR-like)")
    print(f"Number of epochs: {len(dates)}")
    print(f"Synthetic grid: {N_ROWS} x {N_COLS} pixels "
          f"(~{N_ROWS * PIXEL_SIZE_DEG * 111:.0f} m x "
          f"{N_COLS * PIXEL_SIZE_DEG * 111:.0f} m)")

    # Analyze both collapses
    aru1_result = analyze_collapse(
        "ARU-1 COLLAPSE (17 July 2016)",
        COLLAPSE_1,
        volume_mm3=68.0,
        config=config,
        dates=dates,
        lat_grid=lat_grid,
        lon_grid=lon_grid,
    )

    aru2_result = analyze_collapse(
        "ARU-2 COLLAPSE (21 September 2016)",
        COLLAPSE_2,
        volume_mm3=83.0,
        config=config,
        dates=dates,
        lat_grid=lat_grid,
        lon_grid=lon_grid,
    )

    # Compile full results
    results = {
        "site": {
            "name": "Aru Glaciers, Western Tibet",
            "latitude": SITE_LAT,
            "longitude": SITE_LON,
            "description": (
                "Twin glacier collapses in the Aru Range. Low-angle "
                "(~10-15 deg bed slope) glacier failures unprecedented "
                "in the observational record. Subglacial meltwater "
                "pressurization on fine-grained till, amplified by "
                "permafrost degradation."
            ),
        },
        "event_dates": {
            "aru_1": COLLAPSE_1.isoformat(),
            "aru_2": COLLAPSE_2.isoformat(),
        },
        "location": {
            "latitude": SITE_LAT,
            "longitude": SITE_LON,
            "region": "Aru Range, Rutog County, Ngari Prefecture, Tibet",
        },
        "validation_type": "synthetic_retrospective",
        "generated": datetime.now().isoformat(),
        "limitations": [
            "No real NISAR data for 2016 -- all displacement values are synthetic.",
            "Synthetic time series use idealized noise; real InSAR data would have "
            "spatially correlated atmospheric artifacts and coherence loss.",
            "12-day repeat cadence mirrors NISAR design but differs from Landsat "
            "and other sensors used in original studies.",
            "Results demonstrate detection capability on documented signal "
            "magnitude, not an independent discovery.",
            "Glacier geometry and spatial extent are approximated; real glaciers "
            "have complex shapes and variable surface properties.",
            "Permafrost and subglacial conditions cannot be observed by InSAR "
            "alone -- complementary data would be needed operationally.",
        ],
        "literature_references": [
            {
                "key": "Kaab2018",
                "citation": (
                    "Kaab, A., Leinss, S., Gilbert, A., et al. (2018). "
                    "Massive collapse of two glaciers in western Tibet in 2016 "
                    "after surge-like instability. Nature Geoscience, 11, 114-120."
                ),
                "doi": "10.1038/s41561-017-0039-7",
                "used_for": "Pre-collapse velocity values (1 -> 20 m/yr over ~2 months)",
            },
            {
                "key": "Tian2017",
                "citation": (
                    "Tian, L., Yao, T., Gao, Y., et al. (2017). Two glaciers "
                    "collapsed in western Tibet. Journal of Glaciology, 63(237), "
                    "194-197."
                ),
                "doi": "10.1017/jog.2016.122",
                "used_for": "Surging behavior detection in satellite imagery",
            },
            {
                "key": "Gilbert2018",
                "citation": (
                    "Gilbert, A., Leinss, S., Kargel, J., et al. (2018). "
                    "Mechanisms leading to the 2016 giant twin glacier collapses, "
                    "Aru Range, Tibet. The Cryosphere, 12, 2883-2900."
                ),
                "doi": "10.5194/tc-12-2883-2018",
                "used_for": "Thermomechanical collapse mechanism",
            },
        ],
        "event_context": {
            "bed_angle_deg": "10-15 (unusually flat for collapse)",
            "volumes": {
                "aru_1_m3": 68_000_000,
                "aru_2_m3": 83_000_000,
                "total_m3": 151_000_000,
            },
            "debris_runout_km": 8.5,
            "fatalities": 9,
            "livestock_killed": "hundreds",
            "mechanism": (
                "Subglacial meltwater pressurization on fine-grained, "
                "low-permeability till bed. Permafrost degradation beneath "
                "glacier increased basal water input. The low bed angle "
                "meant large areas were near the threshold for sliding; "
                "once destabilized, the entire glacier tongue mobilized."
            ),
            "climate_link": (
                "Permafrost degradation due to regional warming. "
                "Increased meltwater penetration to glacier bed. "
                "These collapses may represent a new failure mode "
                "enabled by climate change."
            ),
        },
        "synthetic_parameters": {
            "background_velocity_m_yr": BACKGROUND_VELOCITY_M_YR,
            "peak_velocity_m_yr": PEAK_VELOCITY_M_YR,
            "acceleration_onset_days_before_collapse": ACCELERATION_ONSET_DAYS,
            "acquisition_cadence_days": 12,
            "grid_size": f"{N_ROWS}x{N_COLS}",
            "pixel_size_deg": PIXEL_SIZE_DEG,
            "noise_model": (
                "spatially correlated atmospheric (3 mm RMS) "
                "+ measurement (1 mm RMS)"
            ),
        },
        "collapses": {
            "aru_1": aru1_result,
            "aru_2": aru2_result,
        },
    }

    # Estimated advance warning summary
    print(f"\n{'=' * 60}")
    print(f"  OVERALL ASSESSMENT")
    print(f"{'=' * 60}")

    all_warnings = []
    for collapse_key in ["aru_1", "aru_2"]:
        collapse_data = results["collapses"][collapse_key]
        for method, w in collapse_data.get("advance_warning", {}).items():
            all_warnings.append({
                "collapse": collapse_key,
                "method": method,
                "advance_warning_days": w["advance_warning_days"],
                "detection_date": w["first_detection_date"],
            })

    results["estimated_advance_warning_days"] = all_warnings

    if all_warnings:
        max_warning = max(w["advance_warning_days"] for w in all_warnings)
        min_warning = min(w["advance_warning_days"] for w in all_warnings)
        print(f"\n  Advance warning range: {min_warning}-{max_warning} days")
        print(f"  (across all methods and both collapses)")
        for w in all_warnings:
            print(f"    {w['collapse']} / {w['method']}: "
                  f"{w['advance_warning_days']} days "
                  f"(detected {w['detection_date']})")
    else:
        print(f"\n  WARNING: No advance detection achieved.")
        print(f"  This may indicate the synthetic signal is too weak or")
        print(f"  the detection thresholds are too conservative.")

    # Cascade summary
    print(f"\n  Cascade Assessment:")
    for collapse_key in ["aru_1", "aru_2"]:
        cascade = results["collapses"][collapse_key].get(
            "cascade_assessment", []
        )
        if cascade:
            top = cascade[0]
            print(f"    {collapse_key}: risk={top['risk_level']}, "
                  f"volume={top['estimated_volume_m3']:,.0f} m3, "
                  f"runout={top['runout_distance_km']:.1f} km")

    # Assessment of GLEWS capability
    detected_aru1 = bool(
        results["collapses"]["aru_1"]["detection_results"]
        ["acceleration_zscore"]["n_flags"]
    )
    detected_aru2 = bool(
        results["collapses"]["aru_2"]["detection_results"]
        ["acceleration_zscore"]["n_flags"]
    )

    if detected_aru1 and detected_aru2:
        verdict = "PASS"
        verdict_detail = (
            "GLEWS detection pipeline successfully identified precursory "
            "acceleration in synthetic data matching the Aru collapse "
            "observations for both events."
        )
    elif detected_aru1 or detected_aru2:
        verdict = "PARTIAL"
        verdict_detail = (
            "GLEWS detection pipeline detected precursory acceleration for "
            "one of the two collapses. Further tuning may improve sensitivity."
        )
    else:
        verdict = "FAIL"
        verdict_detail = (
            "GLEWS detection pipeline did not flag the synthetic precursory "
            "signals. Detection thresholds may need adjustment for this "
            "failure mode (low-angle glacier surge)."
        )

    results["validation_verdict"] = {
        "result": verdict,
        "detail": verdict_detail,
        "note": (
            "This is a synthetic validation -- the system detected a signal "
            "it was designed to detect. Real-world performance depends on "
            "actual InSAR data quality, atmospheric conditions, and temporal "
            "sampling."
        ),
    }

    print(f"\n  Validation verdict: {verdict}")
    print(f"  {verdict_detail}")

    # Save results
    out_path = DATA_DIR / "aru_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
