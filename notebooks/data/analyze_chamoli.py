#!/usr/bin/env python3
"""
Chamoli 2021 rock-ice avalanche — synthetic retrospective validation.

On 7 February 2021, a rock-ice avalanche detached from the north face
of Ronti Peak (~30.38N, 79.73E, ~5,600 m) in the Chamoli district of
Uttarakhand, India.  The collapse released approximately 27 million m3
of rock and glacier ice, generating a catastrophic debris flow that
traveled >15 km down the Rishiganga and Dhauliganga valleys, killing
204 people and destroying two hydropower projects.

NISAR launched in 2024 — no NISAR data exist for this 2021 event.
Sentinel-1 InSAR coverage of the site was intermittent (ascending-only
geometry over steep Himalayan terrain, frequent snow decorrelation).
Retrospective optical analyses (Shugar et al. 2021, Science) identified
progressive crack widening in the months before collapse, but no
published study provides quantified pre-event InSAR displacement rates
at Ronti Peak.

This script therefore constructs a HYPOTHETICAL PRECURSOR SCENARIO:
it generates synthetic displacement time series with an injected
pre-failure acceleration signal whose parameters are physically
plausible assumptions (not measured values), then runs the full GLEWS
detection and cascade risk pipelines on this synthetic data to answer:

    "If GLEWS had been operating with InSAR coverage of this site,
     and the precursor signal matched our assumed scenario, would
     the system have flagged it — and how far in advance?"

The injected signal assumes:
    - 0.5 m/yr background LOS velocity (active rock-glacier creep)
    - Exponential acceleration beginning ~120 days before collapse,
      reaching ~25 m/yr2 — terminal velocity ~15-20 m/yr consistent
      with Voight-type tertiary creep in rock-ice masses
    - Failure wedge ~500 x 250 m (7-pixel radius at ~90 m posting)

These are stated assumptions, not literature-derived measurements.

References:
    Shugar et al. (2021). A massive rock and ice avalanche caused the
        2021 disaster at Chamoli, Indian Himalaya. Science, 373(6552),
        300-306.
    Voight (1988). A method for prediction of volcanic eruptions.
        Nature, 332(6160), 125-130.
    Hungr et al. (2014). The Varnes classification of landslide types,
        an update. Landslides, 11(2), 167-194.
    Scheidegger (1973). On the prediction of the reach and velocity of
        catastrophic landslides. Rock Mechanics, 5(4), 231-236.
"""
from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from glews.cascade import (
    CascadeAssessment,
    assess_cascade_risk,
    estimate_exposure,
    estimate_runout,
    generate_exposure_summary,
)
from glews.detect import AnomalyFlag, detect_anomalies
from glews.process import DisplacementTimeseries
from glews.synthetic import SyntheticConfig, generate_synthetic_scene
from glews.timeseries import compute_acceleration_map

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-20s %(levelname)-7s %(message)s",
)
logger = logging.getLogger("chamoli_validation")

DATA_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "chamoli_2021.yaml"

# ---------------------------------------------------------------------------
# Site parameters
# ---------------------------------------------------------------------------
SITE_LAT = 30.38
SITE_LON = 79.73
EVENT_DATE = date(2021, 2, 7)
FAILURE_ELEVATION_M = 5500.0
SLOPE_ANGLE_DEG = 35.0
WEDGE_LENGTH_M = 500
WEDGE_WIDTH_M = 250
WEDGE_DEPTH_M = 80
LITERATURE_VOLUME_M3 = 27_000_000
RUNOUT_KM_OBSERVED = 27.0  # debris traveled ~27 km (Shugar et al.)


# ---------------------------------------------------------------------------
# 1. ASF search — document the data gap
# ---------------------------------------------------------------------------

def attempt_asf_search() -> dict:
    """
    Attempt to query ASF DAAC for NISAR and Sentinel-1 products
    covering the Chamoli site in the pre-event period.

    NISAR launched in 2024, so no products exist for 2021.
    Sentinel-1 coverage of high Himalayan terrain is limited.
    """
    results = {
        "nisar_goff_count": 0,
        "nisar_gunw_count": 0,
        "sentinel1_count": 0,
        "asf_query_status": "not_attempted",
        "notes": [],
    }

    results["notes"].append(
        "NISAR launched 2024-03-22. No NISAR data exist for the "
        "2021 event period. GOFF/GUNW search skipped."
    )

    # Attempt Sentinel-1 search for pre-event period
    try:
        import asf_search as asf
        from shapely.geometry import Point

        center = Point(SITE_LON, SITE_LAT)
        aoi = center.buffer(0.15)  # ~15 km buffer

        s1_results = asf.search(
            platform=asf.PLATFORM.SENTINEL1,
            processingLevel=asf.PRODUCT_TYPE.SLC,
            beamMode=asf.BEAMMODE.IW,
            intersectsWith=aoi.wkt,
            start="2021-01-01",
            end="2021-02-07",
        )

        results["sentinel1_count"] = len(s1_results)
        results["asf_query_status"] = "success"

        if s1_results:
            tracks = set()
            for r in s1_results:
                tracks.add(r.properties.get("pathNumber", "?"))
            results["sentinel1_tracks"] = sorted(tracks)
            results["notes"].append(
                f"Found {len(s1_results)} Sentinel-1 SLC scenes "
                f"(Jan 2021) on tracks {sorted(tracks)}."
            )
        else:
            results["notes"].append(
                "No Sentinel-1 SLC scenes found for Chamoli "
                "in Jan 2021 search window."
            )

    except ImportError:
        results["asf_query_status"] = "import_error"
        results["notes"].append(
            "asf_search package not installed; Sentinel-1 query skipped."
        )
    except Exception as exc:
        results["asf_query_status"] = f"error: {exc}"
        results["notes"].append(
            f"ASF search failed: {exc}. Proceeding with synthetic data."
        )

    return results


# ---------------------------------------------------------------------------
# 2. Build synthetic displacement time series
# ---------------------------------------------------------------------------

def build_chamoli_synthetic() -> DisplacementTimeseries:
    """
    Generate a synthetic InSAR displacement scene centered on the
    Chamoli/Ronti Peak source zone, with an injected pre-failure
    acceleration signal.

    Covers ~2 years ending on the event date (2021-02-07), with
    12-day revisit (Sentinel-1 cadence).
    """
    # Grid: ~0.2 deg span at ~90 m posting -> ~200x250 pixels
    # center_row/col for the failure zone: place it in the upper
    # portion of the grid (headwall is high on the slope)
    n_rows = 200
    n_cols = 250
    lat_min = SITE_LAT - 0.10
    lat_max = SITE_LAT + 0.10
    lon_min = SITE_LON - 0.10
    lon_max = SITE_LON + 0.10

    # Failure zone center: upper-center of grid
    # Row 0 = lat_max, so row ~60 is high on the slope
    # Wedge is ~500x250 m. At ~90 m pixel size -> radius ~6-7 pixels.
    # Use radius 7 to approximate the 500x250 m footprint.
    center_row = 60
    center_col = 125

    config = SyntheticConfig(
        n_rows=n_rows,
        n_cols=n_cols,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        start_date="2019-02-20",  # ~2 years of history
        end_date="2021-02-07",    # collapse date
        revisit_days=12,
        base_velocity_m_yr=0.5,      # 0.5 m/yr background creep (active rock-glacier)
        velocity_spatial_variation=0.3,
        annual_amplitude_m=0.005,
        semi_annual_amplitude_m=0.002,
        atmospheric_noise_m=0.004,
        measurement_noise_m=0.001,
        base_coherence=0.75,         # lower than default — Himalayan snow
        low_coherence_fraction=0.15,
        failure_zones=[
            {
                "center_row": center_row,
                "center_col": center_col,
                "radius_pixels": 7,  # ~500 m diameter at ~90 m pixels
                "onset_days_before_end": 120,  # acceleration starts ~4 months before
                "max_acceleration_m_yr2": 25.0,  # Voight-type tertiary creep; terminal vel ~15-20 m/yr
                "ramp_type": "exponential",
            },
        ],
    )

    ts = generate_synthetic_scene(config)

    logger.info(
        "Synthetic scene: %d epochs x %d x %d, %s to %s",
        len(ts.dates), ts.displacement.shape[1], ts.displacement.shape[2],
        ts.date_strings[0], ts.date_strings[-1],
    )

    return ts


# ---------------------------------------------------------------------------
# 3. Run GLEWS detection pipeline
# ---------------------------------------------------------------------------

def run_detection(
    ts: DisplacementTimeseries, config: dict, sigma_override: float | None = None
) -> dict:
    """
    Run the full GLEWS Tier 0 detection pipeline on the synthetic data.

    Parameters
    ----------
    ts : DisplacementTimeseries
        Synthetic (or real) displacement time series.
    config : dict
        Parsed site configuration.
    sigma_override : float or None
        If set, overrides the sigma_threshold in the config.

    Returns
    -------
    dict with detection results.
    """
    det_cfg = dict(config)
    if sigma_override is not None:
        det_cfg = json.loads(json.dumps(config))  # deep copy
        det_cfg["detect"]["acceleration"]["sigma_threshold"] = sigma_override

    sigma = det_cfg["detect"]["acceleration"]["sigma_threshold"]

    logger.info("Running detection at sigma=%.1f ...", sigma)

    accel_map = compute_acceleration_map(
        ts.dates,
        ts.displacement,
        window_size_days=det_cfg["detect"]["acceleration"]["window_size_days"],
        step_days=det_cfg["detect"]["acceleration"]["step_days"],
        n_harmonics=det_cfg["detect"].get("n_harmonics", 2),
    )

    flags = detect_anomalies(
        accel_map,
        ts.latitude,
        ts.longitude,
        det_cfg,
        dates=ts.dates,
        displacement=ts.displacement,
    )

    # Identify flags near the injection site.
    # The injected failure zone is radius 7 pixels at ~90 m posting,
    # area ~pi*(7*90)^2 ~ 1.2 Mm2.  We separate:
    #   - "signal flags": near the injection AND plausible area (< 5 Mm2)
    #     AND detected during the acceleration phase (last 120 days)
    #   - "noise flags": near the injection but too large, too early,
    #     or both — these are false positives from background noise
    inject_lat = ts.latitude[60, 125]
    inject_lon = ts.longitude[60, 125]

    # Acceleration onset ordinal
    onset_ordinal = EVENT_DATE.toordinal() - 120

    # The operationally-available date for a detection is
    # window_center + window_size_days/2 (the window must complete
    # before the detection can be reported).
    window_half = det_cfg["detect"]["acceleration"]["window_size_days"] / 2.0
    event_ordinal = EVENT_DATE.toordinal()

    # Max plausible area for the injected zone: the Gaussian taper
    # extends to 3*radius (21 pixels); with a strong signal the full
    # 3-sigma footprint (~pi*(21*90)^2 ≈ 11 Mm2) lights up, so use
    # 5x that to accommodate clustering margin.
    max_signal_area_m2 = 5.0 * np.pi * (21 * 90) ** 2  # ~56 Mm2

    all_near_flags = []
    signal_flags = []
    noise_flags = []

    for f in flags:
        dlat = (f.center_lat - inject_lat) * 111_320
        dlon = (f.center_lon - inject_lon) * 111_320 * np.cos(np.radians(inject_lat))
        dist_m = np.sqrt(dlat**2 + dlon**2)
        if dist_m >= 2000:
            continue

        available_ordinal = f.window_date + window_half

        entry = {
            "flag_id": f.flag_id,
            "score": round(float(f.score), 3),
            "peak_zscore": round(float(f.peak_zscore), 2),
            "area_m2": round(float(f.area_m2), 0),
            "distance_to_injection_m": round(dist_m, 0),
            "window_date": datetime.fromordinal(int(f.window_date)).strftime("%Y-%m-%d"),
            "operationally_available_date": datetime.fromordinal(
                int(available_ordinal)
            ).strftime("%Y-%m-%d"),
            "lead_time_days": round(float(event_ordinal - available_ordinal), 1),
            "detection_tag": f.detection_details.get("tag", "acceleration"),
            "voight_fit": f.voight_fit,
        }

        all_near_flags.append((f, dist_m, entry))

        # Classify: signal if area is plausible AND window is during/after
        # the acceleration phase
        is_in_accel_phase = f.window_date >= onset_ordinal - window_half
        is_plausible_area = f.area_m2 < max_signal_area_m2

        if is_in_accel_phase and is_plausible_area:
            signal_flags.append((f, dist_m, entry))
        else:
            noise_flags.append((f, dist_m, entry))

    # Walk the acceleration z-score map to find earliest crossing
    # in the acceleration phase only
    earliest_crossing = _find_earliest_crossing(
        accel_map, ts, det_cfg, inject_lat, inject_lon, event_ordinal,
        window_half, onset_ordinal,
    )

    # Also find earliest noise crossing (before onset) to characterize
    # false positive rate
    earliest_noise = _find_earliest_crossing(
        accel_map, ts, det_cfg, inject_lat, inject_lon, event_ordinal,
        window_half, onset_ordinal=None,
    )

    # Diagnostic: check raw z-scores at the injection site across windows
    site_zscores = _site_zscore_timeseries(
        accel_map, ts, inject_lat, inject_lon, onset_ordinal
    )

    detected = len(signal_flags) > 0
    signal_lead_times = [e for _, _, e in signal_flags]
    max_lead = max((lt["lead_time_days"] for lt in signal_lead_times), default=0)
    earliest_lead = earliest_crossing.get("lead_time_days") if earliest_crossing else None

    result = {
        "sigma_threshold": sigma,
        "total_flags": len(flags),
        "flags_near_site_total": len(all_near_flags),
        "signal_flags": len(signal_flags),
        "noise_flags_near_site": len(noise_flags),
        "detected": detected,
        "signal_lead_times": signal_lead_times,
        "noise_flags_detail": [e for _, _, e in noise_flags[:5]],
        "max_lead_time_days": round(float(max_lead), 1) if detected else None,
        "earliest_crossing_in_accel_phase": earliest_crossing,
        "earliest_lead_time_days": earliest_lead,
        "earliest_noise_crossing": earliest_noise,
        "site_zscore_diagnostic": site_zscores,
        "false_positive_note": (
            f"{len(noise_flags)} flag(s) near the site are noise "
            f"(pre-onset or oversized clusters). At sigma={sigma}, "
            f"the detector produces {len(flags)} total flags across "
            f"the scene, most of which are false positives."
        ),
    }

    logger.info(
        "  sigma=%.1f: %d total flags, %d signal / %d noise near site, "
        "detected=%s, max_lead=%.0f days, earliest_lead=%s days",
        sigma, len(flags), len(signal_flags), len(noise_flags),
        detected,
        max_lead if detected else 0,
        earliest_lead if earliest_lead is not None else "N/A",
    )

    return result, flags


def _find_earliest_crossing(
    accel_map, ts, config, inject_lat, inject_lon, event_ordinal, window_half,
    onset_ordinal: float | None = None,
) -> dict | None:
    """
    Walk acceleration z-score windows to find the first window where
    the injection zone crosses the sigma threshold with enough pixels
    to form a cluster.

    Parameters
    ----------
    onset_ordinal : float or None
        If set, only consider windows whose center is at or after this
        date (i.e., the acceleration phase).  If None, searches all
        windows (useful for characterizing false-positive rate).
    """
    det = config["detect"]
    sigma = det["acceleration"]["sigma_threshold"]
    min_pixels = det["clustering"]["min_cluster_pixels"]

    n_windows, n_rows, n_cols = accel_map.acceleration_zscore.shape

    # Find pixels near the injection center (within ~700 m, roughly
    # the 3-sigma radius of the injected Gaussian zone)
    dlat = (ts.latitude - inject_lat) * 111_320
    dlon = (ts.longitude - inject_lon) * 111_320 * np.cos(np.radians(inject_lat))
    dist = np.sqrt(dlat**2 + dlon**2)
    near_mask = dist < 700  # tighter radius to focus on the injection zone

    for w in range(n_windows):
        window_center = accel_map.window_centers[w]

        # Skip windows before onset if onset filtering is requested
        if onset_ordinal is not None and window_center < onset_ordinal - window_half:
            continue

        zscore_map = accel_map.acceleration_zscore[w]
        exceeds = (np.abs(zscore_map) > sigma) & near_mask
        n_exceeding = np.count_nonzero(exceeds)

        if n_exceeding >= min_pixels:
            available = window_center + window_half
            lead_days = event_ordinal - available

            return {
                "window_index": int(w),
                "window_date": datetime.fromordinal(int(window_center)).strftime("%Y-%m-%d"),
                "operationally_available_date": datetime.fromordinal(
                    int(available)
                ).strftime("%Y-%m-%d"),
                "lead_time_days": round(float(lead_days), 1),
                "n_pixels_exceeding": int(n_exceeding),
                "peak_zscore": round(float(np.nanmax(np.abs(zscore_map[near_mask]))), 2),
            }

    return None


def _site_zscore_timeseries(
    accel_map, ts, inject_lat, inject_lon, onset_ordinal,
) -> dict:
    """
    Extract the z-score time series at the injection site to diagnose
    whether the signal is present but absorbed into large clusters,
    or genuinely too weak.
    """
    n_windows = accel_map.acceleration_zscore.shape[0]

    # Find the nearest pixel to the injection center
    dlat = np.abs(ts.latitude - inject_lat)
    dlon = np.abs(ts.longitude - inject_lon)
    dist = dlat + dlon  # manhattan distance in degrees, fine for nearest pixel
    ri, ci = np.unravel_index(np.argmin(dist), dist.shape)

    # 3x3 patch around injection center
    hw = 2
    r0 = max(0, ri - hw)
    r1 = min(ts.latitude.shape[0], ri + hw + 1)
    c0 = max(0, ci - hw)
    c1 = min(ts.latitude.shape[1], ci + hw + 1)

    zscores_at_site = []
    accel_at_site = []
    dates_str = []
    in_accel_phase = []

    for w in range(n_windows):
        patch_z = accel_map.acceleration_zscore[w, r0:r1, c0:c1]
        patch_a = accel_map.acceleration[w, r0:r1, c0:c1]
        wc = accel_map.window_centers[w]

        abs_z = np.abs(patch_z)
        if np.all(np.isnan(abs_z)):
            peak_z = 0.0
            mean_z = 0.0
            peak_a = 0.0
        else:
            peak_z = float(np.nanmax(abs_z))
            mean_z = float(np.nanmean(abs_z))
            peak_a = float(patch_a.flat[np.nanargmax(abs_z)])

        zscores_at_site.append(round(peak_z, 3))
        accel_at_site.append(round(peak_a, 6))
        dates_str.append(datetime.fromordinal(int(wc)).strftime("%Y-%m-%d"))
        in_accel_phase.append(bool(wc >= onset_ordinal))

    max_z_overall = max(zscores_at_site) if zscores_at_site else 0
    max_z_in_accel = max(
        (z for z, p in zip(zscores_at_site, in_accel_phase) if p),
        default=0,
    )
    max_z_pre_accel = max(
        (z for z, p in zip(zscores_at_site, in_accel_phase) if not p),
        default=0,
    )

    logger.info(
        "  Site z-score diagnostic: max_overall=%.2f, "
        "max_in_accel_phase=%.2f, max_pre_accel=%.2f",
        max_z_overall, max_z_in_accel, max_z_pre_accel,
    )

    return {
        "pixel_row_col": [int(ri), int(ci)],
        "max_zscore_overall": round(max_z_overall, 3),
        "max_zscore_in_accel_phase": round(max_z_in_accel, 3),
        "max_zscore_pre_accel_phase": round(max_z_pre_accel, 3),
        "n_windows_total": n_windows,
        "n_windows_in_accel_phase": sum(in_accel_phase),
        "zscore_timeseries": [
            {"date": d, "peak_zscore": z, "acceleration_m_yr2": a, "in_accel_phase": p}
            for d, z, a, p in zip(dates_str, zscores_at_site, accel_at_site, in_accel_phase)
        ][-15:],  # last 15 windows for brevity
        "interpretation": (
            f"Peak z-score at injection site: {max_z_in_accel:.2f} "
            f"during acceleration phase vs {max_z_pre_accel:.2f} "
            f"before onset. "
            + (
                "The signal IS distinguishable from noise at the pixel "
                "level but gets absorbed into large spatial clusters by "
                "DBSCAN, preventing focused detection."
                if max_z_in_accel > max_z_pre_accel + 0.5
                else "The signal is NOT clearly distinguishable from "
                "background noise at the assumed acceleration rate."
            )
        ),
    }


# ---------------------------------------------------------------------------
# 4. Cascade risk assessment
# ---------------------------------------------------------------------------

def run_cascade_assessment(
    flags: list[AnomalyFlag], config: dict,
    no_signal_detected: bool = False,
) -> dict:
    """
    Run Tier 1 cascade risk assessment on detected flags,
    plus a literature-informed comparison.
    """
    results = {
        "pipeline_assessments": [],
        "literature_comparison": {},
    }

    if no_signal_detected:
        results["pipeline_assessments_note"] = (
            "No signal flags detected in the acceleration phase. "
            "Pipeline-native cascade assessment skipped because "
            "running it on noise clusters would produce meaningless "
            "volume estimates. See literature_comparison for the "
            "cascade scenario using published event parameters."
        )
        logger.info("  No signal flags — skipping pipeline cascade, "
                     "using literature values only.")
    else:
        # Pipeline-native assessment (synthetic DEM values)
        for f in flags[:5]:  # top 5 by score
            assessment = assess_cascade_risk(flag=f, config=config)
            results["pipeline_assessments"].append({
                "flag_id": f.flag_id,
                "risk_level": assessment.risk_level.value,
                "estimated_volume_m3": round(assessment.estimated_volume_m3, 0),
                "slope_angle_deg": round(assessment.slope_angle_deg, 1),
                "elevation_m": round(assessment.elevation_m, 0),
                "runout_distance_km": round(assessment.runout_distance_m / 1000, 1),
                "population_exposed": assessment.population_exposed,
                "dam_potential": assessment.dam_potential,
                "cascade_score": round(assessment.cascade_score, 3),
                "settlements_exposed": assessment.settlements_exposed,
            })

            logger.info(
                "  Flag %d: risk=%s, vol=%.0f m3, runout=%.1f km, pop=%d",
                f.flag_id,
                assessment.risk_level.value,
                assessment.estimated_volume_m3,
                assessment.runout_distance_m / 1000,
                assessment.population_exposed,
            )

    # Literature-informed runout: use actual observed values
    # Elevation drop: ~5500 m source to ~1400 m at Tapovan = 4100 m
    elevation_drop_m = 4100.0
    horizontal_distance_m = RUNOUT_KM_OBSERVED * 1000

    runout_lit = estimate_runout(
        elevation_drop_m=elevation_drop_m,
        horizontal_distance_m=horizontal_distance_m,
        volume_m3=LITERATURE_VOLUME_M3,
    )

    exposure_lit = estimate_exposure(
        flag_lat=SITE_LAT,
        flag_lon=SITE_LON,
        flag_elevation_m=FAILURE_ELEVATION_M,
        runout_distance_km=runout_lit["runout_distance_km"],
    )

    results["literature_comparison"] = {
        "note": (
            "Runout and exposure computed using literature-reported "
            "values (volume=27 Mm3, elevation drop=4100 m) rather "
            "than pipeline-estimated values. The pipeline's volume "
            "heuristic (depth = 0.1 * sqrt(area)) underestimates "
            "deep-seated rock-ice wedge failures like Chamoli."
        ),
        "volume_m3_literature": LITERATURE_VOLUME_M3,
        "elevation_drop_m": elevation_drop_m,
        "runout_distance_km": round(runout_lit["runout_distance_km"], 1),
        "angle_of_reach_deg": round(runout_lit["angle_of_reach_deg"], 2),
        "observed_runout_km": RUNOUT_KM_OBSERVED,
        "population_exposed": exposure_lit.total_population_exposed,
        "population_by_distance": exposure_lit.population_by_distance,
        "settlements_at_risk": exposure_lit.settlements_at_risk,
        "exposure_score": round(exposure_lit.exposure_score, 1),
    }

    return results


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  GLEWS Validation: Chamoli 2021 Rock-Ice Avalanche")
    print("  Synthetic Retrospective Analysis")
    print("=" * 70)
    print()

    # Load config
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    # Step 1: ASF search (document the gap)
    print("--- Step 1: ASF DAAC search ---")
    asf_results = attempt_asf_search()
    for note in asf_results["notes"]:
        print(f"  {note}")
    print()

    # Step 2: Build synthetic data
    print("--- Step 2: Generate synthetic displacement time series ---")
    ts = build_chamoli_synthetic()
    print(f"  Epochs: {len(ts.dates)}")
    print(f"  Grid: {ts.displacement.shape[1]} x {ts.displacement.shape[2]}")
    print(f"  Date range: {ts.date_strings[0]} – {ts.date_strings[-1]}")
    print(f"  Valid pixels: {np.sum(np.isfinite(ts.velocity)):,}")
    print()

    # Step 3: Detection at sigma=2.0 (config default)
    print("--- Step 3: Run detection (sigma=2.0, config default) ---")
    det_20, flags_20 = run_detection(ts, config, sigma_override=2.0)
    print()

    # Step 3b: Detection at sigma=2.5 (test the config comment claim)
    print("--- Step 3b: Run detection (sigma=2.5, comparison) ---")
    det_25, flags_25 = run_detection(ts, config, sigma_override=2.5)
    print()

    # Step 4: Cascade risk assessment
    print("--- Step 4: Cascade risk assessment ---")
    # Use signal flags from the sigma=2.0 run for cascade assessment.
    # If no signal flags, run cascade on the literature-informed scenario
    # only (skip pipeline assessment on noise clusters, whose volumes
    # are meaningless).
    site_flag_ids = {lt["flag_id"] for lt in det_20["signal_lead_times"]}
    site_flags = [f for f in flags_20 if f.flag_id in site_flag_ids]

    cascade_results = run_cascade_assessment(
        site_flags, config, no_signal_detected=(len(site_flags) == 0)
    )
    print()

    # Step 5: Summary
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    detected_20 = det_20["detected"]
    detected_25 = det_25["detected"]

    if detected_20:
        print(f"  sigma=2.0: DETECTED — {det_20['signal_flags']} signal flag(s), "
              f"{det_20['noise_flags_near_site']} noise flags near site")
        if det_20["earliest_crossing_in_accel_phase"]:
            ec = det_20["earliest_crossing_in_accel_phase"]
            print(f"    Earliest signal crossing: {ec['operationally_available_date']} "
                  f"({ec['lead_time_days']:.0f} days before collapse)")
        if det_20["max_lead_time_days"] is not None:
            print(f"    Max lead time (signal flag): {det_20['max_lead_time_days']:.0f} days")
    else:
        print("  sigma=2.0: NOT DETECTED (in acceleration phase)")
        print(f"    Total flags: {det_20['total_flags']}, "
              f"noise near site: {det_20['noise_flags_near_site']}")

    if detected_25:
        print(f"  sigma=2.5: DETECTED — {det_25['signal_flags']} signal flag(s), "
              f"{det_25['noise_flags_near_site']} noise flags near site")
        if det_25["earliest_crossing_in_accel_phase"]:
            ec = det_25["earliest_crossing_in_accel_phase"]
            print(f"    Earliest signal crossing: {ec['operationally_available_date']} "
                  f"({ec['lead_time_days']:.0f} days before collapse)")
    else:
        print("  sigma=2.5: NOT DETECTED (in acceleration phase)")
        print(f"    Total flags: {det_25['total_flags']}, "
              f"noise near site: {det_25['noise_flags_near_site']}")

    if detected_20 and not detected_25:
        print("\n  The config comment (lines 81-85) claims sigma=2.5 would have")
        print("  caught it ~2 weeks before collapse. Our synthetic scenario shows")
        print("  sigma=2.0 detects it but sigma=2.5 does not, supporting the")
        print("  choice of the lower threshold.")
    elif detected_20 and detected_25:
        lead_20 = det_20.get("earliest_lead_time_days", 0)
        lead_25 = det_25.get("earliest_lead_time_days", 0)
        if lead_20 is not None and lead_25 is not None:
            print(f"\n  sigma=2.0 gives {lead_20:.0f} days warning vs "
                  f"{lead_25:.0f} days at sigma=2.5")
    elif not detected_20 and not detected_25:
        print("\n  Neither threshold detected the signal within the "
              "acceleration phase.")
        if det_20["earliest_noise_crossing"]:
            nc = det_20["earliest_noise_crossing"]
            print(f"  However, noise crossings occur as early as "
                  f"{nc['operationally_available_date']} "
                  f"(sigma=2.0), demonstrating the false positive rate.")

        # Show z-score diagnostic
        diag = det_20.get("site_zscore_diagnostic", {})
        if diag:
            print(f"\n  Z-score at injection site:")
            print(f"    Max in accel phase: {diag['max_zscore_in_accel_phase']:.2f}")
            print(f"    Max pre-accel:      {diag['max_zscore_pre_accel_phase']:.2f}")
            print(f"    {diag['interpretation']}")

    # Pipeline vs literature volume
    if cascade_results["pipeline_assessments"]:
        pipeline_vol = cascade_results["pipeline_assessments"][0]["estimated_volume_m3"]
        print(f"\n  Volume: pipeline={pipeline_vol:,.0f} m3 vs "
              f"literature={LITERATURE_VOLUME_M3:,.0f} m3 "
              f"(ratio={LITERATURE_VOLUME_M3/max(1,pipeline_vol):.1f}x)")
    else:
        print(f"\n  No pipeline volume estimate (no signal detected).")
        print(f"  Literature volume: {LITERATURE_VOLUME_M3:,.0f} m3")

    lit = cascade_results["literature_comparison"]
    print(f"  Runout (literature Scheidegger): {lit['runout_distance_km']:.1f} km "
          f"(observed: {RUNOUT_KM_OBSERVED} km)")
    print(f"  Population exposed (literature scenario): "
          f"{lit['population_exposed']:,}")

    # Build output JSON
    output = {
        "event_date": EVENT_DATE.isoformat(),
        "location": {
            "name": "Chamoli, Uttarakhand, India",
            "latitude": SITE_LAT,
            "longitude": SITE_LON,
            "elevation_m": FAILURE_ELEVATION_M,
            "peak": "Ronti Peak",
        },
        "validation_type": "synthetic_retrospective",
        "generated": datetime.now().isoformat(),
        "asf_search": asf_results,
        "synthetic_parameters": {
            "note": (
                "These are ASSUMED parameters for a hypothetical precursor "
                "scenario. No published study quantifies pre-event InSAR "
                "displacement rates at Ronti Peak. Shugar et al. (2021) "
                "identified crack widening in optical imagery but did not "
                "report InSAR-derived creep rates."
            ),
            "base_velocity_m_yr": 0.5,
            "acceleration_onset_days_before": 120,
            "max_acceleration_m_yr2": 25.0,
            "ramp_type": "exponential",
            "failure_zone_radius_pixels": 7,
            "pixel_size_approx_m": 90,
            "wedge_dimensions": {
                "length_m": WEDGE_LENGTH_M,
                "width_m": WEDGE_WIDTH_M,
                "depth_m": WEDGE_DEPTH_M,
            },
        },
        "detection_results": {
            "sigma_2_0": det_20,
            "sigma_2_5": det_25,
            "would_glews_have_flagged": detected_20,
            "earliest_warning_days": det_20.get("earliest_lead_time_days"),
            "how_many_days_before": det_20.get("earliest_lead_time_days"),
            "assessment_summary": (
                f"At sigma=2.0, GLEWS {'detected' if detected_20 else 'did not detect'} "
                f"the synthetic precursor signal in the acceleration phase "
                f"({det_20['signal_flags']} signal flag(s), "
                f"{det_20['noise_flags_near_site']} noise flag(s) near site). "
                f"At sigma=2.5, GLEWS {'detected' if detected_25 else 'did not detect'} "
                f"it ({det_25['signal_flags']} signal flag(s))."
            ),
        },
        "cascade_assessment": cascade_results,
        "event_facts": {
            "failure_volume_m3": LITERATURE_VOLUME_M3,
            "wedge_dimensions_m": f"{WEDGE_LENGTH_M} x {WEDGE_WIDTH_M} x {WEDGE_DEPTH_M}",
            "slope_angle_deg": SLOPE_ANGLE_DEG,
            "runout_distance_km": RUNOUT_KM_OBSERVED,
            "casualties": 204,
            "infrastructure_destroyed": [
                "Rishiganga Hydroelectric Project (13.2 MW)",
                "Tapovan Vishnugad Hydroelectric Project (520 MW, under construction)",
            ],
        },
        "limitations": [
            "NISAR launched in 2024 — no NISAR GOFF/GUNW data exist for this 2021 event.",
            "No published InSAR-derived pre-event displacement rates at Ronti Peak. "
            "The injected precursor signal uses physically plausible assumptions, "
            "not measured values.",
            "Synthetic data lacks real atmospheric artifacts, DEM errors, and "
            "seasonal snow decorrelation typical of high Himalayan InSAR.",
            "Detection lead times depend entirely on the assumed acceleration "
            "onset and magnitude — different assumptions yield different results.",
            "The pipeline volume heuristic (depth = 0.1 * sqrt(area)) significantly "
            "underestimates deep-seated rock-ice failures. The Chamoli wedge was "
            "~80 m deep vs the heuristic's expectation of ~35 m for a 125,000 m2 area.",
            "Cascade exposure uses synthetic settlements, not actual census data.",
        ],
        "literature_references": [
            {
                "authors": "Shugar et al.",
                "year": 2021,
                "title": "A massive rock and ice avalanche caused the 2021 disaster at Chamoli, Indian Himalaya",
                "journal": "Science",
                "volume": "373(6552)",
                "pages": "300-306",
                "doi": "10.1126/science.abh4455",
                "used_for": "Event description, failure dimensions, debris flow runout distance",
            },
            {
                "authors": "Voight",
                "year": 1988,
                "title": "A method for prediction of volcanic eruptions",
                "journal": "Nature",
                "volume": "332(6160)",
                "pages": "125-130",
                "used_for": "Failure forecasting law applied to acceleration detection",
            },
            {
                "authors": "Scheidegger",
                "year": 1973,
                "title": "On the prediction of the reach and velocity of catastrophic landslides",
                "journal": "Rock Mechanics",
                "volume": "5(4)",
                "pages": "231-236",
                "used_for": "Volume-dependent runout estimation (angle of reach)",
            },
            {
                "authors": "Hungr, Leroueil & Picarelli",
                "year": 2014,
                "title": "The Varnes classification of landslide types, an update",
                "journal": "Landslides",
                "volume": "11(2)",
                "pages": "167-194",
                "used_for": "Failure depth heuristic for volume estimation",
            },
        ],
    }

    out_path = DATA_DIR / "chamoli_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
