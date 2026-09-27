"""
Anomaly detection module — Tier 0 screening for unstable glaciers
and rock slopes.

Takes acceleration maps from the timeseries module and identifies
spatially coherent clusters of anomalous acceleration. Each cluster
is a "flag" — a candidate unstable site that advances to Tier 1
filtering.

Design principles:
    - Each site is compared to its own history (unsupervised)
    - No labeled collapse data required
    - Spatial coherence filters isolated noisy pixels
    - Output is a ranked list of flags with anomaly scores
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import ndimage
from sklearn.cluster import DBSCAN

from glews.timeseries import (
    AccelerationMap,
    bocpd_changepoints,
    detect_step_changes,
    fit_voight,
)

logger = logging.getLogger(__name__)


@dataclass
class AnomalyFlag:
    """
    A spatially coherent cluster of anomalous acceleration.

    This is the output of Tier 0 screening — each flag represents
    a candidate unstable site that should advance to Tier 1 filtering.
    """

    flag_id: int
    score: float                    # composite anomaly score (higher = more anomalous)
    peak_zscore: float              # maximum z-score in the cluster
    mean_zscore: float              # mean z-score across cluster pixels
    n_pixels: int                   # number of pixels in cluster
    area_m2: float                  # approximate area of deforming zone
    center_lat: float
    center_lon: float
    peak_lat: float                 # location of maximum anomaly
    peak_lon: float
    acceleration_m_yr2: float       # peak acceleration in m/yr²
    pixel_indices: np.ndarray       # [n_pixels, 2] row/col indices
    window_index: int               # time window where anomaly peaks
    window_date: float              # ordinal date of that window center

    # Voight analysis (if applicable)
    voight_fit: dict | None = None  # predicted failure time, R², etc.

    # Mean displacement time series over the flagged cluster pixels.
    # dict with "dates" (list of ISO date strings) and "values"
    # (list of floats, displacement in meters).  Populated when raw
    # displacement is supplied to detect_anomalies(); None otherwise.
    timeseries: dict | None = None

    # Metadata for Tier 1
    detection_details: dict = field(default_factory=dict)


def detect_anomalies(
    accel_map: AccelerationMap,
    latitude: np.ndarray,
    longitude: np.ndarray,
    config: dict,
    dates: np.ndarray | None = None,
    displacement: np.ndarray | None = None,
) -> list[AnomalyFlag]:
    """
    Run Tier 0 anomaly detection on an acceleration map.

    Algorithm:
    1. Threshold the z-score map at each time window
    2. Cluster spatially connected anomalous pixels (DBSCAN)
    3. Filter clusters by minimum size
    4. Score and rank surviving clusters
    5. If raw displacement is supplied, also run step-change detection
       and merge (union) its flags with the acceleration-based flags
    6. Optionally fit Voight's law to top candidates

    The acceleration detector catches gradual pre-failure speed-up
    (Voight-style). It misses failures like the Nepal 2026 collapse,
    which produced 3.5 m of displacement in a single 12-day cycle
    with velocities that *decelerated* after the initial rupture —
    no gradual acceleration trend for Voight's law to fit. Supplying
    `dates`/`displacement` enables `_detect_step_change_anomalies()`
    to catch that pattern directly, independent of any trend model.

    Parameters
    ----------
    accel_map : AccelerationMap
        Output from timeseries.compute_acceleration_map().
    latitude, longitude : np.ndarray
        Coordinate grids, shape [n_rows, n_cols].
    config : dict
        Detection configuration (from config YAML 'detect' section).
    dates : np.ndarray, optional
        Acquisition dates as ordinal days, shape [n_epochs]. Required
        (together with `displacement`) to enable step-change detection.
    displacement : np.ndarray, optional
        Raw LOS displacement in meters, shape [n_epochs, n_rows, n_cols].
        Required (together with `dates`) to enable step-change detection.

    Returns
    -------
    list[AnomalyFlag]
        Flags sorted by score (highest first).
    """
    det = config["detect"]
    sigma = det["acceleration"]["sigma_threshold"]
    clust = det["clustering"]
    min_pixels = clust["min_cluster_pixels"]
    min_area = clust["min_area_m2"]

    n_windows, n_rows, n_cols = accel_map.acceleration_zscore.shape

    logger.info(
        "Running Tier 0 detection: σ=%.1f, min_pixels=%d, min_area=%d m²",
        sigma, min_pixels, min_area,
    )

    # Estimate pixel area from coordinate grid
    pixel_area = _estimate_pixel_area(latitude, longitude)

    all_flags: list[AnomalyFlag] = []
    flag_id = 0

    # Process each time window, focusing on the most recent ones
    # (pre-failure acceleration is most diagnostic near the end)
    for w in range(n_windows):
        zscore_map = accel_map.acceleration_zscore[w]
        accel_vals = accel_map.acceleration[w]

        # Threshold: pixels exceeding sigma threshold
        # Use absolute value — both positive and negative acceleration
        # can indicate instability depending on viewing geometry
        anomalous = np.abs(zscore_map) > sigma

        # Skip windows with no anomalies
        if not np.any(anomalous):
            continue

        # Spatial clustering
        clusters = _cluster_anomalous_pixels(
            anomalous,
            latitude,
            longitude,
            max_distance_m=clust["max_distance_m"],
            min_samples=min_pixels,
        )

        for cluster_label in np.unique(clusters):
            if cluster_label == -1:
                continue  # noise label from DBSCAN

            mask = clusters == cluster_label
            n_pix = np.count_nonzero(mask)
            area = n_pix * pixel_area

            if n_pix < min_pixels or area < min_area:
                continue

            # Extract cluster properties
            rows, cols = np.where(mask)
            zscores = zscore_map[mask]
            accels = accel_vals[mask]

            peak_idx = np.argmax(np.abs(zscores))
            peak_row, peak_col = rows[peak_idx], cols[peak_idx]

            # Composite score: combines magnitude, spatial extent, and persistence
            mean_z = np.nanmean(np.abs(zscores))
            peak_z = np.nanmax(np.abs(zscores))
            spatial_weight = np.log10(max(1, area))
            score = peak_z * 0.5 + mean_z * 0.3 + spatial_weight * 0.2

            flag = AnomalyFlag(
                flag_id=flag_id,
                score=score,
                peak_zscore=float(peak_z),
                mean_zscore=float(mean_z),
                n_pixels=n_pix,
                area_m2=float(area),
                center_lat=float(np.mean(latitude[mask])),
                center_lon=float(np.mean(longitude[mask])),
                peak_lat=float(latitude[peak_row, peak_col]),
                peak_lon=float(longitude[peak_row, peak_col]),
                acceleration_m_yr2=float(accels[peak_idx]),
                pixel_indices=np.column_stack([rows, cols]),
                window_index=w,
                window_date=float(accel_map.window_centers[w]),
                detection_details={
                    "tag": "acceleration",
                    "sigma_threshold": sigma,
                    "window_size_days": int(
                        accel_map.window_centers[1] - accel_map.window_centers[0]
                    )
                    if len(accel_map.window_centers) > 1
                    else 0,
                },
            )

            all_flags.append(flag)
            flag_id += 1

    # Step-change detection: catches abrupt, single-epoch jumps that
    # the acceleration detector's trend-based model would miss (the
    # Nepal 2026 failure mode). Merges (union) with the acceleration
    # flags above; overlapping spatial clusters are resolved by
    # _deduplicate_flags, which keeps the higher-scoring flag.
    if dates is not None and displacement is not None:
        step_flags = _detect_step_change_anomalies(
            dates, displacement, latitude, longitude, config, flag_id,
        )
        all_flags.extend(step_flags)
        flag_id += len(step_flags)

    # BOCPD changepoint detection: when enabled via config
    # (detect.changepoint.method == "bocpd"), runs Bayesian Online
    # Changepoint Detection per pixel and flags spatial clusters
    # where changepoints coincide. Opt-in — existing methods remain
    # the default.
    cp_config = det.get("changepoint", {})
    if (
        cp_config.get("method") == "bocpd"
        and dates is not None
        and displacement is not None
    ):
        bocpd_flags = _detect_bocpd_anomalies(
            dates, displacement, latitude, longitude, config, flag_id,
        )
        all_flags.extend(bocpd_flags)
        flag_id += len(bocpd_flags)

    # Deduplicate: if the same spatial cluster appears in multiple
    # time windows (or across detectors), keep only the
    # highest-scoring instance
    flags = _deduplicate_flags(all_flags, latitude, longitude)

    # Populate per-flag displacement time series when raw data is
    # available.  Done after dedup so we only compute for surviving
    # flags.
    if dates is not None and displacement is not None:
        _populate_timeseries(flags, dates, displacement)

    # Spatial coherence: when enabled, run graph-based spatial anomaly
    # detection and annotate each flag with a coherence score based on
    # overlap with spatially coherent clusters.  High coherence boosts
    # the flag score; low coherence penalises it.
    spatial_cfg = det.get("spatial", {})
    if (
        spatial_cfg.get("enabled", False)
        and dates is not None
        and displacement is not None
    ):
        from glews.spatial import detect_spatial_anomalies

        try:
            spatial_anomalies = detect_spatial_anomalies(
                displacement, dates, latitude, longitude,
                config={"spatial": spatial_cfg},
            )

            for flag in flags:
                best_coherence = 0.0
                flag_pixels = set(map(tuple, flag.pixel_indices.tolist()))
                for sa in spatial_anomalies:
                    sa_pixels = set(map(tuple, sa.pixel_indices.tolist()))
                    overlap = len(flag_pixels & sa_pixels)
                    if overlap > 0 and sa.coherence_score > best_coherence:
                        best_coherence = sa.coherence_score

                flag.detection_details["spatial_coherence"] = best_coherence

                # Boost or penalise score based on coherence
                if best_coherence >= 0.7:
                    flag.score *= 1.3   # strong spatial support
                elif best_coherence >= 0.4:
                    flag.score *= 1.0   # neutral
                else:
                    flag.score *= 0.7   # weak spatial support
        except Exception:
            logger.warning(
                "Spatial coherence check failed; skipping",
                exc_info=True,
            )

    # Classifier scoring: when enabled and a trained model exists,
    # score each flag's displacement time series with the transfer
    # learning classifier and record the probability in detection_details.
    cls_config = det.get("classifier", {})
    if (
        cls_config.get("enabled", False)
        and dates is not None
        and displacement is not None
    ):
        model_path = cls_config.get("model_path", "")
        if model_path and Path(model_path).is_file():
            from glews.classifier import PrecursorClassifier, PrecursorFeatureExtractor

            try:
                clf = PrecursorClassifier.load(model_path)
                extractor = PrecursorFeatureExtractor()
                n_epochs_c, n_rows_c, n_cols_c = displacement.shape

                for flag in flags:
                    rows = flag.pixel_indices[:, 0]
                    cols = flag.pixel_indices[:, 1]
                    valid_px = (
                        (rows >= 0) & (rows < n_rows_c)
                        & (cols >= 0) & (cols < n_cols_c)
                    )
                    rows, cols = rows[valid_px], cols[valid_px]
                    if len(rows) == 0:
                        continue

                    mean_disp = np.nanmean(displacement[:, rows, cols], axis=1)
                    feats = extractor.extract_features(dates.astype(float), mean_disp)
                    score = float(clf.predict_proba(feats)[0])
                    flag.detection_details["classifier_score"] = score
            except Exception:
                logger.warning(
                    "Classifier scoring failed; skipping",
                    exc_info=True,
                )
        else:
            logger.debug(
                "Classifier enabled but model not found at %r; skipping",
                model_path,
            )

    # Voight analysis on top candidates
    if det.get("voight", {}).get("enabled", False):
        flags = _apply_voight_analysis(
            flags, accel_map, det["voight"]
        )

    # Sort by score (highest first)
    flags.sort(key=lambda f: f.score, reverse=True)

    logger.info(
        "Tier 0 complete: %d flags from %d raw detections across %d windows",
        len(flags), len(all_flags), n_windows,
    )

    for f in flags[:10]:
        logger.info(
            "  Flag %d: score=%.2f, z=%.1f, area=%.0f m², "
            "lat=%.4f lon=%.4f",
            f.flag_id, f.score, f.peak_zscore, f.area_m2,
            f.center_lat, f.center_lon,
        )

    return flags


def _coords_are_projected(latitude: np.ndarray, longitude: np.ndarray) -> bool:
    """Detect whether coordinates are in a projected CRS (meters) vs geographic (degrees)."""
    lat_range = np.nanmax(latitude) - np.nanmin(latitude)
    lon_range = np.nanmax(longitude) - np.nanmin(longitude)
    # Geographic coordinates: lat ∈ [-90, 90], lon ∈ [-180, 180]
    # Projected (UTM): values in hundreds of thousands to millions
    return lat_range > 1000 or lon_range > 1000


def _to_meters(
    lats: np.ndarray, lons: np.ndarray, projected: bool
) -> np.ndarray:
    """Convert coordinate arrays to [N, 2] meter positions."""
    if projected:
        # Already in meters (UTM or similar)
        return np.column_stack([lats, lons])
    else:
        lat_m = lats * 111_320
        lon_m = lons * 111_320 * np.cos(np.radians(np.mean(lats)))
        return np.column_stack([lat_m, lon_m])


def _cluster_anomalous_pixels(
    anomalous: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    max_distance_m: float,
    min_samples: int,
) -> np.ndarray:
    """
    Cluster spatially connected anomalous pixels using DBSCAN.

    Returns an array of cluster labels (same shape as anomalous),
    with -1 for non-anomalous or noise pixels.
    """
    rows, cols = np.where(anomalous)
    if len(rows) == 0:
        return np.full(anomalous.shape, -1, dtype=int)

    # Convert pixel coordinates to meters
    lats = latitude[rows, cols]
    lons = longitude[rows, cols]
    projected = _coords_are_projected(latitude, longitude)
    coords_m = _to_meters(lats, lons, projected)

    # DBSCAN clustering
    clustering = DBSCAN(
        eps=max_distance_m,
        min_samples=min_samples,
        metric="euclidean",
    ).fit(coords_m)

    labels = np.full(anomalous.shape, -1, dtype=int)
    labels[rows, cols] = clustering.labels_

    return labels


def _estimate_pixel_area(latitude: np.ndarray, longitude: np.ndarray) -> float:
    """Estimate the area of a single pixel in m² from coordinate grids."""
    if latitude.shape[0] < 2 or latitude.shape[1] < 2:
        return 900.0  # default ~30m × 30m for Sentinel-1

    dlat = np.abs(latitude[1, 0] - latitude[0, 0])
    dlon = np.abs(longitude[0, 1] - longitude[0, 0])

    if _coords_are_projected(latitude, longitude):
        # Coordinates already in meters
        return dlat * dlon
    else:
        mean_lat = np.nanmean(latitude)
        lat_m = dlat * 111_320
        lon_m = dlon * 111_320 * np.cos(np.radians(mean_lat))
        return lat_m * lon_m


def _ordinal_to_iso(ordinal: float) -> str:
    """Convert an ordinal date to an ISO date string (YYYY-MM-DD)."""
    return datetime.fromordinal(int(ordinal)).strftime("%Y-%m-%d")


def _populate_timeseries(
    flags: list[AnomalyFlag],
    dates: np.ndarray,
    displacement: np.ndarray,
) -> None:
    """
    Populate the ``timeseries`` field on each flag with the mean
    displacement over the flag's pixel cluster.

    Parameters
    ----------
    flags : list[AnomalyFlag]
        Flags to populate (modified in place).
    dates : np.ndarray
        Acquisition dates as ordinal days, shape [n_epochs].
    displacement : np.ndarray
        Raw LOS displacement in meters, shape [n_epochs, n_rows, n_cols].
    """
    n_epochs, n_rows, n_cols = displacement.shape

    for flag in flags:
        rows = flag.pixel_indices[:, 0]
        cols = flag.pixel_indices[:, 1]

        # Clamp to valid index range
        valid = (rows < n_rows) & (cols < n_cols) & (rows >= 0) & (cols >= 0)
        rows, cols = rows[valid], cols[valid]
        if len(rows) == 0:
            continue

        # Mean displacement across cluster pixels at each epoch
        mean_disp = np.nanmean(displacement[:, rows, cols], axis=1)

        # Keep only finite epochs so downstream consumers (JSON, SVG)
        # never encounter NaN / Inf.
        finite_mask = np.isfinite(mean_disp)
        if not np.any(finite_mask):
            continue

        ts_dates = [_ordinal_to_iso(d) for d, ok in zip(dates, finite_mask) if ok]
        ts_values = [round(float(v), 4) for v, ok in zip(mean_disp, finite_mask) if ok]

        flag.timeseries = {"dates": ts_dates, "values": ts_values}


def _detect_step_change_anomalies(
    dates: np.ndarray,
    displacement: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    config: dict,
    start_flag_id: int,
) -> list[AnomalyFlag]:
    """
    Tier 0 screening for abrupt, single-epoch step-change displacement.

    Complements the acceleration-based detector in detect_anomalies():
    that detector looks for a *sustained* speed-up (Voight-style
    pre-failure acceleration), while this one flags a sudden
    displacement jump between two consecutive epochs with no gradual
    build-up — the pattern seen in the Nepal 2026 collapse (3.5 m in
    one 12-day cycle), where velocities decelerated after the rupture
    rather than continuing to accelerate.

    Parameters
    ----------
    dates : np.ndarray
        Acquisition dates as ordinal days, shape [n_epochs].
    displacement : np.ndarray
        Raw LOS displacement in meters, shape [n_epochs, n_rows, n_cols].
    latitude, longitude : np.ndarray
        Coordinate grids, shape [n_rows, n_cols].
    config : dict
        Detection configuration (from config YAML 'detect' section).
        Reads 'detect.step_change.sigma_threshold' (default 5.0) and
        'detect.step_change.min_displacement_m' (default 0.5).
    start_flag_id : int
        First flag_id to assign, so ids stay unique alongside flags
        from the acceleration-based detector.

    Returns
    -------
    list[AnomalyFlag]
        Step-change flags, each tagged
        detection_details["tag"] == "step_change".
    """
    det = config["detect"]
    sc_config = det.get("step_change", {})
    sigma = sc_config.get("sigma_threshold", 5.0)
    min_disp = sc_config.get("min_displacement_m", 0.5)
    clust = det["clustering"]
    min_pixels = clust["min_cluster_pixels"]
    min_area = clust["min_area_m2"]

    logger.info(
        "Running step-change detection: σ=%.1f, min_displacement=%.2fm, "
        "min_pixels=%d, min_area=%d m²",
        sigma, min_disp, min_pixels, min_area,
    )

    step_map = detect_step_changes(
        dates,
        displacement,
        sigma_threshold=sigma,
        min_displacement_m=min_disp,
    )

    n_epochs = step_map.step_change.shape[0]
    pixel_area = _estimate_pixel_area(latitude, longitude)

    flags: list[AnomalyFlag] = []
    flag_id = start_flag_id

    for e in range(n_epochs):
        step_mask = step_map.step_change[e]
        if not np.any(step_mask):
            continue

        magnitudes = step_map.magnitude[e]

        clusters = _cluster_anomalous_pixels(
            step_mask,
            latitude,
            longitude,
            max_distance_m=clust["max_distance_m"],
            min_samples=min_pixels,
        )

        for cluster_label in np.unique(clusters):
            if cluster_label == -1:
                continue  # noise label from DBSCAN

            mask = clusters == cluster_label
            n_pix = np.count_nonzero(mask)
            area = n_pix * pixel_area

            if n_pix < min_pixels or area < min_area:
                continue

            rows, cols = np.where(mask)
            mags = magnitudes[mask]

            peak_idx = np.argmax(np.abs(mags))
            peak_row, peak_col = rows[peak_idx], cols[peak_idx]

            mean_mag = float(np.nanmean(np.abs(mags)))
            peak_mag = float(np.nanmax(np.abs(mags)))

            # Express the jump in units of the configured minimum, so
            # the score is comparable in scale to the acceleration
            # detector's z-score-based score.
            unit = min_disp if min_disp > 0 else 1.0
            peak_score_units = peak_mag / unit
            mean_score_units = mean_mag / unit

            spatial_weight = np.log10(max(1, area))
            score = (
                peak_score_units * 0.5
                + mean_score_units * 0.3
                + spatial_weight * 0.2
            )

            flag = AnomalyFlag(
                flag_id=flag_id,
                score=score,
                peak_zscore=peak_score_units,
                mean_zscore=mean_score_units,
                n_pixels=n_pix,
                area_m2=float(area),
                center_lat=float(np.mean(latitude[mask])),
                center_lon=float(np.mean(longitude[mask])),
                peak_lat=float(latitude[peak_row, peak_col]),
                peak_lon=float(longitude[peak_row, peak_col]),
                acceleration_m_yr2=float("nan"),
                pixel_indices=np.column_stack([rows, cols]),
                window_index=e,
                window_date=float(dates[e]),
                detection_details={
                    "tag": "step_change",
                    "sigma_threshold": sigma,
                    "min_displacement_m": min_disp,
                    "displacement_jump_m": peak_mag,
                },
            )

            flags.append(flag)
            flag_id += 1

    return flags


def _detect_bocpd_anomalies(
    dates: np.ndarray,
    displacement: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    config: dict,
    start_flag_id: int,
) -> list[AnomalyFlag]:
    """
    Tier 0 screening using Bayesian Online Changepoint Detection.

    Runs BOCPD (Adams & MacKay 2007) per pixel on the displacement
    time series to detect regime changes in deformation rate. Pixels
    whose changepoint probability exceeds the configured threshold at
    the same epoch are clustered spatially and emitted as flags.

    Parameters
    ----------
    dates : np.ndarray
        Acquisition dates as ordinal days, shape [n_epochs].
    displacement : np.ndarray
        Raw LOS displacement in meters, shape [n_epochs, n_rows, n_cols].
    latitude, longitude : np.ndarray
        Coordinate grids, shape [n_rows, n_cols].
    config : dict
        Detection configuration (from config YAML 'detect' section).
    start_flag_id : int
        First flag_id to assign.

    Returns
    -------
    list[AnomalyFlag]
        BOCPD flags, each tagged detection_details["tag"] == "bocpd".
    """
    det = config["detect"]
    cp_config = det.get("changepoint", {})
    hazard_rate = cp_config.get("hazard_rate", 1 / 100)
    prior_variance = cp_config.get("prior_variance", None)
    threshold = cp_config.get("threshold", 0.25)
    clust = det["clustering"]
    min_pixels = clust["min_cluster_pixels"]
    min_area = clust["min_area_m2"]

    n_epochs, n_rows, n_cols = displacement.shape

    logger.info(
        "Running BOCPD changepoint detection: %d epochs, %d×%d pixels, "
        "hazard_rate=%.4f, threshold=%.2f",
        n_epochs, n_rows, n_cols, hazard_rate, threshold,
    )

    # Per-pixel changepoint probability map: for each epoch, the
    # maximum changepoint probability across all pixels that fire there.
    cp_prob_map = np.zeros((n_epochs, n_rows, n_cols))

    # Skip pixels whose displacement range is below a floor (cost guard
    # for large scenes — BOCPD is O(n²) per pixel in a Python loop).
    disp_range = np.nanmax(displacement, axis=0) - np.nanmin(displacement, axis=0)
    min_range = cp_config.get("min_displacement_range_m", 0.0)

    for row in range(n_rows):
        for col in range(n_cols):
            if disp_range[row, col] < min_range:
                continue
            pixel_disp = displacement[:, row, col]
            if np.sum(np.isfinite(pixel_disp)) < 3:
                continue

            result = bocpd_changepoints(
                dates,
                pixel_disp,
                hazard_rate=hazard_rate,
                prior_variance=prior_variance,
                threshold=threshold,
            )
            cp_prob_map[:, row, col] = result.changepoint_probabilities

    pixel_area = _estimate_pixel_area(latitude, longitude)

    flags: list[AnomalyFlag] = []
    flag_id = start_flag_id

    for e in range(n_epochs):
        anomalous = cp_prob_map[e] > threshold
        if not np.any(anomalous):
            continue

        clusters = _cluster_anomalous_pixels(
            anomalous,
            latitude,
            longitude,
            max_distance_m=clust["max_distance_m"],
            min_samples=min_pixels,
        )

        for cluster_label in np.unique(clusters):
            if cluster_label == -1:
                continue

            mask = clusters == cluster_label
            n_pix = np.count_nonzero(mask)
            area = n_pix * pixel_area

            if n_pix < min_pixels or area < min_area:
                continue

            rows, cols = np.where(mask)
            probs = cp_prob_map[e][mask]

            peak_idx = np.argmax(probs)
            peak_row, peak_col = rows[peak_idx], cols[peak_idx]

            mean_prob = float(np.mean(probs))
            peak_prob = float(np.max(probs))

            # Express in units of the threshold so the score scale is
            # comparable to acceleration z-scores and step-change scores.
            unit = threshold if threshold > 0 else 1.0
            peak_score_units = peak_prob / unit
            mean_score_units = mean_prob / unit

            spatial_weight = np.log10(max(1, area))
            score = (
                peak_score_units * 0.5
                + mean_score_units * 0.3
                + spatial_weight * 0.2
            )

            flag = AnomalyFlag(
                flag_id=flag_id,
                score=score,
                peak_zscore=peak_score_units,
                mean_zscore=mean_score_units,
                n_pixels=n_pix,
                area_m2=float(area),
                center_lat=float(np.mean(latitude[mask])),
                center_lon=float(np.mean(longitude[mask])),
                peak_lat=float(latitude[peak_row, peak_col]),
                peak_lon=float(longitude[peak_row, peak_col]),
                acceleration_m_yr2=float("nan"),
                pixel_indices=np.column_stack([rows, cols]),
                window_index=e,
                window_date=float(dates[e]),
                detection_details={
                    "tag": "bocpd",
                    "hazard_rate": hazard_rate,
                    "threshold": threshold,
                    "peak_changepoint_probability": peak_prob,
                },
            )

            flags.append(flag)
            flag_id += 1

    logger.info(
        "BOCPD detection complete: %d flagged clusters",
        len(flags),
    )

    return flags


def _deduplicate_flags(
    flags: list[AnomalyFlag],
    latitude: np.ndarray,
    longitude: np.ndarray,
    merge_distance_m: float = 500.0,
) -> list[AnomalyFlag]:
    """
    Merge flags from different time windows that overlap spatially.

    Keeps the highest-scoring flag from each spatial cluster.
    """
    if not flags:
        return []

    # Group by spatial proximity of cluster centers
    centers = np.array([[f.center_lat, f.center_lon] for f in flags])
    projected = np.max(centers[:, 0]) - np.min(centers[:, 0]) > 1000
    if projected:
        centers_m = centers.copy()
    else:
        mean_lat = np.mean(centers[:, 0])
        centers_m = centers.copy()
        centers_m[:, 0] *= 111_320
        centers_m[:, 1] *= 111_320 * np.cos(np.radians(mean_lat))

    if len(centers_m) == 1:
        return flags

    spatial_clustering = DBSCAN(
        eps=merge_distance_m,
        min_samples=1,
        metric="euclidean",
    ).fit(centers_m)

    # Keep highest-scoring flag per spatial cluster
    best = {}
    for flag, label in zip(flags, spatial_clustering.labels_):
        if label not in best or flag.score > best[label].score:
            best[label] = flag

    return list(best.values())


def _apply_voight_analysis(
    flags: list[AnomalyFlag],
    accel_map: AccelerationMap,
    voight_config: dict,
) -> list[AnomalyFlag]:
    """
    Apply Voight's failure law analysis to flagged sites.

    For each flag, extracts the velocity time series at the peak
    anomaly pixel and fits the inverse-velocity linear trend.
    """
    min_points = voight_config.get("min_points", 5)

    for flag in flags:
        # Get velocity time series at peak pixel location
        peak_row = flag.pixel_indices[
            np.argmax(np.abs(
                accel_map.acceleration_zscore[
                    flag.window_index,
                    flag.pixel_indices[:, 0],
                    flag.pixel_indices[:, 1],
                ]
            ))
        ]
        row, col = peak_row

        velocities = accel_map.velocity_residual[:, row, col]
        dates = accel_map.window_centers

        result = fit_voight(dates, np.abs(velocities), min_points=min_points)
        if result is not None:
            r2_threshold = voight_config.get("r_squared_threshold", 0.7)
            if result["r_squared"] >= r2_threshold:
                flag.voight_fit = result
                # Boost score for flags with Voight fit
                flag.score *= 1.5
                window = result.get("failure_window_days", [None, None])
                window_str = ""
                if window[0] is not None and window[1] is not None:
                    window_str = f" (95%% CI: {window[0]:.0f}–{window[1]:.0f} days)"
                logger.info(
                    "  Flag %d: Voight fit R²=%.2f, predicted failure %s "
                    "(%.0f days from last obs)%s",
                    flag.flag_id,
                    result["r_squared"],
                    result.get("predicted_failure_iso", "?"),
                    result["days_until_failure"],
                    window_str,
                )

    return flags
