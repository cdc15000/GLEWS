"""
Tier 1 multi-sensor cross-check filters for GLEWS anomaly flags.

Cross-checks InSAR anomalies against interferometric quality metrics
and (when available) optical imagery to eliminate false positives
caused by atmospheric noise, snow cover changes, or processing
artifacts.

Filters:
    1. Coherence quality — are flagged pixels in areas of adequate
       interferometric coherence, or in decorrelated zones where
       phase measurements are unreliable?
    2. Spatial consistency — is the displacement pattern spatially
       coherent (contiguous deformation lobe) or scattered noise?
    3. Temporal consistency — does the signal persist across multiple
       SAR acquisitions, or is it a single-epoch artifact?
    4. Optical cross-check — does Sentinel-2 imagery show visible
       surface change (crevassing, scarps, bulging) at the site?
       (Stub — requires optical data access to be wired up.)

Each flag receives a ``tier1_quality`` dict with scores from each
check and an overall recommendation ("retain", "demote", "remove").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from glews.detect import AnomalyFlag

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CrossCheckResult:
    """
    Aggregated result of all Tier 1 cross-check filters for one flag.

    Attributes
    ----------
    quality_score : float
        Interferometric coherence quality score, 0-1. Higher means the
        flagged pixels sit in a well-correlated area where phase
        measurements are trustworthy.
    spatial_consistency : float
        Fraction of the flagged area that forms a single connected
        component of real deformation, 0-1. Low values indicate
        scattered noise pixels rather than a coherent deformation lobe.
    temporal_consistency : float
        Fraction of acquisition epochs in which the displacement signal
        is present and consistent, 0-1. Low values indicate a
        single-epoch artifact (atmospheric screen, snow event).
    optical_confirmation : bool | None
        True if optical imagery shows corroborating surface change,
        False if imagery is clear but shows no change, None if optical
        data is unavailable.
    recommendation : str
        One of "retain" (passes all checks), "demote" (marginal — keep
        but lower priority), or "remove" (likely false positive).
    details : dict
        Per-check diagnostic information for logging and reporting.
    """

    quality_score: float
    spatial_consistency: float
    temporal_consistency: float
    optical_confirmation: bool | None
    recommendation: str
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# InSAR quality checks
# ---------------------------------------------------------------------------

class InSARQualityChecker:
    """
    Checks whether InSAR-derived displacement anomalies are supported
    by the interferometric data quality at the flagged pixels.
    """

    def check_coherence_quality(
        self,
        coherence_map: np.ndarray,
        flag_mask: np.ndarray,
        min_coherence: float = 0.3,
    ) -> float:
        """
        Check if flagged pixels have sufficient interferometric coherence.

        Decorrelated areas (low coherence) produce unreliable phase
        measurements. Flags in such areas are likely noise.

        Parameters
        ----------
        coherence_map : np.ndarray
            Temporal coherence values, shape [n_rows, n_cols], range 0-1.
        flag_mask : np.ndarray
            Boolean mask of flagged pixels, same spatial shape.
        min_coherence : float
            Minimum coherence value considered reliable (default 0.3).

        Returns
        -------
        float
            Quality score 0-1. The fraction of flagged pixels whose
            coherence exceeds ``min_coherence``.
        """
        if not np.any(flag_mask):
            return 0.0

        flagged_coherence = coherence_map[flag_mask]
        valid = np.isfinite(flagged_coherence)
        if not np.any(valid):
            return 0.0

        n_above = np.count_nonzero(flagged_coherence[valid] >= min_coherence)
        return float(n_above / valid.sum())

    def check_spatial_consistency(
        self,
        displacement_map: np.ndarray,
        flag_mask: np.ndarray,
        min_connected_fraction: float = 0.5,
    ) -> float:
        """
        Check if the displacement pattern is spatially consistent.

        Real deformation produces a spatially smooth displacement lobe.
        Atmospheric noise or processing artifacts tend to produce
        scattered pixels that do not form a single connected region.

        Parameters
        ----------
        displacement_map : np.ndarray
            Displacement values at a single epoch, shape [n_rows, n_cols].
        flag_mask : np.ndarray
            Boolean mask of flagged pixels, same spatial shape.
        min_connected_fraction : float
            Minimum fraction of flagged pixels that must belong to the
            largest connected component (default 0.5).

        Returns
        -------
        float
            Spatial consistency score 0-1. The fraction of flagged
            pixels belonging to the largest 4-connected component.
        """
        if not np.any(flag_mask):
            return 0.0

        n_flagged = np.count_nonzero(flag_mask)

        # Label connected components (4-connectivity)
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        labeled, n_components = ndimage.label(flag_mask, structure=structure)

        if n_components == 0:
            return 0.0

        # Find the largest component
        component_sizes = ndimage.sum(
            flag_mask, labeled, range(1, n_components + 1)
        )
        largest = float(max(component_sizes))

        return largest / n_flagged

    def check_temporal_consistency(
        self,
        dates: np.ndarray,
        displacement_stack: np.ndarray,
        flag_mask: np.ndarray,
        max_gap_fraction: float = 0.3,
    ) -> float:
        """
        Check if the displacement signal is temporally consistent.

        Real pre-failure deformation persists and typically grows
        across multiple SAR acquisitions. A signal that appears in
        only one epoch and vanishes is likely an atmospheric phase
        screen, a snow-cover change, or a processing artifact.

        Parameters
        ----------
        dates : np.ndarray
            Acquisition dates as ordinal days, shape [n_epochs].
        displacement_stack : np.ndarray
            LOS displacement in meters, shape [n_epochs, n_rows, n_cols].
        flag_mask : np.ndarray
            Boolean mask of flagged pixels, shape [n_rows, n_cols].
        max_gap_fraction : float
            Maximum allowed fraction of epochs where the signal is
            absent (default 0.3). Higher values are more lenient.

        Returns
        -------
        float
            Temporal consistency score 0-1. The fraction of epochs
            in which the mean displacement over flagged pixels
            deviates from the scene-wide median by more than 1
            standard deviation of the background.
        """
        if not np.any(flag_mask) or len(dates) < 2:
            return 0.0

        n_epochs = displacement_stack.shape[0]
        n_active = 0

        for e in range(n_epochs):
            epoch_disp = displacement_stack[e]
            flagged_vals = epoch_disp[flag_mask]
            valid_flagged = flagged_vals[np.isfinite(flagged_vals)]
            if len(valid_flagged) == 0:
                continue

            # Background: everything outside the flag mask
            bg_mask = ~flag_mask
            bg_vals = epoch_disp[bg_mask]
            valid_bg = bg_vals[np.isfinite(bg_vals)]
            if len(valid_bg) == 0:
                continue

            bg_median = np.median(valid_bg)
            bg_std = np.std(valid_bg)
            if bg_std < 1e-12:
                bg_std = 1e-12

            flag_mean = np.mean(valid_flagged)
            if abs(flag_mean - bg_median) > bg_std:
                n_active += 1

        return float(n_active / n_epochs)


# ---------------------------------------------------------------------------
# Optical cross-check (stub)
# ---------------------------------------------------------------------------

class OpticalCrossChecker:
    """
    Cross-checks InSAR anomalies against optical satellite imagery.

    In a full implementation this would:
    1. Query the Sentinel-2 archive (via Copernicus Data Space or
       Google Earth Engine) for cloud-free scenes covering the
       flagged site within the specified date range.
    2. Compute change-detection indices (e.g. NDVI difference, band
       ratio anomalies) between pre-event and co-event imagery.
    3. Look for visible surface changes that corroborate the InSAR
       signal: new crevasses, head-scarps, lateral shear margins,
       bulging at the toe, turbid water in downstream rivers.
    4. Return a confidence score and thumbnail of the detected change.

    The current implementation is a stub that returns "unavailable"
    since Sentinel-2 data access is not yet wired into the pipeline.
    """

    def check_surface_change(
        self,
        bbox: tuple[float, float, float, float],
        date_range: tuple[str, str],
    ) -> CrossCheckResult:
        """
        Check for visible surface changes in optical imagery.

        In a real system, this would query Sentinel-2 imagery for
        the area defined by ``bbox`` during ``date_range`` and run
        change detection to look for crevassing, scarps, or bulging
        that would corroborate an InSAR deformation anomaly.

        Parameters
        ----------
        bbox : tuple[float, float, float, float]
            Bounding box (min_lon, min_lat, max_lon, max_lat) in
            WGS84 degrees.
        date_range : tuple[str, str]
            (start_date, end_date) as ISO-8601 strings defining the
            period to search for optical imagery.

        Returns
        -------
        CrossCheckResult
            Result with ``optical_confirmation=None`` and
            ``recommendation="retain"`` (benefit of the doubt when
            optical data is unavailable).
        """
        logger.info(
            "Optical cross-check requested for bbox=%s, dates=%s — "
            "returning unavailable (Sentinel-2 access not configured)",
            bbox,
            date_range,
        )

        return CrossCheckResult(
            quality_score=1.0,
            spatial_consistency=1.0,
            temporal_consistency=1.0,
            optical_confirmation=None,
            recommendation="retain",
            details={
                "optical_status": "unavailable",
                "reason": "Sentinel-2 data access not yet configured",
                "bbox": bbox,
                "date_range": date_range,
            },
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def apply_tier1_filters(
    flags: list[AnomalyFlag],
    displacement_stack: np.ndarray,
    coherence_map: np.ndarray,
    dates: np.ndarray,
    config: dict,
    dem: np.ndarray | None = None,
) -> list[AnomalyFlag]:
    """
    Run all Tier 1 cross-check filters on detected anomaly flags.

    For each flag, evaluates coherence quality, spatial consistency,
    and temporal consistency. Assigns a ``tier1_quality`` dict to
    each flag's ``detection_details`` and updates the recommendation.

    Flags recommended for "remove" are dropped. Flags recommended for
    "demote" have their score halved. Flags recommended for "retain"
    are untouched.

    Integration
    -----------
    Call from the CLI pipeline after Tier 0 detection::

        from glews.crosscheck import apply_tier1_filters

        flags = detect_anomalies(accel_map, lat, lon, config,
                                 dates=dates, displacement=disp)
        flags = apply_tier1_filters(flags, disp, coherence, dates, config)
        # flags now have tier1_quality annotations; false positives removed

    Parameters
    ----------
    flags : list[AnomalyFlag]
        Tier 0 anomaly flags from ``detect.detect_anomalies()``.
    displacement_stack : np.ndarray
        LOS displacement, shape [n_epochs, n_rows, n_cols].
    coherence_map : np.ndarray
        Temporal coherence, shape [n_rows, n_cols], range 0-1.
    dates : np.ndarray
        Acquisition dates as ordinal days, shape [n_epochs].
    config : dict
        Configuration dict. Reads ``detect.tier1_filters`` section
        for thresholds (uses sensible defaults when absent).

    Returns
    -------
    list[AnomalyFlag]
        Filtered flags with ``detection_details["tier1_quality"]``
        populated. Flags recommended for removal are excluded.
    """
    t1_config = config.get("detect", {}).get("tier1_filters", {})

    min_coherence = t1_config.get("min_coherence", 0.3)
    min_spatial = t1_config.get("min_spatial_consistency", 0.5)
    min_temporal = t1_config.get("min_temporal_consistency", 0.3)
    # Thresholds for recommendation
    remove_quality = t1_config.get("remove_below_quality", 0.2)
    demote_quality = t1_config.get("demote_below_quality", 0.5)

    insar_checker = InSARQualityChecker()
    n_rows, n_cols = coherence_map.shape

    logger.info(
        "Running Tier 1 cross-check filters on %d flags "
        "(min_coherence=%.2f, min_spatial=%.2f, min_temporal=%.2f)",
        len(flags), min_coherence, min_spatial, min_temporal,
    )

    # APS detection: when a DEM is provided, run atmospheric phase
    # screen analysis once (scene-wide) and use results per flag.
    aps_result = None
    if dem is not None:
        from glews.atmosphere import APSDetector

        try:
            detector = APSDetector()
            aps_thresholds = t1_config.get("aps_thresholds", {})
            aps_result = detector.flag_aps_contaminated_epochs(
                dates, displacement_stack, dem=dem,
                thresholds=aps_thresholds,
            )
        except Exception:
            logger.warning(
                "APS detection failed; skipping atmospheric check",
                exc_info=True,
            )

    retained: list[AnomalyFlag] = []

    for flag in flags:
        # Build a boolean mask from pixel indices
        flag_mask = np.zeros((n_rows, n_cols), dtype=bool)
        rows = flag.pixel_indices[:, 0]
        cols = flag.pixel_indices[:, 1]
        # Clip to array bounds (safety)
        valid = (rows < n_rows) & (cols < n_cols) & (rows >= 0) & (cols >= 0)
        flag_mask[rows[valid], cols[valid]] = True

        # Run checks
        quality = insar_checker.check_coherence_quality(
            coherence_map, flag_mask, min_coherence=min_coherence,
        )

        # Use the displacement at the flag's window epoch for spatial check
        epoch_idx = min(flag.window_index, displacement_stack.shape[0] - 1)
        spatial = insar_checker.check_spatial_consistency(
            displacement_stack[epoch_idx], flag_mask,
            min_connected_fraction=min_spatial,
        )

        temporal = insar_checker.check_temporal_consistency(
            dates, displacement_stack, flag_mask,
            max_gap_fraction=1 - min_temporal,
        )

        # APS score: if APS analysis ran, check whether this flag's
        # epoch is contaminated.  A clean epoch scores 1.0; a
        # contaminated epoch scores 1 - max(stratified, turbulent).
        aps_score = None
        if aps_result is not None:
            epoch_idx = min(flag.window_index, len(aps_result.contaminated_epochs) - 1)
            strat = float(aps_result.stratified_scores[epoch_idx])
            turb = float(aps_result.turbulent_scores[epoch_idx])
            aps_score = max(0.0, 1.0 - max(strat, turb))

        # Overall quality: geometric mean of scores (3-term without
        # APS, 4-term when APS is available)
        if aps_score is not None:
            composite = (quality * spatial * temporal * aps_score) ** (1.0 / 4.0)
        else:
            composite = (quality * spatial * temporal) ** (1.0 / 3.0)

        # Recommendation
        if composite < remove_quality:
            recommendation = "remove"
        elif composite < demote_quality:
            recommendation = "demote"
        else:
            recommendation = "retain"

        result_details: dict = {
            "composite_score": composite,
            "thresholds": {
                "min_coherence": min_coherence,
                "min_spatial": min_spatial,
                "min_temporal": min_temporal,
                "remove_below": remove_quality,
                "demote_below": demote_quality,
            },
        }
        if aps_score is not None:
            result_details["aps"] = {
                "aps_score": aps_score,
                "epoch_contaminated": bool(aps_result.contaminated_epochs[epoch_idx]),
                "stratified_r2": strat,
                "turbulent_fraction": turb,
            }

        result = CrossCheckResult(
            quality_score=quality,
            spatial_consistency=spatial,
            temporal_consistency=temporal,
            optical_confirmation=None,
            recommendation=recommendation,
            details=result_details,
        )

        # Annotate the flag
        tier1_quality: dict = {
            "quality_score": result.quality_score,
            "spatial_consistency": result.spatial_consistency,
            "temporal_consistency": result.temporal_consistency,
            "optical_confirmation": result.optical_confirmation,
            "composite_score": composite,
            "recommendation": result.recommendation,
        }
        if aps_score is not None:
            tier1_quality["aps_score"] = aps_score
        flag.detection_details["tier1_quality"] = tier1_quality

        logger.info(
            "  Flag %d: quality=%.2f, spatial=%.2f, temporal=%.2f, "
            "composite=%.2f -> %s",
            flag.flag_id, quality, spatial, temporal,
            composite, recommendation,
        )

        if recommendation == "remove":
            continue
        elif recommendation == "demote":
            flag.score *= 0.5
        retained.append(flag)

    # Re-sort by (potentially adjusted) score
    retained.sort(key=lambda f: f.score, reverse=True)

    logger.info(
        "Tier 1 cross-check complete: %d of %d flags retained "
        "(%d removed, %d demoted)",
        len(retained),
        len(flags),
        len(flags) - len(retained),
        sum(
            1
            for f in retained
            if f.detection_details.get("tier1_quality", {}).get("recommendation")
            == "demote"
        ),
    )

    return retained
