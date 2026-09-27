"""
Atmospheric Phase Screen (APS) detection and correction for InSAR
displacement time series.

InSAR measurements are contaminated by atmospheric path-delay
variations that masquerade as ground displacement. These atmospheric
phase screens (APS) come in two flavours:

    1. **Stratified APS** — caused by vertical stratification of water
       vapour in the troposphere. The delay correlates with topographic
       elevation: high-altitude pixels accumulate more (or less) path
       delay than low-altitude ones. Detected by regressing
       displacement against a DEM per epoch and looking for high R².

    2. **Turbulent APS** — caused by turbulent mixing in the
       troposphere. Produces spatially correlated noise at scales of
       5–50 km, distinct from tectonic or glacial deformation which
       is typically localised to < 1 km. Detected by examining the
       spatial power spectrum of each displacement epoch.

This module provides:
    - Detection of both APS types per epoch
    - Flagging of contaminated epochs
    - Correction of the stratified component (elevation regression)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class APSResult:
    """
    Results from atmospheric phase screen analysis.

    Attributes
    ----------
    contaminated_epochs : np.ndarray
        Boolean array, shape [n_epochs]. True where the epoch is
        flagged as APS-contaminated (by either stratified or turbulent
        criteria, or both).
    stratified_scores : np.ndarray
        R² of the displacement-vs-elevation regression per epoch,
        shape [n_epochs]. High values (e.g. > 0.6) indicate
        stratified APS dominates that acquisition.
    turbulent_scores : np.ndarray
        Fraction of spatial spectral power in the 5–50 km band
        relative to total power, per epoch, shape [n_epochs]. High
        values indicate turbulent atmospheric noise.
    correction_applied : bool
        Whether stratified APS correction was applied to produce
        ``residual_displacement``.
    residual_displacement : np.ndarray | None
        Displacement stack after removing the elevation-correlated
        component, shape [n_epochs, n_rows, n_cols]. None when no
        correction was applied.
    """

    contaminated_epochs: np.ndarray
    stratified_scores: np.ndarray
    turbulent_scores: np.ndarray
    correction_applied: bool
    residual_displacement: np.ndarray | None = None


class APSDetector:
    """
    Detect and correct atmospheric phase screens in InSAR
    displacement stacks.
    """

    def estimate_aps_correlation(
        self,
        displacement_map: np.ndarray,
        dem: np.ndarray,
        dates: np.ndarray,
    ) -> np.ndarray:
        """
        Compute R² between displacement and elevation for each epoch.

        A high R² indicates that displacement at that epoch is
        dominated by a stratified atmospheric delay that scales with
        topographic height — not real ground motion.

        Parameters
        ----------
        displacement_map : np.ndarray
            Displacement stack, shape [n_epochs, n_rows, n_cols].
        dem : np.ndarray
            Digital elevation model, shape [n_rows, n_cols], in meters.
        dates : np.ndarray
            Acquisition dates as ordinal days, shape [n_epochs].
            Used for logging; the regression is purely spatial.

        Returns
        -------
        np.ndarray
            R² per epoch, shape [n_epochs]. Values in [0, 1].
        """
        n_epochs = displacement_map.shape[0]
        r_squared = np.zeros(n_epochs)

        elev = dem.ravel().astype(float)
        valid_elev = np.isfinite(elev)

        # Check for flat DEM — no elevation variation means R² is
        # undefined; return zeros (no stratified signal detectable).
        elev_range = np.ptp(elev[valid_elev]) if np.any(valid_elev) else 0.0
        if elev_range < 1e-6:
            logger.info(
                "Flat DEM detected (range %.2e m) — stratified APS "
                "scores will be zero for all %d epochs",
                elev_range, n_epochs,
            )
            return r_squared

        for i in range(n_epochs):
            disp = displacement_map[i].ravel().astype(float)
            valid = valid_elev & np.isfinite(disp)
            n_valid = np.count_nonzero(valid)
            if n_valid < 3:
                continue

            e = elev[valid]
            d = disp[valid]

            # Linear regression: d = a * e + b
            e_mean = np.mean(e)
            d_mean = np.mean(d)
            e_centered = e - e_mean
            d_centered = d - d_mean

            ss_ee = np.sum(e_centered ** 2)
            if ss_ee < 1e-12:
                continue

            ss_ed = np.sum(e_centered * d_centered)
            slope = ss_ed / ss_ee
            fitted = slope * e_centered + d_mean

            ss_res = np.sum((d - fitted) ** 2)
            ss_tot = np.sum(d_centered ** 2)

            r_squared[i] = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
            # Clamp to [0, 1] for numerical safety
            r_squared[i] = max(0.0, min(1.0, r_squared[i]))

        logger.info(
            "Stratified APS correlation: max R²=%.3f, "
            "mean R²=%.3f across %d epochs",
            np.max(r_squared), np.mean(r_squared), n_epochs,
        )

        return r_squared

    def detect_turbulent_aps(
        self,
        displacement_map: np.ndarray,
        dates: np.ndarray,
        spatial_scale_km: float = 10.0,
    ) -> np.ndarray:
        """
        Detect turbulent APS via spatial power spectrum analysis.

        Turbulent atmospheric noise concentrates spectral power at
        wavelengths of 5–50 km, whereas localised glacial or
        landslide deformation has power at < 1 km wavelengths. This
        method computes the 2-D power spectrum of each epoch's
        displacement field and measures the fraction of total power
        in the atmospheric band (5–50 km).

        The pixel spacing is derived from ``spatial_scale_km``, which
        gives the full extent of the displacement map in kilometres.
        Actual projects would read pixel spacing from metadata; here
        we parameterise it for flexibility.

        Parameters
        ----------
        displacement_map : np.ndarray
            Displacement stack, shape [n_epochs, n_rows, n_cols].
        dates : np.ndarray
            Acquisition dates as ordinal days, shape [n_epochs].
        spatial_scale_km : float
            Physical extent of the displacement map along one axis,
            in kilometres. Used to convert spatial frequencies to
            wavelengths. Default 10 km.

        Returns
        -------
        np.ndarray
            Turbulent APS score per epoch, shape [n_epochs].
            Fraction of spectral power in the 5–50 km band.
        """
        n_epochs, n_rows, n_cols = displacement_map.shape
        scores = np.zeros(n_epochs)

        if n_rows < 4 or n_cols < 4:
            logger.info(
                "Displacement map too small (%d x %d) for spectral "
                "analysis — turbulent APS scores will be zero",
                n_rows, n_cols,
            )
            return scores

        # Pixel spacing in km
        pixel_km_y = spatial_scale_km / n_rows
        pixel_km_x = spatial_scale_km / n_cols

        # Frequency grids (cycles per km)
        freq_y = np.fft.fftfreq(n_rows, d=pixel_km_y)
        freq_x = np.fft.fftfreq(n_cols, d=pixel_km_x)
        fy, fx = np.meshgrid(freq_y, freq_x, indexing="ij")
        freq_mag = np.sqrt(fy ** 2 + fx ** 2)

        # Wavelength = 1 / frequency (km). Avoid division by zero at DC.
        with np.errstate(divide="ignore", invalid="ignore"):
            wavelength_km = np.where(freq_mag > 0, 1.0 / freq_mag, np.inf)

        # Atmospheric band: 5–50 km wavelengths
        atmos_mask = (wavelength_km >= 5.0) & (wavelength_km <= 50.0)

        for i in range(n_epochs):
            epoch = displacement_map[i].copy()

            # Replace NaN with zero for FFT
            nan_mask = ~np.isfinite(epoch)
            if np.all(nan_mask):
                continue
            epoch[nan_mask] = 0.0

            # Remove mean to suppress DC component
            epoch -= np.mean(epoch)

            # 2-D FFT and power spectrum
            spectrum = np.fft.fft2(epoch)
            power = np.abs(spectrum) ** 2

            total_power = np.sum(power) - power[0, 0]  # exclude DC
            if total_power < 1e-20:
                continue

            atmos_power = np.sum(power[atmos_mask])
            scores[i] = atmos_power / total_power

        logger.info(
            "Turbulent APS analysis: max score=%.3f, "
            "mean score=%.3f across %d epochs",
            np.max(scores), np.mean(scores), n_epochs,
        )

        return scores

    def flag_aps_contaminated_epochs(
        self,
        dates: np.ndarray,
        displacement_stack: np.ndarray,
        dem: np.ndarray | None = None,
        thresholds: dict | None = None,
    ) -> APSResult:
        """
        Identify epochs contaminated by atmospheric phase screens.

        Runs both stratified and turbulent APS detection and flags
        epochs that exceed configurable thresholds for either type.

        Parameters
        ----------
        dates : np.ndarray
            Acquisition dates as ordinal days, shape [n_epochs].
        displacement_stack : np.ndarray
            Displacement in meters, shape [n_epochs, n_rows, n_cols].
        dem : np.ndarray or None
            Digital elevation model, shape [n_rows, n_cols]. Required
            for stratified APS detection. If None, only turbulent
            detection is run.
        thresholds : dict or None
            Override default thresholds. Recognised keys:

            - ``stratified_r2`` (float): R² above which an epoch is
              flagged as stratified-APS-contaminated. Default 0.6.
            - ``turbulent_fraction`` (float): atmospheric-band power
              fraction above which an epoch is flagged as turbulent-
              APS-contaminated. Default 0.4.
            - ``spatial_scale_km`` (float): physical extent of the
              map for spectral analysis. Default 10.0.

        Returns
        -------
        APSResult
            Detection results with per-epoch scores and flags.
        """
        if thresholds is None:
            thresholds = {}

        strat_thresh = thresholds.get("stratified_r2", 0.6)
        turb_thresh = thresholds.get("turbulent_fraction", 0.4)
        scale_km = thresholds.get("spatial_scale_km", 10.0)

        n_epochs = displacement_stack.shape[0]

        # Stratified detection
        if dem is not None:
            strat_scores = self.estimate_aps_correlation(
                displacement_stack, dem, dates,
            )
        else:
            strat_scores = np.zeros(n_epochs)

        # Turbulent detection
        turb_scores = self.detect_turbulent_aps(
            displacement_stack, dates, spatial_scale_km=scale_km,
        )

        # Flag contaminated epochs
        contaminated = (strat_scores >= strat_thresh) | (turb_scores >= turb_thresh)

        n_flagged = int(np.sum(contaminated))
        logger.info(
            "APS flagging complete: %d of %d epochs contaminated "
            "(%d stratified, %d turbulent)",
            n_flagged, n_epochs,
            int(np.sum(strat_scores >= strat_thresh)),
            int(np.sum(turb_scores >= turb_thresh)),
        )

        return APSResult(
            contaminated_epochs=contaminated,
            stratified_scores=strat_scores,
            turbulent_scores=turb_scores,
            correction_applied=False,
            residual_displacement=None,
        )

    def correct_stratified_aps(
        self,
        displacement_map: np.ndarray,
        dem: np.ndarray,
    ) -> np.ndarray:
        """
        Remove the elevation-correlated component from displacement.

        For each epoch, fits a linear model d = a * elevation + b
        and subtracts the fitted surface, leaving only the residual
        displacement that is uncorrelated with topography.

        Parameters
        ----------
        displacement_map : np.ndarray
            Displacement stack, shape [n_epochs, n_rows, n_cols].
        dem : np.ndarray
            Digital elevation model, shape [n_rows, n_cols], in meters.

        Returns
        -------
        np.ndarray
            Corrected displacement stack, same shape as input.
            Pixels that are NaN in the input remain NaN.
        """
        n_epochs, n_rows, n_cols = displacement_map.shape
        corrected = displacement_map.copy()

        elev = dem.ravel().astype(float)
        valid_elev = np.isfinite(elev)

        # Flat DEM: nothing to correct
        elev_range = np.ptp(elev[valid_elev]) if np.any(valid_elev) else 0.0
        if elev_range < 1e-6:
            logger.info(
                "Flat DEM — no stratified correction applied",
            )
            return corrected

        for i in range(n_epochs):
            disp = displacement_map[i].ravel().astype(float)
            valid = valid_elev & np.isfinite(disp)
            n_valid = np.count_nonzero(valid)
            if n_valid < 3:
                continue

            e = elev[valid]
            d = disp[valid]

            # Linear regression: d = slope * e + intercept
            e_mean = np.mean(e)
            d_mean = np.mean(d)
            e_centered = e - e_mean

            ss_ee = np.sum(e_centered ** 2)
            if ss_ee < 1e-12:
                continue

            slope = np.sum(e_centered * (d - d_mean)) / ss_ee
            intercept = d_mean - slope * e_mean

            # Subtract elevation-correlated component from all valid
            # pixels, then restore the mean displacement so we only
            # remove the elevation-dependent *shape*, not the offset.
            correction = slope * elev + intercept
            corrected_flat = disp.copy()
            corrected_flat[valid_elev] -= correction[valid_elev]
            corrected_flat[valid_elev] += d_mean

            corrected[i] = corrected_flat.reshape(n_rows, n_cols)

            # Preserve original NaN pattern
            nan_mask = ~np.isfinite(displacement_map[i])
            corrected[i][nan_mask] = np.nan

        logger.info(
            "Stratified APS correction applied to %d epochs", n_epochs,
        )

        return corrected
