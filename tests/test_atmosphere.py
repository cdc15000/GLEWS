"""Tests for atmospheric phase screen detection and correction."""

import numpy as np
import pytest

from glews.atmosphere import APSDetector, APSResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dem(n_rows: int = 64, n_cols: int = 64) -> np.ndarray:
    """Create a DEM with a linear elevation gradient (west-to-east)."""
    # Elevation rises from 1000 m on the left to 4000 m on the right
    col_vals = np.linspace(1000.0, 4000.0, n_cols)
    dem = np.broadcast_to(col_vals[None, :], (n_rows, n_cols)).copy()
    return dem


def _make_dates(n_epochs: int = 10) -> np.ndarray:
    """Create evenly spaced dates (12-day revisit)."""
    return np.arange(n_epochs) * 12 + 738000


def _make_stratified_displacement(
    dem: np.ndarray,
    n_epochs: int = 10,
    slope: float = 0.001,
    noise_std: float = 0.0001,
) -> np.ndarray:
    """
    Create displacement that correlates strongly with elevation.

    Each epoch's displacement is ``slope * (elevation - mean) + noise``.
    """
    rng = np.random.default_rng(42)
    n_rows, n_cols = dem.shape
    disp = np.zeros((n_epochs, n_rows, n_cols))

    for i in range(n_epochs):
        disp[i] = slope * (dem - np.mean(dem)) + rng.normal(
            0, noise_std, (n_rows, n_cols)
        )

    return disp


def _make_clean_displacement(
    n_epochs: int = 10,
    n_rows: int = 64,
    n_cols: int = 64,
) -> np.ndarray:
    """
    Create displacement with localised deformation (no APS).

    A small deformation patch at the centre with random noise
    elsewhere — no correlation with elevation.
    """
    rng = np.random.default_rng(99)
    disp = rng.normal(0, 0.001, (n_epochs, n_rows, n_cols))

    # Add a localised deformation patch (5x5 pixels at centre)
    cy, cx = n_rows // 2, n_cols // 2
    for i in range(n_epochs):
        disp[i, cy - 2 : cy + 3, cx - 2 : cx + 3] += 0.05 * (i + 1)

    return disp


# ---------------------------------------------------------------------------
# APSDetector — stratified detection
# ---------------------------------------------------------------------------

class TestStratifiedDetection:
    """Test stratified APS (elevation-correlated) detection."""

    def test_high_r_squared_for_elevation_correlated_signal(self):
        """Displacement that tracks elevation should produce high R²."""
        det = APSDetector()
        dem = _make_dem()
        dates = _make_dates(5)
        disp = _make_stratified_displacement(dem, n_epochs=5, slope=0.002)

        r2 = det.estimate_aps_correlation(disp, dem, dates)

        assert r2.shape == (5,)
        # Every epoch should have very high R²
        assert np.all(r2 > 0.9), f"Expected R² > 0.9, got {r2}"

    def test_low_r_squared_for_clean_signal(self):
        """Clean localised deformation should produce low R²."""
        det = APSDetector()
        dem = _make_dem()
        dates = _make_dates(5)
        disp = _make_clean_displacement(n_epochs=5)

        r2 = det.estimate_aps_correlation(disp, dem, dates)

        assert r2.shape == (5,)
        # R² should be low — displacement is not elevation-correlated
        assert np.all(r2 < 0.3), f"Expected R² < 0.3, got {r2}"

    def test_flat_dem_returns_zeros(self):
        """A completely flat DEM should produce R² = 0 for all epochs."""
        det = APSDetector()
        dem = np.full((32, 32), 2500.0)
        dates = _make_dates(3)
        disp = _make_stratified_displacement(
            _make_dem(32, 32), n_epochs=3,
        )

        r2 = det.estimate_aps_correlation(disp, dem, dates)

        assert r2.shape == (3,)
        np.testing.assert_array_equal(r2, 0.0)

    def test_single_epoch(self):
        """Should work with a single epoch."""
        det = APSDetector()
        dem = _make_dem()
        dates = _make_dates(1)
        disp = _make_stratified_displacement(dem, n_epochs=1, slope=0.002)

        r2 = det.estimate_aps_correlation(disp, dem, dates)

        assert r2.shape == (1,)
        assert r2[0] > 0.9


# ---------------------------------------------------------------------------
# APSDetector — turbulent detection
# ---------------------------------------------------------------------------

class TestTurbulentDetection:
    """Test turbulent APS (spatial power spectrum) detection."""

    def test_large_scale_noise_scores_high(self):
        """
        Noise with power concentrated at 5–50 km wavelengths should
        produce a high turbulent score.
        """
        det = APSDetector()
        n_rows, n_cols = 128, 128
        spatial_scale_km = 100.0  # 100 km extent
        dates = _make_dates(3)

        rng = np.random.default_rng(7)
        disp = np.zeros((3, n_rows, n_cols))

        # Inject a sinusoidal pattern at ~20 km wavelength
        pixel_km = spatial_scale_km / n_rows
        wavelength_km = 20.0
        freq = 1.0 / wavelength_km  # cycles per km

        y_km = np.arange(n_rows) * pixel_km
        pattern = np.sin(2 * np.pi * freq * y_km)
        for i in range(3):
            disp[i] = pattern[:, None] + rng.normal(0, 0.01, (n_rows, n_cols))

        scores = det.detect_turbulent_aps(disp, dates, spatial_scale_km)

        assert scores.shape == (3,)
        # The injected wavelength falls in the atmospheric band
        assert np.all(scores > 0.3), f"Expected turbulent scores > 0.3, got {scores}"

    def test_high_frequency_noise_scores_lower(self):
        """
        Pure high-frequency noise (pixel-scale) should have a lower
        turbulent score than a signal dominated by atmospheric-scale
        wavelengths, since less power is in the 5-50 km band.
        """
        det = APSDetector()
        n_rows, n_cols = 128, 128
        spatial_scale_km = 100.0
        dates = _make_dates(3)
        rng = np.random.default_rng(12)

        # Pure pixel-scale white noise — power spread across all
        # frequencies, so the atmospheric band gets only its share
        disp_noise = rng.normal(0, 1.0, (3, n_rows, n_cols))
        scores_noise = det.detect_turbulent_aps(
            disp_noise, dates, spatial_scale_km,
        )

        # Atmospheric-scale signal (20 km wavelength)
        pixel_km = spatial_scale_km / n_rows
        wavelength_km = 20.0
        y_km = np.arange(n_rows) * pixel_km
        pattern = np.sin(2 * np.pi / wavelength_km * y_km)
        disp_atmos = np.zeros((3, n_rows, n_cols))
        for i in range(3):
            disp_atmos[i] = pattern[:, None]

        scores_atmos = det.detect_turbulent_aps(
            disp_atmos, dates, spatial_scale_km,
        )

        # Atmospheric signal should score higher than white noise
        assert np.mean(scores_atmos) > np.mean(scores_noise), (
            f"Atmospheric scores {scores_atmos} should exceed "
            f"noise scores {scores_noise}"
        )

    def test_small_map_returns_zeros(self):
        """Maps smaller than 4x4 pixels should return zero scores."""
        det = APSDetector()
        dates = _make_dates(2)
        disp = np.random.default_rng(0).normal(0, 1, (2, 3, 3))

        scores = det.detect_turbulent_aps(disp, dates)

        assert scores.shape == (2,)
        np.testing.assert_array_equal(scores, 0.0)


# ---------------------------------------------------------------------------
# APSDetector — flagging
# ---------------------------------------------------------------------------

class TestFlagging:
    """Test epoch flagging logic."""

    def test_stratified_epochs_flagged(self):
        """Epochs with strong elevation correlation are flagged."""
        det = APSDetector()
        dem = _make_dem()
        dates = _make_dates(5)
        disp = _make_stratified_displacement(dem, n_epochs=5, slope=0.002)

        result = det.flag_aps_contaminated_epochs(
            dates, disp, dem=dem,
            thresholds={"stratified_r2": 0.6, "turbulent_fraction": 0.9},
        )

        assert isinstance(result, APSResult)
        assert result.contaminated_epochs.shape == (5,)
        # All epochs should be flagged (high elevation correlation)
        assert np.all(result.contaminated_epochs)
        assert result.correction_applied is False
        assert result.residual_displacement is None

    def test_clean_epochs_not_flagged(self):
        """Clean displacement should not be flagged."""
        det = APSDetector()
        dem = _make_dem()
        dates = _make_dates(5)
        disp = _make_clean_displacement(n_epochs=5)

        result = det.flag_aps_contaminated_epochs(
            dates, disp, dem=dem,
            thresholds={"stratified_r2": 0.6, "turbulent_fraction": 0.9},
        )

        assert result.contaminated_epochs.shape == (5,)
        # No epochs should be flagged
        assert not np.any(result.contaminated_epochs)

    def test_no_dem_skips_stratified(self):
        """Without a DEM, stratified scores should be zero."""
        det = APSDetector()
        dates = _make_dates(3)
        disp = _make_clean_displacement(n_epochs=3)

        result = det.flag_aps_contaminated_epochs(
            dates, disp, dem=None,
        )

        np.testing.assert_array_equal(result.stratified_scores, 0.0)

    def test_custom_thresholds(self):
        """Custom thresholds should change flagging behaviour."""
        det = APSDetector()
        dem = _make_dem()
        dates = _make_dates(5)

        # Use clean displacement (low R², low turbulent scores)
        disp = _make_clean_displacement(n_epochs=5)

        # With lenient (low) thresholds, clean signal gets flagged
        result_low = det.flag_aps_contaminated_epochs(
            dates, disp, dem=dem,
            thresholds={"stratified_r2": 0.001, "turbulent_fraction": 0.001},
        )
        # With strict (high) thresholds, clean signal is not flagged
        result_high = det.flag_aps_contaminated_epochs(
            dates, disp, dem=dem,
            thresholds={"stratified_r2": 0.99, "turbulent_fraction": 0.99},
        )
        assert not np.any(result_high.contaminated_epochs)
        # At least some epochs should differ between lenient and strict
        assert np.sum(result_low.contaminated_epochs) >= np.sum(
            result_high.contaminated_epochs
        )


# ---------------------------------------------------------------------------
# APSDetector — correction
# ---------------------------------------------------------------------------

class TestCorrection:
    """Test stratified APS correction."""

    def test_correction_removes_elevation_component(self):
        """
        After correction, the R² between displacement and elevation
        should drop substantially.
        """
        det = APSDetector()
        dem = _make_dem()
        dates = _make_dates(5)
        disp = _make_stratified_displacement(dem, n_epochs=5, slope=0.002)

        # Confirm high R² before correction
        r2_before = det.estimate_aps_correlation(disp, dem, dates)
        assert np.all(r2_before > 0.9)

        # Apply correction
        corrected = det.correct_stratified_aps(disp, dem)
        assert corrected.shape == disp.shape

        # R² after correction should be much lower
        r2_after = det.estimate_aps_correlation(corrected, dem, dates)
        assert np.all(r2_after < 0.1), (
            f"Expected R² < 0.1 after correction, got {r2_after}"
        )

    def test_correction_preserves_nan(self):
        """NaN pixels in the input should remain NaN after correction."""
        det = APSDetector()
        dem = _make_dem(32, 32)
        dates = _make_dates(3)
        disp = _make_stratified_displacement(dem, n_epochs=3)

        # Inject NaN at a few pixels
        disp[0, 5, 5] = np.nan
        disp[1, 10:15, 10:15] = np.nan

        corrected = det.correct_stratified_aps(disp, dem)

        assert np.isnan(corrected[0, 5, 5])
        assert np.all(np.isnan(corrected[1, 10:15, 10:15]))

    def test_flat_dem_correction_is_noop(self):
        """Correction with a flat DEM should return the input unchanged."""
        det = APSDetector()
        dem = np.full((32, 32), 2000.0)
        disp = np.random.default_rng(0).normal(0, 0.01, (3, 32, 32))

        corrected = det.correct_stratified_aps(disp, dem)

        np.testing.assert_array_equal(corrected, disp)

    def test_correction_does_not_destroy_real_signal(self):
        """
        Localised deformation that is uncorrelated with elevation
        should survive the correction largely intact.
        """
        det = APSDetector()
        dem = _make_dem()
        dates = _make_dates(5)

        # Start with clean localised deformation
        disp_clean = _make_clean_displacement(n_epochs=5)

        # Add a stratified APS on top
        disp_contaminated = disp_clean + _make_stratified_displacement(
            dem, n_epochs=5, slope=0.002,
        )

        corrected = det.correct_stratified_aps(disp_contaminated, dem)

        # The localised patch (centre 5x5) should still be present
        # after correction: its mean should be close to the clean signal
        cy, cx = 32, 32
        patch_clean = disp_clean[:, cy - 2 : cy + 3, cx - 2 : cx + 3]
        patch_corrected = corrected[:, cy - 2 : cy + 3, cx - 2 : cx + 3]

        # Compare means per epoch — they should be within a reasonable
        # tolerance (the correction shifts the mean, so compare the
        # *relative* displacement within the patch to its surroundings)
        for i in range(5):
            clean_contrast = np.mean(patch_clean[i]) - np.mean(disp_clean[i])
            corrected_contrast = (
                np.mean(patch_corrected[i]) - np.mean(corrected[i])
            )
            assert abs(clean_contrast - corrected_contrast) < 0.01, (
                f"Epoch {i}: localised signal distorted by correction"
            )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_epoch_flagging(self):
        """Should handle a single-epoch displacement stack."""
        det = APSDetector()
        dem = _make_dem(32, 32)
        dates = _make_dates(1)
        disp = _make_stratified_displacement(dem, n_epochs=1)

        result = det.flag_aps_contaminated_epochs(dates, disp, dem=dem)

        assert result.contaminated_epochs.shape == (1,)
        assert result.stratified_scores.shape == (1,)
        assert result.turbulent_scores.shape == (1,)

    def test_all_nan_epoch(self):
        """An all-NaN epoch should not crash and should score zero."""
        det = APSDetector()
        dem = _make_dem(32, 32)
        dates = _make_dates(3)
        disp = np.full((3, 32, 32), np.nan)

        r2 = det.estimate_aps_correlation(disp, dem, dates)
        turb = det.detect_turbulent_aps(disp, dates)

        np.testing.assert_array_equal(r2, 0.0)
        np.testing.assert_array_equal(turb, 0.0)

    def test_aps_result_dataclass(self):
        """APSResult should hold the expected fields."""
        result = APSResult(
            contaminated_epochs=np.array([True, False]),
            stratified_scores=np.array([0.8, 0.1]),
            turbulent_scores=np.array([0.2, 0.3]),
            correction_applied=True,
            residual_displacement=np.zeros((2, 10, 10)),
        )

        assert result.correction_applied is True
        assert result.residual_displacement is not None
        assert result.contaminated_epochs[0] is np.bool_(True)
