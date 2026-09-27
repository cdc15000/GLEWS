"""Tests for the NISAR data-loading module.

These tests exercise the pure numerical helpers in glews.nisar (velocity
estimation) without touching HDF5 I/O — the GUNW/GOFF file readers are
not covered here since they require real NISAR product files.
"""

from __future__ import annotations

import numpy as np

from glews.nisar import NISARTimeseries, _compute_velocity


class TestComputeVelocity:
    """Tests for the vectorized linear-velocity estimator."""

    def _make_dates(self, n=30, start=738886, step=12):
        return np.arange(start, start + n * step, step)

    def test_linear_displacement_recovers_velocity(self):
        """A pure linear ramp should recover the exact velocity."""
        dates = self._make_dates(30)
        t_yr = (dates - dates[0]).astype(float) / 365.25

        true_velocity = 0.06  # m/yr
        n_rows, n_cols = 5, 5
        displacement = np.zeros((len(dates), n_rows, n_cols))
        for i, t in enumerate(t_yr):
            displacement[i] = true_velocity * t

        velocity = _compute_velocity(dates, displacement)

        assert velocity.shape == (n_rows, n_cols)
        assert np.allclose(velocity, true_velocity, atol=1e-6)

    def test_negative_velocity(self):
        """Should correctly recover a negative (subsiding) velocity."""
        dates = self._make_dates(20)
        t_yr = (dates - dates[0]).astype(float) / 365.25

        true_velocity = -0.12
        displacement = np.zeros((len(dates), 3, 3))
        for i, t in enumerate(t_yr):
            displacement[i] = true_velocity * t

        velocity = _compute_velocity(dates, displacement)
        assert np.allclose(velocity, true_velocity, atol=1e-6)

    def test_insufficient_data_is_nan(self):
        """Pixels with fewer than 3 valid observations should be NaN."""
        dates = self._make_dates(5)
        displacement = np.full((5, 2, 2), np.nan)
        # Only 2 valid observations for every pixel — below the minimum of 3
        displacement[0] = 0.0
        displacement[1] = 0.01

        velocity = _compute_velocity(dates, displacement)
        assert np.all(np.isnan(velocity))

    def test_mixed_valid_and_nan_pixels(self):
        """Should compute velocity per-pixel independently, skipping
        pixels that don't have enough valid samples."""
        dates = self._make_dates(10)
        t_yr = (dates - dates[0]).astype(float) / 365.25

        displacement = np.zeros((len(dates), 2, 2))
        displacement[:, 0, 0] = 0.05 * t_yr  # fully valid, velocity 0.05
        displacement[:, 0, 1] = np.nan       # entirely missing
        displacement[:, 1, 0] = 0.05 * t_yr
        displacement[:2, 1, 0] = np.nan      # still enough valid points

        velocity = _compute_velocity(dates, displacement)

        assert np.isclose(velocity[0, 0], 0.05, atol=1e-6)
        assert np.isnan(velocity[0, 1])
        assert np.isclose(velocity[1, 0], 0.05, atol=1e-6)

    def test_noisy_data_approximate_recovery(self, rng):
        """With small noise, recovered velocity should be close to truth."""
        dates = self._make_dates(50)
        t_yr = (dates - dates[0]).astype(float) / 365.25

        true_velocity = 0.03
        displacement = np.zeros((len(dates), 4, 4))
        for i, t in enumerate(t_yr):
            displacement[i] = true_velocity * t
        displacement += rng.normal(0, 0.001, displacement.shape)

        velocity = _compute_velocity(dates, displacement)
        assert np.allclose(velocity, true_velocity, atol=0.01)


class TestNISARTimeseriesDataclass:
    """Sanity checks on the NISARTimeseries container itself."""

    def test_synthetic_fixture_shapes_are_consistent(self, synthetic_nisar_timeseries):
        ts = synthetic_nisar_timeseries
        assert isinstance(ts, NISARTimeseries)

        n_dates, n_rows, n_cols = ts.displacement.shape
        assert ts.dates.shape == (n_dates,)
        assert len(ts.date_strings) == n_dates
        assert ts.velocity.shape == (n_rows, n_cols)
        assert ts.coherence.shape == (n_rows, n_cols)
        assert ts.latitude.shape == (n_rows, n_cols)
        assert ts.longitude.shape == (n_rows, n_cols)
        assert ts.source == "GUNW"

    def test_synthetic_fixture_velocity_is_positive(self, synthetic_nisar_timeseries):
        # Fixture was built with a constant positive velocity trend.
        ts = synthetic_nisar_timeseries
        valid = np.isfinite(ts.velocity)
        assert valid.any()
        assert np.nanmean(ts.velocity) > 0
