"""Tests for the synthetic data generator."""

import numpy as np
import pytest

from glews.synthetic import SyntheticConfig, generate_synthetic_scene


class TestSyntheticGenerator:
    """Tests for synthetic scene generation."""

    def test_default_config_produces_valid_output(self):
        """Default config should produce a valid DisplacementTimeseries."""
        ts = generate_synthetic_scene()

        assert ts.displacement.ndim == 3
        assert ts.dates.ndim == 1
        assert len(ts.date_strings) == len(ts.dates)
        assert ts.latitude.shape == ts.displacement.shape[1:]
        assert ts.longitude.shape == ts.displacement.shape[1:]
        assert ts.temporal_coherence.shape == ts.displacement.shape[1:]

    def test_dimensions_match_config(self):
        """Output dimensions should match configuration."""
        cfg = SyntheticConfig(
            n_rows=50,
            n_cols=40,
            start_date="2025-06-01",
            end_date="2026-06-01",
            revisit_days=12,
            failure_zones=[],
        )
        ts = generate_synthetic_scene(cfg)

        assert ts.displacement.shape[1] == 50
        assert ts.displacement.shape[2] == 40

        # Approximately 365/12 ≈ 30 dates
        assert 25 <= ts.displacement.shape[0] <= 35

    def test_failure_signal_increases_displacement(self):
        """The failure zone should show more displacement than background."""
        cfg = SyntheticConfig(
            n_rows=100,
            n_cols=100,
            failure_zones=[{
                "center_row": 50,
                "center_col": 50,
                "radius_pixels": 10,
                "onset_days_before_end": 90,
                "max_acceleration_m_yr2": 0.2,
                "ramp_type": "exponential",
            }],
        )
        ts = generate_synthetic_scene(cfg)

        # The failure signal is injected as additional negative displacement
        # (downslope, away from satellite). Compare the difference between
        # the last and first time steps: the failure zone should show more
        # total displacement change than background.
        first = ts.displacement[0]
        last = ts.displacement[-1]
        diff = last - first

        failure_mean = np.nanmean(np.abs(diff[40:60, 40:60]))
        bg_mean = np.nanmean(np.abs(diff[0:20, 0:20]))

        assert failure_mean > bg_mean

    def test_coherence_mask_applied(self):
        """Low-coherence pixels should be NaN."""
        ts = generate_synthetic_scene()

        low_coh = ts.temporal_coherence < 0.5
        if np.any(low_coh):
            # All time steps should be NaN for low-coherence pixels
            for t in range(len(ts.dates)):
                nan_at_t = np.isnan(ts.displacement[t])
                assert np.all(nan_at_t[low_coh])

    def test_coordinates_in_expected_range(self):
        """Coordinates should be in the expected geographic range."""
        ts = generate_synthetic_scene()

        assert np.nanmin(ts.latitude) >= 28.0
        assert np.nanmax(ts.latitude) <= 28.4
        assert np.nanmin(ts.longitude) >= 85.7
        assert np.nanmax(ts.longitude) <= 86.1

    def test_no_failure_zones(self):
        """Should work with no failure zones (background only)."""
        cfg = SyntheticConfig(failure_zones=[])
        ts = generate_synthetic_scene(cfg)

        assert ts.displacement.ndim == 3
        # Should have finite values where coherence is good
        good = ts.temporal_coherence >= 0.5
        assert np.any(np.isfinite(ts.displacement[-1, good]))
