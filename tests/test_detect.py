"""Tests for the anomaly detection module."""

import numpy as np
import pytest

from glews.detect import AnomalyFlag, detect_anomalies
from glews.timeseries import AccelerationMap


def _make_accel_map(
    n_windows=20,
    n_rows=50,
    n_cols=60,
    inject_anomaly=False,
):
    """Create a synthetic AccelerationMap for testing."""
    rng = np.random.default_rng(123)

    window_centers = np.linspace(738886, 738886 + 600, n_windows)

    # Background: low-amplitude, normally distributed z-scores
    zscore = rng.normal(0, 0.8, (n_windows, n_rows, n_cols))
    accel = rng.normal(0, 0.01, (n_windows, n_rows, n_cols))
    velocity = rng.normal(0, 0.02, (n_windows, n_rows, n_cols))

    if inject_anomaly:
        # Inject a strong, spatially coherent anomaly in later windows
        for w in range(n_windows - 5, n_windows):
            progress = (w - (n_windows - 5)) / 5
            strength = 3.0 + 4.0 * progress  # z-score 3 to 7
            zscore[w, 20:28, 25:33] = strength
            accel[w, 20:28, 25:33] = 0.05 * progress

    return AccelerationMap(
        window_centers=window_centers,
        acceleration=accel,
        acceleration_zscore=zscore,
        velocity_residual=velocity,
    )


def _make_coords(n_rows=50, n_cols=60):
    """Create coordinate grids matching the test acceleration map."""
    lat = np.linspace(28.3, 28.1, n_rows)[:, None] * np.ones(n_cols)
    lon = np.ones(n_rows)[:, None] * np.linspace(85.8, 86.0, n_cols)
    return lat, lon


def _make_config(sigma=2.5, min_pixels=5, min_area=5000):
    """Create detection config dict."""
    return {
        "detect": {
            "acceleration": {
                "sigma_threshold": sigma,
                "window_size_days": 60,
                "step_days": 12,
                "changepoint": {
                    "model": "rbf",
                    "penalty": "bic",
                    "min_segment_size": 3,
                },
            },
            "clustering": {
                "min_cluster_pixels": min_pixels,
                "max_distance_m": 500,
                "min_area_m2": min_area,
            },
            "voight": {
                "enabled": False,
            },
        }
    }


class TestDetectAnomalies:
    """Tests for the main anomaly detection function."""

    def test_no_anomalies_in_quiet_field(self):
        """A quiet field should produce few or no flags."""
        accel_map = _make_accel_map(inject_anomaly=False)
        lat, lon = _make_coords()
        config = _make_config(sigma=3.0, min_pixels=5)

        flags = detect_anomalies(accel_map, lat, lon, config)

        # Might get a few noise clusters, but scores should be low
        high_score_flags = [f for f in flags if f.score > 5]
        assert len(high_score_flags) == 0

    def test_detects_injected_anomaly(self):
        """Should detect a strong injected anomaly."""
        accel_map = _make_accel_map(inject_anomaly=True)
        lat, lon = _make_coords()
        config = _make_config(sigma=2.5, min_pixels=3)

        flags = detect_anomalies(accel_map, lat, lon, config)

        assert len(flags) > 0

        # The top flag should be near the injected anomaly location
        top = flags[0]
        assert top.peak_zscore > 3.0
        assert top.n_pixels >= 3

        # Check spatial location (injected at rows 20-28, cols 25-33)
        # which maps to approximately lat 28.22, lon 85.88
        assert 28.15 < top.center_lat < 28.28
        assert 85.85 < top.center_lon < 85.95

    def test_flag_properties(self):
        """Detected flags should have valid properties."""
        accel_map = _make_accel_map(inject_anomaly=True)
        lat, lon = _make_coords()
        config = _make_config(sigma=2.5, min_pixels=3)

        flags = detect_anomalies(accel_map, lat, lon, config)
        assert len(flags) > 0

        for flag in flags:
            assert isinstance(flag, AnomalyFlag)
            assert flag.score > 0
            assert flag.peak_zscore > 0
            assert flag.n_pixels > 0
            assert flag.area_m2 > 0
            assert -90 <= flag.center_lat <= 90
            assert -180 <= flag.center_lon <= 180
            assert flag.pixel_indices.shape[1] == 2

    def test_flags_sorted_by_score(self):
        """Output should be sorted by score, highest first."""
        accel_map = _make_accel_map(inject_anomaly=True)
        lat, lon = _make_coords()
        config = _make_config(sigma=2.0, min_pixels=3)

        flags = detect_anomalies(accel_map, lat, lon, config)

        if len(flags) > 1:
            scores = [f.score for f in flags]
            assert scores == sorted(scores, reverse=True)

    def test_min_area_filter(self):
        """Clusters below minimum area should be filtered out."""
        accel_map = _make_accel_map(inject_anomaly=True)
        lat, lon = _make_coords()

        # Very high minimum area should filter everything
        config = _make_config(sigma=2.5, min_pixels=3, min_area=1e9)

        flags = detect_anomalies(accel_map, lat, lon, config)
        assert len(flags) == 0

    def test_high_threshold_fewer_flags(self):
        """Higher sigma threshold should produce fewer flags."""
        accel_map = _make_accel_map(inject_anomaly=True)
        lat, lon = _make_coords()

        flags_low = detect_anomalies(
            accel_map, lat, lon, _make_config(sigma=2.0, min_pixels=3)
        )
        flags_high = detect_anomalies(
            accel_map, lat, lon, _make_config(sigma=5.0, min_pixels=3)
        )

        assert len(flags_high) <= len(flags_low)


class TestSpatialCoherence:
    """Tests for spatial coherence integration in detect_anomalies."""

    def _make_displacement(self, accel_map, n_rows=50, n_cols=60):
        """Create synthetic displacement with coherent signal at anomaly location."""
        rng = np.random.default_rng(99)
        n_epochs = len(accel_map.window_centers)
        dates = accel_map.window_centers.copy()
        displacement = rng.normal(0, 0.001, (n_epochs, n_rows, n_cols))
        # Add strongly correlated ramp at the anomaly location (rows 20-28,
        # cols 25-33 matches _make_accel_map inject location) so spatial
        # coherence analysis finds a cluster.
        for i in range(n_epochs):
            displacement[i, 20:28, 25:33] += 0.01 * i
        return dates, displacement

    def test_spatial_coherence_added_when_enabled(self):
        """When spatial is enabled, every flag should have spatial_coherence."""
        accel_map = _make_accel_map(inject_anomaly=True)
        lat, lon = _make_coords()
        dates, displacement = self._make_displacement(accel_map)

        config = _make_config(sigma=2.5, min_pixels=3)
        config["detect"]["spatial"] = {
            "enabled": True,
            "max_distance_m": 500,
            "min_correlation": 0.5,
            "min_cluster_size": 3,
            "min_coherence": 0.3,
        }

        flags = detect_anomalies(
            accel_map, lat, lon, config,
            dates=dates, displacement=displacement,
        )

        assert len(flags) > 0
        for flag in flags:
            assert "spatial_coherence" in flag.detection_details
            assert 0.0 <= flag.detection_details["spatial_coherence"] <= 1.0

        # At least one flag should have non-zero coherence to confirm the
        # spatial code path actually matched clusters to anomaly flags.
        max_coherence = max(
            f.detection_details["spatial_coherence"] for f in flags
        )
        assert max_coherence > 0, "No flag matched a spatial cluster"

    def test_spatial_coherence_absent_when_disabled(self):
        """When spatial is not enabled, no spatial_coherence key should appear."""
        accel_map = _make_accel_map(inject_anomaly=True)
        lat, lon = _make_coords()
        config = _make_config(sigma=2.5, min_pixels=3)

        flags = detect_anomalies(accel_map, lat, lon, config)

        assert len(flags) > 0
        for flag in flags:
            assert "spatial_coherence" not in flag.detection_details


class TestTimeseriesPopulation:
    """Tests for displacement time-series population on flags."""

    def _make_displacement(self, accel_map, n_rows=50, n_cols=60):
        """Create synthetic displacement matching the accel_map grid."""
        rng = np.random.default_rng(99)
        n_epochs = len(accel_map.window_centers)
        dates = accel_map.window_centers.copy()
        displacement = rng.normal(0, 0.01, (n_epochs, n_rows, n_cols))
        # Add a ramp at the anomaly location
        for i in range(n_epochs):
            displacement[i, 20:28, 25:33] += 0.002 * i
        return dates, displacement

    def test_timeseries_populated_with_displacement(self):
        """When dates and displacement are supplied, surviving flags
        have timeseries dicts with parallel dates/values arrays."""
        accel_map = _make_accel_map(inject_anomaly=True)
        lat, lon = _make_coords()
        config = _make_config(sigma=2.5, min_pixels=3)
        dates, displacement = self._make_displacement(accel_map)

        flags = detect_anomalies(
            accel_map, lat, lon, config,
            dates=dates, displacement=displacement,
        )

        assert len(flags) > 0
        top = flags[0]
        assert top.timeseries is not None
        assert len(top.timeseries["dates"]) == len(top.timeseries["values"])
        assert len(top.timeseries["dates"]) > 0

        # Dates should be ISO format strings
        for d in top.timeseries["dates"]:
            assert isinstance(d, str)
            assert len(d) == 10  # YYYY-MM-DD

        # Values should be finite floats
        for v in top.timeseries["values"]:
            assert isinstance(v, float)
            assert np.isfinite(v)

    def test_timeseries_none_without_displacement(self):
        """Without dates/displacement, timeseries stays None."""
        accel_map = _make_accel_map(inject_anomaly=True)
        lat, lon = _make_coords()
        config = _make_config(sigma=2.5, min_pixels=3)

        flags = detect_anomalies(accel_map, lat, lon, config)

        assert len(flags) > 0
        for flag in flags:
            assert flag.timeseries is None
