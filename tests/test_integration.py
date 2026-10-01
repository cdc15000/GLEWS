"""End-to-end integration tests for the GLEWS pipeline.

Each test exercises multiple modules together using only synthetic data
--- no network access, no real satellite products.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from glews.alerts import AlertDispatcher
from glews.dashboard import load_flags
from glews.detect import AnomalyFlag, detect_anomalies
from glews.report import generate_report
from glews.synthetic import SyntheticConfig, generate_synthetic_scene
from glews.timeseries import (
    AccelerationMap,
    compute_acceleration_map,
    detect_step_changes,
    bocpd_changepoints,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_config() -> dict:
    """Minimal detection config matching the shape expected by
    ``detect_anomalies`` and ``generate_report``."""
    return {
        "site": {
            "name": "Integration Test Site",
            "event_date": "2026-08-26",
        },
        "detect": {
            "n_harmonics": 2,
            "acceleration": {
                "window_size_days": 60,
                "step_days": 12,
                "sigma_threshold": 2.5,
                "changepoint": {
                    "model": "rbf",
                    "penalty": "bic",
                    "min_segment_size": 3,
                },
            },
            "clustering": {
                "min_cluster_pixels": 5,
                "max_distance_m": 200,
                "min_area_m2": 10000,
            },
            "voight": {
                "enabled": True,
                "min_points": 5,
                "r_squared_threshold": 0.7,
            },
        },
    }


@pytest.fixture(scope="module")
def default_pipeline_result():
    """Run the default synthetic scene pipeline once and cache the result
    for all tests in this module.  The default Nepal config (200x250, 49
    dates) runs in ~0.2 s and reliably produces detectable flags."""
    ts = generate_synthetic_scene()  # default Nepal config
    cfg = _default_config()
    det_cfg = cfg["detect"]

    accel_map = compute_acceleration_map(
        ts.dates,
        ts.displacement,
        window_size_days=det_cfg["acceleration"]["window_size_days"],
        step_days=det_cfg["acceleration"]["step_days"],
        n_harmonics=det_cfg.get("n_harmonics", 2),
    )

    flags = detect_anomalies(accel_map, ts.latitude, ts.longitude, cfg)

    return {
        "ts": ts,
        "cfg": cfg,
        "accel_map": accel_map,
        "flags": flags,
    }


# ---------------------------------------------------------------------------
# 1. Full pipeline on synthetic data
# ---------------------------------------------------------------------------


class TestFullPipelineSynthetic:
    """Generate synthetic data, compute acceleration, detect anomalies,
    and verify the injected failure zone is detected."""

    def test_full_pipeline_synthetic(self, default_pipeline_result):
        r = default_pipeline_result
        accel_map = r["accel_map"]
        flags = r["flags"]

        assert accel_map.acceleration.ndim == 3
        assert accel_map.acceleration_zscore.shape == accel_map.acceleration.shape

        # The injected failure zone should produce at least one flag
        assert len(flags) > 0, "No flags detected; injected failure zone was missed"

        # The top flag should have a meaningful anomaly score
        top = flags[0]
        assert top.peak_zscore > 2.0
        assert top.score > 0

        # The injected failure zone is at row=80, col=120 in the default
        # 200x250 grid (lat ~28.22, lon ~85.90).  At least one high-scoring
        # flag should be within ~0.1 degree of that location.
        near_injection = [
            f for f in flags
            if abs(f.center_lat - 28.22) < 0.1 and abs(f.center_lon - 85.90) < 0.1
        ]
        assert len(near_injection) > 0, (
            "No flags detected near the injected failure zone (28.22N, 85.90E)"
        )


# ---------------------------------------------------------------------------
# 2. BOCPD integration
# ---------------------------------------------------------------------------


class TestBOCPDIntegration:
    """Run detection with BOCPD enabled on synthetic data with a step
    change and verify BOCPD flags are produced."""

    def test_bocpd_integration(self):
        # Use the default scene (200x250) but restrict BOCPD to pixels
        # with large displacement range so it runs quickly.
        ts = generate_synthetic_scene()

        cfg = _default_config()
        cfg["detect"]["changepoint"] = {
            "method": "bocpd",
            "hazard_rate": 1 / 50,
            "threshold": 0.25,
            # Only run BOCPD on pixels with significant displacement
            # range, which keeps the O(n^2)-per-pixel cost manageable.
            "min_displacement_range_m": 0.01,
        }
        cfg["detect"]["clustering"]["min_cluster_pixels"] = 3
        cfg["detect"]["clustering"]["min_area_m2"] = 5000

        det_cfg = cfg["detect"]
        accel_map = compute_acceleration_map(
            ts.dates,
            ts.displacement,
            window_size_days=det_cfg["acceleration"]["window_size_days"],
            step_days=det_cfg["acceleration"]["step_days"],
            n_harmonics=det_cfg.get("n_harmonics", 2),
        )

        flags = detect_anomalies(
            accel_map, ts.latitude, ts.longitude, cfg,
            dates=ts.dates, displacement=ts.displacement,
        )

        # At least one flag should carry the "bocpd" tag
        bocpd_flags = [
            f for f in flags
            if f.detection_details.get("tag") == "bocpd"
        ]
        assert len(bocpd_flags) > 0, (
            "No BOCPD flags produced; expected at least one on synthetic data"
        )

    def test_bocpd_changepoints_single_pixel(self):
        """Sanity check: BOCPD on a single pixel time series with a
        known mean shift should detect a changepoint."""
        rng = np.random.default_rng(99)
        n = 60
        dates = np.arange(738886, 738886 + n * 12, 12)

        # Constant velocity for first half, then a jump
        disp = np.zeros(n)
        vel_1 = 0.0001  # m/day
        vel_2 = 0.001   # 10x velocity after changepoint
        for i in range(n):
            t = (dates[i] - dates[0])
            if i < n // 2:
                disp[i] = vel_1 * t
            else:
                mid_t = dates[n // 2] - dates[0]
                disp[i] = vel_1 * mid_t + vel_2 * (t - mid_t)
        disp += rng.normal(0, 0.0005, n)

        result = bocpd_changepoints(dates, disp, hazard_rate=1 / 30, threshold=0.25)

        assert len(result.changepoint_indices) > 0, "No changepoints detected"
        assert result.changepoint_probabilities.shape == (n,)
        assert result.run_length_posterior.shape == (n, n + 1)


# ---------------------------------------------------------------------------
# 3. Step-change integration
# ---------------------------------------------------------------------------


class TestStepChangeIntegration:
    """Run detection with step-change detector on synthetic data and
    verify step-change flags are produced."""

    def test_step_change_integration(self):
        # Use default scene and supply dates/displacement to enable
        # step-change detection.  Lower the thresholds to catch the
        # synthetic failure signal, which is smooth rather than abrupt.
        ts = generate_synthetic_scene()

        cfg = _default_config()
        cfg["detect"]["step_change"] = {
            "sigma_threshold": 3.0,
            "min_displacement_m": 0.005,
        }
        cfg["detect"]["clustering"]["min_cluster_pixels"] = 3
        cfg["detect"]["clustering"]["min_area_m2"] = 5000
        # Disable Voight: its window_index lookup is only valid for
        # acceleration-based flags, not step-change flags.
        cfg["detect"]["voight"]["enabled"] = False

        det_cfg = cfg["detect"]
        accel_map = compute_acceleration_map(
            ts.dates,
            ts.displacement,
            window_size_days=det_cfg["acceleration"]["window_size_days"],
            step_days=det_cfg["acceleration"]["step_days"],
            n_harmonics=det_cfg.get("n_harmonics", 2),
        )

        flags = detect_anomalies(
            accel_map, ts.latitude, ts.longitude, cfg,
            dates=ts.dates, displacement=ts.displacement,
        )

        step_flags = [
            f for f in flags
            if f.detection_details.get("tag") == "step_change"
        ]
        assert len(step_flags) > 0, (
            "No step-change flags produced; expected at least one"
        )

    def test_detect_step_changes_directly(self):
        """The timeseries.detect_step_changes function should flag a
        large inter-epoch jump."""
        rng = np.random.default_rng(77)
        n_dates, n_rows, n_cols = 30, 10, 10
        dates = np.arange(738886, 738886 + n_dates * 12, 12)

        displacement = rng.normal(0, 0.001, (n_dates, n_rows, n_cols))
        # Inject a large step at epoch 20 for a block of pixels
        displacement[20:, 3:7, 3:7] += 2.0

        result = detect_step_changes(
            dates, displacement, sigma_threshold=3.0, min_displacement_m=0.5,
        )

        assert result.step_change.shape == (n_dates, n_rows, n_cols)
        # The step should be flagged at epoch 20 for pixels (3:7, 3:7)
        assert np.any(result.step_change[20, 3:7, 3:7]), (
            "Step change at epoch 20 was not flagged"
        )
        # No step change at epoch 0 (no prior)
        assert not np.any(result.step_change[0])


# ---------------------------------------------------------------------------
# 4. Multi-detector fusion
# ---------------------------------------------------------------------------


class TestMultiDetectorFusion:
    """Run with all detectors enabled (acceleration + step-change + BOCPD)
    and verify flags from multiple sources are merged correctly."""

    def test_multi_detector_fusion(self):
        ts = generate_synthetic_scene()

        cfg = _default_config()
        cfg["detect"]["changepoint"] = {
            "method": "bocpd",
            "hazard_rate": 1 / 50,
            "threshold": 0.25,
            "min_displacement_range_m": 0.01,
        }
        cfg["detect"]["step_change"] = {
            "sigma_threshold": 3.0,
            "min_displacement_m": 0.005,
        }
        cfg["detect"]["clustering"]["min_cluster_pixels"] = 3
        cfg["detect"]["clustering"]["min_area_m2"] = 5000
        # Disable Voight: its window_index lookup is only valid for
        # acceleration-based flags, not step-change/BOCPD flags.
        cfg["detect"]["voight"]["enabled"] = False

        det_cfg = cfg["detect"]
        accel_map = compute_acceleration_map(
            ts.dates,
            ts.displacement,
            window_size_days=det_cfg["acceleration"]["window_size_days"],
            step_days=det_cfg["acceleration"]["step_days"],
            n_harmonics=det_cfg.get("n_harmonics", 2),
        )

        flags = detect_anomalies(
            accel_map, ts.latitude, ts.longitude, cfg,
            dates=ts.dates, displacement=ts.displacement,
        )

        # Should have flags from at least one detector
        assert len(flags) > 0

        # Collect tags from all flags
        tags = {f.detection_details.get("tag") for f in flags}

        # At minimum the acceleration detector should produce flags
        assert "acceleration" in tags, (
            f"Expected 'acceleration' tag in flags; got tags: {tags}"
        )

        # Flags should be sorted by score (highest first) even after merging
        scores = [f.score for f in flags]
        assert scores == sorted(scores, reverse=True), (
            "Flags not sorted by score after multi-detector fusion"
        )

        # All flag_ids should be unique (deduplication should not break ids)
        flag_ids = [f.flag_id for f in flags]
        assert len(flag_ids) == len(set(flag_ids)), (
            "Duplicate flag_ids found after deduplication"
        )


# ---------------------------------------------------------------------------
# 5. Report generation
# ---------------------------------------------------------------------------


class TestReportGeneration:
    """Run report generation and verify output files exist."""

    def test_report_generation(self, default_pipeline_result, tmp_path):
        r = default_pipeline_result
        ts = r["ts"]
        cfg = r["cfg"]
        accel_map = r["accel_map"]
        flags = r["flags"]

        assert len(flags) > 0, "Need at least one flag to test report generation"

        output_dir = tmp_path / "report_output"
        report_path = generate_report(
            flags=flags,
            assessments=None,
            accel_map=accel_map,
            latitude=ts.latitude,
            longitude=ts.longitude,
            config=cfg,
            output_dir=str(output_dir),
        )

        # Verify output files
        assert report_path.exists(), f"Report file not found: {report_path}"
        assert report_path.name == "report.md"

        geojson_path = output_dir / "flags.geojson"
        assert geojson_path.exists(), "flags.geojson not generated"

        fig_dir = output_dir / "figures"
        assert fig_dir.exists(), "figures/ directory not created"
        assert (fig_dir / "anomaly_map.png").exists(), "anomaly_map.png not generated"
        assert (fig_dir / "acceleration_evolution.png").exists(), (
            "acceleration_evolution.png not generated"
        )

        # At least one per-flag time series plot
        flag_plots = list(fig_dir.glob("flag_*_timeseries.png"))
        assert len(flag_plots) > 0, "No per-flag time series plots generated"

        # Verify GeoJSON content
        geojson = json.loads(geojson_path.read_text())
        assert geojson["type"] == "FeatureCollection"
        assert len(geojson["features"]) == len(flags)
        for feature in geojson["features"]:
            assert feature["type"] == "Feature"
            assert feature["geometry"]["type"] == "Point"
            assert len(feature["geometry"]["coordinates"]) == 2
            props = feature["properties"]
            assert "flag_id" in props
            assert "score" in props

        # Verify report.md content
        report_text = report_path.read_text()
        assert "Integration Test Site" in report_text
        assert "Tier 0" in report_text


# ---------------------------------------------------------------------------
# 6. Dashboard flag loading round-trip
# ---------------------------------------------------------------------------


class TestDashboardFlagLoading:
    """Generate flags, write to GeoJSON, load via dashboard's load_flags().
    Verify round-trip preserves flag data."""

    def test_dashboard_flag_loading(self, default_pipeline_result, tmp_path):
        r = default_pipeline_result
        ts = r["ts"]
        cfg = r["cfg"]
        accel_map = r["accel_map"]
        flags = r["flags"]

        assert len(flags) > 0

        # Generate report to produce the GeoJSON
        generate_report(
            flags=flags,
            assessments=None,
            accel_map=accel_map,
            latitude=ts.latitude,
            longitude=ts.longitude,
            config=cfg,
            output_dir=str(tmp_path),
        )

        # Load flags back via dashboard
        loaded = load_flags(tmp_path)

        assert len(loaded) == len(flags), (
            f"Round-trip lost flags: wrote {len(flags)}, loaded {len(loaded)}"
        )

        # Verify key properties survived the round-trip
        loaded_by_id = {f["flag_id"]: f for f in loaded}
        for original in flags:
            assert original.flag_id in loaded_by_id, (
                f"Flag {original.flag_id} missing after round-trip"
            )
            loaded_flag = loaded_by_id[original.flag_id]

            assert abs(loaded_flag["score"] - original.score) < 0.01
            assert abs(loaded_flag["center_lat"] - original.center_lat) < 0.001
            assert abs(loaded_flag["center_lon"] - original.center_lon) < 0.001

            # Dashboard adds a severity field
            assert "severity" in loaded_flag
            assert loaded_flag["severity"] in ("CRITICAL", "WARNING", "INFO")


# ---------------------------------------------------------------------------
# 7. Alert dispatch on detection (no-op with no channels)
# ---------------------------------------------------------------------------


class TestAlertDispatchOnDetection:
    """Detect anomalies, create AlertDispatcher with no channels configured,
    verify dispatch is a no-op (doesn't crash) and returns an empty dict."""

    def test_alert_dispatch_on_detection(self, default_pipeline_result):
        flags = default_pipeline_result["flags"]
        assert len(flags) > 0

        # AlertDispatcher with no channels (no config section)
        dispatcher = AlertDispatcher.from_config({})
        assert len(dispatcher.channels) == 0

        top_flag = flags[0]
        results = dispatcher.dispatch(
            alert_level="WARNING",
            site_name="Test Site",
            message=f"Anomalous acceleration detected: Flag {top_flag.flag_id}",
            details={
                "max_zscore": top_flag.peak_zscore,
                "n_flags": len(flags),
                "area_m2": top_flag.area_m2,
            },
        )

        # No channels => empty results dict, no crash
        assert results == {}

    def test_alert_dispatch_with_empty_alerts_section(self):
        """Dispatcher built from a config with an empty alerts: {} section
        should also be a safe no-op."""
        dispatcher = AlertDispatcher.from_config({"alerts": {}})
        assert len(dispatcher.channels) == 0

        results = dispatcher.dispatch(
            alert_level="CRITICAL",
            site_name="Test Site",
            message="Test alert",
            details={"key": "value"},
        )
        assert results == {}
