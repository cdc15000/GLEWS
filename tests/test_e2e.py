"""End-to-end integration tests for the GEWS pipeline.

Each test exercises an entire subsystem or multi-module flow using only
synthetic data --- no network access, no real satellite products.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml

from glews.alerts import AlertDispatcher, WebhookChannel
from glews.crosscheck import apply_tier1_filters
from glews.detect import AnomalyFlag, detect_anomalies
from glews.provenance import ProvenanceTracker
from glews.report import generate_report
from glews.synthetic import generate_synthetic_scene
from glews.timeseries import compute_acceleration_map
from glews.validate import validate_config, validate_config_file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detection_config() -> dict:
    """Minimal detection config matching the shape expected by the pipeline."""
    return {
        "site": {
            "name": "E2E Test Site",
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


def _make_flag(flag_id=1, peak_zscore=6.0, center_lat=28.2, center_lon=85.9):
    """Create a minimal AnomalyFlag for testing."""
    return AnomalyFlag(
        flag_id=flag_id,
        score=peak_zscore,
        peak_zscore=peak_zscore,
        mean_zscore=peak_zscore * 0.8,
        n_pixels=10,
        area_m2=50000.0,
        center_lat=center_lat,
        center_lon=center_lon,
        peak_lat=center_lat,
        peak_lon=center_lon,
        acceleration_m_yr2=0.05,
        pixel_indices=np.array([[0, 0], [0, 1]]),
        window_index=5,
        window_date=738900.0,
    )


# ---------------------------------------------------------------------------
# 1. test_full_pipeline_synthetic
# ---------------------------------------------------------------------------


class TestFullPipelineSynthetic:
    """Synthetic scene -> acceleration -> detect -> crosscheck -> report.
    Assert injected signal detected."""

    def test_full_pipeline_synthetic(self, tmp_path):
        # Step 1: Generate synthetic scene
        ts = generate_synthetic_scene()
        cfg = _detection_config()
        det_cfg = cfg["detect"]

        # Step 2: Compute acceleration map
        accel_map = compute_acceleration_map(
            ts.dates,
            ts.displacement,
            window_size_days=det_cfg["acceleration"]["window_size_days"],
            step_days=det_cfg["acceleration"]["step_days"],
            n_harmonics=det_cfg.get("n_harmonics", 2),
        )

        assert accel_map.acceleration.ndim == 3
        assert accel_map.acceleration_zscore.shape == accel_map.acceleration.shape

        # Step 3: Detect anomalies
        flags = detect_anomalies(accel_map, ts.latitude, ts.longitude, cfg)
        assert len(flags) > 0, "No flags detected; injected failure zone missed"

        # The injected failure zone is near lat~28.22, lon~85.90.
        near_injection = [
            f for f in flags
            if abs(f.center_lat - 28.22) < 0.1
            and abs(f.center_lon - 85.90) < 0.1
        ]
        assert len(near_injection) > 0, (
            "No flags detected near injected failure zone (28.22N, 85.90E)"
        )

        # Step 4: Crosscheck (Tier 1 filters)
        tier1_cfg = {
            "detect": {
                "tier1_filters": {
                    "min_coherence": 0.3,
                    "min_spatial_consistency": 0.3,
                    "min_temporal_consistency": 0.2,
                    "remove_below_quality": 0.1,
                    "demote_below_quality": 0.4,
                },
            },
        }
        dates_float = ts.dates.astype(float)
        filtered_flags = apply_tier1_filters(
            flags, ts.displacement, ts.temporal_coherence, dates_float, tier1_cfg,
        )

        # At least verify crosscheck ran and annotated the surviving flags
        for f in filtered_flags:
            assert "tier1_quality" in f.detection_details
            t1q = f.detection_details["tier1_quality"]
            assert "quality_score" in t1q
            assert "recommendation" in t1q

        # Step 5: Generate report
        report_flags = filtered_flags if filtered_flags else flags
        output_dir = tmp_path / "report"
        report_path = generate_report(
            flags=report_flags,
            assessments=None,
            accel_map=accel_map,
            latitude=ts.latitude,
            longitude=ts.longitude,
            config=cfg,
            output_dir=str(output_dir),
        )

        assert report_path.exists()
        assert (output_dir / "flags.geojson").exists()
        geojson = json.loads((output_dir / "flags.geojson").read_text())
        assert geojson["type"] == "FeatureCollection"
        assert len(geojson["features"]) == len(report_flags)


# ---------------------------------------------------------------------------
# 2. test_demo_command
# ---------------------------------------------------------------------------


class TestDemoCommand:
    """Run `glews demo --no-plots` via Click CliRunner, assert exit 0 and
    flags found."""

    def test_demo_command(self, tmp_path):
        from click.testing import CliRunner
        from glews.cli import main

        runner = CliRunner()
        result = runner.invoke(main, [
            "demo",
            "--output", str(tmp_path / "demo_output"),
            "--no-plots",
        ])

        assert result.exit_code == 0, (
            f"demo --no-plots failed (exit {result.exit_code}):\n{result.output}"
        )
        # The demo should report detected flags
        assert "Detected" in result.output
        # With --no-plots it should show the skip message
        assert "Skipping plots" in result.output
        # The Results Summary table should appear
        assert "Results Summary" in result.output


# ---------------------------------------------------------------------------
# 3. test_monitor_single_check
# ---------------------------------------------------------------------------


class TestMonitorSingleCheck:
    """Mock ASF search returning empty, run single check cycle, verify
    no crash."""

    def test_monitor_single_check(self, tmp_path):
        from glews.monitor import run_check_cycle

        site_config = {
            "site": {
                "name": "Monitor Test Site",
                "latitude": 28.2,
                "longitude": 85.9,
            },
            "acquire": {
                "platform": "NISAR",
                "product_types": ["GUNW"],
                "gunw_dir": str(tmp_path / "gunw"),
                "goff_dir": str(tmp_path / "goff"),
            },
            "detect": {
                "n_harmonics": 2,
                "acceleration": {
                    "window_size_days": 60,
                    "step_days": 12,
                    "sigma_threshold": 2.5,
                },
                "clustering": {
                    "min_cluster_pixels": 5,
                    "max_distance_m": 200,
                    "min_area_m2": 10000,
                },
                "voight": {
                    "enabled": False,
                },
            },
            "monitor": {
                "interval_hours": 6,
                "lookback_days": 30,
                "state_dir": str(tmp_path / "state"),
                "provenance_dir": str(tmp_path / "provenance"),
                "audit_dir": str(tmp_path / "audit"),
            },
        }

        # Patch asf_search.search to return empty list (no new data)
        with patch("asf_search.search", return_value=[]):
            alerts = run_check_cycle(site_config)

        # No new data => no flags => no alerts, and no crash
        assert alerts == []

        # State file should have been created
        state_dir = tmp_path / "state"
        state_files = list(state_dir.glob("*.json"))
        assert len(state_files) == 1


# ---------------------------------------------------------------------------
# 4. test_alert_dispatch_integration
# ---------------------------------------------------------------------------


class TestAlertDispatchIntegration:
    """Detection flag through alert system with mock webhook."""

    def test_alert_dispatch_integration(self):
        flag = _make_flag(peak_zscore=8.0)

        # Create a webhook channel with a mock
        with patch("glews.alerts.urllib.request.urlopen") as mock_urlopen:
            dispatcher = AlertDispatcher.from_config({
                "alerts": {
                    "webhook": {
                        "url": "https://example.com/api/alerts",
                        "headers": {"Authorization": "Bearer test-token"},
                    },
                },
            })

            assert len(dispatcher.channels) == 1
            assert isinstance(dispatcher.channels[0], WebhookChannel)

            results = dispatcher.dispatch(
                alert_level="CRITICAL",
                site_name="E2E Test Site",
                message=f"Anomalous acceleration detected: Flag {flag.flag_id}",
                details={
                    "max_zscore": flag.peak_zscore,
                    "n_flags": 1,
                    "area_m2": flag.area_m2,
                    "center_lat": flag.center_lat,
                    "center_lon": flag.center_lon,
                },
            )

            # Webhook should have been called
            assert results == {"webhook": True}
            mock_urlopen.assert_called_once()

            # Verify the payload sent to the webhook
            req = mock_urlopen.call_args[0][0]
            assert req.method == "POST"
            body = json.loads(req.data)
            assert body["alert_level"] == "CRITICAL"
            assert body["site_name"] == "E2E Test Site"
            assert body["details"]["max_zscore"] == 8.0
            assert "timestamp" in body


# ---------------------------------------------------------------------------
# 5. test_validate_all_configs
# ---------------------------------------------------------------------------


class TestValidateAllConfigs:
    """Load every config/*.yaml, run validate_config, assert no errors."""

    def test_validate_all_configs(self):
        config_dir = Path(__file__).parent.parent / "config"
        assert config_dir.is_dir(), f"Config directory not found: {config_dir}"

        yaml_files = sorted(config_dir.glob("*.yaml"))
        assert len(yaml_files) >= 1, "No YAML config files found"

        for config_file in yaml_files:
            config, issues = validate_config_file(str(config_file))
            errors = [i for i in issues if i.startswith("ERROR:")]
            assert errors == [], (
                f"Validation errors in {config_file.name}:\n"
                + "\n".join(errors)
            )


# ---------------------------------------------------------------------------
# 6. test_train_classifier_e2e
# ---------------------------------------------------------------------------


class TestTrainClassifierE2E:
    """Train with small samples, verify model file and accuracy > 0.7."""

    def test_train_classifier_e2e(self, tmp_path):
        from glews.classifier import PrecursorClassifier, train_precursor_model

        model_path = tmp_path / "model.json"
        metrics = train_precursor_model(
            model_path,
            seed=42,
            n_positive=80,
            n_negative=320,
            n_iter=300,
            lr=0.1,
        )

        # Model file should exist and be valid JSON
        assert model_path.is_file()
        data = json.loads(model_path.read_text())
        assert "weights" in data
        assert "bias" in data
        assert "feature_names" in data

        # Accuracy should be > 0.7
        assert metrics["accuracy"] > 0.7, (
            f"Accuracy {metrics['accuracy']:.3f} below 0.7 threshold"
        )

        # Model should be loadable and produce valid predictions
        model = PrecursorClassifier.load(model_path)
        from glews.classifier import LandslideTrainingData

        gen = LandslideTrainingData(seed=999)
        X, y = gen.generate_synthetic_training_set(n_positive=20, n_negative=80)
        proba = model.predict_proba(X)
        assert np.all((proba >= 0) & (proba <= 1))

        # Positive examples should on average score higher than negatives
        assert np.mean(proba[y == 1]) > np.mean(proba[y == 0])


# ---------------------------------------------------------------------------
# 7. test_provenance_audit_trail
# ---------------------------------------------------------------------------


class TestProvenanceAuditTrail:
    """Detection through provenance tracker, verify JSONL entries."""

    def test_provenance_audit_trail(self, tmp_path):
        tracker = ProvenanceTracker(log_dir=tmp_path / "provenance")

        # Simulate a detection pipeline with provenance tracking
        with tracker.track(
            "detect",
            "Audit Trail Test Site",
            inputs=["synthetic_scene.h5"],
            params={"sigma_threshold": 2.5, "window_size_days": 60},
        ) as ctx:
            # Run actual detection on synthetic data
            ts = generate_synthetic_scene()
            cfg = _detection_config()
            det_cfg = cfg["detect"]

            accel_map = compute_acceleration_map(
                ts.dates,
                ts.displacement,
                window_size_days=det_cfg["acceleration"]["window_size_days"],
                step_days=det_cfg["acceleration"]["step_days"],
                n_harmonics=det_cfg.get("n_harmonics", 2),
            )
            flags = detect_anomalies(accel_map, ts.latitude, ts.longitude, cfg)

            ctx.set_result(f"{len(flags)} flags detected")
            ctx.outputs = ["flags.geojson"]

        # Record an alert dispatch event
        tracker.record(
            "alert_dispatch",
            "Audit Trail Test Site",
            summary=f"Dispatched alerts for {len(flags)} flags",
        )

        # Verify JSONL file exists and contains correct entries
        assert tracker.log_path.exists()
        lines = tracker.log_path.read_text().strip().split("\n")
        assert len(lines) == 2

        # Parse and verify each entry
        detect_entry = json.loads(lines[0])
        assert detect_entry["action"] == "detect"
        assert detect_entry["site_name"] == "Audit Trail Test Site"
        assert "flags detected" in detect_entry["result_summary"]
        assert detect_entry["input_files"] == ["synthetic_scene.h5"]
        assert detect_entry["output_files"] == ["flags.geojson"]
        assert detect_entry["parameters"]["sigma_threshold"] == 2.5
        assert detect_entry["duration_seconds"] > 0
        assert "timestamp" in detect_entry

        alert_entry = json.loads(lines[1])
        assert alert_entry["action"] == "alert_dispatch"
        assert "Dispatched" in alert_entry["result_summary"]

        # Query API should also work
        all_records = tracker.query()
        assert len(all_records) == 2

        detect_records = tracker.query(action="detect")
        assert len(detect_records) == 1
        assert detect_records[0].site_name == "Audit Trail Test Site"

        site_records = tracker.query(site_name="Audit Trail Test Site")
        assert len(site_records) == 2
