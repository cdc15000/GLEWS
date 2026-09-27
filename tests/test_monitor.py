"""Tests for the operational monitoring module.

Covers the pure/config-driven pieces of glews.monitor — config fan-out,
state (de)serialization, and alert construction — without touching
ASF search, downloads, or the NISAR/detection pipeline.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from glews.detect import AnomalyFlag
from glews.monitor import (
    DEFAULT_ALERT_LEVELS,
    MonitorState,
    build_alerts,
    iter_site_configs,
)


# --------------------------------------------------------------------------
# iter_site_configs
# --------------------------------------------------------------------------


class TestIterSiteConfigs:
    def test_single_site_config_yielded_once(self, tmp_path, site_config):
        path = tmp_path / "single.yaml"
        path.write_text(yaml.safe_dump(site_config))

        configs = list(iter_site_configs(path))

        assert len(configs) == 1
        assert configs[0]["site"]["name"] == "Test Site"
        assert configs[0]["detect"]["acceleration"]["sigma_threshold"] == 2.5

    def test_multi_site_config_merges_shared_defaults(self, tmp_path):
        raw = {
            "acquire": {"platform": "NISAR", "n_workers": 4},
            "detect": {
                "n_harmonics": 2,
                "acceleration": {
                    "window_size_days": 60,
                    "step_days": 12,
                    "sigma_threshold": 2.5,
                },
            },
            "monitor": {"interval_hours": 6},
            "sites": [
                {"site": {"name": "Site A", "latitude": 28.0, "longitude": 85.0}},
                {
                    "site": {"name": "Site B", "latitude": 30.0, "longitude": 90.0},
                    "detect": {"acceleration": {"sigma_threshold": 3.0}},
                },
            ],
        }
        path = tmp_path / "multi.yaml"
        path.write_text(yaml.safe_dump(raw))

        configs = list(iter_site_configs(path))

        assert len(configs) == 2
        assert configs[0]["site"]["name"] == "Site A"
        assert configs[1]["site"]["name"] == "Site B"

        # Shared defaults present on both
        assert configs[0]["acquire"]["platform"] == "NISAR"
        assert configs[1]["acquire"]["platform"] == "NISAR"

        # Per-site override applied for Site B only (shallow merge within
        # the 'detect' block, not a full replace)
        assert configs[0]["detect"]["acceleration"]["sigma_threshold"] == 2.5
        assert configs[1]["detect"]["acceleration"]["sigma_threshold"] == 3.0
        assert configs[1]["detect"]["n_harmonics"] == 2


# --------------------------------------------------------------------------
# MonitorState
# --------------------------------------------------------------------------


class TestMonitorState:
    def test_round_trip_to_from_dict(self):
        state = MonitorState(
            site_name="Test Site",
            last_check="2026-01-01T00:00:00+00:00",
            known_products={"granule-1", "granule-2"},
            downloaded_products={"granule-1"},
            alerted_signatures={"abc123"},
            n_checks=3,
        )

        restored = MonitorState.from_dict(state.to_dict())

        assert restored.site_name == state.site_name
        assert restored.last_check == state.last_check
        assert restored.known_products == state.known_products
        assert restored.downloaded_products == state.downloaded_products
        assert restored.alerted_signatures == state.alerted_signatures
        assert restored.n_checks == state.n_checks

    def test_to_dict_produces_sorted_lists_not_sets(self):
        state = MonitorState(site_name="X", known_products={"b", "a", "c"})
        d = state.to_dict()

        assert d["known_products"] == ["a", "b", "c"]
        assert isinstance(d["known_products"], list)

    def test_save_and_load_round_trip(self, tmp_path):
        state = MonitorState(
            site_name="Test Site",
            known_products={"g1"},
            n_checks=5,
        )
        path = tmp_path / "state" / "test_site.json"
        state.save(path)

        assert path.exists()
        loaded = MonitorState.load(path, "Test Site")

        assert loaded.site_name == "Test Site"
        assert loaded.known_products == {"g1"}
        assert loaded.n_checks == 5

    def test_load_missing_file_returns_fresh_state(self, tmp_path):
        path = tmp_path / "does_not_exist.json"
        state = MonitorState.load(path, "New Site")

        assert state.site_name == "New Site"
        assert state.known_products == set()
        assert state.n_checks == 0

    def test_load_corrupt_file_returns_fresh_state(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("{not valid json")

        state = MonitorState.load(path, "New Site")

        assert state.site_name == "New Site"
        assert state.known_products == set()


# --------------------------------------------------------------------------
# build_alerts
# --------------------------------------------------------------------------


def _make_flag(flag_id=1, peak_zscore=6.0, center_lat=28.2, center_lon=85.9,
                voight_fit=None):
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
        voight_fit=voight_fit,
    )


class TestBuildAlerts:
    def test_high_zscore_produces_alert(self, site_config):
        state = MonitorState(site_name="Test Site")
        flags = [_make_flag(peak_zscore=6.0)]  # ratio 6.0/2.5 = 2.4 -> WARNING

        alerts = build_alerts(flags, ts=None, site_config=site_config, state=state)

        assert len(alerts) == 1
        alert = alerts[0]
        assert alert["site"] == "Test Site"
        assert alert["level"] in DEFAULT_ALERT_LEVELS
        assert alert["max_zscore"] == 6.0
        assert "alert_id" in alert

    def test_low_zscore_below_all_levels_produces_no_alert(self, site_config):
        state = MonitorState(site_name="Test Site")
        # ratio 0.5/2.5 = 0.2, below the lowest INFO threshold (1.0)
        flags = [_make_flag(peak_zscore=0.5)]

        alerts = build_alerts(flags, ts=None, site_config=site_config, state=state)

        assert alerts == []

    def test_duplicate_signature_not_re_alerted(self, site_config):
        state = MonitorState(site_name="Test Site")
        flags = [_make_flag(peak_zscore=6.0, center_lat=28.2, center_lon=85.9)]

        first = build_alerts(flags, ts=None, site_config=site_config, state=state)
        assert len(first) == 1

        # Same flag (same location) run again against the same state
        second = build_alerts(flags, ts=None, site_config=site_config, state=state)
        assert second == []

    def test_state_alerted_signatures_updated(self, site_config):
        state = MonitorState(site_name="Test Site")
        flags = [_make_flag(peak_zscore=6.0)]

        assert state.alerted_signatures == set()
        build_alerts(flags, ts=None, site_config=site_config, state=state)
        assert len(state.alerted_signatures) == 1

    def test_critical_escalation_from_voight_fit(self, site_config):
        state = MonitorState(site_name="Test Site")
        # Low z-score ratio, but a strong near-term Voight prediction
        # should escalate the alert to CRITICAL regardless.
        flags = [
            _make_flag(
                peak_zscore=3.0,
                voight_fit={"r_squared": 0.9, "days_until_failure": 10},
            )
        ]

        alerts = build_alerts(flags, ts=None, site_config=site_config, state=state)

        assert len(alerts) == 1
        assert alerts[0]["level"] == "CRITICAL"
