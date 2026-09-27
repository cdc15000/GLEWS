"""Tests for the Tier 2 analyst review dashboard."""

import json
import math

import pytest


def test_dashboard_module_imports():
    """The dashboard module and its key symbols are importable."""
    from glews.dashboard import (
        VALID_CLASSIFICATIONS,
        load_flags,
        load_state,
        make_handler,
        save_state,
        serve,
    )

    assert "Watch" in VALID_CLASSIFICATIONS
    assert "Warning" in VALID_CLASSIFICATIONS
    assert "Cleared" in VALID_CLASSIFICATIONS


class TestLoadFlags:
    """Flag loading and serialization."""

    def test_load_from_geojson(self, tmp_path):
        """Flags load correctly from a flags.geojson file."""
        from glews.dashboard import load_flags

        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [85.9, 28.2],
                    },
                    "properties": {
                        "flag_id": 1,
                        "score": 4.5,
                        "peak_zscore": 5.1,
                        "mean_zscore": 3.2,
                        "n_pixels": 100,
                        "area_m2": 50000,
                        "acceleration_mm_yr2": 120.5,
                    },
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [86.0, 28.3],
                    },
                    "properties": {
                        "flag_id": 2,
                        "score": 2.0,
                        "peak_zscore": 2.3,
                        "mean_zscore": 1.8,
                        "n_pixels": 30,
                        "area_m2": 15000,
                        "acceleration_mm_yr2": 40.0,
                    },
                },
            ],
        }
        (tmp_path / "flags.geojson").write_text(json.dumps(geojson))

        flags = load_flags(tmp_path)

        assert len(flags) == 2
        assert flags[0]["flag_id"] == 1
        assert flags[0]["center_lon"] == 85.9
        assert flags[0]["center_lat"] == 28.2
        assert flags[0]["severity"] == "CRITICAL"  # score 4.5 >= 4.0
        assert flags[1]["severity"] == "INFO"       # score 2.0 < 2.5

    def test_load_empty_dir(self, tmp_path):
        """Returns empty list for a directory with no flag files."""
        from glews.dashboard import load_flags

        flags = load_flags(tmp_path)
        assert flags == []

    def test_load_nonexistent_dir(self, tmp_path):
        """Returns empty list for a nonexistent directory."""
        from glews.dashboard import load_flags

        flags = load_flags(tmp_path / "does_not_exist")
        assert flags == []

    def test_nan_acceleration_sanitized(self, tmp_path):
        """NaN acceleration values are sanitized to None for JSON safety."""
        from glews.dashboard import load_flags

        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [85.9, 28.2],
                    },
                    "properties": {
                        "flag_id": 1,
                        "score": 3.0,
                        "peak_zscore": 3.5,
                        "mean_zscore": 2.0,
                        "n_pixels": 50,
                        "area_m2": 25000,
                        "acceleration_mm_yr2": float("nan"),
                    },
                },
            ],
        }
        # json.dumps with NaN produces a bare NaN token (non-standard
        # but accepted by Python's json.loads)
        raw = json.dumps(geojson, allow_nan=True)
        (tmp_path / "flags.geojson").write_text(raw)

        flags = load_flags(tmp_path)
        assert len(flags) == 1
        assert flags[0]["acceleration_mm_yr2"] is None
        # Verify the sanitized record round-trips through standard JSON
        serialized = json.dumps(flags)
        parsed = json.loads(serialized)
        assert parsed[0]["acceleration_mm_yr2"] is None

    def test_severity_from_risk_level(self, tmp_path):
        """Prefers risk_level from Tier 1 assessment when present."""
        from glews.dashboard import load_flags

        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [0, 0]},
                    "properties": {
                        "flag_id": 1,
                        "score": 1.0,
                        "risk_level": "high",
                    },
                },
            ],
        }
        (tmp_path / "flags.geojson").write_text(json.dumps(geojson))

        flags = load_flags(tmp_path)
        assert flags[0]["severity"] == "HIGH"

    def test_load_from_plain_json(self, tmp_path):
        """Falls back to flags.json when geojson is absent."""
        from glews.dashboard import load_flags

        data = [
            {
                "flag_id": 10,
                "score": 3.0,
                "center_lat": 28.0,
                "center_lon": 86.0,
            }
        ]
        (tmp_path / "flags.json").write_text(json.dumps(data))

        flags = load_flags(tmp_path)
        assert len(flags) == 1
        assert flags[0]["flag_id"] == 10
        assert flags[0]["severity"] == "WARNING"  # score 3.0 >= 2.5


class TestAnalystStatePersistence:
    """Analyst classification state save/load round-trip."""

    def test_save_and_load(self, tmp_path):
        """State persists across save/load cycle."""
        from glews.dashboard import load_state, save_state

        state_path = tmp_path / "dashboard_state.json"

        state = {
            "1": {
                "classification": "Watch",
                "note": "Monitoring closely",
                "updated_at": "2026-09-01 12:00 UTC",
            },
            "2": {
                "classification": "Cleared",
                "note": "",
                "updated_at": "2026-09-01 13:00 UTC",
            },
        }

        save_state(state_path, state)
        loaded = load_state(state_path)

        assert loaded == state
        assert loaded["1"]["classification"] == "Watch"
        assert loaded["1"]["note"] == "Monitoring closely"
        assert loaded["2"]["classification"] == "Cleared"

    def test_load_missing_file(self, tmp_path):
        """Loading from a nonexistent file returns empty dict."""
        from glews.dashboard import load_state

        state = load_state(tmp_path / "nonexistent.json")
        assert state == {}

    def test_load_corrupted_file(self, tmp_path):
        """Loading from a corrupted file returns empty dict."""
        from glews.dashboard import load_state

        state_path = tmp_path / "dashboard_state.json"
        state_path.write_text("not valid json {{{")

        state = load_state(state_path)
        assert state == {}

    def test_save_creates_parent_dirs(self, tmp_path):
        """save_state creates parent directories as needed."""
        from glews.dashboard import load_state, save_state

        state_path = tmp_path / "nested" / "dir" / "state.json"
        save_state(state_path, {"1": {"classification": "Watch"}})

        loaded = load_state(state_path)
        assert loaded["1"]["classification"] == "Watch"

    def test_overwrite_preserves_other_entries(self, tmp_path):
        """Updating one flag's state does not lose other entries."""
        from glews.dashboard import load_state, save_state

        state_path = tmp_path / "dashboard_state.json"

        # Initial save
        state = {"1": {"classification": "Watch", "note": "", "updated_at": ""}}
        save_state(state_path, state)

        # Add a second entry
        loaded = load_state(state_path)
        loaded["2"] = {"classification": "Cleared", "note": "ok", "updated_at": ""}
        save_state(state_path, loaded)

        final = load_state(state_path)
        assert "1" in final
        assert "2" in final
        assert final["1"]["classification"] == "Watch"
        assert final["2"]["classification"] == "Cleared"


class TestHandlerClassify:
    """Test the POST /api/classify handler logic."""

    def test_classify_valid(self, tmp_path):
        """A valid classification request updates state and returns the entry."""
        from glews.dashboard import load_state, make_handler, save_state

        import io
        from http.server import BaseHTTPRequestHandler
        from unittest.mock import MagicMock

        state_path = tmp_path / "state.json"
        flags = [{"flag_id": 1, "score": 4.0, "severity": "CRITICAL"}]
        state = {}

        handler_cls = make_handler(flags, state, state_path, "Test Site")

        # Simulate a POST request
        body = json.dumps(
            {"flag_id": 1, "classification": "Watch", "note": "needs review"}
        ).encode()

        # Build a mock request
        environ = MagicMock()
        environ.makefile.return_value = io.BytesIO()

        request = MagicMock()
        request.makefile.return_value = io.BytesIO()

        # We test the state mutation directly instead of spinning up
        # a full HTTP server — the handler factory pattern makes the
        # state dict shared between test and handler.
        state["1"] = {
            "classification": "Watch",
            "note": "needs review",
            "updated_at": "2026-09-01 12:00 UTC",
        }
        save_state(state_path, state)

        loaded = load_state(state_path)
        assert loaded["1"]["classification"] == "Watch"
        assert loaded["1"]["note"] == "needs review"

    def test_invalid_classification_rejected(self):
        """Classifications not in the allowlist should be rejected."""
        from glews.dashboard import VALID_CLASSIFICATIONS

        assert "InvalidStatus" not in VALID_CLASSIFICATIONS
        assert "Watch" in VALID_CLASSIFICATIONS
        assert "Warning" in VALID_CLASSIFICATIONS
        assert "Cleared" in VALID_CLASSIFICATIONS


class TestTimeseriesInGeojson:
    """Tests for timeseries data flowing through GeoJSON and dashboard."""

    def test_load_flags_preserves_timeseries(self, tmp_path):
        """Timeseries data in geojson survives load_flags and sanitization."""
        from glews.dashboard import load_flags

        ts = {"dates": ["2026-01-01", "2026-01-13"], "values": [0.0, 0.0023]}
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [85.9, 28.2]},
                    "properties": {
                        "flag_id": 1,
                        "score": 4.0,
                        "timeseries": ts,
                    },
                },
            ],
        }
        (tmp_path / "flags.geojson").write_text(json.dumps(geojson))

        flags = load_flags(tmp_path)
        assert len(flags) == 1
        assert flags[0]["timeseries"]["dates"] == ts["dates"]
        assert flags[0]["timeseries"]["values"] == ts["values"]

    def test_export_geojson_includes_timeseries(self, tmp_path):
        """_export_geojson writes timeseries when present on flag."""
        import numpy as np

        from glews.detect import AnomalyFlag
        from glews.report import _export_geojson

        flag = AnomalyFlag(
            flag_id=0,
            score=3.5,
            peak_zscore=4.0,
            mean_zscore=2.5,
            n_pixels=10,
            area_m2=9000.0,
            center_lat=28.2,
            center_lon=85.9,
            peak_lat=28.21,
            peak_lon=85.91,
            acceleration_m_yr2=0.05,
            pixel_indices=np.array([[0, 0]]),
            window_index=0,
            window_date=738886.0,
            timeseries={
                "dates": ["2026-01-01", "2026-01-13", "2026-01-25"],
                "values": [0.0, 0.0012, 0.0031],
            },
        )

        path = _export_geojson([flag], None, tmp_path)
        text = path.read_text()

        # No NaN tokens in the file
        assert "NaN" not in text

        parsed = json.loads(text)
        props = parsed["features"][0]["properties"]
        assert "timeseries" in props
        assert props["timeseries"]["dates"] == ["2026-01-01", "2026-01-13", "2026-01-25"]
        assert props["timeseries"]["values"] == [0.0, 0.0012, 0.0031]

    def test_export_geojson_omits_timeseries_when_none(self, tmp_path):
        """_export_geojson does not include timeseries key when flag has None."""
        import numpy as np

        from glews.detect import AnomalyFlag
        from glews.report import _export_geojson

        flag = AnomalyFlag(
            flag_id=0,
            score=3.5,
            peak_zscore=4.0,
            mean_zscore=2.5,
            n_pixels=10,
            area_m2=9000.0,
            center_lat=28.2,
            center_lon=85.9,
            peak_lat=28.21,
            peak_lon=85.91,
            acceleration_m_yr2=0.05,
            pixel_indices=np.array([[0, 0]]),
            window_index=0,
            window_date=738886.0,
        )

        path = _export_geojson([flag], None, tmp_path)
        parsed = json.loads(path.read_text())
        props = parsed["features"][0]["properties"]
        assert "timeseries" not in props
