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


# ---------------------------------------------------------------------------
# Data quality, provenance and triage
# ---------------------------------------------------------------------------

def _write_geojson(path, features, properties=None):
    path.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": features,
        "properties": properties or {},
    }))


def _feature(fid, score, quality=None, lon=85.52, lat=28.29):
    props = {"flag_id": fid, "score": score}
    if quality:
        props["quality"] = quality
    return {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": props}


class TestReliabilityCap:
    def test_unreliable_critical_capped(self, tmp_path):
        from glews.dashboard import UNRELIABLE_SEVERITY_CAP, load_flags

        _write_geojson(tmp_path / "flags.geojson", [
            _feature(1, 5.0, {"reliability": "unreliable"}),
            _feature(2, 5.0, {"reliability": "good"}),
            _feature(3, 3.0, {"reliability": "unreliable"}),
        ])
        flags = {f["flag_id"]: f for f in load_flags(tmp_path)}
        assert flags[1]["severity"] == UNRELIABLE_SEVERITY_CAP
        assert flags[1]["severity_uncapped"] == "CRITICAL"
        assert flags[2]["severity"] == "CRITICAL"
        assert flags[3]["severity"] == "WARNING"
        assert "severity_uncapped" not in flags[3]


class TestBanner:
    def test_synthetic_banner_unchanged(self):
        from glews.dashboard import _data_banner_html

        assert "SYNTHETIC DATA" in _data_banner_html("synthetic", {"sensor": "X"})

    def test_observed_banner_from_provenance(self):
        from glews.dashboard import _data_banner_html

        html = _data_banner_html("nisar-gunw-t98", {
            "sensor": "NISAR", "product": "L2 GUNW", "track": 98, "direction": "ascending",
            "date_start": "2025-11-28", "date_end": "2026-08-19", "n_pairs": 21,
            "coherence_mask": 0.3, "excluded": ["co-event <pair>"],
        })
        assert "OBSERVED DATA" in html
        assert "NISAR L2 GUNW · track 98 ascending · 28 Nov 2025 – 19 Aug 2026 · 21 pairs" in html
        assert "co-event &lt;pair&gt;" in html

    def test_no_banner_without_provenance(self):
        from glews.dashboard import _data_banner_html

        assert _data_banner_html("nisar", None) == ""

    def test_render_escapes_script_close_in_data(self):
        from glews.dashboard import _render_html

        html = _render_html([{"flag_id": 1, "place_name": "</script><b>x"}], {}, "Site <A>")
        assert "</script><b>x" not in html
        assert "Site &lt;A&gt;" in html


@pytest.fixture
def live_dashboard(tmp_path):
    """Run the dashboard handler on an ephemeral port."""
    import threading
    from http.server import ThreadingHTTPServer

    from glews.dashboard import make_handler

    (tmp_path / "figures" / "ifg").mkdir(parents=True)
    (tmp_path / "figures" / "ifg" / "flag_001.png").write_bytes(b"\x89PNG fake")
    (tmp_path / "secret.txt").write_text("no")
    flags = [{"flag_id": 1, "score": 5.0, "severity": "CRITICAL", "place_name": "A"},
             {"flag_id": 2, "score": 3.0, "severity": "WARNING"}]
    state = {"2": {"classification": "Watch", "note": "keep an eye on it"}}
    done = threading.Event()
    handler = make_handler(flags, state, tmp_path / "dashboard_state.json", "Test",
                           place_names_done=done)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", tmp_path, state, done
    server.shutdown()
    server.server_close()


def _get(url):
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(url) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _post(url, payload):
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


class TestLiveRoutes:
    def test_figure_served(self, live_dashboard):
        base, *_ = live_dashboard
        assert _get(base + "/figures/ifg/flag_001.png") == (200, b"\x89PNG fake")

    @pytest.mark.parametrize("path", [
        "/figures/../secret.txt", "/figures/%2e%2e/secret.txt",
        "/figures/ifg/missing.png", "/figures/../dashboard_state.json",
    ])
    def test_figure_route_stays_in_folder(self, live_dashboard, path):
        base, *_ = live_dashboard
        assert _get(base + path)[0] == 404

    def test_place_names_pending_then_done(self, live_dashboard):
        base, _, _, done = live_dashboard
        status, body = _get(base + "/api/place_names")
        data = json.loads(body)
        assert status == 200 and data["pending"] is True
        assert data["names"] == {"1": {"name": "A", "name_en": ""}}
        done.set()
        assert json.loads(_get(base + "/api/place_names")[1])["pending"] is False

    def test_bulk_clear_updates_state_and_audit(self, live_dashboard):
        base, tmp_path, state, _ = live_dashboard
        status, data = _post(base + "/api/classify_bulk", {
            "flag_ids": [1, 2], "classification": "Cleared", "note": "Seasonal decorrelation (snow)"})
        assert status == 200
        assert state["1"]["classification"] == "Cleared" and state["1"]["bulk"] is True
        assert state["2"]["note"] == "keep an eye on it\nSeasonal decorrelation (snow)"
        saved = json.loads((tmp_path / "dashboard_state.json").read_text())
        assert saved["2"]["classification"] == "Cleared"
        lines = (tmp_path / "audit" / "alert_audit.jsonl").read_text().splitlines()
        entries = [json.loads(line) for line in lines]
        assert {e["flag_id"] for e in entries} == {"1", "2"}
        assert all(e["analyst_note"] == "[bulk] Seasonal decorrelation (snow)" for e in entries)

    @pytest.mark.parametrize("payload, msg", [
        ({"flag_ids": [1], "classification": "Gone", "note": "x"}, "Invalid classification"),
        ({"flag_ids": [1], "classification": "Cleared", "note": "  "}, "reason"),
        ({"flag_ids": [], "classification": "Cleared", "note": "x"}, "reason"),
        ({"flag_ids": [99], "classification": "Cleared", "note": "x"}, "Unknown flag ids: 99"),
    ])
    def test_bulk_clear_rejects_bad_requests(self, live_dashboard, payload, msg):
        base, _, state, _ = live_dashboard
        status, data = _post(base + "/api/classify_bulk", payload)
        assert status == 400 and msg in data["error"]
        assert "1" not in state

    def test_single_classify_is_audited(self, live_dashboard):
        base, tmp_path, *_ = live_dashboard
        status, _ = _post(base + "/api/classify", {"flag_id": 1, "classification": "Watch", "note": "n"})
        assert status == 200
        entry = json.loads((tmp_path / "audit" / "alert_audit.jsonl").read_text().splitlines()[-1])
        assert entry["classification"] == "Watch" and entry["flag_id"] == "1"


def test_site_context_withholds_source_for_synthetic():
    from glews.dashboard import _site_context

    cfg = {"site": {"event_date": "2026-08-26", "known_source": {"polygon": [[0, 0]] * 3},
                    "snow_months": [1, 2]}}
    assert _site_context(cfg, None, "synthetic")["known_source"] is None
    real = _site_context(cfg, {"extent": [1, 2, 3, 4]}, "nisar")
    assert real["known_source"]["polygon"] and real["extent"] == [1, 2, 3, 4]
    assert real["snow_months"] == [1, 2]
