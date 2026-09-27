"""Tests for the interactive GeoJSON map viewer."""

import json

import pytest


class TestImports:
    """Module and key symbols are importable."""

    def test_mapview_module_imports(self):
        from glews.mapview import (
            MAP_HTML,
            load_geojson,
            make_handler,
            serve,
        )

        assert callable(load_geojson)
        assert callable(make_handler)
        assert callable(serve)
        assert isinstance(MAP_HTML, str)


class TestLoadGeojson:
    """GeoJSON loading from the data directory."""

    def test_load_existing_geojson(self, tmp_path):
        """Loads and parses a valid flags.geojson file."""
        from glews.mapview import load_geojson

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
                    },
                },
            ],
        }
        (tmp_path / "flags.geojson").write_text(json.dumps(geojson))

        result = load_geojson(tmp_path)

        assert result["type"] == "FeatureCollection"
        assert len(result["features"]) == 2
        assert result["features"][0]["properties"]["flag_id"] == 1
        assert result["features"][1]["properties"]["score"] == 2.0

    def test_load_empty_directory(self, tmp_path):
        """Returns empty FeatureCollection when no file exists."""
        from glews.mapview import load_geojson

        result = load_geojson(tmp_path)

        assert result["type"] == "FeatureCollection"
        assert result["features"] == []

    def test_load_nonexistent_directory(self, tmp_path):
        """Returns empty FeatureCollection for a nonexistent path."""
        from glews.mapview import load_geojson

        result = load_geojson(tmp_path / "does_not_exist")

        assert result["type"] == "FeatureCollection"
        assert result["features"] == []


class TestHTMLContent:
    """The generated HTML page includes required components."""

    def test_html_includes_leaflet_js(self):
        from glews.mapview import MAP_HTML

        assert "leaflet/1.9.4/leaflet.min.js" in MAP_HTML
        assert "cdnjs.cloudflare.com" in MAP_HTML

    def test_html_includes_leaflet_css(self):
        """Leaflet CSS is inlined in the page."""
        from glews.mapview import MAP_HTML

        assert ".leaflet-container" in MAP_HTML
        assert ".leaflet-control" in MAP_HTML

    def test_html_includes_openstreetmap_tiles(self):
        from glews.mapview import MAP_HTML

        assert "tile.openstreetmap.org" in MAP_HTML

    def test_html_includes_severity_colors(self):
        from glews.mapview import MAP_HTML

        assert "CRITICAL" in MAP_HTML
        assert "WARNING" in MAP_HTML
        assert "INFO" in MAP_HTML

    def test_html_includes_legend(self):
        from glews.mapview import MAP_HTML

        assert "legend" in MAP_HTML.lower()

    def test_html_fetches_geojson_api(self):
        from glews.mapview import MAP_HTML

        assert "/api/flags.geojson" in MAP_HTML


class TestMakeHandler:
    """Handler factory produces a working HTTP handler."""

    def test_make_handler_returns_class(self, tmp_path):
        from http.server import BaseHTTPRequestHandler

        from glews.mapview import make_handler

        handler_cls = make_handler(tmp_path)

        assert issubclass(handler_cls, BaseHTTPRequestHandler)


class TestSeverityForFeature:
    """Severity derivation from feature properties."""

    def test_critical_from_score(self):
        from glews.mapview import _severity_for_feature

        feature = {"properties": {"score": 5.0}}
        assert _severity_for_feature(feature) == "CRITICAL"

    def test_warning_from_score(self):
        from glews.mapview import _severity_for_feature

        feature = {"properties": {"score": 3.0}}
        assert _severity_for_feature(feature) == "WARNING"

    def test_info_from_low_score(self):
        from glews.mapview import _severity_for_feature

        feature = {"properties": {"score": 1.0}}
        assert _severity_for_feature(feature) == "INFO"

    def test_risk_level_overrides_score(self):
        from glews.mapview import _severity_for_feature

        feature = {"properties": {"score": 1.0, "risk_level": "high"}}
        assert _severity_for_feature(feature) == "HIGH"
