"""Tests for configuration validation."""

import textwrap

import pytest
import yaml

from glews.validate import validate_config, validate_config_file


def _minimal_valid_config() -> dict:
    """Return a minimal config that passes all required-field checks."""
    return {
        "site": {
            "name": "Test Site",
            "latitude": 28.20,
            "longitude": 85.90,
        },
    }


def _full_valid_config() -> dict:
    """Return a realistic config covering most sections."""
    return {
        "site": {
            "name": "Nepal-Tibet Border 2026",
            "latitude": 28.20,
            "longitude": 85.90,
            "buffer_km": 15,
        },
        "acquire": {
            "platform": "SENTINEL-1",
            "start_date": "2025-01-01",
            "end_date": "2026-08-26",
            "n_workers": 4,
            "max_scenes": 100,
        },
        "detect": {
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
                "enabled": True,
                "min_points": 5,
                "r_squared_threshold": 0.7,
            },
            "step_change": {
                "sigma_threshold": 5.0,
                "min_displacement_m": 0.5,
            },
        },
        "cascade": {
            "min_volume_m3": 100000,
            "exposure": {
                "max_runout_km": 50,
            },
            "valley_width_threshold_m": 500,
        },
        "monitor": {
            "interval_hours": 6,
            "lookback_days": 30,
        },
    }


# -------------------------------------------------------------------
# Valid configs
# -------------------------------------------------------------------


class TestValidConfig:
    def test_minimal_valid(self):
        issues = validate_config(_minimal_valid_config())
        errors = [i for i in issues if i.startswith("ERROR:")]
        assert errors == []

    def test_full_valid(self):
        issues = validate_config(_full_valid_config())
        errors = [i for i in issues if i.startswith("ERROR:")]
        assert errors == []

    def test_alias_lat_lon(self):
        """site.lat / site.lon should be accepted as aliases."""
        cfg = {
            "site": {
                "name": "Test",
                "lat": 28.20,
                "lon": 85.90,
            },
        }
        issues = validate_config(cfg)
        errors = [i for i in issues if i.startswith("ERROR:")]
        assert errors == []


# -------------------------------------------------------------------
# Missing required fields
# -------------------------------------------------------------------


class TestMissingRequired:
    def test_missing_site_name(self):
        cfg = _minimal_valid_config()
        del cfg["site"]["name"]
        issues = validate_config(cfg)
        assert any("site.name" in i for i in issues if i.startswith("ERROR:"))

    def test_missing_latitude(self):
        cfg = _minimal_valid_config()
        del cfg["site"]["latitude"]
        issues = validate_config(cfg)
        assert any("site.latitude" in i for i in issues if i.startswith("ERROR:"))

    def test_missing_longitude(self):
        cfg = _minimal_valid_config()
        del cfg["site"]["longitude"]
        issues = validate_config(cfg)
        assert any("site.longitude" in i for i in issues if i.startswith("ERROR:"))

    def test_missing_site_section(self):
        issues = validate_config({"acquire": {}})
        errors = [i for i in issues if i.startswith("ERROR:")]
        # All three site fields should be missing
        assert len(errors) >= 3


# -------------------------------------------------------------------
# Wrong types
# -------------------------------------------------------------------


class TestWrongTypes:
    def test_name_not_string(self):
        cfg = _minimal_valid_config()
        cfg["site"]["name"] = 12345
        issues = validate_config(cfg)
        assert any(
            "site.name" in i and "str" in i
            for i in issues
            if i.startswith("ERROR:")
        )

    def test_latitude_not_numeric(self):
        cfg = _minimal_valid_config()
        cfg["site"]["latitude"] = "not_a_number"
        issues = validate_config(cfg)
        assert any(
            "site.latitude" in i for i in issues if i.startswith("ERROR:")
        )

    def test_positive_field_not_numeric(self):
        cfg = _full_valid_config()
        cfg["detect"]["acceleration"]["sigma_threshold"] = "high"
        issues = validate_config(cfg)
        assert any(
            "sigma_threshold" in i and "numeric" in i
            for i in issues
            if i.startswith("ERROR:")
        )

    def test_non_dict_config(self):
        issues = validate_config("just a string")
        assert any("mapping" in i for i in issues if i.startswith("ERROR:"))


# -------------------------------------------------------------------
# Out-of-range values
# -------------------------------------------------------------------


class TestOutOfRange:
    def test_latitude_out_of_range(self):
        cfg = _minimal_valid_config()
        cfg["site"]["latitude"] = 95.0
        issues = validate_config(cfg)
        assert any("site.latitude" in i and "range" in i for i in issues)

    def test_longitude_out_of_range(self):
        cfg = _minimal_valid_config()
        cfg["site"]["longitude"] = -200.0
        issues = validate_config(cfg)
        assert any("site.longitude" in i and "range" in i for i in issues)

    def test_sigma_threshold_zero(self):
        cfg = _full_valid_config()
        cfg["detect"]["acceleration"]["sigma_threshold"] = 0
        issues = validate_config(cfg)
        assert any("sigma_threshold" in i and "> 0" in i for i in issues)

    def test_sigma_threshold_negative(self):
        cfg = _full_valid_config()
        cfg["detect"]["acceleration"]["sigma_threshold"] = -1
        issues = validate_config(cfg)
        assert any("sigma_threshold" in i and "> 0" in i for i in issues)

    def test_buffer_negative(self):
        cfg = _minimal_valid_config()
        cfg["site"]["buffer_km"] = -5
        issues = validate_config(cfg)
        assert any("buffer_km" in i and "> 0" in i for i in issues)

    def test_r_squared_out_of_unit_range(self):
        cfg = _full_valid_config()
        cfg["detect"]["voight"]["r_squared_threshold"] = 1.5
        issues = validate_config(cfg)
        assert any("r_squared_threshold" in i and "[0, 1]" in i for i in issues)


# -------------------------------------------------------------------
# Unknown keys
# -------------------------------------------------------------------


class TestUnknownKeys:
    def test_unknown_top_level_key(self):
        cfg = _minimal_valid_config()
        cfg["foobar"] = "something"
        issues = validate_config(cfg)
        warnings = [i for i in issues if i.startswith("WARNING:")]
        assert any("foobar" in w for w in warnings)

    def test_known_keys_no_warning(self):
        cfg = _full_valid_config()
        issues = validate_config(cfg)
        warnings = [i for i in issues if i.startswith("WARNING:")]
        assert warnings == []


# -------------------------------------------------------------------
# Cross-field checks
# -------------------------------------------------------------------


class TestCrossField:
    def test_start_date_after_end_date(self):
        cfg = _full_valid_config()
        cfg["acquire"]["start_date"] = "2027-01-01"
        cfg["acquire"]["end_date"] = "2026-08-26"
        issues = validate_config(cfg)
        assert any("start_date" in i and "before" in i for i in issues)

    def test_start_date_equals_end_date(self):
        cfg = _full_valid_config()
        cfg["acquire"]["start_date"] = "2026-08-26"
        cfg["acquire"]["end_date"] = "2026-08-26"
        issues = validate_config(cfg)
        assert any("start_date" in i and "before" in i for i in issues)

    def test_valid_date_order(self):
        cfg = _full_valid_config()
        issues = validate_config(cfg)
        assert not any("start_date" in i for i in issues if i.startswith("ERROR:"))


# -------------------------------------------------------------------
# Multi-site config
# -------------------------------------------------------------------


class TestMultiSite:
    def test_valid_multi_site(self):
        cfg = {
            "detect": {"acceleration": {"sigma_threshold": 2.5}},
            "sites": [
                {
                    "site": {
                        "name": "Site A",
                        "latitude": 28.20,
                        "longitude": 85.90,
                    },
                },
                {
                    "site": {
                        "name": "Site B",
                        "latitude": 30.00,
                        "longitude": 79.00,
                    },
                },
            ],
        }
        issues = validate_config(cfg)
        errors = [i for i in issues if i.startswith("ERROR:")]
        assert errors == []

    def test_multi_site_missing_site_section(self):
        cfg = {
            "sites": [
                {"acquire": {"platform": "NISAR"}},
            ],
        }
        issues = validate_config(cfg)
        assert any("missing required 'site' section" in i for i in issues)

    def test_multi_site_per_site_error_annotated(self):
        cfg = {
            "sites": [
                {
                    "site": {
                        "name": "Bad Site",
                        "latitude": 999,
                        "longitude": 85.0,
                    },
                },
            ],
        }
        issues = validate_config(cfg)
        errors = [i for i in issues if i.startswith("ERROR:")]
        assert any("Bad Site" in i for i in errors)

    def test_empty_sites_list(self):
        cfg = {"sites": []}
        issues = validate_config(cfg)
        assert any("empty" in i for i in issues if i.startswith("ERROR:"))


# -------------------------------------------------------------------
# File-level validation
# -------------------------------------------------------------------


class TestValidateConfigFile:
    def test_missing_file(self, tmp_path):
        _, issues = validate_config_file(str(tmp_path / "nonexistent.yaml"))
        assert any("not found" in i for i in issues)

    def test_invalid_yaml(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(":\n  :\n    - [unbalanced")
        _, issues = validate_config_file(str(bad))
        assert any("YAML" in i for i in issues if i.startswith("ERROR:"))

    def test_empty_file(self, tmp_path):
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        _, issues = validate_config_file(str(empty))
        assert any("empty" in i for i in issues if i.startswith("ERROR:"))

    def test_valid_file(self, tmp_path):
        cfg_file = tmp_path / "good.yaml"
        cfg_file.write_text(yaml.dump(_minimal_valid_config()))
        config, issues = validate_config_file(str(cfg_file))
        errors = [i for i in issues if i.startswith("ERROR:")]
        assert errors == []
        assert config["site"]["name"] == "Test Site"

    def test_real_nepal_config(self):
        """Validate the shipped nepal_2026.yaml config."""
        from pathlib import Path

        config_path = (
            Path(__file__).parent.parent / "config" / "nepal_2026.yaml"
        )
        if not config_path.exists():
            pytest.skip("nepal_2026.yaml not found")
        _, issues = validate_config_file(str(config_path))
        errors = [i for i in issues if i.startswith("ERROR:")]
        assert errors == [], f"Errors in shipped config: {errors}"
