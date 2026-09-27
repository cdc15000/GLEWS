"""Tests for the seismic/infrasound tactical warning module (glews.tactical).

Covers arrival-time computation with known distances, trigger evaluation
logic, alert message generation, escalation with a mock dispatcher, and
edge cases (zero distance, boundary thresholds, invalid confidence).
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from glews.tactical import (
    ArrivalTimeEstimate,
    DownstreamCommunity,
    SeismicTrigger,
    TacticalWarningSystem,
    _haversine_km,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def site_config():
    """Minimal site config for tactical tests."""
    return {
        "site": {
            "name": "Test Glacier",
            "latitude": 28.0,
            "longitude": 85.0,
        },
        "tactical": {
            "seismic_sources": [
                {"name": "USGS", "url": "https://earthquake.usgs.gov/fdsnws/event/1/query"},
            ],
            "trigger_thresholds": {
                "min_magnitude": 2.0,
                "max_distance_km": 10.0,
                "min_confidence": 0.7,
            },
            "wave_speed_m_s": 5.0,
            "wave_speed_uncertainty": 0.3,
            "downstream_communities": [
                {
                    "name": "Rivertown",
                    "lat": 27.9,
                    "lon": 85.0,
                    "population": 12000,
                    "alert_channels": ["siren", "sms"],
                },
                {
                    "name": "Bridgeport",
                    "lat": 27.8,
                    "lon": 85.1,
                    "population": 4500,
                    "alert_channels": ["sms"],
                },
            ],
        },
    }


@pytest.fixture
def tws(site_config):
    return TacticalWarningSystem.from_config(site_config)


@pytest.fixture
def trigger():
    return SeismicTrigger(
        timestamp=datetime(2026, 9, 7, 12, 0, 0, tzinfo=timezone.utc),
        magnitude=3.2,
        lat=28.005,
        lon=85.005,
        source="USGS",
        confidence=0.85,
    )


# ---------------------------------------------------------------------------
# SeismicTrigger dataclass
# ---------------------------------------------------------------------------


class TestSeismicTrigger:
    def test_valid_construction(self, trigger):
        assert trigger.magnitude == 3.2
        assert trigger.source == "USGS"
        assert trigger.confidence == 0.85

    def test_confidence_zero_is_valid(self):
        t = SeismicTrigger(
            timestamp=datetime.now(timezone.utc),
            magnitude=1.0, lat=0.0, lon=0.0,
            source="test", confidence=0.0,
        )
        assert t.confidence == 0.0

    def test_confidence_one_is_valid(self):
        t = SeismicTrigger(
            timestamp=datetime.now(timezone.utc),
            magnitude=1.0, lat=0.0, lon=0.0,
            source="test", confidence=1.0,
        )
        assert t.confidence == 1.0

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValueError, match="confidence must be in"):
            SeismicTrigger(
                timestamp=datetime.now(timezone.utc),
                magnitude=1.0, lat=0.0, lon=0.0,
                source="test", confidence=1.5,
            )

    def test_negative_confidence_raises(self):
        with pytest.raises(ValueError, match="confidence must be in"):
            SeismicTrigger(
                timestamp=datetime.now(timezone.utc),
                magnitude=1.0, lat=0.0, lon=0.0,
                source="test", confidence=-0.1,
            )


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------


class TestHaversine:
    def test_same_point_is_zero(self):
        assert _haversine_km(28.0, 85.0, 28.0, 85.0) == 0.0

    def test_known_distance(self):
        # Kathmandu (27.7172, 85.3240) to Pokhara (28.2096, 83.9856)
        # is approximately 133 km.
        dist = _haversine_km(27.7172, 85.3240, 28.2096, 83.9856)
        assert 130 < dist < 145

    def test_one_degree_latitude(self):
        # 1 degree of latitude ~ 111 km
        dist = _haversine_km(0.0, 0.0, 1.0, 0.0)
        assert 110 < dist < 112


# ---------------------------------------------------------------------------
# TacticalWarningSystem construction
# ---------------------------------------------------------------------------


class TestTWSConstruction:
    def test_from_config(self, tws, site_config):
        assert len(tws.seismic_sources) == 1
        assert tws.wave_speed_m_s == 5.0
        assert len(tws.communities) == 2
        assert tws.communities[0].name == "Rivertown"
        assert tws.communities[0].population == 12000
        assert tws.communities[0].alert_channels == ["siren", "sms"]

    def test_empty_config_uses_defaults(self):
        tws = TacticalWarningSystem({})
        assert tws.wave_speed_m_s == 5.0
        assert tws.communities == []
        assert tws.trigger_thresholds["min_magnitude"] == 1.5

    def test_from_config_classmethod(self, site_config):
        tws = TacticalWarningSystem.from_config(site_config)
        assert len(tws.communities) == 2


# ---------------------------------------------------------------------------
# check_seismic_feeds (stub)
# ---------------------------------------------------------------------------


class TestCheckSeismicFeeds:
    def test_stub_returns_empty_list(self, tws):
        result = tws.check_seismic_feeds()
        assert result == []

    def test_stub_accepts_sites_arg(self, tws, site_config):
        result = tws.check_seismic_feeds(sites=[site_config])
        assert result == []


# ---------------------------------------------------------------------------
# evaluate_trigger
# ---------------------------------------------------------------------------


class TestEvaluateTrigger:
    def test_trigger_within_thresholds_accepted(self, tws, trigger, site_config):
        # trigger is at (28.005, 85.005), site at (28.0, 85.0)
        # distance ~ 0.7 km, magnitude 3.2 >= 2.0, confidence 0.85 >= 0.7
        assert tws.evaluate_trigger(trigger, site_config) is True

    def test_trigger_too_far_rejected(self, tws, site_config):
        far_trigger = SeismicTrigger(
            timestamp=datetime.now(timezone.utc),
            magnitude=5.0, lat=29.0, lon=85.0,  # ~111 km away
            source="USGS", confidence=0.9,
        )
        assert tws.evaluate_trigger(far_trigger, site_config) is False

    def test_trigger_below_min_magnitude_rejected(self, tws, site_config):
        weak_trigger = SeismicTrigger(
            timestamp=datetime.now(timezone.utc),
            magnitude=1.0,  # below min_magnitude=2.0
            lat=28.001, lon=85.001,
            source="USGS", confidence=0.9,
        )
        assert tws.evaluate_trigger(weak_trigger, site_config) is False

    def test_trigger_below_min_confidence_rejected(self, tws, site_config):
        low_conf_trigger = SeismicTrigger(
            timestamp=datetime.now(timezone.utc),
            magnitude=3.0,
            lat=28.001, lon=85.001,
            source="USGS", confidence=0.5,  # below min_confidence=0.7
        )
        assert tws.evaluate_trigger(low_conf_trigger, site_config) is False

    def test_trigger_at_exact_threshold_accepted(self, tws, site_config):
        """Boundary: magnitude and confidence exactly at threshold."""
        exact_trigger = SeismicTrigger(
            timestamp=datetime.now(timezone.utc),
            magnitude=2.0,  # exactly min_magnitude
            lat=28.0, lon=85.0,  # distance 0
            source="IRIS", confidence=0.7,  # exactly min_confidence
        )
        assert tws.evaluate_trigger(exact_trigger, site_config) is True


# ---------------------------------------------------------------------------
# compute_arrival_times
# ---------------------------------------------------------------------------


class TestComputeArrivalTimes:
    def test_known_distance_arrival(self):
        """10 km at 5 m/s = 2000 s = 33.3 min."""
        tws = TacticalWarningSystem({
            "tactical": {
                "wave_speed_m_s": 5.0,
                "wave_speed_uncertainty": 0.0,
            },
        })
        communities = [
            DownstreamCommunity(name="Town A", lat=0.0, lon=0.0),
        ]
        # Place collapse ~10 km north (0.09 degrees latitude ~ 10 km)
        estimates = tws.compute_arrival_times(
            collapse_lat=0.09, collapse_lon=0.0,
            downstream_communities=communities,
        )
        assert len(estimates) == 1
        est = estimates[0]
        # ~10 km / 5 m/s = 2000 s = 33.3 min
        assert 32.0 < est.arrival_minutes < 35.0
        assert est.wave_speed_m_s == 5.0

    def test_zero_distance_yields_zero_arrival(self, tws):
        communities = [
            DownstreamCommunity(name="On Site", lat=28.0, lon=85.0),
        ]
        estimates = tws.compute_arrival_times(
            collapse_lat=28.0, collapse_lon=85.0,
            downstream_communities=communities,
        )
        assert len(estimates) == 1
        assert estimates[0].arrival_minutes == 0.0
        assert estimates[0].uncertainty_minutes == 0.0

    def test_sorted_by_arrival_time(self, tws):
        near = DownstreamCommunity(name="Near", lat=28.01, lon=85.0)
        far = DownstreamCommunity(name="Far", lat=27.5, lon=85.0)
        estimates = tws.compute_arrival_times(
            collapse_lat=28.0, collapse_lon=85.0,
            downstream_communities=[far, near],  # out of order
        )
        assert estimates[0].community_name == "Near"
        assert estimates[1].community_name == "Far"
        assert estimates[0].arrival_minutes <= estimates[1].arrival_minutes

    def test_uncertainty_propagation(self):
        """With 30% speed uncertainty, arrival uncertainty should be
        t * 0.3 / 0.7 ~ 0.429 * t."""
        tws = TacticalWarningSystem({
            "tactical": {
                "wave_speed_m_s": 10.0,
                "wave_speed_uncertainty": 0.3,
            },
        })
        communities = [
            DownstreamCommunity(name="Town", lat=0.0, lon=0.0),
        ]
        estimates = tws.compute_arrival_times(
            collapse_lat=0.09, collapse_lon=0.0,
            downstream_communities=communities,
        )
        est = estimates[0]
        expected_ratio = 0.3 / 0.7  # ~0.4286
        actual_ratio = est.uncertainty_minutes / est.arrival_minutes if est.arrival_minutes else 0
        assert abs(actual_ratio - expected_ratio) < 0.05

    def test_zero_uncertainty(self):
        """Zero speed uncertainty produces zero arrival uncertainty."""
        tws = TacticalWarningSystem({
            "tactical": {
                "wave_speed_m_s": 5.0,
                "wave_speed_uncertainty": 0.0,
            },
        })
        communities = [
            DownstreamCommunity(name="Town", lat=0.1, lon=0.0),
        ]
        estimates = tws.compute_arrival_times(
            collapse_lat=0.0, collapse_lon=0.0,
            downstream_communities=communities,
        )
        assert estimates[0].uncertainty_minutes == 0.0

    def test_uses_instance_communities_as_default(self, tws):
        estimates = tws.compute_arrival_times(
            collapse_lat=28.0, collapse_lon=85.0,
        )
        assert len(estimates) == 2
        names = {e.community_name for e in estimates}
        assert names == {"Rivertown", "Bridgeport"}

    def test_empty_communities_returns_empty(self, tws):
        estimates = tws.compute_arrival_times(
            collapse_lat=28.0, collapse_lon=85.0,
            downstream_communities=[],
        )
        assert estimates == []

    def test_non_positive_wave_speed_skips(self):
        """Non-positive wave speed should skip (not crash)."""
        tws = TacticalWarningSystem({
            "tactical": {"wave_speed_m_s": 0.0},
        })
        communities = [
            DownstreamCommunity(name="Town", lat=0.1, lon=0.0),
        ]
        estimates = tws.compute_arrival_times(
            collapse_lat=0.0, collapse_lon=0.0,
            downstream_communities=communities,
        )
        assert estimates == []


# ---------------------------------------------------------------------------
# generate_tactical_alert
# ---------------------------------------------------------------------------


class TestGenerateTacticalAlert:
    def test_alert_structure(self, tws, trigger, site_config):
        arrival_times = [
            ArrivalTimeEstimate(
                community_name="Rivertown",
                distance_km=11.1,
                wave_speed_m_s=5.0,
                arrival_minutes=37.0,
                uncertainty_minutes=16.0,
            ),
        ]
        alert = tws.generate_tactical_alert(trigger, site_config, arrival_times)

        assert alert["level"] == "CRITICAL"
        assert alert["type"] == "tactical"
        assert alert["site"] == "Test Glacier"
        assert "alert_id" in alert
        assert "tactical-Test_Glacier-" in alert["alert_id"]
        assert "timestamp" in alert

    def test_alert_message_contains_key_info(self, tws, trigger, site_config):
        arrival_times = [
            ArrivalTimeEstimate(
                community_name="Rivertown",
                distance_km=11.1,
                wave_speed_m_s=5.0,
                arrival_minutes=37.0,
                uncertainty_minutes=16.0,
            ),
        ]
        alert = tws.generate_tactical_alert(trigger, site_config, arrival_times)

        assert "TACTICAL WARNING" in alert["message"]
        assert "Test Glacier" in alert["message"]
        assert "M3.2" in alert["message"]
        assert "USGS" in alert["message"]
        assert "Rivertown" in alert["message"]
        assert "37 min" in alert["message"]
        assert "85%" in alert["message"]

    def test_alert_trigger_field(self, tws, trigger, site_config):
        alert = tws.generate_tactical_alert(trigger, site_config, [])

        assert alert["trigger"]["magnitude"] == 3.2
        assert alert["trigger"]["source"] == "USGS"
        assert alert["trigger"]["confidence"] == 0.85

    def test_alert_arrival_times_field(self, tws, trigger, site_config):
        arrival_times = [
            ArrivalTimeEstimate("A", 5.0, 5.0, 17.0, 7.0),
            ArrivalTimeEstimate("B", 10.0, 5.0, 33.0, 14.0),
        ]
        alert = tws.generate_tactical_alert(trigger, site_config, arrival_times)

        assert len(alert["arrival_times"]) == 2
        assert alert["arrival_times"][0]["community"] == "A"
        assert alert["arrival_times"][0]["arrival_minutes"] == 17.0
        assert alert["arrival_times"][1]["community"] == "B"

    def test_alert_with_no_communities(self, tws, trigger, site_config):
        alert = tws.generate_tactical_alert(trigger, site_config, [])

        assert "No downstream communities configured" in alert["message"]
        assert alert["arrival_times"] == []


# ---------------------------------------------------------------------------
# escalate
# ---------------------------------------------------------------------------


class TestEscalate:
    def test_dispatches_at_critical_level(self, tws, trigger, site_config):
        mock_dispatcher = MagicMock()
        mock_dispatcher.dispatch.return_value = {"slack": True}

        alert = tws.generate_tactical_alert(trigger, site_config, [])
        result = tws.escalate(alert, mock_dispatcher)

        mock_dispatcher.dispatch.assert_called_once()
        call_kwargs = mock_dispatcher.dispatch.call_args
        assert call_kwargs[1]["alert_level"] == "CRITICAL" or call_kwargs[0][0] == "CRITICAL"
        assert result == {"slack": True}

    def test_escalate_passes_site_name(self, tws, trigger, site_config):
        mock_dispatcher = MagicMock()
        mock_dispatcher.dispatch.return_value = {}

        alert = tws.generate_tactical_alert(trigger, site_config, [])
        tws.escalate(alert, mock_dispatcher)

        call_args = mock_dispatcher.dispatch.call_args
        assert call_args[1]["site_name"] == "Test Glacier" or call_args[0][1] == "Test Glacier"

    def test_escalate_passes_message(self, tws, trigger, site_config):
        mock_dispatcher = MagicMock()
        mock_dispatcher.dispatch.return_value = {}

        alert = tws.generate_tactical_alert(trigger, site_config, [])
        tws.escalate(alert, mock_dispatcher)

        call_args = mock_dispatcher.dispatch.call_args
        msg = call_args[1].get("message") or call_args[0][2]
        assert "TACTICAL WARNING" in msg

    def test_escalate_details_exclude_level_site_message(self, tws, trigger, site_config):
        mock_dispatcher = MagicMock()
        mock_dispatcher.dispatch.return_value = {}

        alert = tws.generate_tactical_alert(trigger, site_config, [])
        tws.escalate(alert, mock_dispatcher)

        call_args = mock_dispatcher.dispatch.call_args
        details = call_args[1].get("details") or call_args[0][3]
        assert "level" not in details
        assert "site" not in details
        assert "message" not in details
        assert "alert_id" in details
        assert "trigger" in details

    def test_escalate_returns_dispatcher_result(self, tws, trigger, site_config):
        mock_dispatcher = MagicMock()
        mock_dispatcher.dispatch.return_value = {"email": False, "sms": True}

        alert = tws.generate_tactical_alert(trigger, site_config, [])
        result = tws.escalate(alert, mock_dispatcher)

        assert result == {"email": False, "sms": True}


# ---------------------------------------------------------------------------
# DownstreamCommunity dataclass
# ---------------------------------------------------------------------------


class TestDownstreamCommunity:
    def test_defaults(self):
        c = DownstreamCommunity(name="Town", lat=0.0, lon=0.0)
        assert c.population == 0
        assert c.distance_km == 0.0
        assert c.estimated_arrival_minutes == 0.0
        assert c.alert_channels == []

    def test_full_construction(self):
        c = DownstreamCommunity(
            name="Big City", lat=27.5, lon=85.5,
            population=50000, distance_km=30.0,
            estimated_arrival_minutes=100.0,
            alert_channels=["siren", "sms", "radio"],
        )
        assert c.name == "Big City"
        assert c.alert_channels == ["siren", "sms", "radio"]


# ---------------------------------------------------------------------------
# ArrivalTimeEstimate dataclass
# ---------------------------------------------------------------------------


class TestArrivalTimeEstimate:
    def test_construction(self):
        est = ArrivalTimeEstimate(
            community_name="Town",
            distance_km=15.0,
            wave_speed_m_s=5.0,
            arrival_minutes=50.0,
            uncertainty_minutes=21.4,
        )
        assert est.community_name == "Town"
        assert est.distance_km == 15.0
        assert est.arrival_minutes == 50.0
