"""
Seismic/infrasound tactical warning integration for GLEWS.

Once strategic monitoring (InSAR anomaly detection) flags a site as
high-risk, tactical sensors — seismometers and infrasound arrays near
the site — provide the final seconds-to-minutes warning when collapse
actually begins.  This module bridges the gap between "this slope is
unstable" (hours/days) and "it is moving now" (seconds).

The tactical warning chain:

    1. Seismic/infrasound networks detect an event near a flagged site.
    2. The event is evaluated against collapse signatures (proximity,
       magnitude, frequency content, timing relative to strategic
       warning level).
    3. If the trigger matches, flood/debris arrival times are computed
       for each downstream community using distance and empirical
       wave-speed models.
    4. An urgent alert with community-specific countdown timers is
       generated and dispatched at CRITICAL level through the existing
       AlertDispatcher.

Configuration lives under the ``tactical:`` key in the site YAML::

    tactical:
      seismic_sources:
        - name: USGS
          url: https://earthquake.usgs.gov/fdsnws/event/1/query
        - name: IRIS
          url: https://service.iris.edu/fdsnws/event/1/query
      trigger_thresholds:
        min_magnitude: 1.5
        max_distance_km: 15.0
        min_confidence: 0.6
      wave_speed_m_s: 5.0          # empirical debris-flow front speed
      wave_speed_uncertainty: 0.3   # fractional uncertainty (+/- 30 %)
      downstream_communities:
        - name: Rivertown
          lat: 28.10
          lon: 85.35
          population: 12000
          alert_channels: [siren, sms]
        - name: Bridgeport
          lat: 28.05
          lon: 85.40
          population: 4500
          alert_channels: [sms]

Usage::

    from glews.tactical import TacticalWarningSystem

    tws = TacticalWarningSystem.from_config(site_config)
    triggers = tws.check_seismic_feeds(sites=[site_config])
    for trigger in triggers:
        arrival_times = tws.compute_arrival_times(
            trigger.lat, trigger.lon, tws.communities,
        )
        alert = tws.generate_tactical_alert(trigger, site_config, arrival_times)
        tws.escalate(alert, dispatcher)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Earth's mean radius in km — used for Haversine distance calculations.
_EARTH_RADIUS_KM = 6371.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SeismicTrigger:
    """A seismic or infrasound event detected near a monitored site.

    Attributes
    ----------
    timestamp : datetime
        UTC time of the event origin.
    magnitude : float
        Event magnitude (local or moment magnitude).
    lat : float
        Event latitude (degrees).
    lon : float
        Event longitude (degrees).
    source : str
        Data source identifier (e.g. "USGS", "IRIS", "local_network").
    confidence : float
        Confidence that this event represents a collapse trigger,
        ranging from 0.0 (no confidence) to 1.0 (certain).
    """

    timestamp: datetime
    magnitude: float
    lat: float
    lon: float
    source: str
    confidence: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence}"
            )


@dataclass
class DownstreamCommunity:
    """A population center downstream of a monitored site that must
    receive tactical warnings.

    Attributes
    ----------
    name : str
        Human-readable community name.
    lat : float
        Community latitude (degrees).
    lon : float
        Community longitude (degrees).
    population : int
        Approximate population at risk.
    distance_km : float
        Pre-computed or dynamically computed distance from the monitored
        site (km).  May be overwritten by arrival-time computation.
    estimated_arrival_minutes : float
        Estimated debris/flood arrival time (minutes from collapse).
        Filled in by ``compute_arrival_times``.
    alert_channels : list[str]
        Notification channels for this community (e.g. ["siren", "sms"]).
    """

    name: str
    lat: float
    lon: float
    population: int = 0
    distance_km: float = 0.0
    estimated_arrival_minutes: float = 0.0
    alert_channels: list[str] = field(default_factory=list)


@dataclass
class ArrivalTimeEstimate:
    """Estimated arrival time of a flood/debris wave at a downstream
    community.

    Attributes
    ----------
    community_name : str
        Name of the downstream community.
    distance_km : float
        Distance from the collapse origin to the community (km).
    wave_speed_m_s : float
        Assumed debris-flow front speed (m/s).
    arrival_minutes : float
        Estimated arrival time (minutes from collapse origin).
    uncertainty_minutes : float
        Uncertainty band on the arrival estimate (minutes).
    """

    community_name: str
    distance_km: float
    wave_speed_m_s: float
    arrival_minutes: float
    uncertainty_minutes: float


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points on Earth (km)."""
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    )
    return _EARTH_RADIUS_KM * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# TacticalWarningSystem
# ---------------------------------------------------------------------------


class TacticalWarningSystem:
    """Integrates seismic/infrasound detections with the GLEWS strategic
    warning pipeline to issue tactical (seconds-to-minutes) alerts for
    downstream communities when collapse begins.

    Parameters
    ----------
    config : dict
        Site configuration dict containing a ``tactical:`` section.
    """

    def __init__(self, config: dict) -> None:
        tactical = config.get("tactical", {})

        self.seismic_sources: list[dict] = tactical.get("seismic_sources", [])
        self.trigger_thresholds: dict = tactical.get("trigger_thresholds", {
            "min_magnitude": 1.5,
            "max_distance_km": 15.0,
            "min_confidence": 0.6,
        })
        self.wave_speed_m_s: float = float(tactical.get("wave_speed_m_s", 5.0))
        self.wave_speed_uncertainty: float = float(
            tactical.get("wave_speed_uncertainty", 0.3)
        )

        raw_communities = tactical.get("downstream_communities", [])
        self.communities: list[DownstreamCommunity] = [
            DownstreamCommunity(
                name=c["name"],
                lat=float(c["lat"]),
                lon=float(c["lon"]),
                population=int(c.get("population", 0)),
                alert_channels=c.get("alert_channels", []),
            )
            for c in raw_communities
        ]

    @classmethod
    def from_config(cls, site_config: dict) -> "TacticalWarningSystem":
        """Convenience constructor matching the project convention."""
        return cls(site_config)

    # -- (a) poll seismic feeds ------------------------------------------------

    def check_seismic_feeds(
        self, sites: list[dict] | None = None,
    ) -> list[SeismicTrigger]:
        """Poll configured seismic data sources for events near monitored
        sites.

        .. note::

            This is a **stub**.  A production implementation would:

            1. For each source in ``self.seismic_sources``, issue an FDSN
               web-service query (``/fdsnws/event/1/query``) with
               parameters::

                   starttime = <last_check or now - polling_interval>
                   endtime   = <now>
                   latitude  = <site_lat>
                   longitude = <site_lon>
                   maxradius = <max_distance_km converted to degrees>
                   minmagnitude = <min_magnitude>
                   format    = geojson

            2. Parse the GeoJSON feature collection returned by
               USGS/IRIS/local networks.

            3. Convert each feature to a ``SeismicTrigger`` with a
               confidence score derived from:
               - Network detection quality / number of reporting stations
               - Spectral characteristics (collapse vs tectonic signature)
               - Proximity to a flagged site

            4. Return the list of triggers, newest first.

        Parameters
        ----------
        sites : list[dict] or None
            Site configs to check.  Each must have ``site.latitude`` and
            ``site.longitude``.  Ignored in the stub.

        Returns
        -------
        list[SeismicTrigger]
            Empty in this stub implementation.
        """
        logger.debug(
            "check_seismic_feeds: stub — would query %d source(s)",
            len(self.seismic_sources),
        )
        return []

    # -- (b) evaluate a seismic trigger ----------------------------------------

    def evaluate_trigger(
        self,
        seismic_event: SeismicTrigger,
        site_config: dict,
    ) -> bool:
        """Evaluate whether a seismic event matches a collapse signature
        for a monitored site.

        The evaluation checks:

        - **Proximity**: the event must be within ``max_distance_km``
          of the site.
        - **Magnitude**: the event must meet or exceed ``min_magnitude``.
        - **Confidence**: the event's confidence score must meet or
          exceed ``min_confidence``.

        A production system would additionally examine:

        - Frequency content (collapse generates broadband, low-frequency
          signals distinct from tectonic earthquakes).
        - Duration and amplitude envelope (landslide signals ramp up
          more slowly than tectonic P-wave onsets).
        - Correlation with the current strategic warning level — a site
          at WARNING level lowers the trigger threshold.

        Parameters
        ----------
        seismic_event : SeismicTrigger
            Detected seismic event.
        site_config : dict
            Site config with ``site.latitude``, ``site.longitude``.

        Returns
        -------
        bool
            True if the event should trigger a tactical alert.
        """
        site = site_config["site"]
        site_lat = float(site["latitude"])
        site_lon = float(site["longitude"])

        thresholds = self.trigger_thresholds
        max_dist = float(thresholds.get("max_distance_km", 15.0))
        min_mag = float(thresholds.get("min_magnitude", 1.5))
        min_conf = float(thresholds.get("min_confidence", 0.6))

        distance = _haversine_km(
            seismic_event.lat, seismic_event.lon, site_lat, site_lon,
        )

        if distance > max_dist:
            logger.debug(
                "Trigger rejected: distance %.1f km > max %.1f km",
                distance, max_dist,
            )
            return False

        if seismic_event.magnitude < min_mag:
            logger.debug(
                "Trigger rejected: magnitude %.1f < min %.1f",
                seismic_event.magnitude, min_mag,
            )
            return False

        if seismic_event.confidence < min_conf:
            logger.debug(
                "Trigger rejected: confidence %.2f < min %.2f",
                seismic_event.confidence, min_conf,
            )
            return False

        logger.info(
            "Trigger ACCEPTED: M%.1f at (%.3f, %.3f), %.1f km from site, "
            "confidence %.2f",
            seismic_event.magnitude,
            seismic_event.lat,
            seismic_event.lon,
            distance,
            seismic_event.confidence,
        )
        return True

    # -- (c) compute arrival times ---------------------------------------------

    def compute_arrival_times(
        self,
        collapse_lat: float,
        collapse_lon: float,
        downstream_communities: list[DownstreamCommunity] | None = None,
    ) -> list[ArrivalTimeEstimate]:
        """Estimate flood/debris arrival times at downstream points.

        Uses straight-line (Haversine) distance and an empirical
        wave-speed model.  A production system would use channel
        geometry and hydraulic routing models for more accurate
        estimates, but this provides a conservative first approximation.

        Parameters
        ----------
        collapse_lat, collapse_lon : float
            Coordinates of the collapse origin (degrees).
        downstream_communities : list[DownstreamCommunity] or None
            Communities to compute for.  Defaults to
            ``self.communities``.

        Returns
        -------
        list[ArrivalTimeEstimate]
            One estimate per community, sorted by ascending arrival time.
        """
        communities = downstream_communities if downstream_communities is not None else self.communities
        estimates: list[ArrivalTimeEstimate] = []

        for community in communities:
            distance_km = _haversine_km(
                collapse_lat, collapse_lon, community.lat, community.lon,
            )
            distance_m = distance_km * 1000.0

            if self.wave_speed_m_s <= 0:
                logger.warning(
                    "wave_speed_m_s is non-positive (%.2f); "
                    "cannot compute arrival for %s",
                    self.wave_speed_m_s, community.name,
                )
                continue

            arrival_seconds = distance_m / self.wave_speed_m_s
            arrival_minutes = arrival_seconds / 60.0

            # Uncertainty propagation: if speed is v +/- u*v, then
            # arrival time t = d/v has uncertainty ~ t * u / (1 - u)
            # (from the slow-end bound d / (v*(1-u))).
            u = self.wave_speed_uncertainty
            if u < 1.0:
                uncertainty_minutes = arrival_minutes * u / (1.0 - u)
            else:
                uncertainty_minutes = float("inf")

            estimates.append(ArrivalTimeEstimate(
                community_name=community.name,
                distance_km=round(distance_km, 2),
                wave_speed_m_s=self.wave_speed_m_s,
                arrival_minutes=round(arrival_minutes, 1),
                uncertainty_minutes=round(uncertainty_minutes, 1),
            ))

        estimates.sort(key=lambda e: e.arrival_minutes)
        return estimates

    # -- (d) generate tactical alert -------------------------------------------

    def generate_tactical_alert(
        self,
        trigger: SeismicTrigger,
        site_config: dict,
        arrival_times: list[ArrivalTimeEstimate],
    ) -> dict[str, Any]:
        """Create an urgent alert message with community-specific
        countdown timers.

        Parameters
        ----------
        trigger : SeismicTrigger
            The seismic event that triggered the alert.
        site_config : dict
            Site configuration (for the site name and coordinates).
        arrival_times : list[ArrivalTimeEstimate]
            Pre-computed arrival time estimates for downstream
            communities.

        Returns
        -------
        dict
            Alert record compatible with the existing alert pipeline,
            with additional tactical fields.
        """
        site = site_config["site"]
        site_name = site["name"]
        now = datetime.now(timezone.utc)

        # Build per-community countdown lines
        countdown_lines: list[str] = []
        for est in arrival_times:
            countdown_lines.append(
                f"  {est.community_name}: ~{est.arrival_minutes:.0f} min "
                f"(+/- {est.uncertainty_minutes:.0f} min), "
                f"{est.distance_km:.1f} km downstream"
            )

        countdown_text = (
            "\n".join(countdown_lines)
            if countdown_lines
            else "  No downstream communities configured."
        )

        message = (
            f"TACTICAL WARNING: Seismic trigger detected near {site_name}. "
            f"M{trigger.magnitude:.1f} event at "
            f"({trigger.lat:.4f}, {trigger.lon:.4f}) "
            f"detected by {trigger.source} at "
            f"{trigger.timestamp.strftime('%H:%M:%S UTC')}. "
            f"Confidence: {trigger.confidence:.0%}.\n"
            f"Estimated arrival times:\n{countdown_text}"
        )

        alert: dict[str, Any] = {
            "alert_id": (
                f"tactical-{site_name.replace(' ', '_')}-"
                f"{trigger.timestamp.strftime('%Y%m%dT%H%M%S')}"
            ),
            "site": site_name,
            "timestamp": now.isoformat(),
            "level": "CRITICAL",
            "type": "tactical",
            "message": message,
            "trigger": {
                "magnitude": trigger.magnitude,
                "lat": trigger.lat,
                "lon": trigger.lon,
                "source": trigger.source,
                "confidence": trigger.confidence,
                "event_time": trigger.timestamp.isoformat(),
            },
            "arrival_times": [
                {
                    "community": est.community_name,
                    "distance_km": est.distance_km,
                    "arrival_minutes": est.arrival_minutes,
                    "uncertainty_minutes": est.uncertainty_minutes,
                }
                for est in arrival_times
            ],
        }

        return alert

    # -- (e) escalate via AlertDispatcher --------------------------------------

    def escalate(
        self,
        alert: dict[str, Any],
        dispatcher: Any,
    ) -> dict[str, bool]:
        """Send a tactical alert via the AlertDispatcher at CRITICAL level.

        Parameters
        ----------
        alert : dict
            Alert record from ``generate_tactical_alert``.
        dispatcher : AlertDispatcher
            Configured dispatcher instance (from ``glews.alerts``).

        Returns
        -------
        dict[str, bool]
            Per-channel success/failure from the dispatcher.
        """
        logger.critical(
            "[%s] TACTICAL ALERT: %s", alert["site"], alert["message"],
        )

        details = {
            k: v for k, v in alert.items()
            if k not in ("level", "site", "message")
        }

        return dispatcher.dispatch(
            alert_level="CRITICAL",
            site_name=alert["site"],
            message=alert["message"],
            details=details,
        )
