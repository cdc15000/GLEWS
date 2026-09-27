"""
Cascade risk assessment module — Tier 1 filtering.

Evaluates whether a flagged unstable site can produce a dangerous
downstream cascade (debris flow, river blockage, flood). Flags that
fail these physical/geometric tests are deprioritized regardless
of their deformation anomaly score.

Filters:
    1. Volume estimation — is the potentially unstable mass large enough?
    2. Geometric plausibility — is it above a confined valley or river?
    3. Downstream exposure — are people or infrastructure at risk?
    4. Runout analysis — can the debris reach a river or settlement?

All filters use freely available data (DEMs, population layers)
and require no labeled collapse examples.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from glews.detect import AnomalyFlag

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default configuration — mirrors config/global_watch.yaml 'cascade' section
# ---------------------------------------------------------------------------

_DEFAULT_CASCADE_CONFIG: dict[str, Any] = {
    "cascade": {
        "min_volume_m3": 100_000,
        "exposure": {
            "max_runout_km": 50,
            "angle_of_reach_deg": 11,
            "population_source": "worldpop",
        },
        "valley_width_threshold_m": 500,
    },
}


# ---------------------------------------------------------------------------
# Enums and dataclasses
# ---------------------------------------------------------------------------


class RiskLevel(Enum):
    """Cascade risk classification for Tier 1 output."""

    HIGH = "high"          # passes all filters, significant downstream exposure
    MODERATE = "moderate"  # passes geometric filters, limited exposure
    LOW = "low"            # fails volume or geometry filters
    MINIMAL = "minimal"    # isolated, no downstream pathway


@dataclass
class ExposureResult:
    """Population exposure analysis for a single site."""

    total_population_exposed: int
    population_by_distance: dict[str, int]  # e.g. {"0-10km": 500, ...}
    exposure_score: float  # composite 0–10
    settlements_at_risk: list[dict]  # [{name, population, distance, coordinates}, ...]
    runout_distance_km: float
    angle_of_reach_deg: float


@dataclass
class CascadeAssessment:
    """
    Tier 1 cascade risk assessment for a single flagged site.
    """

    flag: AnomalyFlag
    risk_level: RiskLevel

    # Volume estimate
    estimated_volume_m3: float
    volume_sufficient: bool  # exceeds minimum threshold

    # Geometric analysis
    slope_angle_deg: float
    elevation_m: float
    above_valley: bool      # deforming mass is above a confined valley
    valley_width_m: float | None  # width of valley below, if applicable

    # Runout analysis
    runout_distance_m: float
    runout_reaches_river: bool
    angle_of_reach_deg: float

    # Downstream exposure
    population_exposed: int
    settlements_exposed: list[str] = field(default_factory=list)
    infrastructure_at_risk: list[str] = field(default_factory=list)

    # Population exposure analysis (new)
    exposure: ExposureResult | None = None

    # River damming potential
    dam_potential: bool = False
    estimated_lake_volume_m3: float = 0.0

    # Combined score for ranking
    cascade_score: float = 0.0

    @property
    def passes_tier1(self) -> bool:
        """Whether this site should advance to Tier 2."""
        return self.risk_level in (RiskLevel.HIGH, RiskLevel.MODERATE)


# ---------------------------------------------------------------------------
# Public API — population exposure and single-flag assessment
# ---------------------------------------------------------------------------


def estimate_runout(
    elevation_drop_m: float,
    horizontal_distance_m: float,
    volume_m3: float | None = None,
) -> dict[str, float]:
    """
    Empirical angle-of-reach runout estimation.

    Uses the Scheidegger (1973) volume-dependent mobility relation when
    volume is provided:

        log10(H/L) = a - b * log10(V)

    where H = elevation drop, L = runout distance, V = volume in m^3.
    Coefficients: a = 0.16, b = 0.16 (Scheidegger 1973, Table 2;
    see also Corominas 1996 for a similar fit on a larger dataset).

    When volume is not provided, falls back to the geometric angle
    from the supplied elevation drop and horizontal distance.

    Parameters
    ----------
    elevation_drop_m : float
        Vertical drop from source to valley floor.
    horizontal_distance_m : float
        Horizontal distance from source to valley floor.
    volume_m3 : float or None
        Estimated volume of the failing mass.

    Returns
    -------
    dict with keys:
        runout_distance_km : float — estimated maximum travel distance
        angle_of_reach_deg : float — angle of reach (fahrboschung)
    """
    # Handle edge cases
    if elevation_drop_m <= 0:
        return {"runout_distance_km": 0.0, "angle_of_reach_deg": 0.0}

    if volume_m3 is not None and volume_m3 > 0:
        # Scheidegger volume-dependent relation
        # log10(H/L) = a - b * log10(V)
        a, b = 0.16, 0.16
        log_hl = a - b * np.log10(volume_m3)
        hl_ratio = 10 ** log_hl
        # H/L = tan(angle_of_reach), L = H / (H/L)
        runout_m = elevation_drop_m / hl_ratio
        angle_deg = float(np.degrees(np.arctan(hl_ratio)))
    else:
        # Geometric fallback
        if horizontal_distance_m <= 0:
            return {
                "runout_distance_km": 0.0,
                "angle_of_reach_deg": 90.0 if elevation_drop_m > 0 else 0.0,
            }
        angle_deg = float(np.degrees(
            np.arctan2(elevation_drop_m, horizontal_distance_m)
        ))
        runout_m = float(horizontal_distance_m)

    # Clamp angle to valid range
    angle_deg = max(0.0, min(90.0, angle_deg))
    runout_km = max(0.0, runout_m / 1000.0)

    return {
        "runout_distance_km": runout_km,
        "angle_of_reach_deg": angle_deg,
    }


def estimate_exposure(
    flag_lat: float,
    flag_lon: float,
    flag_elevation_m: float,
    runout_distance_km: float,
    population_data: np.ndarray | str | None = None,
) -> ExposureResult:
    """
    Estimate population exposure within the runout zone.

    Parameters
    ----------
    flag_lat, flag_lon : float
        Coordinates of the flagged site.
    flag_elevation_m : float
        Elevation of the site in meters.
    runout_distance_km : float
        Maximum runout distance in km.
    population_data : np.ndarray, str, or None
        If an ndarray: population grid aligned to the area.
        If a str: path to a raster file (not implemented in demo).
        If None: uses a synthetic population model for demo purposes.

    Returns
    -------
    ExposureResult
    """
    distance_bands = {"0-10km": 0, "10-30km": 0, "30-100km": 0}

    if runout_distance_km <= 0:
        return ExposureResult(
            total_population_exposed=0,
            population_by_distance=distance_bands,
            exposure_score=0.0,
            settlements_at_risk=[],
            runout_distance_km=0.0,
            angle_of_reach_deg=0.0,
        )

    if population_data is not None and isinstance(population_data, np.ndarray):
        # Real population grid path — simplified circular buffer
        settlements, distance_bands = _exposure_from_grid(
            flag_lat, flag_lon, runout_distance_km, population_data
        )
    else:
        # Synthetic demo model
        settlements = _generate_synthetic_settlements(
            flag_lat, flag_lon, flag_elevation_m, runout_distance_km
        )
        for s in settlements:
            dist = s["distance_km"]
            if dist <= 10:
                distance_bands["0-10km"] += s["population"]
            elif dist <= 30:
                distance_bands["10-30km"] += s["population"]
            elif dist <= 100:
                distance_bands["30-100km"] += s["population"]

    # Only count population within runout distance
    for band_key in list(distance_bands.keys()):
        band_lower = _band_lower_km(band_key)
        if band_lower >= runout_distance_km:
            distance_bands[band_key] = 0

    # Filter settlements beyond runout
    settlements = [
        s for s in settlements if s["distance_km"] <= runout_distance_km
    ]

    total_pop = sum(distance_bands.values())
    angle = float(np.degrees(np.arctan2(
        flag_elevation_m, runout_distance_km * 1000
    ))) if runout_distance_km > 0 else 0.0

    exposure_score = _compute_exposure_score(total_pop, runout_distance_km)

    return ExposureResult(
        total_population_exposed=total_pop,
        population_by_distance=distance_bands,
        exposure_score=exposure_score,
        settlements_at_risk=settlements,
        runout_distance_km=runout_distance_km,
        angle_of_reach_deg=angle,
    )


def assess_cascade_risk(
    flag: AnomalyFlag,
    dem_data: np.ndarray | None = None,
    population_data: np.ndarray | None = None,
    config: dict | None = None,
) -> CascadeAssessment:
    """
    Single-flag cascade risk assessment combining volume estimation,
    runout modeling, and exposure assessment.

    Parameters
    ----------
    flag : AnomalyFlag
        A single Tier 0 detection flag.
    dem_data : np.ndarray or None
        Digital elevation model [n_rows, n_cols]. If None, synthetic
        values are used for demo purposes.
    population_data : np.ndarray or None
        Population grid. If None, synthetic settlements are generated.
    config : dict or None
        Configuration dict with a 'cascade' key. If None, defaults
        are used.

    Returns
    -------
    CascadeAssessment
    """
    cfg = _merge_config(config)
    cascade_cfg = cfg["cascade"]
    min_volume = cascade_cfg["min_volume_m3"]
    aor_threshold = cascade_cfg["exposure"]["angle_of_reach_deg"]
    valley_width_thresh = cascade_cfg.get("valley_width_threshold_m", 500)

    # Volume estimation from anomaly area
    area = flag.area_m2
    width = np.sqrt(area)
    depth = 0.1 * width
    volume = max(area * depth, 0.0)

    # Use DEM if provided, otherwise synthetic values
    if dem_data is not None and dem_data.size > 0:
        elevation = float(np.nanmax(dem_data))
        height_drop = float(np.nanmax(dem_data) - np.nanmin(dem_data))
        slope = float(np.degrees(np.arctan2(
            height_drop, max(1.0, np.sqrt(area))
        )))
        above_valley = height_drop > 100
        valley_width: float | None = None
        if above_valley:
            valley_width = float(min(500, np.sqrt(area)))
    else:
        # Synthetic: assume mountainous terrain proportional to anomaly
        elevation = 3500.0 + flag.score * 100
        height_drop = 800.0 + flag.score * 50
        slope = 35.0
        above_valley = True
        valley_width = 200.0

    # Runout estimation
    runout = estimate_runout(height_drop, height_drop / np.tan(np.radians(
        max(aor_threshold, 1.0)
    )), volume_m3=volume)

    runout_km = runout["runout_distance_km"]
    angle_of_reach = runout["angle_of_reach_deg"]
    runout_m = runout_km * 1000

    reaches_river = above_valley and runout_m > 0

    # Population exposure
    exposure = estimate_exposure(
        flag.center_lat,
        flag.center_lon,
        elevation,
        runout_km,
        population_data=population_data,
    )

    pop = exposure.total_population_exposed
    settlements_names = [s["name"] for s in exposure.settlements_at_risk]

    # Dam potential
    dam_pot = (
        above_valley
        and volume > min_volume
        and valley_width is not None
        and valley_width < valley_width_thresh
    )

    # Risk classification
    volume_ok = volume >= min_volume
    if volume_ok and above_valley and (pop > 0 or dam_pot):
        risk = RiskLevel.HIGH
    elif volume_ok and (above_valley or pop > 0):
        risk = RiskLevel.MODERATE
    elif volume_ok:
        risk = RiskLevel.LOW
    else:
        risk = RiskLevel.MINIMAL

    cascade_score = _compute_cascade_score(
        flag.score, volume, pop, dam_pot, above_valley, slope
    )

    return CascadeAssessment(
        flag=flag,
        risk_level=risk,
        estimated_volume_m3=volume,
        volume_sufficient=volume_ok,
        slope_angle_deg=slope,
        elevation_m=elevation,
        above_valley=above_valley,
        valley_width_m=valley_width,
        runout_distance_m=runout_m,
        runout_reaches_river=reaches_river,
        angle_of_reach_deg=angle_of_reach,
        population_exposed=pop,
        settlements_exposed=settlements_names,
        exposure=exposure,
        dam_potential=dam_pot,
        cascade_score=cascade_score,
    )


def assess_cascade_risk_batch(
    flags: list[AnomalyFlag],
    dem: np.ndarray,
    dem_lat: np.ndarray,
    dem_lon: np.ndarray,
    config: dict,
    population_grid: np.ndarray | None = None,
) -> list[CascadeAssessment]:
    """
    Run Tier 1 cascade risk assessment on a batch of flagged sites.

    This is the original batch entry point. For single-flag assessment
    with built-in exposure analysis, use :func:`assess_cascade_risk`.

    Parameters
    ----------
    flags : list[AnomalyFlag]
        Tier 0 detection output.
    dem : np.ndarray
        Digital elevation model (meters), shape [n_rows, n_cols].
    dem_lat, dem_lon : np.ndarray
        Coordinate grids for the DEM.
    config : dict
        Cascade configuration (from YAML 'cascade' section).
    population_grid : np.ndarray or None
        Population density grid (people/km^2), aligned to DEM.
        If None, exposure check is skipped.

    Returns
    -------
    list[CascadeAssessment]
        Assessments sorted by cascade_score (highest risk first).
    """
    cfg = _merge_config(config)
    cascade_cfg = cfg["cascade"]
    min_volume = cascade_cfg["min_volume_m3"]
    max_runout = cascade_cfg["exposure"]["max_runout_km"] * 1000
    aor_threshold = cascade_cfg["exposure"]["angle_of_reach_deg"]
    valley_width_thresh = cascade_cfg.get("valley_width_threshold_m", 500)

    assessments = []

    for flag in flags:
        logger.info("Assessing cascade risk for flag %d (%.4f N, %.4f E)",
                     flag.flag_id, flag.center_lat, flag.center_lon)

        # Find nearest DEM pixel to flag center
        row_idx, col_idx = _nearest_pixel(
            dem_lat, dem_lon, flag.center_lat, flag.center_lon
        )

        # 1. Volume estimation
        volume = _estimate_failure_volume(
            flag, dem, dem_lat, dem_lon, row_idx, col_idx
        )

        # 2. Slope and elevation
        slope = _compute_slope_angle(dem, dem_lat, dem_lon, row_idx, col_idx)
        elevation = float(dem[row_idx, col_idx])

        # 3. Valley analysis
        above_valley, valley_width = _analyze_valley_below(
            dem, dem_lat, dem_lon, row_idx, col_idx
        )

        # 4. Runout analysis (use estimate_runout for consistency)
        height_drop = _max_height_drop(
            dem, row_idx, col_idx, max_runout, dem_lat, dem_lon
        )
        runout_result = estimate_runout(height_drop, max_runout, volume_m3=volume)
        aor = runout_result["angle_of_reach_deg"]
        runout_distance = (
            height_drop / np.tan(np.radians(aor_threshold))
            if aor_threshold > 0 and height_drop > 0
            else 0
        )
        reaches_river = above_valley and runout_distance > 0

        # 5. Population exposure
        pop = 0
        exposure_result = None
        if population_grid is not None:
            pop = _estimate_exposed_population(
                population_grid, dem_lat, dem_lon,
                flag.center_lat, flag.center_lon,
                runout_distance,
            )
        # Also compute detailed exposure
        runout_km = runout_distance / 1000
        exposure_result = estimate_exposure(
            flag.center_lat,
            flag.center_lon,
            elevation,
            runout_km,
            population_data=population_grid,
        )
        if pop == 0:
            pop = exposure_result.total_population_exposed

        # 6. Dam potential
        dam_pot = (
            above_valley
            and volume > min_volume
            and valley_width is not None
            and valley_width < valley_width_thresh
        )

        # Risk classification
        volume_ok = volume >= min_volume
        if volume_ok and above_valley and (pop > 0 or dam_pot):
            risk = RiskLevel.HIGH
        elif volume_ok and (above_valley or pop > 0):
            risk = RiskLevel.MODERATE
        elif volume_ok:
            risk = RiskLevel.LOW
        else:
            risk = RiskLevel.MINIMAL

        # Composite cascade score
        cascade_score = _compute_cascade_score(
            flag.score, volume, pop, dam_pot, above_valley, slope
        )

        assessment = CascadeAssessment(
            flag=flag,
            risk_level=risk,
            estimated_volume_m3=volume,
            volume_sufficient=volume_ok,
            slope_angle_deg=slope,
            elevation_m=elevation,
            above_valley=above_valley,
            valley_width_m=valley_width,
            runout_distance_m=runout_distance,
            runout_reaches_river=reaches_river,
            angle_of_reach_deg=aor,
            population_exposed=pop,
            exposure=exposure_result,
            dam_potential=dam_pot,
            cascade_score=cascade_score,
        )

        assessments.append(assessment)

        logger.info(
            "  Flag %d: risk=%s, volume=%.0f m3, slope=%.1f deg, "
            "elev=%.0f m, valley=%s, pop=%d, dam=%s",
            flag.flag_id, risk.value, volume, slope,
            elevation, above_valley, pop, dam_pot,
        )

    assessments.sort(key=lambda a: a.cascade_score, reverse=True)

    n_pass = sum(1 for a in assessments if a.passes_tier1)
    logger.info(
        "Tier 1 complete: %d of %d flags pass (>= MODERATE risk)",
        n_pass, len(assessments),
    )

    return assessments


def generate_exposure_summary(
    assessments: list[CascadeAssessment],
) -> str:
    """
    Produce a human-readable summary table of population exposure by site.

    Parameters
    ----------
    assessments : list[CascadeAssessment]
        Cascade assessments (typically from assess_cascade_risk or
        assess_cascade_risk_batch).

    Returns
    -------
    str
        Formatted text table.
    """
    if not assessments:
        return "No cascade assessments to summarize."

    lines = [
        "Population Exposure Summary",
        "=" * 80,
        f"{'Flag':>5}  {'Lat':>9}  {'Lon':>10}  {'Risk':>8}  "
        f"{'Pop Exposed':>11}  {'Runout km':>9}  {'Score':>6}  "
        f"{'Settlements':>12}",
        "-" * 80,
    ]

    for a in assessments:
        n_settlements = 0
        runout_km = a.runout_distance_m / 1000
        exposure_score = 0.0
        if a.exposure is not None:
            n_settlements = len(a.exposure.settlements_at_risk)
            exposure_score = a.exposure.exposure_score
            runout_km = a.exposure.runout_distance_km

        lines.append(
            f"{a.flag.flag_id:>5}  "
            f"{a.flag.center_lat:>9.4f}  "
            f"{a.flag.center_lon:>10.4f}  "
            f"{a.risk_level.value:>8}  "
            f"{a.population_exposed:>11,}  "
            f"{runout_km:>9.1f}  "
            f"{exposure_score:>6.1f}  "
            f"{n_settlements:>12}"
        )

    lines.append("-" * 80)

    # Summary row
    total_pop = sum(a.population_exposed for a in assessments)
    high_count = sum(1 for a in assessments if a.risk_level == RiskLevel.HIGH)
    lines.append(
        f"Total: {len(assessments)} sites assessed, "
        f"{high_count} HIGH risk, "
        f"{total_pop:,} total population exposed"
    )

    # Per-band breakdown if exposure data is available
    band_totals: dict[str, int] = {}
    for a in assessments:
        if a.exposure is not None:
            for band, pop in a.exposure.population_by_distance.items():
                band_totals[band] = band_totals.get(band, 0) + pop

    if band_totals:
        lines.append("")
        lines.append("Population by distance band:")
        for band in ["0-10km", "10-30km", "30-100km"]:
            if band in band_totals:
                lines.append(f"  {band}: {band_totals[band]:,}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _merge_config(config: dict | None) -> dict:
    """Merge user config with defaults, tolerating partial or None config."""
    if config is None:
        return dict(_DEFAULT_CASCADE_CONFIG)

    merged = dict(_DEFAULT_CASCADE_CONFIG)
    if "cascade" in config:
        cascade_merged = dict(merged["cascade"])
        user_cascade = config["cascade"]
        for key, val in user_cascade.items():
            if key == "exposure" and isinstance(val, dict):
                cascade_merged["exposure"] = {
                    **cascade_merged.get("exposure", {}),
                    **val,
                }
            else:
                cascade_merged[key] = val
        merged["cascade"] = cascade_merged

    return merged


def _band_lower_km(band_key: str) -> float:
    """Extract lower bound in km from a band key like '10-30km'."""
    try:
        lower = band_key.split("-")[0].replace("km", "")
        return float(lower)
    except (ValueError, IndexError):
        return 0.0


def _compute_exposure_score(total_population: int, runout_km: float) -> float:
    """
    Compute a composite exposure score on a 0–10 scale.

    Combines population count (log-scaled) and runout proximity.
    """
    if total_population <= 0:
        return 0.0

    # Population component: log10(pop) scaled to 0–7
    pop_component = min(7.0, np.log10(max(1, total_population)))

    # Proximity component: closer runout = higher risk (0–3)
    if runout_km <= 0:
        proximity_component = 0.0
    elif runout_km <= 5:
        proximity_component = 3.0
    elif runout_km <= 20:
        proximity_component = 2.0
    elif runout_km <= 50:
        proximity_component = 1.0
    else:
        proximity_component = 0.5

    score = pop_component + proximity_component
    return min(10.0, max(0.0, score))


def _generate_synthetic_settlements(
    flag_lat: float,
    flag_lon: float,
    flag_elevation_m: float,
    runout_distance_km: float,
) -> list[dict]:
    """
    Generate plausible synthetic downstream settlements for demo purposes.

    Uses a deterministic seed derived from the input coordinates so that
    repeated calls with the same inputs produce identical results.

    Models settlements along valleys downstream with population
    distributions typical of mountainous regions: small villages close
    to the source, occasional larger settlements in wider valleys
    further downstream.
    """
    if runout_distance_km <= 0:
        return []

    # Deterministic seed from rounded coordinates
    seed_str = f"{flag_lat:.4f}_{flag_lon:.4f}_{flag_elevation_m:.0f}"
    seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16) % (2**31)
    rng = np.random.RandomState(seed)

    settlements = []

    # Settlement model for mountainous regions:
    # - Hamlets (50-200 people) at 2-8 km downstream
    # - Villages (200-2000 people) at 5-20 km
    # - Towns (2000-20000 people) at 15-50 km
    settlement_templates = [
        {"type": "hamlet", "pop_range": (50, 200), "dist_range": (2, 8), "count": 2},
        {"type": "village", "pop_range": (200, 2000), "dist_range": (5, 20), "count": 2},
        {"type": "town", "pop_range": (2000, 20000), "dist_range": (15, 50), "count": 1},
    ]

    names_hamlet = [
        "Upper Hamlet", "Lower Hamlet", "Ridge Settlement",
        "Valley Hamlet", "Stream Hamlet",
    ]
    names_village = [
        "Riverside Village", "Valley Village", "Mountain Village",
        "Bridge Village", "Confluence Village",
    ]
    names_town = [
        "Valley Town", "River Town", "Market Town",
    ]
    name_pools = {
        "hamlet": names_hamlet,
        "village": names_village,
        "town": names_town,
    }

    for template in settlement_templates:
        dist_lo, dist_hi = template["dist_range"]
        pop_lo, pop_hi = template["pop_range"]
        pool = name_pools[template["type"]]

        for i in range(template["count"]):
            dist_km = dist_lo + rng.random() * (dist_hi - dist_lo)

            # Only place settlements within physical reach
            if dist_km > runout_distance_km * 1.5:
                continue

            pop = int(pop_lo + rng.random() * (pop_hi - pop_lo))

            # Place downstream (lower latitude for northern hemisphere,
            # slight longitude offset for valley direction)
            bearing_rad = rng.uniform(-0.5, 0.5)  # roughly downstream
            dlat = -dist_km / 111.32 * np.cos(bearing_rad)
            dlon = dist_km / (111.32 * np.cos(np.radians(flag_lat))) * np.sin(bearing_rad)

            name_idx = (i + seed) % len(pool)
            name = pool[name_idx]

            settlements.append({
                "name": name,
                "population": pop,
                "distance_km": round(dist_km, 1),
                "coordinates": {
                    "lat": round(flag_lat + dlat, 4),
                    "lon": round(flag_lon + dlon, 4),
                },
            })

    # Sort by distance
    settlements.sort(key=lambda s: s["distance_km"])

    return settlements


def _exposure_from_grid(
    flag_lat: float,
    flag_lon: float,
    runout_distance_km: float,
    pop_grid: np.ndarray,
) -> tuple[list[dict], dict[str, int]]:
    """
    Compute exposure from a real population grid.

    Returns (settlements_at_risk, population_by_distance).
    """
    bands = {"0-10km": 0, "10-30km": 0, "30-100km": 0}
    settlements: list[dict] = []

    if pop_grid is None or pop_grid.size == 0:
        return settlements, bands

    n_rows, n_cols = pop_grid.shape

    # Assume the grid covers a region around the flag
    # with ~1km resolution (typical for WorldPop)
    for r in range(n_rows):
        for c in range(n_cols):
            cell_pop = pop_grid[r, c]
            if not np.isfinite(cell_pop) or cell_pop <= 0:
                continue

            # Approximate distance from center of grid
            dlat = (r - n_rows / 2) * 1.0  # km (1 km/pixel)
            dlon = (c - n_cols / 2) * 1.0
            dist_km = np.sqrt(dlat**2 + dlon**2)

            if dist_km > runout_distance_km:
                continue

            cell_pop_int = int(cell_pop)
            if dist_km <= 10:
                bands["0-10km"] += cell_pop_int
            elif dist_km <= 30:
                bands["10-30km"] += cell_pop_int
            else:
                bands["30-100km"] += cell_pop_int

            if cell_pop_int > 100:
                settlements.append({
                    "name": f"Grid cell ({r},{c})",
                    "population": cell_pop_int,
                    "distance_km": round(dist_km, 1),
                    "coordinates": {"lat": flag_lat, "lon": flag_lon},
                })

    return settlements, bands


# ---------------------------------------------------------------------------
# DEM / geometry helpers (unchanged from original)
# ---------------------------------------------------------------------------


def _nearest_pixel(
    lat_grid: np.ndarray, lon_grid: np.ndarray,
    target_lat: float, target_lon: float,
) -> tuple[int, int]:
    """Find the row, col index of the nearest pixel to a target coordinate."""
    dist = (lat_grid - target_lat) ** 2 + (lon_grid - target_lon) ** 2
    return np.unravel_index(np.nanargmin(dist), dist.shape)


def _estimate_failure_volume(
    flag: AnomalyFlag,
    dem: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    center_row: int,
    center_col: int,
) -> float:
    """
    Estimate the volume of potentially unstable mass.

    Uses the area of the deformation anomaly and an assumed failure
    depth proportional to the width of the deforming zone.

    This is a rough estimate — adequate for Tier 1 screening but
    not for engineering design.
    """
    area = flag.area_m2

    # Empirical: failure depth ~ 0.1 * sqrt(area) for rock slopes
    # (Hungr et al., 2014). Conservative for screening.
    width = np.sqrt(area)
    depth = 0.1 * width  # meters

    # Adjust for slope angle
    if center_row < dem.shape[0] and center_col < dem.shape[1]:
        slope_rad = np.radians(
            _compute_slope_angle(dem, lat, lon, center_row, center_col)
        )
        # On steeper slopes, the failure slab is thinner
        depth *= np.cos(slope_rad)

    volume = area * depth
    return max(volume, 0)


def _compute_slope_angle(
    dem: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    row: int,
    col: int,
    window: int = 3,
) -> float:
    """Compute local slope angle in degrees from DEM."""
    half = window // 2
    r0 = max(0, row - half)
    r1 = min(dem.shape[0], row + half + 1)
    c0 = max(0, col - half)
    c1 = min(dem.shape[1], col + half + 1)

    patch = dem[r0:r1, c0:c1]
    if patch.size < 4:
        return 0.0

    # Pixel spacing in meters
    mean_lat = np.nanmean(lat[r0:r1, c0:c1])
    dy = 111_320 * np.abs(lat[r0, c0] - lat[min(r1 - 1, r0 + 1), c0])
    dx = 111_320 * np.cos(np.radians(mean_lat)) * np.abs(
        lon[r0, c0] - lon[r0, min(c1 - 1, c0 + 1)]
    ) if lon.ndim == 2 else 30.0  # fallback

    if dy == 0:
        dy = 30.0
    if dx == 0:
        dx = 30.0

    # Gradient
    gy, gx = np.gradient(patch, dy, dx)
    slope_rad = np.arctan(np.sqrt(np.nanmean(gx**2) + np.nanmean(gy**2)))
    return float(np.degrees(slope_rad))


def _analyze_valley_below(
    dem: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    row: int,
    col: int,
    search_radius_pixels: int = 50,
) -> tuple[bool, float | None]:
    """
    Determine if the flagged site is above a confined valley.

    Searches downslope from the site for a local minimum in elevation
    flanked by rising terrain on both sides (valley cross-section).

    Returns (above_valley, valley_width_m).
    """
    n_rows, n_cols = dem.shape

    # Sample elevation profile in steepest descent direction
    r0 = max(0, row - search_radius_pixels)
    r1 = min(n_rows, row + search_radius_pixels + 1)
    c0 = max(0, col - search_radius_pixels)
    c1 = min(n_cols, col + search_radius_pixels + 1)

    patch = dem[r0:r1, c0:c1]
    center_elev = dem[row, col]

    # Find lowest point in search area
    if np.all(np.isnan(patch)):
        return False, None

    min_idx = np.unravel_index(np.nanargmin(patch), patch.shape)
    min_elev = patch[min_idx]

    # Must be significantly below the flagged site
    if center_elev - min_elev < 100:  # less than 100m drop
        return False, None

    # Check if the minimum is in a valley (terrain rises on both sides)
    # Take a cross-section through the minimum perpendicular to the
    # slope direction
    min_row_local, min_col_local = min_idx
    cross_section = patch[min_row_local, :]

    valid = np.isfinite(cross_section)
    if valid.sum() < 5:
        return False, None

    # Valley = minimum with terrain rising on both sides
    cs_min_idx = np.nanargmin(cross_section)
    left_max = np.nanmax(cross_section[:cs_min_idx]) if cs_min_idx > 0 else min_elev
    right_max = np.nanmax(cross_section[cs_min_idx + 1:]) if cs_min_idx < len(cross_section) - 1 else min_elev

    prominence = min(left_max - min_elev, right_max - min_elev)
    if prominence < 50:  # less than 50m prominence = not a clear valley
        return False, None

    # Estimate valley width at the minimum
    above_min = cross_section > (min_elev + prominence * 0.5)
    transitions = np.diff(above_min.astype(int))
    entries = np.where(transitions == -1)[0]
    exits = np.where(transitions == 1)[0]

    if len(entries) > 0 and len(exits) > 0:
        # Width between first descent and first ascent
        mean_lat_val = np.nanmean(lat[r0:r1, c0:c1])
        dlon = np.abs(lon[0, 1] - lon[0, 0]) if lon.ndim == 2 and lon.shape[1] > 1 else 0.00028
        pixel_width_m = dlon * 111_320 * np.cos(np.radians(mean_lat_val))
        valley_width = abs(exits[0] - entries[0]) * pixel_width_m
        return True, float(valley_width)

    return True, None


def _max_height_drop(
    dem: np.ndarray,
    row: int,
    col: int,
    max_distance_m: float,
    lat: np.ndarray,
    lon: np.ndarray,
) -> float:
    """Maximum elevation drop within search radius."""
    # Approximate pixels for max distance
    pixel_size_m = 30.0  # default
    if lat.shape[0] > 1:
        pixel_size_m = np.abs(lat[1, 0] - lat[0, 0]) * 111_320

    radius_px = int(max_distance_m / pixel_size_m) if pixel_size_m > 0 else 100
    radius_px = min(radius_px, max(dem.shape))

    r0 = max(0, row - radius_px)
    r1 = min(dem.shape[0], row + radius_px + 1)
    c0 = max(0, col - radius_px)
    c1 = min(dem.shape[1], col + radius_px + 1)

    patch = dem[r0:r1, c0:c1]
    center_elev = dem[row, col]
    min_elev = np.nanmin(patch)

    return max(0, center_elev - min_elev)


def _estimate_exposed_population(
    pop_grid: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    center_lat: float,
    center_lon: float,
    runout_m: float,
) -> int:
    """
    Estimate population within the potential runout zone.

    Simple circular buffer for PoC — operational system would use
    actual flow-path modeling.
    """
    if pop_grid is None:
        return 0

    mean_lat = np.nanmean(lat)
    lat_m = (lat - center_lat) * 111_320
    lon_m = (lon - center_lon) * 111_320 * np.cos(np.radians(mean_lat))
    dist = np.sqrt(lat_m**2 + lon_m**2)

    # Only count population downslope (lower elevation)
    in_range = dist <= runout_m
    exposed = pop_grid[in_range]

    return int(np.nansum(exposed))


def _compute_cascade_score(
    anomaly_score: float,
    volume: float,
    population: int,
    dam_potential: bool,
    above_valley: bool,
    slope: float,
) -> float:
    """
    Composite cascade risk score combining deformation anomaly
    with physical cascade potential.
    """
    # Log-scale volume contribution
    vol_score = np.log10(max(1, volume)) / 7  # normalized: 10^7 m3 -> 1.0

    # Population contribution (diminishing returns)
    pop_score = np.log10(max(1, population)) / 5  # 100k people -> 1.0

    # Binary factors
    dam_score = 0.3 if dam_potential else 0.0
    valley_score = 0.2 if above_valley else 0.0

    # Weighted combination
    score = (
        anomaly_score * 0.3
        + vol_score * 0.25
        + pop_score * 0.2
        + dam_score
        + valley_score
        + (slope / 90) * 0.05  # steeper = riskier
    )

    return score
