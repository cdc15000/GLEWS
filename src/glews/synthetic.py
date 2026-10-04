"""
Synthetic data generator for testing and demonstration.

Generates realistic InSAR displacement time-series data with:
    - Steady-state glacier motion at configurable velocity
    - Seasonal (annual + semi-annual) variation
    - Atmospheric noise
    - One or more injected pre-failure acceleration signals

This allows the full pipeline to be tested and demonstrated
without requiring actual Sentinel-1 data or InSAR processing.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from glews.process import DisplacementTimeseries

logger = logging.getLogger(__name__)


@dataclass
class SyntheticConfig:
    """Configuration for synthetic scene generation."""

    # Spatial dimensions
    n_rows: int = 200
    n_cols: int = 250

    # Coordinate bounds (centered on approximate Nepal site)
    lat_min: float = 28.10
    lat_max: float = 28.30
    lon_min: float = 85.80
    lon_max: float = 86.00

    # Temporal parameters
    start_date: str = "2025-01-08"
    end_date: str = "2026-08-18"
    revisit_days: int = 12  # Sentinel-1 revisit

    # Background glacier motion
    base_velocity_m_yr: float = 0.05  # 50 mm/yr LOS, typical for slow glaciers
    velocity_spatial_variation: float = 0.3  # fraction of spatial variation

    # Seasonal signal
    annual_amplitude_m: float = 0.005  # 5 mm annual cycle
    semi_annual_amplitude_m: float = 0.002

    # Noise
    atmospheric_noise_m: float = 0.004  # 4 mm atmospheric noise per scene
    measurement_noise_m: float = 0.001  # 1 mm instrument noise

    # Coherence
    base_coherence: float = 0.85
    low_coherence_fraction: float = 0.1  # fraction of pixels with poor coherence

    # Injected failure signal
    failure_zones: list[dict] | None = None
    # Each zone: {center_row, center_col, radius_pixels, onset_days_before_end,
    #             max_acceleration_m_yr2, ramp_type: "exponential"|"linear"}


def generate_synthetic_scene(
    config: SyntheticConfig | None = None,
) -> DisplacementTimeseries:
    """
    Generate a complete synthetic displacement time-series dataset.

    Parameters
    ----------
    config : SyntheticConfig or None
        Generation parameters. If None, uses defaults with one
        pre-failure zone mimicking the Nepal 2026 scenario.

    Returns
    -------
    DisplacementTimeseries
        Synthetic data in the same format as real InSAR output.
    """
    if config is None:
        config = _default_nepal_config()

    rng = np.random.default_rng(42)

    # Generate date grid
    start = datetime.strptime(config.start_date, "%Y-%m-%d")
    end = datetime.strptime(config.end_date, "%Y-%m-%d")
    n_days = (end - start).days
    date_list = []
    d = start
    while d <= end:
        date_list.append(d)
        d += timedelta(days=config.revisit_days)

    n_dates = len(date_list)
    date_strings = [d.strftime("%Y%m%d") for d in date_list]
    dates = np.array([d.toordinal() for d in date_list])

    logger.info(
        "Generating synthetic scene: %d dates × %d×%d pixels, %s to %s",
        n_dates, config.n_rows, config.n_cols,
        date_strings[0], date_strings[-1],
    )

    n_rows = config.n_rows
    n_cols = config.n_cols

    # Coordinate grids
    latitude = np.linspace(config.lat_max, config.lat_min, n_rows)[:, None] * np.ones(n_cols)
    longitude = np.ones(n_rows)[:, None] * np.linspace(config.lon_min, config.lon_max, n_cols)

    # Time axis (years from start)
    t0 = dates[0]
    t_years = (dates - t0).astype(float) / 365.25
    t_days = (dates - t0).astype(float)

    # Spatial velocity field — smooth random variation
    velocity_field = _generate_velocity_field(
        n_rows, n_cols, config.base_velocity_m_yr,
        config.velocity_spatial_variation, rng,
    )

    # Build displacement cube
    displacement = np.zeros((n_dates, n_rows, n_cols))

    for i in range(n_dates):
        # Linear trend
        displacement[i] = velocity_field * t_years[i]

        # Seasonal signal
        omega1 = 2 * np.pi / 365.25
        omega2 = 2 * omega1
        seasonal = (
            config.annual_amplitude_m * np.sin(omega1 * t_days[i])
            + config.semi_annual_amplitude_m * np.sin(omega2 * t_days[i] + 0.5)
        )
        displacement[i] += seasonal

        # Atmospheric noise (spatially correlated)
        atm = _generate_atmospheric_noise(
            n_rows, n_cols, config.atmospheric_noise_m, rng
        )
        displacement[i] += atm

        # Measurement noise
        displacement[i] += rng.normal(0, config.measurement_noise_m, (n_rows, n_cols))

    # Inject failure signals
    failure_zones = config.failure_zones or []
    for zone in failure_zones:
        displacement = _inject_failure_signal(displacement, dates, zone, rng)
        logger.info(
            "  Injected failure zone at row=%d, col=%d, "
            "onset=%d days before end, max_accel=%.3f m/yr²",
            zone["center_row"], zone["center_col"],
            zone["onset_days_before_end"],
            zone["max_acceleration_m_yr2"],
        )

    # Temporal coherence
    coherence = _generate_coherence_field(
        n_rows, n_cols, config.base_coherence,
        config.low_coherence_fraction, rng,
    )

    # Mask low-coherence pixels
    mask = coherence < 0.5
    displacement[:, mask] = np.nan

    # Velocity from linear fit
    velocity = np.full((n_rows, n_cols), np.nan)
    valid = ~mask
    if np.any(valid):
        velocity[valid] = velocity_field[valid]

    logger.info(
        "Synthetic scene complete: %d valid pixels (%.1f%%)",
        np.count_nonzero(valid),
        100 * np.count_nonzero(valid) / valid.size,
    )

    return DisplacementTimeseries(
        dates=dates,
        date_strings=date_strings,
        displacement=displacement,
        velocity=velocity,
        temporal_coherence=coherence,
        latitude=latitude,
        longitude=longitude,
        metadata={
            "source": "synthetic",
            "config": {
                "n_failure_zones": len(failure_zones),
                "base_velocity_m_yr": config.base_velocity_m_yr,
                "atmospheric_noise_m": config.atmospheric_noise_m,
            },
            "wavelength_m": 0.05546,
        },
    )


def _default_nepal_config() -> SyntheticConfig:
    """
    Default configuration mimicking the Nepal 2026 scenario.

    Injects one pre-failure acceleration zone with parameters
    inspired by Shirzaei's observations: ~10 mm/month velocity
    with acceleration detectable in the final weeks.
    """
    return SyntheticConfig(
        failure_zones=[
            {
                "center_row": 80,
                "center_col": 120,
                "radius_pixels": 12,
                "onset_days_before_end": 90,  # acceleration begins ~3 months before
                "max_acceleration_m_yr2": 0.5,
                "ramp_type": "voight",
            },
            # Second zone: a glacier that moves but doesn't fail (false positive test)
            {
                "center_row": 140,
                "center_col": 180,
                "radius_pixels": 8,
                "onset_days_before_end": 180,
                "max_acceleration_m_yr2": 0.03,  # mild, non-dangerous
                "ramp_type": "linear",
            },
        ],
    )


def _generate_velocity_field(
    n_rows: int, n_cols: int,
    base_vel: float,
    variation: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a spatially smooth velocity field."""
    # Low-frequency random field
    from scipy.ndimage import gaussian_filter

    raw = rng.normal(0, 1, (n_rows, n_cols))
    smooth = gaussian_filter(raw, sigma=20)
    smooth = smooth / np.std(smooth)  # normalize

    velocity = base_vel * (1 + variation * smooth)
    return velocity


def _generate_atmospheric_noise(
    n_rows: int, n_cols: int,
    amplitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate spatially correlated atmospheric noise.

    Atmospheric phase screens in mountain regions have long-wavelength
    structure correlated with topography. We simulate this with a
    smooth random field.
    """
    from scipy.ndimage import gaussian_filter

    raw = rng.normal(0, 1, (n_rows, n_cols))
    smooth = gaussian_filter(raw, sigma=30)
    smooth = smooth / np.std(smooth) * amplitude
    return smooth


def _generate_coherence_field(
    n_rows: int, n_cols: int,
    base_coherence: float,
    low_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a realistic coherence field."""
    from scipy.ndimage import gaussian_filter

    coherence = np.full((n_rows, n_cols), base_coherence)

    # Add spatial variation
    noise = gaussian_filter(rng.normal(0, 0.1, (n_rows, n_cols)), sigma=10)
    coherence += noise

    # Low-coherence patches (snow, water, vegetation)
    n_patches = int(low_fraction * n_rows * n_cols / 100)
    for _ in range(n_patches):
        r = rng.integers(0, n_rows)
        c = rng.integers(0, n_cols)
        radius = rng.integers(3, 15)
        yy, xx = np.ogrid[-r : n_rows - r, -c : n_cols - c]
        mask = xx**2 + yy**2 <= radius**2
        coherence[mask] *= rng.uniform(0.2, 0.5)

    return np.clip(coherence, 0, 1)


def _inject_failure_signal(
    displacement: np.ndarray,
    dates: np.ndarray,
    zone: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Inject a pre-failure acceleration signal into the displacement field.

    Models the physics: before catastrophic failure, the unstable mass
    accelerates non-linearly. The signal is strongest at the center
    and tapers with distance.
    """
    n_dates, n_rows, n_cols = displacement.shape
    cr = zone["center_row"]
    cc = zone["center_col"]
    radius = zone["radius_pixels"]
    onset = zone["onset_days_before_end"]
    max_accel = zone["max_acceleration_m_yr2"]
    ramp_type = zone.get("ramp_type", "exponential")

    # Spatial taper: Gaussian falloff from center
    yy, xx = np.ogrid[:n_rows, :n_cols]
    dist = np.sqrt((yy - cr) ** 2 + (xx - cc) ** 2)
    spatial = np.exp(-0.5 * (dist / radius) ** 2)
    spatial[dist > 3 * radius] = 0

    # Temporal ramp: begins at onset, intensifies toward end
    end_date = dates[-1]
    onset_date = end_date - onset

    for i in range(n_dates):
        t = dates[i]
        if t < onset_date:
            continue

        # Progress: 0 at onset, 1 at end
        progress = (t - onset_date) / max(1, end_date - onset_date)
        progress = min(progress, 1.0)

        if ramp_type == "voight":
            tf = onset * 1.1
            t_since = progress * onset
            t_since = min(t_since, tf * 0.99)
            voight_disp = math.log(tf / (tf - t_since))
            voight_max = math.log(tf / (tf - onset))
            temporal = voight_disp / voight_max if voight_max > 0 else progress
        elif ramp_type == "exponential":
            # Exponential ramp — slow start, rapid increase
            temporal = (np.exp(3 * progress) - 1) / (np.e**3 - 1)
        else:
            temporal = progress

        t_since_onset_yr = (t - onset_date) / 365.25
        if ramp_type == "voight":
            accel_disp = max_accel * (onset / 365.25) * temporal
        else:
            accel_disp = 0.5 * max_accel * temporal * t_since_onset_yr**2

        # Add to displacement (negative = downslope motion away from satellite)
        displacement[i] -= accel_disp * spatial

        # Add some noise to the signal (real failures are messy)
        noise_scale = 0.1 * accel_disp
        if noise_scale > 0:
            displacement[i] -= noise_scale * spatial * rng.normal(0, 1, (n_rows, n_cols))

    return displacement


def generate_demo_instruments(flags: list, output_dir: str) -> None:
    """Generate a demo instruments.json with synthetic sensor data near flags."""
    import json
    from pathlib import Path

    rng = np.random.default_rng(42)
    instruments = []
    inst_id = 1

    voight_flags = [f for f in flags if hasattr(f, "voight_fit") and f.voight_fit]
    if not voight_flags:
        voight_flags = flags[:3] if len(flags) >= 3 else flags

    types = [
        ("gnss", "mm", 0.5, 50),
        ("crackmeter", "mm", 0.0, 10),
        ("tiltmeter", "degrees", 0.0, 2),
    ]

    for flag in voight_flags[:3]:
        lat, lon = flag.center_lat, flag.center_lon
        for itype, unit, base_val, scale in types:
            offset_lat = rng.uniform(-0.02, 0.02)
            offset_lon = rng.uniform(-0.02, 0.02)
            readings = []
            val = base_val + rng.uniform(0, scale * 0.1)
            for day in range(0, 90, 3):
                dt = (datetime.now() - timedelta(days=90 - day)).strftime("%Y-%m-%d")
                if itype == "gnss":
                    val += rng.uniform(0.1, 0.8)
                elif itype == "crackmeter":
                    val += rng.uniform(0.0, 0.3)
                else:
                    val += rng.uniform(0.0, 0.05)
                readings.append({
                    "date": dt,
                    "value": round(val, 3),
                    "unit": unit,
                })
            status = rng.choice(["active", "active", "active", "maintenance"], p=[0.7, 0.1, 0.1, 0.1])
            instruments.append({
                "id": f"INST-{inst_id:03d}",
                "type": itype,
                "lat": round(lat + offset_lat, 4),
                "lon": round(lon + offset_lon, 4),
                "status": status,
                "installed_date": (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d"),
                "readings": readings,
            })
            inst_id += 1

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "instruments.json").write_text(
        json.dumps({"instruments": instruments}, indent=2)
    )
    logger.info("Wrote %d instruments to %s", len(instruments), out / "instruments.json")


def generate_demo_field_reports(flags: list, output_dir: str) -> None:
    """Generate demo field_reports.json with synthetic reconnaissance data."""
    import json
    from pathlib import Path

    rng = np.random.default_rng(99)
    reports = []

    voight_flags = [f for f in flags if hasattr(f, "voight_fit") and f.voight_fit]
    if not voight_flags:
        voight_flags = flags[:2] if len(flags) >= 2 else flags

    authors = ["Dr. K. Tamang", "R. Sherpa", "L. Chen"]
    seepage_vals = ["none", "minor seepage", "active spring", "saturated ground"]
    vegetation_vals = ["none", "minor tilting", "fallen trees", "bare scarp"]
    cracking_vals = ["none observed", "hairline cracks", "tension cracks 2-5cm", "major fractures >10cm"]

    for i, flag in enumerate(voight_flags[:3]):
        lat, lon = flag.center_lat, flag.center_lon
        days_ago = rng.integers(3, 30)
        risk = rng.choice(["low", "moderate", "high", "critical"],
                          p=[0.1, 0.3, 0.4, 0.2])
        reports.append({
            "id": f"FR-{i + 1:03d}",
            "date": (datetime.now() - timedelta(days=int(days_ago))).strftime("%Y-%m-%d"),
            "author": authors[i % len(authors)],
            "lat": round(lat + rng.uniform(-0.005, 0.005), 4),
            "lon": round(lon + rng.uniform(-0.005, 0.005), 4),
            "observations": {
                "crack_width_cm": round(float(rng.uniform(0.5, 15)), 1),
                "scarp_height_m": round(float(rng.uniform(1, 25)), 1),
                "seepage": rng.choice(seepage_vals),
                "vegetation_disturbance": rng.choice(vegetation_vals),
                "ground_cracking": rng.choice(cracking_vals),
            },
            "risk_assessment": risk,
            "notes": [
                "Visible tension cracks trending NE-SW. Local residents report increased rockfall over past 2 weeks.",
                "Fresh scarp face with unstable blocks. Access road showing subsidence. Recommended immediate monitoring.",
                "Seasonal seepage higher than previous years. Ground deformation visible near ridge crest.",
            ][i % 3],
        })

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "field_reports.json").write_text(
        json.dumps({"reports": reports}, indent=2)
    )
    logger.info("Wrote %d field reports to %s", len(reports), out / "field_reports.json")
