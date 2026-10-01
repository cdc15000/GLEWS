"""Shared pytest fixtures for GLEWS tests.

All fixtures produce purely synthetic data — no real HDF5/NISAR products
or network access are required to run the test suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from glews.nisar import NISARTimeseries


@pytest.fixture
def rng():
    """Deterministic random generator for reproducible synthetic data."""
    return np.random.default_rng(42)


@pytest.fixture
def small_dates():
    """A short run of evenly spaced ordinal dates (12-day revisit)."""
    return np.arange(738886, 738886 + 40 * 12, 12)


@pytest.fixture
def small_displacement(small_dates, rng):
    """A small [n_dates, n_rows, n_cols] displacement cube with a
    constant velocity plus a little noise — no injected anomaly."""
    n_dates = len(small_dates)
    n_rows, n_cols = 10, 12
    t_yr = (small_dates - small_dates[0]).astype(float) / 365.25

    velocity = 0.04  # m/yr
    displacement = np.zeros((n_dates, n_rows, n_cols), dtype=np.float64)
    for i in range(n_dates):
        displacement[i] = velocity * t_yr[i]
    displacement += rng.normal(0, 0.001, displacement.shape)

    return displacement


@pytest.fixture
def synthetic_nisar_timeseries(small_dates, small_displacement):
    """A minimal, valid NISARTimeseries built from synthetic arrays,
    exercising the same dataclass the loaders in nisar.py produce."""
    n_dates, n_rows, n_cols = small_displacement.shape

    lat = np.linspace(28.3, 28.1, n_rows)[:, None] * np.ones(n_cols)
    lon = np.ones(n_rows)[:, None] * np.linspace(85.8, 86.0, n_cols)

    from glews.nisar import _compute_velocity

    velocity = _compute_velocity(small_dates, small_displacement)
    coherence = np.full((n_rows, n_cols), 0.8, dtype=np.float32)

    date_strings = [
        f"2024{1 + (i // 28):02d}{1 + (i % 28):02d}" for i in range(n_dates)
    ]

    return NISARTimeseries(
        dates=small_dates,
        date_strings=date_strings,
        displacement=small_displacement,
        velocity=velocity,
        coherence=coherence,
        latitude=lat,
        longitude=lon,
        source="GUNW",
        metadata={"wavelength_m": 0.2384, "n_interferograms": n_dates - 1,
                  "sensor": "NISAR", "band": "L", "track": 98},
    )


@pytest.fixture
def site_config():
    """A minimal single-site config dict matching the shape expected by
    detect_anomalies / monitor / cli — no I/O required."""
    return {
        "site": {
            "name": "Test Site",
            "latitude": 28.2,
            "longitude": 85.9,
            "event_date": "2026-08-26",
        },
        "acquire": {
            "platform": "NISAR",
            "product_types": ["GUNW", "GOFF"],
            "gunw_dir": "data/nisar/gunw",
            "goff_dir": "data/nisar/goff",
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
        "monitor": {
            "interval_hours": 6,
            "lookback_days": 30,
        },
    }
