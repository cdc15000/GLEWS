"""Tests for radar/optical cross-sensor geometry."""

import numpy as np
import pytest

from glews.fusion import agreement, los_enu, project_to_radar, slope_parallel_from_radar

# NISAR track 98 over the Langtang Lirung source (left-looking, ascending)
LOS = los_enu(0.6128, 0.1519)
AT = (-0.2681, 0.9634)


def test_los_is_unit():
    assert np.linalg.norm(LOS) == pytest.approx(1.0)
    assert LOS[2] > 0


def test_motion_toward_sensor_shortens_range():
    rng, _ = project_to_radar(1.0, 0.0, 0.0, LOS, AT)   # east = toward this sensor
    assert rng < 0


def test_round_trip_slope_parallel():
    grad = (0.707, -0.081)                    # west-facing 35 deg slope
    e, n = -0.8, 0.1
    u = grad[0] * e + grad[1] * n
    rng, at = project_to_radar(e, n, u, LOS, AT)
    e2, n2, u2 = slope_parallel_from_radar(rng, at, LOS, AT, grad)
    assert (e2, n2, u2) == pytest.approx((e, n, u))


def test_vectorised_inversion():
    grad = (0.3, 0.2)
    e = np.array([0.1, -0.5]); n = np.array([0.4, 0.0]); u = grad[0] * e + grad[1] * n
    rng, at = project_to_radar(e, n, u, LOS, AT)
    e2, n2, _ = slope_parallel_from_radar(rng, at, LOS, AT, grad)
    assert np.allclose(e2, e) and np.allclose(n2, n)


def test_agreement():
    assert agreement(0.6, 0.1, 0.5, 0.1)["agree"]
    assert not agreement(0.6, 0.1, -0.5, 0.1)["agree"]
    assert not agreement(0.6, 0.05, 0.1, 0.05)["agree"]
