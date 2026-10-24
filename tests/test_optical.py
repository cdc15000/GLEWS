"""Tests for optical pixel-offset tracking (synthetic images, no network)."""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, shift as nd_shift

from glews.optical import (
    OffsetField,
    _scene_from_feature,
    highpass,
    offset_field,
    reference_offsets,
    scl_fractions,
)


def _texture(n=160, seed=0):
    rng = np.random.default_rng(seed)
    return gaussian_filter(rng.normal(size=(n, n)), 1.5).astype(np.float32)


class TestOffsetField:
    @pytest.mark.parametrize("sx,sy", [(0.0, 0.0), (1.3, -0.4), (-3.75, 2.2), (6.1, 5.05)])
    def test_recovers_uniform_subpixel_shift(self, sx, sy):
        ref = _texture()
        sec = nd_shift(ref, (sy, sx), order=3, mode="wrap")   # content moves +sx east, +sy south
        f = offset_field(ref, sec, window=32, step=16)
        inner = (slice(1, -1), slice(1, -1))
        assert np.nanmedian(f.dx[inner]) == pytest.approx(sx, abs=0.06)
        assert np.nanmedian(f.dy[inner]) == pytest.approx(sy, abs=0.06)
        assert np.nanmedian(f.peak[inner]) > 0.3

    def test_localised_motion_is_localised(self):
        ref = _texture(192, seed=1)
        sec = ref.copy()
        moved = nd_shift(ref, (0, 4.0), order=3, mode="wrap")
        sec[64:128, 64:128] = moved[64:128, 64:128]
        f = offset_field(ref, sec, window=32, step=8)
        rr, cc = np.meshgrid(f.rows, f.cols, indexing="ij")
        inside = (rr > 84) & (rr < 108) & (cc > 84) & (cc < 108)
        outside = (rr < 40) | (rr > 152)
        assert np.nanmedian(f.dx[inside]) == pytest.approx(4.0, abs=0.15)
        assert abs(np.nanmedian(f.dx[outside])) < 0.05

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            offset_field(np.zeros((40, 40)), np.zeros((40, 41)))

    def test_flat_window_is_nan(self):
        f = offset_field(np.ones((64, 64)), np.ones((64, 64)), window=32, step=32)
        assert np.isnan(f.dx).all()


class TestReferenceOffsets:
    def test_removes_stable_median_and_reports_sigma(self):
        rng = np.random.default_rng(2)
        dx = 0.5 + rng.normal(0, 0.1, (20, 20))
        dy = -0.3 + rng.normal(0, 0.1, (20, 20))
        dx[8:12, 8:12] += 3
        f = OffsetField(dx, dy, np.ones_like(dx), np.arange(20), np.arange(20), 32, 8)
        stable = np.zeros((20, 20), bool)
        stable[:, :5] = True
        rx, ry, sx, sy = reference_offsets(f, stable)
        assert abs(np.median(rx[stable])) < 1e-9
        assert np.median(rx[8:12, 8:12]) == pytest.approx(3, abs=0.15)
        assert 0.06 < sx < 0.15

    def test_too_few_stable_windows_raise(self):
        f = OffsetField(np.zeros((5, 5)), np.zeros((5, 5)), np.zeros((5, 5)), np.arange(5), np.arange(5), 32, 8)
        with pytest.raises(ValueError):
            reference_offsets(f, np.ones((5, 5), bool), min_peak=0.5)


def test_scl_fractions():
    scl = np.array([[0, 3, 8, 11], [4, 4, 5, 2]])
    m = np.ones_like(scl, bool)
    fr = scl_fractions(scl, m)
    assert fr == {"cloud": 0.125, "shadow": 0.25, "snow": 0.125, "nodata": 0.125}


def test_highpass_removes_gradient():
    yy, xx = np.mgrid[:100, :100]
    img = xx * 10.0 + _texture(100)
    hp = highpass(img, sigma=4)
    assert abs(np.corrcoef(hp[10:-10, 10:-10].ravel(), xx[10:-10, 10:-10].ravel())[0, 1]) < 0.05


def test_scene_from_feature_parses_orbit():
    f = {"id": "S2B_T45RUM_20251230T050343_L2A",
         "properties": {"datetime": "2025-12-30T05:03:43Z", "eo:cloud_cover": 0.5,
                        "view:sun_elevation": 35.8, "view:sun_azimuth": 160.0, "proj:epsg": 32645,
                        "s2:product_uri": "S2B_MSIL2A_20251230T050119_N0511_R119_T45RUM_20251230T072458.SAFE"},
         "assets": {"nir": {"href": "https://x/B08.tif"}, "thumbnail": {"href": "https://x/t.jpg"}}}
    sc = _scene_from_feature(f)
    assert sc.relative_orbit == 119 and sc.date == "2025-12-30"
    assert sc.hrefs == {"nir": "https://x/B08.tif"}
