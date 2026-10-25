"""Tests for real-exposure reach analysis (glews.exposure)."""
import numpy as np

from glews.exposure import (
    d8_receivers, fill_depressions, parse_overpass, rasterize_assets, reach_exposure,
)


def _valley(n=60, post=100.0):
    """V-shaped valley draining south (down the rows): side slopes 0.3, floor gradient 0.05."""
    r, c = np.indices((n, n)).astype(float)
    return 3000.0 - 0.05 * r * post + 0.3 * np.abs(c - n // 2) * post


class TestFill:
    def test_pit_is_filled_and_drains(self):
        z = np.full((7, 7), 10.0); z[0, :] = z[:, 0] = z[:, -1] = 10.0
        z[-1, :] = 5.0; z[3, 3] = 1.0                           # pit in the middle, outlet on the south edge
        zf = fill_depressions(z)
        assert zf[3, 3] > 9.99
        rec = d8_receivers(zf)
        cur, seen = 3 * 7 + 3, 0
        while rec[cur] >= 0 and seen < 50:
            cur, seen = rec[cur], seen + 1
        assert cur // 7 in (0, 6) or cur % 7 in (0, 6)       # reaches the edge

    def test_nan_cells_act_as_outlets(self):
        z = np.full((5, 5), 10.0); z[2, 2] = np.nan
        zf = fill_depressions(z)
        assert np.isnan(zf[2, 2]) and np.isfinite(zf[0, 0])


class TestReceivers:
    def test_flow_goes_down_the_valley(self):
        z = _valley()
        rec = d8_receivers(fill_depressions(z))
        n = z.shape[1]
        r, c = divmod(rec[10 * n + 30], n)
        assert (r, c) == (11, 30)                                  # floor flows south
        r, c = divmod(rec[10 * n + 20], n)
        assert c > 20                                              # side slope flows to the floor


class TestReach:
    def test_reach_limited_by_travel_angle(self):
        z, post = _valley(), 100.0
        exposed = np.zeros(z.shape, bool); exposed[50, 30] = True    # village on the floor, 4 km below row 10
        src = np.zeros(z.shape, bool); src[10, 30] = True; src[10, 20] = True
        # floor gradient 0.05 < 0.1 -> the floor source cannot reach 4 km down at H/L 0.1
        assert not reach_exposure(z, post, exposed, src, min_hl=0.1, buffer_m=0)[10, 30]
        # but it can at H/L 0.04
        assert reach_exposure(z, post, exposed, src, min_hl=0.04, buffer_m=0)[10, 30]
        # the side-slope source drops 300 m first, so it carries further at the same angle
        side = reach_exposure(z, post, exposed, src, min_hl=0.06, buffer_m=0)
        assert side[10, 20] and not side[10, 30]

    def test_buffer_counts_assets_beside_the_path(self):
        z, post = _valley(), 100.0
        exposed = np.zeros(z.shape, bool); exposed[20, 33] = True    # 300 m off the channel
        src = np.zeros(z.shape, bool); src[10, 30] = True
        assert not reach_exposure(z, post, exposed, src, min_hl=0.04, buffer_m=0)[10, 30]
        assert reach_exposure(z, post, exposed, src, min_hl=0.04, buffer_m=300)[10, 30]

    def test_only_sources_are_marked(self):
        z, post = _valley(), 100.0
        exposed = np.zeros(z.shape, bool); exposed[55, :] = True
        src = np.zeros(z.shape, bool); src[5, 5] = True
        out = reach_exposure(z, post, exposed, src, min_hl=0.01)
        assert out.sum() == 1 and out[5, 5]


class TestAssets:
    def test_parse_overpass_kinds_and_geometry(self):
        payload = {"elements": [
            {"type": "node", "lon": 85.3, "lat": 28.2, "tags": {"place": "village", "name": "A"}},
            {"type": "way", "geometry": [{"lon": 85.0, "lat": 28.0}, {"lon": 85.1, "lat": 28.1}],
             "tags": {"highway": "trunk"}},
            {"type": "way", "center": {"lon": 85.2, "lat": 28.1}, "tags": {"power": "plant"}},
            {"type": "node", "lon": 1, "lat": 1, "tags": {"amenity": "cafe"}},
        ]}
        a = parse_overpass(payload)
        assert [x["kind"] for x in a] == ["settlement", "road", "power"]
        assert a[1]["lon"] == [85.0, 85.1] and a[0]["name"] == "A"

    def test_rasterize_densifies_lines(self):
        g = rasterize_assets([(np.array([5.0, 995.0]), np.array([-5.0, -5.0]))], 0.0, 0.0, 10.0, (5, 100))
        assert g[0].all() and g.sum() == 100
        g = rasterize_assets([(np.array([55.0]), np.array([-25.0]))], 0.0, 0.0, 10.0, (5, 10))
        assert g[2, 5] and g.sum() == 1
