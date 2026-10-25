"""Tests for the frozen E4 detector (glews.e4) and the transverse Mercator helper (glews.tm)."""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from glews import e4, tm

ROOT = Path(__file__).resolve().parents[1]


class TestTM:
    def test_zone_meridians(self):
        assert tm.utm_lon0("EPSG:32645") == 87.0
        assert tm.utm_lon0("EPSG:32644") == 81.0
        assert tm.utm_crs(83.9) == "EPSG:32644" and tm.utm_crs(85.5) == "EPSG:32645"
        with pytest.raises(ValueError):
            tm.utm_lon0("EPSG:4326")

    def test_round_trip(self):
        for lon0 in (75.0, 81.0, 87.0):
            lat, lon = 30.38, lon0 - 1.3
            x, y = tm.fwd(lat, lon, lon0)
            la, lo = tm.inv(x, y, lon0)
            assert abs(la - lat) < 1e-8 and abs(lo - lon) < 1e-8   # ~1 mm

    def test_matches_2026_helper(self):
        sys.path.insert(0, str(ROOT / "scripts/nepal_2026"))
        import utm
        assert tm.fwd(28.288, 85.524) == utm.fwd(28.288, 85.524)


class TestSeason:
    def test_windows(self):
        w = e4.season_dates(2027)
        assert w == dict(base_start="20261201", base_end="20270115", pm_start="20270115", pm_end="20270527")


def _P(ratio, incr, target=None, cutoffs=None):
    ratio, incr = np.asarray(ratio, float), np.asarray(incr, float)
    return dict(ratio=ratio, incr=incr, target=np.zeros(ratio.shape[1:], bool) if target is None else target,
                cutoffs=cutoffs or [f"2027{m:02d}01" for m in range(2, 2 + ratio.shape[0])])


class TestFlags:
    def test_group_size_and_thresholds(self):
        r = np.zeros((1, 10, 10)); i = np.zeros((1, 10, 10))
        r[0, 1, 1:7] = 2.0; i[0, 1, 1:7] = 20.0          # 6 px: a flag
        r[0, 5, 1:6] = 2.0; i[0, 5, 1:6] = 20.0          # 5 px: too small
        r[0, 8, 1:8] = 2.0; i[0, 8, 1:8] = 11.0          # increase below 12 m/yr
        lab, n = e4.flags_at(_P(r, i), 0)
        assert n == 1 and (lab[1, 1:7] > 0).all() and lab[5].sum() == 0 and lab[8].sum() == 0

    def test_persistent_first_allows_one_later_miss(self):
        assert e4.persistent_first([False, True, True, False, True]) == 1
        assert e4.persistent_first([True, False, False, True, True]) == 3
        assert e4.persistent_first([False, False]) is None

    def test_run_respects_before(self):
        r = np.zeros((3, 8, 8)); i = np.zeros((3, 8, 8))
        r[2, 2, 0:6] = 2.0; i[2, 2, 0:6] = 20.0          # flag only at the last cutoff
        P = _P(r, i, cutoffs=["20270301", "20270401", "20270501"])
        assert e4.run(P)["fa"] == 1
        assert e4.run(P, before="20270501")["fa"] == 0

    def test_lead(self):
        assert e4.lead_days("20260220", date(2026, 8, 26)) == 187
        assert e4.lead_days(None, date(2026, 8, 26)) is None


SITES_2026 = [
    ("nepal", "data/nisar/nepal_2026/goff_t98_crop", None, 0, "20260220"),
    ("ganesh_east", "data/nisar/control_ganesh_east/goff_t98_crop", (28.36, 85.22), 0, None),
    ("jugal", "data/nisar/control_jugal/goff_t98_crop", (28.13, 85.86), 2, None),
    ("annapurna_sanctuary", "data/nisar/control_annapurna_sanctuary/goff_t98_crop", (28.55, 83.90), 12, None),
    ("langtang_east", "data/nisar/control_langtang_east/goff_t98_crop", (28.25, 85.72), 4, None),
]


@pytest.mark.skipif(not all((ROOT / g).exists() for _, g, *_ in SITES_2026), reason="2026 GOFF crops not present")
class TestRegression2026:
    """Freeze condition (docs/prereg_2027_prospective.md): reproduce Phase 2c exactly."""

    @pytest.fixture(scope="class")
    def nepal_cfg(self):
        import yaml
        return yaml.safe_load(open(ROOT / "config/nepal_2026.yaml"))["site"]

    @pytest.mark.parametrize("name,g,centre,fa,first", SITES_2026)
    def test_phase2c_counts(self, nepal_cfg, name, g, centre, fa, first):
        polys = None
        if centre is None:
            centre = (nepal_cfg["latitude"], nepal_cfg["longitude"])
            polys = [nepal_cfg["known_source"]["polygon"], nepal_cfg["reference_areas"][0]["polygon"]]
        P = e4.prepare(ROOT / g, centre, 2026, track=98, crs="EPSG:32645", target_polygons=polys)
        r = e4.run(P)
        assert P["n_base"] == 4
        assert r["fa"] == fa and r["first"] == first

    def test_arrays_match_2026_code(self, nepal_cfg):
        sys.path.insert(0, str(ROOT / "scripts/nepal_2026"))
        import dev_premelt_detector as D
        old = D.prepare("jugal", "data/nisar/control_jugal/goff_t98_crop", (28.13, 85.86), False)
        new = e4.prepare(ROOT / "data/nisar/control_jugal/goff_t98_crop", (28.13, 85.86), 2026, track=98)
        k0 = new["cutoffs"].index(old["cutoffs"][0])         # calendar rule adds the 27 Jan cutoff first
        assert new["cutoffs"][k0:] == old["cutoffs"]
        np.testing.assert_array_equal(new["ratio"][k0:], old["ratio"])
        np.testing.assert_array_equal(new["incr"][k0:], old["incr"])
        assert np.isnan(new["ratio"][:k0]).all()              # too few window pairs to alarm
