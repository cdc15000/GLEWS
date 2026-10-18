"""Tests for per-flag data-quality grading and interferogram strips."""

import json

import numpy as np
import pytest

from glews.detect import AnomalyFlag
from glews.quality import (
    assess_flag_quality,
    grade_reliability,
    render_interferogram_strips,
)


def _flag(flag_id=1, pixels=((1, 1), (1, 2)), lat=28.29, lon=85.52):
    return AnomalyFlag(
        flag_id=flag_id, score=4.5, peak_zscore=5.0, mean_zscore=3.0,
        n_pixels=len(pixels), area_m2=12800.0, center_lat=lat, center_lon=lon,
        peak_lat=lat, peak_lon=lon, acceleration_m_yr2=0.1,
        pixel_indices=np.array(pixels), window_index=3, window_date=739800.0,
    )


class TestGradeReliability:
    def test_good(self):
        assert grade_reliability(0.5, 0.1) == "good"

    def test_marginal(self):
        assert grade_reliability(0.3, 0.1) == "marginal"
        assert grade_reliability(0.5, 0.3) == "marginal"

    def test_unreliable_from_unwrapping(self):
        assert grade_reliability(0.6, 0.5) == "unreliable"

    def test_unreliable_from_coherence(self):
        assert grade_reliability(0.15, 0.0) == "unreliable"

    def test_nan_is_unreliable(self):
        assert grade_reliability(float("nan"), 0.0) == "unreliable"


class TestAssessFlagQuality:
    def test_summary_and_pairs(self):
        coh = np.full((3, 4, 4), 0.5, dtype=np.float32)
        coh[2] = 0.1                       # last pair decorrelated
        unrel = np.zeros((3, 4, 4), dtype=bool)
        unrel[1, 1, 1] = True              # half the flag's pixels in pair 2
        dates = [("2026-01-01", "2026-01-13"), ("2026-01-13", "2026-01-25"),
                 ("2026-01-25", "2026-02-06")]
        f = _flag()
        assess_flag_quality([f], coh, unrel, dates)
        q = f.detection_details["quality"]
        assert q["n_pairs"] == 3
        assert q["n_pairs_usable"] == 2    # pair 3 fails coherence; pair 2 at 0.5 unreliable passes
        assert q["mean_coherence"] == pytest.approx((0.5 + 0.5 + 0.1) / 3, abs=1e-3)
        assert q["unreliable_fraction"] == pytest.approx(1 / 6, abs=1e-3)
        assert q["reliability"] == "good"          # 0.367 coherence, 17% unreliable
        assert q["pairs"][2] == {"ref": "2026-01-25", "sec": "2026-02-06", "coherence": 0.1,
                                 "unreliable_fraction": 0.0, "usable": False}

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            assess_flag_quality([_flag()], np.zeros((2, 4, 4)), np.zeros((2, 4, 4), bool),
                                [("a", "b")])


class TestInterferogramStrips:
    def test_writes_strip_for_flags_inside_tiles(self, tmp_path):
        lat, lon = np.meshgrid(np.linspace(28.30, 28.28, 40), np.linspace(85.51, 85.53, 40),
                               indexing="ij")
        tiles = [dict(ref="2026-01-01", sec="2026-01-13", phase=np.zeros((40, 40)), lat=lat, lon=lon),
                 dict(ref="2026-01-13", sec="2026-01-25", phase=np.ones((40, 40)), lat=lat, lon=lon)]
        inside = _flag(flag_id=7, lat=28.29, lon=85.52)
        outside = _flag(flag_id=8, lat=27.0, lon=84.0)
        n = render_interferogram_strips([inside, outside], tiles, tmp_path, half_width_m=500)
        assert n == 1
        assert inside.detection_details["ifg_strip"] == "figures/ifg/flag_007.png"
        assert (tmp_path / "figures/ifg/flag_007.png").stat().st_size > 0
        assert "ifg_strip" not in outside.detection_details


class TestExportGeojsonQuality:
    def test_exports_peak_date_quality_strip_and_provenance(self, tmp_path):
        from glews.report import _export_geojson

        f = _flag()
        f.detection_details["quality"] = {"reliability": "good", "mean_coherence": 0.5}
        f.detection_details["ifg_strip"] = "figures/ifg/flag_001.png"
        prov = {"sensor": "NISAR", "track": 98}
        path = _export_geojson([f], None, tmp_path, data_source="nisar", provenance=prov)
        data = json.loads(path.read_text())
        props = data["features"][0]["properties"]
        assert props["peak_window_date"] == "2026-07-03"
        assert props["quality"]["reliability"] == "good"
        assert props["ifg_strip"] == "figures/ifg/flag_001.png"
        assert data["properties"]["provenance"] == prov
