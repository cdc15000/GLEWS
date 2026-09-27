"""Tests for cascade risk assessment with population exposure mapping."""

from __future__ import annotations

import numpy as np
import pytest

from glews.cascade import (
    CascadeAssessment,
    ExposureResult,
    RiskLevel,
    assess_cascade_risk,
    assess_cascade_risk_batch,
    estimate_exposure,
    estimate_runout,
    generate_exposure_summary,
)
from glews.detect import AnomalyFlag


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_flag() -> AnomalyFlag:
    """A realistic anomaly flag for testing."""
    return AnomalyFlag(
        flag_id=1,
        score=3.5,
        peak_zscore=5.0,
        mean_zscore=3.0,
        n_pixels=100,
        area_m2=250_000,  # 0.25 km^2
        center_lat=28.5,
        center_lon=84.7,
        peak_lat=28.51,
        peak_lon=84.71,
        acceleration_m_yr2=0.05,
        pixel_indices=np.column_stack([
            np.arange(100) // 10,
            np.arange(100) % 10,
        ]),
        window_index=0,
        window_date=738886.0,
    )


@pytest.fixture
def small_flag() -> AnomalyFlag:
    """A small anomaly flag (below volume threshold)."""
    return AnomalyFlag(
        flag_id=2,
        score=1.0,
        peak_zscore=2.0,
        mean_zscore=1.5,
        n_pixels=5,
        area_m2=500,  # very small
        center_lat=28.5,
        center_lon=84.7,
        peak_lat=28.5,
        peak_lon=84.7,
        acceleration_m_yr2=0.01,
        pixel_indices=np.column_stack([np.arange(5), np.arange(5)]),
        window_index=0,
        window_date=738886.0,
    )


# ---------------------------------------------------------------------------
# estimate_runout tests
# ---------------------------------------------------------------------------


class TestEstimateRunout:
    """Tests for the Scheidegger angle-of-reach runout model."""

    def test_basic_geometry(self):
        """With no volume, angle comes from elevation drop / distance."""
        result = estimate_runout(1000.0, 5000.0)
        assert result["runout_distance_km"] == pytest.approx(5.0, rel=1e-3)
        assert 0 < result["angle_of_reach_deg"] < 90

    def test_larger_volume_longer_runout(self):
        """Larger volume produces longer runout (Scheidegger mobility)."""
        small = estimate_runout(1000.0, 5000.0, volume_m3=1e5)
        large = estimate_runout(1000.0, 5000.0, volume_m3=1e8)
        assert large["runout_distance_km"] > small["runout_distance_km"]

    def test_larger_volume_smaller_angle(self):
        """Larger volume produces smaller angle of reach."""
        small = estimate_runout(1000.0, 5000.0, volume_m3=1e5)
        large = estimate_runout(1000.0, 5000.0, volume_m3=1e8)
        assert large["angle_of_reach_deg"] < small["angle_of_reach_deg"]

    def test_angle_in_valid_range(self):
        """Angle of reach must be between 0 and 90 degrees."""
        for vol in [None, 0, 1e3, 1e6, 1e9]:
            result = estimate_runout(500.0, 3000.0, volume_m3=vol)
            assert 0 <= result["angle_of_reach_deg"] <= 90

    def test_runout_non_negative(self):
        """Runout distance is always non-negative."""
        for h in [0, 100, 1000]:
            for d in [0, 100, 5000]:
                result = estimate_runout(float(h), float(d))
                assert result["runout_distance_km"] >= 0

    def test_round_trip_consistency(self):
        """Runout ~ H / tan(angle) when volume-based."""
        H = 800.0
        result = estimate_runout(H, 5000.0, volume_m3=1e6)
        angle_rad = np.radians(result["angle_of_reach_deg"])
        if angle_rad > 0:
            expected_km = (H / np.tan(angle_rad)) / 1000
            assert result["runout_distance_km"] == pytest.approx(
                expected_km, rel=1e-3
            )

    def test_flat_terrain(self):
        """Zero elevation drop produces zero runout and zero angle."""
        result = estimate_runout(0.0, 5000.0)
        assert result["runout_distance_km"] == 0.0
        assert result["angle_of_reach_deg"] == 0.0

    def test_flat_terrain_with_volume(self):
        """Zero elevation drop with volume still produces zero runout."""
        result = estimate_runout(0.0, 5000.0, volume_m3=1e6)
        assert result["runout_distance_km"] == 0.0
        assert result["angle_of_reach_deg"] == 0.0

    def test_zero_horizontal_distance(self):
        """Zero horizontal distance does not cause division by zero."""
        result = estimate_runout(500.0, 0.0)
        assert result["angle_of_reach_deg"] == 90.0
        assert result["runout_distance_km"] == 0.0

    def test_zero_volume(self):
        """Volume of zero falls back to geometric calculation."""
        result = estimate_runout(500.0, 3000.0, volume_m3=0)
        # Should use geometric fallback since volume is 0
        assert result["runout_distance_km"] == pytest.approx(3.0, rel=1e-3)

    def test_monotonic_volume_runout(self):
        """Runout increases monotonically with volume."""
        volumes = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
        runouts = [
            estimate_runout(500.0, 3000.0, volume_m3=v)["runout_distance_km"]
            for v in volumes
        ]
        for i in range(len(runouts) - 1):
            assert runouts[i + 1] > runouts[i], (
                f"Runout should increase: vol={volumes[i]:.0e} -> {runouts[i]:.2f}, "
                f"vol={volumes[i+1]:.0e} -> {runouts[i+1]:.2f}"
            )


# ---------------------------------------------------------------------------
# estimate_exposure tests
# ---------------------------------------------------------------------------


class TestEstimateExposure:
    """Tests for the population exposure estimation."""

    def test_basic_synthetic(self):
        """Synthetic model produces plausible settlements."""
        result = estimate_exposure(28.5, 84.7, 3500.0, 30.0)
        assert isinstance(result, ExposureResult)
        assert result.total_population_exposed >= 0
        assert result.runout_distance_km == 30.0

    def test_distance_bands_present(self):
        """All three standard distance bands must be present."""
        result = estimate_exposure(28.5, 84.7, 3500.0, 50.0)
        assert "0-10km" in result.population_by_distance
        assert "10-30km" in result.population_by_distance
        assert "30-100km" in result.population_by_distance

    def test_total_equals_sum_of_bands(self):
        """Total population equals sum of distance bands."""
        result = estimate_exposure(28.5, 84.7, 3500.0, 50.0)
        band_sum = sum(result.population_by_distance.values())
        assert result.total_population_exposed == band_sum

    def test_bands_beyond_runout_are_zero(self):
        """Distance bands entirely beyond runout should have zero pop."""
        result = estimate_exposure(28.5, 84.7, 3500.0, 5.0)
        # With 5km runout, the 10-30km and 30-100km bands should be zero
        assert result.population_by_distance["10-30km"] == 0
        assert result.population_by_distance["30-100km"] == 0

    def test_zero_runout_zero_exposure(self):
        """Zero runout distance produces zero exposure."""
        result = estimate_exposure(28.5, 84.7, 3500.0, 0.0)
        assert result.total_population_exposed == 0
        assert result.exposure_score == 0.0
        assert len(result.settlements_at_risk) == 0

    def test_score_range_zero_pop(self):
        """Exposure score is 0 when there is no population."""
        result = estimate_exposure(28.5, 84.7, 3500.0, 0.0)
        assert result.exposure_score == 0.0

    def test_score_range_upper_bound(self):
        """Exposure score does not exceed 10."""
        # Large runout should get large exposure
        result = estimate_exposure(28.5, 84.7, 3500.0, 200.0)
        assert result.exposure_score <= 10.0

    def test_score_range_lower_bound(self):
        """Exposure score is at least 0."""
        result = estimate_exposure(28.5, 84.7, 3500.0, 1.0)
        assert result.exposure_score >= 0.0

    def test_deterministic_synthetic(self):
        """Same inputs produce identical synthetic settlements."""
        r1 = estimate_exposure(28.5, 84.7, 3500.0, 30.0)
        r2 = estimate_exposure(28.5, 84.7, 3500.0, 30.0)
        assert r1.total_population_exposed == r2.total_population_exposed
        assert len(r1.settlements_at_risk) == len(r2.settlements_at_risk)
        for s1, s2 in zip(r1.settlements_at_risk, r2.settlements_at_risk):
            assert s1["name"] == s2["name"]
            assert s1["population"] == s2["population"]

    def test_settlements_have_required_fields(self):
        """Each settlement dict has name, population, distance, coordinates."""
        result = estimate_exposure(28.5, 84.7, 3500.0, 30.0)
        for s in result.settlements_at_risk:
            assert "name" in s
            assert "population" in s
            assert "distance_km" in s
            assert "coordinates" in s
            assert "lat" in s["coordinates"]
            assert "lon" in s["coordinates"]

    def test_with_population_grid(self):
        """Real population grid path works."""
        # Create a simple pop grid with some populated cells
        pop_grid = np.zeros((20, 20))
        pop_grid[5, 5] = 500
        pop_grid[10, 10] = 1000
        result = estimate_exposure(28.5, 84.7, 3500.0, 20.0,
                                   population_data=pop_grid)
        assert isinstance(result, ExposureResult)
        assert result.total_population_exposed >= 0

    def test_settlements_within_runout(self):
        """All reported settlements are within the runout distance."""
        result = estimate_exposure(28.5, 84.7, 3500.0, 15.0)
        for s in result.settlements_at_risk:
            assert s["distance_km"] <= 15.0


# ---------------------------------------------------------------------------
# assess_cascade_risk (single-flag) tests
# ---------------------------------------------------------------------------


class TestAssessCascadeRisk:
    """Tests for the single-flag cascade risk entry point."""

    def test_basic_no_data(self, sample_flag):
        """Works without any DEM or population data (synthetic path)."""
        result = assess_cascade_risk(sample_flag)
        assert isinstance(result, CascadeAssessment)
        assert result.flag is sample_flag
        assert isinstance(result.risk_level, RiskLevel)
        assert result.estimated_volume_m3 > 0
        assert result.exposure is not None

    def test_returns_exposure(self, sample_flag):
        """Assessment includes an ExposureResult."""
        result = assess_cascade_risk(sample_flag)
        assert result.exposure is not None
        assert isinstance(result.exposure, ExposureResult)
        assert result.exposure.total_population_exposed >= 0

    def test_small_flag_low_risk(self, small_flag):
        """Small anomaly area produces low volume and MINIMAL risk."""
        result = assess_cascade_risk(small_flag)
        # area_m2 = 500, volume = 500 * 0.1*sqrt(500) ~ 1118 m3
        # well below default min_volume of 100,000
        assert result.risk_level in (RiskLevel.LOW, RiskLevel.MINIMAL)
        assert not result.volume_sufficient

    def test_with_dem_data(self, sample_flag):
        """Works with a DEM array provided."""
        dem = np.linspace(4000, 2000, 100).reshape(10, 10)
        result = assess_cascade_risk(sample_flag, dem_data=dem)
        assert isinstance(result, CascadeAssessment)
        assert result.elevation_m > 0

    def test_with_config(self, sample_flag):
        """Respects custom configuration."""
        config = {
            "cascade": {
                "min_volume_m3": 1,  # very low threshold
            }
        }
        result = assess_cascade_risk(sample_flag, config=config)
        assert result.volume_sufficient

    def test_cascade_score_positive(self, sample_flag):
        """Cascade score is positive for a real anomaly."""
        result = assess_cascade_risk(sample_flag)
        assert result.cascade_score > 0

    def test_passes_tier1_for_high(self, sample_flag):
        """High risk flags pass tier 1."""
        result = assess_cascade_risk(sample_flag)
        if result.risk_level == RiskLevel.HIGH:
            assert result.passes_tier1

    def test_flat_dem(self, sample_flag):
        """Flat DEM (no elevation variation) handled gracefully."""
        dem = np.full((10, 10), 3000.0)
        result = assess_cascade_risk(sample_flag, dem_data=dem)
        assert isinstance(result, CascadeAssessment)
        # Flat terrain: no height drop, no valley
        assert result.elevation_m == 3000.0


# ---------------------------------------------------------------------------
# assess_cascade_risk_batch tests
# ---------------------------------------------------------------------------


class TestAssessCascadeRiskBatch:
    """Tests for the batch cascade risk assessment."""

    def test_batch_returns_sorted(self, sample_flag):
        """Batch results are sorted by cascade_score descending."""
        flag2 = AnomalyFlag(
            flag_id=2,
            score=1.0,
            peak_zscore=2.0,
            mean_zscore=1.5,
            n_pixels=10,
            area_m2=5000,
            center_lat=28.4,
            center_lon=84.6,
            peak_lat=28.4,
            peak_lon=84.6,
            acceleration_m_yr2=0.01,
            pixel_indices=np.column_stack([np.arange(10), np.arange(10)]),
            window_index=0,
            window_date=738886.0,
        )

        n_rows, n_cols = 20, 20
        lat = np.linspace(28.6, 28.3, n_rows)[:, None] * np.ones(n_cols)
        lon = np.ones((n_rows, 1)) * np.linspace(84.5, 84.8, n_cols)
        dem = np.linspace(4000, 2000, n_rows)[:, None] * np.ones(n_cols)

        config = {"cascade": {
            "min_volume_m3": 100_000,
            "exposure": {"max_runout_km": 50, "angle_of_reach_deg": 11},
            "valley_width_threshold_m": 500,
        }}

        results = assess_cascade_risk_batch(
            [sample_flag, flag2], dem, lat, lon, config
        )
        assert len(results) == 2
        assert results[0].cascade_score >= results[1].cascade_score


# ---------------------------------------------------------------------------
# generate_exposure_summary tests
# ---------------------------------------------------------------------------


class TestGenerateExposureSummary:
    """Tests for the summary table generation."""

    def test_empty_list(self):
        """Empty assessments list produces a valid string."""
        result = generate_exposure_summary([])
        assert isinstance(result, str)
        assert "No cascade assessments" in result

    def test_with_assessments(self, sample_flag):
        """Summary includes flag info and population data."""
        assessment = assess_cascade_risk(sample_flag)
        summary = generate_exposure_summary([assessment])
        assert isinstance(summary, str)
        assert "Population Exposure Summary" in summary
        assert str(sample_flag.flag_id) in summary

    def test_summary_contains_bands(self, sample_flag):
        """Summary includes distance band breakdown."""
        assessment = assess_cascade_risk(sample_flag)
        summary = generate_exposure_summary([assessment])
        # Should contain band labels if exposure data exists
        if assessment.exposure is not None:
            assert "Population by distance band" in summary

    def test_multiple_assessments(self, sample_flag, small_flag):
        """Summary works with multiple assessments."""
        a1 = assess_cascade_risk(sample_flag)
        a2 = assess_cascade_risk(small_flag)
        summary = generate_exposure_summary([a1, a2])
        assert "2 sites assessed" in summary


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for robustness."""

    def test_negative_elevation_drop(self):
        """Negative elevation drop treated as zero."""
        result = estimate_runout(-100.0, 5000.0)
        assert result["runout_distance_km"] == 0.0
        assert result["angle_of_reach_deg"] == 0.0

    def test_very_large_volume(self):
        """Extremely large volume does not produce invalid results."""
        result = estimate_runout(1000.0, 5000.0, volume_m3=1e12)
        assert np.isfinite(result["runout_distance_km"])
        assert 0 <= result["angle_of_reach_deg"] <= 90

    def test_none_volume(self):
        """volume_m3=None uses geometric fallback without error."""
        result = estimate_runout(500.0, 3000.0, volume_m3=None)
        assert result["runout_distance_km"] > 0
        assert 0 < result["angle_of_reach_deg"] < 90

    def test_exposure_result_dataclass(self):
        """ExposureResult can be constructed directly."""
        er = ExposureResult(
            total_population_exposed=1000,
            population_by_distance={"0-10km": 500, "10-30km": 300, "30-100km": 200},
            exposure_score=5.0,
            settlements_at_risk=[],
            runout_distance_km=20.0,
            angle_of_reach_deg=5.0,
        )
        assert er.total_population_exposed == 1000
        assert er.exposure_score == 5.0
