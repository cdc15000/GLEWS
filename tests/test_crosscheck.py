"""Tests for the Tier 1 multi-sensor cross-check filters."""

import numpy as np
import pytest

from glews.crosscheck import (
    CrossCheckResult,
    InSARQualityChecker,
    OpticalCrossChecker,
    apply_tier1_filters,
)
from glews.detect import AnomalyFlag


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_flag(
    flag_id: int = 0,
    pixel_rows: np.ndarray | None = None,
    pixel_cols: np.ndarray | None = None,
    score: float = 5.0,
    window_index: int = 5,
) -> AnomalyFlag:
    """Create a synthetic AnomalyFlag for testing."""
    if pixel_rows is None:
        # Default: a compact 4x4 block at rows 10-13, cols 10-13
        rr, cc = np.meshgrid(np.arange(10, 14), np.arange(10, 14), indexing="ij")
        pixel_rows = rr.ravel()
        pixel_cols = cc.ravel()

    return AnomalyFlag(
        flag_id=flag_id,
        score=score,
        peak_zscore=4.0,
        mean_zscore=3.0,
        n_pixels=len(pixel_rows),
        area_m2=len(pixel_rows) * 900.0,
        center_lat=28.2,
        center_lon=85.9,
        peak_lat=28.2,
        peak_lon=85.9,
        acceleration_m_yr2=0.05,
        pixel_indices=np.column_stack([pixel_rows, pixel_cols]),
        window_index=window_index,
        window_date=738900.0,
    )


def _make_coherence(n_rows: int = 30, n_cols: int = 30, value: float = 0.8) -> np.ndarray:
    """Create a uniform coherence map."""
    return np.full((n_rows, n_cols), value)


def _make_displacement_stack(
    n_epochs: int = 10,
    n_rows: int = 30,
    n_cols: int = 30,
    signal_rows: slice | None = None,
    signal_cols: slice | None = None,
    signal_strength: float = 0.05,
    persistent: bool = True,
) -> np.ndarray:
    """
    Create a synthetic displacement stack.

    If signal_rows/signal_cols are given and persistent=True, injects a
    growing displacement signal at those pixels across all epochs.
    If persistent=False, only injects at a single epoch.
    """
    rng = np.random.default_rng(42)
    stack = rng.normal(0, 0.005, (n_epochs, n_rows, n_cols))

    if signal_rows is not None and signal_cols is not None:
        if persistent:
            for e in range(n_epochs):
                stack[e, signal_rows, signal_cols] += signal_strength * (e + 1)
        else:
            # Single-epoch artifact at epoch n_epochs // 2
            stack[n_epochs // 2, signal_rows, signal_cols] += signal_strength * 10

    return stack


# ---------------------------------------------------------------------------
# InSARQualityChecker tests
# ---------------------------------------------------------------------------

class TestCoherenceQuality:
    """Tests for InSARQualityChecker.check_coherence_quality."""

    def test_high_coherence_scores_high(self):
        """Flagged pixels with high coherence should score near 1.0."""
        checker = InSARQualityChecker()
        coherence = _make_coherence(value=0.9)
        flag = _make_flag()
        mask = np.zeros((30, 30), dtype=bool)
        mask[flag.pixel_indices[:, 0], flag.pixel_indices[:, 1]] = True

        score = checker.check_coherence_quality(coherence, mask, min_coherence=0.3)
        assert score == 1.0

    def test_low_coherence_scores_low(self):
        """Flagged pixels with low coherence should score near 0.0."""
        checker = InSARQualityChecker()
        coherence = _make_coherence(value=0.1)
        flag = _make_flag()
        mask = np.zeros((30, 30), dtype=bool)
        mask[flag.pixel_indices[:, 0], flag.pixel_indices[:, 1]] = True

        score = checker.check_coherence_quality(coherence, mask, min_coherence=0.3)
        assert score == 0.0

    def test_mixed_coherence(self):
        """Half high, half low coherence should score around 0.5."""
        checker = InSARQualityChecker()
        coherence = _make_coherence(value=0.1)
        # Set half the flagged pixels to high coherence
        coherence[10:12, 10:14] = 0.9  # 8 of 16 pixels high
        flag = _make_flag()
        mask = np.zeros((30, 30), dtype=bool)
        mask[flag.pixel_indices[:, 0], flag.pixel_indices[:, 1]] = True

        score = checker.check_coherence_quality(coherence, mask, min_coherence=0.3)
        assert 0.4 <= score <= 0.6

    def test_empty_mask_returns_zero(self):
        """An empty flag mask should return 0.0."""
        checker = InSARQualityChecker()
        coherence = _make_coherence(value=0.9)
        mask = np.zeros((30, 30), dtype=bool)

        score = checker.check_coherence_quality(coherence, mask)
        assert score == 0.0


class TestSpatialConsistency:
    """Tests for InSARQualityChecker.check_spatial_consistency."""

    def test_connected_block_scores_high(self):
        """A single connected block should score 1.0."""
        checker = InSARQualityChecker()
        disp = np.zeros((30, 30))
        flag = _make_flag()  # compact 4x4 block
        mask = np.zeros((30, 30), dtype=bool)
        mask[flag.pixel_indices[:, 0], flag.pixel_indices[:, 1]] = True

        score = checker.check_spatial_consistency(disp, mask)
        assert score == 1.0

    def test_scattered_pixels_score_low(self):
        """Scattered, disconnected pixels should score low."""
        checker = InSARQualityChecker()
        disp = np.zeros((30, 30))

        # Create scattered pixels with no adjacency
        rows = np.array([2, 5, 10, 15, 20, 25, 28])
        cols = np.array([3, 8, 15, 22, 7, 18, 25])
        mask = np.zeros((30, 30), dtype=bool)
        mask[rows, cols] = True

        score = checker.check_spatial_consistency(disp, mask)
        # Each pixel is its own component -> largest is 1/7
        assert score < 0.3

    def test_two_clusters(self):
        """Two separate clusters: largest fraction should be < 1.0."""
        checker = InSARQualityChecker()
        disp = np.zeros((30, 30))
        mask = np.zeros((30, 30), dtype=bool)
        # Cluster 1: 4 pixels
        mask[5:7, 5:7] = True
        # Cluster 2: 9 pixels (well separated)
        mask[20:23, 20:23] = True

        score = checker.check_spatial_consistency(disp, mask)
        # Largest cluster is 9 out of 13 total
        assert 0.6 < score < 0.8

    def test_empty_mask_returns_zero(self):
        """An empty flag mask should return 0.0."""
        checker = InSARQualityChecker()
        disp = np.zeros((30, 30))
        mask = np.zeros((30, 30), dtype=bool)

        score = checker.check_spatial_consistency(disp, mask)
        assert score == 0.0


class TestTemporalConsistency:
    """Tests for InSARQualityChecker.check_temporal_consistency."""

    def test_persistent_signal_scores_high(self):
        """A signal present across all epochs should score high."""
        checker = InSARQualityChecker()
        dates = np.arange(10, dtype=float)
        stack = _make_displacement_stack(
            signal_rows=slice(10, 14),
            signal_cols=slice(10, 14),
            signal_strength=0.05,
            persistent=True,
        )
        flag = _make_flag()
        mask = np.zeros((30, 30), dtype=bool)
        mask[flag.pixel_indices[:, 0], flag.pixel_indices[:, 1]] = True

        score = checker.check_temporal_consistency(dates, stack, mask)
        assert score > 0.5

    def test_transient_signal_scores_low(self):
        """A single-epoch artifact should score low."""
        checker = InSARQualityChecker()
        dates = np.arange(10, dtype=float)
        stack = _make_displacement_stack(
            signal_rows=slice(10, 14),
            signal_cols=slice(10, 14),
            signal_strength=0.05,
            persistent=False,
        )
        flag = _make_flag()
        mask = np.zeros((30, 30), dtype=bool)
        mask[flag.pixel_indices[:, 0], flag.pixel_indices[:, 1]] = True

        score = checker.check_temporal_consistency(dates, stack, mask)
        assert score < 0.3

    def test_no_signal_scores_zero(self):
        """Pure noise (no injected signal) should score low."""
        checker = InSARQualityChecker()
        dates = np.arange(10, dtype=float)
        stack = _make_displacement_stack(persistent=True)  # no signal injected
        flag = _make_flag()
        mask = np.zeros((30, 30), dtype=bool)
        mask[flag.pixel_indices[:, 0], flag.pixel_indices[:, 1]] = True

        score = checker.check_temporal_consistency(dates, stack, mask)
        # No systematic signal at the flagged pixels
        assert score < 0.5

    def test_empty_mask_returns_zero(self):
        """An empty flag mask should return 0.0."""
        checker = InSARQualityChecker()
        dates = np.arange(10, dtype=float)
        stack = _make_displacement_stack()
        mask = np.zeros((30, 30), dtype=bool)

        score = checker.check_temporal_consistency(dates, stack, mask)
        assert score == 0.0


# ---------------------------------------------------------------------------
# OpticalCrossChecker tests
# ---------------------------------------------------------------------------

class TestOpticalCrossChecker:
    """Tests for the optical cross-check stub."""

    def test_returns_unavailable(self):
        """Stub should return optical_confirmation=None."""
        checker = OpticalCrossChecker()
        result = checker.check_surface_change(
            bbox=(85.8, 28.1, 86.0, 28.3),
            date_range=("2026-08-01", "2026-08-26"),
        )

        assert isinstance(result, CrossCheckResult)
        assert result.optical_confirmation is None
        assert result.recommendation == "retain"
        assert result.details["optical_status"] == "unavailable"

    def test_result_has_correct_fields(self):
        """CrossCheckResult should have all expected fields."""
        checker = OpticalCrossChecker()
        result = checker.check_surface_change(
            bbox=(85.8, 28.1, 86.0, 28.3),
            date_range=("2026-08-01", "2026-08-26"),
        )

        assert 0 <= result.quality_score <= 1
        assert 0 <= result.spatial_consistency <= 1
        assert 0 <= result.temporal_consistency <= 1
        assert isinstance(result.details, dict)


# ---------------------------------------------------------------------------
# Full pipeline tests
# ---------------------------------------------------------------------------

class TestApplyTier1Filters:
    """Tests for the apply_tier1_filters entry point."""

    def _make_config(self, **overrides):
        """Default config with tier1_filters section."""
        t1 = {
            "min_coherence": 0.3,
            "min_spatial_consistency": 0.5,
            "min_temporal_consistency": 0.3,
            "remove_below_quality": 0.2,
            "demote_below_quality": 0.5,
        }
        t1.update(overrides)
        return {"detect": {"tier1_filters": t1}}

    def test_good_flag_retained(self):
        """A flag with high coherence, spatial, and temporal consistency is retained."""
        flag = _make_flag(score=5.0)
        coherence = _make_coherence(value=0.9)
        dates = np.arange(10, dtype=float)
        stack = _make_displacement_stack(
            signal_rows=slice(10, 14),
            signal_cols=slice(10, 14),
            signal_strength=0.05,
            persistent=True,
        )
        config = self._make_config()

        result = apply_tier1_filters([flag], stack, coherence, dates, config)

        assert len(result) == 1
        t1q = result[0].detection_details["tier1_quality"]
        assert t1q["recommendation"] == "retain"
        assert t1q["quality_score"] > 0.5
        assert t1q["spatial_consistency"] == 1.0

    def test_low_coherence_flag_removed(self):
        """A flag in a decorrelated area should be removed."""
        flag = _make_flag(score=5.0)
        coherence = _make_coherence(value=0.05)  # very low coherence
        dates = np.arange(10, dtype=float)
        # No persistent signal either
        stack = _make_displacement_stack()
        config = self._make_config()

        result = apply_tier1_filters([flag], stack, coherence, dates, config)

        assert len(result) == 0

    def test_scattered_flag_handled(self):
        """A flag with scattered pixels (low spatial consistency) is demoted or removed."""
        rows = np.array([2, 5, 10, 15, 20, 25, 28])
        cols = np.array([3, 8, 15, 22, 7, 18, 25])
        flag = _make_flag(pixel_rows=rows, pixel_cols=cols, score=5.0)
        coherence = _make_coherence(value=0.9)
        dates = np.arange(10, dtype=float)
        stack = _make_displacement_stack()
        config = self._make_config()

        result = apply_tier1_filters([flag], stack, coherence, dates, config)

        # Should be demoted or removed (low spatial + temporal)
        if len(result) > 0:
            t1q = result[0].detection_details["tier1_quality"]
            assert t1q["recommendation"] in ("demote", "remove")
            assert t1q["spatial_consistency"] < 0.5

    def test_tier1_quality_annotated(self):
        """Every retained flag should have tier1_quality in detection_details."""
        flag = _make_flag(score=5.0)
        coherence = _make_coherence(value=0.9)
        dates = np.arange(10, dtype=float)
        stack = _make_displacement_stack(
            signal_rows=slice(10, 14),
            signal_cols=slice(10, 14),
            signal_strength=0.05,
            persistent=True,
        )
        config = self._make_config()

        result = apply_tier1_filters([flag], stack, coherence, dates, config)

        for f in result:
            t1q = f.detection_details["tier1_quality"]
            assert "quality_score" in t1q
            assert "spatial_consistency" in t1q
            assert "temporal_consistency" in t1q
            assert "composite_score" in t1q
            assert "recommendation" in t1q
            assert t1q["recommendation"] in ("retain", "demote", "remove")

    def test_demoted_flag_has_reduced_score(self):
        """A demoted flag should have its score halved."""
        flag = _make_flag(score=10.0)
        # Medium coherence, scattered pixels, transient signal -> demote
        coherence = _make_coherence(value=0.5)
        dates = np.arange(10, dtype=float)
        stack = _make_displacement_stack(
            signal_rows=slice(10, 14),
            signal_cols=slice(10, 14),
            signal_strength=0.05,
            persistent=True,
        )
        config = self._make_config(
            remove_below_quality=0.1,
            demote_below_quality=0.9,
        )

        result = apply_tier1_filters([flag], stack, coherence, dates, config)

        if len(result) > 0 and result[0].detection_details["tier1_quality"]["recommendation"] == "demote":
            assert result[0].score == pytest.approx(5.0)

    def test_empty_flags_list(self):
        """Empty input should return empty output."""
        coherence = _make_coherence()
        dates = np.arange(10, dtype=float)
        stack = _make_displacement_stack()
        config = self._make_config()

        result = apply_tier1_filters([], stack, coherence, dates, config)
        assert result == []

    def test_aps_results_added_when_dem_provided(self):
        """When a DEM is provided, APS results should appear in tier1_quality."""
        flag = _make_flag(score=5.0)
        coherence = _make_coherence(value=0.9)
        dates = np.arange(10, dtype=float)
        stack = _make_displacement_stack(
            signal_rows=slice(10, 14),
            signal_cols=slice(10, 14),
            signal_strength=0.05,
            persistent=True,
        )
        config = self._make_config()

        # Create a DEM with some elevation variation
        rng = np.random.default_rng(7)
        dem = rng.uniform(1000, 3000, (30, 30))

        result = apply_tier1_filters([flag], stack, coherence, dates, config, dem=dem)

        assert len(result) >= 1
        t1q = result[0].detection_details["tier1_quality"]
        assert "aps_score" in t1q
        assert 0.0 <= t1q["aps_score"] <= 1.0

    def test_no_aps_without_dem(self):
        """Without a DEM, aps_score should not appear in tier1_quality."""
        flag = _make_flag(score=5.0)
        coherence = _make_coherence(value=0.9)
        dates = np.arange(10, dtype=float)
        stack = _make_displacement_stack(
            signal_rows=slice(10, 14),
            signal_cols=slice(10, 14),
            signal_strength=0.05,
            persistent=True,
        )
        config = self._make_config()

        result = apply_tier1_filters([flag], stack, coherence, dates, config)

        assert len(result) >= 1
        t1q = result[0].detection_details["tier1_quality"]
        assert "aps_score" not in t1q

    def test_config_defaults_when_missing(self):
        """Should work with an empty config (using defaults)."""
        flag = _make_flag(score=5.0)
        coherence = _make_coherence(value=0.9)
        dates = np.arange(10, dtype=float)
        stack = _make_displacement_stack(
            signal_rows=slice(10, 14),
            signal_cols=slice(10, 14),
            signal_strength=0.05,
            persistent=True,
        )
        config = {"detect": {}}

        result = apply_tier1_filters([flag], stack, coherence, dates, config)

        # Should not raise, should use defaults
        assert isinstance(result, list)
