"""Tests for graph-based spatial anomaly detection."""

import numpy as np
import pytest

from glews.spatial import (
    DeformationGraph,
    SpatialAnomaly,
    detect_spatial_anomalies,
    _pearson_correlation,
)


def _make_coords(n_rows=20, n_cols=20):
    """Create geographic coordinate grids (~30m spacing)."""
    # ~0.00027 degrees ≈ 30m at mid-latitudes
    lat = np.linspace(28.30, 28.30 - 0.00027 * n_rows, n_rows)[:, None] * np.ones(n_cols)
    lon = np.ones(n_rows)[:, None] * np.linspace(85.80, 85.80 + 0.00027 * n_cols, n_cols)
    return lat, lon


def _make_displacement_with_block(
    n_epochs=30,
    n_rows=20,
    n_cols=20,
    block_slice=(slice(5, 12), slice(5, 12)),
    rng=None,
):
    """Create synthetic displacement: correlated block + random noise.

    The block pixels share a common linear trend with small noise,
    while the rest of the grid is pure uncorrelated noise.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    disp = rng.normal(0, 0.01, (n_epochs, n_rows, n_cols))

    # Inject a correlated deformation signal into the block
    trend = np.linspace(0, 0.5, n_epochs)  # 0.5 m over the series
    r_block, c_block = block_slice
    block_h = r_block.stop - r_block.start
    block_w = c_block.stop - c_block.start
    for i in range(block_h):
        for j in range(block_w):
            disp[:, r_block.start + i, c_block.start + j] = (
                trend + rng.normal(0, 0.005, n_epochs)
            )

    return disp


class TestPearsonCorrelation:
    """Tests for the internal correlation helper."""

    def test_perfect_correlation(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _pearson_correlation(a, a) == pytest.approx(1.0)

    def test_negative_correlation(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        assert _pearson_correlation(a, b) == pytest.approx(-1.0)

    def test_uncorrelated(self):
        rng = np.random.default_rng(99)
        a = rng.normal(0, 1, 1000)
        b = rng.normal(0, 1, 1000)
        r = _pearson_correlation(a, b)
        assert abs(r) < 0.1

    def test_nan_handling(self):
        a = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        b = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        # Only indices 0, 3, 4 are jointly valid
        r = _pearson_correlation(a, b)
        assert np.isfinite(r)
        assert r == pytest.approx(1.0)

    def test_too_few_valid(self):
        a = np.array([1.0, np.nan, np.nan])
        b = np.array([np.nan, 2.0, np.nan])
        assert np.isnan(_pearson_correlation(a, b))

    def test_constant_series(self):
        a = np.array([3.0, 3.0, 3.0, 3.0])
        b = np.array([1.0, 2.0, 3.0, 4.0])
        # Zero variance in a → correlation is 0.0
        assert _pearson_correlation(a, b) == pytest.approx(0.0)


class TestGraphConstruction:
    """Tests for DeformationGraph.build_graph with known geometry."""

    def test_4connectivity_edges(self):
        """A 3x3 grid should produce 12 edges (4-connectivity)."""
        rng = np.random.default_rng(1)
        disp = rng.normal(0, 1, (10, 3, 3))
        lat, lon = _make_coords(3, 3)

        graph = DeformationGraph()
        graph.build_graph(disp, lat, lon, max_distance_m=200)

        assert graph.n_nodes == 9
        # 3x3 grid, 4-connectivity: 2*3 + 3*2 = 12 edges
        assert len(graph.edges) == 12

    def test_single_pixel(self):
        """A 1x1 grid should have 1 node and 0 edges."""
        disp = np.ones((5, 1, 1))
        lat = np.array([[28.3]])
        lon = np.array([[85.8]])

        graph = DeformationGraph()
        graph.build_graph(disp, lat, lon)

        assert graph.n_nodes == 1
        assert len(graph.edges) == 0

    def test_all_nan_excluded(self):
        """Pixels with all-NaN displacement are excluded as nodes."""
        disp = np.ones((10, 3, 3))
        disp[:, 1, 1] = np.nan  # center pixel is all NaN
        lat, lon = _make_coords(3, 3)

        graph = DeformationGraph()
        graph.build_graph(disp, lat, lon)

        assert graph.n_nodes == 8
        assert (1, 1) not in graph._pixel_to_node

    def test_correlated_neighbors_have_high_weight(self):
        """Neighbors sharing the same signal should have correlation ~1."""
        disp = np.zeros((20, 2, 1))
        trend = np.linspace(0, 1, 20)
        disp[:, 0, 0] = trend + np.random.default_rng(1).normal(0, 0.01, 20)
        disp[:, 1, 0] = trend + np.random.default_rng(2).normal(0, 0.01, 20)
        lat = np.array([[28.3], [28.2999]])
        lon = np.array([[85.8], [85.8]])

        graph = DeformationGraph()
        graph.build_graph(disp, lat, lon)

        assert len(graph.edges) == 1
        assert graph.edges[0, 2] > 0.95  # high correlation

    def test_empty_displacement(self):
        """All-NaN input yields an empty graph."""
        disp = np.full((5, 3, 3), np.nan)
        lat, lon = _make_coords(3, 3)

        graph = DeformationGraph()
        graph.build_graph(disp, lat, lon)

        assert graph.n_nodes == 0
        assert len(graph.edges) == 0


class TestCoherentClusterDetection:
    """Tests for detect_coherent_clusters."""

    def test_correlated_block_detected(self):
        """A block of correlated pixels forms a single cluster."""
        disp = _make_displacement_with_block()
        lat, lon = _make_coords()

        graph = DeformationGraph()
        graph.build_graph(disp, lat, lon)
        clusters = graph.detect_coherent_clusters(
            min_correlation=0.7, min_size=5,
        )

        # Should find at least one cluster
        assert len(clusters) >= 1
        # The largest cluster should contain the injected block
        largest = max(clusters, key=len)
        # The injected block is 7x7 = 49 pixels
        assert len(largest) >= 30  # most of the block should be captured

    def test_noise_only_no_clusters(self):
        """Pure random noise should produce no large coherent clusters."""
        rng = np.random.default_rng(77)
        disp = rng.normal(0, 0.01, (30, 15, 15))
        lat, lon = _make_coords(15, 15)

        graph = DeformationGraph()
        graph.build_graph(disp, lat, lon)
        clusters = graph.detect_coherent_clusters(
            min_correlation=0.7, min_size=5,
        )

        assert len(clusters) == 0

    def test_min_size_filter(self):
        """Clusters smaller than min_size are excluded."""
        # Create a small 3-pixel correlated patch
        rng = np.random.default_rng(10)
        disp = rng.normal(0, 0.01, (20, 10, 10))
        trend = np.linspace(0, 0.3, 20)
        for r, c in [(2, 2), (2, 3), (3, 2)]:
            disp[:, r, c] = trend + rng.normal(0, 0.002, 20)
        lat, lon = _make_coords(10, 10)

        graph = DeformationGraph()
        graph.build_graph(disp, lat, lon)

        # With min_size=5, the 3-pixel patch should be excluded
        clusters_strict = graph.detect_coherent_clusters(
            min_correlation=0.5, min_size=5,
        )
        # With min_size=2, it should be included
        clusters_loose = graph.detect_coherent_clusters(
            min_correlation=0.5, min_size=2,
        )

        # The 3-pixel patch should show up with min_size=2 but not 5
        small_in_loose = any(len(c) == 3 for c in clusters_loose)
        small_in_strict = any(len(c) == 3 for c in clusters_strict)
        assert small_in_loose or len(clusters_loose) > len(clusters_strict)
        assert not small_in_strict


class TestScoreClusterCoherence:
    """Tests for score_cluster_coherence."""

    def test_perfectly_correlated_block(self):
        """Identical time series across a block should score ~1.0."""
        disp = np.zeros((20, 5, 5))
        trend = np.linspace(0, 1, 20)
        for r in range(2, 5):
            for c in range(2, 5):
                disp[:, r, c] = trend

        lat, lon = _make_coords(5, 5)
        graph = DeformationGraph()
        graph.build_graph(disp, lat, lon)

        mask = np.zeros((5, 5), dtype=bool)
        mask[2:5, 2:5] = True

        score = graph.score_cluster_coherence(disp, mask)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_random_block_low_score(self):
        """Uncorrelated random pixels should score near 0."""
        rng = np.random.default_rng(55)
        disp = rng.normal(0, 1, (50, 5, 5))
        lat, lon = _make_coords(5, 5)
        graph = DeformationGraph()
        graph.build_graph(disp, lat, lon)

        mask = np.zeros((5, 5), dtype=bool)
        mask[1:4, 1:4] = True

        score = graph.score_cluster_coherence(disp, mask)
        assert abs(score) < 0.3

    def test_single_pixel_score(self):
        """A single-pixel cluster should return 1.0."""
        disp = np.ones((10, 3, 3))
        lat, lon = _make_coords(3, 3)
        graph = DeformationGraph()
        graph.build_graph(disp, lat, lon)

        mask = np.zeros((3, 3), dtype=bool)
        mask[1, 1] = True

        assert graph.score_cluster_coherence(disp, mask) == 1.0

    def test_empty_mask(self):
        """An empty mask should return NaN."""
        disp = np.ones((10, 3, 3))
        lat, lon = _make_coords(3, 3)
        graph = DeformationGraph()
        graph.build_graph(disp, lat, lon)

        mask = np.zeros((3, 3), dtype=bool)
        assert np.isnan(graph.score_cluster_coherence(disp, mask))


class TestNoiseFiltering:
    """Tests for filter_noise_clusters."""

    def test_removes_incoherent_clusters(self):
        """Clusters of random noise are filtered out."""
        rng = np.random.default_rng(33)
        disp = rng.normal(0, 0.01, (30, 20, 20))

        # Inject one coherent block and leave the rest as noise
        trend = np.linspace(0, 0.5, 30)
        for r in range(3, 10):
            for c in range(3, 10):
                disp[:, r, c] = trend + rng.normal(0, 0.003, 30)

        lat, lon = _make_coords()

        graph = DeformationGraph()
        graph.build_graph(disp, lat, lon)

        # Use a low correlation to get more raw clusters
        clusters = graph.detect_coherent_clusters(
            min_correlation=0.3, min_size=5,
        )

        # Filter with high coherence threshold
        surviving = graph.filter_noise_clusters(
            clusters, disp, min_coherence=0.7,
        )

        # The coherent block should survive; noise clusters should not
        assert len(surviving) >= 1
        # Check the surviving cluster has high coherence
        for _, score in surviving:
            assert score >= 0.7

    def test_keeps_coherent_clusters(self):
        """Highly correlated clusters survive noise filtering."""
        disp = _make_displacement_with_block()
        lat, lon = _make_coords()

        graph = DeformationGraph()
        graph.build_graph(disp, lat, lon)
        clusters = graph.detect_coherent_clusters(
            min_correlation=0.5, min_size=5,
        )

        surviving = graph.filter_noise_clusters(
            clusters, disp, min_coherence=0.5,
        )

        assert len(surviving) >= 1


class TestDetectSpatialAnomalies:
    """Tests for the main entry point."""

    def test_finds_injected_anomaly(self):
        """detect_spatial_anomalies finds the injected correlated block."""
        disp = _make_displacement_with_block()
        lat, lon = _make_coords()
        dates = np.arange(738886, 738886 + 30)

        anomalies = detect_spatial_anomalies(
            disp, dates, lat, lon,
            config={
                "spatial": {
                    "min_correlation": 0.7,
                    "min_cluster_size": 5,
                    "min_coherence": 0.5,
                }
            },
        )

        assert len(anomalies) >= 1
        best = anomalies[0]
        assert isinstance(best, SpatialAnomaly)
        assert best.coherence_score >= 0.5
        assert best.pixel_indices.shape[1] == 2
        assert best.area_m2 > 0

    def test_quiet_field_no_anomalies(self):
        """Pure noise produces no spatial anomalies."""
        rng = np.random.default_rng(88)
        disp = rng.normal(0, 0.01, (30, 15, 15))
        lat, lon = _make_coords(15, 15)
        dates = np.arange(738886, 738886 + 30)

        anomalies = detect_spatial_anomalies(disp, dates, lat, lon)
        assert len(anomalies) == 0

    def test_uniform_displacement(self):
        """Uniform (constant) displacement: all pixels correlate but
        zero variance means correlation is 0.0, so no clusters form."""
        disp = np.full((20, 10, 10), 0.5)
        lat, lon = _make_coords(10, 10)
        dates = np.arange(738886, 738886 + 20)

        anomalies = detect_spatial_anomalies(disp, dates, lat, lon)
        # Constant series → zero variance → correlation = 0
        assert len(anomalies) == 0

    def test_all_nan(self):
        """All-NaN displacement produces no anomalies."""
        disp = np.full((10, 5, 5), np.nan)
        lat, lon = _make_coords(5, 5)
        dates = np.arange(738886, 738886 + 10)

        anomalies = detect_spatial_anomalies(disp, dates, lat, lon)
        assert len(anomalies) == 0

    def test_default_config(self):
        """Works with no config (uses defaults)."""
        disp = _make_displacement_with_block()
        lat, lon = _make_coords()
        dates = np.arange(738886, 738886 + 30)

        # Should not raise
        anomalies = detect_spatial_anomalies(disp, dates, lat, lon)
        assert isinstance(anomalies, list)

    def test_anomaly_dataclass_fields(self):
        """SpatialAnomaly has all expected fields."""
        disp = _make_displacement_with_block()
        lat, lon = _make_coords()
        dates = np.arange(738886, 738886 + 30)

        anomalies = detect_spatial_anomalies(
            disp, dates, lat, lon,
            config={"spatial": {"min_correlation": 0.5, "min_cluster_size": 3, "min_coherence": 0.3}},
        )

        assert len(anomalies) >= 1
        a = anomalies[0]
        assert hasattr(a, "cluster_id")
        assert hasattr(a, "pixel_indices")
        assert hasattr(a, "centroid_lat")
        assert hasattr(a, "centroid_lon")
        assert hasattr(a, "coherence_score")
        assert hasattr(a, "mean_displacement")
        assert hasattr(a, "area_m2")
        assert a.centroid_lat > 0
        assert a.centroid_lon > 0

    def test_sorted_by_coherence(self):
        """Results are sorted by coherence_score descending."""
        disp = _make_displacement_with_block()
        lat, lon = _make_coords()
        dates = np.arange(738886, 738886 + 30)

        anomalies = detect_spatial_anomalies(
            disp, dates, lat, lon,
            config={"spatial": {"min_correlation": 0.3, "min_cluster_size": 3, "min_coherence": 0.3}},
        )

        if len(anomalies) >= 2:
            for i in range(len(anomalies) - 1):
                assert anomalies[i].coherence_score >= anomalies[i + 1].coherence_score
