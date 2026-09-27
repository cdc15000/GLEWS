"""
Graph-based spatial anomaly detection — filters single-pixel artifacts
by requiring spatially coherent deformation.

True pre-failure deformation affects spatially connected pixels.
Isolated pixel-level anomalies are more likely noise.  This module
builds an adjacency graph on the spatial deformation field and uses
connected-component analysis on thresholded correlation edges to
identify clusters of pixels with coherent (temporally correlated)
displacement patterns.

Design principles:
    - Pure numpy — no networkx or other graph libraries
    - 4-connectivity (or distance-based) pixel adjacency
    - Edge weights = temporal displacement correlation
    - Connected components on thresholded edges find real deformation
    - Isolated noisy pixels are filtered out naturally

Integration with detect.py:
    Run detect_spatial_anomalies() on raw displacement data to get a
    list of SpatialAnomaly objects.  Each anomaly's pixel_indices can
    be intersected with AnomalyFlag.pixel_indices from detect.py to
    confirm that a flagged site shows spatially coherent deformation,
    or used as an independent pre-filter before acceleration-based
    detection:

        spatial_anomalies = detect_spatial_anomalies(
            displacement_stack, dates, lat, lon,
        )
        # Keep only detect.py flags whose pixels overlap a coherent cluster
        confirmed = [
            flag for flag in anomaly_flags
            if any(
                _clusters_overlap(flag.pixel_indices, sa.pixel_indices)
                for sa in spatial_anomalies
            )
        ]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SpatialAnomaly:
    """A spatially coherent cluster of deforming pixels.

    Attributes
    ----------
    cluster_id : int
        Unique identifier for this cluster.
    pixel_indices : np.ndarray
        [n_pixels, 2] array of (row, col) indices into the
        displacement grid.
    centroid_lat : float
        Latitude of the cluster centroid.
    centroid_lon : float
        Longitude of the cluster centroid.
    coherence_score : float
        Mean pairwise temporal correlation within the cluster.
        1.0 = perfectly correlated, 0.0 = uncorrelated.
    mean_displacement : float
        Mean displacement (meters) across cluster pixels at the
        last epoch.
    area_m2 : float
        Approximate area of the cluster in square meters.
    """

    cluster_id: int
    pixel_indices: np.ndarray
    centroid_lat: float
    centroid_lon: float
    coherence_score: float
    mean_displacement: float
    area_m2: float


class DeformationGraph:
    """Adjacency graph over a displacement field for spatial filtering.

    Nodes are pixels with valid displacement data.  Edges connect
    spatially neighboring pixels (4-connectivity or distance-based),
    weighted by the temporal correlation of their displacement time
    series.

    The graph is stored as sparse adjacency lists using numpy arrays
    (no networkx dependency).
    """

    def __init__(self) -> None:
        # Adjacency: list of (i, j, weight) edges where i < j
        self.edges: np.ndarray = np.empty((0, 3), dtype=np.float64)
        # Number of nodes
        self.n_nodes: int = 0
        # Mapping from (row, col) to node index
        self._pixel_to_node: dict[tuple[int, int], int] = {}
        # Inverse mapping
        self._node_to_pixel: np.ndarray = np.empty((0, 2), dtype=int)

    def build_graph(
        self,
        displacement_map: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray,
        max_distance_m: float = 200.0,
    ) -> DeformationGraph:
        """Build an adjacency graph from a displacement stack.

        Parameters
        ----------
        displacement_map : np.ndarray
            Displacement time series, shape [n_epochs, n_rows, n_cols].
        lat, lon : np.ndarray
            Coordinate grids, shape [n_rows, n_cols].
        max_distance_m : float
            Maximum distance (meters) for edge connectivity.  When
            pixel spacing is smaller than this, 4-connectivity is
            used (immediate N/S/E/W neighbors).  When spacing is
            larger, distance-based connectivity includes all pixels
            within max_distance_m.

        Returns
        -------
        DeformationGraph
            self, for chaining.
        """
        n_epochs, n_rows, n_cols = displacement_map.shape

        # Identify valid pixels: must have at least 3 finite observations
        valid_count = np.sum(np.isfinite(displacement_map), axis=0)
        valid_mask = valid_count >= 3

        valid_positions = np.argwhere(valid_mask)  # [n_valid, 2]
        n_valid = len(valid_positions)

        if n_valid == 0:
            self.n_nodes = 0
            self._pixel_to_node = {}
            self._node_to_pixel = np.empty((0, 2), dtype=int)
            self.edges = np.empty((0, 3), dtype=np.float64)
            return self

        # Build node index mapping
        self._node_to_pixel = valid_positions
        self._pixel_to_node = {
            (int(r), int(c)): idx
            for idx, (r, c) in enumerate(valid_positions)
        }
        self.n_nodes = n_valid

        # Estimate pixel spacing to decide connectivity strategy
        pixel_spacing_m = _estimate_pixel_spacing(lat, lon)

        # Build edges: 4-connectivity when pixel spacing <= max_distance_m,
        # otherwise distance-based search
        edge_list: list[tuple[int, int]] = []

        if pixel_spacing_m <= max_distance_m:
            # 4-connectivity: check N, S, E, W neighbors
            for idx, (r, c) in enumerate(valid_positions):
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = int(r) + dr, int(c) + dc
                    neighbor_idx = self._pixel_to_node.get((nr, nc))
                    if neighbor_idx is not None and neighbor_idx > idx:
                        edge_list.append((idx, neighbor_idx))
        else:
            # Distance-based: all pairs within max_distance_m
            projected = _coords_are_projected(lat, lon)
            node_lats = lat[valid_positions[:, 0], valid_positions[:, 1]]
            node_lons = lon[valid_positions[:, 0], valid_positions[:, 1]]
            coords_m = _to_meters(node_lats, node_lons, projected)

            for i in range(n_valid):
                for j in range(i + 1, n_valid):
                    dist = np.sqrt(np.sum((coords_m[i] - coords_m[j]) ** 2))
                    if dist <= max_distance_m:
                        edge_list.append((i, j))

        # Compute edge weights = temporal correlation
        if not edge_list:
            self.edges = np.empty((0, 3), dtype=np.float64)
            return self

        edges_arr = np.array(edge_list, dtype=int)
        n_edges = len(edges_arr)
        weights = np.zeros(n_edges, dtype=np.float64)

        # Extract displacement time series for all valid nodes
        node_series = displacement_map[
            :, valid_positions[:, 0], valid_positions[:, 1]
        ]  # [n_epochs, n_valid]

        for e_idx in range(n_edges):
            i, j = edges_arr[e_idx]
            weights[e_idx] = _pearson_correlation(
                node_series[:, i], node_series[:, j]
            )

        self.edges = np.column_stack([
            edges_arr.astype(np.float64), weights
        ])

        logger.info(
            "Built deformation graph: %d nodes, %d edges, "
            "mean correlation=%.3f",
            self.n_nodes, n_edges,
            np.nanmean(weights) if n_edges > 0 else 0.0,
        )

        return self

    def detect_coherent_clusters(
        self,
        min_correlation: float = 0.7,
        min_size: int = 5,
    ) -> list[np.ndarray]:
        """Find clusters of coherently deforming pixels.

        Thresholds the graph edges by correlation and finds connected
        components on the surviving subgraph.  Only components with
        at least ``min_size`` pixels are returned.

        Parameters
        ----------
        min_correlation : float
            Minimum edge correlation to retain.
        min_size : int
            Minimum cluster size (number of pixels).

        Returns
        -------
        list[np.ndarray]
            Each element is an array of node indices belonging to
            one coherent cluster.
        """
        if self.n_nodes == 0:
            return []

        # Build adjacency from thresholded edges
        adj = _build_adjacency_list(self.n_nodes, self.edges, min_correlation)

        # Connected components via BFS
        components = _connected_components(self.n_nodes, adj)

        # Filter by size
        clusters = [c for c in components if len(c) >= min_size]

        logger.info(
            "Found %d coherent clusters (min_corr=%.2f, min_size=%d) "
            "from %d components",
            len(clusters), min_correlation, min_size, len(components),
        )

        return clusters

    def score_cluster_coherence(
        self,
        displacement_stack: np.ndarray,
        cluster_mask: np.ndarray,
    ) -> float:
        """Compute intra-cluster correlation score.

        Parameters
        ----------
        displacement_stack : np.ndarray
            Displacement time series, shape [n_epochs, n_rows, n_cols].
        cluster_mask : np.ndarray
            Boolean mask, shape [n_rows, n_cols], True for cluster pixels.

        Returns
        -------
        float
            Mean pairwise temporal correlation within the cluster.
            Returns 1.0 for single-pixel clusters, NaN if no valid
            pairs exist.
        """
        rows, cols = np.where(cluster_mask)
        n_pix = len(rows)

        if n_pix == 0:
            return np.nan
        if n_pix == 1:
            return 1.0

        series = displacement_stack[:, rows, cols]  # [n_epochs, n_pix]

        # Compute pairwise correlations
        correlations: list[float] = []
        for i in range(n_pix):
            for j in range(i + 1, n_pix):
                r = _pearson_correlation(series[:, i], series[:, j])
                if np.isfinite(r):
                    correlations.append(r)

        if not correlations:
            return np.nan

        return float(np.mean(correlations))

    def filter_noise_clusters(
        self,
        clusters: list[np.ndarray],
        displacement_stack: np.ndarray,
        min_coherence: float = 0.5,
    ) -> list[tuple[np.ndarray, float]]:
        """Remove clusters with incoherent internal displacement.

        Parameters
        ----------
        clusters : list[np.ndarray]
            Cluster node-index arrays from detect_coherent_clusters().
        displacement_stack : np.ndarray
            Displacement time series, shape [n_epochs, n_rows, n_cols].
        min_coherence : float
            Minimum mean pairwise correlation to keep a cluster.

        Returns
        -------
        list[tuple[np.ndarray, float]]
            Surviving (cluster_node_indices, coherence_score) pairs.
        """
        surviving: list[tuple[np.ndarray, float]] = []

        for cluster_nodes in clusters:
            # Convert node indices to (row, col)
            pixels = self._node_to_pixel[cluster_nodes]
            mask = np.zeros(displacement_stack.shape[1:], dtype=bool)
            mask[pixels[:, 0], pixels[:, 1]] = True

            score = self.score_cluster_coherence(displacement_stack, mask)

            if np.isfinite(score) and score >= min_coherence:
                surviving.append((cluster_nodes, score))
            else:
                logger.debug(
                    "Filtered cluster of %d pixels (coherence=%.3f < %.3f)",
                    len(cluster_nodes), score if np.isfinite(score) else 0.0,
                    min_coherence,
                )

        logger.info(
            "Noise filtering: %d/%d clusters survived (min_coherence=%.2f)",
            len(surviving), len(clusters), min_coherence,
        )

        return surviving


def detect_spatial_anomalies(
    displacement_stack: np.ndarray,
    dates: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    config: dict | None = None,
) -> list[SpatialAnomaly]:
    """Detect spatially coherent deformation anomalies.

    Main entry point for graph-based spatial anomaly detection.
    Builds an adjacency graph from the displacement field, finds
    coherent clusters, filters noise, and returns a list of
    SpatialAnomaly objects.

    Parameters
    ----------
    displacement_stack : np.ndarray
        Displacement time series in meters,
        shape [n_epochs, n_rows, n_cols].
    dates : np.ndarray
        Acquisition dates as ordinal days, shape [n_epochs].
        Reserved for future use (e.g. temporal weighting); not
        currently used in the spatial analysis.
    lat, lon : np.ndarray
        Coordinate grids, shape [n_rows, n_cols].
    config : dict, optional
        Configuration overrides.  Recognized keys (all under a
        top-level ``"spatial"`` key):

        - ``max_distance_m`` (float): max edge distance, default 200.
        - ``min_correlation`` (float): edge threshold, default 0.7.
        - ``min_cluster_size`` (int): minimum pixels, default 5.
        - ``min_coherence`` (float): intra-cluster filter, default 0.5.

    Returns
    -------
    list[SpatialAnomaly]
        Detected anomalies sorted by coherence score (highest first).
    """
    cfg = (config or {}).get("spatial", {})
    max_distance_m = cfg.get("max_distance_m", 200.0)
    min_correlation = cfg.get("min_correlation", 0.7)
    min_cluster_size = cfg.get("min_cluster_size", 5)
    min_coherence = cfg.get("min_coherence", 0.5)

    logger.info(
        "Spatial anomaly detection: %d epochs, %dx%d grid, "
        "max_dist=%dm, min_corr=%.2f, min_size=%d, min_coh=%.2f",
        displacement_stack.shape[0],
        displacement_stack.shape[1],
        displacement_stack.shape[2],
        max_distance_m,
        min_correlation,
        min_cluster_size,
        min_coherence,
    )

    graph = DeformationGraph()
    graph.build_graph(displacement_stack, lat, lon, max_distance_m)

    clusters = graph.detect_coherent_clusters(min_correlation, min_cluster_size)
    surviving = graph.filter_noise_clusters(
        clusters, displacement_stack, min_coherence,
    )

    pixel_area = _estimate_pixel_area(lat, lon)

    anomalies: list[SpatialAnomaly] = []
    for cluster_id, (cluster_nodes, coherence) in enumerate(surviving):
        pixels = graph._node_to_pixel[cluster_nodes]
        rows, cols = pixels[:, 0], pixels[:, 1]

        # Mean displacement at the last epoch
        last_epoch = displacement_stack[-1, rows, cols]
        mean_disp = float(np.nanmean(last_epoch))

        anomaly = SpatialAnomaly(
            cluster_id=cluster_id,
            pixel_indices=pixels.copy(),
            centroid_lat=float(np.mean(lat[rows, cols])),
            centroid_lon=float(np.mean(lon[rows, cols])),
            coherence_score=coherence,
            mean_displacement=mean_disp,
            area_m2=float(len(rows) * pixel_area),
        )
        anomalies.append(anomaly)

    anomalies.sort(key=lambda a: a.coherence_score, reverse=True)

    logger.info(
        "Spatial anomaly detection complete: %d anomalies detected",
        len(anomalies),
    )

    return anomalies


# -------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------

def _pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two 1-D arrays, ignoring NaN pairs."""
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3:
        return np.nan

    x = a[valid]
    y = b[valid]

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    xd = x - x_mean
    yd = y - y_mean

    denom = np.sqrt(np.sum(xd ** 2) * np.sum(yd ** 2))
    if denom < 1e-15:
        return 0.0

    return float(np.sum(xd * yd) / denom)


def _build_adjacency_list(
    n_nodes: int,
    edges: np.ndarray,
    min_weight: float,
) -> list[list[int]]:
    """Build adjacency list from edges above a weight threshold."""
    adj: list[list[int]] = [[] for _ in range(n_nodes)]

    for e in range(len(edges)):
        i, j, w = int(edges[e, 0]), int(edges[e, 1]), edges[e, 2]
        if np.isfinite(w) and w >= min_weight:
            adj[i].append(j)
            adj[j].append(i)

    return adj


def _connected_components(
    n_nodes: int,
    adj: list[list[int]],
) -> list[np.ndarray]:
    """Find connected components via iterative BFS."""
    visited = np.zeros(n_nodes, dtype=bool)
    components: list[np.ndarray] = []

    for start in range(n_nodes):
        if visited[start]:
            continue

        # BFS
        component: list[int] = []
        queue = [start]
        visited[start] = True

        while queue:
            node = queue.pop(0)
            component.append(node)
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

        components.append(np.array(component, dtype=int))

    return components


def _coords_are_projected(lat: np.ndarray, lon: np.ndarray) -> bool:
    """Detect whether coordinates are projected (meters) vs geographic (degrees)."""
    lat_range = np.nanmax(lat) - np.nanmin(lat)
    lon_range = np.nanmax(lon) - np.nanmin(lon)
    return lat_range > 1000 or lon_range > 1000


def _to_meters(
    lats: np.ndarray, lons: np.ndarray, projected: bool,
) -> np.ndarray:
    """Convert coordinate arrays to [N, 2] meter positions."""
    if projected:
        return np.column_stack([lats, lons])
    lat_m = lats * 111_320
    lon_m = lons * 111_320 * np.cos(np.radians(np.mean(lats)))
    return np.column_stack([lat_m, lon_m])


def _estimate_pixel_spacing(lat: np.ndarray, lon: np.ndarray) -> float:
    """Estimate spacing between adjacent pixels in meters."""
    if lat.shape[0] < 2 or lat.shape[1] < 2:
        return 30.0  # default Sentinel-1

    if _coords_are_projected(lat, lon):
        dlat = abs(float(lat[1, 0] - lat[0, 0]))
        dlon = abs(float(lon[0, 1] - lon[0, 0]))
        return max(dlat, dlon) if max(dlat, dlon) > 0 else 30.0

    dlat = abs(float(lat[1, 0] - lat[0, 0]))
    dlon = abs(float(lon[0, 1] - lon[0, 0]))
    mean_lat = float(np.nanmean(lat))
    lat_m = dlat * 111_320
    lon_m = dlon * 111_320 * np.cos(np.radians(mean_lat))
    return max(lat_m, lon_m) if max(lat_m, lon_m) > 0 else 30.0


def _estimate_pixel_area(lat: np.ndarray, lon: np.ndarray) -> float:
    """Estimate area of a single pixel in m^2."""
    if lat.shape[0] < 2 or lat.shape[1] < 2:
        return 900.0

    if _coords_are_projected(lat, lon):
        dlat = abs(float(lat[1, 0] - lat[0, 0]))
        dlon = abs(float(lon[0, 1] - lon[0, 0]))
        return dlat * dlon if dlat * dlon > 0 else 900.0

    dlat = abs(float(lat[1, 0] - lat[0, 0]))
    dlon = abs(float(lon[0, 1] - lon[0, 0]))
    mean_lat = float(np.nanmean(lat))
    lat_m = dlat * 111_320
    lon_m = dlon * 111_320 * np.cos(np.radians(mean_lat))
    return lat_m * lon_m if lat_m * lon_m > 0 else 900.0
