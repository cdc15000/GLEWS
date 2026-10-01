"""
Performance benchmarking for the GLEWS pipeline.

Measures throughput and memory usage at different scales to estimate
global screening time for ~215,000 glaciers.  Each benchmark generates
synthetic data of the requested size, runs the relevant pipeline stage,
and records wall-clock time plus peak resident memory.
"""

from __future__ import annotations

import logging
import time
import tracemalloc
from dataclasses import dataclass

import numpy as np

from glews.timeseries import (
    bocpd_changepoints,
    compute_acceleration_map,
    detect_step_changes,
)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Timing and throughput result for a single benchmark run."""

    stage_name: str
    n_pixels: int
    n_dates: int
    elapsed_seconds: float
    pixels_per_second: float
    memory_mb: float


def _make_synthetic_cube(
    n_pixels: int,
    n_dates: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (dates, displacement) arrays for benchmarking.

    ``displacement`` has shape ``[n_dates, n_rows, n_cols]`` where
    ``n_rows * n_cols == n_pixels`` (closest square layout).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    n_cols = max(1, int(np.sqrt(n_pixels)))
    n_rows = max(1, n_pixels // n_cols)

    dates = np.arange(738886, 738886 + n_dates * 12, 12)[:n_dates]
    t_yr = (dates - dates[0]).astype(float) / 365.25

    velocity = 0.05  # m/yr
    displacement = np.empty((n_dates, n_rows, n_cols))
    for i in range(n_dates):
        displacement[i] = velocity * t_yr[i] + rng.normal(
            0, 0.003, (n_rows, n_cols)
        )

    return dates, displacement


class PipelineBenchmark:
    """Run and collect benchmarks for each pipeline stage."""

    def __init__(self) -> None:
        self.results: list[BenchmarkResult] = []

    # ------------------------------------------------------------------
    # Individual stage benchmarks
    # ------------------------------------------------------------------

    def benchmark_timeseries(
        self,
        n_pixels_list: list[int] | None = None,
        n_dates: int = 50,
    ) -> list[BenchmarkResult]:
        """Time the acceleration-map computation at different pixel counts."""
        if n_pixels_list is None:
            n_pixels_list = [1000, 5000, 10000, 50000]

        results: list[BenchmarkResult] = []
        rng = np.random.default_rng(1)

        for n_pixels in n_pixels_list:
            dates, disp = _make_synthetic_cube(n_pixels, n_dates, rng)
            actual_pixels = disp.shape[1] * disp.shape[2]

            tracemalloc.start()
            t0 = time.perf_counter()

            compute_acceleration_map(dates, disp)

            elapsed = time.perf_counter() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            r = BenchmarkResult(
                stage_name="timeseries",
                n_pixels=actual_pixels,
                n_dates=n_dates,
                elapsed_seconds=elapsed,
                pixels_per_second=actual_pixels / elapsed if elapsed > 0 else 0,
                memory_mb=peak / 1024 / 1024,
            )
            results.append(r)
            self.results.append(r)
            logger.info(
                "timeseries  %6d px x %d dates  %.2fs  %.0f px/s  %.1f MB",
                actual_pixels, n_dates, elapsed, r.pixels_per_second, r.memory_mb,
            )

        return results

    def benchmark_detection(
        self,
        n_pixels_list: list[int] | None = None,
        n_dates: int = 50,
    ) -> list[BenchmarkResult]:
        """Time the full detection pipeline (acceleration + step-change)."""
        if n_pixels_list is None:
            n_pixels_list = [1000, 5000, 10000, 50000]

        results: list[BenchmarkResult] = []
        rng = np.random.default_rng(2)

        for n_pixels in n_pixels_list:
            dates, disp = _make_synthetic_cube(n_pixels, n_dates, rng)
            actual_pixels = disp.shape[1] * disp.shape[2]

            tracemalloc.start()
            t0 = time.perf_counter()

            compute_acceleration_map(dates, disp)
            detect_step_changes(dates, disp)

            elapsed = time.perf_counter() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            r = BenchmarkResult(
                stage_name="detection",
                n_pixels=actual_pixels,
                n_dates=n_dates,
                elapsed_seconds=elapsed,
                pixels_per_second=actual_pixels / elapsed if elapsed > 0 else 0,
                memory_mb=peak / 1024 / 1024,
            )
            results.append(r)
            self.results.append(r)
            logger.info(
                "detection   %6d px x %d dates  %.2fs  %.0f px/s  %.1f MB",
                actual_pixels, n_dates, elapsed, r.pixels_per_second, r.memory_mb,
            )

        return results

    def benchmark_bocpd(
        self,
        n_pixels_list: list[int] | None = None,
        n_dates: int = 50,
    ) -> list[BenchmarkResult]:
        """Time BOCPD changepoint detection (the most expensive detector).

        BOCPD runs per-pixel in a Python loop, so this benchmark uses a
        representative sample of pixels rather than the full grid.
        """
        if n_pixels_list is None:
            n_pixels_list = [1000, 5000, 10000, 50000]

        results: list[BenchmarkResult] = []
        rng = np.random.default_rng(3)

        for n_pixels in n_pixels_list:
            # BOCPD is O(n_dates^2) per pixel and runs in pure Python,
            # so we benchmark a manageable sample and extrapolate.
            sample_size = min(n_pixels, 50)
            dates = np.arange(738886, 738886 + n_dates * 12, 12)[:n_dates]
            t_yr = (dates - dates[0]).astype(float) / 365.25

            tracemalloc.start()
            t0 = time.perf_counter()

            for _ in range(sample_size):
                disp_1d = 0.05 * t_yr + rng.normal(0, 0.003, n_dates)
                bocpd_changepoints(dates, disp_1d)

            elapsed_sample = time.perf_counter() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Extrapolate to the full pixel count
            elapsed = elapsed_sample * (n_pixels / sample_size)

            r = BenchmarkResult(
                stage_name="bocpd",
                n_pixels=n_pixels,
                n_dates=n_dates,
                elapsed_seconds=elapsed,
                pixels_per_second=n_pixels / elapsed if elapsed > 0 else 0,
                memory_mb=peak / 1024 / 1024,
            )
            results.append(r)
            self.results.append(r)
            logger.info(
                "bocpd       %6d px x %d dates  %.2fs (extrapolated)  %.0f px/s  %.1f MB",
                n_pixels, n_dates, elapsed, r.pixels_per_second, r.memory_mb,
            )

        return results

    def benchmark_step_change(
        self,
        n_pixels_list: list[int] | None = None,
        n_dates: int = 50,
    ) -> list[BenchmarkResult]:
        """Time step-change detection at different pixel counts."""
        if n_pixels_list is None:
            n_pixels_list = [1000, 5000, 10000, 50000]

        results: list[BenchmarkResult] = []
        rng = np.random.default_rng(4)

        for n_pixels in n_pixels_list:
            dates, disp = _make_synthetic_cube(n_pixels, n_dates, rng)
            actual_pixels = disp.shape[1] * disp.shape[2]

            tracemalloc.start()
            t0 = time.perf_counter()

            detect_step_changes(dates, disp)

            elapsed = time.perf_counter() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            r = BenchmarkResult(
                stage_name="step_change",
                n_pixels=actual_pixels,
                n_dates=n_dates,
                elapsed_seconds=elapsed,
                pixels_per_second=actual_pixels / elapsed if elapsed > 0 else 0,
                memory_mb=peak / 1024 / 1024,
            )
            results.append(r)
            self.results.append(r)
            logger.info(
                "step_change %6d px x %d dates  %.2fs  %.0f px/s  %.1f MB",
                actual_pixels, n_dates, elapsed, r.pixels_per_second, r.memory_mb,
            )

        return results

    # ------------------------------------------------------------------
    # Global runtime estimation
    # ------------------------------------------------------------------

    def estimate_global_runtime(
        self,
        n_glaciers: int = 215_000,
        avg_pixels_per_glacier: int = 1000,
        n_dates: int = 50,
    ) -> dict:
        """Extrapolate from collected benchmarks to estimate global screening time.

        Returns a dict keyed by stage name with estimated hours and
        a ``"total"`` entry.
        """
        total_pixels = n_glaciers * avg_pixels_per_glacier

        estimates: dict[str, float] = {}
        for stage in ("timeseries", "detection", "step_change", "bocpd"):
            stage_results = [
                r for r in self.results
                if r.stage_name == stage and r.n_dates == n_dates
            ]
            if not stage_results:
                continue

            # Use the largest-scale result for the most representative rate
            best = max(stage_results, key=lambda r: r.n_pixels)
            rate = best.pixels_per_second
            if rate > 0:
                seconds = total_pixels / rate
                estimates[stage] = seconds / 3600  # hours
            else:
                estimates[stage] = float("inf")

        estimates["total"] = sum(estimates.values())

        return {
            "n_glaciers": n_glaciers,
            "avg_pixels_per_glacier": avg_pixels_per_glacier,
            "total_pixels": total_pixels,
            "n_dates": n_dates,
            "estimated_hours": estimates,
        }

    # ------------------------------------------------------------------
    # Report formatting
    # ------------------------------------------------------------------

    def generate_report(self) -> str:
        """Format all collected results as a readable text table."""
        if not self.results:
            return "No benchmark results collected."

        lines: list[str] = []
        lines.append("")
        lines.append("GLEWS Pipeline Performance Benchmarks")
        lines.append("=" * 72)

        header = (
            f"{'Stage':<14} {'Pixels':>8} {'Dates':>6} "
            f"{'Time (s)':>10} {'px/s':>12} {'Mem (MB)':>10}"
        )
        lines.append(header)
        lines.append("-" * 72)

        for r in self.results:
            lines.append(
                f"{r.stage_name:<14} {r.n_pixels:>8,} {r.n_dates:>6} "
                f"{r.elapsed_seconds:>10.3f} {r.pixels_per_second:>12,.0f} "
                f"{r.memory_mb:>10.1f}"
            )

        lines.append("-" * 72)

        # Add global estimate if we have enough data
        est = self.estimate_global_runtime()
        if est["estimated_hours"]:
            lines.append("")
            lines.append("Global Screening Estimate")
            lines.append(
                f"  {est['n_glaciers']:,} glaciers x "
                f"{est['avg_pixels_per_glacier']:,} px/glacier = "
                f"{est['total_pixels']:,} pixels"
            )
            for stage, hours in est["estimated_hours"].items():
                if hours == float("inf"):
                    lines.append(f"  {stage:<14}  (no data)")
                else:
                    lines.append(f"  {stage:<14}  {hours:>8.1f} hours")

        lines.append("")
        return "\n".join(lines)
