"""Tests for the performance benchmarking module."""

import numpy as np
import pytest

from glews.benchmark import BenchmarkResult, PipelineBenchmark, _make_synthetic_cube


class TestBenchmarkResult:
    """Tests for the BenchmarkResult dataclass."""

    def test_fields(self):
        r = BenchmarkResult(
            stage_name="timeseries",
            n_pixels=1000,
            n_dates=50,
            elapsed_seconds=1.5,
            pixels_per_second=666.7,
            memory_mb=12.3,
        )
        assert r.stage_name == "timeseries"
        assert r.n_pixels == 1000
        assert r.n_dates == 50
        assert r.elapsed_seconds == 1.5
        assert r.pixels_per_second == pytest.approx(666.7)
        assert r.memory_mb == pytest.approx(12.3)

    def test_equality(self):
        kwargs = dict(
            stage_name="detection",
            n_pixels=500,
            n_dates=30,
            elapsed_seconds=0.5,
            pixels_per_second=1000.0,
            memory_mb=5.0,
        )
        assert BenchmarkResult(**kwargs) == BenchmarkResult(**kwargs)


class TestMakeSyntheticCube:
    """Tests for the internal synthetic data helper."""

    def test_shape(self):
        dates, disp = _make_synthetic_cube(100, 20)
        n_rows, n_cols = disp.shape[1], disp.shape[2]
        assert n_rows * n_cols >= 90  # at least close to requested
        assert disp.shape[0] == 20
        assert len(dates) == 20

    def test_finite(self):
        _, disp = _make_synthetic_cube(50, 10)
        assert np.all(np.isfinite(disp))


class TestPipelineBenchmark:
    """Tests for PipelineBenchmark benchmark runs."""

    def test_benchmark_timeseries_small(self):
        bench = PipelineBenchmark()
        results = bench.benchmark_timeseries(n_pixels_list=[100], n_dates=15)
        assert len(results) == 1
        r = results[0]
        assert r.stage_name == "timeseries"
        assert r.elapsed_seconds > 0
        assert r.pixels_per_second > 0
        assert r.memory_mb >= 0
        assert r.n_dates == 15

    def test_benchmark_step_change_small(self):
        bench = PipelineBenchmark()
        results = bench.benchmark_step_change(n_pixels_list=[100], n_dates=15)
        assert len(results) == 1
        r = results[0]
        assert r.stage_name == "step_change"
        assert r.elapsed_seconds > 0
        assert r.pixels_per_second > 0

    def test_benchmark_bocpd_small(self):
        bench = PipelineBenchmark()
        results = bench.benchmark_bocpd(n_pixels_list=[100], n_dates=15)
        assert len(results) == 1
        r = results[0]
        assert r.stage_name == "bocpd"
        assert r.elapsed_seconds > 0
        assert r.pixels_per_second > 0

    def test_benchmark_detection_small(self):
        bench = PipelineBenchmark()
        results = bench.benchmark_detection(n_pixels_list=[100], n_dates=15)
        assert len(results) == 1
        r = results[0]
        assert r.stage_name == "detection"
        assert r.elapsed_seconds > 0

    def test_results_accumulate(self):
        bench = PipelineBenchmark()
        bench.benchmark_timeseries(n_pixels_list=[100], n_dates=15)
        bench.benchmark_step_change(n_pixels_list=[100], n_dates=15)
        assert len(bench.results) == 2
        stages = {r.stage_name for r in bench.results}
        assert stages == {"timeseries", "step_change"}

    def test_estimate_global_runtime(self):
        bench = PipelineBenchmark()
        bench.benchmark_timeseries(n_pixels_list=[100], n_dates=15)
        est = bench.estimate_global_runtime(
            n_glaciers=100, avg_pixels_per_glacier=100, n_dates=15,
        )
        assert est["n_glaciers"] == 100
        assert est["total_pixels"] == 10_000
        assert "timeseries" in est["estimated_hours"]
        assert est["estimated_hours"]["total"] > 0

    def test_estimate_empty(self):
        bench = PipelineBenchmark()
        est = bench.estimate_global_runtime()
        assert est["estimated_hours"] == {"total": 0}


class TestGenerateReport:
    """Tests for report formatting."""

    def test_empty_report(self):
        bench = PipelineBenchmark()
        report = bench.generate_report()
        assert "No benchmark results" in report

    def test_report_has_header_and_data(self):
        bench = PipelineBenchmark()
        bench.benchmark_timeseries(n_pixels_list=[100], n_dates=15)
        report = bench.generate_report()
        assert "GEWS Pipeline Performance Benchmarks" in report
        assert "Stage" in report
        assert "timeseries" in report
        assert "px/s" in report

    def test_report_includes_global_estimate(self):
        bench = PipelineBenchmark()
        bench.benchmark_timeseries(n_pixels_list=[100], n_dates=50)
        report = bench.generate_report()
        assert "Global Screening Estimate" in report
        assert "215,000" in report
        assert "hours" in report
