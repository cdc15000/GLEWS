"""
GEWS command-line interface.

Usage:
    gews search   --config CONFIG     Search for available Sentinel-1 scenes
    gews download --config CONFIG     Download SLC scenes from ASF
    gews process  --config CONFIG     Run InSAR processing (ISCE-2 + MintPy)
    gews detect   --config CONFIG     Run Tier 0 anomaly detection
    gews assess   --config CONFIG     Run Tier 1 cascade risk assessment
    gews report   --config CONFIG     Generate analysis report
    gews monitor  --config CONFIG     Continuous monitoring for new NISAR data
    gews dashboard --config CONFIG    Launch Tier 2 analyst review dashboard
    gews map       --data-dir DIR     Launch interactive GeoJSON map viewer
    gews demo                         Run full pipeline on synthetic data
    gews train                        Train precursor classifier model
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import yaml

logger = logging.getLogger("gews")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        click.echo(f"Configuration file not found: {path}", err=True)
        sys.exit(1)

    with open(path) as f:
        config = yaml.safe_load(f)

    # Run validation and emit warnings (non-blocking)
    from gews.validate import validate_config

    issues = validate_config(config) if isinstance(config, dict) else []
    for issue in issues:
        if issue.startswith("WARNING:"):
            logger.warning(issue.removeprefix("WARNING: "))
        elif issue.startswith("ERROR:"):
            logger.warning("Config issue: %s", issue.removeprefix("ERROR: "))

    return config


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging")
def main(verbose: bool) -> None:
    """GEWS — Glacier Early Warning System for unstable glacier detection."""
    _setup_logging(verbose)


@main.command()
@click.option("--config", "-c", required=True, help="Path to site config YAML")
@click.option("--site", "-s", default=None, help="Site name filter (substring match; for multi-site configs)")
def search(config: str, site: str | None) -> None:
    """Search ASF archive for Sentinel-1 scenes covering the study area.

    Works with both single-site configs (e.g., nepal_2026.yaml) and
    multi-site configs (e.g., hkh_priority.yaml, global_watch.yaml).
    Use --site to filter multi-site configs by name.
    """
    from gews.acquire import search_scenes, select_track, summarize_scenes
    from gews.monitor import iter_site_configs

    site_configs = list(iter_site_configs(config))

    if site:
        site_configs = [
            c for c in site_configs
            if site.lower() in c["site"]["name"].lower()
        ]
        if not site_configs:
            click.echo(f"No sites matching '{site}' in {config}", err=True)
            sys.exit(1)

    for i, cfg in enumerate(site_configs):
        site_name = cfg["site"]["name"]
        if len(site_configs) > 1:
            click.echo(f"\n{'═' * 60}")
            click.echo(f"  [{i + 1}/{len(site_configs)}] {site_name}")
            click.echo(f"{'═' * 60}")

        try:
            scenes = search_scenes(cfg)
            click.echo(summarize_scenes(scenes))

            # Show recommended track
            track = select_track(scenes, cfg["acquire"].get("path_number"))
            click.echo(f"\nRecommended: {len(track)} scenes on selected track")
        except Exception as e:
            click.echo(f"  Error searching {site_name}: {e}", err=True)


@main.command()
@click.option("--config", "-c", required=True, help="Path to site config YAML")
@click.option("--site", "-s", default=None, help="Site name filter (substring match)")
def download(config: str, site: str | None) -> None:
    """Download SLC scenes from ASF DAAC.

    Works with both single-site and multi-site configs.
    Use --site to filter by name.
    """
    from gews.acquire import search_scenes, select_track, download_scenes
    from gews.monitor import iter_site_configs

    site_configs = list(iter_site_configs(config))
    if site:
        site_configs = [
            c for c in site_configs
            if site.lower() in c["site"]["name"].lower()
        ]
        if not site_configs:
            click.echo(f"No sites matching '{site}' in {config}", err=True)
            sys.exit(1)

    for cfg in site_configs:
        site_name = cfg["site"]["name"]
        acq = cfg.get("acquire", {})
        click.echo(f"\nDownloading: {site_name}")

        try:
            scenes = search_scenes(cfg)
            track = select_track(scenes, acq.get("path_number"))

            # Route products to the correct directory
            product_types = {s.product_type for s in track}
            for pt in sorted(product_types):
                pt_scenes = [s for s in track if s.product_type == pt]
                if pt == "GUNW":
                    out = acq.get("gunw_dir", "data/gunw")
                elif pt == "GOFF":
                    out = acq.get("goff_dir", "data/goff")
                else:
                    out = acq.get("output_dir", "data/slc")
                download_scenes(
                    pt_scenes,
                    output_dir=out,
                    n_workers=acq.get("n_workers", 4),
                )
        except Exception as e:
            click.echo(f"  Error downloading {site_name}: {e}", err=True)


@main.command()
@click.option("--config", "-c", required=True, help="Path to site config YAML")
@click.option("--dry-run", is_flag=True, help="Print commands without executing")
@click.option(
    "--step",
    type=click.Choice(["prepare", "isce", "mintpy", "all"]),
    default="all",
    help="Which processing step to run",
)
def process(config: str, dry_run: bool, step: str) -> None:
    """Run InSAR processing pipeline (ISCE-2 + MintPy)."""
    from gews.process import InSARProcessor

    cfg = _load_config(config)
    proc = InSARProcessor(cfg)

    if step in ("prepare", "all"):
        proc.prepare_stack()

    if step in ("isce", "all"):
        proc.run_isce(dry_run=dry_run)

    if step in ("mintpy", "all"):
        if step == "mintpy":
            proc.generate_mintpy_config()
        proc.run_mintpy(dry_run=dry_run)

    click.echo("Processing complete.")


@main.command()
@click.option("--config", "-c", required=True, help="Path to site config YAML")
@click.option("--output", "-o", default="output", help="Output directory")
def detect(config: str, output: str) -> None:
    """Run Tier 0 anomaly detection on processed InSAR data."""
    from gews.detect import detect_anomalies
    from gews.process import InSARProcessor
    from gews.timeseries import compute_acceleration_map

    cfg = _load_config(config)

    # Load processed time series
    proc = InSARProcessor(cfg)
    ts = proc.load_timeseries()

    # Compute acceleration
    det_cfg = cfg["detect"]
    accel_map = compute_acceleration_map(
        ts.dates,
        ts.displacement,
        window_size_days=det_cfg["acceleration"]["window_size_days"],
        step_days=det_cfg["acceleration"]["step_days"],
        n_harmonics=det_cfg.get("n_harmonics", 2),
    )

    # Detect anomalies
    flags = detect_anomalies(accel_map, ts.latitude, ts.longitude, cfg)

    click.echo(f"\nTier 0 complete: {len(flags)} flags detected")
    for f in flags[:10]:
        click.echo(
            f"  Flag {f.flag_id}: score={f.score:.2f}, "
            f"z={f.peak_zscore:.1f}, "
            f"area={f.area_m2:,.0f} m², "
            f"({f.center_lat:.4f}°N, {f.center_lon:.4f}°E)"
        )


@main.command()
@click.option("--config", "-c", required=True, help="Path to site config YAML")
def assess(config: str) -> None:
    """Run Tier 1 cascade risk assessment."""
    click.echo("Tier 1 assessment requires DEM and population data.")
    click.echo("Use 'gews demo' for a complete demonstration with synthetic data.")


@main.command()
@click.option("--output", "-o", default="output", help="Output directory")
@click.option("--no-plots", is_flag=True, help="Skip generating plots")
def demo(output: str, no_plots: bool) -> None:
    """
    Run the full GEWS pipeline on synthetic data.

    Generates a realistic synthetic InSAR scene with an injected
    pre-failure acceleration signal, then runs Tier 0 detection
    and Tier 1 filtering. No real satellite data or processing
    tools required.

    This demonstrates the anomaly detection and cascade assessment
    algorithms on data mimicking the Nepal 2026 scenario.
    """
    from gews.detect import detect_anomalies
    from gews.report import generate_report
    from gews.synthetic import generate_synthetic_scene
    from gews.timeseries import compute_acceleration_map

    click.echo("=" * 60)
    click.echo("GEWS Demo — Synthetic Nepal 2026 Scenario")
    click.echo("=" * 60)
    click.echo()

    # Load default config for detection parameters
    config_path = Path(__file__).parent.parent.parent / "config" / "nepal_2026.yaml"
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        # Minimal inline config
        cfg = {
            "site": {
                "name": "Nepal-Tibet Border 2026 (Synthetic)",
                "event_date": "2026-08-26",
            },
            "detect": {
                "n_harmonics": 2,
                "acceleration": {
                    "window_size_days": 60,
                    "step_days": 12,
                    "sigma_threshold": 2.5,
                    "changepoint": {
                        "model": "rbf",
                        "penalty": "bic",
                        "min_segment_size": 3,
                    },
                },
                "clustering": {
                    "min_cluster_pixels": 5,
                    "max_distance_m": 200,
                    "min_area_m2": 10000,
                },
                "voight": {
                    "enabled": True,
                    "min_points": 5,
                    "r_squared_threshold": 0.7,
                },
            },
        }
    cfg["site"]["name"] = "Nepal-Tibet Border 2026 (Synthetic Demo)"

    # Step 1: Generate synthetic data
    click.echo("Step 1/3  Generating synthetic InSAR data...")
    ts = generate_synthetic_scene()
    click.echo(
        f"  {len(ts.date_strings)} dates × "
        f"{ts.displacement.shape[1]}×{ts.displacement.shape[2]} pixels"
    )
    click.echo(
        f"  Date range: {ts.date_strings[0]} to {ts.date_strings[-1]}"
    )
    click.echo()

    # Step 2: Compute acceleration and detect anomalies
    click.echo("Step 2/3  Running anomaly detection...")
    det_cfg = cfg["detect"]
    accel_map = compute_acceleration_map(
        ts.dates,
        ts.displacement,
        window_size_days=det_cfg["acceleration"]["window_size_days"],
        step_days=det_cfg["acceleration"]["step_days"],
        n_harmonics=det_cfg.get("n_harmonics", 2),
    )

    flags = detect_anomalies(accel_map, ts.latitude, ts.longitude, cfg)
    click.echo(f"  Detected {len(flags)} anomaly flags")
    click.echo()

    # Step 3: Report
    if flags and not no_plots:
        click.echo("Step 3/3  Generating report...")
        report_path = generate_report(
            flags=flags,
            assessments=None,  # no DEM for cascade assessment in demo
            accel_map=accel_map,
            latitude=ts.latitude,
            longitude=ts.longitude,
            config=cfg,
            output_dir=output,
        )
        click.echo(f"  Report: {report_path}")
        click.echo(f"  Figures: {Path(output) / 'figures'}")
        click.echo(f"  GeoJSON: {Path(output) / 'flags.geojson'}")
    elif not flags:
        click.echo("  No flags detected (adjust sigma_threshold if needed)")
    else:
        click.echo("Step 3/3  Skipping plots (--no-plots)")

    # Summary
    click.echo()
    click.echo("=" * 60)
    click.echo("Results Summary")
    click.echo("=" * 60)

    if flags:
        click.echo(f"\n{'Flag':>6} {'Score':>7} {'Peak Z':>8} {'Area (m²)':>12} "
                    f"{'Accel (mm/yr²)':>16} {'Voight':>8}  {'Failure Estimate':>20}")
        click.echo("-" * 90)
        for f in flags[:10]:
            voight_str = (
                f"R²={f.voight_fit['r_squared']:.2f}"
                if f.voight_fit
                else "—"
            )
            if f.voight_fit and "predicted_failure_iso" in f.voight_fit:
                vf = f.voight_fit
                failure_str = vf["predicted_failure_iso"]
                window = vf.get("failure_window_days", [None, None])
                if window[0] is not None:
                    failure_str += f" (±{abs(window[1] - window[0]) / 2:.0f}d)"
            else:
                failure_str = "—"
            click.echo(
                f"{f.flag_id:>6d} {f.score:>7.2f} {f.peak_zscore:>8.1f} "
                f"{f.area_m2:>12,.0f} "
                f"{f.acceleration_m_yr2 * 1000:>16.2f} "
                f"{voight_str:>8}  {failure_str:>20}"
            )

    click.echo()
    click.echo(
        "The injected failure zone was at row=80, col=120 "
        "(~28.22°N, 85.90°E)."
    )
    if flags:
        nearest = min(
            flags,
            key=lambda f: abs(f.center_lat - 28.22) + abs(f.center_lon - 85.90),
        )
        click.echo(
            f"Nearest detection: Flag {nearest.flag_id} at "
            f"{nearest.center_lat:.4f}°N, {nearest.center_lon:.4f}°E "
            f"(score={nearest.score:.2f})"
        )


@main.command()
@click.option("--config", "-c", required=True, help="Path to site/multi-site config YAML")
@click.option(
    "--interval",
    type=float,
    default=None,
    help="Check interval in hours (overrides config; default from config or 6)",
)
@click.option(
    "--check-now",
    is_flag=True,
    help="Run a single check cycle and exit (no loop)",
)
def monitor(config: str, interval: float | None, check_now: bool) -> None:
    """
    Run continuous monitoring for new NISAR acquisitions.

    Watches one or more sites for new data on ASF DAAC, downloads new
    scenes, runs the anomaly detection pipeline, and generates alerts.

    The config can be a single-site YAML (with a top-level 'site' key)
    or a multi-site YAML (with a 'sites' list and shared defaults).

    Examples:
        gews monitor -c config/nepal_2026.yaml --check-now
        gews monitor -c config/global_watch.yaml --interval 6
    """
    from gews.monitor import run_monitoring_loop, run_check_cycle, iter_site_configs

    # iter_site_configs and run_monitoring_loop take a file path,
    # not a loaded dict — they parse YAML internally.
    config_path = config

    if check_now:
        click.echo("Running single check cycle...")
        for site_cfg in iter_site_configs(config_path):
            site_name = site_cfg.get("site", {}).get("name", "unnamed")
            click.echo(f"\n{'─' * 50}")
            click.echo(f"Site: {site_name}")
            click.echo(f"{'─' * 50}")
            alerts = run_check_cycle(site_cfg)
            if alerts:
                for a in alerts:
                    level = a.get("level", "INFO")
                    msg = a.get("message", "")
                    click.echo(f"  [{level}] {msg}")
            else:
                click.echo("  No new alerts.")
        click.echo("\nDone.")
    else:
        cfg = _load_config(config)
        interval_hours = interval or cfg.get("monitor", {}).get("interval_hours", 6)
        click.echo(f"Starting monitoring loop (interval: {interval_hours}h)")
        click.echo(f"Press Ctrl+C to stop.\n")
        try:
            run_monitoring_loop(config_path, interval_hours=interval_hours)
        except KeyboardInterrupt:
            click.echo("\nMonitoring stopped.")


@main.command()
@click.option("--config", "-c", required=True, help="Path to site config YAML")
@click.option("--port", default=8080, help="Port to serve on")
@click.option("--data-dir", default="output", help="Directory containing flags.geojson")
def dashboard(config: str, port: int, data_dir: str) -> None:
    """Launch the Tier 2 analyst review dashboard."""
    from gews.dashboard import serve

    cfg = _load_config(config)
    serve(data_dir=data_dir, port=port, config=cfg)


@main.command(name="map")
@click.option("--data-dir", default="output", help="Directory containing flags.geojson")
@click.option("--port", default=8050, type=int, help="Port to serve on")
def map_view(data_dir: str, port: int) -> None:
    """Launch an interactive map viewer for flagged sites."""
    from gews.mapview import serve

    serve(data_dir=data_dir, port=port)


@main.command()
@click.option("--config", "-c", required=True, help="Path to site config YAML")
def validate(config: str) -> None:
    """Validate a configuration file and report errors/warnings."""
    from gews.validate import validate_config_file

    cfg, issues = validate_config_file(config)

    errors = [i for i in issues if i.startswith("ERROR:")]
    warnings = [i for i in issues if i.startswith("WARNING:")]

    if not issues:
        click.echo(f"Config OK: {config}")
        sys.exit(0)

    for issue in issues:
        if issue.startswith("ERROR:"):
            click.secho(issue, fg="red", err=True)
        else:
            click.secho(issue, fg="yellow", err=True)

    click.echo()
    click.echo(f"{len(errors)} error(s), {len(warnings)} warning(s)")

    if errors:
        sys.exit(1)


@main.command()
def info() -> None:
    """Show version, Python info, dependency status, and available configs."""
    import importlib

    from gews import __version__

    click.echo(f"GEWS v{__version__}")
    click.echo(f"Python {sys.version}")
    click.echo()

    # Dependency status
    click.echo("Dependencies:")
    deps = [
        "click",
        "yaml",
        "numpy",
        "scipy",
        "h5py",
        "matplotlib",
        "asf_search",
    ]
    for dep in deps:
        mod_name = dep if dep != "yaml" else "yaml"
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", "installed")
            click.echo(f"  {dep:20s} {ver}")
        except ImportError:
            click.secho(f"  {dep:20s} NOT INSTALLED", fg="yellow")

    click.echo()

    # Available configs
    config_dir = Path(__file__).parent.parent.parent / "config"
    if config_dir.is_dir():
        configs = sorted(config_dir.glob("*.yaml"))
        click.echo(f"Available configs ({config_dir}):")
        for c in configs:
            click.echo(f"  {c.name}")
    else:
        click.echo("No config directory found.")


@main.command()
@click.option(
    "--output", "-o",
    default="models/precursor_classifier.json",
    help="Path to save the trained model JSON",
)
@click.option("--seed", type=int, default=42, help="Random seed for reproducibility")
@click.option("--n-positive", type=int, default=200, help="Number of positive training samples")
@click.option("--n-negative", type=int, default=800, help="Number of negative training samples")
@click.option("--n-iter", type=int, default=500, help="Training iterations")
@click.option("--lr", type=float, default=0.1, help="Learning rate")
def train(output: str, seed: int, n_positive: int, n_negative: int, n_iter: int, lr: float) -> None:
    """Train the precursor classifier on synthetic landslide data."""
    from gews.classifier import train_precursor_model

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    click.echo(f"Training precursor classifier (seed={seed}, "
               f"n_positive={n_positive}, n_negative={n_negative}, "
               f"n_iter={n_iter}, lr={lr})...")

    metrics = train_precursor_model(
        output_path,
        seed=seed,
        n_positive=n_positive,
        n_negative=n_negative,
        n_iter=n_iter,
        lr=lr,
    )

    click.echo(f"\nTraining complete. Model saved to: {output_path}")
    click.echo(f"  Accuracy:  {metrics['accuracy']:.3f}")
    click.echo(f"  Precision: {metrics['precision']:.3f}")
    click.echo(f"  Recall:    {metrics['recall']:.3f}")
    click.echo(f"  Loss:      {metrics['final_loss']:.4f}")
    click.echo(f"  Train/Test: {metrics['n_train']}/{metrics['n_test']}")


@main.command(name="benchmark")
@click.option("--quick", is_flag=True, help="Run quick benchmark only")
def benchmark_cmd(quick: bool) -> None:
    """Run pipeline performance benchmarks."""
    from gews.benchmark import PipelineBenchmark

    bench = PipelineBenchmark()

    if quick:
        pixel_list = [1000, 5000]
    else:
        pixel_list = [1000, 5000, 10000, 50000]

    click.echo("Running timeseries benchmark...")
    bench.benchmark_timeseries(n_pixels_list=pixel_list)

    click.echo("Running step-change benchmark...")
    bench.benchmark_step_change(n_pixels_list=pixel_list)

    click.echo("Running BOCPD benchmark...")
    bench.benchmark_bocpd(n_pixels_list=pixel_list)

    click.echo("Running detection benchmark...")
    bench.benchmark_detection(n_pixels_list=pixel_list)

    report = bench.generate_report()
    click.echo(report)


@main.command()
def version() -> None:
    """Show version information."""
    from gews import __version__

    click.echo(f"GEWS v{__version__}")
    click.echo("Glacier Early Warning System — PoC InSAR Pipeline")


@main.command()
@click.option("--site", default=None, help="Filter by site name")
@click.option("--since", default=None, help="Start date (YYYY-MM-DD)")
@click.option("--provenance-dir", default="data/provenance", help="Provenance log directory")
@click.option("--audit-dir", default="data/audit", help="Alert audit log directory")
def audit(site: str | None, since: str | None, provenance_dir: str, audit_dir: str) -> None:
    """Show provenance and alert audit trail."""
    from gews.provenance import AlertAuditLog, ProvenanceTracker

    tracker = ProvenanceTracker(log_dir=provenance_dir)
    audit_log = AlertAuditLog(log_dir=audit_dir)

    # Provenance report
    prov_report = tracker.generate_audit_report(site_name=site)
    click.echo(prov_report)

    click.echo()

    # Alert history
    alert_report = audit_log.generate_alert_history(site_name=site)
    click.echo(alert_report)


if __name__ == "__main__":
    main()
