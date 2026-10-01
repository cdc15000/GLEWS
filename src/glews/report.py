"""
Reporting module — generate visual reports and GeoJSON output
for flagged sites.

Produces:
    - Time-series plots for each flagged site
    - Spatial anomaly maps
    - Cascade risk summary
    - GeoJSON export for GIS integration
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from glews.cascade import CascadeAssessment, RiskLevel
from glews.detect import AnomalyFlag
from glews.timeseries import AccelerationMap

logger = logging.getLogger(__name__)


def generate_report(
    flags: list[AnomalyFlag],
    assessments: list[CascadeAssessment] | None,
    accel_map: AccelerationMap,
    latitude: np.ndarray,
    longitude: np.ndarray,
    config: dict,
    output_dir: str | Path = "output",
) -> Path:
    """
    Generate a complete analysis report.

    Parameters
    ----------
    flags : list[AnomalyFlag]
        Tier 0 detections.
    assessments : list[CascadeAssessment] or None
        Tier 1 assessments, if available.
    accel_map : AccelerationMap
        Acceleration data for plotting.
    latitude, longitude : np.ndarray
        Coordinate grids.
    config : dict
        Full configuration.
    output_dir : str or Path
        Output directory.

    Returns
    -------
    Path
        Path to the main report file.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    site_name = config["site"]["name"]

    # 1. Spatial anomaly map — latest window
    fig, ax = plt.subplots(figsize=(10, 8))
    latest_zscore = accel_map.acceleration_zscore[-1]

    vmax = max(3, np.nanpercentile(np.abs(latest_zscore), 99))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.pcolormesh(
        longitude, latitude, latest_zscore,
        cmap="RdBu_r", norm=norm, shading="auto",
    )
    plt.colorbar(im, ax=ax, label="Acceleration z-score", shrink=0.8)

    # Mark flagged sites
    for flag in flags[:20]:
        ax.plot(
            flag.center_lon, flag.center_lat,
            "k^", markersize=8, markeredgewidth=1.5, markerfacecolor="none",
        )
        ax.annotate(
            f"F{flag.flag_id}",
            (flag.center_lon, flag.center_lat),
            fontsize=7, ha="left", va="bottom",
            xytext=(4, 4), textcoords="offset points",
        )

    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(f"{site_name} — Acceleration Anomaly Map\n"
                 f"Window ending {_ordinal_to_str(accel_map.window_centers[-1])}")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(fig_dir / "anomaly_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Time-series plots for top flags
    for flag in flags[:10]:
        _plot_flag_timeseries(flag, accel_map, fig_dir)

    # 3. Acceleration evolution plot
    _plot_acceleration_evolution(flags[:5], accel_map, fig_dir, site_name)

    # 4. GeoJSON export
    geojson_path = _export_geojson(flags, assessments, output_dir)

    # 5. Summary text report
    report_path = output_dir / "report.md"
    _write_summary(flags, assessments, config, report_path)

    logger.info("Report generated in %s", output_dir)
    logger.info("  Anomaly map: %s", fig_dir / "anomaly_map.png")
    logger.info("  GeoJSON: %s", geojson_path)
    logger.info("  Summary: %s", report_path)

    return report_path


def _plot_flag_timeseries(
    flag: AnomalyFlag,
    accel_map: AccelerationMap,
    fig_dir: Path,
) -> None:
    """Plot acceleration time series for a flagged site."""
    import matplotlib.pyplot as plt

    # Average acceleration z-score across cluster pixels
    rows = flag.pixel_indices[:, 0]
    cols = flag.pixel_indices[:, 1]

    # Clamp indices to valid range
    n_rows, n_cols = accel_map.acceleration_zscore.shape[1:]
    valid = (rows < n_rows) & (cols < n_cols)
    rows, cols = rows[valid], cols[valid]
    if len(rows) == 0:
        return

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_zscore = np.nanmean(
            accel_map.acceleration_zscore[:, rows, cols], axis=1
        )
        mean_accel = np.nanmean(
            accel_map.acceleration[:, rows, cols], axis=1
        )

    dates = [_ordinal_to_date(d) for d in accel_map.window_centers]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Acceleration
    ax1.plot(dates, mean_accel * 1000, "b-o", markersize=3, linewidth=1)
    ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax1.set_ylabel("Acceleration (mm/yr²)")
    ax1.set_title(
        f"Flag {flag.flag_id} — {flag.center_lat:.4f}°N, "
        f"{flag.center_lon:.4f}°E ({flag.n_pixels} pixels, "
        f"{flag.area_m2:.0f} m²)"
    )

    # Z-score with threshold
    sigma = flag.detection_details.get("sigma_threshold", 2.5)
    ax2.fill_between(dates, mean_zscore, 0, alpha=0.3, color="steelblue")
    ax2.plot(dates, mean_zscore, "b-o", markersize=3, linewidth=1)
    ax2.axhline(sigma, color="red", linewidth=1, linestyle="--", label=f"σ = {sigma}")
    ax2.axhline(-sigma, color="red", linewidth=1, linestyle="--")
    ax2.set_ylabel("Z-score (vs. baseline)")
    ax2.set_xlabel("Date")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(fig_dir / f"flag_{flag.flag_id}_timeseries.png", dpi=150)
    plt.close(fig)


def _plot_acceleration_evolution(
    flags: list[AnomalyFlag],
    accel_map: AccelerationMap,
    fig_dir: Path,
    site_name: str,
) -> None:
    """Plot acceleration evolution for top N flags on one figure."""
    import matplotlib.pyplot as plt

    if not flags:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    dates = [_ordinal_to_date(d) for d in accel_map.window_centers]

    colors = plt.cm.Set2(np.linspace(0, 1, max(len(flags), 3)))

    for i, flag in enumerate(flags):
        rows = flag.pixel_indices[:, 0]
        cols = flag.pixel_indices[:, 1]
        n_rows, n_cols = accel_map.acceleration_zscore.shape[1:]
        valid = (rows < n_rows) & (cols < n_cols)
        rows, cols = rows[valid], cols[valid]
        if len(rows) == 0:
            continue
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_z = np.nanmean(
                accel_map.acceleration_zscore[:, rows, cols], axis=1
            )
        ax.plot(
            dates, mean_z,
            "-o", markersize=3, linewidth=1.5, color=colors[i],
            label=f"Flag {flag.flag_id} (score={flag.score:.1f})",
        )

    sigma = flags[0].detection_details.get("sigma_threshold", 2.5)
    ax.axhline(sigma, color="red", linewidth=1, linestyle="--",
               alpha=0.6, label=f"Threshold (σ={sigma})")
    ax.axhline(-sigma, color="red", linewidth=1, linestyle="--", alpha=0.6)

    ax.set_xlabel("Date")
    ax.set_ylabel("Acceleration z-score")
    ax.set_title(f"{site_name} — Top {len(flags)} Anomalies")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / "acceleration_evolution.png", dpi=150)
    plt.close(fig)


def _export_geojson(
    flags: list[AnomalyFlag],
    assessments: list[CascadeAssessment] | None,
    output_dir: Path,
) -> Path:
    """Export flags as GeoJSON for viewing in QGIS, Google Earth, etc."""
    assessment_map = {}
    if assessments:
        assessment_map = {a.flag.flag_id: a for a in assessments}

    features = []
    for flag in flags:
        props = {
            "flag_id": flag.flag_id,
            "score": round(flag.score, 2),
            "peak_zscore": round(flag.peak_zscore, 2),
            "mean_zscore": round(flag.mean_zscore, 2),
            "n_pixels": flag.n_pixels,
            "area_m2": round(flag.area_m2, 0),
            "acceleration_mm_yr2": round(flag.acceleration_m_yr2 * 1000, 2),
        }

        if flag.voight_fit:
            vf = flag.voight_fit
            props["voight_r2"] = round(vf["r_squared"], 3)
            props["voight_days_to_failure"] = round(vf["days_until_failure"], 1)
            if "predicted_failure_iso" in vf:
                props["predicted_failure_date"] = vf["predicted_failure_iso"]
            if "failure_window_early_iso" in vf:
                props["failure_window"] = (
                    f"{vf['failure_window_early_iso']} to "
                    f"{vf['failure_window_late_iso']}"
                )
                props["confidence_level"] = vf.get("confidence_level", 0.95)

        if flag.timeseries is not None:
            props["timeseries"] = flag.timeseries

        assessment = assessment_map.get(flag.flag_id)
        if assessment:
            props["risk_level"] = assessment.risk_level.value
            props["volume_m3"] = round(assessment.estimated_volume_m3, 0)
            props["population_exposed"] = assessment.population_exposed
            props["dam_potential"] = assessment.dam_potential
            props["passes_tier1"] = assessment.passes_tier1

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [flag.center_lon, flag.center_lat],
            },
            "properties": props,
        })

    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "name": "GLEWS Tier 0/1 Flags",
            "generated": datetime.utcnow().isoformat() + "Z",
            "n_flags": len(flags),
        },
    }

    path = output_dir / "flags.geojson"
    path.write_text(json.dumps(geojson, indent=2, default=_json_default))
    return path


def _json_default(obj):
    """Handle numpy types in JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _write_summary(
    flags: list[AnomalyFlag],
    assessments: list[CascadeAssessment] | None,
    config: dict,
    path: Path,
) -> None:
    """Write a Markdown summary report."""
    lines = [
        f"# GLEWS Analysis Report — {config['site']['name']}",
        "",
        f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Event date:** {config['site']['event_date']}",
        "",
        "## Tier 0 — Anomaly Detection",
        "",
        f"- **Total flags:** {len(flags)}",
    ]

    if flags:
        lines.extend([
            f"- **Highest score:** {flags[0].score:.2f} "
            f"(Flag {flags[0].flag_id})",
            f"- **Peak z-score:** {max(f.peak_zscore for f in flags):.1f}",
            "",
            "### Top 10 Flags",
            "",
            "| Flag | Score | Peak Z | Area (m²) | Accel (mm/yr²) | Location |",
            "|------|-------|--------|-----------|----------------|----------|",
        ])

        for f in flags[:10]:
            lines.append(
                f"| {f.flag_id} | {f.score:.2f} | {f.peak_zscore:.1f} | "
                f"{f.area_m2:,.0f} | {f.acceleration_m_yr2 * 1000:.2f} | "
                f"{f.center_lat:.4f}°N, {f.center_lon:.4f}°E |"
            )

    if assessments:
        lines.extend([
            "",
            "## Tier 1 — Cascade Risk Assessment",
            "",
        ])

        for risk in RiskLevel:
            count = sum(1 for a in assessments if a.risk_level == risk)
            if count > 0:
                lines.append(f"- **{risk.value.upper()}:** {count} sites")

        passing = [a for a in assessments if a.passes_tier1]
        if passing:
            lines.extend([
                "",
                "### Sites advancing to Tier 2",
                "",
                "| Flag | Risk | Volume (m³) | Pop. Exposed | Dam Risk | Score |",
                "|------|------|-------------|-------------|----------|-------|",
            ])
            for a in passing:
                lines.append(
                    f"| {a.flag.flag_id} | {a.risk_level.value} | "
                    f"{a.estimated_volume_m3:,.0f} | {a.population_exposed:,} | "
                    f"{'Yes' if a.dam_potential else 'No'} | "
                    f"{a.cascade_score:.2f} |"
                )

    lines.extend([
        "",
        "---",
        f"*GLEWS v0.1.0 — Configuration: {config['site']['name']}*",
    ])

    path.write_text("\n".join(lines))


def _ordinal_to_str(ordinal: float) -> str:
    return datetime.fromordinal(int(ordinal)).strftime("%Y-%m-%d")


def _ordinal_to_date(ordinal: float) -> datetime:
    return datetime.fromordinal(int(ordinal))
