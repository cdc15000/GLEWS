"""
Operational monitoring module — turns the retrospective GLEWS pipeline
into a standing watch over configured glacier/rock-slope sites.

Design notes (read before touching the SBAS/z-score code paths):

    NISAR displacement time series are NOT incrementally update-able at
    the inversion layer. `nisar.load_nisar_gunw_stack` builds a single
    global SBAS design matrix over *all* dates in the product directory
    and solves it with one `lstsq` call; `timeseries.compute_acceleration_map`
    z-scores each pixel's acceleration against a baseline drawn from the
    first ~70% of windows and fits harmonics over the full series. A new
    acquisition changes every residual and every z-score, not just the
    newest sample. Appending a raw displacement value onto a cached array
    would silently produce an inconsistent SBAS reference and a broken
    harmonic fit.

    So the incrementality this module provides lives at the *acquisition*
    layer, not the *inversion* layer:
        - MonitorState tracks which product granules have already been
          seen and downloaded.
        - Only new granules are downloaded into the site's product
          directory.
        - The full (now larger) set of products in that directory is
          re-loaded and re-inverted through the existing, unmodified
          nisar.py / timeseries.py / detect.py pipeline.
    This keeps the science correct; the "incremental" saving is that we
    never re-download or re-search for products we already have, and the
    recompute cost (SBAS inversion + acceleration map) is cheap relative
    to a satellite download.

Usage:
    from glews.monitor import run_monitoring_loop

    run_monitoring_loop("config/global_watch.yaml", interval_hours=6)

    # or a single check-now pass:
    from glews.monitor import MonitorState, check_new_data, process_new_acquisition
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Default lookback window for ASF searches. NISAR products are typically
# published some days after acquisition, so we must not diff against
# `last_check` as the search start bound — a granule acquired before the
# last check can still appear in the archive afterward. Instead we always
# re-search a rolling window and diff the *returned product IDs* against
# what we already know about.
DEFAULT_LOOKBACK_DAYS = 30

# Default alert levels, expressed as multiples of the configured
# sigma_threshold. Overridable via config["monitor"]["alert_levels"].
DEFAULT_ALERT_LEVELS = {
    "INFO": 1.0,       # >= 1.0x sigma_threshold
    "WARNING": 1.5,    # >= 1.5x sigma_threshold
    "CRITICAL": 2.5,   # >= 2.5x sigma_threshold (or a strong Voight fit)
}


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------


@dataclass
class MonitorState:
    """
    Persistent state for the monitoring loop, checkpointed to disk so a
    restart resumes rather than re-processing or re-alerting.

    Attributes
    ----------
    site_name : str
        Identifier matching config["site"]["name"].
    last_check : str or None
        ISO timestamp of the most recent completed check (informational —
        NOT used as the ASF search lower bound; see module docstring).
    known_products : set[str]
        ASF granule/fileID's already seen (searched, or already downloaded).
    downloaded_products : set[str]
        Granules already present on disk — avoids re-downloading.
    alerted_signatures : set[str]
        Deterministic signatures (see `_alert_signature`) of anomalies
        already alerted on, so re-running detection on a grown time
        series (whose z-scores drift as the baseline grows) does not
        re-alert the same physical anomaly.
    n_checks : int
        Number of completed check cycles (diagnostic counter).
    """

    site_name: str
    last_check: str | None = None
    known_products: set[str] = field(default_factory=set)
    downloaded_products: set[str] = field(default_factory=set)
    alerted_signatures: set[str] = field(default_factory=set)
    n_checks: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["known_products"] = sorted(self.known_products)
        d["downloaded_products"] = sorted(self.downloaded_products)
        d["alerted_signatures"] = sorted(self.alerted_signatures)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MonitorState":
        return cls(
            site_name=d["site_name"],
            last_check=d.get("last_check"),
            known_products=set(d.get("known_products", [])),
            downloaded_products=set(d.get("downloaded_products", [])),
            alerted_signatures=set(d.get("alerted_signatures", [])),
            n_checks=d.get("n_checks", 0),
        )

    @classmethod
    def load(cls, path: str | Path, site_name: str) -> "MonitorState":
        path = Path(path)
        if not path.exists():
            return cls(site_name=site_name)
        try:
            with open(path) as f:
                return cls.from_dict(json.load(f))
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Could not parse state file %s (%s); starting fresh", path, e)
            return cls(site_name=site_name)

    def save(self, path: str | Path) -> None:
        """Write state atomically (tmp file + rename) so a crash mid-write
        never leaves a corrupt/half-written state file behind."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        tmp_path.replace(path)


def _state_path(site_config: dict) -> Path:
    mon = site_config.get("monitor", {})
    state_dir = Path(mon.get("state_dir", "data/monitor_state"))
    safe_name = site_config["site"]["name"].replace(" ", "_").replace("/", "-")
    return state_dir / f"{safe_name}.json"


# --------------------------------------------------------------------------
# Step (a): check for new data
# --------------------------------------------------------------------------


def check_new_data(site_config: dict, state: MonitorState | None = None) -> list:
    """
    Query ASF DAAC for NISAR GUNW/GOFF products at a configured site that
    have not been seen before.

    Parameters
    ----------
    site_config : dict
        Parsed YAML config with 'site', 'acquire', and optionally
        'monitor' sections (a single-site slice, as produced by
        `iter_site_configs`).
    state : MonitorState or None
        Existing state to diff against. If None, loads/creates state
        from the configured state file.

    Returns
    -------
    list[glews.acquire.SceneInfo]
        Newly discovered products not present in state.known_products.
        Note: `known_products` is NOT updated here — callers decide
        when a product graduates from "seen" to "known" (typically
        after a successful download).
    """
    import asf_search as asf
    from shapely.geometry import Point

    site = site_config["site"]
    acq = site_config.get("acquire", {})
    mon = site_config.get("monitor", {})

    if state is None:
        state = MonitorState.load(_state_path(site_config), site["name"])

    lookback_days = mon.get("lookback_days", DEFAULT_LOOKBACK_DAYS)
    search_start = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    center = Point(site["longitude"], site["latitude"])
    buffer_deg = site.get("buffer_km", 15) / 111.0
    aoi = center.buffer(buffer_deg)

    product_types = acq.get("product_types", ["GUNW", "GOFF"])
    asf_product_types = [getattr(asf.PRODUCT_TYPE, pt) for pt in product_types]

    logger.info(
        "[%s] Searching ASF for NISAR %s since %s (lookback=%dd)",
        site["name"], "/".join(product_types), search_start, lookback_days,
    )

    results = asf.search(
        platform=asf.PLATFORM.NISAR,
        processingLevel=asf_product_types,
        intersectsWith=aoi.wkt,
        start=search_start,
    )

    from glews.acquire import SceneInfo

    new_products = []
    for r in results:
        granule = r.properties.get("fileID") or r.properties.get("sceneName")
        if granule in state.known_products:
            continue
        try:
            scene = SceneInfo.from_asf_result(r)
        except (KeyError, TypeError):
            # NISAR properties don't always populate the same keys as
            # Sentinel-1 (e.g. pathNumber, bytes as dict); fall back to
            # a loose wrapper.
            scene = _scene_from_nisar_result(r, granule)
        new_products.append(scene)

    logger.info(
        "[%s] %d new product(s) of %d total found",
        site["name"], len(new_products), len(results),
    )
    return new_products


def _scene_from_nisar_result(result, granule: str):
    """Best-effort SceneInfo construction for NISAR products whose
    metadata schema differs slightly from Sentinel-1 SLC."""
    from glews.acquire import SceneInfo

    props = result.properties

    # bytes can be a dict, int, or None depending on the product type
    raw_bytes = props.get("bytes", 0)
    if isinstance(raw_bytes, dict):
        raw_bytes = raw_bytes.get("bytes", 0) or 0
    raw_bytes = raw_bytes or 0

    return SceneInfo(
        granule=granule,
        start_time=props.get("startTime", ""),
        path_number=props.get("pathNumber") or props.get("relativeOrbit") or 0,
        frame_number=props.get("frameNumber", 0),
        polarization=props.get("polarization", ""),
        url=props.get("url", ""),
        file_size_mb=float(raw_bytes) / 1e6,
        geometry=result.geometry,
    )


# --------------------------------------------------------------------------
# Step (b): process a new acquisition
# --------------------------------------------------------------------------


def process_new_acquisition(
    products: list,
    site_config: dict,
    state: MonitorState,
) -> dict[str, Any]:
    """
    Download newly found products, re-run the detection pipeline on the
    grown product set, and return the results for alerting.

    This does NOT try to incrementally patch the displacement time
    series in memory (see module docstring) — it downloads the new
    granules into the site's product directory and reloads/re-inverts
    the full stack via the existing nisar.py loaders.

    Parameters
    ----------
    products : list[SceneInfo]
        New products from `check_new_data` to download and fold in.
    site_config : dict
        Single-site config slice.
    state : MonitorState
        Mutated in place: `known_products` and `downloaded_products`
        are updated as products are downloaded.

    Returns
    -------
    dict with keys:
        'flags' : list[AnomalyFlag]
        'ts' : NISARTimeseries | None (whichever source produced flags)
        'accel_map' : AccelerationMap | None
        'source' : "GUNW" | "GOFF" | None
    """
    from glews.acquire import download_scenes
    from glews.detect import detect_anomalies
    from glews.nisar import load_nisar_goff_stack, load_nisar_gunw_stack
    from glews.timeseries import compute_acceleration_map

    acq = site_config.get("acquire", {})
    site_name = site_config["site"]["name"]

    gunw_dir = Path(acq.get("gunw_dir", "data/nisar/gunw"))
    goff_dir = Path(acq.get("goff_dir", "data/nisar/goff"))
    gunw_dir.mkdir(parents=True, exist_ok=True)
    goff_dir.mkdir(parents=True, exist_ok=True)

    if products:
        gunw_products = [p for p in products if "GUNW" in p.granule.upper()]
        goff_products = [p for p in products if "GOFF" in p.granule.upper()]

        if gunw_products:
            logger.info("[%s] Downloading %d new GUNW product(s)", site_name, len(gunw_products))
            download_scenes(gunw_products, output_dir=gunw_dir, n_workers=acq.get("n_workers", 4))
        if goff_products:
            logger.info("[%s] Downloading %d new GOFF product(s)", site_name, len(goff_products))
            download_scenes(goff_products, output_dir=goff_dir, n_workers=acq.get("n_workers", 4))

        for p in products:
            state.known_products.add(p.granule)
            state.downloaded_products.add(p.granule)

    det_cfg = site_config["detect"]
    result: dict[str, Any] = {"flags": [], "ts": None, "accel_map": None, "source": None}

    # Prefer GUNW (phase-based, higher precision); fall back to GOFF
    # (amplitude offset tracking) when GUNW is unavailable or too sparse.
    for loader, src_dir, source_name in (
        (load_nisar_gunw_stack, gunw_dir, "GUNW"),
        (load_nisar_goff_stack, goff_dir, "GOFF"),
    ):
        try:
            ts = loader(src_dir, track=acq.get("path_number"))
        except (FileNotFoundError, ValueError) as e:
            logger.info("[%s] %s not usable: %s", site_name, source_name, e)
            continue

        accel_map = compute_acceleration_map(
            ts.dates,
            ts.displacement,
            window_size_days=det_cfg["acceleration"]["window_size_days"],
            step_days=det_cfg["acceleration"]["step_days"],
            n_harmonics=det_cfg.get("n_harmonics", 2),
        )
        flags = detect_anomalies(accel_map, ts.latitude, ts.longitude, site_config)

        result = {"flags": flags, "ts": ts, "accel_map": accel_map, "source": source_name}
        break  # GUNW succeeded — no need to also run GOFF

    return result


# --------------------------------------------------------------------------
# Alerting
# --------------------------------------------------------------------------


def _coords_are_projected(latitude, longitude) -> bool:
    """Mirror of detect._coords_are_projected — grids may be UTM meters
    rather than degrees; alert lat/lon and distances must not assume
    geographic coordinates."""
    import numpy as np

    lat_range = np.nanmax(latitude) - np.nanmin(latitude)
    lon_range = np.nanmax(longitude) - np.nanmin(longitude)
    return lat_range > 1000 or lon_range > 1000


def _distance_km(lat1, lon1, lat2, lon2, projected: bool) -> float:
    import numpy as np

    if projected:
        return float(np.hypot(lat1 - lat2, lon1 - lon2) / 1000.0)
    # Haversine-ish flat-earth approximation, adequate at these scales.
    dlat_km = (lat1 - lat2) * 111.32
    dlon_km = (lon1 - lon2) * 111.32 * np.cos(np.radians((lat1 + lat2) / 2))
    return float(np.hypot(dlat_km, dlon_km))


def _alert_signature(site_name: str, flag) -> str:
    """Deterministic id for a physical anomaly, stable across runs even
    as z-scores drift with a growing baseline. Used both as the alert_id
    and as the dedup key in MonitorState.alerted_signatures."""
    key = f"{site_name}|{round(flag.center_lat, 3)}|{round(flag.center_lon, 3)}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def _alert_level(flag, sigma_threshold: float, alert_levels: dict) -> str | None:
    """Classify a flag's severity. Returns None if below the lowest level."""
    ratio = flag.peak_zscore / sigma_threshold if sigma_threshold else 0.0

    level = None
    for name, threshold_ratio in sorted(alert_levels.items(), key=lambda kv: kv[1]):
        if ratio >= threshold_ratio:
            level = name

    # A strong, short-horizon Voight fit escalates regardless of z-score ratio.
    if flag.voight_fit and flag.voight_fit.get("days_until_failure", 999) < 30:
        level = "CRITICAL"

    return level


def build_alerts(
    flags: list,
    ts,
    site_config: dict,
    state: MonitorState,
) -> list[dict]:
    """
    Convert newly-detected anomaly flags into alert records, skipping
    flags already alerted on (tracked in `state.alerted_signatures`).

    Parameters
    ----------
    flags : list[AnomalyFlag]
        Output of detect_anomalies, sorted by score.
    ts : NISARTimeseries or None
        Needed to look up displacement at each flag's peak pixel.
    site_config : dict
        Single-site config slice.
    state : MonitorState
        Mutated in place: matching signatures are added to
        `alerted_signatures`.

    Returns
    -------
    list[dict]
        New alert records (schema documented in the module and README).
    """
    site = site_config["site"]
    det_cfg = site_config["detect"]
    sigma_threshold = det_cfg["acceleration"]["sigma_threshold"]
    alert_levels = site_config.get("monitor", {}).get("alert_levels", DEFAULT_ALERT_LEVELS)

    alerts = []
    now = datetime.now(timezone.utc).isoformat()

    for flag in flags:
        sig = _alert_signature(site["name"], flag)
        if sig in state.alerted_signatures:
            continue

        level = _alert_level(flag, sigma_threshold, alert_levels)
        if level is None:
            continue

        projected = False
        displacement_mm = None
        if ts is not None:
            projected = _coords_are_projected(ts.latitude, ts.longitude)
            # Mean displacement across the flagged cluster's pixels, latest date.
            try:
                disp = ts.displacement[-1, flag.pixel_indices[:, 0], flag.pixel_indices[:, 1]]
                displacement_mm = round(float(_nanmean(disp)) * 1000, 1)
            except Exception:
                displacement_mm = None

        nearest_flag_km = _distance_km(
            flag.center_lat, flag.center_lon, site["latitude"], site["longitude"],
            projected=projected,
        )

        alert = {
            "alert_id": sig,
            "site": site["name"],
            "timestamp": now,
            "level": level,
            "max_zscore": round(float(flag.peak_zscore), 2),
            "n_flags": 1,
            "nearest_flag_km": round(nearest_flag_km, 2),
            "displacement_mm": displacement_mm,
            "message": _alert_message(flag, level, sigma_threshold),
        }

        # Include failure date estimate when Voight fit is available
        if flag.voight_fit:
            vf = flag.voight_fit
            alert["voight_r2"] = round(vf["r_squared"], 3)
            alert["days_until_failure"] = round(vf["days_until_failure"], 1)
            if "predicted_failure_iso" in vf:
                alert["predicted_failure_date"] = vf["predicted_failure_iso"]
            if "failure_window_early_iso" in vf:
                alert["failure_window"] = {
                    "early": vf["failure_window_early_iso"],
                    "late": vf["failure_window_late_iso"],
                    "confidence": vf.get("confidence_level", 0.95),
                }
        alerts.append(alert)
        state.alerted_signatures.add(sig)

    return alerts


def _nanmean(arr):
    import numpy as np

    return np.nanmean(arr)


def _alert_message(flag, level: str, sigma_threshold: float) -> str:
    msg = (
        f"Anomalous acceleration detected: peak z-score {flag.peak_zscore:.1f} "
        f"(threshold {sigma_threshold:.1f}sigma), {flag.n_pixels} pixels, "
        f"{flag.area_m2:,.0f} m^2 at ({flag.center_lat:.4f}, {flag.center_lon:.4f})."
    )
    if flag.voight_fit:
        vf = flag.voight_fit
        msg += (
            f" Voight fit (R^2={vf['r_squared']:.2f}) predicts "
            f"failure in ~{vf['days_until_failure']:.0f} days"
        )
        if "predicted_failure_iso" in vf:
            msg += f" (~{vf['predicted_failure_iso']})"
        window = vf.get("failure_window_days", [None, None])
        if window[0] is not None and window[1] is not None:
            msg += (
                f", 95% CI: {window[0]:.0f}–{window[1]:.0f} days "
                f"({vf.get('failure_window_early_iso', '?')} to "
                f"{vf.get('failure_window_late_iso', '?')})"
            )
        msg += "."
    if level == "CRITICAL":
        msg = "CRITICAL: " + msg
    return msg


def write_alerts(alerts: list[dict], site_config: dict) -> Path | None:
    """
    Append new alerts to the site's alert JSON file.

    Format: a JSON array of alert records, one file per site, appended
    to on each run. This is the placeholder sink for now — swap for an
    email/webhook/Slack notifier by replacing this function.
    """
    if not alerts:
        return None

    mon = site_config.get("monitor", {})
    alert_dir = Path(mon.get("alert_dir", "output/alerts"))
    alert_dir.mkdir(parents=True, exist_ok=True)

    safe_name = site_config["site"]["name"].replace(" ", "_").replace("/", "-")
    path = alert_dir / f"{safe_name}_alerts.json"

    existing = []
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            logger.warning("Could not parse existing alert file %s; overwriting", path)

    existing.extend(alerts)
    path.write_text(json.dumps(existing, indent=2))

    for a in alerts:
        logger.warning(
            "[%s] ALERT %s (%s): z=%.1f, %s",
            a["site"], a["alert_id"], a["level"], a["max_zscore"], a["message"],
        )

    return path


# --------------------------------------------------------------------------
# Config helpers
# --------------------------------------------------------------------------


def iter_site_configs(config_path: str | Path):
    """
    Yield one per-site config dict for each site defined in `config_path`.

    Supports two shapes:
      - single-site configs (like config/nepal_2026.yaml): a top-level
        'site' mapping. Yields the config unchanged (once).
      - multi-site configs (like config/global_watch.yaml): a top-level
        'sites' list, each entry a 'site' mapping, plus shared
        'acquire'/'detect'/'cascade'/'monitor' sections at the top level
        that are merged in (per-site keys take precedence).

    Each yielded dict still exposes cfg["detect"] and
    cfg["site"]["name"]/["event_date"] exactly as detect_anomalies() and
    report.py expect, so neither module needs to change.
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    if "sites" in raw:
        shared = {k: v for k, v in raw.items() if k != "sites"}
        for site_entry in raw["sites"]:
            cfg = {**shared}
            for key, value in site_entry.items():
                if key in cfg and isinstance(cfg[key], dict) and isinstance(value, dict):
                    cfg[key] = {**cfg[key], **value}
                else:
                    cfg[key] = value
            yield cfg
    else:
        yield raw


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------


def run_check_cycle(site_config: dict) -> list[dict]:
    """
    Run one complete check-download-detect-alert cycle for a single site.

    Returns the list of new alerts generated (may be empty).
    """
    from glews.provenance import AlertAuditLog, ProvenanceTracker

    site_name = site_config["site"]["name"]
    state_path = _state_path(site_config)
    state = MonitorState.load(state_path, site_name)

    mon = site_config.get("monitor", {})
    tracker = ProvenanceTracker(log_dir=mon.get("provenance_dir", "data/provenance"))
    audit_log = AlertAuditLog(log_dir=mon.get("audit_dir", "data/audit"))

    try:
        with tracker.track("check_cycle", site_name) as cycle_ctx:
            new_products = check_new_data(site_config, state=state)
            result = process_new_acquisition(new_products, site_config, state)

            alerts = []
            if result["flags"]:
                alerts = build_alerts(result["flags"], result["ts"], site_config, state)
                write_alerts(alerts, site_config)

                # Dispatch alerts to configured notification channels
                # (email, Slack, webhook).  If all channels fail for a given
                # alert, remove its signature so the next cycle retries.
                from glews.alerts import AlertDispatcher

                dispatcher = AlertDispatcher.from_config(site_config)
                for alert in alerts:
                    dispatch_results = dispatcher.dispatch(
                        alert["level"],
                        alert["site"],
                        alert["message"],
                        {k: v for k, v in alert.items()
                         if k not in ("level", "site", "message")},
                    )

                    # Log each alert dispatch to the audit log
                    channels_sent = [
                        ch for ch, ok in dispatch_results.items() if ok
                    ]
                    channels_failed = [
                        ch for ch, ok in dispatch_results.items() if not ok
                    ]
                    audit_log.log_alert(
                        alert["level"],
                        alert["site"],
                        alert["message"],
                        channels_sent=channels_sent,
                        channels_failed=channels_failed,
                    )

                    if dispatch_results and not any(dispatch_results.values()):
                        state.alerted_signatures.discard(alert["alert_id"])

            cycle_ctx.set_result(
                f"{len(new_products)} new products, "
                f"{len(result['flags'])} flags, "
                f"{len(alerts)} alerts"
            )

        state.last_check = datetime.now(timezone.utc).isoformat()
        state.n_checks += 1
        state.save(state_path)

        logger.info(
            "[%s] Check complete: %d new products, %d flags, %d new alerts",
            site_name, len(new_products), len(result["flags"]), len(alerts),
        )
        return alerts

    except Exception:
        logger.exception("[%s] Check cycle failed", site_name)
        return []


def run_monitoring_loop(
    config_path: str | Path,
    interval_hours: float = 6,
    check_now_only: bool = False,
    max_iterations: int | None = None,
    _sleep_fn=time.sleep,
) -> None:
    """
    Main operational loop: periodically checks every configured site for
    new NISAR acquisitions, processes them, and raises alerts.

    Parameters
    ----------
    config_path : str or Path
        Path to a single-site or multi-site (config["sites"]) YAML config.
    interval_hours : float
        Hours to sleep between check cycles.
    check_now_only : bool
        If True, run exactly one cycle across all sites and return
        (used by `glews monitor --check-now`).
    max_iterations : int or None
        Cap on loop iterations — for tests/diagnostics; None runs forever.
    _sleep_fn : callable
        Injection seam for tests to avoid actually sleeping.
    """
    site_configs = list(iter_site_configs(config_path))
    logger.info(
        "Starting monitoring loop: %d site(s), interval=%.1fh",
        len(site_configs), interval_hours,
    )

    iteration = 0
    while True:
        iteration += 1
        logger.info("=== Check cycle %d ===", iteration)

        for site_config in site_configs:
            run_check_cycle(site_config)

        if check_now_only:
            break
        if max_iterations is not None and iteration >= max_iterations:
            break

        logger.info("Sleeping %.1fh until next check", interval_hours)
        _sleep_fn(interval_hours * 3600)
