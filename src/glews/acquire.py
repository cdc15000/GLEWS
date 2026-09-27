"""
Data acquisition module — search and download SAR scenes and
interferometric products from the Alaska Satellite Facility (ASF)
DAAC archive.

Supports:
    - Sentinel-1 SLC (raw scenes for ISCE-2 processing)
    - NISAR GUNW (L-band unwrapped interferograms)
    - NISAR GOFF (L-band pixel offset tracking)
    - ARIA S1 GUNW (pre-processed Sentinel-1 interferograms)

Usage:
    from glews.acquire import search_scenes, select_track, download_scenes

    scenes = search_scenes(config)
    track_scenes = select_track(scenes)
    download_scenes(track_scenes, output_dir="data/gunw")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlparse

import asf_search as asf
from shapely.geometry import Point

logger = logging.getLogger(__name__)


def _extract_file_size_mb(bytes_field) -> float:
    if isinstance(bytes_field, (int, float)):
        return bytes_field / 1e6
    if isinstance(bytes_field, dict):
        total = 0
        for v in bytes_field.values():
            if isinstance(v, (int, float)):
                total += v
            elif isinstance(v, dict):
                total += v.get("bytes", 0)
        return total / 1e6
    return 0.0


def _filename_from_url(url: str) -> str:
    return Path(urlparse(url).path).name


@dataclass
class SceneInfo:
    """Metadata for a single SAR scene or interferometric product."""

    granule: str
    start_time: str
    path_number: int
    frame_number: int
    polarization: str
    url: str
    file_size_mb: float
    geometry: dict  # GeoJSON geometry
    product_type: str = "SLC"
    stop_time: str | None = None
    perpendicular_baseline: float | None = None
    file_name: str | None = None

    @classmethod
    def from_asf_result(cls, result: asf.ASFProduct, product_type: str = "SLC") -> SceneInfo:
        props = result.properties
        return cls(
            granule=props["fileID"],
            start_time=props["startTime"],
            path_number=props.get("pathNumber", 0) or 0,
            frame_number=props.get("frameNumber", 0) or 0,
            polarization=props.get("polarization") or "N/A",
            url=props["url"],
            file_size_mb=_extract_file_size_mb(props.get("bytes", 0)),
            geometry=result.geometry,
            product_type=product_type,
            stop_time=props.get("stopTime"),
            perpendicular_baseline=props.get("perpendicularBaseline"),
            file_name=props.get("fileName"),
        )


def _resolve_platform(platform_str: str):
    mapping = {
        "NISAR": asf.PLATFORM.NISAR,
        "SENTINEL1": asf.PLATFORM.SENTINEL1,
        "SENTINEL-1": asf.PLATFORM.SENTINEL1,
        "ALOS": asf.PLATFORM.ALOS,
    }
    return mapping.get(platform_str.upper(), asf.PLATFORM.SENTINEL1)


def _resolve_product_type(pt_str: str):
    mapping = {
        "GUNW": asf.PRODUCT_TYPE.GUNW,
        "GOFF": asf.PRODUCT_TYPE.GOFF,
        "SLC": asf.PRODUCT_TYPE.SLC,
        "CSLC": asf.PRODUCT_TYPE.CSLC,
        "RSLC": asf.PRODUCT_TYPE.RSLC,
    }
    return mapping.get(pt_str.upper(), asf.PRODUCT_TYPE.SLC)


def _resolve_dataset(platform_str: str, product_type_str: str):
    if platform_str.upper() == "NISAR":
        return asf.DATASET.NISAR
    if platform_str.upper() in ("SENTINEL1", "SENTINEL-1") and product_type_str.upper() == "GUNW":
        return asf.DATASET.ARIA_S1_GUNW
    return None


def search_scenes(config: dict) -> list[SceneInfo]:
    """
    Search ASF archive for SAR scenes or interferometric products.

    Dispatches based on the `acquire.platform` and `acquire.product_types`
    config fields. Defaults to Sentinel-1 SLC for backward compatibility.
    """
    site = config["site"]
    acq = config.get("acquire", {})

    center = Point(site["longitude"], site["latitude"])
    buffer_deg = site.get("buffer_km", 15) / 111.0
    aoi = center.buffer(buffer_deg)

    end_date = acq.get("end_date", date.today().isoformat())
    if "start_date" in acq:
        start_date = acq["start_date"]
    else:
        lookback_days = acq.get("lookback_days", 180)
        start_date = (date.today() - timedelta(days=lookback_days)).isoformat()

    platform_str = acq.get("platform", "SENTINEL1")
    product_types = acq.get("product_types", ["SLC"])

    all_scenes: list[SceneInfo] = []

    for pt_str in product_types:
        logger.info(
            "Searching ASF for %s %s: %.2f°N, %.2f°E ± %d km, %s to %s",
            platform_str, pt_str,
            site["latitude"], site["longitude"],
            site.get("buffer_km", 15),
            start_date, end_date,
        )

        dataset = _resolve_dataset(platform_str, pt_str)
        product_type = _resolve_product_type(pt_str)

        search_kwargs: dict = dict(
            intersectsWith=aoi.wkt,
            start=start_date,
            end=end_date,
        )

        if dataset:
            search_kwargs["dataset"] = dataset
            search_kwargs["processingLevel"] = product_type
        else:
            search_kwargs["platform"] = _resolve_platform(platform_str)
            search_kwargs["processingLevel"] = product_type
            if pt_str.upper() == "SLC" and platform_str.upper() in ("SENTINEL1", "SENTINEL-1"):
                search_kwargs["beamMode"] = asf.BEAMMODE.IW

        if acq.get("path_number"):
            search_kwargs["relativeOrbit"] = [acq["path_number"]]

        results = asf.search(**search_kwargs)
        scenes = [SceneInfo.from_asf_result(r, product_type=pt_str.upper()) for r in results]
        all_scenes.extend(scenes)

        logger.info("Found %d %s products", len(scenes), pt_str)

    all_scenes.sort(key=lambda s: s.start_time)

    max_scenes = acq.get("max_scenes", 200)
    if len(all_scenes) > max_scenes:
        logger.warning(
            "Found %d products, limiting to %d most recent",
            len(all_scenes), max_scenes,
        )
        all_scenes = all_scenes[-max_scenes:]

    logger.info(
        "Total: %d products across %d orbital tracks",
        len(all_scenes), _count_tracks(all_scenes),
    )
    return all_scenes


def _count_tracks(scenes: list[SceneInfo]) -> int:
    return len({s.path_number for s in scenes})


def select_track(
    scenes: list[SceneInfo], path_number: int | None = None
) -> list[SceneInfo]:
    """
    Select scenes from a single orbital track for consistent viewing geometry.

    If path_number is None, selects the track with the most scenes
    (maximizing temporal sampling).
    """
    tracks: dict[int, list[SceneInfo]] = {}
    for s in scenes:
        tracks.setdefault(s.path_number, []).append(s)

    if path_number is not None:
        if path_number not in tracks:
            available = sorted(tracks.keys())
            raise ValueError(
                f"Track {path_number} not found. Available: {available}"
            )
        selected = tracks[path_number]
    else:
        path_number = max(tracks, key=lambda k: len(tracks[k]))
        selected = tracks[path_number]

    logger.info(
        "Selected track %d: %d products, %s to %s",
        path_number,
        len(selected),
        selected[0].start_time[:10],
        selected[-1].start_time[:10],
    )

    for p, s in sorted(tracks.items()):
        if p != path_number:
            logger.info("  Skipped track %d: %d products", p, len(s))

    return selected


def _create_session(
    username: str | None = None,
    password: str | None = None,
) -> asf.ASFSession:
    session = asf.ASFSession()

    if username and password:
        session.auth_with_creds(username, password)
        return session

    env_user = os.environ.get("EARTHDATA_USER")
    env_pass = os.environ.get("EARTHDATA_PASS")
    if env_user and env_pass:
        session.auth_with_creds(env_user, env_pass)
        return session

    # Fall through to ~/.netrc (requests reads it automatically)
    return session


def download_scenes(
    scenes: list[SceneInfo],
    output_dir: str | Path = "data/slc",
    n_workers: int = 4,
    username: str | None = None,
    password: str | None = None,
) -> list[Path]:
    """
    Download scenes/products from ASF.

    Requires an Earthdata Login account. Credentials can be provided
    directly, via environment variables (EARTHDATA_USER/EARTHDATA_PASS),
    or via ~/.netrc file (recommended).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_names = {p.stem for p in output_dir.iterdir() if p.is_file()}
    to_download = []
    for s in scenes:
        fname = _filename_from_url(s.url) if s.url else s.granule
        if Path(fname).stem not in existing_names:
            to_download.append(s)

    if len(to_download) < len(scenes):
        logger.info(
            "%d of %d products already downloaded, downloading %d remaining",
            len(scenes) - len(to_download),
            len(scenes),
            len(to_download),
        )

    if not to_download:
        logger.info("All products already downloaded")
        return [p for p in output_dir.iterdir() if p.is_file()]

    total_gb = sum(s.file_size_mb for s in to_download) / 1024
    logger.info(
        "Downloading %d products (%.1f GB) with %d workers",
        len(to_download),
        total_gb,
        n_workers,
    )

    session = _create_session(username, password)

    urls = [s.url for s in to_download]
    asf.download_urls(
        urls=urls,
        path=str(output_dir),
        session=session,
        processes=n_workers,
    )

    downloaded = [p for p in output_dir.iterdir() if p.is_file()]
    logger.info("Download complete: %d files in %s", len(downloaded), output_dir)
    return downloaded


def summarize_scenes(scenes: list[SceneInfo]) -> str:
    """Return a human-readable summary of a scene/product collection."""
    if not scenes:
        return "No scenes found."

    product_types = sorted({s.product_type for s in scenes})
    pt_label = "/".join(product_types)

    tracks = {}
    for s in scenes:
        tracks.setdefault(s.path_number, []).append(s)

    lines = [
        f"Total: {len(scenes)} {pt_label} products, "
        f"{len(tracks)} track(s), "
        f"{scenes[0].start_time[:10]} to {scenes[-1].start_time[:10]}",
        "",
    ]
    for path, track_scenes in sorted(tracks.items()):
        interval = "—"
        if len(track_scenes) > 1:
            from datetime import datetime

            dates = [
                datetime.fromisoformat(s.start_time.replace("Z", "+00:00"))
                for s in track_scenes
            ]
            gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
            nonzero_gaps = [g for g in gaps if g > 0]
            if nonzero_gaps:
                interval = f"median {sorted(nonzero_gaps)[len(nonzero_gaps)//2]}d"

        pt_counts = {}
        for s in track_scenes:
            pt_counts[s.product_type] = pt_counts.get(s.product_type, 0) + 1
        pt_str = ", ".join(f"{v} {k}" for k, v in sorted(pt_counts.items()))

        lines.append(
            f"  Track {path:>3d}: {pt_str}, "
            f"{track_scenes[0].start_time[:10]} → {track_scenes[-1].start_time[:10]}, "
            f"revisit {interval}"
        )

    total_gb = sum(s.file_size_mb for s in scenes) / 1024
    lines.append(f"\nTotal download: {total_gb:.1f} GB")
    return "\n".join(lines)
