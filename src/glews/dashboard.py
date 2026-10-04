"""
Tier 2 analyst review dashboard — lightweight web UI for glaciologists
to review flagged sites from the GLEWS anomaly detection pipeline.

Serves a single-page HTML dashboard using only Python's built-in
http.server module. No additional dependencies required beyond the
Python standard library.

Usage (via CLI):
    glews dashboard -c config/nepal_2026.yaml --data-dir output --port 8080
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Severity thresholds — derived from anomaly score when no risk_level
# is present in the GeoJSON properties (i.e. no Tier 1 assessment ran).
# ---------------------------------------------------------------------------
SEVERITY_CRITICAL_THRESHOLD = 4.0
SEVERITY_WARNING_THRESHOLD = 2.5

# Analyst classification choices (distinct from severity)
VALID_CLASSIFICATIONS = {"Watch", "Warning", "Cleared"}

_USER_AGENT = "GLEWS/0.1 (glacier-landslide-warning; https://github.com/glews)"
_API_TIMEOUT = 10


def _severity_from_score(score: float) -> str:
    """Derive a display severity from the composite anomaly score."""
    if score >= SEVERITY_CRITICAL_THRESHOLD:
        return "CRITICAL"
    if score >= SEVERITY_WARNING_THRESHOLD:
        return "WARNING"
    return "INFO"


def _sanitize_for_json(value):
    """Replace non-finite floats with None so browser JSON.parse succeeds."""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(v) for v in value]
    return value


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Flag loading
# ---------------------------------------------------------------------------

def load_flags(data_dir: str | Path) -> list[dict]:
    """
    Load anomaly flags from the output directory.

    Tries ``flags.geojson`` first (the standard report output), then
    falls back to ``flags.json`` (a plain list or ``{"flags": [...]}``)
    format.  Returns an empty list when the directory or files do not
    exist.

    Each returned dict has at least: flag_id, score, peak_zscore,
    mean_zscore, n_pixels, area_m2, acceleration_mm_yr2, center_lat,
    center_lon, severity.
    """
    data_dir = Path(data_dir)

    geojson_path = data_dir / "flags.geojson"
    json_path = data_dir / "flags.json"

    flags: list[dict] = []

    if geojson_path.exists():
        raw = json.loads(geojson_path.read_text())
        for feature in raw.get("features", []):
            props = dict(feature.get("properties", {}))
            coords = feature.get("geometry", {}).get("coordinates", [None, None])
            props["center_lon"] = coords[0]
            props["center_lat"] = coords[1]
            flags.append(props)
    elif json_path.exists():
        raw = json.loads(json_path.read_text())
        if isinstance(raw, list):
            flags = raw
        elif isinstance(raw, dict) and "flags" in raw:
            flags = raw["flags"]

    for f in flags:
        if "voight_r2" in f and "voight_fit" not in f:
            f["voight_fit"] = {
                "r_squared": f["voight_r2"],
                "days_to_failure": f.get("voight_days_to_failure"),
                "predicted_failure_date": f.get("predicted_failure_date"),
                "failure_window": f.get("failure_window"),
            }

    # Derive severity for every flag
    for f in flags:
        if "severity" not in f:
            risk = f.get("risk_level")
            if risk:
                f["severity"] = risk.upper()
            else:
                f["severity"] = _severity_from_score(f.get("score", 0))

    # Sanitize non-finite floats (NaN from step-change detector)
    flags = [_sanitize_for_json(f) for f in flags]

    return flags


# ---------------------------------------------------------------------------
# Analyst state persistence
# ---------------------------------------------------------------------------

def load_state(state_path: str | Path) -> dict:
    """Load analyst classifications from disk.  Returns {} on missing file."""
    state_path = Path(state_path)
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text())
    except (json.JSONDecodeError, OSError):
        logger.warning("Corrupted state file %s, starting fresh", state_path)
        return {}


def save_state(state_path: str | Path, state: dict) -> None:
    """Atomically persist analyst state via temp-file + rename."""
    state_path = Path(state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(state_path.parent), suffix=".tmp", prefix=".dashboard_"
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, str(state_path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# External API helpers
# ---------------------------------------------------------------------------

_api_cache: dict[str, tuple[object, float]] = {}
_cache_lock = threading.Lock()


def _fetch_json(url: str, *, data: bytes | None = None,
                headers: dict | None = None, timeout: int = _API_TIMEOUT):
    """Fetch JSON from a URL with identifying User-Agent."""
    hdrs = {"User-Agent": _USER_AGENT}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=data, headers=hdrs)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _cached_call(key: str, func, *args, ttl: int = 300):
    """Return cached result or call func and cache for ttl seconds."""
    with _cache_lock:
        if key in _api_cache:
            result, ts = _api_cache[key]
            if time.time() - ts < ttl:
                return result
    result = func(*args)
    with _cache_lock:
        _api_cache[key] = (result, time.time())
    return result


def _get_elevation(lat: float, lon: float) -> dict:
    """Query 3x3 DEM grid around point, compute slope and aspect."""
    delta = 0.001  # ~111 m spacing
    lats = []
    lons = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            lats.append(str(round(lat + dy * delta, 6)))
            lons.append(str(round(lon + dx * delta, 6)))
    url = (
        "https://api.open-meteo.com/v1/elevation?latitude="
        + ",".join(lats)
        + "&longitude="
        + ",".join(lons)
    )
    data = _fetch_json(url, timeout=15)
    e = data["elevation"]

    # Grid indices (row-major, south to north):
    # 0=SW  1=S   2=SE
    # 3=W   4=C   5=E
    # 6=NW  7=N   8=NE
    cell_x = delta * 111320.0 * math.cos(math.radians(lat))
    cell_y = delta * 110540.0

    # Horn's method
    dz_dx = ((e[2] + 2 * e[5] + e[8]) - (e[0] + 2 * e[3] + e[6])) / (8 * cell_x)
    dz_dy = ((e[6] + 2 * e[7] + e[8]) - (e[0] + 2 * e[1] + e[2])) / (8 * cell_y)

    slope_deg = math.degrees(math.atan(math.sqrt(dz_dx ** 2 + dz_dy ** 2)))
    aspect_math = math.degrees(math.atan2(-dz_dy, dz_dx))
    if aspect_math < 0:
        aspect_math += 360
    aspect_compass = (90 - aspect_math) % 360

    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    aspect_dir = dirs[int((aspect_compass + 22.5) % 360 / 45)]

    return {
        "elevation_m": round(e[4], 1),
        "slope_deg": round(slope_deg, 1),
        "aspect_deg": round(aspect_compass, 1),
        "aspect_dir": aspect_dir,
    }


_OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]


_NOMINATIM_CACHE: dict[str, list[dict]] = {}


def _nominatim_fallback(lat: float, lon: float, radius_km: int = 20) -> list[dict]:
    """Find nearby settlements via Nominatim when Overpass is down."""
    grid_key = f"{lat:.1f},{lon:.1f}"
    if grid_key in _NOMINATIM_CACHE:
        cached = _NOMINATIM_CACHE[grid_key]
        return [
            {**f, "distance_km": round(_haversine(lat, lon, f["lat"], f["lon"]), 2)}
            for f in cached
            if _haversine(lat, lon, f["lat"], f["lon"]) <= radius_km
        ]

    delta = radius_km / 111.0
    features: list[dict] = []
    seen: set[str] = set()

    def _add(name: str, ftype: str, flat: float, flon: float,
             name_en: str = ""):
        key = f"{name}:{flat:.3f}"
        if key in seen or not name:
            return
        seen.add(key)
        entry = {
            "name": name, "type": ftype,
            "lat": round(flat, 4), "lon": round(flon, 4),
            "distance_km": 0.0,
        }
        if name_en and name_en != name:
            entry["name_en"] = name_en
        features.append(entry)

    try:
        url = (
            "https://nominatim.openstreetmap.org/search?format=jsonv2"
            f"&q=village&viewbox={lon - delta},{lat + delta},{lon + delta},{lat - delta}"
            "&bounded=1&limit=50&namedetails=1"
        )
        data = _fetch_json(url, timeout=10)
        for p in data:
            ptype = p.get("type", "village")
            if ptype in ("administrative", "county", "state", "country"):
                continue
            nd = p.get("namedetails", {})
            local_name = nd.get("name", p.get("name", ""))
            en_name = (
                nd.get("name:en")
                or nd.get("name:zh-Latn-pinyin")
                or nd.get("int_name")
                or ""
            )
            _add(local_name, ptype,
                 float(p.get("lat", 0)), float(p.get("lon", 0)),
                 name_en=en_name)
    except Exception as exc:
        logger.info("Nominatim search failed: %s", exc)

    offsets = [(-0.08, -0.08), (-0.08, 0.08), (0, 0),
               (0.08, -0.08), (0.08, 0.08)]
    for dlat, dlon in offsets:
        rlat, rlon = lat + dlat, lon + dlon
        try:
            url = (
                "https://nominatim.openstreetmap.org/reverse?format=jsonv2"
                f"&lat={rlat}&lon={rlon}&zoom=16&namedetails=1"
            )
            time.sleep(1.05)
            data = _fetch_json(url, timeout=10)
            rtype = data.get("type", "")
            if rtype not in ("administrative", "county", "state", "country", ""):
                nd = data.get("namedetails", {})
                local_name = nd.get("name", data.get("name", ""))
                en_name = (
                    nd.get("name:en")
                    or nd.get("name:zh-Latn-pinyin")
                    or nd.get("int_name")
                    or ""
                )
                _add(local_name, rtype,
                     float(data.get("lat", 0)), float(data.get("lon", 0)),
                     name_en=en_name)
        except Exception:
            pass

    _NOMINATIM_CACHE[grid_key] = features
    return [
        {**f, "distance_km": round(_haversine(lat, lon, f["lat"], f["lon"]), 2)}
        for f in features
        if _haversine(lat, lon, f["lat"], f["lon"]) <= radius_km
    ]


def _get_exposure(lat: float, lon: float) -> dict:
    """Query Overpass for settlements/infrastructure; Nominatim fallback."""
    radius_m = 20000
    query = (
        f"[out:json][timeout:25];\n"
        f"(\n"
        f'  node["place"~"village|town|city|hamlet"](around:{radius_m},{lat},{lon});\n'
        f'  node["amenity"~"hospital|school"](around:{radius_m},{lat},{lon});\n'
        f");\n"
        f"out body;"
    )
    post_data = urllib.parse.urlencode({"data": query}).encode()
    result = None
    for url in _OVERPASS_ENDPOINTS:
        try:
            result = _fetch_json(url, data=post_data, timeout=10)
            break
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.info("Overpass %s failed: %s", url.split("/")[2], exc)

    features = []
    if result is not None:
        for elem in result.get("elements", []):
            tags = elem.get("tags", {})
            name = tags.get("name", tags.get("place", tags.get("amenity", "unknown")))
            name_en = (
                tags.get("name:en")
                or tags.get("name:zh-Latn-pinyin")
                or tags.get("int_name")
                or ""
            )
            feat_type = tags.get("place") or tags.get("amenity") or "unknown"
            el_lat = elem.get("lat")
            el_lon = elem.get("lon")
            if el_lat is None or el_lon is None:
                continue
            dist = _haversine(lat, lon, el_lat, el_lon)
            entry = {
                "name": name,
                "type": feat_type,
                "lat": round(el_lat, 4),
                "lon": round(el_lon, 4),
                "distance_km": round(dist, 2),
            }
            if name_en and name_en != name:
                entry["name_en"] = name_en
            features.append(entry)
    else:
        features = _nominatim_fallback(lat, lon)
        if not features:
            return {"radius_km": 20, "features": [], "unavailable": True}

    features.sort(key=lambda x: x["distance_km"])
    return {"radius_km": 20, "features": features[:20]}


def _get_sentinel2(lat: float, lon: float) -> dict:
    """Query Element84 STAC for recent Sentinel-2 scenes."""
    bbox_delta = 0.02
    bbox = [lon - bbox_delta, lat - bbox_delta, lon + bbox_delta, lat + bbox_delta]
    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=30)).strftime("%Y-%m-%dT00:00:00Z")
    end = now.strftime("%Y-%m-%dT23:59:59Z")

    payload = json.dumps({
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "datetime": f"{start}/{end}",
        "limit": 5,
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
    }).encode()

    result = _fetch_json(
        "https://earth-search.aws.element84.com/v1/search",
        data=payload,
        headers={"Content-Type": "application/json"},
        timeout=_API_TIMEOUT,
    )

    scenes = []
    for feat in result.get("features", [])[:5]:
        props = feat.get("properties", {})
        dt = props.get("datetime", "")[:10]
        cloud = props.get("eo:cloud_cover")
        assets = feat.get("assets", {})
        thumbnail = assets.get("thumbnail", {}).get("href")
        scene = {
            "date": dt,
            "cloud_pct": round(cloud, 1) if cloud is not None else None,
        }
        if thumbnail:
            scene["thumbnail"] = thumbnail
        scenes.append(scene)

    return {
        "total_scenes": result.get("numberMatched", 0),
        "scenes": scenes,
    }


def _get_weather(lat: float, lon: float) -> dict:
    """Query Open-Meteo for current weather conditions."""
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,"
        f"precipitation,wind_speed_10m,cloud_cover"
        f"&timezone=auto"
    )
    data = _fetch_json(url)
    current = data.get("current", {})
    return {
        "temperature": current.get("temperature_2m"),
        "humidity": current.get("relative_humidity_2m"),
        "wind_speed": current.get("wind_speed_10m"),
        "cloud_cover": current.get("cloud_cover"),
        "precipitation": current.get("precipitation"),
    }


def _get_history(lat: float, lon: float, data_dir: Path) -> dict:
    """Scan monitor_state for prior flags within 1 km of this location."""
    state_dir = data_dir / "monitor_state"
    if not state_dir.exists():
        return {"message": "No prior monitoring runs found", "prior_flags": []}

    prior = []
    for fpath in sorted(state_dir.glob("*.geojson")):
        try:
            raw = json.loads(fpath.read_text())
            for feat in raw.get("features", []):
                props = feat.get("properties", {})
                coords = feat.get("geometry", {}).get("coordinates", [None, None])
                if coords[0] is None or coords[1] is None:
                    continue
                dist_km = _haversine(lat, lon, coords[1], coords[0])
                if dist_km < 1.0:
                    score = props.get("score", 0)
                    sev = props.get("risk_level", "")
                    if sev:
                        sev = sev.upper()
                    else:
                        sev = _severity_from_score(score)
                    prior.append({
                        "date": fpath.stem,
                        "score": score,
                        "severity": sev,
                        "distance_m": round(dist_km * 1000, 1),
                        "flag_id": props.get("flag_id"),
                    })
        except (json.JSONDecodeError, OSError):
            continue

    prior.sort(key=lambda x: x["date"], reverse=True)
    return {"prior_flags": prior[:20]}


def _get_monitoring_cadence(lat: float, lon: float) -> dict:
    """Query ASF SearchAPI for Sentinel-1 acquisition cadence."""
    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=90)).strftime("%Y-%m-%dT00:00:00Z")
    end = now.strftime("%Y-%m-%dT23:59:59Z")
    bbox_delta = 0.1
    bbox = f"{lon - bbox_delta},{lat - bbox_delta},{lon + bbox_delta},{lat + bbox_delta}"

    scenes_asc: list[str] = []
    scenes_desc: list[str] = []

    try:
        url = (
            "https://api.daac.asf.alaska.edu/services/search/param?"
            f"platform=SENTINEL-1&bbox={bbox}"
            f"&start={start}&end={end}"
            "&processingLevel=SLC&output=json"
        )
        data = _fetch_json(url, timeout=15)
        if isinstance(data, list) and data and isinstance(data[0], list):
            results = data[0]
        elif isinstance(data, list):
            results = data
        else:
            results = data.get("results", data.get("features", []))
        for item in results:
            props = item if "startTime" in item else item.get("properties", item)
            dt = (props.get("startTime") or props.get("datetime", ""))[:10]
            direction = props.get("flightDirection", props.get("orbit_direction", ""))
            if not dt:
                continue
            if direction.upper() == "ASCENDING":
                scenes_asc.append(dt)
            elif direction.upper() == "DESCENDING":
                scenes_desc.append(dt)
            else:
                scenes_desc.append(dt)
    except Exception as exc:
        logger.info("ASF search failed: %s", exc)
        return {
            "error_detail": str(exc),
            "total_scenes": 0,
            "ascending": [],
            "descending": [],
            "cadence_days": None,
            "next_expected": None,
        }

    scenes_asc.sort()
    scenes_desc.sort()
    all_scenes = sorted(set(scenes_asc + scenes_desc))

    cadence_days = None
    if len(all_scenes) >= 2:
        gaps = []
        for i in range(1, len(all_scenes)):
            d1 = datetime.strptime(all_scenes[i - 1], "%Y-%m-%d")
            d2 = datetime.strptime(all_scenes[i], "%Y-%m-%d")
            gaps.append((d2 - d1).days)
        cadence_days = round(sum(gaps) / len(gaps), 1)

    next_expected = None
    if all_scenes and cadence_days:
        last = datetime.strptime(all_scenes[-1], "%Y-%m-%d")
        next_dt = last + timedelta(days=round(cadence_days))
        next_expected = next_dt.strftime("%Y-%m-%d")

    return {
        "total_scenes": len(all_scenes),
        "ascending": scenes_asc[-5:],
        "descending": scenes_desc[-5:],
        "cadence_days": cadence_days,
        "last_acquisition": all_scenes[-1] if all_scenes else None,
        "next_expected": next_expected,
    }


def _get_instruments(lat: float, lon: float, data_dir: Path) -> dict:
    """Read deployed in-situ instruments from instruments.json."""
    instruments_path = data_dir / "instruments.json"
    if not instruments_path.exists():
        return {"instruments": [], "message": "No instruments.json found"}

    try:
        raw = json.loads(instruments_path.read_text())
        instruments = raw if isinstance(raw, list) else raw.get("instruments", [])
    except (json.JSONDecodeError, OSError):
        return {"instruments": [], "message": "Error reading instruments.json"}

    nearby = []
    for inst in instruments:
        ilat = inst.get("lat")
        ilon = inst.get("lon")
        if ilat is None or ilon is None:
            continue
        dist = _haversine(lat, lon, ilat, ilon)
        if dist <= 25:
            readings = inst.get("readings", [])
            last_reading = readings[-1] if readings else None
            trend = None
            if len(readings) >= 2:
                prev_val = readings[-2].get("value", 0)
                cur_val = readings[-1].get("value", 0)
                if cur_val > prev_val * 1.01:
                    trend = "increasing"
                elif cur_val < prev_val * 0.99:
                    trend = "decreasing"
                else:
                    trend = "stable"
            nearby.append({
                "id": inst.get("id", "unknown"),
                "type": inst.get("type", "unknown"),
                "lat": ilat,
                "lon": ilon,
                "distance_km": round(dist, 2),
                "status": inst.get("status", "unknown"),
                "installed_date": inst.get("installed_date"),
                "last_reading": last_reading,
                "trend": trend,
            })

    nearby.sort(key=lambda x: x["distance_km"])
    return {"instruments": nearby[:20]}


def _get_field_reports(lat: float, lon: float, data_dir: Path) -> dict:
    """Read field reconnaissance reports from field_reports.json."""
    reports_path = data_dir / "field_reports.json"
    if not reports_path.exists():
        return {"reports": [], "message": "No field_reports.json found"}

    try:
        raw = json.loads(reports_path.read_text())
        reports = raw if isinstance(raw, list) else raw.get("reports", [])
    except (json.JSONDecodeError, OSError):
        return {"reports": [], "message": "Error reading field_reports.json"}

    nearby = []
    for rpt in reports:
        rlat = rpt.get("lat")
        rlon = rpt.get("lon")
        if rlat is None or rlon is None:
            continue
        dist = _haversine(lat, lon, rlat, rlon)
        if dist <= 10:
            nearby.append({
                "id": rpt.get("id", "unknown"),
                "date": rpt.get("date"),
                "author": rpt.get("author", "Unknown"),
                "distance_km": round(dist, 2),
                "observations": rpt.get("observations", {}),
                "risk_assessment": rpt.get("risk_assessment", "unknown"),
                "notes": rpt.get("notes", ""),
            })

    nearby.sort(key=lambda x: x.get("date", ""), reverse=True)
    return {"reports": nearby[:10]}


def _get_news(lat: float, lon: float, region: str = "") -> dict:
    """Fetch related news from OpenAlex and Crossref."""
    query_terms = "landslide glacier hazard"
    if region:
        query_terms += " " + region

    articles: list[dict] = []

    try:
        q = urllib.parse.quote(query_terms)
        url = (
            f"https://api.openalex.org/works?search={q}"
            "&per_page=5&sort=publication_date:desc"
            "&select=title,doi,publication_date,primary_location"
        )
        data = _fetch_json(url, timeout=12)
        for r in data.get("results", []):
            title = r.get("title", "")
            if not title:
                continue
            loc = r.get("primary_location") or {}
            src = loc.get("source") or {}
            articles.append({
                "title": title,
                "url": r.get("doi", ""),
                "date": (r.get("publication_date") or "")[:10],
                "source": src.get("display_name", ""),
                "provider": "OpenAlex",
            })
    except Exception as exc:
        logger.info("OpenAlex query failed: %s", exc)

    try:
        q = urllib.parse.quote(query_terms)
        url = (
            f"https://api.crossref.org/works?query={q}"
            "&rows=5&sort=relevance&order=desc"
            "&select=title,URL,published-print,container-title"
        )
        data = _fetch_json(url, timeout=12)
        for it in data.get("message", {}).get("items", []):
            title_list = it.get("title", [])
            title = title_list[0] if title_list else ""
            if not title:
                continue
            journal_list = it.get("container-title", [])
            journal = journal_list[0] if journal_list else ""
            pub = it.get("published-print") or it.get("published-online") or {}
            parts = pub.get("date-parts", [[]])[0]
            year = str(parts[0]) if parts else ""
            articles.append({
                "title": title,
                "url": it.get("URL", ""),
                "date": year,
                "source": journal,
                "provider": "Crossref",
            })
    except Exception as exc:
        logger.info("Crossref query failed: %s", exc)

    seen_titles: set[str] = set()
    unique: list[dict] = []
    for a in articles:
        key = a["title"].lower()[:60]
        if key not in seen_titles:
            seen_titles.add(key)
            unique.append(a)

    return {"articles": unique[:8]}


def _resolve_flag_place_names(flags: list[dict], data_dir: Path) -> None:
    """Resolve nearest place name for each flag via one Overpass query.

    Writes results to place_names.json and annotates flags in-place
    with 'place_name' and 'place_name_en' keys.
    """
    cache_path = data_dir / "place_names.json"
    cached: dict[str, dict] = {}
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            cached = {}

    all_resolved = True
    for f in flags:
        fid = str(f.get("flag_id"))
        if fid in cached:
            f["place_name"] = cached[fid].get("name", "")
            f["place_name_en"] = cached[fid].get("name_en", "")
        else:
            all_resolved = False

    if all_resolved:
        return

    lats = [f["center_lat"] for f in flags
            if f.get("center_lat") is not None]
    lons = [f["center_lon"] for f in flags
            if f.get("center_lon") is not None]
    if not lats:
        return

    margin = 0.15
    south, north = min(lats) - margin, max(lats) + margin
    west, east = min(lons) - margin, max(lons) + margin

    query = (
        f"[out:json][timeout:30];\n"
        f'node["place"~"village|town|city|hamlet"]'
        f"({south},{west},{north},{east});\n"
        f"out body;"
    )
    post_data = urllib.parse.urlencode({"data": query}).encode()
    places: list[dict] = []
    for url in _OVERPASS_ENDPOINTS:
        try:
            result = _fetch_json(url, data=post_data, timeout=30)
            for elem in result.get("elements", []):
                tags = elem.get("tags", {})
                name = tags.get("name", "")
                if not name:
                    continue
                name_en = (
                    tags.get("name:en")
                    or tags.get("name:zh-Latn-pinyin")
                    or tags.get("int_name")
                    or ""
                )
                places.append({
                    "name": name,
                    "name_en": name_en,
                    "lat": elem["lat"],
                    "lon": elem["lon"],
                })
            break
        except Exception as exc:
            logger.info("Overpass place name query failed (%s): %s",
                        url.split("/")[2], exc)

    if not places:
        return

    for f in flags:
        fid = str(f.get("flag_id"))
        if fid in cached:
            continue
        flat = f.get("center_lat")
        flon = f.get("center_lon")
        if flat is None or flon is None:
            continue
        best = None
        best_dist = float("inf")
        for p in places:
            d = _haversine(flat, flon, p["lat"], p["lon"])
            if d < best_dist:
                best_dist = d
                best = p
        if best and best_dist < 30:
            entry = {"name": best["name"], "name_en": best.get("name_en", "")}
            cached[fid] = entry
            f["place_name"] = entry["name"]
            f["place_name_en"] = entry.get("name_en", "")

    try:
        cache_path.write_text(json.dumps(cached, ensure_ascii=False, indent=2))
    except OSError as exc:
        logger.warning("Could not write place_names.json: %s", exc)


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------
# Uses $$PLACEHOLDER$$ tokens to avoid conflicts with CSS/JS braces.

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GLEWS Tier 2 Analyst Dashboard</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; display: flex; height: 100vh; background: #f5f6fa; color: #2d3436; }

/* Sidebar */
.sidebar { width: 340px; min-width: 340px; background: #1e272e; color: #dfe6e9; display: flex; flex-direction: column; overflow: hidden; }
.sidebar-header { padding: 20px; border-bottom: 1px solid #485460; }
.sidebar-header h1 { font-size: 18px; color: #fff; margin-bottom: 4px; }
.sidebar-header .site-name { font-size: 13px; color: #b2bec3; }
.sidebar-stats { padding: 12px 20px; border-bottom: 1px solid #485460; display: flex; gap: 12px; }
.sidebar-stats .stat { text-align: center; flex: 1; }
.sidebar-stats .stat-value { font-size: 20px; font-weight: 700; color: #fff; font-variant-numeric: tabular-nums; }
.sidebar-stats .stat-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; color: #636e72; }
.sidebar-stats .stat-value.critical { color: #d63031; }
.sidebar-stats .stat-value.warning { color: #fdcb6e; }
.sidebar-stats .stat-value.info { color: #74b9ff; }
.sidebar-filter { padding: 12px 20px; border-bottom: 1px solid #485460; display: flex; gap: 6px; }
.sidebar-filter button { padding: 4px 10px; border: 1px solid #636e72; border-radius: 4px; background: transparent; color: #b2bec3; cursor: pointer; font-size: 12px; }
.sidebar-filter button.active { background: #0984e3; border-color: #0984e3; color: #fff; }
.flag-list { flex: 1; overflow-y: auto; }
.flag-item { padding: 14px 20px; border-bottom: 1px solid #2d3e50; cursor: pointer; transition: background 0.15s; }
.flag-item:hover { background: #2d3e50; }
.flag-item.selected { background: #0984e3; }
.flag-item .flag-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
.flag-item .flag-id { font-weight: 600; font-size: 14px; }
.flag-item .flag-score { font-size: 13px; color: #b2bec3; }
.flag-item.selected .flag-score { color: #dfe6e9; }
.flag-item .flag-meta { font-size: 12px; color: #636e72; }
.flag-item.selected .flag-meta { color: #b2bec3; }
.severity-badge { display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 11px; font-weight: 700; letter-spacing: 0.5px; }
.severity-CRITICAL { background: #d63031; color: #fff; }
.severity-WARNING { background: #fdcb6e; color: #2d3436; }
.severity-INFO { background: #74b9ff; color: #2d3436; }
.classification-badge { display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 11px; font-weight: 600; margin-left: 6px; }
.classification-Watch { background: #e17055; color: #fff; }
.classification-Warning { background: #fdcb6e; color: #2d3436; }
.classification-Cleared { background: #00b894; color: #fff; }

/* Main content */
.main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
.main-header { padding: 20px 30px; background: #fff; border-bottom: 1px solid #dfe6e9; }
.main-header h2 { font-size: 20px; color: #2d3436; }
.main-body { flex: 1; overflow-y: auto; padding: 30px; }

/* Detail card */
.detail-card { background: #fff; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); padding: 24px; margin-bottom: 20px; }
.detail-card h3 { font-size: 16px; margin-bottom: 16px; color: #2d3436; border-bottom: 1px solid #dfe6e9; padding-bottom: 10px; }
.metrics-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 16px; }
.metric { background: #f5f6fa; border-radius: 6px; padding: 14px; }
.metric .label { font-size: 11px; text-transform: uppercase; color: #636e72; letter-spacing: 0.5px; margin-bottom: 4px; }
.metric .value { font-size: 20px; font-weight: 700; color: #2d3436; }
.metric .unit { font-size: 12px; color: #636e72; margin-left: 2px; }

/* Map */
.map-container { margin-bottom: 20px; background: #fff; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); overflow: hidden; }
.map-container h3 { font-size: 16px; padding: 16px 24px 10px; color: #2d3436; border-bottom: 1px solid #dfe6e9; margin: 0; }
#flagMap { height: 300px; }

/* Chart area */
.chart-container { margin-top: 16px; padding: 16px; background: #f5f6fa; border-radius: 6px; text-align: center; }
.chart-container .no-data { color: #636e72; font-size: 14px; padding: 40px; }

/* Info panels */
.info-panels { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
.info-panel { background: #fff; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); overflow: hidden; }
.info-panel h3 { font-size: 14px; padding: 12px 16px; border-bottom: 1px solid #dfe6e9; color: #2d3436; margin: 0; display: flex; align-items: center; gap: 8px; }
.info-panel h3 .panel-icon { font-size: 16px; opacity: 0.7; }
.panel-content { padding: 16px; min-height: 80px; }
.panel-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(110px, 1fr)); gap: 10px; }
.panel-metric { text-align: center; padding: 8px; background: #f5f6fa; border-radius: 4px; }
.panel-metric .pm-label { font-size: 10px; text-transform: uppercase; color: #636e72; letter-spacing: 0.5px; margin-bottom: 2px; }
.panel-metric .pm-value { font-size: 16px; font-weight: 700; color: #2d3436; }
.panel-metric .pm-unit { font-size: 11px; color: #636e72; }
.panel-loading { color: #636e72; font-size: 13px; padding: 20px 0; text-align: center; }
.panel-error { color: #d63031; font-size: 13px; padding: 8px 0; }
.panel-empty { color: #636e72; font-size: 13px; padding: 8px 0; }
.panel-table { width: 100%; font-size: 13px; border-collapse: collapse; }
.panel-table th { text-align: left; padding: 6px 8px; border-bottom: 1px solid #dfe6e9; font-size: 11px; text-transform: uppercase; color: #636e72; letter-spacing: 0.3px; }
.panel-table td { padding: 6px 8px; border-bottom: 1px solid #f5f6fa; }
.escalation-banner { background: #d63031; color: #fff; padding: 8px 12px; border-radius: 4px; margin-top: 10px; font-size: 13px; font-weight: 600; }

/* Response protocol */
.response-protocol { background: #fff; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); padding: 24px; border-left: 4px solid #d63031; margin-bottom: 20px; }
.response-protocol h3 { font-size: 16px; margin-bottom: 16px; color: #d63031; border-bottom: 1px solid #dfe6e9; padding-bottom: 10px; }
.protocol-actions { display: flex; gap: 12px; flex-wrap: wrap; }
.protocol-btn { padding: 10px 18px; border: none; border-radius: 6px; font-size: 13px; font-weight: 600; cursor: pointer; transition: all 0.15s; }
.protocol-btn:hover { filter: brightness(0.9); }
.btn-notify { background: #0984e3; color: #fff; }
.btn-evacuate { background: #d63031; color: #fff; }
.draft-output { margin-top: 16px; background: #f5f6fa; border-radius: 6px; padding: 16px; font-size: 13px; line-height: 1.6; white-space: pre-wrap; font-family: -apple-system, BlinkMacSystemFont, sans-serif; position: relative; max-height: 400px; overflow-y: auto; }
.draft-output .draft-header { font-weight: 700; font-size: 14px; margin-bottom: 8px; color: #2d3436; }
.copy-btn { position: absolute; top: 8px; right: 8px; padding: 4px 12px; border: 1px solid #dfe6e9; border-radius: 4px; background: #fff; font-size: 12px; cursor: pointer; }
.copy-btn:hover { background: #dfe6e9; }

/* Instrument status */
.status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 4px; vertical-align: middle; }
.status-active { background: #00b894; }
.status-offline { background: #d63031; }
.status-maintenance { background: #fdcb6e; }
.status-unknown { background: #636e72; }
.trend-arrow { font-size: 14px; margin-left: 4px; }
.trend-increasing { color: #d63031; }
.trend-decreasing { color: #00b894; }
.trend-stable { color: #636e72; }
.risk-badge { display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 11px; font-weight: 700; letter-spacing: 0.5px; }
.risk-critical { background: #d63031; color: #fff; }
.risk-high { background: #e17055; color: #fff; }
.risk-moderate { background: #fdcb6e; color: #2d3436; }
.risk-low { background: #00b894; color: #fff; }
.risk-unknown { background: #636e72; color: #fff; }
.enhance-toggle { display: flex; align-items: center; gap: 8px; margin-top: 10px; padding: 8px 12px; background: #f5f6fa; border-radius: 4px; }
.enhance-toggle button { padding: 4px 12px; border: 1px solid #0984e3; border-radius: 4px; background: #fff; color: #0984e3; font-size: 12px; font-weight: 600; cursor: pointer; transition: all 0.15s; }
.enhance-toggle button.active { background: #0984e3; color: #fff; }
.enhance-toggle button:hover { filter: brightness(0.95); }
.obs-list { list-style: none; padding: 0; margin: 0; font-size: 13px; }
.obs-list li { padding: 4px 0; border-bottom: 1px solid #f5f6fa; }
.obs-label { color: #636e72; font-size: 11px; text-transform: uppercase; letter-spacing: 0.3px; }

/* Analyst actions */
.actions-card { background: #fff; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); padding: 24px; }
.actions-card h3 { font-size: 16px; margin-bottom: 16px; color: #2d3436; border-bottom: 1px solid #dfe6e9; padding-bottom: 10px; }
.action-row { display: flex; gap: 12px; align-items: flex-start; flex-wrap: wrap; }
.action-row .btn-group { display: flex; gap: 8px; }
.action-row button { padding: 8px 20px; border: 2px solid #dfe6e9; border-radius: 6px; background: #fff; cursor: pointer; font-size: 14px; font-weight: 600; transition: all 0.15s; }
.action-row button:hover { border-color: #0984e3; }
.action-row button.active-Watch { background: #e17055; color: #fff; border-color: #e17055; }
.action-row button.active-Warning { background: #fdcb6e; color: #2d3436; border-color: #fdcb6e; }
.action-row button.active-Cleared { background: #00b894; color: #fff; border-color: #00b894; }
.note-area { flex: 1; min-width: 250px; }
.note-area textarea { width: 100%; height: 70px; padding: 10px; border: 2px solid #dfe6e9; border-radius: 6px; font-family: inherit; font-size: 13px; resize: vertical; }
.note-area textarea:focus { outline: none; border-color: #0984e3; }
.save-indicator { font-size: 13px; color: #00b894; margin-top: 8px; opacity: 0; transition: opacity 0.3s; }
.save-indicator.visible { opacity: 1; }

/* Empty state */
.empty-state { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; color: #636e72; }
.empty-state .icon { font-size: 48px; margin-bottom: 16px; }
.empty-state p { font-size: 16px; }

@media (max-width: 900px) {
    .info-panels { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<div class="sidebar">
    <div class="sidebar-header">
        <h1>GLEWS Dashboard</h1>
        <div class="site-name" id="siteName">$$SITE_NAME$$</div>
    </div>
    <div class="sidebar-stats" id="sidebarStats"></div>
    <div class="sidebar-filter">
        <button class="active" onclick="filterFlags('ALL')">All</button>
        <button onclick="filterFlags('CRITICAL')">Critical</button>
        <button onclick="filterFlags('WARNING')">Warning</button>
        <button onclick="filterFlags('INFO')">Info</button>
    </div>
    <div class="flag-list" id="flagList"></div>
</div>
<div class="main">
    <div class="main-header">
        <h2 id="mainTitle">Tier 2 Analyst Review</h2>
    </div>
    <div class="main-body" id="mainBody">
        <div class="empty-state">
            <div class="icon">&#9650;</div>
            <p>Select a flagged site from the sidebar to review</p>
        </div>
    </div>
</div>

<script>
var FLAGS = $$FLAGS_JSON$$;
var STATE = $$STATE_JSON$$;
var currentFilter = 'ALL';
var selectedFlagId = null;
var cachedExposure = null;

function filterFlags(severity) {
    currentFilter = severity;
    document.querySelectorAll('.sidebar-filter button').forEach(function(btn) {
        btn.classList.toggle('active', btn.textContent.toLowerCase() === severity.toLowerCase() || (severity === 'ALL' && btn.textContent === 'All'));
    });
    renderFlagList();
}

function renderFlagList() {
    var list = document.getElementById('flagList');
    var filtered = FLAGS;
    if (currentFilter !== 'ALL') {
        filtered = FLAGS.filter(function(f) { return f.severity === currentFilter; });
    }
    var html = '';
    for (var i = 0; i < filtered.length; i++) {
        var f = filtered[i];
        var fid = String(f.flag_id);
        var st = STATE[fid] || {};
        var sel = (f.flag_id === selectedFlagId) ? ' selected' : '';
        html += '<div class="flag-item' + sel + '" onclick="selectFlag(' + f.flag_id + ')">';
        html += '<div class="flag-header">';
        html += '<span class="flag-id">Flag ' + f.flag_id + '</span>';
        html += '<span><span class="severity-badge severity-' + f.severity + '">' + f.severity + '</span>';
        if (st.classification) {
            html += '<span class="classification-badge classification-' + st.classification + '">' + st.classification + '</span>';
        }
        html += '</span></div>';
        if (f.place_name) {
            var placeLabel = escapeHtml(f.place_name);
            if (f.place_name_en && f.place_name_en !== f.place_name) placeLabel += ' (' + escapeHtml(f.place_name_en) + ')';
            html += '<div class="flag-meta" style="margin-bottom:2px;">' + placeLabel + '</div>';
        }
        html += '<div class="flag-meta">Score: ' + (f.score != null ? f.score.toFixed(2) : 'N/A');
        html += ' &middot; Z: ' + (f.peak_zscore != null ? f.peak_zscore.toFixed(1) : 'N/A');
        html += ' &middot; ' + formatArea(f.area_m2) + '</div>';
        html += '<div class="flag-meta">' + formatCoord(f.center_lat, f.center_lon) + '</div>';
        html += '</div>';
    }
    if (filtered.length === 0) {
        html = '<div style="padding:30px 20px;color:#636e72;text-align:center;">No flags match this filter</div>';
    }
    list.innerHTML = html;
}

function renderStats() {
    var total = FLAGS.length;
    var crit = 0, warn = 0, info = 0;
    for (var i = 0; i < FLAGS.length; i++) {
        if (FLAGS[i].severity === 'CRITICAL') crit++;
        else if (FLAGS[i].severity === 'WARNING') warn++;
        else info++;
    }
    var el = document.getElementById('sidebarStats');
    el.innerHTML =
        '<div class="stat"><div class="stat-value">' + total + '</div><div class="stat-label">Total</div></div>' +
        '<div class="stat"><div class="stat-value critical">' + crit + '</div><div class="stat-label">Critical</div></div>' +
        '<div class="stat"><div class="stat-value warning">' + warn + '</div><div class="stat-label">Warning</div></div>' +
        '<div class="stat"><div class="stat-value info">' + info + '</div><div class="stat-label">Info</div></div>';
}

function formatArea(a) {
    if (a == null) return 'N/A';
    if (a >= 1e6) return (a / 1e6).toFixed(2) + ' km²';
    return a.toLocaleString() + ' m²';
}

function formatCoord(lat, lon) {
    if (lat == null || lon == null) return '';
    return lat.toFixed(4) + '°N, ' + lon.toFixed(4) + '°E';
}

function selectFlag(flagId) {
    selectedFlagId = flagId;
    renderFlagList();
    renderDetail(flagId);
}

function escapeHtml(s) {
    var div = document.createElement('div');
    div.appendChild(document.createTextNode(s));
    return div.innerHTML;
}

function metric(label, value, unit) {
    return '<div class="metric"><div class="label">' + label + '</div><div class="value">' + value + '<span class="unit">' + unit + '</span></div></div>';
}

function panelMetric(label, value, unit) {
    return '<div class="panel-metric"><div class="pm-label">' + escapeHtml(String(label)) + '</div><div class="pm-value">' + escapeHtml(String(value)) + '<span class="pm-unit">' + escapeHtml(String(unit)) + '</span></div></div>';
}

function renderDetail(flagId) {
    var f = FLAGS.find(function(x) { return x.flag_id === flagId; });
    if (!f) return;
    var fid = String(flagId);
    var st = STATE[fid] || {};
    var body = document.getElementById('mainBody');
    var accel = f.acceleration_mm_yr2;
    var accelStr = (accel != null) ? accel.toFixed(2) + ' mm/yr²' : 'N/A';

    var html = '<div class="map-container"><h3>Location</h3><div id="flagMap"></div></div>';
    html += '<div class="detail-card">';
    var locationLabel = formatCoord(f.center_lat, f.center_lon);
    if (f.place_name) {
        var pn = escapeHtml(f.place_name);
        if (f.place_name_en && f.place_name_en !== f.place_name) pn += ' (' + escapeHtml(f.place_name_en) + ')';
        locationLabel = pn + ' — ' + locationLabel;
    }
    html += '<h3>Flag ' + f.flag_id + ' — ' + locationLabel;
    html += ' <span class="severity-badge severity-' + f.severity + '">' + f.severity + '</span></h3>';
    html += '<div class="metrics-grid">';
    html += metric('Anomaly Score', f.score != null ? f.score.toFixed(2) : 'N/A', '');
    html += metric('Peak Z-Score', f.peak_zscore != null ? f.peak_zscore.toFixed(1) : 'N/A', 'σ');
    html += metric('Mean Z-Score', f.mean_zscore != null ? f.mean_zscore.toFixed(1) : 'N/A', 'σ');
    html += metric('Area', formatArea(f.area_m2), '');
    html += metric('Pixels', f.n_pixels != null ? f.n_pixels.toLocaleString() : 'N/A', '');
    html += metric('Acceleration', accelStr, '');
    html += '</div>';

    // Chart area
    html += '<div class="chart-container">';
    if (f.timeseries && f.timeseries.dates && f.timeseries.values) {
        html += renderTimeseriesSVG(f.timeseries.dates, f.timeseries.values, f.flag_id);
    } else {
        html += '<div class="no-data">No displacement time-series data available for this flag.<br>';
        html += '<span style="font-size:12px;color:#b2bec3;">Run the full pipeline with time-series export to populate this chart.</span></div>';
    }
    html += '</div>';
    html += '</div>';

    // --- Info panels row 1: Voight + Elevation ---
    html += '<div class="info-panels">';
    html += '<div class="info-panel"><h3><span class="panel-icon">&#9888;</span> Failure Forecast (Voight)</h3><div class="panel-content" id="panel-voight"></div></div>';
    html += '<div class="info-panel"><h3><span class="panel-icon">&#9650;</span> Elevation &amp; Slope</h3><div class="panel-content" id="panel-elevation"></div></div>';
    html += '</div>';

    // --- Info panels row 2: Sentinel-2 + Weather ---
    html += '<div class="info-panels">';
    html += '<div class="info-panel"><h3><span class="panel-icon">&#128752;</span> Sentinel-2 Optical</h3><div class="panel-content" id="panel-sentinel2"></div></div>';
    html += '<div class="info-panel"><h3><span class="panel-icon">&#9729;</span> Current Weather</h3><div class="panel-content" id="panel-weather"></div></div>';
    html += '</div>';

    // --- Info panels row 3: Exposure + History ---
    html += '<div class="info-panels">';
    html += '<div class="info-panel"><h3><span class="panel-icon">&#127968;</span> Downstream Exposure</h3><div class="panel-content" id="panel-exposure"></div></div>';
    html += '<div class="info-panel"><h3><span class="panel-icon">&#128197;</span> Flag History</h3><div class="panel-content" id="panel-history"></div></div>';
    html += '</div>';

    // --- Info panels row 4: Monitoring Cadence + In-Situ Instruments ---
    html += '<div class="info-panels">';
    html += '<div class="info-panel"><h3><span class="panel-icon">&#128225;</span> Monitoring Cadence</h3><div class="panel-content" id="panel-monitoring"></div></div>';
    html += '<div class="info-panel"><h3><span class="panel-icon">&#128296;</span> In-Situ Instruments</h3><div class="panel-content" id="panel-instruments"></div></div>';
    html += '</div>';

    // --- Info panels row 5: Field Reconnaissance ---
    html += '<div class="info-panels">';
    html += '<div class="info-panel" style="grid-column: 1 / -1;"><h3><span class="panel-icon">&#128269;</span> Field Reconnaissance</h3><div class="panel-content" id="panel-fieldreports"></div></div>';
    html += '</div>';

    // --- Info panels row 6: News Feed ---
    html += '<div class="info-panels">';
    html += '<div class="info-panel" style="grid-column: 1 / -1;"><h3><span class="panel-icon">&#128240;</span> News Feed</h3><div class="panel-content" id="panel-news"></div></div>';
    html += '</div>';

    // Response protocol (shown when Voight escalation triggers)
    html += '<div class="response-protocol" id="responseProtocol" style="display:none;">';
    html += '<h3>Response Protocol</h3>';
    html += '<div class="protocol-actions">';
    html += '<button class="protocol-btn btn-notify" onclick="draftNotification()">Draft Authority Notification</button>';
    html += '<button class="protocol-btn btn-evacuate" onclick="draftEvacuation()">Draft Evacuation Advisory</button>';
    html += '</div>';
    html += '<div id="draftOutput"></div>';
    html += '</div>';

    // Actions card
    html += '<div class="actions-card">';
    html += '<h3>Analyst Classification</h3>';
    html += '<div class="action-row">';
    html += '<div class="btn-group">';
    var classes = ['Watch', 'Warning', 'Cleared'];
    for (var i = 0; i < classes.length; i++) {
        var c = classes[i];
        var active = (st.classification === c) ? ' active-' + c : '';
        html += '<button class="' + active + '" onclick="classify(' + flagId + ',\'' + c + '\')">' + c + '</button>';
    }
    html += '</div>';
    html += '<div class="note-area">';
    html += '<textarea id="noteInput" placeholder="Add analyst notes..." onchange="saveNote(' + flagId + ')">' + escapeHtml(st.note || '') + '</textarea>';
    html += '</div>';
    html += '</div>';
    if (st.updated_at) {
        html += '<div style="font-size:12px;color:#636e72;margin-top:10px;">Last updated: ' + escapeHtml(st.updated_at) + '</div>';
    }
    html += '<div class="save-indicator" id="saveIndicator">Saved</div>';
    html += '</div>';

    body.innerHTML = html;
    var titleText = 'Flag ' + flagId;
    if (f.place_name) {
        titleText += ' — ' + f.place_name;
        if (f.place_name_en && f.place_name_en !== f.place_name) titleText += ' (' + f.place_name_en + ')';
    }
    titleText += ' — Detail View';
    document.getElementById('mainTitle').textContent = titleText;
    initMap(f);

    // Populate panels
    renderVoightPanel(f);
    if (f.center_lat != null && f.center_lon != null) {
        loadPanel('elevation', f.center_lat, f.center_lon);
        loadPanel('sentinel2', f.center_lat, f.center_lon);
        loadPanel('weather', f.center_lat, f.center_lon);
        loadPanel('exposure', f.center_lat, f.center_lon);
        loadPanel('history', f.center_lat, f.center_lon);
        loadPanel('monitoring', f.center_lat, f.center_lon);
        loadPanel('instruments', f.center_lat, f.center_lon);
        loadPanel('fieldreports', f.center_lat, f.center_lon);
        var newsRegion = '';
        if (f.place_name_en) newsRegion = f.place_name_en;
        else if (f.place_name) newsRegion = f.place_name;
        loadPanel('news', f.center_lat, f.center_lon, newsRegion);
    }
}

// ---- Map with basemap toggle ----
var dashMap = null;
function initMap(flag) {
    if (flag.center_lat == null || flag.center_lon == null) return;
    if (dashMap) { dashMap.remove(); dashMap = null; }

    var osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors',
        maxZoom: 18
    });
    var satellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: '© Esri',
        maxZoom: 18
    });

    dashMap = L.map('flagMap', { layers: [osm] }).setView([flag.center_lat, flag.center_lon], 13);
    L.control.layers({ 'Street': osm, 'Satellite': satellite }).addTo(dashMap);

    var sevColors = {CRITICAL: '#d63031', WARNING: '#fdcb6e', INFO: '#74b9ff'};
    for (var i = 0; i < FLAGS.length; i++) {
        var fl = FLAGS[i];
        if (fl.center_lat == null || fl.center_lon == null) continue;
        var col = sevColors[fl.severity] || '#636e72';
        var isSelected = (fl.flag_id === flag.flag_id);
        L.circleMarker([fl.center_lat, fl.center_lon], {
            radius: isSelected ? 10 : 5,
            color: isSelected ? '#fff' : col,
            weight: isSelected ? 3 : 1,
            fillColor: col,
            fillOpacity: isSelected ? 0.9 : 0.5
        }).addTo(dashMap).bindPopup('Flag ' + fl.flag_id + ' (' + fl.severity + ')');
    }
}

// ---- Async panel loading ----
function loadPanel(panel, lat, lon, region) {
    var el = document.getElementById('panel-' + panel);
    if (!el) return;
    el.innerHTML = '<div class="panel-loading">Loading…</div>';
    var requestedFlag = selectedFlagId;
    var xhr = new XMLHttpRequest();
    var url = '/api/' + panel + '?lat=' + lat + '&lon=' + lon;
    if (region) url += '&region=' + encodeURIComponent(region);
    xhr.open('GET', url, true);
    xhr.timeout = 45000;
    xhr.onload = function() {
        if (selectedFlagId !== requestedFlag) return;
        if (xhr.status === 200) {
            try {
                var data = JSON.parse(xhr.responseText);
                renderPanelData(panel, data, el);
            } catch (e) {
                el.innerHTML = '<div class="panel-error">Parse error</div>';
            }
        } else {
            el.innerHTML = '<div class="panel-error">Failed to load (' + xhr.status + ')</div>';
        }
    };
    xhr.onerror = function() {
        if (selectedFlagId !== requestedFlag) return;
        el.innerHTML = '<div class="panel-error">Network error</div>';
    };
    xhr.ontimeout = function() {
        if (selectedFlagId !== requestedFlag) return;
        el.innerHTML = '<div class="panel-error">Request timed out</div>';
    };
    xhr.send();
}

function renderPanelData(panel, data, el) {
    if (data.error) {
        el.innerHTML = '<div class="panel-error">' + escapeHtml(data.error) + '</div>';
        return;
    }
    switch (panel) {
        case 'elevation': renderElevationPanel(data, el); break;
        case 'sentinel2': renderSentinelPanel(data, el); break;
        case 'weather': renderWeatherPanel(data, el); break;
        case 'exposure': renderExposurePanel(data, el); break;
        case 'history': renderHistoryPanel(data, el); break;
        case 'monitoring': renderMonitoringPanel(data, el); break;
        case 'instruments': renderInstrumentsPanel(data, el); break;
        case 'fieldreports': renderFieldReportsPanel(data, el); break;
        case 'news': renderNewsPanel(data, el); break;
        default: el.innerHTML = '<div class="panel-empty">Unknown panel</div>';
    }
}

// ---- Panel renderers ----

function renderVoightPanel(flag) {
    var el = document.getElementById('panel-voight');
    if (!el) return;
    if (!flag.voight_fit || flag.voight_fit.r_squared == null) {
        el.innerHTML = '<div class="panel-empty">No Voight analysis available.<br><span style="font-size:12px;">Run the full pipeline with time-series export to enable failure forecasting.</span></div>';
        return;
    }
    var v = flag.voight_fit;
    var daysFromToday = null;
    if (v.predicted_failure_date) {
        var failMs = new Date(v.predicted_failure_date + 'T00:00:00').getTime();
        var nowMs = new Date().setHours(0,0,0,0);
        daysFromToday = Math.round((failMs - nowMs) / 86400000);
    }
    var daysLabel = 'N/A';
    if (daysFromToday != null) {
        daysLabel = daysFromToday > 0 ? daysFromToday : Math.abs(daysFromToday) + ' ago';
    }
    var html = '<div class="panel-grid">';
    html += panelMetric('Days to Failure', daysLabel, '');
    html += panelMetric('Fit R²', v.r_squared != null ? v.r_squared.toFixed(3) : 'N/A', '');
    html += panelMetric('Failure Date', v.predicted_failure_date || 'N/A', '');
    html += '</div>';
    if (v.failure_window) {
        html += '<div style="font-size:12px;color:#636e72;margin-top:8px;">95% confidence window: ' + escapeHtml(v.failure_window) + '</div>';
    }
    if (daysFromToday != null && daysFromToday <= 0) {
        html += '<div class="escalation-banner">Projected failure date has passed — immediate field verification required</div>';
    } else if (daysFromToday != null && daysFromToday <= 30) {
        html += '<div class="escalation-banner">Projected failure within 30 days — escalation recommended</div>';
    }
    el.innerHTML = html;
    updateProtocolVisibility();
}

function renderElevationPanel(data, el) {
    var html = '<div class="panel-grid">';
    html += panelMetric('Elevation', data.elevation_m, 'm');
    html += panelMetric('Slope', data.slope_deg, '°');
    html += panelMetric('Aspect', data.aspect_deg + '° ' + data.aspect_dir, '');
    html += '</div>';
    el.innerHTML = html;
}

function safeLink(url, text) {
    try {
        var u = new URL(url);
        if (u.protocol !== 'https:' && u.protocol !== 'http:') return escapeHtml(text);
        var safe = u.href.replace(/"/g, '&quot;');
        return '<a href="' + safe + '" target="_blank" rel="noopener noreferrer" style="color:#0984e3;text-decoration:none;">' + escapeHtml(text) + '</a>';
    } catch(e) { return escapeHtml(text); }
}

function renderSentinelPanel(data, el) {
    if (!data.scenes || data.scenes.length === 0) {
        el.innerHTML = '<div class="panel-empty">No recent Sentinel-2 scenes found</div>';
        return;
    }
    var html = '<div class="panel-grid">';
    html += panelMetric('Latest Scene', data.scenes[0].date, '');
    html += panelMetric('Cloud Cover', data.scenes[0].cloud_pct != null ? data.scenes[0].cloud_pct : 'N/A', '%');
    html += panelMetric('Scenes (30d)', data.total_scenes, '');
    html += '</div>';
    html += '<div style="font-size:12px;color:#636e72;margin-top:10px;">Recent scenes: ';
    for (var i = 0; i < Math.min(data.scenes.length, 5); i++) {
        if (i > 0) html += ', ';
        var s = data.scenes[i];
        var label = s.date;
        if (s.cloud_pct != null) label += ' (' + s.cloud_pct + '%)';
        if (s.thumbnail) {
            html += safeLink(s.thumbnail, label);
        } else {
            html += escapeHtml(label);
        }
    }
    html += '</div>';
    el.innerHTML = html;
}

function renderWeatherPanel(data, el) {
    var html = '<div class="panel-grid">';
    html += panelMetric('Temp', data.temperature != null ? data.temperature : 'N/A', '°C');
    html += panelMetric('Humidity', data.humidity != null ? data.humidity : 'N/A', '%');
    html += panelMetric('Wind', data.wind_speed != null ? data.wind_speed : 'N/A', ' km/h');
    html += panelMetric('Clouds', data.cloud_cover != null ? data.cloud_cover : 'N/A', '%');
    html += panelMetric('Precip', data.precipitation != null ? data.precipitation : 'N/A', ' mm');
    html += '</div>';
    el.innerHTML = html;
}

function renderExposurePanel(data, el) {
    cachedExposure = data;
    updateProtocolVisibility();
    if (data.unavailable) {
        el.innerHTML = '<div class="panel-empty">Overpass API unavailable — settlement data could not be loaded.<br><span style="font-size:12px;color:#636e72;">Try reloading the page later.</span></div>';
        return;
    }
    if (!data.features || data.features.length === 0) {
        el.innerHTML = '<div class="panel-empty">No settlements or infrastructure found within ' + (data.radius_km || 10) + ' km</div>';
        return;
    }
    var html = '<div style="font-size:11px;color:#636e72;margin-bottom:8px;">Within ' + data.radius_km + ' km (straight-line, not flow-routed)</div>';
    html += '<table class="panel-table"><tr><th>Name</th><th>Type</th><th>Dist</th></tr>';
    for (var i = 0; i < data.features.length; i++) {
        var feat = data.features[i];
        var displayName = escapeHtml(feat.name);
        if (feat.name_en) displayName += ' <span style="color:#636e72;font-size:12px;">(' + escapeHtml(feat.name_en) + ')</span>';
        html += '<tr><td>' + displayName + '</td>';
        html += '<td style="color:#636e72;">' + escapeHtml(feat.type) + '</td>';
        html += '<td>' + feat.distance_km.toFixed(1) + ' km</td></tr>';
    }
    html += '</table>';
    el.innerHTML = html;
}

function renderHistoryPanel(data, el) {
    if (!data.prior_flags || data.prior_flags.length === 0) {
        el.innerHTML = '<div class="panel-empty">' + escapeHtml(data.message || 'No prior flags found at this location') + '</div>';
        return;
    }
    var html = '<table class="panel-table"><tr><th>Date</th><th>Score</th><th>Severity</th><th>Dist</th></tr>';
    for (var i = 0; i < data.prior_flags.length; i++) {
        var pf = data.prior_flags[i];
        html += '<tr><td>' + escapeHtml(pf.date) + '</td>';
        html += '<td>' + (pf.score != null ? pf.score.toFixed(2) : 'N/A') + '</td>';
        html += '<td><span class="severity-badge severity-' + escapeHtml(pf.severity) + '">' + escapeHtml(pf.severity) + '</span></td>';
        html += '<td>' + pf.distance_m.toFixed(0) + ' m</td></tr>';
    }
    html += '</table>';
    el.innerHTML = html;
}

function renderMonitoringPanel(data, el) {
    if (data.error_detail) {
        el.innerHTML = '<div class="panel-error">ASF API unavailable: ' + escapeHtml(data.error_detail.substring(0, 80)) + '</div>';
        return;
    }
    if (data.total_scenes === 0) {
        el.innerHTML = '<div class="panel-empty">No Sentinel-1 acquisitions found in the last 90 days</div>';
        return;
    }
    var html = '<div class="panel-grid">';
    html += panelMetric('Scenes (90d)', data.total_scenes, '');
    html += panelMetric('Cadence', data.cadence_days != null ? data.cadence_days : 'N/A', ' days');
    html += panelMetric('Last Acq.', data.last_acquisition || 'N/A', '');
    html += panelMetric('Next Expected', data.next_expected || 'N/A', '');
    html += panelMetric('Ascending', data.ascending ? data.ascending.length : 0, '');
    html += panelMetric('Descending', data.descending ? data.descending.length : 0, '');
    html += '</div>';
    var f = getSelectedFlag();
    var fid = f ? String(f.flag_id) : '';
    var st = STATE[fid] || {};
    var enhanced = st.enhanced_monitoring || false;
    html += '<div class="enhance-toggle">';
    html += '<span style="font-size:12px;color:#636e72;">Enhanced Monitoring:</span>';
    html += '<button class="' + (enhanced ? 'active' : '') + '" onclick="toggleEnhancedMonitoring()">' + (enhanced ? 'Requested' : 'Request') + '</button>';
    if (st.enhanced_monitoring_at) {
        html += '<span style="font-size:11px;color:#636e72;">since ' + escapeHtml(st.enhanced_monitoring_at) + '</span>';
    }
    html += '</div>';
    el.innerHTML = html;
}

function toggleEnhancedMonitoring() {
    var f = getSelectedFlag();
    if (!f) return;
    var fid = String(f.flag_id);
    var st = STATE[fid] || {};
    var newVal = !st.enhanced_monitoring;
    var payload = JSON.stringify({flag_id: f.flag_id, enhanced: newVal});
    var xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/monitoring/enhance', true);
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.onload = function() {
        if (xhr.status === 200) {
            if (!STATE[fid]) STATE[fid] = {};
            STATE[fid].enhanced_monitoring = newVal;
            STATE[fid].enhanced_monitoring_at = new Date().toISOString().slice(0, 16).replace('T', ' ') + ' UTC';
            loadPanel('monitoring', f.center_lat, f.center_lon);
        }
    };
    xhr.send(payload);
}

function renderInstrumentsPanel(data, el) {
    if (data.message && (!data.instruments || data.instruments.length === 0)) {
        el.innerHTML = '<div class="panel-empty">' + escapeHtml(data.message) + '<br><span style="font-size:12px;color:#636e72;">Place an instruments.json file in the data directory to display deployed sensors.</span></div>';
        return;
    }
    if (!data.instruments || data.instruments.length === 0) {
        el.innerHTML = '<div class="panel-empty">No instruments deployed within 25 km</div>';
        return;
    }
    var html = '<table class="panel-table"><tr><th>Type</th><th>Status</th><th>Last Reading</th><th>Trend</th><th>Dist</th></tr>';
    for (var i = 0; i < data.instruments.length; i++) {
        var inst = data.instruments[i];
        var statusClass = 'status-' + (inst.status || 'unknown');
        var trendArrow = '';
        if (inst.trend === 'increasing') trendArrow = '<span class="trend-arrow trend-increasing">&#9650;</span>';
        else if (inst.trend === 'decreasing') trendArrow = '<span class="trend-arrow trend-decreasing">&#9660;</span>';
        else if (inst.trend === 'stable') trendArrow = '<span class="trend-arrow trend-stable">&#9644;</span>';
        var readingStr = 'N/A';
        if (inst.last_reading) {
            readingStr = inst.last_reading.value;
            if (inst.last_reading.unit) readingStr += ' ' + inst.last_reading.unit;
        }
        html += '<tr>';
        html += '<td>' + escapeHtml(inst.type) + '</td>';
        html += '<td><span class="status-dot ' + statusClass + '"></span>' + escapeHtml(inst.status || 'unknown') + '</td>';
        html += '<td>' + escapeHtml(String(readingStr)) + '</td>';
        html += '<td>' + trendArrow + '</td>';
        html += '<td>' + inst.distance_km.toFixed(1) + ' km</td>';
        html += '</tr>';
    }
    html += '</table>';
    el.innerHTML = html;
}

function renderFieldReportsPanel(data, el) {
    if (data.message && (!data.reports || data.reports.length === 0)) {
        el.innerHTML = '<div class="panel-empty">' + escapeHtml(data.message) + '<br><span style="font-size:12px;color:#636e72;">Place a field_reports.json file in the data directory to display reconnaissance data.</span></div>';
        return;
    }
    if (!data.reports || data.reports.length === 0) {
        el.innerHTML = '<div class="panel-empty">No field reports within 10 km</div>';
        return;
    }
    var html = '';
    for (var i = 0; i < data.reports.length; i++) {
        var rpt = data.reports[i];
        var riskClass = 'risk-' + (rpt.risk_assessment || 'unknown');
        html += '<div style="padding:10px;background:#f5f6fa;border-radius:6px;margin-bottom:8px;">';
        html += '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">';
        html += '<span style="font-weight:600;font-size:13px;">' + escapeHtml(rpt.date || 'Unknown date') + ' — ' + escapeHtml(rpt.author) + '</span>';
        html += '<span class="risk-badge ' + riskClass + '">' + escapeHtml((rpt.risk_assessment || 'unknown').toUpperCase()) + '</span>';
        html += '</div>';
        var obs = rpt.observations || {};
        var obsItems = [];
        if (obs.crack_width_cm != null) obsItems.push('Crack width: ' + obs.crack_width_cm + ' cm');
        if (obs.scarp_height_m != null) obsItems.push('Scarp height: ' + obs.scarp_height_m + ' m');
        if (obs.seepage) obsItems.push('Seepage: ' + obs.seepage);
        if (obs.vegetation_disturbance) obsItems.push('Vegetation: ' + obs.vegetation_disturbance);
        if (obs.rock_fall_activity) obsItems.push('Rock fall: ' + obs.rock_fall_activity);
        if (obs.ground_cracking) obsItems.push('Ground cracking: ' + obs.ground_cracking);
        if (obsItems.length > 0) {
            html += '<ul class="obs-list">';
            for (var j = 0; j < obsItems.length; j++) {
                html += '<li>' + escapeHtml(obsItems[j]) + '</li>';
            }
            html += '</ul>';
        }
        if (rpt.notes) {
            html += '<div style="font-size:12px;color:#636e72;margin-top:6px;font-style:italic;">' + escapeHtml(rpt.notes) + '</div>';
        }
        html += '</div>';
    }
    el.innerHTML = html;
}

function renderNewsPanel(data, el) {
    if (!data.articles || data.articles.length === 0) {
        el.innerHTML = '<div class="panel-empty">No related articles found<br><span style="font-size:12px;color:#636e72;">Sources: OpenAlex, Crossref (academic and media)</span></div>';
        return;
    }
    var html = '<div style="font-size:11px;color:#636e72;margin-bottom:8px;">Related articles from academic and media sources (OpenAlex, Crossref)</div>';
    html += '<table class="panel-table"><tr><th>Title</th><th>Source</th><th>Date</th></tr>';
    for (var i = 0; i < data.articles.length; i++) {
        var a = data.articles[i];
        var titleHtml = escapeHtml(a.title);
        if (a.url) titleHtml = safeLink(a.url, a.title);
        html += '<tr><td style="max-width:400px;">' + titleHtml + '</td>';
        html += '<td style="color:#636e72;font-size:12px;white-space:nowrap;">' + escapeHtml(a.source || a.provider) + '</td>';
        html += '<td style="white-space:nowrap;">' + escapeHtml(a.date || '') + '</td></tr>';
    }
    html += '</table>';
    el.innerHTML = html;
}

// ---- Response Protocol ----

function updateProtocolVisibility() {
    var el = document.getElementById('responseProtocol');
    if (!el) return;
    var f = FLAGS.find(function(x) { return x.flag_id === selectedFlagId; });
    if (!f || !f.voight_fit || !f.voight_fit.predicted_failure_date) { el.style.display = 'none'; return; }
    var failMs = new Date(f.voight_fit.predicted_failure_date + 'T00:00:00').getTime();
    var nowMs = new Date().setHours(0,0,0,0);
    var daysFromToday = Math.round((failMs - nowMs) / 86400000);
    el.style.display = (daysFromToday <= 30) ? '' : 'none';
}

function getSelectedFlag() {
    return FLAGS.find(function(x) { return x.flag_id === selectedFlagId; });
}

function formatSettlementList(features) {
    if (!features || features.length === 0) return '  (No settlement data available)\n';
    var txt = '';
    for (var i = 0; i < features.length; i++) {
        var f = features[i];
        txt += '  - ' + f.name + ' (' + f.type + '), ' + f.distance_km.toFixed(1) + ' km from hazard zone\n';
    }
    return txt;
}

function draftNotification() {
    var f = getSelectedFlag();
    if (!f) return;
    var v = f.voight_fit || {};
    var today = new Date().toISOString().slice(0, 10);
    var failMs = new Date(v.predicted_failure_date + 'T00:00:00').getTime();
    var nowMs = new Date().setHours(0,0,0,0);
    var daysFromToday = Math.round((failMs - nowMs) / 86400000);
    var urgency = daysFromToday <= 0 ? 'IMMEDIATE' : 'URGENT';

    var settlements = cachedExposure && cachedExposure.features ? cachedExposure.features : [];
    var villageNames = [];
    for (var i = 0; i < settlements.length; i++) {
        if (settlements[i].type === 'village' || settlements[i].type === 'town' || settlements[i].type === 'city') {
            villageNames.push(settlements[i].name);
        }
    }

    var txt = '';
    txt += urgency + ' — GLEWS Hazard Notification\n';
    txt += '========================================\n\n';
    txt += 'Date Issued: ' + today + '\n';
    txt += 'Issuing System: GLEWS (Glacier and Landslide Early Warning System)\n';
    txt += 'Hazard Type: Potential slope failure / mass movement\n\n';
    txt += 'LOCATION\n';
    txt += '  Coordinates: ' + f.center_lat.toFixed(4) + '°N, ' + f.center_lon.toFixed(4) + '°E\n';
    txt += '  Region: Nyalam County, Shigatse Prefecture, Xizang\n';
    txt += '  Flag ID: ' + f.flag_id + ' | Severity: ' + f.severity + '\n\n';
    txt += 'HAZARD ASSESSMENT\n';
    txt += '  Anomaly Score: ' + (f.score != null ? f.score.toFixed(2) : 'N/A') + '\n';
    txt += '  Voight Fit R²: ' + (v.r_squared != null ? v.r_squared.toFixed(3) : 'N/A') + '\n';
    txt += '  Predicted Failure Date: ' + (v.predicted_failure_date || 'N/A') + '\n';
    txt += '  95% Confidence Window: ' + (v.failure_window || 'N/A') + '\n';
    if (daysFromToday <= 0) {
        txt += '  STATUS: Projected failure date has PASSED (' + Math.abs(daysFromToday) + ' days ago)\n';
    } else {
        txt += '  STATUS: Failure projected in ' + daysFromToday + ' days\n';
    }
    txt += '\nCOMMUNITIES WITHIN POTENTIAL IMPACT ZONE\n';
    txt += formatSettlementList(settlements);
    if (villageNames.length > 0) {
        txt += '  Priority communities: ' + villageNames.slice(0, 5).join(', ') + '\n';
    }
    txt += '\nRECOMMENDED ACTIONS\n';
    txt += '  1. Alert local disaster management authorities in Nyalam County\n';
    txt += '  2. Commission immediate field reconnaissance of the hazard site\n';
    txt += '  3. Establish communication with community leaders in affected villages\n';
    txt += '  4. Pre-position emergency response resources if not already in place\n';
    txt += '  5. Increase monitoring cadence (request additional SAR acquisitions)\n';
    txt += '\nThis notification is generated from satellite-based InSAR analysis.\n';
    txt += 'Field verification is required before public alert issuance.\n';

    showDraft(txt);
}

function draftEvacuation() {
    var f = getSelectedFlag();
    if (!f) return;
    var v = f.voight_fit || {};
    var today = new Date().toISOString().slice(0, 10);
    var failMs = new Date(v.predicted_failure_date + 'T00:00:00').getTime();
    var nowMs = new Date().setHours(0,0,0,0);
    var daysFromToday = Math.round((failMs - nowMs) / 86400000);

    var settlements = cachedExposure && cachedExposure.features ? cachedExposure.features : [];
    var zones = { immediate: [], warning: [], advisory: [] };
    for (var i = 0; i < settlements.length; i++) {
        var s = settlements[i];
        if (s.type !== 'village' && s.type !== 'town' && s.type !== 'city' && s.type !== 'hamlet') continue;
        if (s.distance_km <= 10) zones.immediate.push(s);
        else if (s.distance_km <= 15) zones.warning.push(s);
        else zones.advisory.push(s);
    }

    var txt = '';
    txt += 'EVACUATION ADVISORY — DRAFT\n';
    txt += '========================================\n\n';
    txt += 'Date Prepared: ' + today + '\n';
    txt += 'Prepared By: GLEWS Tier 2 Analyst (REQUIRES REVIEW)\n';
    txt += 'Status: DRAFT — NOT FOR PUBLIC RELEASE\n\n';
    txt += 'HAZARD SUMMARY\n';
    txt += '  A potential slope failure has been identified at\n';
    txt += '  ' + f.center_lat.toFixed(4) + '°N, ' + f.center_lon.toFixed(4) + '°E\n';
    txt += '  via satellite InSAR displacement analysis.\n\n';
    if (daysFromToday <= 0) {
        txt += '  The projected failure date (' + v.predicted_failure_date + ') has PASSED.\n';
        txt += '  The slope may be in an advanced failure state or the model\n';
        txt += '  parameters may have shifted. Immediate field verification is critical.\n\n';
    } else {
        txt += '  Projected failure date: ' + v.predicted_failure_date + ' (' + daysFromToday + ' days from today)\n';
        txt += '  Confidence window: ' + (v.failure_window || 'N/A') + '\n\n';
    }
    txt += 'AFFECTED ZONES\n\n';
    if (zones.immediate.length > 0) {
        txt += '  ZONE 1 — IMMEDIATE (< 10 km from hazard)\n';
        txt += '  Action: Prepare for evacuation; await field verification\n';
        for (var i = 0; i < zones.immediate.length; i++) {
            txt += '    - ' + zones.immediate[i].name + ' (' + zones.immediate[i].distance_km.toFixed(1) + ' km)\n';
        }
        txt += '\n';
    }
    if (zones.warning.length > 0) {
        txt += '  ZONE 2 — WARNING (10–15 km from hazard)\n';
        txt += '  Action: Alert community leaders; identify evacuation routes\n';
        for (var i = 0; i < zones.warning.length; i++) {
            txt += '    - ' + zones.warning[i].name + ' (' + zones.warning[i].distance_km.toFixed(1) + ' km)\n';
        }
        txt += '\n';
    }
    if (zones.advisory.length > 0) {
        txt += '  ZONE 3 — ADVISORY (15–20 km from hazard)\n';
        txt += '  Action: Monitor situation; no immediate action required\n';
        for (var i = 0; i < zones.advisory.length; i++) {
            txt += '    - ' + zones.advisory[i].name + ' (' + zones.advisory[i].distance_km.toFixed(1) + ' km)\n';
        }
        txt += '\n';
    }
    txt += 'EVACUATION GUIDANCE\n';
    txt += '  - Move AWAY from valley floors and drainage channels\n';
    txt += '  - Move to higher ground perpendicular to potential flow path\n';
    txt += '  - Avoid downstream river valleys — debris flows follow drainages\n';
    txt += '  - Identified road: G318 (China National Highway 318) for evacuation routing\n\n';
    txt += 'NEXT STEPS\n';
    txt += '  1. Field team to verify ground conditions at hazard site\n';
    txt += '  2. Local authority review and approval of this advisory\n';
    txt += '  3. Translation into local languages before community distribution\n';
    txt += '  4. Establish community communication channels\n\n';
    txt += 'THIS IS A DRAFT PREPARED FROM REMOTE SENSING DATA.\n';
    txt += 'IT MUST BE VERIFIED BY FIELD ASSESSMENT AND APPROVED BY\n';
    txt += 'LOCAL DISASTER MANAGEMENT AUTHORITIES BEFORE DISTRIBUTION.\n';

    showDraft(txt);
}

function showDraft(text) {
    var el = document.getElementById('draftOutput');
    if (!el) return;
    var html = '<div class="draft-output">';
    html += '<button class="copy-btn" onclick="copyDraft()">Copy</button>';
    html += '<pre style="margin:0;white-space:pre-wrap;font-family:inherit;font-size:inherit;" id="draftText">' + escapeHtml(text) + '</pre>';
    html += '</div>';
    el.innerHTML = html;
}

function copyDraft() {
    var el = document.getElementById('draftText');
    if (!el) return;
    navigator.clipboard.writeText(el.textContent).then(function() {
        var btn = document.querySelector('.copy-btn');
        if (btn) { btn.textContent = 'Copied'; setTimeout(function() { btn.textContent = 'Copy'; }, 2000); }
    });
}

// ---- Timeseries SVG ----
function renderTimeseriesSVG(dates, values, flagId) {
    var w = 700, h = 200, pad = 50;
    var n = dates.length;
    if (n < 2) return '<div class="no-data">Insufficient data points</div>';
    var vmin = Math.min.apply(null, values);
    var vmax = Math.max.apply(null, values);
    if (vmin === vmax) { vmin -= 1; vmax += 1; }
    var margin = (vmax - vmin) * 0.1;
    vmin -= margin; vmax += margin;

    var svg = '<svg viewBox="0 0 ' + (w + 2 * pad) + ' ' + (h + 2 * pad) + '" style="max-width:100%;height:auto;">';
    svg += '<line x1="' + pad + '" y1="' + (h + pad) + '" x2="' + (w + pad) + '" y2="' + (h + pad) + '" stroke="#636e72" stroke-width="1"/>';
    svg += '<line x1="' + pad + '" y1="' + pad + '" x2="' + pad + '" y2="' + (h + pad) + '" stroke="#636e72" stroke-width="1"/>';
    for (var y = 0; y <= 4; y++) {
        var yv = vmin + (vmax - vmin) * y / 4;
        var yp = h + pad - (y / 4) * h;
        var range = vmax - vmin;
        var dp = range < 0.1 ? 4 : range < 1 ? 3 : 2;
        svg += '<text x="' + (pad - 5) + '" y="' + yp + '" text-anchor="end" font-size="10" fill="#636e72">' + yv.toFixed(dp) + '</text>';
        svg += '<line x1="' + pad + '" y1="' + yp + '" x2="' + (w + pad) + '" y2="' + yp + '" stroke="#dfe6e9" stroke-width="0.5"/>';
    }
    var points = '';
    for (var i = 0; i < n; i++) {
        var x = pad + (i / (n - 1)) * w;
        var yVal = h + pad - ((values[i] - vmin) / (vmax - vmin)) * h;
        points += x + ',' + yVal + ' ';
    }
    svg += '<polyline points="' + points.trim() + '" fill="none" stroke="#0984e3" stroke-width="2"/>';
    for (var i = 0; i < n; i++) {
        var x = pad + (i / (n - 1)) * w;
        var yVal = h + pad - ((values[i] - vmin) / (vmax - vmin)) * h;
        svg += '<circle cx="' + x + '" cy="' + yVal + '" r="3" fill="#0984e3"/>';
    }
    var xLabels = [0, Math.floor(n / 2), n - 1];
    for (var k = 0; k < xLabels.length; k++) {
        var idx = xLabels[k];
        var x = pad + (idx / (n - 1)) * w;
        svg += '<text x="' + x + '" y="' + (h + pad + 16) + '" text-anchor="middle" font-size="10" fill="#636e72">' + dates[idx] + '</text>';
    }
    svg += '<text x="' + (w / 2 + pad) + '" y="' + (h + pad + 35) + '" text-anchor="middle" font-size="12" fill="#2d3436">Date</text>';
    svg += '<text transform="rotate(-90)" x="' + (-(h / 2 + pad)) + '" y="14" text-anchor="middle" font-size="12" fill="#2d3436">Displacement (m)</text>';
    svg += '</svg>';
    return svg;
}

// ---- Classify + notes ----
function classify(flagId, classification) {
    var fid = String(flagId);
    var note = '';
    var noteEl = document.getElementById('noteInput');
    if (noteEl) note = noteEl.value;

    var payload = JSON.stringify({flag_id: flagId, classification: classification, note: note});
    var xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/classify', true);
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.onload = function() {
        if (xhr.status === 200) {
            var resp = JSON.parse(xhr.responseText);
            STATE[fid] = resp;
            renderDetail(flagId);
            renderFlagList();
            showSaved();
        }
    };
    xhr.send(payload);
}

function saveNote(flagId) {
    var st = STATE[String(flagId)] || {};
    if (st.classification) {
        classify(flagId, st.classification);
    }
}

function showSaved() {
    var el = document.getElementById('saveIndicator');
    if (el) {
        el.classList.add('visible');
        setTimeout(function() { el.classList.remove('visible'); }, 2000);
    }
}

// Initial render
renderStats();
renderFlagList();
</script>
</body>
</html>
"""


def _render_html(flags: list[dict], state: dict, site_name: str) -> str:
    """Render the dashboard HTML with flag data injected."""
    flags_json = json.dumps(flags, default=str)
    state_json = json.dumps(state)
    html = _HTML_TEMPLATE
    html = html.replace("$$FLAGS_JSON$$", flags_json)
    html = html.replace("$$STATE_JSON$$", state_json)
    html = html.replace("$$SITE_NAME$$", site_name)
    return html


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

def make_handler(
    flags: list[dict], state: dict, state_path: Path, site_name: str
):
    """
    Factory that returns a BaseHTTPRequestHandler subclass bound to
    the given flags and state.  Avoids module-level globals.
    """
    state_lock = threading.Lock()
    data_dir = state_path.parent

    class DashboardHandler(BaseHTTPRequestHandler):

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            params = parse_qs(parsed.query)

            if path in ("/", "/index.html"):
                html = _render_html(flags, state, site_name)
                self._respond(200, "text/html; charset=utf-8", html.encode("utf-8"))

            elif path == "/api/flags":
                merged = _merge_flags_state(flags, state)
                body = json.dumps(merged)
                self._respond(200, "application/json", body.encode())

            elif path == "/api/elevation":
                self._handle_geo_api(params, lambda lat, lon: _cached_call(
                    f"elev:{lat:.4f},{lon:.4f}", _get_elevation, lat, lon
                ))

            elif path == "/api/exposure":
                self._handle_geo_api(params, lambda lat, lon: _cached_call(
                    f"expo:{lat:.4f},{lon:.4f}", _get_exposure, lat, lon,
                    ttl=600,
                ))

            elif path == "/api/sentinel2":
                self._handle_geo_api(params, lambda lat, lon: _cached_call(
                    f"s2:{lat:.2f},{lon:.2f}", _get_sentinel2, lat, lon,
                    ttl=600,
                ))

            elif path == "/api/weather":
                self._handle_geo_api(params, lambda lat, lon: _cached_call(
                    f"wx:{lat:.2f},{lon:.2f}", _get_weather, lat, lon,
                    ttl=300,
                ))

            elif path == "/api/history":
                self._handle_geo_api(params, lambda lat, lon: _cached_call(
                    f"hist:{lat:.4f},{lon:.4f}", _get_history, lat, lon, data_dir,
                    ttl=60,
                ))

            elif path == "/api/monitoring":
                self._handle_geo_api(params, lambda lat, lon: _cached_call(
                    f"mon:{lat:.2f},{lon:.2f}", _get_monitoring_cadence, lat, lon,
                    ttl=600,
                ))

            elif path == "/api/instruments":
                self._handle_geo_api(params, lambda lat, lon: _cached_call(
                    f"inst:{lat:.4f},{lon:.4f}", _get_instruments, lat, lon, data_dir,
                    ttl=60,
                ))

            elif path == "/api/fieldreports":
                self._handle_geo_api(params, lambda lat, lon: _cached_call(
                    f"fr:{lat:.4f},{lon:.4f}", _get_field_reports, lat, lon, data_dir,
                    ttl=60,
                ))

            elif path == "/api/news":
                region = params.get("region", [""])[0]
                self._handle_geo_api(params, lambda lat, lon: _cached_call(
                    f"news:{region or 'default'}",
                    _get_news, lat, lon, region,
                    ttl=3600,
                ))

            else:
                self._respond(404, "text/plain", b"Not Found")

        def _handle_geo_api(self, params, handler):
            try:
                lat = float(params["lat"][0])
                lon = float(params["lon"][0])
            except (KeyError, IndexError, ValueError):
                self._respond(
                    400, "application/json",
                    json.dumps({"error": "lat and lon required"}).encode(),
                )
                return
            try:
                result = handler(lat, lon)
                body = json.dumps(result)
                self._respond(200, "application/json", body.encode())
            except Exception as exc:
                logger.warning("API error for %s: %s", self.path, exc)
                body = json.dumps({"error": str(exc)})
                self._respond(502, "application/json", body.encode())

        def do_POST(self):
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/api/classify":
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length)
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    self._respond(
                        400,
                        "application/json",
                        json.dumps({"error": "Invalid JSON"}).encode(),
                    )
                    return

                flag_id = data.get("flag_id")
                classification = data.get("classification")
                note = data.get("note", "")

                if classification not in VALID_CLASSIFICATIONS:
                    self._respond(
                        400,
                        "application/json",
                        json.dumps(
                            {
                                "error": (
                                    "Invalid classification. Must be one of: "
                                    + ", ".join(sorted(VALID_CLASSIFICATIONS))
                                )
                            }
                        ).encode(),
                    )
                    return

                fid = str(flag_id)
                entry = {
                    "classification": classification,
                    "note": note,
                    "updated_at": datetime.now(timezone.utc).strftime(
                        "%Y-%m-%d %H:%M UTC"
                    ),
                }
                with state_lock:
                    state[fid] = entry
                    save_state(state_path, state)

                self._respond(
                    200, "application/json", json.dumps(entry).encode()
                )

            elif path == "/api/monitoring/enhance":
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length)
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    self._respond(400, "application/json",
                                  json.dumps({"error": "Invalid JSON"}).encode())
                    return
                fid = str(data.get("flag_id"))
                enhanced = bool(data.get("enhanced", False))
                with state_lock:
                    if fid not in state:
                        state[fid] = {}
                    state[fid]["enhanced_monitoring"] = enhanced
                    state[fid]["enhanced_monitoring_at"] = datetime.now(
                        timezone.utc
                    ).strftime("%Y-%m-%d %H:%M UTC")
                    save_state(state_path, state)
                self._respond(200, "application/json",
                              json.dumps({"enhanced": enhanced}).encode())
            else:
                self._respond(404, "text/plain", b"Not Found")

        def _respond(self, code: int, content_type: str, body: bytes):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            logger.info(fmt, *args)

    return DashboardHandler


def _merge_flags_state(flags: list[dict], state: dict) -> list[dict]:
    """Return flags with analyst state merged in."""
    result = []
    for f in flags:
        merged = dict(f)
        fid = str(f.get("flag_id"))
        if fid in state:
            merged["analyst"] = state[fid]
        result.append(merged)
    return result


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------

def serve(data_dir: str, port: int, config: dict) -> None:
    """Start the dashboard HTTP server (blocking)."""
    data_path = Path(data_dir)
    state_path = data_path / "dashboard_state.json"

    site_name = config.get("site", {}).get("name", "GLEWS Site")

    flags = load_flags(data_path)
    state = load_state(state_path)

    _resolve_flag_place_names(flags, data_path)

    handler_cls = make_handler(flags, state, state_path, site_name)

    try:
        from http.server import ThreadingHTTPServer
        server = ThreadingHTTPServer(("127.0.0.1", port), handler_cls)
    except ImportError:
        server = HTTPServer(("127.0.0.1", port), handler_cls)

    logger.info(
        "Dashboard serving %d flags at http://127.0.0.1:%d/", len(flags), port
    )
    print(f"GLEWS Tier 2 Dashboard - {site_name}")
    print(f"Serving {len(flags)} flags at http://127.0.0.1:{port}/")
    print("Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()
