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
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from html import escape as html_escape
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

# Flags graded "unreliable" by glews.quality (low coherence or unreliable
# unwrapping under most of their pixels) cannot display above this severity.
UNRELIABLE_SEVERITY_CAP = "WARNING"

_USER_AGENT = "GLEWS/0.1 (glacier-landslide-warning; https://github.com/glews)"
_API_TIMEOUT = 10


def _severity_from_score(score: float) -> str:
    """Derive a display severity from the composite anomaly score."""
    if score >= SEVERITY_CRITICAL_THRESHOLD:
        return "CRITICAL"
    if score >= SEVERITY_WARNING_THRESHOLD:
        return "WARNING"
    return "INFO"


def _cap_unreliable_severity(flags: list[dict]) -> None:
    """Stop flags built on unreliable data from displaying as CRITICAL."""
    for f in flags:
        quality = f.get("quality") or {}
        if quality.get("reliability") == "unreliable" and f.get("severity") == "CRITICAL":
            f["severity_uncapped"] = "CRITICAL"
            f["severity"] = UNRELIABLE_SEVERITY_CAP


def _reclassify_post_events(flags: list[dict], event_date: str | None) -> None:
    """Downgrade CRITICAL flags whose predicted failure is in the past to POST-EVENT."""
    from datetime import date

    if event_date:
        try:
            ref = date.fromisoformat(event_date)
        except ValueError:
            ref = date.today()
    else:
        ref = date.today()

    for f in flags:
        if f.get("severity") != "CRITICAL":
            continue
        pfd = f.get("predicted_failure_date") or (
            f.get("voight_fit", {}) or {}
        ).get("predicted_failure_date")
        if not pfd:
            continue
        try:
            if date.fromisoformat(pfd) < ref:
                f["severity"] = "POST-EVENT"
        except ValueError:
            pass


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

    _cap_unreliable_severity(flags)

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


# ---------------------------------------------------------------------------
# English place names
# ---------------------------------------------------------------------------

_EN_NAME_TAGS = ("name:en", "int_name", "name:zh-Latn-pinyin", "name:ne-Latn", "name:bo-Latn")


def _romanize(name: str) -> str:
    """Romanize a Chinese (pinyin) or Devanagari (IAST) name; "" if not possible.

    Used only when OpenStreetMap has no English or Latin-script name, and
    always labelled as a romanization rather than an official English name.
    """
    if not name:
        return ""
    if any("\u4e00" <= ch <= "\u9fff" for ch in name):
        try:
            from pypinyin import lazy_pinyin
        except ImportError:
            return ""
        return "".join(lazy_pinyin(name)).capitalize()
    if any("\u0900" <= ch <= "\u097f" for ch in name):
        try:
            from indic_transliteration import sanscript
        except ImportError:
            return ""
        words = []
        for word in name.split():
            latin = sanscript.transliterate(word, sanscript.DEVANAGARI, sanscript.IAST)
            # Nepali drops the inherent final vowel that IAST writes out
            if len(latin) > 3 and latin.endswith("a") and word[-1] not in "\u093e\u0906\u0905":
                latin = latin[:-1]
            words.append(latin.capitalize())
        return " ".join(words)
    return ""


def _english_name(tags: dict, name: str | None = None) -> tuple[str, str]:
    """Return (english_name, source) for an OSM element.

    source is "osm" for a name tagged in OpenStreetMap, "romanized" for a
    pinyin/IAST fallback, or "" when nothing is available.
    """
    for key in _EN_NAME_TAGS:
        if tags.get(key):
            return tags[key], "osm"
    roman = _romanize(name if name is not None else tags.get("name", ""))
    return (roman, "romanized") if roman else ("", "")


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
             name_en: str = "", name_en_source: str = ""):
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
            entry["name_en_source"] = name_en_source
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
            en_name, en_src = _english_name(nd, local_name)
            _add(local_name, ptype,
                 float(p.get("lat", 0)), float(p.get("lon", 0)),
                 name_en=en_name, name_en_source=en_src)
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
                en_name, en_src = _english_name(nd, local_name)
                _add(local_name, rtype,
                     float(data.get("lat", 0)), float(data.get("lon", 0)),
                     name_en=en_name, name_en_source=en_src)
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
            # Unnamed OSM nodes used to fall back to their type ("hamlet"),
            # which then showed in the Name column. Keep the name empty
            # instead and let the panel label it as unnamed.
            name = (tags.get("name") or tags.get("official_name")
                    or tags.get("alt_name") or "")
            name_en, name_en_source = _english_name(tags, name)
            if not name and name_en:
                name, name_en, name_en_source = name_en, "", ""
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
                entry["name_en_source"] = name_en_source
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


WINDY_KEY_ENV = "GLEWS_WINDY_API_KEY"
_WINDY_WEBCAMS_URL = "https://api.windy.com/webcams/api/v3/webcams"
# Free-tier image URLs carry a token that expires after 10 minutes, so
# results are cached for less than that and never written to disk.
WEBCAM_CACHE_TTL_S = 300
WEBCAM_RADII_KM = (50, 150)


YOUTUBE_KEY_ENV = "GLEWS_YOUTUBE_API_KEY"
DEFAULT_WEBCAM_REGISTRY = Path(__file__).resolve().parents[2] / "config" / "webcams.yaml"


def load_webcam_registry(path: str | Path | None = None) -> list[dict]:
    """Load the curated webcam list (config/webcams.yaml); [] if absent or invalid."""
    import yaml

    path = Path(path) if path else DEFAULT_WEBCAM_REGISTRY
    if not path.exists():
        return []
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("Could not read webcam registry %s: %s", path, exc)
        return []
    providers = raw.get("providers", {}) or {}
    cams = []
    for w in raw.get("webcams", []) or []:
        if w.get("lat") is None or w.get("lon") is None or not w.get("youtube_video_id"):
            logger.warning("Skipping webcam entry without lat/lon/youtube_video_id: %s", w.get("id"))
            continue
        prov = providers.get(w.get("provider"), {})
        cams.append({**w, "provider_name": prov.get("name", w.get("provider", "")),
                     "provider_url": prov.get("url", ""),
                     "youtube_channel_id": prov.get("youtube_channel_id", "")})
    return cams


# Channel scans cost ~1 quota unit per 50 uploads plus 1 per 50 videos
# checked (~8 units for a 150-video channel), so hourly is ~200 units/day
# of the 10,000 free. Failures are retried after 15 minutes.
LIVE_SCAN_TTL_S = 3600
LIVE_SCAN_RETRY_S = 900
_live_scan_cache: dict[str, tuple[list[dict], float]] = {}
_live_scan_lock = threading.Lock()


def _youtube_get(path: str, key: str, **params) -> dict:
    query = urllib.parse.urlencode(params)
    return _fetch_json(f"https://www.googleapis.com/youtube/v3/{path}?{query}",
                       headers={"X-Goog-Api-Key": key}, timeout=15)


def _channel_live_streams(channel_id: str, key: str) -> list[dict]:
    """Currently live videos on a channel, as [{"id", "title"}], cached hourly.

    Scans the channel's uploads playlist rather than using search
    (eventType=live), which costs 100 units and often misses long-running
    streams.
    """
    now = time.time()
    with _live_scan_lock:
        hit = _live_scan_cache.get(channel_id)
        if hit and now < hit[1]:
            return hit[0]
    try:
        uploads = "UU" + channel_id[2:]
        ids, token = [], None
        for _ in range(10):                      # cap at 500 uploads
            params = {"part": "contentDetails", "playlistId": uploads, "maxResults": 50}
            if token:
                params["pageToken"] = token
            page = _youtube_get("playlistItems", key, **params)
            ids += [i["contentDetails"]["videoId"] for i in page.get("items", [])]
            token = page.get("nextPageToken")
            if not token:
                break
        live = []
        for k in range(0, len(ids), 50):
            vids = _youtube_get("videos", key, part="snippet", id=",".join(ids[k:k + 50]))
            live += [{"id": v["id"], "title": v["snippet"].get("title", "")}
                     for v in vids.get("items", [])
                     if v.get("snippet", {}).get("liveBroadcastContent") == "live"]
        expires = now + LIVE_SCAN_TTL_S
    except Exception as exc:
        logger.info("YouTube channel scan failed for %s: %s", channel_id, exc)
        live, expires = [], now + LIVE_SCAN_RETRY_S
    with _live_scan_lock:
        _live_scan_cache[channel_id] = (live, expires)
    return live


def _resolve_live_video_ids(registry: list[dict]) -> dict[str, str]:
    """Map camera id -> current live video id, matched by title keywords."""
    key = os.environ.get(YOUTUBE_KEY_ENV, "").strip()
    if not key:
        return {}
    resolved = {}
    for channel in {w["youtube_channel_id"] for w in registry if w.get("youtube_channel_id")}:
        live = _channel_live_streams(channel, key)
        for w in registry:
            words = [m.upper() for m in w.get("match") or []]
            if w.get("youtube_channel_id") != channel or not words:
                continue
            for v in live:
                if all(word in v["title"].upper() for word in words):
                    resolved[w.get("id")] = v["id"]
                    break
    return resolved


def _youtube_live_status(video_ids: list[str]) -> dict[str, str]:
    """Map video id -> "live" / "upcoming" / "ended" / "unavailable" via the YouTube Data API.

    Returns {} when GLEWS_YOUTUBE_API_KEY is unset or the call fails, and the
    panel then reports the status as unknown.
    """
    key = os.environ.get(YOUTUBE_KEY_ENV, "").strip()
    if not key or not video_ids:
        return {}
    query = urllib.parse.urlencode({"part": "snippet", "id": ",".join(video_ids[:50])})
    try:
        data = _fetch_json(f"https://www.googleapis.com/youtube/v3/videos?{query}",
                           headers={"X-Goog-Api-Key": key}, timeout=10)
    except Exception as exc:
        logger.info("YouTube status lookup failed: %s", exc)
        return {}
    found = {item["id"]: (item.get("snippet", {}).get("liveBroadcastContent") or "none")
             for item in data.get("items", [])}
    return {vid: {"live": "live", "upcoming": "upcoming"}.get(found[vid], "ended")
            if vid in found else "unavailable" for vid in video_ids}


def _curated_webcams(lat: float, lon: float, registry: list[dict], radius_km: float,
                     live_ids: dict[str, str] | None = None) -> list[dict]:
    live_ids = live_ids or {}
    out = []
    for w in registry:
        d = _haversine(lat, lon, w["lat"], w["lon"])
        if d > radius_km:
            continue
        vid = live_ids.get(w.get("id"), w["youtube_video_id"])
        out.append({
            "id": w.get("id", vid), "title": w.get("name", "Webcam"),
            "city": "", "lat": w["lat"], "lon": w["lon"],
            "distance_km": round(d, 1),
            # hqdefault_live.jpg is YouTube's thumbnail for a live stream
            "image": f"https://i.ytimg.com/vi/{vid}/hqdefault_live.jpg",
            "image_daylight": f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg",
            "last_updated": "", "url": f"https://www.youtube.com/watch?v={vid}",
            "faces": w.get("faces", ""), "elevation_m": w.get("elevation_m"),
            "provider": w.get("provider_name", ""), "provider_url": w.get("provider_url", ""),
            "approximate_location": w.get("coord_precision", "approximate") != "exact",
            "video_id": vid,
            "status": "live" if w.get("id") in live_ids else None,
            "stream_id_updated": vid != w["youtube_video_id"],
        })
    return out


def _windy_webcams(lat: float, lon: float, radius: float, key: str) -> list[dict]:
    query = urllib.parse.urlencode({
        "nearby": f"{lat:.4f},{lon:.4f},{radius}",
        "include": "images,location,urls",
        "limit": 20,
    })
    data = _fetch_json(f"{_WINDY_WEBCAMS_URL}?{query}",
                       headers={"x-windy-api-key": key}, timeout=10)
    cams = []
    for w in data.get("webcams", []) or []:
        loc = w.get("location") or {}
        wlat, wlon = loc.get("latitude"), loc.get("longitude")
        if wlat is None or wlon is None:
            continue
        images = w.get("images") or {}
        current = images.get("current") or {}
        daylight = images.get("daylight") or {}
        urls = w.get("urls") or {}
        cams.append({
            "id": w.get("webcamId"),
            "title": w.get("title") or "Untitled webcam",
            "status": w.get("status"),
            "city": loc.get("city") or loc.get("region") or "",
            "lat": wlat, "lon": wlon,
            "distance_km": round(_haversine(lat, lon, wlat, wlon), 1),
            "image": current.get("preview") or current.get("thumbnail") or "",
            "image_daylight": daylight.get("preview") or daylight.get("thumbnail") or "",
            "last_updated": w.get("lastUpdatedOn") or "",
            "url": urls.get("detail") or (f"https://www.windy.com/webcams/{w['webcamId']}"
                                          if w.get("webcamId") else ""),
            "provider": "Windy", "provider_url": "https://www.windy.com/",
        })
    return cams


def _get_webcams(lat: float, lon: float, registry: list[dict] | tuple = ()) -> dict:
    """Nearby webcams: the curated registry plus Windy's listings (key from env).

    Searches 50 km, then 150 km. Windy is queried server-side so its key
    never reaches the browser.
    """
    key = os.environ.get(WINDY_KEY_ENV, "").strip()
    if not key and not registry:
        return {"configured": False, "webcams": []}
    windy_error = None
    live_ids = _resolve_live_video_ids(list(registry)) if registry else {}
    for radius in WEBCAM_RADII_KM:
        cams = _curated_webcams(lat, lon, list(registry), radius, live_ids)
        if key and windy_error is None:
            try:
                cams += _windy_webcams(lat, lon, radius, key)
            except urllib.error.HTTPError as exc:
                if exc.code not in (401, 403):
                    raise
                windy_error = (f"Windy rejected the API key (HTTP {exc.code}). "
                               f"Check {WINDY_KEY_ENV} and that the key is for the Webcams API.")
        if cams:
            pending = [c["video_id"] for c in cams if c.get("video_id") and not c.get("status")]
            status = _youtube_live_status(pending)
            for c in cams:
                if c.get("video_id") and not c.get("status"):
                    c["status"] = status.get(c["video_id"], "unknown")
            cams.sort(key=lambda c: c["distance_km"])
            if radius > WEBCAM_RADII_KM[0]:
                # Distant cameras only give regional context, so show one per site
                seen, unique = set(), []
                for c in cams:
                    spot = (round(c["lat"], 2), round(c["lon"], 2))
                    if spot not in seen:
                        seen.add(spot)
                        unique.append(c)
                cams = unique
            out = {"configured": True, "radius_km": radius, "webcams": cams[:8]}
            if windy_error:
                out["error"] = windy_error
            return out
    out = {"configured": True, "radius_km": WEBCAM_RADII_KM[-1], "webcams": []}
    if windy_error:
        out["error"] = windy_error
    return out


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


_country_cache: dict[str, str] = {}


def _reverse_country(lat: float, lon: float) -> str:
    """Reverse-geocode lat/lon to country name via Nominatim, cached."""
    key = f"{lat:.1f},{lon:.1f}"
    if key in _country_cache:
        return _country_cache[key]
    try:
        url = (
            f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}"
            "&format=json&zoom=3&accept-language=en"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "GLEWS/0.1"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
        country = data.get("address", {}).get("country", "")
        _country_cache[key] = country
        return country
    except Exception:
        _country_cache[key] = ""
        return ""


def _get_news(lat: float, lon: float, region: str = "") -> dict:
    """Fetch related news from Google News RSS (media) plus OpenAlex and Crossref (academic)."""
    import xml.etree.ElementTree as ET

    country = _reverse_country(lat, lon)

    media_articles: list[dict] = []
    academic_articles: list[dict] = []

    news_geo = country or region or "Himalaya"

    def _parse_rss_date(pub: str) -> str:
        """Convert RSS pubDate like 'Sat, 03 Oct 2026 07:00:00 GMT' to YYYY-MM-DD."""
        if not pub:
            return ""
        try:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(pub)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return pub[:16]

    _SOURCE_NAMES: dict[str, str] = {
        "markets.businessinsider.com": "Business Insider",
        "scmp.com": "South China Morning Post",
        "eos.org": "Eos",
        "phys.org": "Phys.org",
        "apnews.com": "AP News",
        "bbc.com": "BBC News",
        "bbc.co.uk": "BBC News",
        "theguardian.com": "The Guardian",
        "reuters.com": "Reuters",
        "france24.com": "France 24",
        "dw.com": "Deutsche Welle",
        "aljazeera.com": "Al Jazeera",
        "cnn.com": "CNN",
        "washingtonpost.com": "Washington Post",
        "nytimes.com": "New York Times",
    }

    def _normalize_source(source: str) -> str:
        """Remove marketing slogans and map URLs to organization names."""
        if not source:
            return source
        for domain, name in _SOURCE_NAMES.items():
            if domain in source.lower():
                return name
        sep = source.find(" - ")
        if sep > 0:
            return source[:sep].strip()
        sep = source.find(" | ")
        if sep > 0:
            return source[:sep].strip()
        sep = source.find(" – ")
        if sep > 0:
            candidate = source[:sep].strip()
            if len(candidate) >= 3:
                return candidate
        return source

    def _strip_source_suffix(title: str, source: str) -> str:
        """Remove trailing ' - Source Name' appended by Google News."""
        if source and title.endswith(" - " + source):
            return title[: -(len(source) + 3)]
        return title

    def _fetch_rss(query: str, hl: str = "en", gl: str = "US",
                   ceid: str = "US:en", limit: int = 8) -> list[dict]:
        items: list[dict] = []
        try:
            q = urllib.parse.quote(query)
            url = (
                f"https://news.google.com/rss/search?q={q}"
                f"&hl={hl}&gl={gl}&ceid={ceid}"
            )
            req = urllib.request.Request(url, headers={"User-Agent": "GLEWS/0.1"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                xml_data = resp.read()
            root = ET.fromstring(xml_data)
            for item in root.findall(".//item")[:limit]:
                title_el = item.find("title")
                title = title_el.text.strip() if title_el is not None and title_el.text else ""
                if not title:
                    continue
                source_el = item.find("source")
                source = source_el.text.strip() if source_el is not None and source_el.text else ""
                title = _strip_source_suffix(title, source)
                source = _normalize_source(source)
                link_el = item.find("link")
                link = link_el.text.strip() if link_el is not None and link_el.text else ""
                pub_el = item.find("pubDate")
                pub = pub_el.text.strip() if pub_el is not None and pub_el.text else ""
                items.append({
                    "title": title,
                    "url": link,
                    "date": _parse_rss_date(pub),
                    "source": source,
                    "type": "media",
                })
        except Exception as exc:
            logger.info("Google News RSS query failed (%s): %s", gl, exc)
        return items

    rss_query = f"landslide OR glacier OR rockfall {news_geo}"
    media_articles = _fetch_rss(rss_query, hl="en", gl="US", ceid="US:en", limit=5)
    media_articles += _fetch_rss(rss_query, hl="en", gl="GB", ceid="GB:en", limit=5)

    local_query = f"landslide OR flood OR disaster {region}" if region else ""
    if local_query:
        media_articles += _fetch_rss(local_query, hl="en", gl="US", ceid="US:en", limit=3)

    academic_articles: list[dict] = []
    academic_geo = " ".join(filter(None, [region, country]))
    if not academic_geo:
        academic_geo = "Himalaya"
    academic_query = f"landslide glacier hazard {academic_geo}"

    try:
        q = urllib.parse.quote(academic_query)
        url = (
            f"https://api.openalex.org/works?search={q}"
            "&per_page=3&sort=publication_date:desc"
            "&select=title,doi,publication_date,primary_location"
        )
        data = _fetch_json(url, timeout=12)
        for r in data.get("results", []):
            title = r.get("title", "")
            if not title:
                continue
            loc = r.get("primary_location") or {}
            src = loc.get("source") or {}
            academic_articles.append({
                "title": title,
                "url": r.get("doi", ""),
                "date": (r.get("publication_date") or "")[:10],
                "source": src.get("display_name", ""),
                "type": "academic",
            })
    except Exception as exc:
        logger.info("OpenAlex query failed: %s", exc)

    try:
        q = urllib.parse.quote(academic_query)
        url = (
            f"https://api.crossref.org/works?query={q}"
            "&rows=3&sort=relevance&order=desc"
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
            date_str = "-".join(str(p).zfill(2) for p in parts) if parts else ""
            academic_articles.append({
                "title": title,
                "url": it.get("URL", ""),
                "date": date_str,
                "source": journal,
                "type": "academic",
            })
    except Exception as exc:
        logger.info("Crossref query failed: %s", exc)

    seen_titles: set[str] = set()
    unique_media: list[dict] = []
    for a in media_articles:
        key = a["title"].lower()[:60]
        if key not in seen_titles:
            seen_titles.add(key)
            unique_media.append(a)
    unique_academic: list[dict] = []
    for a in academic_articles:
        key = a["title"].lower()[:60]
        if key not in seen_titles:
            seen_titles.add(key)
            unique_academic.append(a)

    interleaved: list[dict] = []
    ai, mi = 0, 0
    while len(interleaved) < 15 and (mi < len(unique_media) or ai < len(unique_academic)):
        if mi < len(unique_media):
            interleaved.append(unique_media[mi]); mi += 1
        if len(interleaved) >= 15:
            break
        if mi < len(unique_media):
            interleaved.append(unique_media[mi]); mi += 1
        if len(interleaved) >= 15:
            break
        if ai < len(unique_academic):
            interleaved.append(unique_academic[ai]); ai += 1
        if len(interleaved) >= 15:
            break
    while len(interleaved) < 15 and mi < len(unique_media):
        interleaved.append(unique_media[mi]); mi += 1
    while len(interleaved) < 15 and ai < len(unique_academic):
        interleaved.append(unique_academic[ai]); ai += 1

    return {"articles": interleaved[:15]}


def _apply_place_name(flag: dict, entry: dict) -> None:
    """Copy a cached place-name entry onto a flag, romanizing if no English name."""
    name = entry.get("name", "")
    name_en = entry.get("name_en", "")
    source = entry.get("name_en_source", "osm" if name_en else "")
    if not name_en:
        name_en = _romanize(name)
        source = "romanized" if name_en else ""
    flag["place_name"] = name
    flag["place_name_en"] = name_en
    flag["place_name_en_source"] = source


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
            _apply_place_name(f, cached[fid])
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
                name_en, name_en_source = _english_name(tags, name)
                places.append({
                    "name": name,
                    "name_en": name_en,
                    "name_en_source": name_en_source,
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
            entry = {"name": best["name"], "name_en": best.get("name_en", ""),
                     "name_en_source": best.get("name_en_source", "")}
            cached[fid] = entry
            _apply_place_name(f, entry)

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
:root {
    --bg-page: #f5f6fa;
    --bg-card: #fff;
    --bg-inset: #f5f6fa;
    --banner-bg: #fff3cd;
    --banner-fg: #7a5a00;
    --banner-border: #f0d78c;
    --text-primary: #2d3436;
    --text-secondary: #636e72;
    --text-muted: #b2bec3;
    --border: #dfe6e9;
    --border-dark: #485460;
    --sidebar-bg: #1e272e;
    --sidebar-border: #485460;
    --sidebar-hover: #2d3e50;
    --sidebar-text: #dfe6e9;
    --shadow: rgba(0,0,0,0.06);
    --accent: #0984e3;
    --chart-line: #0984e3;
    --chart-grid: #dfe6e9;
    --chart-axis: #636e72;
    --chart-label: #2d3436;
    --chart-snow: rgba(116,185,255,0.14);
    --chart-coh: #a4b0be;
    --chart-coh-bad: #e17055;
    --info-bg: #e8f1fb;
    --info-fg: #1d4f7a;
    --info-border: #bcd6ee;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg-page: #1a1a2e;
    --bg-card: #16213e;
    --bg-inset: #1a1a2e;
    --banner-bg: #3a2f0b;
    --banner-fg: #f5d77a;
    --banner-border: #6b5414;
    --text-primary: #e0e0e0;
    --text-secondary: #a0a0b0;
    --text-muted: #6c6c80;
    --border: #2a2a4a;
    --border-dark: #2a2a4a;
    --sidebar-bg: #0f0f23;
    --sidebar-border: #2a2a4a;
    --sidebar-hover: #1a1a3e;
    --sidebar-text: #d0d0e0;
    --shadow: rgba(0,0,0,0.3);
    --accent: #4dabf7;
    --chart-line: #4dabf7;
    --chart-grid: #2a2a4a;
    --chart-axis: #a0a0b0;
    --chart-label: #e0e0e0;
    --chart-snow: rgba(116,185,255,0.10);
    --chart-coh: #5a6380;
    --chart-coh-bad: #e17055;
    --info-bg: #0f2a44;
    --info-fg: #a9d1f5;
    --info-border: #1f4a73;
  }
}
:root[data-theme="dark"] {
    --bg-page: #1a1a2e;
    --bg-card: #16213e;
    --bg-inset: #1a1a2e;
    --banner-bg: #3a2f0b;
    --banner-fg: #f5d77a;
    --banner-border: #6b5414;
    --text-primary: #e0e0e0;
    --text-secondary: #a0a0b0;
    --text-muted: #6c6c80;
    --border: #2a2a4a;
    --border-dark: #2a2a4a;
    --sidebar-bg: #0f0f23;
    --sidebar-border: #2a2a4a;
    --sidebar-hover: #1a1a3e;
    --sidebar-text: #d0d0e0;
    --shadow: rgba(0,0,0,0.3);
    --accent: #4dabf7;
    --chart-line: #4dabf7;
    --chart-grid: #2a2a4a;
    --chart-axis: #a0a0b0;
    --chart-label: #e0e0e0;
    --chart-snow: rgba(116,185,255,0.10);
    --chart-coh: #5a6380;
    --chart-coh-bad: #e17055;
    --info-bg: #0f2a44;
    --info-fg: #a9d1f5;
    --info-border: #1f4a73;
}

* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; display: flex; height: 100vh; background: var(--bg-page); color: var(--text-primary); }

/* Sidebar */
.sidebar { width: 340px; min-width: 340px; background: var(--sidebar-bg); color: var(--sidebar-text); display: flex; flex-direction: column; overflow: hidden; }
.sidebar-header { padding: 20px; border-bottom: 1px solid var(--sidebar-border); }
.sidebar-header h1 { font-size: 18px; color: #fff; margin-bottom: 4px; }
.sidebar-header .site-name { font-size: 13px; color: var(--text-muted); }
.sidebar-stats { padding: 12px 20px; border-bottom: 1px solid var(--sidebar-border); display: flex; gap: 12px; }
.sidebar-stats .stat { text-align: center; flex: 1; }
.sidebar-stats .stat-value { font-size: 20px; font-weight: 700; color: #fff; font-variant-numeric: tabular-nums; }
.sidebar-stats .stat-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-secondary); }
.sidebar-stats .stat-value.critical { color: #d63031; }
.sidebar-stats .stat-value.warning { color: #fdcb6e; }
.sidebar-stats .stat-value.info { color: #74b9ff; }
.sidebar-filter { padding: 12px 20px; border-bottom: 1px solid var(--sidebar-border); display: flex; gap: 6px; }
.sidebar-filter button { padding: 4px 10px; border: 1px solid var(--text-secondary); border-radius: 4px; background: transparent; color: var(--text-muted); cursor: pointer; font-size: 12px; }
.sidebar-filter button.active { background: var(--accent); border-color: var(--accent); color: #fff; }
.flag-list { flex: 1; overflow-y: auto; }
.flag-item { padding: 14px 20px; border-bottom: 1px solid var(--sidebar-hover); cursor: pointer; transition: background 0.15s; }
.flag-item:hover { background: var(--sidebar-hover); }
.flag-item.selected { background: var(--accent); }
.flag-item .flag-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
.flag-item .flag-id { font-weight: 600; font-size: 14px; }
.flag-item .flag-score { font-size: 13px; color: var(--text-muted); }
.flag-item.selected .flag-score { color: var(--sidebar-text); }
.flag-item .flag-meta { font-size: 12px; color: var(--text-secondary); }
.flag-item.selected .flag-meta { color: var(--text-muted); }
.severity-badge { display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 11px; font-weight: 700; letter-spacing: 0.5px; }
.severity-CRITICAL { background: #d63031; color: #fff; }
.severity-POST-EVENT { background: #636e72; color: #fff; }
.severity-WARNING { background: #fdcb6e; color: #2d3436; }
.severity-INFO { background: #74b9ff; color: #2d3436; }
.classification-badge { display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 11px; font-weight: 600; margin-left: 6px; }
.classification-Watch { background: #e17055; color: #fff; }
.classification-Warning { background: #fdcb6e; color: #2d3436; }
.classification-Cleared { background: #00b894; color: #fff; }

/* Theme toggle */
.theme-toggle { position: fixed; bottom: 16px; right: 16px; z-index: 1000; width: 36px; height: 36px; border-radius: 50%; border: 1px solid var(--border); background: var(--bg-card); color: var(--text-primary); cursor: pointer; font-size: 18px; display: flex; align-items: center; justify-content: center; box-shadow: 0 2px 8px var(--shadow); transition: all 0.2s; }
.theme-toggle:hover { border-color: var(--accent); }

/* Main content */
.main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
.main-header { padding: 20px 30px; background: var(--bg-card); border-bottom: 1px solid var(--border); }
.main-header h2 { font-size: 20px; color: var(--text-primary); }
.data-banner { padding: 8px 30px; background: var(--banner-bg); color: var(--banner-fg); border-bottom: 1px solid var(--banner-border); font-size: 13px; line-height: 1.4; }
.data-banner strong { letter-spacing: 0.06em; margin-right: 6px; }
.data-banner-observed { background: var(--info-bg); color: var(--info-fg); border-bottom-color: var(--info-border); }
.data-banner .banner-detail { opacity: 0.85; }

/* Data reliability */
.rel-badge { display: inline-block; padding: 2px 7px; border-radius: 3px; font-size: 10.5px; font-weight: 600; letter-spacing: 0.3px; color: #fff; margin-left: 6px; white-space: nowrap; }
.rel-good { background: #0b7a5a; }
.rel-marginal { background: #9a6b12; }
.rel-unreliable { background: #57606f; }
.quality-row { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 14px; margin-top: 16px; padding: 12px 14px; background: var(--bg-inset); border-radius: 6px; font-size: 13px; color: var(--text-secondary); font-variant-numeric: tabular-nums; }
.quality-row .rel-badge { margin-left: 0; }
.quality-row strong { color: var(--text-primary); font-weight: 600; }
.cap-note { flex-basis: 100%; color: var(--text-primary); }

/* Sidebar triage tools (sidebar is dark in both themes) */
.sidebar-tools { padding: 10px 20px 12px; border-bottom: 1px solid var(--sidebar-border); display: flex; flex-direction: column; gap: 10px; }
.tool-row { display: flex; gap: 10px; }
.tool-row label { flex: 1; display: flex; flex-direction: column; gap: 3px; font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; color: #a0a0b0; }
.tool-row select { padding: 4px 6px; border: 1px solid #485460; border-radius: 4px; background: #2d3e50; color: #dfe6e9; font-size: 12px; text-transform: none; letter-spacing: 0; }
.tool-row select:focus-visible, .bulk-form select:focus-visible, .bulk-form input:focus-visible, .sidebar-tools button:focus-visible { outline: 2px solid #74b9ff; outline-offset: 1px; }
.hist-title { font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; color: #a0a0b0; display: flex; justify-content: space-between; }
.hist-title button { background: none; border: none; color: #74b9ff; font-size: 11px; cursor: pointer; padding: 0; text-transform: none; letter-spacing: 0; }
.peak-hist svg { display: block; width: 100%; height: auto; }
.peak-hist rect.bar { cursor: pointer; }
.bulk-btn { padding: 5px 10px; border: 1px solid #485460; border-radius: 4px; background: transparent; color: #dfe6e9; font-size: 12px; cursor: pointer; align-self: flex-start; }
.bulk-btn:hover { border-color: #74b9ff; }
.bulk-form { display: flex; flex-direction: column; gap: 6px; padding: 10px; background: #2d3e50; border-radius: 6px; font-size: 12px; color: #dfe6e9; }
.bulk-form select, .bulk-form input { padding: 5px 6px; border: 1px solid #485460; border-radius: 4px; background: #1e272e; color: #dfe6e9; font-size: 12px; }
.bulk-form .bulk-actions { display: flex; gap: 6px; }
.bulk-form .bulk-actions button { padding: 5px 10px; border-radius: 4px; font-size: 12px; cursor: pointer; border: 1px solid #485460; background: transparent; color: #dfe6e9; }
.bulk-form .bulk-actions button.confirm { background: #00b894; border-color: #00b894; color: #fff; font-weight: 600; }
.bulk-form .bulk-error { color: #ff7675; }

/* Webcams */
.webcam-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 12px; }
.webcam { display: block; text-decoration: none; color: inherit; background: var(--bg-inset); border-radius: 6px; overflow: hidden; }
.webcam:hover, .webcam:focus-visible { outline: 2px solid var(--accent); }
.webcam img, .webcam-noimg { width: 100%; aspect-ratio: 16 / 9; object-fit: cover; display: block; background: var(--border); }
.webcam-noimg { display: flex; align-items: center; justify-content: center; font-size: 12px; color: var(--text-secondary); }
.webcam-title { font-size: 12.5px; font-weight: 600; padding: 6px 8px 0; color: var(--text-primary); }
.webcam-meta { font-size: 11.5px; color: var(--text-secondary); padding: 2px 8px 8px; }
.webcam-provider { padding-top: 0; font-size: 11px; opacity: 0.85; }
.webcam-credit { font-size: 11px; color: var(--text-secondary); margin-top: 8px; }
.webcam-credit a { color: var(--accent); }

/* Interferogram strip */
.ifg-strip { width: 100%; height: auto; display: block; border-radius: 4px; background: #fff; }
.ifg-caption { font-size: 12px; color: var(--text-secondary); margin-top: 8px; line-height: 1.5; }
.main-body { flex: 1; overflow-y: auto; padding: 30px; }

/* Detail card */
.detail-card { background: var(--bg-card); border-radius: 8px; box-shadow: 0 2px 8px var(--shadow); padding: 24px; margin-bottom: 20px; }
.detail-card h3 { font-size: 16px; margin-bottom: 16px; color: var(--text-primary); border-bottom: 1px solid var(--border); padding-bottom: 10px; }
.metrics-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 16px; }
.metric { background: var(--bg-inset); border-radius: 6px; padding: 14px; }
.metric .label { font-size: 11px; text-transform: uppercase; color: var(--text-secondary); letter-spacing: 0.5px; margin-bottom: 4px; }
.metric .value { font-size: 20px; font-weight: 700; color: var(--text-primary); }
.metric .unit { font-size: 12px; color: var(--text-secondary); margin-left: 2px; }

/* Map */
.map-container { margin-bottom: 20px; background: var(--bg-card); border-radius: 8px; box-shadow: 0 2px 8px var(--shadow); overflow: hidden; }
.map-container h3 { font-size: 16px; padding: 16px 24px 10px; color: var(--text-primary); border-bottom: 1px solid var(--border); margin: 0; }
#flagMap { height: 300px; }

/* Chart area */
.chart-container { margin-top: 16px; padding: 16px; background: var(--bg-inset); border-radius: 6px; text-align: center; }
.chart-container .no-data { color: var(--text-secondary); font-size: 14px; padding: 40px; }

/* Info panels */
.info-panels { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
.info-panel { background: var(--bg-card); border-radius: 8px; box-shadow: 0 2px 8px var(--shadow); overflow: hidden; }
.info-panel h3 { font-size: 14px; padding: 12px 16px; border-bottom: 1px solid var(--border); color: var(--text-primary); margin: 0; display: flex; align-items: center; gap: 8px; }
.info-panel h3 .panel-icon { font-size: 16px; opacity: 0.7; }
.panel-content { padding: 16px; min-height: 80px; }
.panel-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(110px, 1fr)); gap: 10px; }
.panel-metric { text-align: center; padding: 8px; background: var(--bg-inset); border-radius: 4px; }
.panel-metric .pm-label { font-size: 10px; text-transform: uppercase; color: var(--text-secondary); letter-spacing: 0.5px; margin-bottom: 2px; }
.panel-metric .pm-value { font-size: 16px; font-weight: 700; color: var(--text-primary); }
.panel-metric .pm-unit { font-size: 11px; color: var(--text-secondary); }
.panel-loading { color: var(--text-secondary); font-size: 13px; padding: 20px 0; text-align: center; }
.panel-error { color: #d63031; font-size: 13px; padding: 8px 0; }
.panel-empty { color: var(--text-secondary); font-size: 13px; padding: 8px 0; }
.panel-table { width: 100%; font-size: 13px; border-collapse: collapse; }
.panel-table th { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); font-size: 11px; text-transform: uppercase; color: var(--text-secondary); letter-spacing: 0.3px; }
.panel-table td { padding: 6px 8px; border-bottom: 1px solid var(--bg-inset); }
.escalation-banner { background: #d63031; color: #fff; padding: 8px 12px; border-radius: 4px; margin-top: 10px; font-size: 13px; font-weight: 600; }

/* Response protocol */
.response-protocol { background: var(--bg-card); border-radius: 8px; box-shadow: 0 2px 8px var(--shadow); padding: 24px; border-left: 4px solid #d63031; margin-bottom: 20px; }
.response-protocol h3 { font-size: 16px; margin-bottom: 16px; color: #d63031; border-bottom: 1px solid var(--border); padding-bottom: 10px; }
.protocol-actions { display: flex; gap: 12px; flex-wrap: wrap; }
.protocol-btn { padding: 10px 18px; border: none; border-radius: 6px; font-size: 13px; font-weight: 600; cursor: pointer; transition: all 0.15s; }
.protocol-btn:hover { filter: brightness(0.9); }
.btn-notify { background: var(--accent); color: #fff; }
.btn-evacuate { background: #d63031; color: #fff; }
.draft-output { margin-top: 16px; background: var(--bg-inset); border-radius: 6px; padding: 16px; font-size: 13px; line-height: 1.6; white-space: pre-wrap; font-family: -apple-system, BlinkMacSystemFont, sans-serif; position: relative; max-height: 400px; overflow-y: auto; color: var(--text-primary); }
.draft-output .draft-header { font-weight: 700; font-size: 14px; margin-bottom: 8px; color: var(--text-primary); }
.draft-btn-group { position: absolute; top: 8px; right: 8px; display: flex; gap: 4px; }
.copy-btn, .close-draft-btn { padding: 4px 12px; border: 1px solid var(--border); border-radius: 4px; background: var(--bg-card); color: var(--text-primary); font-size: 12px; cursor: pointer; }
.copy-btn:hover, .close-draft-btn:hover { background: var(--bg-inset); }
@keyframes marker-pulse { 0% { r: 10; opacity: 0.9; } 50% { r: 16; opacity: 0.4; } 100% { r: 10; opacity: 0.9; } }

/* Instrument status */
.status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 4px; vertical-align: middle; }
.status-active { background: #00b894; }
.status-offline { background: #d63031; }
.status-maintenance { background: #fdcb6e; }
.status-unknown { background: var(--text-secondary); }
.trend-arrow { font-size: 14px; margin-left: 4px; }
.trend-increasing { color: #d63031; }
.trend-decreasing { color: #00b894; }
.trend-stable { color: var(--text-secondary); }
.risk-badge { display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 11px; font-weight: 700; letter-spacing: 0.5px; }
.risk-critical { background: #d63031; color: #fff; }
.risk-high { background: #e17055; color: #fff; }
.risk-moderate { background: #fdcb6e; color: #2d3436; }
.risk-low { background: #00b894; color: #fff; }
.risk-unknown { background: var(--text-secondary); color: #fff; }
.enhance-toggle { display: flex; align-items: center; gap: 8px; margin-top: 10px; padding: 8px 12px; background: var(--bg-inset); border-radius: 4px; }
.enhance-toggle button { padding: 4px 12px; border: 1px solid var(--accent); border-radius: 4px; background: var(--bg-card); color: var(--accent); font-size: 12px; font-weight: 600; cursor: pointer; transition: all 0.15s; }
.enhance-toggle button.active { background: var(--accent); color: #fff; }
.enhance-toggle button:hover { filter: brightness(0.95); }
.obs-list { list-style: none; padding: 0; margin: 0; font-size: 13px; }
.obs-list li { padding: 4px 0; border-bottom: 1px solid var(--bg-inset); }
.obs-label { color: var(--text-secondary); font-size: 11px; text-transform: uppercase; letter-spacing: 0.3px; }

/* News ticker */
.news-ticker { position: relative; overflow: hidden; }
.news-ticker table.panel-table { table-layout: fixed; width: 100%; }
.news-ticker table.panel-table tbody { display: block; height: 200px; overflow: hidden; }
.news-ticker table.panel-table thead, .news-ticker table.panel-table tbody tr { display: table; width: 100%; table-layout: fixed; }
.news-ticker table.panel-table th:nth-child(1), .news-ticker table.panel-table td:nth-child(1) { width: 60%; }
.news-ticker table.panel-table th:nth-child(2), .news-ticker table.panel-table td:nth-child(2) { width: 25%; }
.news-ticker table.panel-table th:nth-child(3), .news-ticker table.panel-table td:nth-child(3) { width: 15%; }
.news-ticker-row { transition: opacity 0.4s ease; }
.news-ticker-row.hidden { opacity: 0; position: absolute; pointer-events: none; }
.news-badge { display: inline-block; padding: 1px 5px; border-radius: 3px; font-size: 9px; font-weight: 700; margin-left: 4px; vertical-align: middle; }
.news-badge-academic { background: #6c5ce7; color: #fff; }
.news-badge-media { background: var(--accent); color: #fff; }
.news-ticker-dots { display: flex; justify-content: center; gap: 4px; margin-top: 8px; }
.news-ticker-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--text-muted); cursor: pointer; transition: background 0.2s; }
.news-ticker-dot.active { background: var(--accent); }

/* Analyst actions */
.actions-card { background: var(--bg-card); border-radius: 8px; box-shadow: 0 2px 8px var(--shadow); padding: 24px; margin-bottom: 20px; }
.actions-card h3 { font-size: 16px; margin-bottom: 16px; color: var(--text-primary); border-bottom: 1px solid var(--border); padding-bottom: 10px; }
.action-row { display: flex; gap: 12px; align-items: flex-start; flex-wrap: wrap; }
.action-row .btn-group { display: flex; gap: 8px; }
.action-row button { padding: 8px 20px; border: 2px solid var(--border); border-radius: 6px; background: var(--bg-card); color: var(--text-primary); cursor: pointer; font-size: 14px; font-weight: 600; transition: all 0.15s; }
.action-row button:hover { border-color: var(--accent); }
.action-row button.active-Watch { background: #e17055; color: #fff; border-color: #e17055; }
.action-row button.active-Warning { background: #fdcb6e; color: #2d3436; border-color: #fdcb6e; }
.action-row button.active-Cleared { background: #00b894; color: #fff; border-color: #00b894; }
.note-area { flex: 1; min-width: 250px; }
.note-area textarea { width: 100%; height: 70px; padding: 10px; border: 2px solid var(--border); border-radius: 6px; font-family: inherit; font-size: 13px; resize: vertical; background: var(--bg-card); color: var(--text-primary); }
.note-area textarea:focus { outline: none; border-color: var(--accent); }
.save-indicator { font-size: 13px; color: #00b894; margin-top: 8px; opacity: 0; transition: opacity 0.3s; }
.save-indicator.visible { opacity: 1; }

/* Empty state */
.empty-state { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; color: var(--text-secondary); }
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
        <button onclick="filterFlags('POST-EVENT')">Post-Event</button>
        <button onclick="filterFlags('WARNING')">Warning</button>
        <button onclick="filterFlags('INFO')">Info</button>
    </div>
    <div class="sidebar-tools">
        <div class="tool-row">
            <label>Sort
                <select id="sortSelect" onchange="setSort(this.value)">
                    <option value="score">Anomaly score</option>
                    <option value="distance" id="sortDistanceOpt">Distance to source</option>
                    <option value="date">Peak date</option>
                </select>
            </label>
            <label id="relFilterLabel">Data quality
                <select id="relSelect" onchange="setReliability(this.value)">
                    <option value="ALL">All</option>
                    <option value="good">Good</option>
                    <option value="marginal">Marginal</option>
                    <option value="unreliable">Unreliable</option>
                </select>
            </label>
        </div>
        <div class="peak-hist" id="peakHist"></div>
        <div id="bulkArea"></div>
    </div>
    <div class="flag-list" id="flagList"></div>
</div>
<div class="main">
    $$DATA_BANNER$$
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
var SITE = $$SITE_JSON$$;
var currentFilter = 'ALL';
var currentSort = 'score';
var currentReliability = 'ALL';
var currentMonth = null;
var selectedFlagId = null;
var cachedExposure = null;
var MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

// ---- Known source + reliability helpers ----
function _kmXY(lat, lon, lat0) {
    return [lon * 111.32 * Math.cos(lat0 * Math.PI / 180), lat * 110.57];
}
// Distance (km) from a point to the known-source polygon; 0 inside, null if no source configured.
function distanceToSource(lat, lon) {
    var ks = SITE.known_source;
    if (!ks || !ks.polygon || ks.polygon.length < 3 || lat == null || lon == null) return null;
    var poly = ks.polygon, lat0 = poly[0][0], p = _kmXY(lat, lon, lat0);
    var inside = false, best = Infinity;
    for (var i = 0, j = poly.length - 1; i < poly.length; j = i++) {
        var a = _kmXY(poly[i][0], poly[i][1], lat0), b = _kmXY(poly[j][0], poly[j][1], lat0);
        if (((a[1] > p[1]) !== (b[1] > p[1])) && (p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0])) inside = !inside;
        var dx = b[0] - a[0], dy = b[1] - a[1];
        var t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy || 1);
        t = Math.max(0, Math.min(1, t));
        var ex = a[0] + t * dx - p[0], ey = a[1] + t * dy - p[1];
        best = Math.min(best, Math.sqrt(ex * ex + ey * ey));
    }
    return inside ? 0 : best;
}
function formatDistance(d) {
    if (d == null) return '';
    if (d === 0) return 'on known source';
    return (d < 10 ? d.toFixed(1) : Math.round(d)) + ' km from source';
}
// "Local (English)" label; romanized names are italic and say so on hover
function placeLabel(name, nameEn, source) {
    var out = escapeHtml(name || '');
    if (nameEn && nameEn !== name) {
        var en = escapeHtml(nameEn);
        if (source === 'romanized') en = '<i title="Romanized: OpenStreetMap has no English name for this place">' + en + '</i>';
        out += out ? ' (' + en + ')' : en;
    }
    return out;
}
function reliabilityOf(f) { return (f.quality && f.quality.reliability) || null; }
function qualitySummary(q) {
    var parts = [];
    if (q.mean_coherence != null) parts.push('mean coherence ' + q.mean_coherence.toFixed(2));
    if (q.unreliable_fraction != null) parts.push(Math.round(q.unreliable_fraction * 100) + '% unreliably unwrapped');
    if (q.n_pairs) parts.push(q.n_pairs_usable + ' of ' + q.n_pairs + ' pairs usable');
    return parts.join(' · ');
}
function reliabilityBadge(f) {
    var r = reliabilityOf(f);
    if (!r) return '';
    var label = {good: 'Good data', marginal: 'Marginal data', unreliable: 'Unreliable data'}[r] || r;
    return '<span class="rel-badge rel-' + r + '" title="' + escapeHtml(qualitySummary(f.quality)) + '">' + label + '</span>';
}
function peakMonth(f) { return f.peak_window_date ? f.peak_window_date.slice(0, 7) : null; }
function monthLabel(ym) { return MONTHS[parseInt(ym.slice(5, 7), 10) - 1] + ' ' + ym.slice(0, 4); }

for (var _i = 0; _i < FLAGS.length; _i++) {
    FLAGS[_i]._dist = distanceToSource(FLAGS[_i].center_lat, FLAGS[_i].center_lon);
}

// ---- Filtering + sorting ----
function filterFlags(severity) {
    currentFilter = severity;
    document.querySelectorAll('.sidebar-filter button').forEach(function(btn) {
        btn.classList.toggle('active', btn.textContent.toLowerCase() === severity.toLowerCase() || (severity === 'ALL' && btn.textContent === 'All'));
    });
    refreshSidebar();
}
function setSort(v) { currentSort = v; refreshSidebar(); }
function setReliability(v) { currentReliability = v; refreshSidebar(); }
function setMonth(ym) { currentMonth = (currentMonth === ym) ? null : ym; refreshSidebar(); }

// Flags passing severity + reliability filters (the histogram's population)
function baseFiltered() {
    return FLAGS.filter(function(f) {
        if (currentFilter !== 'ALL' && f.severity !== currentFilter) return false;
        if (currentReliability !== 'ALL' && reliabilityOf(f) !== currentReliability) return false;
        return true;
    });
}
function visibleFlags() {
    var list = baseFiltered().filter(function(f) { return !currentMonth || peakMonth(f) === currentMonth; });
    var cmp = {
        score: function(a, b) { return (b.score || 0) - (a.score || 0); },
        distance: function(a, b) {
            var da = a._dist == null ? Infinity : a._dist, db = b._dist == null ? Infinity : b._dist;
            return da - db || (b.score || 0) - (a.score || 0);
        },
        date: function(a, b) { return String(a.peak_window_date || '9999').localeCompare(String(b.peak_window_date || '9999')); }
    }[currentSort];
    return list.slice().sort(cmp);
}
function refreshSidebar() {
    renderPeakHistogram();
    renderFlagList();
    renderBulkArea();
}

function renderFlagList() {
    var list = document.getElementById('flagList');
    var filtered = visibleFlags();
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
            html += '<div class="flag-meta" style="margin-bottom:2px;">' + placeLabel(f.place_name, f.place_name_en, f.place_name_en_source) + '</div>';
        }
        html += '<div class="flag-meta">Score: ' + (f.score != null ? f.score.toFixed(2) : 'N/A');
        html += ' &middot; Z: ' + (f.peak_zscore != null ? f.peak_zscore.toFixed(1) : 'N/A');
        html += ' &middot; ' + formatArea(f.area_m2) + '</div>';
        var extra = [];
        if (f.peak_window_date) extra.push('peak ' + f.peak_window_date);
        if (f._dist != null) extra.push(formatDistance(f._dist));
        html += '<div class="flag-meta">' + formatCoord(f.center_lat, f.center_lon) + reliabilityBadge(f) + '</div>';
        if (extra.length) html += '<div class="flag-meta">' + extra.join(' &middot; ') + '</div>';
        html += '</div>';
    }
    if (filtered.length === 0) {
        html = '<div style="padding:30px 20px;color:var(--text-secondary);text-align:center;">No flags match these filters</div>';
    }
    list.innerHTML = html;
}

// ---- Peak-window histogram (by month) ----
function renderPeakHistogram() {
    var el = document.getElementById('peakHist');
    var pop = baseFiltered();
    var counts = {}, months = [];
    for (var i = 0; i < FLAGS.length; i++) { var m = peakMonth(FLAGS[i]); if (m && months.indexOf(m) < 0) months.push(m); }
    if (months.length === 0) { el.innerHTML = ''; return; }
    months.sort();
    // fill gaps so the axis is continuous
    var all = [], y = +months[0].slice(0, 4), mo = +months[0].slice(5, 7), last = months[months.length - 1];
    while (true) {
        var ym = y + '-' + (mo < 10 ? '0' : '') + mo;
        all.push(ym);
        if (ym >= last || all.length > 36) break;
        mo++; if (mo > 12) { mo = 1; y++; }
    }
    for (var k = 0; k < pop.length; k++) { var pm = peakMonth(pop[k]); if (pm) counts[pm] = (counts[pm] || 0) + 1; }
    var maxC = 1; all.forEach(function(m) { maxC = Math.max(maxC, counts[m] || 0); });
    var W = 300, H = 74, top = 12, base = 58, bw = W / all.length;
    var snow = SITE.snow_months || [];
    var svg = '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Flags by peak month">';
    all.forEach(function(m, i) {
        var x = i * bw, c = counts[m] || 0, h = (base - top) * c / maxC;
        if (snow.indexOf(+m.slice(5, 7)) >= 0) svg += '<rect x="' + x + '" y="0" width="' + bw + '" height="' + base + '" fill="rgba(116,185,255,0.10)"/>';
        var on = currentMonth === m;
        var fill = currentMonth && !on ? 'rgba(116,185,255,0.35)' : (on ? '#ffffff' : '#74b9ff');
        svg += '<rect class="bar" x="' + (x + bw * 0.18) + '" y="' + (base - h) + '" width="' + (bw * 0.64) + '" height="' + Math.max(h, c ? 1 : 0) + '" fill="' + fill + '" onclick="setMonth(\'' + m + '\')"><title>' + monthLabel(m) + ': ' + c + ' flag' + (c === 1 ? '' : 's') + '</title></rect>';
        svg += '<rect x="' + x + '" y="0" width="' + bw + '" height="' + H + '" fill="transparent" style="cursor:pointer" onclick="setMonth(\'' + m + '\')"><title>' + monthLabel(m) + ': ' + c + '</title></rect>';
        if (c) svg += '<text x="' + (x + bw / 2) + '" y="' + (base - h - 3) + '" text-anchor="middle" font-size="8.5" fill="#dfe6e9">' + c + '</text>';
        svg += '<text x="' + (x + bw / 2) + '" y="' + (H - 4) + '" text-anchor="middle" font-size="8.5" fill="' + (on ? '#ffffff' : '#a0a0b0') + '">' + MONTHS[+m.slice(5, 7) - 1].charAt(0) + '</text>';
    });
    svg += '</svg>';
    var title = '<div class="hist-title"><span>Flags by peak month' + (snow.length ? ' · shaded = snow season' : '') + '</span>';
    if (currentMonth) title += '<button onclick="setMonth(null)">Clear ' + monthLabel(currentMonth) + '</button>';
    title += '</div>';
    el.innerHTML = title + svg;
}

// ---- Bulk classification ----
var BULK_REASONS = ['Seasonal decorrelation (snow)', 'Unwrapping artefact', 'Atmospheric or topographic phase', 'Outside area of interest'];
var bulkOpen = false;
function renderBulkArea(errorMsg) {
    var el = document.getElementById('bulkArea');
    var n = visibleFlags().length;
    if (n === 0) { el.innerHTML = ''; bulkOpen = false; return; }
    if (!bulkOpen) {
        el.innerHTML = '<button class="bulk-btn" onclick="openBulk()">Clear the ' + n + ' flag' + (n === 1 ? '' : 's') + ' shown…</button>';
        return;
    }
    var html = '<div class="bulk-form"><div>Mark the <strong>' + n + '</strong> flag' + (n === 1 ? '' : 's') + ' shown as Cleared. Each is recorded in the audit log with this reason.</div>';
    html += '<select id="bulkReason" aria-label="Reason">';
    for (var i = 0; i < BULK_REASONS.length; i++) html += '<option>' + BULK_REASONS[i] + '</option>';
    html += '</select><input id="bulkDetail" type="text" placeholder="Add detail (optional)" aria-label="Detail">';
    if (errorMsg) html += '<div class="bulk-error">' + escapeHtml(errorMsg) + '</div>';
    html += '<div class="bulk-actions"><button class="confirm" onclick="submitBulk()">Clear ' + n + '</button><button onclick="closeBulk()">Cancel</button></div></div>';
    el.innerHTML = html;
}
function openBulk() { bulkOpen = true; renderBulkArea(); }
function closeBulk() { bulkOpen = false; renderBulkArea(); }
function submitBulk() {
    var ids = visibleFlags().map(function(f) { return f.flag_id; });
    var reason = document.getElementById('bulkReason').value;
    var detail = document.getElementById('bulkDetail').value.trim();
    if (detail) reason += ' — ' + detail;
    var xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/classify_bulk', true);
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.onload = function() {
        var resp = {};
        try { resp = JSON.parse(xhr.responseText); } catch (e) {}
        if (xhr.status !== 200) { renderBulkArea(resp.error || ('Request failed (' + xhr.status + ')')); return; }
        for (var fid in resp.updated) STATE[fid] = resp.updated[fid];
        bulkOpen = false;
        refreshSidebar();
        if (selectedFlagId != null && resp.updated[String(selectedFlagId)]) renderDetail(selectedFlagId);
    };
    xhr.onerror = function() { renderBulkArea('Network error — nothing was changed'); };
    xhr.send(JSON.stringify({flag_ids: ids, classification: 'Cleared', note: reason}));
}

// ---- Place names arrive in the background ----
function pollPlaceNames() {
    var xhr = new XMLHttpRequest();
    xhr.open('GET', '/api/place_names', true);
    xhr.onload = function() {
        if (xhr.status !== 200) return;
        var resp = JSON.parse(xhr.responseText), changed = false;
        for (var i = 0; i < FLAGS.length; i++) {
            var n = resp.names[String(FLAGS[i].flag_id)];
            if (n && (n.name !== FLAGS[i].place_name || n.name_en !== FLAGS[i].place_name_en)) {
                FLAGS[i].place_name = n.name; FLAGS[i].place_name_en = n.name_en;
                FLAGS[i].place_name_en_source = n.name_en_source; changed = true;
            }
        }
        if (changed) renderFlagList();
        if (resp.pending) setTimeout(pollPlaceNames, 3000);
    };
    xhr.send();
}

function renderStats() {
    var total = FLAGS.length;
    var crit = 0, post = 0, warn = 0, info = 0;
    for (var i = 0; i < FLAGS.length; i++) {
        if (FLAGS[i].severity === 'CRITICAL') crit++;
        else if (FLAGS[i].severity === 'POST-EVENT') post++;
        else if (FLAGS[i].severity === 'WARNING') warn++;
        else info++;
    }
    var el = document.getElementById('sidebarStats');
    el.innerHTML =
        '<div class="stat"><div class="stat-value">' + total + '</div><div class="stat-label">Total</div></div>' +
        '<div class="stat"><div class="stat-value critical">' + crit + '</div><div class="stat-label">Critical</div></div>' +
        '<div class="stat"><div class="stat-value" style="color:#636e72">' + post + '</div><div class="stat-label">Post-Event</div></div>' +
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
        locationLabel = placeLabel(f.place_name, f.place_name_en, f.place_name_en_source) + ' — ' + locationLabel;
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
    if (f.peak_window_date) html += metric('Peak window', f.peak_window_date, '');
    if (f._dist != null) html += metric('Distance to source', f._dist === 0 ? 'On source' : f._dist.toFixed(2), f._dist === 0 ? '' : ' km');
    html += '</div>';

    if (f.quality) {
        var q = f.quality;
        html += '<div class="quality-row">' + reliabilityBadge(f);
        html += '<span>Mean coherence <strong>' + (q.mean_coherence != null ? q.mean_coherence.toFixed(2) : 'N/A') + '</strong></span>';
        html += '<span><strong>' + Math.round((q.unreliable_fraction || 0) * 100) + '%</strong> of pixel-pairs unreliably unwrapped</span>';
        html += '<span><strong>' + q.n_pairs_usable + ' of ' + q.n_pairs + '</strong> pairs usable</span>';
        if (f.severity_uncapped) {
            html += '<span class="cap-note">The anomaly score alone would make this ' + f.severity_uncapped + '. It is shown as ' + f.severity + ' because the data under it is unreliable.</span>';
        }
        html += '</div>';
    }

    // Chart area
    html += '<div class="chart-container" id="tsChart">';
    if (f.timeseries && f.timeseries.dates && f.timeseries.values) {
        html += renderTimeseriesSVG(f);
    } else {
        html += '<div class="no-data">No displacement time-series data available for this flag.<br>';
        html += '<span style="font-size:12px;color:#b2bec3;">Run the full pipeline with time-series export to populate this chart.</span></div>';
    }
    html += '</div>';
    html += '</div>';

    if (f.ifg_strip) {
        html += '<div class="detail-card"><h3>Interferograms</h3>';
        html += '<img class="ifg-strip" src="/figures/' + encodeURI(f.ifg_strip.replace(/^figures\//, '')) + '" alt="Wrapped-phase tiles for each interferogram pair around flag ' + f.flag_id + '">';
        html += '<div class="ifg-caption">Wrapped phase around the flag for every pair; the cross marks the flag centre. Motion confined to the flag that grows towards the event suggests real deformation. Fringes that also appear in winter pairs, or that follow the terrain across the whole tile, point to atmospheric or topographic phase instead.</div>';
        html += '</div>';
    }

    // --- Analyst Classification (between flag overview and detail panels) ---
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
        html += '<div style="font-size:12px;color:var(--text-secondary);margin-top:10px;">Last updated: ' + escapeHtml(st.updated_at) + '</div>';
    }
    html += '<div class="save-indicator" id="saveIndicator">Saved</div>';
    // Response protocol (inside Analyst Classification, shown when Voight escalation triggers)
    html += '<div id="responseProtocol" style="display:none;margin-top:16px;padding-top:16px;border-top:1px solid var(--border);">';
    html += '<h4 style="font-size:14px;color:#d63031;margin-bottom:10px;">Response Protocol</h4>';
    html += '<div class="protocol-actions">';
    html += '<button class="protocol-btn btn-notify" onclick="draftNotification()">Draft Authority Notification</button>';
    html += '<button class="protocol-btn btn-evacuate" onclick="draftEvacuation()">Draft Evacuation Advisory</button>';
    html += '</div>';
    html += '<div id="draftOutput"></div>';
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

    // --- Webcams (full width) ---
    html += '<div class="info-panels">';
    html += '<div class="info-panel" style="grid-column: 1 / -1;"><h3><span class="panel-icon">&#128247;</span> Webcams</h3><div class="panel-content" id="panel-webcams"></div></div>';
    html += '</div>';

    // --- Info panels row 5: Field Reconnaissance ---
    html += '<div class="info-panels">';
    html += '<div class="info-panel" style="grid-column: 1 / -1;"><h3><span class="panel-icon">&#128269;</span> Field Reconnaissance</h3><div class="panel-content" id="panel-fieldreports"></div></div>';
    html += '</div>';

    // --- Info panels row 6: News Feed ---
    html += '<div class="info-panels">';
    html += '<div class="info-panel" style="grid-column: 1 / -1;"><h3><span class="panel-icon">&#128240;</span> News Feed</h3><div class="panel-content" id="panel-news"></div></div>';
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
        loadPanel('webcams', f.center_lat, f.center_lon);
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

    var sevColors = {CRITICAL: '#d63031', WARNING: '#fdcb6e', INFO: '#74b9ff', 'POST-EVENT': '#636e72'};
    for (var i = 0; i < FLAGS.length; i++) {
        var fl = FLAGS[i];
        if (fl.center_lat == null || fl.center_lon == null) continue;
        var col = sevColors[fl.severity] || '#636e72';
        var isSelected = (fl.flag_id === flag.flag_id);
        if (isSelected) {
            var pulseIcon = L.divIcon({
                className: '',
                html: '<svg width="60" height="60" viewBox="0 0 60 60" style="overflow:visible;">' +
                      '<circle cx="30" cy="30" r="12" fill="none" stroke="' + col + '" stroke-width="3" stroke-opacity="0.8">' +
                      '<animate attributeName="r" values="12;26;12" dur="1.8s" repeatCount="indefinite"/>' +
                      '<animate attributeName="stroke-opacity" values="0.8;0;0.8" dur="1.8s" repeatCount="indefinite"/>' +
                      '<animate attributeName="stroke-width" values="3;1;3" dur="1.8s" repeatCount="indefinite"/>' +
                      '</circle>' +
                      '<circle cx="30" cy="30" r="12" fill="none" stroke="' + col + '" stroke-width="2" stroke-opacity="0.5">' +
                      '<animate attributeName="r" values="12;20;12" dur="1.8s" begin="0.3s" repeatCount="indefinite"/>' +
                      '<animate attributeName="stroke-opacity" values="0.5;0;0.5" dur="1.8s" begin="0.3s" repeatCount="indefinite"/>' +
                      '</circle></svg>',
                iconSize: [60, 60],
                iconAnchor: [30, 30]
            });
            L.marker([fl.center_lat, fl.center_lon], {icon: pulseIcon, interactive: false, pane: 'markerPane'}).addTo(dashMap);
        }
        L.circleMarker([fl.center_lat, fl.center_lon], {
            radius: isSelected ? 10 : 5,
            color: isSelected ? '#fff' : col,
            weight: isSelected ? 3 : 1,
            fillColor: col,
            fillOpacity: isSelected ? 0.9 : 0.5
        }).addTo(dashMap).bindPopup('Flag ' + fl.flag_id + ' (' + fl.severity + ')');
    }

    var overlays = {};
    var ks = SITE.known_source;
    if (ks && ks.polygon && ks.polygon.length >= 3) {
        overlays['Known source'] = L.polygon(ks.polygon, {
            color: '#e84393', weight: 2, dashArray: '6 4', fillColor: '#e84393', fillOpacity: 0.10
        }).bindTooltip(escapeHtml(ks.label || 'Known source'), {sticky: true}).addTo(dashMap);
    }
    var ex = SITE.extent;
    if (ex && ex.length === 4) {
        overlays['Analysed area'] = L.rectangle([[ex[0], ex[1]], [ex[2], ex[3]]], {
            color: '#636e72', weight: 1.5, dashArray: '2 5', fill: false, interactive: false
        }).addTo(dashMap);
    }
    L.control.layers({ 'Street': osm, 'Satellite': satellite }, overlays).addTo(dashMap);
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
        case 'webcams': renderWebcamsPanel(data, el); break;
        default: el.innerHTML = '<div class="panel-empty">Unknown panel</div>';
    }
}

// ---- Panel renderers ----

function renderVoightPanel(flag) {
    var el = document.getElementById('panel-voight');
    if (!el) return;
    if (!flag.voight_fit || flag.voight_fit.r_squared == null) {
        el.innerHTML = '<div class="panel-empty">No Voight analysis available.<br><span style="font-size:12px;">Inverse-velocity fit did not meet the R² threshold for this flag.</span></div>';
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
        html += '<div style="font-size:12px;color:var(--text-secondary);margin-top:8px;">95% confidence window: ' + escapeHtml(v.failure_window) + '</div>';
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
    html += '<div style="font-size:12px;color:var(--text-secondary);margin-top:10px;">Recent scenes: ';
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
        el.innerHTML = '<div class="panel-empty">Overpass API unavailable — settlement data could not be loaded.<br><span style="font-size:12px;color:var(--text-secondary);">Try reloading the page later.</span></div>';
        return;
    }
    if (!data.features || data.features.length === 0) {
        el.innerHTML = '<div class="panel-empty">No settlements or infrastructure found within ' + (data.radius_km || 10) + ' km</div>';
        return;
    }
    var html = '<div style="font-size:11px;color:var(--text-secondary);margin-bottom:8px;">Within ' + data.radius_km + ' km (straight-line, not flow-routed)</div>';
    html += '<table class="panel-table"><tr><th>Name</th><th>Type</th><th>Dist</th></tr>';
    for (var i = 0; i < data.features.length; i++) {
        var feat = data.features[i];
        var displayName;
        if (!feat.name) {
            displayName = '<span style="color:var(--text-secondary);font-style:italic;">Unnamed ' + escapeHtml(feat.type) + '</span>';
        } else {
            displayName = escapeHtml(feat.name);
            if (feat.name_en) {
                var en = escapeHtml(feat.name_en);
                if (feat.name_en_source === 'romanized') en = '<i title="Romanized: OpenStreetMap has no English name for this place">' + en + '</i>';
                displayName += ' <span style="color:var(--text-secondary);font-size:12px;">(' + en + ')</span>';
            }
        }
        html += '<tr><td>' + displayName + '</td>';
        html += '<td style="color:var(--text-secondary);">' + escapeHtml(feat.type) + '</td>';
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
    html += '<span style="font-size:12px;color:var(--text-secondary);">Enhanced Monitoring:</span>';
    html += '<button class="' + (enhanced ? 'active' : '') + '" onclick="toggleEnhancedMonitoring()">' + (enhanced ? 'Requested' : 'Request') + '</button>';
    if (st.enhanced_monitoring_at) {
        html += '<span style="font-size:11px;color:var(--text-secondary);">since ' + escapeHtml(st.enhanced_monitoring_at) + '</span>';
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
        el.innerHTML = '<div class="panel-empty">No in-situ instruments are registered for this site.<br><span style="font-size:12px;color:var(--text-secondary);">To list deployed sensors, add an instruments.json file to this run’s data folder.</span></div>';
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

function renderWebcamsPanel(data, el) {
    if (!data.configured) {
        el.innerHTML = '<div class="panel-empty">Webcams are not set up.<br><span style="font-size:12px;color:var(--text-secondary);">Add cameras to config/webcams.yaml, or set the GLEWS_WINDY_API_KEY environment variable to a Windy Webcams API key, and restart the dashboard.</span></div>';
        return;
    }
    var cams = data.webcams || [];
    var providers = {};
    cams.forEach(function(c) { if (c.provider) providers[c.provider] = c.provider_url; });
    if (!cams.length) providers['Windy'] = 'https://www.windy.com/';
    var credit = '<div class="webcam-credit">Webcams provided by ' + Object.keys(providers).map(function(p) {
        return providers[p] ? '<a href="' + escapeHtml(providers[p]) + '" target="_blank" rel="noopener">' + escapeHtml(p) + '</a>' : escapeHtml(p);
    }).join(', ') + '</div>';
    var err = data.error ? '<div class="panel-error">' + escapeHtml(data.error) + '</div>' : '';
    if (!cams.length) {
        el.innerHTML = err + '<div class="panel-empty">No webcams within ' + data.radius_km + ' km of this flag.</div>' + credit;
        return;
    }
    var statusLabel = {live: 'Live', upcoming: 'Scheduled', ended: 'Stream ended', unavailable: 'Stream removed', unknown: 'Live status unknown'};
    var html = err + '<div class="webcam-grid">';
    for (var i = 0; i < cams.length; i++) {
        var c = cams[i];
        var img = c.image || c.image_daylight;
        var fallback = c.image_daylight && c.image_daylight !== img ? ' onerror="this.onerror=null;this.src=\'' + escapeHtml(c.image_daylight) + '\'"' : '';
        html += '<a class="webcam" href="' + escapeHtml(c.url) + '" target="_blank" rel="noopener">';
        html += img ? '<img src="' + escapeHtml(img) + '" alt="' + escapeHtml(c.title) + '" loading="lazy"' + fallback + '>' : '<div class="webcam-noimg">No image</div>';
        html += '<div class="webcam-title">' + escapeHtml(c.title) + '</div>';
        var meta = [c.distance_km.toFixed(1) + ' km away'];
        if (c.elevation_m) meta.push(c.elevation_m.toLocaleString() + ' m');
        if (c.status && c.status !== 'active') meta.push(statusLabel[c.status] || escapeHtml(c.status));
        if (c.stream_id_updated) meta.push('<span title="The stream restarted under a new YouTube ID; GLEWS found it from the channel">new stream ID</span>');
        html += '<div class="webcam-meta">' + meta.join(' · ') + '</div>';
        if (c.faces) html += '<div class="webcam-meta">Faces: ' + escapeHtml(c.faces) + ' <span title="Taken from the stream title; not checked against the view">(unverified)</span></div>';
        html += '<div class="webcam-meta webcam-provider">' + escapeHtml(c.provider || '') + (c.approximate_location ? ' · location approximate' : '') + '</div>';
        html += '</a>';
    }
    html += '</div>';
    if (data.radius_km > 50) html += '<div class="webcam-meta" style="margin-top:6px;">None within 50 km; showing the nearest within ' + data.radius_km + ' km. Distant cameras show regional weather and snow cover, not the flagged slope.</div>';
    el.innerHTML = html + credit;
}

function renderFieldReportsPanel(data, el) {
    if (data.message && (!data.reports || data.reports.length === 0)) {
        el.innerHTML = '<div class="panel-empty">No field reconnaissance has been reported for this site.<br><span style="font-size:12px;color:var(--text-secondary);">To show field observations, add a field_reports.json file to this run’s data folder.</span></div>';
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
        html += '<div style="padding:10px;background:var(--bg-inset);border-radius:6px;margin-bottom:8px;">';
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
            html += '<div style="font-size:12px;color:var(--text-secondary);margin-top:6px;font-style:italic;">' + escapeHtml(rpt.notes) + '</div>';
        }
        html += '</div>';
    }
    el.innerHTML = html;
}

var _newsInterval = null;
function renderNewsPanel(data, el) {
    if (_newsInterval) { clearInterval(_newsInterval); _newsInterval = null; }
    if (!data.articles || data.articles.length === 0) {
        el.innerHTML = '<div class="panel-empty">No related articles found<br><span style="font-size:12px;color:var(--text-secondary);">Sources: Google News, OpenAlex, Crossref</span></div>';
        return;
    }
    var arts = data.articles;
    var perPage = 5;
    var pages = Math.ceil(arts.length / perPage);
    var curPage = 0;

    function fmtDate(d) {
        if (!d) return '';
        var m = d.match(/^(\d{4})-(\d{2})-(\d{2})$/);
        if (m) {
            var months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
            return m[3] + ' ' + months[parseInt(m[2],10)-1] + ' ' + m[1];
        }
        return d;
    }

    function renderPage(p) {
        var start = p * perPage;
        var end = Math.min(start + perPage, arts.length);
        var rows = '';
        for (var i = start; i < end; i++) {
            var a = arts[i];
            var titleHtml = escapeHtml(a.title);
            if (a.url) titleHtml = safeLink(a.url, a.title);
            rows += '<tr class="news-ticker-row"><td>' + titleHtml + '</td>';
            rows += '<td style="color:var(--text-secondary);font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">' + escapeHtml(a.source || '') + '</td>';
            rows += '<td style="white-space:nowrap;font-size:12px;">' + escapeHtml(fmtDate(a.date)) + '</td></tr>';
        }
        while (end - start < perPage) {
            rows += '<tr class="news-ticker-row"><td>&nbsp;</td><td></td><td></td></tr>';
            end++;
        }
        return rows;
    }

    function buildDots() {
        if (pages <= 1) return '';
        var d = '<div class="news-ticker-dots">';
        for (var i = 0; i < pages; i++) {
            d += '<span class="news-ticker-dot' + (i === curPage ? ' active' : '') + '" data-page="' + i + '"></span>';
        }
        d += '</div>';
        return d;
    }

    var header = '<div style="font-size:11px;color:var(--text-secondary);margin-bottom:8px;">Related articles from media and academic sources</div>';
    el.innerHTML = header + '<div class="news-ticker"><table class="panel-table"><thead><tr><th>Title</th><th>Source</th><th>Date</th></tr></thead><tbody id="newsRows">' + renderPage(0) + '</tbody></table>' + buildDots() + '</div>';

    if (pages <= 1) return;

    function showPage(p) {
        curPage = p;
        var tbody = document.getElementById('newsRows');
        if (tbody) tbody.innerHTML = renderPage(p);
        var dots = el.querySelectorAll('.news-ticker-dot');
        for (var i = 0; i < dots.length; i++) {
            dots[i].className = 'news-ticker-dot' + (i === p ? ' active' : '');
        }
    }

    el.addEventListener('click', function(e) {
        if (e.target.classList.contains('news-ticker-dot')) {
            var p = parseInt(e.target.getAttribute('data-page'));
            showPage(p);
            if (_newsInterval) { clearInterval(_newsInterval); }
            _newsInterval = setInterval(function() { showPage((curPage + 1) % pages); }, 5000);
        }
    });

    _newsInterval = setInterval(function() { showPage((curPage + 1) % pages); }, 5000);
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
    html += '<div class="draft-btn-group">';
    html += '<button class="copy-btn" onclick="copyDraft()">Copy</button>';
    html += '<button class="close-draft-btn" onclick="closeDraft()">Close</button>';
    html += '</div>';
    html += '<pre style="margin:0;white-space:pre-wrap;font-family:inherit;font-size:inherit;" id="draftText">' + escapeHtml(text) + '</pre>';
    html += '</div>';
    el.innerHTML = html;
}

function closeDraft() {
    var el = document.getElementById('draftOutput');
    if (el) el.innerHTML = '';
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
// Time-proportional axis so acquisition gaps show; snow season shaded; event
// date marked; per-pair coherence drawn beneath when quality data exist.
function renderTimeseriesSVG(f) {
    var dates = f.timeseries.dates, values = f.timeseries.values;
    var pts = [];
    for (var i = 0; i < dates.length; i++) {
        if (values[i] == null || isNaN(values[i])) continue;
        pts.push({t: Date.parse(dates[i] + 'T00:00:00Z'), v: values[i], d: dates[i]});
    }
    if (pts.length < 2) return '<div class="no-data">Insufficient data points</div>';
    var pairs = (f.quality && f.quality.pairs) || [];
    var DAY = 86400000;
    var t0 = pts[0].t, t1 = pts[pts.length - 1].t;
    var ev = SITE.event_date ? Date.parse(SITE.event_date + 'T00:00:00Z') : null;
    if (ev != null && ev > t1 && ev - t1 < 90 * DAY) t1 = ev;
    t1 += 4 * DAY;

    var w = 700, h = 190, padL = 70, padR = 20, padT = 30;
    var cohH = pairs.length ? 46 : 0, gap = pairs.length ? 22 : 0;
    var axisY = padT + h + gap + cohH;
    var totalH = axisY + 44;
    var vmin = Infinity, vmax = -Infinity;
    pts.forEach(function(p) { vmin = Math.min(vmin, p.v); vmax = Math.max(vmax, p.v); });
    if (vmin === vmax) { vmin -= 0.01; vmax += 0.01; }
    var m = (vmax - vmin) * 0.1; vmin -= m; vmax += m;
    function X(t) { return padL + (t - t0) / (t1 - t0) * w; }
    function Y(v) { return padT + h - (v - vmin) / (vmax - vmin) * h; }

    var cs = getComputedStyle(document.documentElement);
    function tok(n, d) { return cs.getPropertyValue(n).trim() || d; }
    var cAxis = tok('--chart-axis', '#636e72'), cGrid = tok('--chart-grid', '#dfe6e9'), cLine = tok('--chart-line', '#0984e3');
    var cLabel = tok('--chart-label', '#2d3436'), cSnow = tok('--chart-snow', 'rgba(116,185,255,0.14)');
    var cCoh = tok('--chart-coh', '#a4b0be'), cBad = tok('--chart-coh-bad', '#e17055'), cCard = tok('--bg-inset', '#f5f6fa');

    var svg = '<svg viewBox="0 0 ' + (w + padL + padR) + ' ' + totalH + '" style="max-width:100%;height:auto;" role="img" aria-label="Displacement time series">';
    // snow-season shading (month spans) across plot + coherence strip
    var snow = SITE.snow_months || [];
    if (snow.length) {
        var d0 = new Date(t0), cur = Date.UTC(d0.getUTCFullYear(), d0.getUTCMonth(), 1), shaded = false;
        while (cur < t1) {
            var dt = new Date(cur), next = Date.UTC(dt.getUTCFullYear(), dt.getUTCMonth() + 1, 1);
            if (snow.indexOf(dt.getUTCMonth() + 1) >= 0) {
                var xa = X(Math.max(cur, t0)), xb = X(Math.min(next, t1));
                svg += '<rect x="' + xa + '" y="' + padT + '" width="' + Math.max(0, xb - xa) + '" height="' + (axisY - padT) + '" fill="' + cSnow + '"/>';
                shaded = true;
            }
            cur = next;
        }
    }
    // y grid + labels
    var range = vmax - vmin, rangeCm = range * 100, dp = rangeCm < 1 ? 2 : rangeCm < 10 ? 1 : 0;
    for (var yk = 0; yk <= 4; yk++) {
        var yv = vmin + range * yk / 4, yp = Y(yv);
        svg += '<line x1="' + padL + '" y1="' + yp + '" x2="' + (padL + w) + '" y2="' + yp + '" stroke="' + cGrid + '" stroke-width="0.5"/>';
        svg += '<text x="' + (padL - 8) + '" y="' + (yp + 3) + '" text-anchor="end" font-size="10" fill="' + cAxis + '">' + (yv * 100).toFixed(dp) + '</text>';
    }
    svg += '<line x1="' + padL + '" y1="' + padT + '" x2="' + padL + '" y2="' + (padT + h) + '" stroke="' + cAxis + '"/>';
    // month ticks along the bottom axis
    svg += '<line x1="' + padL + '" y1="' + axisY + '" x2="' + (padL + w) + '" y2="' + axisY + '" stroke="' + cAxis + '"/>';
    var d0b = new Date(t0), tick = Date.UTC(d0b.getUTCFullYear(), d0b.getUTCMonth() + 1, 1);
    var nMonths = (t1 - t0) / (30.4 * DAY), every = nMonths > 14 ? 3 : nMonths > 7 ? 2 : 1, mi = 0;
    while (tick < t1) {
        var td = new Date(tick);
        if (mi % every === 0) {
            var tx = X(tick);
            svg += '<line x1="' + tx + '" y1="' + axisY + '" x2="' + tx + '" y2="' + (axisY + 4) + '" stroke="' + cAxis + '"/>';
            var lab = MONTHS[td.getUTCMonth()] + (td.getUTCMonth() === 0 || mi === 0 ? ' ' + td.getUTCFullYear() : '');
            svg += '<text x="' + tx + '" y="' + (axisY + 16) + '" text-anchor="middle" font-size="10" fill="' + cAxis + '">' + lab + '</text>';
        }
        tick = Date.UTC(td.getUTCFullYear(), td.getUTCMonth() + 1, 1); mi++;
    }
    // coherence strip: one bar per pair spanning its dates
    if (pairs.length) {
        var cy0 = padT + h + gap, cy1 = cy0 + cohH;
        svg += '<text x="' + (padL - 8) + '" y="' + (cy0 + cohH / 2 + 3) + '" text-anchor="end" font-size="10" fill="' + cAxis + '">Coherence</text>';
        svg += '<line x1="' + padL + '" y1="' + (cy1 - 0.3 * cohH) + '" x2="' + (padL + w) + '" y2="' + (cy1 - 0.3 * cohH) + '" stroke="' + cAxis + '" stroke-dasharray="2 3" stroke-width="0.7"/>';
        for (var k = 0; k < pairs.length; k++) {
            var pr = pairs[k], ra = Date.parse(pr.ref + 'T00:00:00Z'), rb = Date.parse(pr.sec + 'T00:00:00Z');
            var bx = X(Math.max(ra, t0)), bw = Math.max(1, X(Math.min(rb, t1)) - bx - 1.5), bh = Math.max(0, Math.min(1, pr.coherence || 0)) * cohH;
            svg += '<rect x="' + bx + '" y="' + (cy1 - bh) + '" width="' + bw + '" height="' + bh + '" fill="' + (pr.usable ? cCoh : cBad) + '">' +
                   '<title>' + pr.ref + ' → ' + pr.sec + ': coherence ' + (pr.coherence != null ? pr.coherence.toFixed(2) : 'N/A') + ', ' + Math.round((pr.unreliable_fraction || 0) * 100) + '% unreliably unwrapped' + (pr.usable ? '' : ' (not usable)') + '</title></rect>';
        }
    }
    // series: dashed across gaps longer than 1.6x the typical spacing
    var steps = [];
    for (var s2 = 1; s2 < pts.length; s2++) steps.push(pts[s2].t - pts[s2 - 1].t);
    var sorted = steps.slice().sort(function(a, b) { return a - b; }), med = sorted[Math.floor(sorted.length / 2)];
    for (var q2 = 1; q2 < pts.length; q2++) {
        var isGap = steps[q2 - 1] > 1.6 * med;
        svg += '<line x1="' + X(pts[q2 - 1].t) + '" y1="' + Y(pts[q2 - 1].v) + '" x2="' + X(pts[q2].t) + '" y2="' + Y(pts[q2].v) + '" stroke="' + cLine + '" stroke-width="2"' + (isGap ? ' stroke-dasharray="4 4" stroke-opacity="0.6"' : '') + '><title>' + (isGap ? 'Acquisition gap: ' + Math.round(steps[q2 - 1] / DAY) + ' days' : '') + '</title></line>';
    }
    // points: hollow when the pair ending on that date is unusable
    var unusableEnd = {};
    pairs.forEach(function(pr) { if (!pr.usable) unusableEnd[pr.sec] = true; });
    pts.forEach(function(p) {
        var bad = unusableEnd[p.d];
        svg += '<circle cx="' + X(p.t) + '" cy="' + Y(p.v) + '" r="3.2" fill="' + (bad ? cCard : cLine) + '" stroke="' + cLine + '" stroke-width="1.5"><title>' + p.d + ': ' + (p.v * 100).toFixed(2) + ' cm' + (bad ? ' (pair unusable)' : '') + '</title></circle>';
    });
    // event marker
    if (ev != null && ev >= t0 && ev <= t1) {
        var exx = X(ev);
        svg += '<line x1="' + exx + '" y1="' + (padT - 6) + '" x2="' + exx + '" y2="' + axisY + '" stroke="#d63031" stroke-width="1.5" stroke-dasharray="5 3"/>';
        svg += '<text x="' + (exx - 4) + '" y="' + (padT - 10) + '" text-anchor="end" font-size="10" font-weight="600" fill="#d63031">Event ' + SITE.event_date + '</text>';
    }
    // legend + axis title
    var lx = padL;
    svg += '<text x="' + lx + '" y="14" font-size="10" fill="' + cAxis + '">';
    svg += (snow.length ? '<tspan fill="#74b9ff">■</tspan> snow season   ' : '') + '● usable pair   ○ unusable pair   - - acquisition gap' + (pairs.length ? '   ▮ coherence (dashed line = 0.3; orange = unusable)' : '');
    svg += '</text>';
    svg += '<text transform="rotate(-90)" x="' + (-(padT + h / 2)) + '" y="16" text-anchor="middle" font-size="12" fill="' + cLabel + '">LOS displacement (cm)</text>';
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

// ---- Theme toggle ----
function toggleTheme() {
    var root = document.documentElement;
    var cur = root.getAttribute('data-theme');
    if (cur === 'dark') { root.setAttribute('data-theme', 'light'); }
    else if (cur === 'light') { root.removeAttribute('data-theme'); }
    else { root.setAttribute('data-theme', 'dark'); }
    updateThemeIcon();
    rerenderChart();
}
// The chart reads theme colours when drawn, so redraw it after a theme change
function rerenderChart() {
    var el = document.getElementById('tsChart');
    var f = FLAGS.find(function(x) { return x.flag_id === selectedFlagId; });
    if (el && f && f.timeseries && f.timeseries.dates && f.timeseries.values) el.innerHTML = renderTimeseriesSVG(f);
}
function updateThemeIcon() {
    var btn = document.getElementById('themeToggle');
    if (!btn) return;
    var dark = document.documentElement.getAttribute('data-theme') === 'dark' ||
        (!document.documentElement.getAttribute('data-theme') && window.matchMedia('(prefers-color-scheme: dark)').matches);
    btn.textContent = dark ? '☀' : '☾';
}
updateThemeIcon();
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function() { updateThemeIcon(); rerenderChart(); });

// Initial render
(function initSidebarTools() {
    var hasSource = FLAGS.some(function(f) { return f._dist != null; });
    var hasQuality = FLAGS.some(function(f) { return !!f.quality; });
    if (!hasSource) document.getElementById('sortDistanceOpt').remove();
    if (!hasQuality) document.getElementById('relFilterLabel').style.display = 'none';
})();
renderStats();
refreshSidebar();
pollPlaceNames();
</script>
<button class="theme-toggle" id="themeToggle" onclick="toggleTheme()" title="Toggle dark mode">&#9790;</button>
</body>
</html>
"""


def _collection_properties(data_dir: str | Path) -> dict:
    path = Path(data_dir) / "flags.geojson"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text()).get("properties", {}) or {}
    except (json.JSONDecodeError, OSError):
        return {}


def load_data_source(data_dir: str | Path) -> str | None:
    """Return the ``data_source`` tag from flags.geojson collection properties."""
    return _collection_properties(data_dir).get("data_source")


def load_provenance(data_dir: str | Path) -> dict | None:
    """Return the run ``provenance`` dict from flags.geojson, if recorded."""
    return _collection_properties(data_dir).get("provenance")


def _fmt_date(iso: str | None) -> str:
    from datetime import date
    try:
        d = date.fromisoformat(iso)
        return f"{d.day} {d:%b %Y}"
    except (TypeError, ValueError):
        return iso or "?"


def _data_banner_html(data_source: str | None, provenance: dict | None = None) -> str:
    if data_source == "synthetic":
        return (
            '<div class="data-banner" role="status"><strong>SYNTHETIC DATA</strong>'
            "Flags, displacement series and failure forecasts are generated for "
            "demonstration and do not represent observed ground motion.</div>"
        )
    if not provenance:
        return ""
    p = provenance
    product = " ".join(str(x) for x in (p.get("sensor"), p.get("product")) if x)
    track = ""
    if p.get("track") is not None:
        track = f"track {p['track']}"
        if p.get("direction"):
            track += f" {p['direction']}"
    parts = [x for x in (product, track) if x]
    if p.get("date_start") or p.get("date_end"):
        span = f"{_fmt_date(p.get('date_start'))} – {_fmt_date(p.get('date_end'))}"
        if p.get("n_pairs"):
            span += f" · {p['n_pairs']} pairs"
        parts.append(span)
    if p.get("posting_m"):
        parts.append(f"{p['posting_m']} m grid")
    if p.get("coherence_mask") is not None:
        parts.append(f"coherence mask {p['coherence_mask']}")
    summary = html_escape(" · ".join(parts))
    details = []
    if p.get("reference"):
        details.append("Referenced to " + html_escape(str(p["reference"])) + ".")
    if p.get("excluded"):
        details.append("Excluded: " + html_escape("; ".join(map(str, p["excluded"]))) + ".")
    detail_html = (' <span class="banner-detail">' + " ".join(details) + "</span>") if details else ""
    return (
        '<div class="data-banner data-banner-observed" role="status">'
        f"<strong>OBSERVED DATA</strong>{summary}.{detail_html}</div>"
    )


def _site_context(
    config: dict, provenance: dict | None, data_source: str | None = None
) -> dict:
    """Site facts the dashboard JS needs: known source, event date, snow season, extent.

    Synthetic scenes are not placed on the real source, so the known source
    is withheld for them rather than reporting meaningless distances.
    """
    site = config.get("site", {}) if config else {}
    ctx = {
        "event_date": site.get("event_date"),
        "known_source": None if data_source == "synthetic" else site.get("known_source"),
        "snow_months": site.get("snow_months") or [],
        "extent": (provenance or {}).get("extent"),
    }
    return _sanitize_for_json(ctx)


def _render_html(
    flags: list[dict], state: dict, site_name: str, data_source: str | None = None,
    provenance: dict | None = None, site: dict | None = None,
) -> str:
    """Render the dashboard HTML with flag data injected."""
    def _js(obj) -> str:
        # Keep "</script>" in any string from closing the inline script early
        return json.dumps(obj, default=str).replace("</", "<\\/")

    html = _HTML_TEMPLATE
    html = html.replace("$$FLAGS_JSON$$", _js(flags))
    html = html.replace("$$STATE_JSON$$", _js(state))
    html = html.replace("$$SITE_JSON$$", _js(site or {}))
    html = html.replace("$$SITE_NAME$$", html_escape(site_name))
    html = html.replace("$$DATA_BANNER$$", _data_banner_html(data_source, provenance))
    return html


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

def make_handler(
    flags: list[dict], state: dict, state_path: Path, site_name: str,
    data_source: str | None = None,
    provenance: dict | None = None,
    site: dict | None = None,
    place_names_done: threading.Event | None = None,
    webcam_registry: list[dict] | None = None,
):
    """
    Factory that returns a BaseHTTPRequestHandler subclass bound to
    the given flags and state.  Avoids module-level globals.

    ``place_names_done`` is set by the background place-name lookup;
    until then ``/api/place_names`` reports ``pending``.
    """
    state_lock = threading.Lock()
    data_dir = state_path.parent
    figures_root = (data_dir / "figures").resolve()
    flag_ids = {str(f.get("flag_id")) for f in flags}

    def _audit(flag_id: str, classification: str, note: str) -> None:
        try:
            from glews.provenance import AlertAuditLog
            AlertAuditLog(log_dir=data_dir / "audit").log_classification(
                site_name, flag_id, classification, note
            )
        except Exception as exc:  # audit must never block a classification
            logger.warning("Audit log write failed: %s", exc)

    class DashboardHandler(BaseHTTPRequestHandler):

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            params = parse_qs(parsed.query)

            if path in ("/", "/index.html"):
                html = _render_html(flags, state, site_name, data_source, provenance, site)
                self._respond(200, "text/html; charset=utf-8", html.encode("utf-8"))

            elif path.startswith("/figures/"):
                self._serve_figure(path[len("/figures/"):])

            elif path == "/api/place_names":
                names = {
                    str(f.get("flag_id")): {
                        "name": f.get("place_name", ""),
                        "name_en": f.get("place_name_en", ""),
                        "name_en_source": f.get("place_name_en_source", ""),
                    }
                    for f in flags if f.get("place_name")
                }
                pending = place_names_done is not None and not place_names_done.is_set()
                body = json.dumps({"names": names, "pending": pending})
                self._respond(200, "application/json", body.encode())

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

            elif path == "/api/webcams":
                self._handle_geo_api(params, lambda lat, lon: _cached_call(
                    f"cam:{lat:.3f},{lon:.3f}", _get_webcams, lat, lon,
                    tuple(webcam_registry or ()), ttl=WEBCAM_CACHE_TTL_S,
                ))

            elif path == "/api/news":
                region = params.get("region", [""])[0]
                self._handle_geo_api(params, lambda lat, lon: _cached_call(
                    f"news:{lat:.2f},{lon:.2f}:{region or 'default'}",
                    _get_news, lat, lon, region,
                    ttl=3600,
                ))

            else:
                self._respond(404, "text/plain", b"Not Found")

        def _serve_figure(self, rel: str):
            """Serve a PNG from <data_dir>/figures without leaving that folder."""
            target = (figures_root / urllib.parse.unquote(rel)).resolve()
            if (figures_root not in target.parents or target.suffix.lower() != ".png"
                    or not target.is_file()):
                self._respond(404, "text/plain", b"Not Found")
                return
            self._respond(200, "image/png", target.read_bytes())

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
                _audit(fid, classification, note)

                self._respond(
                    200, "application/json", json.dumps(entry).encode()
                )

            elif path == "/api/classify_bulk":
                length = int(self.headers.get("Content-Length", 0))
                try:
                    data = json.loads(self.rfile.read(length))
                except json.JSONDecodeError:
                    self._respond(400, "application/json",
                                  json.dumps({"error": "Invalid JSON"}).encode())
                    return
                classification = data.get("classification")
                reason = str(data.get("note", "")).strip()
                ids = [str(i) for i in data.get("flag_ids", [])]
                if classification not in VALID_CLASSIFICATIONS:
                    self._respond(400, "application/json",
                                  json.dumps({"error": "Invalid classification"}).encode())
                    return
                if not ids or not reason:
                    self._respond(400, "application/json", json.dumps(
                        {"error": "flag_ids and a reason (note) are required"}).encode())
                    return
                unknown = [i for i in ids if i not in flag_ids]
                if unknown:
                    self._respond(400, "application/json", json.dumps(
                        {"error": "Unknown flag ids: " + ", ".join(unknown)}).encode())
                    return
                stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                updated = {}
                with state_lock:
                    for fid in ids:
                        prev = state.get(fid, {})
                        note = reason
                        if prev.get("note") and reason not in prev["note"]:
                            note = prev["note"] + "\n" + reason
                        entry = dict(prev)
                        entry.update({"classification": classification, "note": note,
                                      "updated_at": stamp, "bulk": True})
                        state[fid] = entry
                        updated[fid] = entry
                    save_state(state_path, state)
                for fid in ids:
                    _audit(fid, classification, "[bulk] " + reason)
                self._respond(200, "application/json",
                              json.dumps({"updated": updated}).encode())

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
    event_date = config.get("site", {}).get("event_date")

    flags = load_flags(data_path)
    _reclassify_post_events(flags, event_date)
    state = load_state(state_path)

    data_source = load_data_source(data_path)
    provenance = load_provenance(data_path)
    site = _site_context(config, provenance, data_source)

    # Place names come from Overpass, which can take tens of seconds or fail.
    # Resolve them in the background so the page is served immediately; the
    # browser polls /api/place_names and fills them in when ready.
    place_names_done = threading.Event()
    try:  # names cached by an earlier run are applied before the first page load
        cached_names = json.loads((data_path / "place_names.json").read_text())
        for f in flags:
            entry = cached_names.get(str(f.get("flag_id")))
            if entry:
                _apply_place_name(f, entry)
    except (OSError, json.JSONDecodeError):
        pass

    def _resolve_names():
        try:
            _resolve_flag_place_names(flags, data_path)
        except Exception as exc:
            logger.info("Place-name lookup failed: %s", exc)
        finally:
            place_names_done.set()

    threading.Thread(target=_resolve_names, name="place-names", daemon=True).start()

    handler_cls = make_handler(
        flags, state, state_path, site_name, data_source,
        provenance=provenance, site=site, place_names_done=place_names_done,
        webcam_registry=load_webcam_registry(config.get("webcams_file")),
    )

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
