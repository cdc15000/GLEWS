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
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Severity thresholds — derived from anomaly score when no risk_level
# is present in the GeoJSON properties (i.e. no Tier 1 assessment ran).
# ---------------------------------------------------------------------------
SEVERITY_CRITICAL_THRESHOLD = 4.0
SEVERITY_WARNING_THRESHOLD = 2.5

# Analyst classification choices (distinct from severity)
VALID_CLASSIFICATIONS = {"Watch", "Warning", "Cleared"}


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
</style>
</head>
<body>
<div class="sidebar">
    <div class="sidebar-header">
        <h1>GLEWS Dashboard</h1>
        <div class="site-name" id="siteName">$$SITE_NAME$$</div>
    </div>
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
    html += '<h3>Flag ' + f.flag_id + ' — ' + formatCoord(f.center_lat, f.center_lon);
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
    document.getElementById('mainTitle').textContent = 'Flag ' + flagId + ' — Detail View';
    initMap(f);
}

var dashMap = null;
function initMap(flag) {
    if (flag.center_lat == null || flag.center_lon == null) return;
    if (dashMap) { dashMap.remove(); dashMap = null; }
    dashMap = L.map('flagMap').setView([flag.center_lat, flag.center_lon], 13);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors',
        maxZoom: 18
    }).addTo(dashMap);
    var sevColors = {CRITICAL: '#d63031', WARNING: '#fdcb6e', INFO: '#74b9ff'};
    // Show all flags as small circles
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

function escapeHtml(s) {
    var div = document.createElement('div');
    div.appendChild(document.createTextNode(s));
    return div.innerHTML;
}

function metric(label, value, unit) {
    return '<div class="metric"><div class="label">' + label + '</div><div class="value">' + value + '<span class="unit">' + unit + '</span></div></div>';
}

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
    // Axes
    svg += '<line x1="' + pad + '" y1="' + (h + pad) + '" x2="' + (w + pad) + '" y2="' + (h + pad) + '" stroke="#636e72" stroke-width="1"/>';
    svg += '<line x1="' + pad + '" y1="' + pad + '" x2="' + pad + '" y2="' + (h + pad) + '" stroke="#636e72" stroke-width="1"/>';
    // Y-axis labels
    for (var y = 0; y <= 4; y++) {
        var yv = vmin + (vmax - vmin) * y / 4;
        var yp = h + pad - (y / 4) * h;
        var range = vmax - vmin;
        var dp = range < 0.1 ? 4 : range < 1 ? 3 : 2;
        svg += '<text x="' + (pad - 5) + '" y="' + yp + '" text-anchor="end" font-size="10" fill="#636e72">' + yv.toFixed(dp) + '</text>';
        svg += '<line x1="' + pad + '" y1="' + yp + '" x2="' + (w + pad) + '" y2="' + yp + '" stroke="#dfe6e9" stroke-width="0.5"/>';
    }
    // Data line
    var points = '';
    for (var i = 0; i < n; i++) {
        var x = pad + (i / (n - 1)) * w;
        var yVal = h + pad - ((values[i] - vmin) / (vmax - vmin)) * h;
        points += x + ',' + yVal + ' ';
    }
    svg += '<polyline points="' + points.trim() + '" fill="none" stroke="#0984e3" stroke-width="2"/>';
    // Data points
    for (var i = 0; i < n; i++) {
        var x = pad + (i / (n - 1)) * w;
        var yVal = h + pad - ((values[i] - vmin) / (vmax - vmin)) * h;
        svg += '<circle cx="' + x + '" cy="' + yVal + '" r="3" fill="#0984e3"/>';
    }
    // X-axis labels (first, middle, last)
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

    class DashboardHandler(BaseHTTPRequestHandler):

        def do_GET(self):
            if self.path == "/" or self.path == "/index.html":
                html = _render_html(flags, state, site_name)
                self._respond(200, "text/html; charset=utf-8", html.encode("utf-8"))
            elif self.path == "/api/flags":
                merged = _merge_flags_state(flags, state)
                body = json.dumps(merged)
                self._respond(200, "application/json", body.encode())
            else:
                self._respond(404, "text/plain", b"Not Found")

        def do_POST(self):
            if self.path == "/api/classify":
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
                state[fid] = entry
                save_state(state_path, state)

                self._respond(
                    200, "application/json", json.dumps(entry).encode()
                )
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

    handler_cls = make_handler(flags, state, state_path, site_name)
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
