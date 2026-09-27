"""
Interactive GeoJSON map viewer for GEWS anomaly flags.

Serves a Leaflet.js-based map showing color-coded markers for each
flagged site, with popups containing anomaly details, severity-based
layer control, and auto-zoom to fit all markers.

Uses only Python's built-in http.server — no additional dependencies.

Usage (via CLI):
    glews map --data-dir output --port 8050
"""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GeoJSON loading (reuses the same flags.geojson produced by report.py)
# ---------------------------------------------------------------------------

def load_geojson(data_dir: str | Path) -> dict:
    """
    Load flags.geojson from the given directory.

    Returns the parsed GeoJSON dict, or an empty FeatureCollection
    when the file does not exist.
    """
    data_dir = Path(data_dir)
    geojson_path = data_dir / "flags.geojson"

    if geojson_path.exists():
        return json.loads(geojson_path.read_text())

    return {"type": "FeatureCollection", "features": []}


# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------

SEVERITY_CRITICAL_THRESHOLD = 4.0
SEVERITY_WARNING_THRESHOLD = 2.5


def _severity_for_feature(feature: dict) -> str:
    """Derive severity string for a GeoJSON feature."""
    props = feature.get("properties", {})
    risk = props.get("risk_level")
    if risk:
        return risk.upper()
    score = props.get("score", 0)
    if score >= SEVERITY_CRITICAL_THRESHOLD:
        return "CRITICAL"
    if score >= SEVERITY_WARNING_THRESHOLD:
        return "WARNING"
    return "INFO"


# ---------------------------------------------------------------------------
# HTML page
# ---------------------------------------------------------------------------

MAP_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GEWS Map Viewer</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
/* Leaflet CSS — inlined from leaflet 1.9.4 */
.leaflet-pane,.leaflet-tile,.leaflet-marker-icon,.leaflet-marker-shadow,
.leaflet-tile-container,.leaflet-pane>svg,.leaflet-pane>canvas,
.leaflet-zoom-box,.leaflet-image-layer,.leaflet-layer{position:absolute;left:0;top:0}
.leaflet-container{overflow:hidden}
.leaflet-tile,.leaflet-marker-icon,.leaflet-marker-shadow{-webkit-user-select:none;
-moz-user-select:none;user-select:none;-webkit-user-drag:none}
.leaflet-tile::selection{background:transparent}
.leaflet-safari .leaflet-tile{image-rendering:-webkit-optimize-contrast}
.leaflet-safari .leaflet-tile-container{width:1600px;height:1600px;
-webkit-transform-origin:0 0}
.leaflet-marker-icon,.leaflet-marker-shadow{display:block}
.leaflet-container .leaflet-overlay-pane svg{max-width:none!important;
max-height:none!important}
.leaflet-container .leaflet-marker-pane img,
.leaflet-container .leaflet-shadow-pane img,
.leaflet-container .leaflet-tile-pane img,
.leaflet-container img.leaflet-image-layer,
.leaflet-container .leaflet-tile{max-width:none!important;
max-height:none!important;width:auto;padding:0}
.leaflet-container.leaflet-touch-zoom{-ms-touch-action:pan-x pan-y;
touch-action:pan-x pan-y}
.leaflet-container.leaflet-touch-drag{-ms-touch-action:pinch-zoom;
touch-action:none;touch-action:pinch-zoom}
.leaflet-container.leaflet-touch-drag.leaflet-touch-zoom{
-ms-touch-action:none;touch-action:none}
.leaflet-container{-webkit-tap-highlight-color:transparent}
.leaflet-container a{-webkit-tap-highlight-color:rgba(51,181,229,.4)}
.leaflet-tile{filter:inherit;visibility:hidden}
.leaflet-tile-loaded{visibility:inherit}
.leaflet-zoom-box{width:0;height:0;-moz-box-sizing:border-box;
box-sizing:border-box;z-index:800}
.leaflet-overlay-pane svg{-moz-user-select:none}
.leaflet-pane{z-index:400}
.leaflet-tile-pane{z-index:200}
.leaflet-overlay-pane{z-index:400}
.leaflet-shadow-pane{z-index:500}
.leaflet-marker-pane{z-index:600}
.leaflet-tooltip-pane{z-index:650}
.leaflet-popup-pane{z-index:700}
.leaflet-map-pane canvas{z-index:100}
.leaflet-map-pane svg{z-index:200}
.leaflet-vml-shape{width:1px;height:1px}
.lvml{behavior:url(#default#VML);display:inline-block;position:absolute}
.leaflet-control{position:relative;z-index:800;pointer-events:visiblePainted;
pointer-events:auto}
.leaflet-top,.leaflet-bottom{position:absolute;z-index:1000;pointer-events:none}
.leaflet-top{top:0}.leaflet-right{right:0}
.leaflet-bottom{bottom:0}.leaflet-left{left:0}
.leaflet-control{float:left;clear:both}
.leaflet-right .leaflet-control{float:right}
.leaflet-top .leaflet-control{margin-top:10px}
.leaflet-bottom .leaflet-control{margin-bottom:10px}
.leaflet-left .leaflet-control{margin-left:10px}
.leaflet-right .leaflet-control{margin-right:10px}
.leaflet-fade-anim .leaflet-popup{opacity:1;
-webkit-transition:opacity .2s linear;-moz-transition:opacity .2s linear;transition:opacity .2s linear}
.leaflet-fade-anim .leaflet-map-pane .leaflet-popup{opacity:0}
.leaflet-zoom-animated{-webkit-transform-origin:0 0;-ms-transform-origin:0 0;
transform-origin:0 0}
.leaflet-zoom-anim .leaflet-zoom-animated{will-change:transform;
-webkit-transition:-webkit-transform .25s cubic-bezier(0,0,.25,1);
-moz-transition:-moz-transform .25s cubic-bezier(0,0,.25,1);
transition:transform .25s cubic-bezier(0,0,.25,1)}
.leaflet-zoom-anim .leaflet-tile,.leaflet-pan-anim .leaflet-tile{
-webkit-transition:none;-moz-transition:none;transition:none}
.leaflet-zoom-anim .leaflet-zoom-hide{visibility:hidden}
.leaflet-interactive{cursor:pointer}
.leaflet-grab{cursor:-webkit-grab;cursor:-moz-grab;cursor:grab}
.leaflet-crosshair,.leaflet-crosshair .leaflet-interactive{cursor:crosshair}
.leaflet-popup-pane,.leaflet-control{cursor:auto}
.leaflet-dragging .leaflet-grab,.leaflet-dragging .leaflet-grab .leaflet-interactive,
.leaflet-dragging .leaflet-marker-draggable{cursor:move;cursor:-webkit-grabbing;
cursor:-moz-grabbing;cursor:grabbing}
.leaflet-marker-icon,.leaflet-marker-shadow,.leaflet-image-layer,
.leaflet-pane>svg path,.leaflet-tile-container{pointer-events:none}
.leaflet-marker-icon.leaflet-interactive,.leaflet-image-layer.leaflet-interactive,
.leaflet-pane>svg path.leaflet-interactive,svg.leaflet-image-layer.leaflet-interactive path{
pointer-events:visiblePainted;pointer-events:auto}
.leaflet-container{background:#ddd;outline-offset:1px}
.leaflet-container a{color:#0078A8}
.leaflet-zoom-box{border:2px dotted #38f;background:rgba(255,255,255,.5)}
.leaflet-container{font-family:"Helvetica Neue",Arial,Helvetica,sans-serif;
font-size:12px;font-size:.75rem;line-height:1.5}
.leaflet-bar{box-shadow:0 1px 5px rgba(0,0,0,.65);border-radius:4px}
.leaflet-bar a,.leaflet-bar a:hover{background-color:#fff;border-bottom:1px solid #ccc;
width:26px;height:26px;line-height:26px;display:block;text-align:center;
text-decoration:none;color:#000}
.leaflet-bar a,.leaflet-control-layers-toggle{background-position:50% 50%;
background-repeat:no-repeat;display:block}
.leaflet-bar a:hover,.leaflet-bar a:focus{background-color:#f4f4f4}
.leaflet-bar a:first-child{border-top-left-radius:4px;border-top-right-radius:4px}
.leaflet-bar a:last-child{border-bottom-left-radius:4px;
border-bottom-right-radius:4px;border-bottom:none}
.leaflet-bar a.leaflet-disabled{cursor:default;background-color:#f4f4f4;color:#bbb}
.leaflet-touch .leaflet-bar a{width:30px;height:30px;line-height:30px}
.leaflet-touch .leaflet-bar a:first-child{border-top-left-radius:2px;
border-top-right-radius:2px}
.leaflet-touch .leaflet-bar a:last-child{border-bottom-left-radius:2px;
border-bottom-right-radius:2px}
.leaflet-control-zoom-in,.leaflet-control-zoom-out{font:bold 18px 'Lucida Console',
Monaco,monospace;text-indent:1px}
.leaflet-touch .leaflet-control-zoom-in,.leaflet-touch .leaflet-control-zoom-out{
font-size:22px}
.leaflet-control-layers{box-shadow:0 1px 5px rgba(0,0,0,.4);background:#fff;
border-radius:5px}
.leaflet-control-layers-toggle{background-image:url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABoAAAAaCAQAAAADQ4RFAAACf0lEQVR4AY1TA7RlQQyz03444/+f8bZt27NlO5N29u2M0eo2C/QLfQklCKw6AKQsEOAAB7jeEYBqTgBYsQJNy+LACjYBYIklAXDlhR53AUCIZSkQcHvDmIeBABFQJBEKBBy8rcMdWwGA2LgSEHB714KfYwBQxbYIEHB776JfYwFQxa4OcJ9h3+P48IARQBX7FoA7O+4+sAIIovJlvwJwX9dBRwAJosp1CwCEBL3mCABRZYcIgMCnb+Jj9gCgit3LQR73MvT7wYQUHlfBri366CjZ0BQxf4lCOz0TN93kQAiqt4LATh05+i8bwEHkwq6F0PAXz3r7iwE1FTQ0TiCYMiPMR4mCEhFBR2No4gGfh/90YiAKooKOhtH0Rj0KcZTNAKqqKCjcRSNQS8ynqIRUEVR2d44ggD74T6MJ4CAqorK9sa/EIf+ivo4kUBVRWX78F/E8X6J8ThBQFVFZXvjX8Thfoz1BEFAVUVle+MoGsy/Yj1BI6CKovJ/LY5i4BcBz9AIqKKo/F+LI+j3Cfcp2g5VFJX/63EEAz6O/TBBQBVFZf96HEHQZ37cSPB2ouqTv0wCJIkp3xMYBJjC5X/6FuNxAQC33X39bwEAMNc8CXDAD/kpJwEArHAvAeCufoB+YQkAVbkYAIAfcFQ/NQeAOlwOgHv6PkfhMwGgjHMCcGfPYcfeAQDU4VwAnF38en6MBYA8jswC3NX1x+F3AgAYcWQM4O6uA4+cAoBqNi8F5O4Ffw09BwDlnBiABzdNDz0HABBz4h0AT6zrO34TANBwfDfggW6iIy4D6nheBphqC8CVpwC2ORUwzU0C6Hg6QH0rDADK+z//AaL9JX3PQvLhAAAAAElFTkSuQmCC);
width:36px;height:36px}
.leaflet-retina .leaflet-control-layers-toggle{background-image:url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADQAAAA0CAQAAABvcdNgAAAEsklEQVR4AWL4TydIhpZK1kAAAABJRU5ErkJggg==);
background-size:26px 26px}
.leaflet-touch .leaflet-control-layers-toggle{width:44px;height:44px}
.leaflet-control-layers .leaflet-control-layers-list,
.leaflet-control-layers-expanded .leaflet-control-layers-toggle{display:none}
.leaflet-control-layers-expanded .leaflet-control-layers-list{display:block;
position:relative}
.leaflet-control-layers-expanded{padding:6px 10px 6px 6px;color:#333;background:#fff}
.leaflet-control-layers-scrollbar{overflow-y:scroll;padding-right:5px}
.leaflet-control-layers-selector{margin-top:2px;position:relative;top:1px}
.leaflet-control-layers label{display:block;font-size:13px;font-size:.8125rem}
.leaflet-control-layers-separator{height:0;border-top:1px solid #ddd;margin:5px -10px 5px -6px}
.leaflet-default-icon-path{background-image:url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAApCAYAAADAk4LOAAAFgUlEQVR4Aa1XA5BjWRTN2oW17d3YassZLmB2AQFBBQV0UiC0BAmigRIJELGBIIFSSIIPahqEBgXBB1HIR0KGDBS/d/fqaqa6vt/Mfda55165d0/5n3LOec/5Nq + zzs5OFnJCVjbcWPdIr + TItkT+OssYtTqDgIGxABGPYJGCjFJI7qhEBqafnmgSohDrMIqFCZhlmkO1aB8UL7v3X3loeBBgA7BGkAwX7ngJxHEI4cHF7zS8MPl0F5AwAgIjSrR2LHuKGrp2JJ3hhLCBmF + Fb2D0Td4BXmI0CroYFOO3eJ0kZ3RfVKdTjj/lCVSVPWrGIIBAgDWEGqxEH7rNWUhLyIQMBTQZ4CYaE0VwqhMLbEVAqq0MYJnhAGDAIhjcYB + ay9Bwg4cLK4xJ/mK7AAGCATuzIxwvEU/rM1Ikf7FYH/FCYR6+UPC0bVlPjGo0sCnPnhDrLi7Wd + 7i + Oaf9Tw0EgxFhEJypBIENH3HlMnNHUOLcDxKw0DGKJWQ5cI3HiQ2X8DNjf0IbQe1C/nFOWmJDf + FRW3PecoB6BgAEHh0J2DePA6c6Ake/2JMy75E8IWDiAQFxBNZSYn7fDMRYFYSCBw1mgHkgU/y2kGtyb6GZ6FKR9Mp6nLaFg6oLqjS0OLwS5cLjIz/zMJviHCsyIqPNmq1y9v40exkMu0DfR7d97IbcCM0jvjx1/FcmNWBhYKAA/6rJLTHZev1vdBnlr1G8l0K6OP9AJhRPNNfKG0hBBNbY4NjGMmz+1BzZKjLOfKNv3O0Y/CgILNJC8gFnE8HWADzLiMQdrOHv0bWm2c6lSIVFIhVFCFz8RpJN2S6HjuKcA/WFLtIIkza9x4frbcvLf7fpGqOIcvNJ+PYbT4OKQGP0BRFSbFrKL6MHG5Cdl/NSvOqJPUldI + avrHSCV0jU + OfQP2b3VGa79pZ5+FDXM3FPwMuV6g0YCAA);
}
.leaflet-container .leaflet-control-attribution{background:#fff;background:rgba(255,255,255,.8);
margin:0}
.leaflet-control-attribution,.leaflet-control-scale-line{padding:0 5px;color:#333;
line-height:1.4}
.leaflet-control-attribution a{text-decoration:none}
.leaflet-control-attribution a:hover,.leaflet-control-attribution a:focus{
text-decoration:underline}
.leaflet-attribution-flag{display:inline!important;vertical-align:baseline!important;
width:1em;height:.6669em}
.leaflet-left .leaflet-control-scale{margin-left:5px}
.leaflet-bottom .leaflet-control-scale{margin-bottom:5px}
.leaflet-control-scale-line{border:2px solid #777;border-top:none;
line-height:1.1;padding:2px 5px 1px;white-space:nowrap;-moz-box-sizing:border-box;
box-sizing:border-box;background:rgba(255,255,255,.5);text-shadow:1px 1px #fff}
.leaflet-control-scale-line:not(:first-child){border-top:2px solid #777;
border-bottom:none;margin-top:-2px}
.leaflet-control-scale-line:not(:first-child):not(:last-child){border-bottom:2px solid #777}
.leaflet-touch .leaflet-control-attribution,.leaflet-touch .leaflet-control-layers,
.leaflet-touch .leaflet-bar{box-shadow:none}
.leaflet-touch .leaflet-control-layers,.leaflet-touch .leaflet-bar{border:2px solid rgba(0,0,0,.2);
background-clip:padding-box}
.leaflet-popup{position:absolute;text-align:center;margin-bottom:20px}
.leaflet-popup-content-wrapper{padding:1px;text-align:left;border-radius:12px}
.leaflet-popup-content{margin:13px 24px 13px 20px;line-height:1.3;
font-size:13px;font-size:.8125rem;min-height:1px}
.leaflet-popup-content p{margin:17px 0;margin:1.3em 0}
.leaflet-popup-tip-container{width:40px;height:20px;position:absolute;
left:50%;margin-top:-1px;margin-left:-20px;overflow:hidden;pointer-events:none}
.leaflet-popup-tip{width:17px;height:17px;padding:1px;margin:-10px auto 0;
pointer-events:auto;-webkit-transform:rotate(45deg);-moz-transform:rotate(45deg);
-ms-transform:rotate(45deg);transform:rotate(45deg)}
.leaflet-popup-content-wrapper,.leaflet-popup-tip{background:white;color:#333;
box-shadow:0 3px 14px rgba(0,0,0,.4)}
.leaflet-container a.leaflet-popup-close-button{position:absolute;top:0;right:0;
border:none;text-align:center;width:24px;height:24px;font:16px/24px Tahoma,Verdana,
sans-serif;color:#757575;text-decoration:none;background:transparent}
.leaflet-container a.leaflet-popup-close-button:hover,.leaflet-container a.leaflet-popup-close-button:focus{
color:#585858}
.leaflet-popup-scrolled{overflow:auto}
.leaflet-oldie .leaflet-popup-content-wrapper{-ms-zoom:1}
.leaflet-oldie .leaflet-popup-tip{width:24px;margin:0 auto;
-ms-filter:"progid:DXImageTransform.Microsoft.Matrix(M11=0.70710678,M12=0.70710678,M21=-0.70710678,M22=0.70710678)";
filter:progid:DXImageTransform.Microsoft.Matrix(M11=0.70710678,M12=0.70710678,M21=-0.70710678,M22=0.70710678)}
.leaflet-oldie .leaflet-control-zoom,.leaflet-oldie .leaflet-control-layers,
.leaflet-oldie .leaflet-popup-content-wrapper,.leaflet-oldie .leaflet-popup-tip{
border:1px solid #999}
.leaflet-div-icon{background:#fff;border:1px solid #666}
.leaflet-tooltip{position:absolute;padding:6px;background-color:#fff;
border:1px solid #fff;border-radius:3px;color:#222;white-space:nowrap;
-webkit-user-select:none;-moz-user-select:none;-ms-user-select:none;
user-select:none;pointer-events:none;box-shadow:0 1px 3px rgba(0,0,0,.4)}
.leaflet-tooltip.leaflet-interactive{cursor:pointer;pointer-events:auto}
.leaflet-tooltip-top:before,.leaflet-tooltip-bottom:before,
.leaflet-tooltip-left:before,.leaflet-tooltip-right:before{
position:absolute;pointer-events:none;border:6px solid transparent;
background:transparent;content:""}
.leaflet-tooltip-bottom{margin-top:6px}
.leaflet-tooltip-top{margin-top:-6px}
.leaflet-tooltip-bottom:before,.leaflet-tooltip-top:before{
left:50%;margin-left:-6px}
.leaflet-tooltip-top:before{bottom:0;margin-bottom:-12px;
border-top-color:#fff}
.leaflet-tooltip-bottom:before{top:0;margin-top:-12px;margin-left:-6px;
border-bottom-color:#fff}
.leaflet-tooltip-left{margin-left:-6px}
.leaflet-tooltip-right{margin-left:6px}
.leaflet-tooltip-left:before{right:0;margin-right:-12px;
border-left-color:#fff}
.leaflet-tooltip-right:before{left:0;margin-left:-12px;
border-right-color:#fff}

/* Page styles */
html, body { margin: 0; padding: 0; height: 100%; font-family: sans-serif; }
#map { width: 100%; height: 100%; }
.legend {
    background: white; padding: 10px 14px; border-radius: 5px;
    box-shadow: 0 1px 5px rgba(0,0,0,.4); line-height: 1.8;
    font-size: 13px;
}
.legend i {
    width: 14px; height: 14px; display: inline-block;
    margin-right: 6px; border-radius: 50%; vertical-align: middle;
}
.legend-title { font-weight: bold; margin-bottom: 4px; }
</style>
</head>
<body>
<div id="map"></div>
<script>
(function () {
    var COLORS = { CRITICAL: '#e53935', WARNING: '#fb8c00', INFO: '#1e88e5' };

    var map = L.map('map').setView([28.2, 86.0], 8);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(map);

    // Layer groups by severity
    var layers = {
        CRITICAL: L.layerGroup().addTo(map),
        WARNING: L.layerGroup().addTo(map),
        INFO: L.layerGroup().addTo(map)
    };

    fetch('/api/flags.geojson')
        .then(function (r) { return r.json(); })
        .then(function (geojson) {
            var bounds = [];

            (geojson.features || []).forEach(function (f) {
                var coords = f.geometry.coordinates;
                var lng = coords[0], lat = coords[1];
                var p = f.properties || {};

                // Determine severity
                var severity = 'INFO';
                if (p.risk_level) {
                    severity = p.risk_level.toUpperCase();
                } else {
                    var score = p.score || 0;
                    if (score >= 4.0) severity = 'CRITICAL';
                    else if (score >= 2.5) severity = 'WARNING';
                }
                if (!COLORS[severity]) severity = 'INFO';

                var color = COLORS[severity];

                var marker = L.circleMarker([lat, lng], {
                    radius: 8,
                    fillColor: color,
                    color: '#333',
                    weight: 1,
                    opacity: 1,
                    fillOpacity: 0.85
                });

                // Popup content
                var html = '<div style="min-width:200px">';
                html += '<strong>Flag ' + (p.flag_id != null ? p.flag_id : '?') + '</strong>';
                html += ' <span style="color:' + color + ';font-weight:bold">[' + severity + ']</span><br>';
                html += '<b>Score:</b> ' + (p.score != null ? p.score : 'N/A') + '<br>';
                html += '<b>Peak Z-score:</b> ' + (p.peak_zscore != null ? p.peak_zscore : 'N/A') + '<br>';
                html += '<b>Mean Z-score:</b> ' + (p.mean_zscore != null ? p.mean_zscore : 'N/A') + '<br>';
                html += '<b>Pixels:</b> ' + (p.n_pixels != null ? p.n_pixels : 'N/A') + '<br>';
                html += '<b>Area:</b> ' + (p.area_m2 != null ? Number(p.area_m2).toLocaleString() + ' m&sup2;' : 'N/A') + '<br>';
                html += '<b>Acceleration:</b> ' + (p.acceleration_mm_yr2 != null ? p.acceleration_mm_yr2 + ' mm/yr&sup2;' : 'N/A') + '<br>';
                html += '<b>Location:</b> ' + lat.toFixed(4) + '&deg;N, ' + lng.toFixed(4) + '&deg;E<br>';
                if (p.voight_r2 != null) {
                    html += '<b>Voight R&sup2;:</b> ' + p.voight_r2 + '<br>';
                }
                if (p.voight_days_to_failure != null) {
                    html += '<b>Days to failure:</b> ' + p.voight_days_to_failure + '<br>';
                }
                if (p.volume_m3 != null) {
                    html += '<b>Volume:</b> ' + Number(p.volume_m3).toLocaleString() + ' m&sup3;<br>';
                }
                if (p.population_exposed != null) {
                    html += '<b>Pop. exposed:</b> ' + Number(p.population_exposed).toLocaleString() + '<br>';
                }
                html += '</div>';

                marker.bindPopup(html);
                marker.addTo(layers[severity]);
                bounds.push([lat, lng]);
            });

            // Auto-zoom to fit all markers
            if (bounds.length > 0) {
                map.fitBounds(bounds, { padding: [40, 40] });
            }
        })
        .catch(function (err) {
            console.error('Failed to load flags:', err);
        });

    // Layer control
    L.control.layers(null, {
        '<span style="color:#e53935">&#9679;</span> Critical': layers.CRITICAL,
        '<span style="color:#fb8c00">&#9679;</span> Warning': layers.WARNING,
        '<span style="color:#1e88e5">&#9679;</span> Info': layers.INFO
    }, { collapsed: false }).addTo(map);

    // Legend
    var legend = L.control({ position: 'bottomright' });
    legend.onAdd = function () {
        var div = L.DomUtil.create('div', 'legend');
        div.innerHTML =
            '<div class="legend-title">Severity</div>' +
            '<i style="background:#e53935"></i> Critical (score &ge; 4.0)<br>' +
            '<i style="background:#fb8c00"></i> Warning (score &ge; 2.5)<br>' +
            '<i style="background:#1e88e5"></i> Info (score &lt; 2.5)';
        return div;
    };
    legend.addTo(map);
})();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# HTTP handler and server
# ---------------------------------------------------------------------------

def make_handler(data_dir: str | Path):
    """
    Create a request handler class bound to the given data directory.

    Returns a BaseHTTPRequestHandler subclass.
    """
    data_dir = Path(data_dir)

    class MapHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/" or self.path == "":
                self._serve_html()
            elif self.path == "/api/flags.geojson":
                self._serve_geojson()
            else:
                self.send_error(404, "Not Found")

        def _serve_html(self):
            body = MAP_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_geojson(self):
            geojson = load_geojson(data_dir)
            body = json.dumps(geojson).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/geo+json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            logger.info(format, *args)

    return MapHandler


def serve(data_dir: str | Path = "output", port: int = 8050) -> None:
    """
    Start the map viewer HTTP server.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing flags.geojson.
    port : int
        Port to listen on.
    """
    data_dir = Path(data_dir)
    handler_cls = make_handler(data_dir)
    server = HTTPServer(("", port), handler_cls)
    logger.info("Map viewer serving at http://localhost:%d", port)
    logger.info("Data directory: %s", data_dir)
    print(f"GEWS Map Viewer running at http://localhost:{port}")
    print(f"Data directory: {data_dir}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
