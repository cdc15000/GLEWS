"""
Configuration validation for GEWS.

Validates site configuration files against expected structure, types,
value ranges, and cross-field consistency rules.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("glews")

# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------

# Required fields as dot-separated paths with expected types
REQUIRED_FIELDS: dict[str, type | tuple[type, ...]] = {
    "site.name": str,
    "site.latitude": (int, float),
    "site.longitude": (int, float),
}

# Aliases: some configs use shorthand keys
_FIELD_ALIASES: dict[str, str] = {
    "site.lat": "site.latitude",
    "site.lon": "site.longitude",
}

# Known top-level sections
KNOWN_SECTIONS: set[str] = {
    "site",
    "acquire",
    "process",
    "detect",
    "cascade",
    "monitor",
    "alerts",
    "report",
    "sites",
}

# Numeric fields that must be positive
POSITIVE_FIELDS: list[str] = [
    "site.buffer_km",
    "detect.acceleration.sigma_threshold",
    "detect.acceleration.window_size_days",
    "detect.acceleration.step_days",
    "detect.clustering.min_cluster_pixels",
    "detect.clustering.max_distance_m",
    "detect.clustering.min_area_m2",
    "detect.step_change.sigma_threshold",
    "detect.step_change.min_displacement_m",
    "detect.voight.min_points",
    "cascade.min_volume_m3",
    "cascade.exposure.max_runout_km",
    "cascade.valley_width_threshold_m",
    "monitor.interval_hours",
    "monitor.lookback_days",
    "acquire.n_workers",
    "acquire.max_scenes",
]

# Fields that must be in [0, 1]
UNIT_RANGE_FIELDS: list[str] = [
    "detect.voight.r_squared_threshold",
    "process.unwrapping.coherence_threshold",
    "process.mintpy.min_temporal_coherence",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MissingSentinel:
    """Sentinel for missing config values."""

    def __repr__(self) -> str:
        return "<MISSING>"


_MISSING = _MissingSentinel()


def _get_nested(config: dict, dotpath: str) -> Any:
    """Retrieve a value from a nested dict using a dot-separated path.

    Returns _MISSING if any key along the path is absent.
    """
    keys = dotpath.split(".")
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return _MISSING
        current = current[key]
    return current


def _resolve_aliases(config: dict) -> None:
    """Resolve known field aliases in-place (e.g. site.lat -> site.latitude)."""
    for alias, canonical in _FIELD_ALIASES.items():
        alias_val = _get_nested(config, alias)
        canonical_val = _get_nested(config, canonical)
        if not isinstance(alias_val, _MissingSentinel) and isinstance(
            canonical_val, _MissingSentinel
        ):
            # Set the canonical path from the alias
            parts = canonical.split(".")
            target = config
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = alias_val


def _type_name(t: type | tuple[type, ...]) -> str:
    if isinstance(t, tuple):
        return " or ".join(cls.__name__ for cls in t)
    return t.__name__


def _deep_copy_dict(d: dict) -> dict:
    """Shallow-ish copy sufficient for one-level merge."""
    return {k: (dict(v) if isinstance(v, dict) else v) for k, v in d.items()}


# ---------------------------------------------------------------------------
# Core validation
# ---------------------------------------------------------------------------


def validate_config(config: dict) -> list[str]:
    """Validate a GEWS configuration dict.

    Returns a list of diagnostic strings.  Each starts with ``ERROR:``
    or ``WARNING:`` so callers can distinguish blocking issues from
    non-fatal advisories.

    An empty list means the configuration passed all checks.
    """
    issues: list[str] = []

    if not isinstance(config, dict):
        return [
            "ERROR: Config must be a YAML mapping (dict), got "
            + type(config).__name__
        ]

    # Multi-site configs: validate each site entry separately
    if "sites" in config:
        _validate_multi_site(config, issues)
        return issues

    # Resolve aliases before checking
    _resolve_aliases(config)

    # 1. Required fields
    for dotpath, expected_type in REQUIRED_FIELDS.items():
        val = _get_nested(config, dotpath)
        if isinstance(val, _MissingSentinel):
            issues.append(f"ERROR: Missing required field '{dotpath}'")
        elif not isinstance(val, expected_type):
            issues.append(
                f"ERROR: Field '{dotpath}' must be {_type_name(expected_type)}, "
                f"got {type(val).__name__}"
            )

    # 2. Latitude / longitude ranges
    lat = _get_nested(config, "site.latitude")
    if isinstance(lat, (int, float)) and not (-90 <= lat <= 90):
        issues.append(f"ERROR: site.latitude={lat} out of range [-90, 90]")

    lon = _get_nested(config, "site.longitude")
    if isinstance(lon, (int, float)) and not (-180 <= lon <= 180):
        issues.append(f"ERROR: site.longitude={lon} out of range [-180, 180]")

    # 3. Positive-value fields
    for dotpath in POSITIVE_FIELDS:
        val = _get_nested(config, dotpath)
        if isinstance(val, _MissingSentinel):
            continue
        if not isinstance(val, (int, float)):
            issues.append(
                f"ERROR: Field '{dotpath}' must be numeric, got {type(val).__name__}"
            )
        elif val <= 0:
            issues.append(f"ERROR: Field '{dotpath}' must be > 0, got {val}")

    # 4. Unit-range fields
    for dotpath in UNIT_RANGE_FIELDS:
        val = _get_nested(config, dotpath)
        if isinstance(val, _MissingSentinel):
            continue
        if isinstance(val, (int, float)) and not (0 <= val <= 1):
            issues.append(
                f"ERROR: Field '{dotpath}' must be in [0, 1], got {val}"
            )

    # 5. Cross-field consistency: start_date < end_date
    start = _get_nested(config, "acquire.start_date")
    end = _get_nested(config, "acquire.end_date")
    if not isinstance(start, _MissingSentinel) and not isinstance(
        end, _MissingSentinel
    ):
        try:
            if str(start) >= str(end):
                issues.append(
                    f"ERROR: acquire.start_date ({start}) must be before "
                    f"acquire.end_date ({end})"
                )
        except TypeError:
            pass

    # 6. Unknown top-level keys
    for key in config:
        if key not in KNOWN_SECTIONS:
            issues.append(f"WARNING: Unknown top-level key '{key}'")

    return issues


def _validate_multi_site(config: dict, issues: list[str]) -> None:
    """Validate a multi-site configuration."""
    sites = config.get("sites")
    if not isinstance(sites, list):
        issues.append("ERROR: 'sites' must be a list")
        return
    if len(sites) == 0:
        issues.append("ERROR: 'sites' list is empty")
        return

    for idx, entry in enumerate(sites):
        if not isinstance(entry, dict):
            issues.append(f"ERROR: sites[{idx}] must be a mapping")
            continue
        if "site" not in entry:
            issues.append(f"ERROR: sites[{idx}] missing required 'site' section")
            continue

        # Build a merged config for this site: shared defaults + per-site overrides
        merged: dict[str, Any] = {}
        for section in KNOWN_SECTIONS - {"sites"}:
            if section in config:
                merged[section] = _deep_copy_dict(config[section])
        for key, val in entry.items():
            if (
                isinstance(val, dict)
                and key in merged
                and isinstance(merged[key], dict)
            ):
                merged[key] = {**merged[key], **val}
            else:
                merged[key] = val

        site_issues = validate_config(merged)
        site_name = entry.get("site", {}).get("name", f"index {idx}")
        for issue in site_issues:
            issues.append(f"{issue} [site: {site_name}]")


# ---------------------------------------------------------------------------
# File-level validation
# ---------------------------------------------------------------------------


def validate_config_file(path: str) -> tuple[dict, list[str]]:
    """Load a YAML config file and validate it.

    Returns ``(config_dict, issues)`` where *issues* is the same list
    produced by :func:`validate_config`.  If the file cannot be read or
    parsed, *config_dict* will be an empty dict and *issues* will
    contain an ``ERROR:`` entry.
    """
    filepath = Path(path)
    if not filepath.exists():
        return {}, [f"ERROR: Config file not found: {path}"]

    try:
        with open(filepath) as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        return {}, [f"ERROR: Invalid YAML: {exc}"]

    if config is None:
        return {}, ["ERROR: Config file is empty"]

    if not isinstance(config, dict):
        return config, [
            "ERROR: Config must be a YAML mapping (dict), got "
            + type(config).__name__
        ]

    issues = validate_config(config)
    return config, issues
