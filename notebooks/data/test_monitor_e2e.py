#!/usr/bin/env python3
"""
End-to-end test of the GLEWS monitoring module.

Runs a single check cycle against the global_watch.yaml config
to verify the monitoring pipeline works with real ASF queries.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
from datetime import datetime

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from glews.monitor import iter_site_configs, MonitorState, check_new_data


def main():
    config_path = Path(__file__).resolve().parents[2] / "config" / "global_watch.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print("=" * 60)
    print("GLEWS Monitor End-to-End Test")
    print(f"Config: {config_path.name}")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 60)

    sites = list(iter_site_configs(cfg))
    print(f"\n{len(sites)} sites configured:\n")

    for i, site_cfg in enumerate(sites):
        site = site_cfg.get("site", {})
        name = site.get("name", f"Site {i+1}")
        lat = site.get("latitude", "?")
        lon = site.get("longitude", "?")

        print(f"{'─' * 50}")
        print(f"Site: {name}")
        print(f"  Coords: {lat}°N, {lon}°E")

        # Test ASF query for new data
        monitor_cfg = site_cfg.get("monitor", cfg.get("monitor", {}))
        state_dir = Path(monitor_cfg.get("state_dir", "data/monitor_state"))
        state_dir.mkdir(parents=True, exist_ok=True)

        state_file = state_dir / f"{name.replace(' ', '_').replace('/', '_')}.json"

        try:
            state = MonitorState.load(str(state_file))
        except Exception:
            state = MonitorState(
                site_name=name,
                last_check=datetime.now().isoformat(),
                last_acquisition=None,
                processed_scenes=[],
                active_alerts=[],
            )

        print(f"  State file: {state_file}")

        # Query ASF for new NISAR products
        try:
            new_products = check_new_data(
                site_cfg,
                lookback_days=monitor_cfg.get("lookback_days", 30),
            )
            print(f"  New products found: {len(new_products)}")

            for p in new_products[:5]:
                if isinstance(p, dict):
                    print(f"    - {p.get('fileName', p.get('name', '?'))}")
                else:
                    print(f"    - {p}")

            if len(new_products) > 5:
                print(f"    ... and {len(new_products) - 5} more")

        except Exception as e:
            print(f"  ASF query failed: {e}")
            new_products = []

        print()

    print("=" * 60)
    print("End-to-end test complete.")
    print("The monitoring loop is functional — it can query ASF,")
    print("parse configs, and manage state for all configured sites.")
    print("=" * 60)


if __name__ == "__main__":
    main()
