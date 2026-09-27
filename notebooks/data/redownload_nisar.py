#!/usr/bin/env python3
"""Re-download NISAR products from ASF. Run from the glews project root."""
import asf_search as asf
import netrc
from pathlib import Path

n = netrc.netrc()
auth = n.authenticators("urs.earthdata.nasa.gov")
session = asf.ASFSession()
session.auth_with_creds(auth[0], auth[2])

SITE_LAT, SITE_LON = 28.20, 85.90

for level, subdir in [("GUNW", "nisar/gunw"), ("GOFF", "nisar/goff")]:
    results = asf.search(
        platform="NISAR", processingLevel=level,
        intersectsWith=f"POINT({SITE_LON} {SITE_LAT})",
        start="2025-01-01", end="2026-08-26", maxResults=200,
    )
    out = Path(f"notebooks/data/{subdir}")
    out.mkdir(parents=True, exist_ok=True)
    existing = {f.name for f in out.iterdir()}
    for i, r in enumerate(sorted(results, key=lambda x: x.properties["startTime"])):
        fname = r.properties["fileName"]
        if fname in existing:
            continue
        print(f"{level} {i+1}/{len(results)}: {fname[:60]}...", flush=True)
        r.download(out, session=session)
    print(f"{level}: {len(results)} products total, {len(list(out.glob('*.h5')))} on disk")

print("Done.")
