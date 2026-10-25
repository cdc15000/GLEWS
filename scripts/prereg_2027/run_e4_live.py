"""2027 prospective test (docs/prereg_2027_prospective.md): frozen E4, run live.

Usage (from the repository root):
    python -u scripts/prereg_2027/run_e4_live.py eligibility          # on/after 15 Feb 2027; metadata only
    python -u scripts/prereg_2027/run_e4_live.py run                  # each new GOFF granule, to 27 May 2027
    python -u scripts/prereg_2027/run_e4_live.py replay NAME GOFF_DIR LAT LON SEASON LOG [--osm-date ISO]

- eligibility: for each site in config/hkh_priority.yaml, the track by
  glews.goff_crop.choose_track and the count of pairs with secondary date in
  [1 Dec 2026, 15 Jan 2027]; >= 3 is eligible. Writes alarms/e4_2027/eligibility.json.
- run: for each eligible site, crops every new pair (secondary date before
  today), runs glews.e4 and appends one line per new cutoff to
  alarms/e4_2027/alarm_log.jsonl. Nothing is pushed by this script.
- replay: the same logging over an existing crop directory, cutoff by
  cutoff as if live (dry run on seen 2026 data); writes to LOG.

Each logged flag carries its pixel list, centroid, size and whether it lies
on exposure-relevant terrain (glews.exposure reach, H/L >= 0.15, buffer
250 m, OSM as of 1 Dec 2026 unless --osm-date is given).
"""
import sys, json, subprocess, logging, argparse, numpy as np
from datetime import datetime, timezone, date
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from glews import e4, goff_crop as gc
from glews.tm import fwd, inv, utm_lon0


logging.basicConfig(level=logging.WARNING)
SEASON = 2027
OSM_DATE = "2026-12-01T00:00:00Z"
HL, BUFFER, POST, HALF_DOMAIN = 0.15, 250.0, 90.0, 46000.0
ALARM_DIR = ROOT / "alarms/e4_2027"
CROPS = ROOT / "data/nisar/prereg_2027"
EXPO_CACHE = ROOT / "data/exposure/prereg_2027"
CODE_TAG = "e4-frozen-2027"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def slug(name: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in name.split(",")[0]).strip("_")


def relevance(name, centre, X, Y, crs, osm_date=OSM_DATE):
    """Crop cells whose D8 path reaches an OSM asset at H/L >= HL (glews.exposure)."""
    from glews.exposure import fetch_osm_assets, rasterize_assets, reach_exposure, glo30_mosaic
    lon0 = utm_lon0(crs)
    cx, cy = fwd(*centre, lon0)
    x0, y0 = round((cx - HALF_DOMAIN) / POST) * POST, round((cy + HALF_DOMAIN) / POST) * POST
    n = int(2 * HALF_DOMAIN / POST)
    z = glo30_mosaic(x0, y0, POST, (n, n), crs=crs, cache=EXPO_CACHE / f"{name}_dem90.npy")
    lats, lons = inv(np.array([x0, x0 + n * POST]), np.array([y0 - n * POST, y0]), lon0)
    assets = fetch_osm_assets(float(lats.min()), float(lons.min()), float(lats.max()), float(lons.max()), date=osm_date,
                              cache=EXPO_CACHE / f"{name}_osm_{osm_date[:10].replace('-', '')}.json")
    xy = [fwd(np.array(a["lat"]), np.array(a["lon"]), lon0) for a in assets]
    exposed = rasterize_assets(xy, x0, y0, POST, (n, n))
    ri = np.clip(((y0 - Y) / POST).astype(int), 0, n - 1)
    ci = np.clip(((X - x0) / POST).astype(int), 0, n - 1)
    src = np.zeros((n, n), bool); src[ri, ci] = True
    return reach_exposure(z, POST, exposed, src, min_hl=HL, buffer_m=BUFFER)[ri, ci]


def logged_cutoffs(log: Path, name: str) -> set:
    if not log.exists():
        return set()
    return {(r["site"], r["cutoff"]) for r in map(json.loads, log.read_text().splitlines()) if r["site"] == name}


def log_site(name, goff_dir, centre, season, log, track=None, osm_date=OSM_DATE, upto=None):
    """Append one line per not-yet-logged cutoff (secondary date <= upto, if given)."""
    crs = gc.crop_crs(goff_dir)
    lon0 = utm_lon0(crs)
    P = e4.prepare(goff_dir, centre, season, track=None if track is None else track[0], crs=crs)
    rel = relevance(name, centre, P["x"], P["y"], crs, osm_date)
    done = logged_cutoffs(log, name)
    lines = []
    for k, c in enumerate(P["cutoffs"]):
        if (name, c) in done or (upto and c > upto):
            continue
        lab, n = e4.flags_at(P, k)
        flags = []
        for i in np.unique(lab[lab > 0]):
            r_, c_ = np.nonzero(lab == i)
            la, lo = inv(P["x"][r_, c_].mean(), P["y"][r_, c_].mean(), lon0)
            flags.append(dict(n_px=int(r_.size), centroid=[round(float(la), 5), round(float(lo), 5)],
                              pixels=[[int(a), int(b)] for a, b in zip(r_, c_)],
                              relevant=bool(rel[r_, c_].any())))
        lines.append(dict(site=name, cutoff=c, season=season, crs=crs, track=track, n_base=P["n_base"],
                          floor_m_per_12d=round(P["floor"], 4), n_flags=n, flags=flags,
                          run_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                          code_tag=CODE_TAG, git_commit=git_commit()))
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as fo:
        for ln in lines:
            fo.write(json.dumps(ln) + "\n")
    return lines


def eligibility():
    import yaml
    W = e4.season_dates(SEASON)
    rows = []
    for s in yaml.safe_load(open(ROOT / "config/hkh_priority.yaml"))["sites"]:
        S = s["site"]; lat, lon = S["latitude"], S["longitude"]
        res = gc.search(lat, lon, f"{SEASON - 1}-11-01", f"{SEASON}-01-31")
        trk = gc.choose_track(res, lat, lon, W["base_start"], W["base_end"])
        n = 0
        if trk:
            ps = [r["properties"] for r in res if (r["properties"]["pathNumber"], r["properties"]["frameNumber"],
                                                    r["properties"]["flightDirection"]) == trk]
            n = sum(1 for (_, sec) in gc.unique_pairs(ps) if W["base_start"] <= sec <= W["base_end"])
        rows.append(dict(name=S["name"], slug=slug(S["name"]), lat=lat, lon=lon, track=list(trk) if trk else None,
                         n_base_pairs=n, eligible=n >= 3,
                         reason=None if n >= 3 else ("no track holds the crop plus 1 km" if not trk else f"{n} baseline pairs < 3")))
        print(f"{S['name'][:44]:44s} {str(trk):28s} base {n}  {'ELIGIBLE' if n >= 3 else ''}", flush=True)
    ALARM_DIR.mkdir(parents=True, exist_ok=True)
    (ALARM_DIR / "eligibility.json").write_text(json.dumps(dict(
        judged_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), season=SEASON,
        rule="track per glews.goff_crop.choose_track; >= 3 pairs with secondary date in [1 Dec, 15 Jan]; metadata only",
        sites=rows), indent=1))


def run():
    el = json.loads((ALARM_DIR / "eligibility.json").read_text())
    today = date.today().strftime("%Y%m%d")
    for s in [r for r in el["sites"] if r["eligible"]]:
        trk = tuple(s["track"])
        res = gc.search(s["lat"], s["lon"], f"{SEASON - 1}-11-01", date.today().isoformat())
        d = CROPS / s["slug"]
        gc.extract(s["lat"], s["lon"], d, res, trk, sec_before=today)
        new = log_site(s["slug"], d, (s["lat"], s["lon"]), SEASON, ALARM_DIR / "alarm_log.jsonl", track=trk)
        for ln in new:
            print(f"{s['slug']:28s} {ln['cutoff']}  flags {ln['n_flags']}  relevant {sum(f['relevant'] for f in ln['flags'])}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("eligibility"); sub.add_parser("run")
    rp = sub.add_parser("replay")
    for a in ("name", "goff_dir"):
        rp.add_argument(a)
    rp.add_argument("lat", type=float); rp.add_argument("lon", type=float)
    rp.add_argument("season", type=int); rp.add_argument("log", type=Path)
    rp.add_argument("--osm-date", default=OSM_DATE)
    a = ap.parse_args()
    if a.cmd == "eligibility":
        eligibility()
    elif a.cmd == "run":
        run()
    else:
        # E4 is causal (cutoff k reads pairs up to k, plus the winter baseline),
        # so logging every cutoff at once equals replaying pair by pair.
        for ln in log_site(a.name, a.goff_dir, (a.lat, a.lon), a.season, a.log, osm_date=a.osm_date):
            print(f"{a.name} {ln['cutoff']} flags {ln['n_flags']} relevant {sum(f['relevant'] for f in ln['flags'])}", flush=True)
