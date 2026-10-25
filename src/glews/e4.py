"""
Detector E4: pre-melt acceleration on NISAR GOFF slant-range offsets.

Frozen for the 2027 prospective test (docs/prereg_2027_prospective.md).
The logic and thresholds are those of Phase 2c
(scripts/nepal_2026/dev_premelt_detector.py); only the season handling is
calendar-based instead of index-based:

- baseline b: median rate over pairs with secondary date in
  [1 Dec (season - 1), 15 Jan], needing >= 3 valid per pixel;
- cutoffs: every pair with secondary date after 15 Jan;
- pre-melt window at a cutoff: pairs with secondary date in
  (15 Jan, min(cutoff, 27 May)], needing >= 3 valid; pm is their median.

With s = sign(b), a moving pixel (|b| >= the ring noise floor) alarms when

    s * pm / |b| >= 1.5    and    (s * pm - |b|) * 365/12 >= 12 m/yr

and a flag is an 8-connected group of >= 6 alarm pixels. Rates are
slant-range metres per 12 days, 3x3 nan-median smoothed, from GOFF layer 2
at SNR >= 5, referenced to the 2-5 km ring around the crop centre, with
loader blunder rejection at 5/12 m per day. Pairs longer than 24 days are
not used.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
from matplotlib.path import Path as _MPath
from scipy import ndimage as ndi

from .nisar import read_goff_pairs
from .tm import fwd, utm_lon0

R0 = 1.5
I0 = 12.0          # m/yr
MIN_PX = 6
MAX_RATE = 5.0 / 12  # m/day, loader blunder rejection
MIN_PAIRS = 3
MAX_PAIR_DAYS = 24
RING_M = (2000.0, 5000.0)
_EIGHT = np.ones((3, 3))


def season_dates(season: int) -> dict:
    """Calendar windows (YYYYMMDD strings) for the season ending in ``season``."""
    return dict(base_start=f"{season - 1}1201", base_end=f"{season}0115",
                pm_start=f"{season}0115", pm_end=f"{season}0527")


def prepare(goff_dir: str | Path, centre: tuple[float, float], season: int, *, track: int | None = None,
            crs: str = "EPSG:32645", target_polygons: list | None = None) -> dict:
    """
    Per-cutoff ratio and increase maps for one crop.

    ``centre`` is (lat, lon). ``target_polygons`` are lists of (lat, lon)
    vertices marking a known source (scoring only; excluded from the ring).
    Returns ``ratio`` and ``incr`` (cutoff x row x col), ``cutoffs``
    (secondary dates), ``target``, ``moving``, ``n_base`` and ``floor``.
    """
    lon0 = utm_lon0(crs)
    W = season_dates(season)
    probe = read_goff_pairs(goff_dir, layer="layer2", min_snr=0, track=track)
    X, Y = np.meshgrid(probe[0]["x_coords"], probe[0]["y_coords"])
    pts = np.c_[X.ravel(), Y.ravel()]
    inside = lambda ll: _MPath(np.array([fwd(a, b, lon0) for a, b in ll])).contains_points(pts).reshape(X.shape)
    target = np.zeros(X.shape, bool)
    for poly in target_polygons or []:
        target |= inside(poly)
    cx, cy = fwd(*centre, lon0)
    d = np.hypot(X - cx, Y - cy)
    ring = (d > RING_M[0]) & (d < RING_M[1]) & ~target
    pairs = read_goff_pairs(goff_dir, layer="layer2", min_snr=5, track=track, reference_mask=ring, max_rate=MAX_RATE)
    pairs = [p for p in pairs if p["days"] <= MAX_PAIR_DAYS]
    dates = [p["sec_date"] for p in pairs]
    Rr = np.array([ndi.generic_filter(p["range_offset"] * 12.0 / p["days"], np.nanmedian, size=3, mode="nearest")
                   for p in pairs])
    S = np.array([1.4826 * np.median(np.abs(r[ring & np.isfinite(r)] - np.median(r[ring & np.isfinite(r)])))
                  for r in Rr])
    base = [i for i, dt in enumerate(dates) if W["base_start"] <= dt <= W["base_end"]]
    if not base:
        raise ValueError(f"no baseline pairs in [{W['base_start']}, {W['base_end']}] in {goff_dir}")
    b = np.where(np.isfinite(Rr[base]).sum(0) >= MIN_PAIRS, np.nanmedian(Rr[base], axis=0), np.nan)
    floor = float(np.median(S[base]))
    moving = np.isfinite(b) & (np.abs(b) >= floor)
    s = np.sign(np.nan_to_num(b))
    ratio, incr, cutoffs = [], [], []
    for k, ck in enumerate(dates):
        if ck <= W["pm_start"]:
            continue
        sel = [i for i, dt in enumerate(dates[:k + 1]) if W["pm_start"] < dt <= W["pm_end"]]
        n = np.isfinite(Rr[sel]).sum(0)
        pm = np.where(n >= MIN_PAIRS, np.nanmedian(Rr[sel], axis=0), np.nan)
        ratio.append(np.where(moving, s * pm / np.abs(b), np.nan))
        incr.append(np.where(moving, (s * pm - np.abs(b)) * 365 / 12, np.nan))
        cutoffs.append(ck)
    return dict(ratio=np.array(ratio), incr=np.array(incr), cutoffs=cutoffs, target=target, moving=moving,
                n_base=len(base), floor=floor, x=X, y=Y, crs=crs)


def flags_at(P: dict, k: int) -> tuple[np.ndarray, int]:
    """Labelled flags (8-connected groups of >= MIN_PX alarm pixels) at cutoff index ``k``."""
    alarm = (np.nan_to_num(P["ratio"][k], nan=0) >= R0) & (np.nan_to_num(P["incr"][k], nan=-1e9) >= I0)
    lab, n = ndi.label(alarm, structure=_EIGHT)
    sz = ndi.sum(alarm, lab, range(1, n + 1))
    keep = [i + 1 for i, v in enumerate(sz) if v >= MIN_PX]
    out = np.where(np.isin(lab, keep), lab, 0)
    return out, len(keep)


def persistent_first(hits: list[bool]) -> int | None:
    """First index flagged and flagged at all later indices except at most one."""
    for i, h in enumerate(hits):
        if h and sum(1 for x in hits[i:] if not x) <= 1:
            return i
    return None


def run(P: dict, before: str | None = None) -> dict:
    """
    False alarms and target hit over the cutoffs (only those with secondary
    date before ``before``, YYYYMMDD, when given).

    ``fa``: distinct 8-connected groups in the union over cutoffs of flags
    not touching the target. ``first``: first persistent cutoff of a flag
    touching the target, or None.
    """
    idx = [k for k, c in enumerate(P["cutoffs"]) if before is None or c < before]
    fa_union, hits = np.zeros(P["target"].shape, bool), []
    for k in idx:
        lab, _ = flags_at(P, k)
        hit = False
        for i in np.unique(lab[lab > 0]):
            m = lab == i
            if (m & P["target"]).any():
                hit = True
            else:
                fa_union |= m
        hits.append(hit)
    i0 = persistent_first(hits)
    return dict(fa=int(ndi.label(fa_union, structure=_EIGHT)[1]), first=P["cutoffs"][idx[i0]] if i0 is not None else None,
                hits=hits, cutoffs=[P["cutoffs"][k] for k in idx], fa_mask=fa_union)


def lead_days(first: str | None, event: date) -> int | None:
    if first is None:
        return None
    return (event - date(int(first[:4]), int(first[4:6]), int(first[6:]))).days
