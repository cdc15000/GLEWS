"""
Per-flag data-quality assessment and interferogram thumbnails.

Tier 0 flags are only as trustworthy as the pixels underneath them. A
flag built on low-coherence or unreliably unwrapped pixels can carry a
large z-score that is nothing but unwrapping noise. This module grades
each flag from the per-pair coherence and unwrapping-reliability layers
so the Tier 2 dashboard can show (and cap) unreliable flags, and renders
a strip of wrapped-phase tiles per flag for visual review.

Both functions take plain arrays, so they work for any sensor/product
that provides per-pair coherence and a reliability mask (for NISAR GUNW,
``connectedComponents == 0`` marks unreliable unwrapping).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Grading thresholds
GOOD_MIN_COHERENCE = 0.35
GOOD_MAX_UNRELIABLE = 0.2
UNRELIABLE_MAX_COHERENCE = 0.2
UNRELIABLE_MIN_UNRELIABLE = 0.5
# A pair "contributes" to a flag when its pixels are coherent and unwrapped
PAIR_MIN_COHERENCE = 0.3
PAIR_MAX_UNRELIABLE = 0.5


def grade_reliability(mean_coherence: float, unreliable_fraction: float) -> str:
    """Return "good", "marginal" or "unreliable" for a flag."""
    if not np.isfinite(mean_coherence):
        return "unreliable"
    if (unreliable_fraction >= UNRELIABLE_MIN_UNRELIABLE
            or mean_coherence < UNRELIABLE_MAX_COHERENCE):
        return "unreliable"
    if mean_coherence >= GOOD_MIN_COHERENCE and unreliable_fraction <= GOOD_MAX_UNRELIABLE:
        return "good"
    return "marginal"


def assess_flag_quality(
    flags: list,
    pair_coherence: np.ndarray,
    pair_unreliable: np.ndarray,
    pair_dates: list[tuple[str, str]],
) -> None:
    """
    Attach a quality summary to each flag's ``detection_details["quality"]``.

    Parameters
    ----------
    flags : list[AnomalyFlag]
        Flags whose ``pixel_indices`` index the same grid as the stacks.
    pair_coherence : ndarray [n_pairs, rows, cols]
        Coherence per interferogram pair.
    pair_unreliable : ndarray [n_pairs, rows, cols] of bool
        True where unwrapping is unreliable in that pair.
    pair_dates : list of (ref, sec) ISO date strings, one per pair.
    """
    n_pairs = len(pair_dates)
    if pair_coherence.shape[0] != n_pairs or pair_unreliable.shape[0] != n_pairs:
        raise ValueError("pair stacks and pair_dates must have the same length")

    for flag in flags:
        idx = np.asarray(flag.pixel_indices)
        if idx.size == 0:
            continue
        rows, cols = idx[:, 0], idx[:, 1]
        coh = pair_coherence[:, rows, cols].astype(float)      # [n_pairs, n_px]
        unrel = pair_unreliable[:, rows, cols].astype(float)

        pairs = []
        n_usable = 0
        for k, (ref, sec) in enumerate(pair_dates):
            c = float(np.nanmean(coh[k]))
            u = float(np.nanmean(unrel[k]))
            usable = bool(c >= PAIR_MIN_COHERENCE and u <= PAIR_MAX_UNRELIABLE)
            n_usable += usable
            pairs.append({
                "ref": ref, "sec": sec,
                "coherence": round(c, 3),
                "unreliable_fraction": round(u, 3),
                "usable": usable,
            })

        mean_coh = float(np.nanmean(coh))
        unrel_frac = float(np.nanmean(unrel))
        flag.detection_details["quality"] = {
            "mean_coherence": round(mean_coh, 3),
            "unreliable_fraction": round(unrel_frac, 3),
            "n_pairs": n_pairs,
            "n_pairs_usable": n_usable,
            "reliability": grade_reliability(mean_coh, unrel_frac),
            "pairs": pairs,
        }


def render_interferogram_strips(
    flags: list,
    tiles: list[dict],
    output_dir: str | Path,
    half_width_m: float = 1000.0,
    max_flags: int | None = None,
) -> int:
    """
    Render one PNG strip of wrapped-phase tiles per flag.

    Parameters
    ----------
    flags : list[AnomalyFlag]
        Flags with ``center_lat`` / ``center_lon``.
    tiles : list of dict
        One per pair, chronologically, each with ``ref``, ``sec`` (ISO
        dates), ``phase`` (2-D wrapped phase in radians), and ``lat`` /
        ``lon`` (2-D grids matching ``phase``).
    output_dir : path
        Report output directory; strips go in ``figures/ifg/``.
    half_width_m : float
        Half-width of each tile around the flag centre.

    Sets ``detection_details["ifg_strip"]`` to the path relative to
    ``output_dir`` and returns the number of strips written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    strip_dir = out / "figures" / "ifg"
    strip_dir.mkdir(parents=True, exist_ok=True)
    if not tiles:
        return 0

    n = len(tiles)
    ncols = min(n, 7)
    nrows = int(np.ceil(n / ncols))
    written = 0
    for flag in flags[:max_flags] if max_flags else flags:
        lat0, lon0 = flag.center_lat, flag.center_lon
        dlat = half_width_m / 111_320.0
        dlon = half_width_m / (111_320.0 * np.cos(np.radians(lat0)))
        inside_any = False
        fig, axes = plt.subplots(nrows, ncols, figsize=(1.6 * ncols, 1.9 * nrows + 0.3), squeeze=False)
        for k, ax in enumerate(axes.flat):
            ax.set_xticks([]); ax.set_yticks([])
            if k >= n:
                ax.axis("off")
                continue
            t = tiles[k]
            lat, lon, ph = t["lat"], t["lon"], t["phase"]
            m = ((lat >= lat0 - dlat) & (lat <= lat0 + dlat)
                 & (lon >= lon0 - dlon) & (lon <= lon0 + dlon))
            if m.sum() < 4:
                ax.text(0.5, 0.5, "outside\ncrop", ha="center", va="center", fontsize=6, color="0.5")
            else:
                inside_any = True
                r = np.where(m.any(axis=1))[0]
                c = np.where(m.any(axis=0))[0]
                sub = ph[r[0]:r[-1] + 1, c[0]:c[-1] + 1]
                ax.imshow(sub, cmap="twilight_shifted", vmin=-np.pi, vmax=np.pi,
                          interpolation="nearest", origin="upper" if lat[0, 0] > lat[-1, 0] else "lower")
                ax.plot(sub.shape[1] / 2 - 0.5, sub.shape[0] / 2 - 0.5, "k+", ms=6, mew=1)
            ax.set_title(f"{t['ref'][5:]}→{t['sec'][5:]}", fontsize=6.5, pad=2)
        fig.suptitle(f"Flag {flag.flag_id}: wrapped phase, ±{half_width_m / 1000:.1f} km "
                     "(one colour cycle = λ/2 LOS)", fontsize=7.5)
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=1 - 0.45 / (1.9 * nrows + 0.3),
                            wspace=0.06, hspace=0.32)
        if inside_any:
            path = strip_dir / f"flag_{flag.flag_id:03d}.png"
            fig.savefig(path, dpi=110)
            flag.detection_details["ifg_strip"] = str(path.relative_to(out))
            written += 1
        plt.close(fig)
    return written
