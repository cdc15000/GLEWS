"""Phase 1e (docs/acceptance_phase1e.md): replay NISAR GOFF pair by pair through two detectors.

Usage (from the repository root):
    python -u scripts/nepal_2026/replay_goff.py [GOFF_DIR]

E1 runs GLEWS's existing pipeline (load_nisar_goff_stack ->
compute_acceleration_map -> detect_anomalies, config unchanged) on the
pairs available at each cutoff. E2 runs the rate-change detector specified
in the criteria. Both were pre-registered before this script was run.
Writes output/acceptance/E_replay.json and E_replay.png.
"""
import sys, json, logging, tempfile, yaml, numpy as np
from datetime import date
from pathlib import Path
from matplotlib.path import Path as MP
from scipy import ndimage as ndi
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).parent))
from glews.nisar import load_nisar_goff_stack, read_goff_pairs
from glews.timeseries import compute_acceleration_map
from glews.detect import detect_anomalies
from utm import fwd, inv

logging.basicConfig(level=logging.ERROR)
GOFF = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/nisar/nepal_2026/goff_t98_crop"
OUT = ROOT / "output/acceptance"
EVENT = date(2026, 8, 26)
LATEST_OK = "20260726"
cfg = yaml.safe_load(open(ROOT / "config/nepal_2026.yaml"))
site = cfg["site"]
cx, cy = fwd(site["latitude"], site["longitude"])
icimod = np.array([fwd(a, b) for a, b in site["known_source"]["polygon"]])
outlet = np.array([fwd(a, b) for a, b in site["reference_areas"][0]["polygon"]])

files = sorted(GOFF.glob("NISAR_*GOFF*.h5"), key=lambda p: p.name.split("_")[13])
probe = read_goff_pairs(GOFF, layer="layer2", min_snr=0, track=98)[0]
X, Y = np.meshgrid(probe["x_coords"], probe["y_coords"])
inside = lambda P: MP(P).contains_points(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
src, out_m = inside(icimod), inside(outlet)
target = src | out_m
d = np.hypot(X - cx, Y - cy)
ring = (d > 2000) & (d < 5000) & ~src
LAT, LON = inv(X, Y)
pairs_all = read_goff_pairs(GOFF, layer="layer2", min_snr=5, track=98, reference_mask=ring)
cut_dates = [p["sec_date"] for p in pairs_all][5:]
print(f"{len(pairs_all)} pairs; cutoffs {cut_dates[0]} .. {cut_dates[-1]}; target {target.sum()} px", flush=True)


def lead(first):
    return (EVENT - date(int(first[:4]), int(first[4:6]), int(first[6:]))).days


def persistent_first(hits):
    """First cutoff whose hit persists to the end with at most one miss."""
    for i, h in enumerate(hits):
        if h and sum(1 for x in hits[i:] if not x) <= 1:
            return i
    return None


# ================================ E1 ============================================
e1 = []
for k, cd in enumerate(cut_dates, 1):
    with tempfile.TemporaryDirectory() as td:
        for f in files:
            if f.name.split("_")[13][:8] <= cd:
                (Path(td) / f.name).symlink_to(f.resolve())
        ts = load_nisar_goff_stack(td, track=98, reference_mask=ring)
    am = compute_acceleration_map(ts.dates, ts.displacement,
                                  window_size_days=cfg["detect"]["acceleration"]["window_size_days"],
                                  step_days=cfg["detect"]["acceleration"]["step_days"],
                                  n_harmonics=cfg["detect"].get("n_harmonics", 2))
    flags = detect_anomalies(am, LAT, LON, cfg, dates=ts.dates, displacement=ts.displacement)
    flags = sorted(flags, key=lambda f: -f.score)
    hit_ranks = [i + 1 for i, f in enumerate(flags)
                 if len(f.pixel_indices) and target[f.pixel_indices[:, 0], f.pixel_indices[:, 1]].any()]
    e1.append(dict(cutoff=cd, n_flags=len(flags), hit=bool(hit_ranks), hit_rank=hit_ranks[0] if hit_ranks else None,
                   false_alarms=len(flags) - len(hit_ranks)))
    q = e1[-1]
    print(f"E1 [{'█' * k}{'░' * (len(cut_dates) - k)}] {cd}: {q['n_flags']:3d} flags, "
          f"hit rank {q['hit_rank']}, false alarms {q['false_alarms']}", flush=True)
i1 = persistent_first([q["hit"] for q in e1])
first1 = e1[i1]["cutoff"] if i1 is not None else None
fa_jul = next((q["false_alarms"] for q in e1 if q["cutoff"] == LATEST_OK), None)
e1_pass = bool(first1 and first1 <= LATEST_OK and e1[i1]["hit_rank"] <= 3 and fa_jul is not None and fa_jul <= 10)
E1 = dict(cutoffs=e1, first_persistent_hit=first1, lead_days=lead(first1) if first1 else None,
          rank_at_first=e1[i1]["hit_rank"] if i1 is not None else None, false_alarms_26jul=fa_jul,
          verdict="PASS" if e1_pass else "FAIL")
print(f"E1: first persistent hit {first1} (lead {E1['lead_days']} d, rank {E1['rank_at_first']}), "
      f"false alarms at 26 Jul {fa_jul} -> {E1['verdict']}\n", flush=True)

# ================================ E2 ============================================
R = []
S = []
for p in pairs_all:
    r = ndi.generic_filter(p["range_offset"] * 12.0 / p["days"], np.nanmedian, size=3, mode="nearest")
    R.append(r)
    v = r[ring & np.isfinite(r)]
    S.append(1.4826 * np.median(np.abs(v - np.median(v))))
R = np.array(R); S = np.array(S)
base = R[:4]
nb = np.isfinite(base).sum(axis=0)
b = np.where(nb >= 3, np.nanmedian(base, axis=0), np.nan)
sb = 1.2533 * np.median(S[:4]) / np.sqrt(np.maximum(nb, 1))
Z = np.abs(R - b) / np.sqrt(S[:, None, None] ** 2 + sb ** 2)
e2, fa_union = [], np.zeros_like(target)
for k in range(5, len(pairs_all)):
    win = Z[max(4, k - 3):k + 1]
    alarm = (np.nan_to_num(win, nan=0) >= 3).sum(axis=0) >= 3
    lab, n = ndi.label(alarm, structure=np.ones((3, 3)))
    sizes = ndi.sum(alarm, lab, range(1, n + 1))
    flags = [i + 1 for i, s_ in enumerate(sizes) if s_ >= 4]
    hits = [i for i in flags if (target & (lab == i)).any()]
    fas = [i for i in flags if i not in hits]
    for i in fas:
        fa_union |= lab == i
    q = dict(cutoff=pairs_all[k]["sec_date"], n_flags=len(flags), hit=bool(hits), false_alarms=len(fas),
             false_alarm_km2=round(float(sum(sizes[i - 1] for i in fas)) * 0.0064, 3),
             hit_pixels_in_target=int(sum((target & (lab == i)).sum() for i in hits)))
    e2.append(q)
    print(f"E2 {q['cutoff']}: {q['n_flags']} flags, hit {q['hit']} ({q['hit_pixels_in_target']} target px), "
          f"false alarms {q['false_alarms']} ({q['false_alarm_km2']} km2)", flush=True)
labu, n_fa = ndi.label(fa_union, structure=np.ones((3, 3)))
i2 = persistent_first([q["hit"] for q in e2])
first2 = e2[i2]["cutoff"] if i2 is not None else None
any2 = next((q["cutoff"] for q in e2 if q["hit"]), None)
e2_pass = bool(first2 and first2 <= LATEST_OK and n_fa <= 5)
E2 = dict(cutoffs=e2, first_persistent_hit=first2, lead_days=lead(first2) if first2 else None,
          first_any_hit=any2, distinct_false_alarm_flags=int(n_fa),
          false_alarm_union_km2=round(float(fa_union.sum()) * 0.0064, 3), verdict="PASS" if e2_pass else "FAIL")
print(f"E2: first persistent hit {first2} (lead {E2['lead_days']} d; first any hit {any2}), "
      f"distinct false-alarm flags {n_fa} ({E2['false_alarm_union_km2']} km2) -> {E2['verdict']}")
(OUT / "E_replay.json").write_text(json.dumps(dict(criteria="docs/acceptance_phase1e.md", E1=E1, E2=E2), indent=1, default=float))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 2, figsize=(11, 5))
ext = [X[0, 0] - 40, X[0, -1] + 40, Y[-1, 0] - 40, Y[0, 0] + 40]
k_last = len(pairs_all) - 1
axs[0].imshow(np.nan_to_num(Z[k_last], nan=0), extent=ext, cmap="magma", vmin=0, vmax=8)
axs[0].set_title(f"E2 z-score, pair ending {pairs_all[k_last]['sec_date']}", fontsize=9)
axs[1].imshow(np.where(fa_union, 1.0, np.nan), extent=ext, cmap="autumn", vmin=0, vmax=1)
axs[1].set_title(f"E2 false-alarm pixels, all cutoffs ({n_fa} flags)", fontsize=9)
for ax in axs:
    ax.plot(*np.vstack([icimod, icimod[:1]]).T, c="cyan", lw=1)
    ax.plot(*np.vstack([outlet, outlet[:1]]).T, c="lime", lw=1)
    ax.tick_params(labelsize=6)
fig.tight_layout(); fig.savefig(OUT / "E_replay.png", dpi=110)
print("wrote", OUT / "E_replay.json")
