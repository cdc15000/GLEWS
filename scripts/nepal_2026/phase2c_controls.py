"""Phase 2c (docs/acceptance_phase2c.md): pre-melt detector E4 on fresh control sites.

Usage (from the repository root):
    python -u scripts/nepal_2026/phase2c_controls.py

Coverage floor first (validity mask only), then E4 (frozen, imported from
dev_premelt_detector.py), E2 and E2 + blunder rejection at each control,
the difficulty check, the pass rule, and the Nepal E4 hit (in-sample).
Writes output/acceptance/phase2c.json. No plots.
"""
import sys, json, logging, numpy as np
from pathlib import Path
from scipy import ndimage as ndi
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).parent))
from glews.nisar import read_goff_pairs
from utm import fwd
import dev_premelt_detector as D

logging.basicConfig(level=logging.ERROR)
R0, I0, MIN_PX = 1.5, 12.0, 6
NEIGHBOURS = [(1.3, 12.0, 6), (1.7, 12.0, 6), (1.5, 10.0, 6), (1.5, 14.0, 6), (1.5, 12.0, 4)]
COVERAGE_FLOOR, LIMIT, INFORMATIVE = 0.50, 2, 10
CONTROLS = [("annapurna_sanctuary", "data/nisar/control_annapurna_sanctuary/goff_t98_crop", (28.55, 83.90), False),
            ("langtang_east", "data/nisar/control_langtang_east/goff_t98_crop", (28.25, 85.72), False)]
FALLBACKS = [("manaslu_west", "data/nisar/control_manaslu_west/goff_t98_crop", (28.58, 84.48), False),
             ("lapchi", "data/nisar/control_lapchi/goff_t98_crop", (28.00, 86.18), False)]


def coverage(g):
    ps = read_goff_pairs(ROOT / g, layer="layer2", min_snr=5, track=98)
    return len(ps), float(np.median([np.isfinite(p["range_offset"]).mean() for p in ps]))


def e2_flags(g, centre, max_rate):
    """E2 exactly as Phase 1e / Phase 2 (no filters); every flag counts at a control."""
    pairs = read_goff_pairs(ROOT / g, layer="layer2", min_snr=0, track=98)
    X, Y = np.meshgrid(pairs[0]["x_coords"], pairs[0]["y_coords"])
    cx, cy = fwd(*centre); d = np.hypot(X - cx, Y - cy); ring = (d > 2000) & (d < 5000)
    pairs = read_goff_pairs(ROOT / g, layer="layer2", min_snr=5, track=98, reference_mask=ring, max_rate=max_rate)
    Rr = np.array([ndi.generic_filter(p["range_offset"] * 12.0 / p["days"], np.nanmedian, size=3, mode="nearest") for p in pairs])
    S = np.array([1.4826 * np.median(np.abs(r[ring & np.isfinite(r)] - np.median(r[ring & np.isfinite(r)]))) for r in Rr])
    nb = np.isfinite(Rr[:4]).sum(axis=0)
    b = np.where(nb >= 3, np.nanmedian(Rr[:4], axis=0), np.nan)
    sb = 1.2533 * np.median(S[:4]) / np.sqrt(np.maximum(nb, 1))
    Z = np.abs(Rr - b) / np.sqrt(S[:, None, None] ** 2 + sb ** 2)
    union = np.zeros(X.shape, bool)
    for k in range(5, len(pairs)):
        alarm = (np.nan_to_num(Z[max(4, k - 3):k + 1], nan=0) >= 3).sum(axis=0) >= 3
        lab, n = ndi.label(alarm, structure=np.ones((3, 3)))
        sizes = ndi.sum(alarm, lab, range(1, n + 1))
        union |= np.isin(lab, [i + 1 for i, s_ in enumerate(sizes) if s_ >= 4])
    return int(ndi.label(union, structure=np.ones((3, 3)))[1]), round(float(union.sum()) * 0.0064, 2)


def e4_detail(P):
    """E4 flags with their first cutoff and area, for the record."""
    union, first = np.zeros(P["target"].shape, bool), {}
    for k, cd in enumerate(P["cutoffs"]):
        a = (np.nan_to_num(P["ratio"][k], nan=0) >= R0) & (np.nan_to_num(P["incr"][k], nan=-1e9) >= I0)
        lab, n = ndi.label(a, structure=np.ones((3, 3))); sz = ndi.sum(a, lab, range(1, n + 1))
        for i in [i + 1 for i, v in enumerate(sz) if v >= MIN_PX]:
            union |= (lab == i) & ~P["target"]
    lab, n = ndi.label(union, structure=np.ones((3, 3)))
    for k, cd in enumerate(P["cutoffs"]):
        a = (np.nan_to_num(P["ratio"][k], nan=0) >= R0) & (np.nan_to_num(P["incr"][k], nan=-1e9) >= I0)
        for i in range(1, n + 1):
            if i not in first and (a & (lab == i)).sum() >= 1:
                first[i] = cd
    return dict(area_km2=round(float(union.sum()) * 0.0064, 3),
                flags=[dict(px=int((lab == i).sum()), first_cutoff=first.get(i)) for i in range(1, n + 1)])


sites, queue = [], list(CONTROLS)
fallbacks = list(FALLBACKS)
report = {}
while queue:
    s = queue.pop(0)
    if not (ROOT / s[1]).exists():
        print(f"{s[0]}: GOFF crop not extracted ({s[1]}); stopping"); sys.exit(1)
    n, cov = coverage(s[1])
    print(f"{s[0]}: {n} pairs, median valid share {cov:.2f} (floor {COVERAGE_FLOOR})", flush=True)
    report.setdefault("coverage", {})[s[0]] = dict(n_pairs=n, median_valid_share=round(cov, 3))
    if n == 21 and cov >= COVERAGE_FLOOR:
        sites.append(s)
    else:
        rep = fallbacks.pop(0)
        print(f"  -> fails coverage; replaced by {rep[0]}", flush=True)
        report.setdefault("replacements", []).append(dict(failed=s[0], replacement=rep[0]))
        queue.append(rep)

results = {}
for s in sites:
    print(f"\n== {s[0]}", flush=True)
    P = D.prepare(*s)
    e4 = D.run(P, R0, I0, MIN_PX)
    e2 = e2_flags(s[1], s[2], None); e2b = e2_flags(s[1], s[2], 5.0 / 12)
    sens = {f"{a}/{b}/{c}": D.run(P, a, b, c)["fa"] for a, b, c in NEIGHBOURS}
    results[s[0]] = dict(E4_flags=e4["fa"], E4_detail=e4_detail(P), E2_flags=e2[0], E2_km2=e2[1],
                         E2_blunder_flags=e2b[0], E2_blunder_km2=e2b[1],
                         informative=e2b[0] >= INFORMATIVE, E4_sensitivity=sens,
                         rate_per_100km2=round(e4["fa"] / P["area_km2"] * 100, 2))
    r = results[s[0]]
    print(f"  E4 false-alarm flags {r['E4_flags']} ({r['rate_per_100km2']}/100 km2) {r['E4_detail']}\n"
          f"  E2 {r['E2_flags']} flags; E2 + blunder rejection {r['E2_blunder_flags']} flags -> "
          f"{'informative' if r['informative'] else 'UNINFORMATIVE (quiet site)'}\n  sensitivity (R0/I0/px: flags) {sens}", flush=True)

nepal = D.run(D.prepare(*D.SITES[0]), R0, I0, MIN_PX)
passes = all(r["E4_flags"] <= LIMIT for r in results.values()) and len(results) == 2
informative = any(r["informative"] for r in results.values())
verdict = "PASS" if passes and informative else ("PASS (uninformative)" if passes else "FAIL")
print(f"\nNepal E4 (in-sample): first persistent hit {nepal['first']}, lead {nepal['lead']} d, false alarms {nepal['fa']}")
print(f"Verdict: {verdict}")
report.update(criteria="docs/acceptance_phase2c.md", setting=dict(R0=R0, I0=I0, min_px=MIN_PX), controls=results,
              nepal_in_sample=nepal, verdict=verdict)
(D.OUT / "phase2c.json").write_text(json.dumps(report, indent=1, default=float))
print("wrote", D.OUT / "phase2c.json")
