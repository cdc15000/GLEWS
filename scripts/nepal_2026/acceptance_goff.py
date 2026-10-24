"""Phase 1 acceptance test A1 (docs/acceptance_phase1.md): NISAR GOFF over the source.

Usage (from the repository root):
    python scripts/nepal_2026/acceptance_goff.py [GOFF_CROP_DIR] [--sweep]

Pre-registered setting: pair 26 Jul -> 19 Aug 2026, slant range, layer 2,
SNR >= 5, referenced to the stable ring (2-5 km, source excluded).
Pass if |signal| >= 5 sigma and the signal ranks >= 95th percentile
among 500 random source-shaped patches. --sweep also reports every
layer x SNR setting for every pair, for transparency; the verdict comes
from the pre-registered setting only. Writes JSON next to the crops' output.
"""
import sys, json, yaml, numpy as np
from pathlib import Path
from matplotlib.path import Path as MP
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).parent))
from glews.nisar import read_goff_pairs
from utm import inv

args = [a for a in sys.argv[1:] if not a.startswith("--")]
GOFF = Path(args[0]) if args else ROOT / "data/nisar/nepal_2026/goff_t98_crop"
SWEEP = "--sweep" in sys.argv
PAIR = ("20260726", "20260819")
N_PATCH, K_SIGMA, PCTL = 500, 5.0, 95.0

site = yaml.safe_load(open(ROOT / "config/nepal_2026.yaml"))["site"]
poly = np.array(site["known_source"]["polygon"])


def masks(p):
    X, Y = np.meshgrid(p["x_coords"], p["y_coords"])
    la, lo = inv(X, Y)
    src = MP(poly[:, ::-1]).contains_points(np.c_[lo.ravel(), la.ravel()]).reshape(la.shape)
    d = np.hypot(la - site["latitude"], (lo - site["longitude"]) * np.cos(np.radians(site["latitude"]))) * 111.2
    return src, (d > 2) & (d < 5) & ~src


def patch_rank(field, src, rng):
    """Percentile of the source median among medians of shifted copies of the source mask."""
    r, c = np.nonzero(src)
    sig = np.nanmedian(field[src])
    H, W = field.shape
    meds = []
    for _ in range(20 * N_PATCH):
        if len(meds) == N_PATCH:
            break
        dr = rng.integers(-r.min(), H - r.max()); dc = rng.integers(-c.min(), W - c.max())
        rr, cc = r + dr, c + dc
        if src[rr, cc].any():
            continue
        v = field[rr, cc]
        if np.isfinite(v).sum() >= max(5, 0.5 * np.isfinite(field[src]).sum()):
            meds.append(np.nanmedian(v))
    meds = np.abs(np.array(meds))
    return float(100 * np.mean(meds < abs(sig))), len(meds)


def evaluate(layer, snr, pair_filter=None):
    probe = read_goff_pairs(GOFF, layer=layer, min_snr=0, track=98)
    src, ring = masks(probe[0])
    pairs = read_goff_pairs(GOFF, layer=layer, min_snr=snr, track=98, reference_mask=ring)
    rows = []
    for p in pairs:
        if pair_filter and (p["ref_date"], p["sec_date"]) != pair_filter:
            continue
        rg = p["range_offset"]
        n_ok = int(np.isfinite(rg[src]).sum())
        row = dict(ref=p["ref_date"], sec=p["sec_date"], days=p["days"], layer=layer, min_snr=snr,
                   source_px=int(src.sum()), source_px_valid=n_ok)
        if n_ok >= 5:
            sig = float(np.nanmedian(rg[src])); sigma = p["noise_sigma"][0]
            rank, n = patch_rank(rg, src, np.random.default_rng(0))
            row.update(signal_m=round(sig, 3), sigma_m=round(sigma, 3), snr_sigma=round(abs(sig) / sigma, 2),
                       along_track_m=round(float(np.nanmedian(p["along_track_offset"][src])), 3),
                       patch_rank_pct=round(rank, 1), n_patches=n)
        rows.append(row)
    return rows


main = evaluate("layer2", 5.0, PAIR)
verdict = {"criterion": "A1", "setting": {"pair": PAIR, "layer": "layer2", "min_snr": 5.0},
           "result": main[0] if main else None}
if main and "signal_m" in main[0]:
    m = main[0]
    verdict["pass_sigma"] = m["snr_sigma"] >= K_SIGMA
    verdict["pass_rank"] = m["patch_rank_pct"] >= PCTL
    verdict["pass"] = verdict["pass_sigma"] and verdict["pass_rank"]
else:
    verdict["pass"] = False
    verdict["note"] = "too few valid source pixels"
print(json.dumps(verdict, indent=1))

if SWEEP:
    sweep = []
    print("\nlayer   snr  pair               days  src ok  signal m  sigma m  |s|/sigma  along m  rank%")
    for layer in ("layer1", "layer2", "layer3"):
        for snr in (0.0, 3.0, 5.0, 10.0):
            for r in evaluate(layer, snr):
                sweep.append(r)
                if "signal_m" in r:
                    print(f"{layer} {snr:5.1f}  {r['ref']}-{r['sec']}  {r['days']:3d}  {r['source_px_valid']:3d}/{r['source_px']}"
                          f"  {r['signal_m']:7.3f}  {r['sigma_m']:7.3f}  {r['snr_sigma']:8.2f}  {r['along_track_m']:7.3f}  {r['patch_rank_pct']:5.1f}")
                else:
                    print(f"{layer} {snr:5.1f}  {r['ref']}-{r['sec']}  {r['days']:3d}  {r['source_px_valid']:3d}/{r['source_px']}  too few valid pixels")
    verdict["sweep"] = sweep
out = ROOT / "output/acceptance"; out.mkdir(parents=True, exist_ok=True)
(out / "A1_goff.json").write_text(json.dumps(verdict, indent=1))
print("wrote", out / "A1_goff.json")
