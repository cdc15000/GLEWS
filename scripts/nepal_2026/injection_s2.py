"""Injection test for the post-hoc Sentinel-2 re-test: can it see motion inside the scar?

Usage (from the repository root):
    python scripts/nepal_2026/injection_s2.py [S2_DIR]

For clean spring pairs, the secondary image inside the scar is replaced
by a copy of itself shifted west by a known amount (0.6 and 1.2 px a
year, i.e. 6 and 12 m/yr), then the same speed_map as posthoc_s2.py
is run. If the injected motion comes back, a null over the scar means
"no motion of that size"; if not, it means "not measurable here".
Also reports the fraction of scar and ring windows with correlation
>= 0.6. Writes output/acceptance/injection_s2.json.
"""
import sys, json, runpy, numpy as np
from pathlib import Path
from scipy.ndimage import shift as nd_shift
ROOT = Path(__file__).resolve().parents[2]
sys.argv = sys.argv[:2]
# Reuse the post-hoc setup (scar, windows, helpers) without re-running its pair loop
src = (Path(__file__).parent / "posthoc_s2.py").read_text()
setup = src[:src.index("rows, maps = [], {}")]
g = {"__file__": str(Path(__file__).parent / "posthoc_s2.py"), "__name__": "posthoc_setup"}
exec(compile(setup, "posthoc_s2_setup", "exec"), g)
img, CROP, scar, wscar, ring, ndays = g["img"], g["CROP"], g["scar"], g["wscar"], g["ring"], g["ndays"]
offset_field, MIN_PEAK, STABLE_PEAK, PIX = g["offset_field"], g["MIN_PEAK"], g["STABLE_PEAK"], g["PIX"]
WINDOW, STEP = g["WINDOW"], g["STEP"]

PAIRS = [("2024-03-05", "2025-03-05"), ("2019-03-27", "2020-03-26"), ("2023-03-16", "2024-03-15")]
scar_c = scar[CROP, CROP]
out = []
for a, b in PAIRS:
    if a not in g["by_date"] or b not in g["by_date"]:
        print("missing", a, b); continue
    yr = ndays(a, b) / 365.25
    ia, ib = img(a), img(b)
    for rate in (0.0, 6.0, 12.0):
        px = -rate * yr / PIX                       # westward = negative column shift
        sec = ib.copy()
        if rate:
            moved = nd_shift(ib, (0, px), order=3, mode="nearest")
            sec[scar_c] = moved[scar_c]
        f = offset_field(ia, sec, WINDOW, STEP, passes=2)
        good = np.isfinite(f.dx) & (f.peak >= MIN_PEAK)
        st = ring & good & (f.peak >= STABLE_PEAK)
        e = (f.dx - np.median(f.dx[st])) * PIX / yr
        n = -(f.dy - np.median(f.dy[st])) * PIX / yr
        ok = wscar & good
        r = dict(ref=a, sec=b, injected_m_yr=rate, recovered_east_m_yr=round(float(np.median(e[ok])), 2),
                 recovered_north_m_yr=round(float(np.median(n[ok])), 2),
                 scar_peak_ge_0_6=round(float(np.mean(f.peak[wscar] >= 0.6)), 2),
                 ring_peak_ge_0_6=round(float(np.mean(f.peak[ring] >= 0.6)), 2),
                 scar_median_peak=round(float(np.median(f.peak[wscar])), 2))
        out.append(r); print(r, flush=True)
(ROOT / "output/acceptance/injection_s2.json").write_text(json.dumps(out, indent=1))
