# Gyirong–Rasuwa 2026 NISAR re-analysis

One-off scripts that re-ran GLEWS Tier 0 over the Langtang Lirung source
(28.288°N, 85.524°E, read from ICIMOD 2026 Figure 9) using NISAR GUNW
products from ascending track 98.

Products are read with HTTP range requests, so only a small crop of each
~2.4 GB file is fetched. Authentication uses `~/.netrc` (Earthdata Login).
Crops land in `data/nisar/nepal_2026/` (gitignored).

| Script | Purpose |
|--------|---------|
| `extract_gunw_crops.py OUT_DIR` | 12 × 12 km crops of the 80 m unwrapped layers for every pre-event track-98 pair (co-event pairs dropped, one product per date pair) |
| `extract_wrapped_crops.py OUT_DIR GUNW_CROP_DIR` | 6 × 6 km crops of the 20 m wrapped interferogram for the same pairs |
| `run_glews.py MIN_COH OUT_DIR` | GLEWS loader → stable-ring referencing → acceleration map → detectors → report (`OUT_DIR` of `-` skips the report). No alerts are dispatched |
| `check_flags.py` | Flags inside or near the Figure 9 extent, flag timing by month, connected-component reliability |
| `diag_pairs.py GUNW_CROP_DIR` | Per-pair coherence, connected components and referenced source displacement |
| `plot_wrapped.py`, `diag_ls_unwrap.py`, `diag_profile.py` | 20 m wrapped-phase plots and two attempted unwrapping diagnostics (both inconclusive) |
| `utm.py` | WGS84 UTM forward/inverse (no pyproj dependency) |

Run from the repository root, e.g.:

```bash
python scripts/nepal_2026/extract_gunw_crops.py data/nisar/nepal_2026/gunw_t98_crop
python scripts/nepal_2026/extract_wrapped_crops.py data/nisar/nepal_2026/gunw_t98_wrapped20m data/nisar/nepal_2026/gunw_t98_crop
python scripts/nepal_2026/run_glews.py 0.3 output/nepal_2026_nisar
glews dashboard -c config/nepal_2026.yaml --data-dir output/nepal_2026_nisar
```

Result (default coherence 0.3): 53 flags over the 144 km² crop; every pixel
in the failed-slope extent is unreliably unwrapped (connected component 0)
in most pairs; source LOS ≈ −5 cm versus ICIMOD's ~30 cm for the same
26 Jul–19 Aug pair. The standard GLEWS NISAR path does not single out the
source.
