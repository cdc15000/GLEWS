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
| `unwrap_20m_snaphu.py WRAPPED_DIR OUT_DIR` | Unwraps the 20 m crops with SNAPHU (`pip install snaphu`; deformation cost, ~20 effective looks) and writes them in GUNW layout |
| `run_glews.py MIN_COH OUT_DIR [STACK_DIR]` | GLEWS loader (per-pair referencing to the stable ring within its connected component; `connectedComponents` 0 dropped; gaps per `GLEWS_GAP_POLICY`, default `break`) → acceleration map → detectors → report (`OUT_DIR` of `-` skips the report). No alerts are dispatched |
| `check_flags.py` | Flags inside or near the Figure 9 extent, flag timing by month, connected-component reliability |
| `diag_pairs.py GUNW_CROP_DIR` | Per-pair coherence, connected components and referenced source displacement |
| `plot_wrapped.py`, `diag_ls_unwrap.py`, `diag_profile.py` | 20 m wrapped-phase plots and two attempted unwrapping diagnostics (both inconclusive) |
| `extract_goff_crops.py OUT_DIR`, `diag_goff.py` | GOFF (pixel offset) crops, all three layers, and a per-pair source/ring table |
| `acceptance_goff.py [GOFF_DIR] [--sweep]` | Phase 1 acceptance test A1 (`docs/acceptance_phase1.md`); `--sweep` reports every layer × SNR setting |
| `map_goff.py [GOFF_DIR] [REF] [SEC] [LAYER]` | Map of one GOFF pair's range and along-track offsets with the source and scar outlines |
| `fetch_s2_crops.py [OUT_DIR] [ORBIT]` | Sentinel-2 L2A (collection 1) scene classification for every orbit-119 scene 2017–2026, then 10 m NIR crops for scenes with a clear source |
| `fetch_s2_scar.py [OUT_DIR]` | Matched pre/post-event scenes (21 Oct 2025, 21 Oct 2026) for mapping the failure scar |
| `acceptance_s2.py [S2_DIR]` | Pre-registered Sentinel-2 acceptance tests A2 and A3 |
| `terrain_rank_goff.py [GOFF_DIR]` | Phase 1b: GOFF source signal against terrain of the same slope and aspect (`docs/acceptance_phase1b.md`) |
| `register_wang_fig1b.py FIG1B_PNG` | Georeferences Wang et al. (2026) Fig. 1b against our 12 Aug 2026 image and traces their measurement area (stored in `config/nepal_2026.yaml`) |
| `acceptance_phase1c.py [S2_DIR] [GOFF_DIR]` | Phase 1c tests C1–C4 over that area (`docs/acceptance_phase1c.md`) |
| `injection_s2.py [S2_DIR]` | Injects 6 and 12 m/yr westward motion into the scar to check that the post-hoc re-test can see it |
| `posthoc_s2.py [S2_DIR]` | Post-hoc Sentinel-2 re-test over the mapped scar (labelled post hoc; see the acceptance doc) |
| `utm.py` | WGS84 UTM forward/inverse (no pyproj dependency) |

Run from the repository root, e.g.:

```bash
python scripts/nepal_2026/extract_gunw_crops.py data/nisar/nepal_2026/gunw_t98_crop
python scripts/nepal_2026/extract_wrapped_crops.py data/nisar/nepal_2026/gunw_t98_wrapped20m data/nisar/nepal_2026/gunw_t98_crop
python scripts/nepal_2026/run_glews.py 0.3 output/nepal_2026_nisar
python scripts/nepal_2026/unwrap_20m_snaphu.py data/nisar/nepal_2026/gunw_t98_wrapped20m data/nisar/nepal_2026/gunw_t98_20m_snaphu
python scripts/nepal_2026/run_glews.py 0.3 output/nepal_2026_nisar_20m data/nisar/nepal_2026/gunw_t98_20m_snaphu
glews dashboard -c config/nepal_2026.yaml --data-dir output/nepal_2026_nisar
# Phase 1 acceptance tests
python scripts/nepal_2026/extract_goff_crops.py data/nisar/nepal_2026/goff_t98_crop
python scripts/nepal_2026/acceptance_goff.py --sweep
python scripts/nepal_2026/fetch_s2_crops.py
python scripts/nepal_2026/fetch_s2_scar.py
python scripts/nepal_2026/acceptance_s2.py
python scripts/nepal_2026/posthoc_s2.py
python scripts/nepal_2026/map_goff.py
```

Phase 1 results are in `docs/acceptance_phase1.md`. With the fixed loader
(connected-component masking and per-pair referencing), no failed-slope
pixel survives the default `break` gap policy. The source cannot be
measured from GUNW phase.

Result with the loader before the Phase 1 fixes (no per-pair connected-component
masking; referencing after inversion), default coherence 0.3: 53 flags over the 144 km² crop; every pixel
in the failed-slope extent is unreliably unwrapped (connected component 0)
in most pairs; source LOS ≈ −5 cm versus ICIMOD's ~30 cm for the same
26 Jul–19 Aug pair. The standard GLEWS NISAR path does not single out the
source.

SNAPHU at 20 m does not change that. Only 0–30% of failed-slope pixels
join any connected component, and those that do rarely connect to stable
ground. Where the slope shares a component with ground 200–800 m outside
it, the relative motion is 0–2 cm per pair in winter and summer alike;
from March to May no component reaches the slope at all. The 20 m run
gives 16 flags over 36 km², none on the failed slope.
