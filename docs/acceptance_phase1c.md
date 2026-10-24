# Phase 1c criteria: Wang et al.'s measurement area (pre-registered)

Written on 24 Oct 2026 and committed **before** any offsets were computed
over this area. Results go in a *Results* section added afterwards; nothing
above it is edited.

## Why

Wang et al. (2026, Landslides 23:3489, doi:10.1007/s10346-026-02842-6)
report their > 100 m for a small "measurement area" (Fig. 1b): the outlet
hanging glacier on the north-east edge of the failed slope. Fig. 2c gives
**7.4 m/yr from 2018 to late 2025, rising to 55 m/yr from January 2026**.
Phases 1 and 1b measured the whole failed slope, which dilutes a small,
fast patch. This phase tests GLEWS over their area.

## Prior knowledge (disclosed)

- Exploratory Sentinel-2 maps (`acceptance_phase1.md`, A2 diagnosis) already
  showed motion along the south-east edge of the ICIMOD outline in clean
  spring pairs. The triangle was not delineated from those maps.
- The NISAR GOFF per-pair source values (Phase 1b) are known for the whole
  outline, not for the triangle.

## Inputs

- **Area:** `site.reference_areas[0]` in `config/nepal_2026.yaml`. It was
  traced from Wang et al. Fig. 1b after registering the panel to our own
  12 Aug 2026 Sentinel-2 image (`scripts/nepal_2026/register_wang_fig1b.py`;
  NCC 0.75, 2.1 m per figure pixel). ~0.067 km², about 150 m × 550 m.
- **Sentinel-2:**
  - Scenes: orbit 119, collection 1, band 8, with scene selection
    corrected. A scene is excluded only if cloud plus cloud shadow covers
    more than 30 % of the source (SCL classes 3, 8, 9, 10). Class 2 "dark
    area" is recorded but not used to exclude scenes; see
    `fetch_s2_crops.py`.
  - Matching: high-pass σ 8 px, 2 passes. Windows are **16 px (160 m)
    every 2 px** (primary, since the area is ~150 m wide), with 32 px
    every 4 px reported as secondary.
  - Quality: window correlation ≥ 0.2. Stable reference = windows
    1.5–5 km from the source centre, outside the ICIMOD outline dilated
    by 300 m, with correlation ≥ 0.6. σ = 1.4826 × MAD of the referenced
    stable offsets (east and north combined).
  - **Signal:** magnitude of the median offset vector over windows whose
    centre lies inside the area. A pair needs ≥ 5 such windows.
- **GOFF:** layer 2, SNR ≥ 5, slant range, referenced to the stable ring
  (as in A1). Signal: median over 80 m pixels inside the area.

## C1. Pre-2026 rate (Wang: 7.4 m/yr)

All spring year-apart pairs (Mar–May, partner within ±15 days of day of
year), 2018→2019 through 2024→2025. For each year pair, take the median
speed over its combinations. Pass if **both**:

1. the median of those annual speeds is between 3.7 and 14.8 m/yr (a
   factor of two of 7.4);
2. it is ≥ 2 × the median annual σ.

## C2. 2026 acceleration (Wang: 55 m/yr from January 2026)

Pairs with reference scene 1 May–15 Jun 2026 and secondary scene
1 Aug–25 Aug 2026, every combination. Speed = median over combinations.
Pass if **both**:

1. the 2026 speed is ≥ 2 × the C1 rate;
2. (2026 speed − C1 rate) ≥ 3 × combined σ, where the σs are converted to
   m/yr and combined in quadrature.

If no such pair has ≥ 5 valid windows: **not testable**.

## C3. Cross-sensor consistency (replaces A4)

For each C2 Sentinel-2 pair, take the GOFF pairs whose dates best cover
the same interval (sum of consecutive GOFF pairs from the GOFF date
nearest the S2 reference to the GOFF date nearest the S2 secondary). Then:

1. Scale both to m/yr.
2. Project the S2 east/north rate into slant range: assume motion parallel
   to the slope, using the GLO-30 gradient over the area and the NISAR LOS
   at 5,000 m (`output/acceptance/nisar_los_t98.json`).
3. Compare with GOFF's range rate.

Pass if they agree in sign and are within 2 combined σ, using the median
over combinations. GOFF σ = ring σ of each pair combined in quadrature.

## C4. Rate history (reported, no pass/fail)

The GOFF per-pair range rate over the area, 28 Nov 2025 → 19 Aug 2026,
alongside the Phase 1b whole-outline series.

## Results

*(added after the run)*
