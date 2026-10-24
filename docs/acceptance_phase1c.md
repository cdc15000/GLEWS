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

*(added after the run, 24 Oct 2026; `scripts/nepal_2026/acceptance_phase1c.py`,
`output/acceptance/C_phase1c.json`, `C_phase1c.png`)*

308 scenes passed the corrected selection, including 3, 5 and 12 June and
12 and 24 August 2026. The area holds 166 window centres at 16 px and 41
at 32 px.

### C1 (pre-2026 rate): FAIL

| Year pair | Pairs | 16 px (m/yr) | σ | 32 px (m/yr) | σ |
|---|---|---|---|---|---|
| 2018→19 | 4 | 4.0 | 8.1 | 6.6 | 3.7 |
| 2019→20 | 57 | 3.7 | 5.5 | 6.3 | 3.6 |
| 2020→21 | 63 | 3.1 | 4.8 | 3.6 | 3.7 |
| 2021→22 | 12 | 1.6 | 5.1 | 1.6 | 3.5 |
| 2022→23 | 10 | 1.6 | 7.1 | 6.1 | 4.9 |
| 2023→24 | 54 | 2.8 | 6.1 | 3.0 | 4.3 |
| 2024→25 | 56 | 5.7 | 3.8 | 9.3 | 2.7 |
| **Median** | | **3.1** | **5.5** | **6.1** | **3.7** |

- Primary (16 px): 3.1 m/yr, below the 3.7–14.8 band and below 2 σ.
- Secondary (32 px): 6.1 m/yr is inside the band but below 2 σ (7.3).
- Directions are mostly 280–320°, close to the area's downslope direction
  (285°). The 2024→25 pair at 32 px is 9.3 m/yr (3.4 σ).

Sentinel-2 sees a rate of the order of Wang et al.'s 7.4 m/yr. Year-pair
noise of 3–8 m/yr keeps it below the pre-registered significance.

### C2 (2026 acceleration): FAIL

12 combinations (ref 4 May–5 Jun, sec 12 or 24 Aug 2026):

- **16 px:** median 34.6 m/yr, all 12 combinations between 23 and 42 m/yr,
  directions 210–292°. That is 11 × C1, but only 0.5 combined σ.
- **32 px:** median 25.8 m/yr (4.2 × C1), 0.8 σ.

The criterion uses per-window ring σ converted to m/yr. For 70–112-day
summer pairs that is 14–120 m/yr, so even a tenfold rate increase does
not reach 3 σ. The 12 combinations agree with each other far more closely
than that σ implies, but the criterion did not use their spread.

### C3 (cross-sensor): PASS, but weak

| | GOFF slant range (m/yr) | S2 projected to range (m/yr) |
|---|---|---|
| Median over the 12 C2 pairs | 54.4 (± 0.9) | 29.0 (± 14–77) |

All 12 agree in sign and lie within 2 combined σ. The pass comes from the
large Sentinel-2 σ, not from close agreement: GOFF is about twice the
projected Sentinel-2 rate. 16 px windows straddle the 150 m-wide strip and
slower ground, which dilutes the optical signal.

### C4 (GOFF rate history over the area, reported)

The area covers 9 GOFF pixels (80 m); 4–9 are valid per pair. All 21
pairs are positive (away from the radar). In total there is **28.1 m of
slant range over 264 days**.

| Period (2025–26) | Pairs | Slant-range rate (m/yr) | ≈ Horizontal, slope-parallel (m/yr) |
|---|---|---|---|
| 28 Nov – 15 Jan | 4 | 16.5 ± 1.6 | ~19 |
| 15 Jan – 28 Mar | 6 | 36.9 ± 1.7 | ~43 |
| 28 Mar – 27 May | 5 | 35.4 ± 2.8 | ~41 |
| 27 May – 26 Jul | 5 | 54.3 ± 1.1 | ~63 |
| 26 Jul – 19 Aug | 1 | 60.2 ± 1.4 | ~70 |

σ is the ring per-pixel σ combined over pairs, so it is conservative for
a median over several pixels. "Horizontal" divides by 0.86, the slant-range
sensitivity to slope-parallel motion toward 285° on this 22° slope.

Reading (descriptive; C4 had no pass/fail):

- The rate **more than doubles at mid-January 2026** and climbs again in
  June–August. This matches Wang et al.'s January onset and their 55 m/yr
  2026 rate.
- Unlike the whole-outline series (Phase 1b), there is no spring-only
  pulse here. The rise is sustained.
- Caveat: Wang et al.'s steadily moving main glacier (42 m/yr) borders
  the area. GOFF's matching windows are much larger than 80 m, so part of
  the winter baseline may be leakage from it. The step change and the
  continued rise cannot come from a steady glacier.

### Summary

| Test | Result |
|---|---|
| C1 S2 pre-2026 rate | FAIL: 3.1 (16 px) and 6.1 (32 px) m/yr, right order of magnitude, < 2 σ |
| C2 S2 2026 acceleration | FAIL: 26–35 m/yr in summer 2026 (4–11 × C1), consistent across 12 combinations, < 1 σ by the pre-registered σ |
| C3 S2 vs GOFF | PASS (weak): same sign, within 2 σ; GOFF ≈ 2 × projected S2 |
| C4 GOFF history | Reported: 16.5 → 37 m/yr slant range at mid-January, rising to 54–60 m/yr by July–August |
