# Phase 1d criteria: empirical noise and the main-glacier control (pre-registered)

Written on 24 Oct 2026 and committed **before** either test was run.
Results go in a *Results* section added afterwards; nothing above it is
edited.

## Why

1. Phase 1c's C2 failed on a noise estimate (per-window σ of stable
   ground) that ignores averaging over the area. The 12 summer
   combinations agreed far more closely than that σ allowed. D1 replaces it
   with an **empirical null for the statistic actually used**. This makes
   D1 a re-test of C1/C2 with a different noise model, specified before
   running.
2. The GOFF acceleration over the measurement area (C4) could partly be
   leakage from Wang et al.'s main glacier next to it. They report the
   main glacier moving at a steady 42.2 m/yr. D2 measures it.

## Prior knowledge (disclosed)

- The C1/C2 area speeds and the C4 GOFF series for the measurement area
  are known (`acceptance_phase1c.md`).
- From C4, the area's GOFF slant-range rate ratio, (27 May – 19 Aug) to
  (28 Nov – 15 Jan), is ≈ 3.4.
- Nothing has been measured over the main-glacier box.

## Inputs

- **Areas:** `site.reference_areas` in `config/nepal_2026.yaml`:
  - [0] the measurement area (Fig. 1b);
  - [1] the main-glacier box (Fig. 2a, 427 × 250 m);
  - [2] the stable box (Fig. 2a), not used in the criteria.
- **Sentinel-2:** as Phase 1c primary: 16 px windows every 2 px, corrected
  scene selection, high-pass σ 8 px, 2 passes, window correlation ≥ 0.2,
  referenced to stable-ring windows with correlation ≥ 0.6.
- **GOFF:** as A1/C4: layer 2, SNR ≥ 5, slant range, ring-referenced.

## Empirical noise model (used by D1)

For each Sentinel-2 pair:

1. Translate the measurement area's 166 window centres by 100 random
   offsets (seed 0). Keep a translation only if every centre falls in the
   stable mask: 1.5–5 km from the source centre, outside the ICIMOD outline
   dilated by 300 m.
2. Compute the area statistic for each copy: the median east and north
   offset over windows with correlation ≥ 0.2, and keep the copy only if
   ≥ 50 % of its windows have correlation ≥ 0.6.
3. Take σ_e and σ_n as 1.4826 × MAD of the copies' medians, and
   σ_patch = √(σ_e² + σ_n²), converted to m/yr.

The pair is used only if ≥ 30 copies qualify. Each pair also reports the
area speed's percentile rank among the copies' speeds.

## D1. Re-test of C1 and C2 with the empirical noise model

- **D1a (pre-2026 rate):** spring year pairs 2018→19 … 2024→25, as C1.
  The annual speed is the median over combinations, and annual σ is the
  median σ_patch. v1 and σ1 are the medians across years. Pass if
  3.7 ≤ v1 ≤ 14.8 m/yr **and** v1 ≥ 2 σ1.
- **D1b (2026 acceleration):** the 12 C2 combinations. v2 is the median
  speed, σ2 the median σ_patch. Pass if v2 ≥ 2 v1 **and**
  (v2 − v1) ≥ 3 √(σ1² + σ2²).

## D2. Main-glacier control (GOFF)

Rate = sum of per-pair median slant-range offsets over the box ÷ days, for
the same periods as C4. Define R = rate(27 May – 19 Aug) ÷
rate(28 Nov – 15 Jan) for the box (R_main) and for the measurement area
(R_area, from the same run).

**Pass ("the acceleration belongs to the outlet, not the main glacier")**
if **both**:

1. R_main ≤ 1.5;
2. R_area ≥ 2 × R_main.

Also reported, without pass/fail:

- the main glacier's rate converted to horizontal under the slope-parallel
  assumption, against Wang et al.'s 42.2 m/yr;
- the correlation between the two areas' per-pair series.

## Results

*(added after the run, 24 Oct 2026; `scripts/nepal_2026/acceptance_phase1d.py`,
`output/acceptance/D_phase1d.json`)*

### D2 (main-glacier control): PASS

GOFF slant-range rate (m/yr) by period:

| Area | Pixels | 28 Nov–15 Jan | 15 Jan–28 Mar | 28 Mar–27 May | 27 May–26 Jul | 26 Jul–19 Aug | R |
|---|---|---|---|---|---|---|---|
| Measurement area (outlet) | 9 | 16.5 | 36.9 | 35.4 | 54.3 | 60.2 | **3.39** |
| Main-glacier box | 16 | 15.7 | 14.2 | 14.3 | 19.9 | 24.7 | **1.36** |

R_main = 1.36 ≤ 1.5, and R_area = 3.39 ≥ 2 × R_main. The per-pair series
of the two areas correlate weakly (r = 0.24).

The main glacier is steady from November to May with a modest summer
rise. The outlet's mid-January step and continued acceleration are not
leakage from it. The main glacier averages 16.7 m/yr in slant range, or
≈ 27 m/yr horizontal under the slope-parallel assumption (range
sensitivity 0.61 on its slope). Wang et al. give 42.2 m/yr: the same order,
lower here.

### D1 (empirical noise model): D1a FAIL, D1b FAIL

There are 100 stable copies of the area. The empirical σ of the area
statistic is 1.9–3.2 m/yr for year pairs, against 3.8–8.1 m/yr from
per-window σ in C1.

| Year pair | Pairs | Speed (m/yr) | σ_patch | Rank among stable copies |
|---|---|---|---|---|
| 2018→19 | 4 | 4.0 | 2.7 | 62 % |
| 2019→20 | 56 | 3.7 | 2.2 | 71 % |
| 2020→21 | 63 | 3.1 | 2.1 | 65 % |
| 2021→22 | 12 | 1.6 | 2.7 | 32 % |
| 2022→23 | 10 | 1.6 | 3.2 | 24 % |
| 2023→24 | 54 | 2.8 | 2.6 | 57 % |
| 2024→25 | 56 | 5.7 | 1.9 | 75 % |

- **D1a:** v1 = 3.1 m/yr, σ1 = 2.6. That is below the 3.7 floor and
  below 2 σ1.
- **D1b:** v2 = 34.6 m/yr, σ2 = 14.5. v2 ≥ 2 v1, but
  (v2 − v1) / combined σ = **2.13**, short of 3.
- **Decisive detail:** the area's 2026 speed ranks at only the 50th–89th
  percentile among stable copies in the same 12 summer pairs. Stable ground
  shows apparent speeds of the same size in those pairs. The consistency
  across the 12 combinations noted in C2 reflects shared scenes (every
  pair ends on 12 or 24 Aug), not a distinguishable signal.

**Conclusion:** at 10 m, Sentinel-2 does not separate the measurement
area's motion from noise, before 2026 or in 2026. Year-pair ranks are
24–75 %, and summer-2026 ranks 50–89 %. GLEWS's evidence for the
precursor rests on NISAR GOFF (C4, D2), not on Sentinel-2.

### Phase 1 overall (A–D)

| Evidence | Status |
|---|---|
| NISAR GOFF over the whole failed slope (A1) | Pass vs the crop; source moving in all 21 pairs (1b post hoc), not a slope-geometry bias |
| NISAR GOFF over Wang et al.'s outlet area (C4) | 16.5 → 37 m/yr slant range at mid-January 2026, 54–60 m/yr by Jul–Aug; 28 m in 264 days (reported) |
| Main-glacier control (D2) | **Pass:** the acceleration is specific to the outlet |
| Sentinel-2 offsets (A2, A3, C1, C2, D1) | Fail: not separable from noise at this scale and season |
| NISAR GUNW phase | Cannot measure the source |
