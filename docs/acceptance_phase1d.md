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

*(added after the run)*
