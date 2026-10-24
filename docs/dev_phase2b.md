# Phase 2b (development): magnitude and ratio gates on E2

**Development only. No pass/fail.** Written 24 Oct 2026 after the run.
Script `scripts/nepal_2026/dev_gates.py`, output
`output/acceptance/dev_gates.json`.

## Purpose

Phase 2 left one cheap candidate for specificity: gate E2 on the size of
the rate and on its ratio to the pixel's own Nov–Jan baseline. Before any
fresh control sites are chosen and extracted, this sweep asks whether
*any* gate setting works **in-sample**. If none does, the gates are not
worth an out-of-sample test.

## Development set

- Nepal, Ganesh East and Jugal Himal. The two Phase 2 controls were seen
  in Phase 2, so they are now development data.
- **Development criterion (go / no-go for fresh controls):**
  - both Ganesh East and Jugal have ≤ 2 false-alarm flags;
  - Nepal has a persistent hit with a lead of ≥ 31 days.

## Gates swept (on top of E2: z ≥ 3 in ≥ 3 of the last 4 pairs)

- **Magnitude M:** |r| ≥ M m per 12 days (slant range, 3×3-smoothed),
  0–2.0.
- **Ratio K, direction-aware:** s·r ≥ K·max(|b|, b_floor), where:
  - b is the Nov–Jan baseline rate (the same window as D2);
  - b_floor is the median ring noise over the baseline pairs (Nepal: 0.10);
  - s = sign(b) if |b| ≥ b_floor, otherwise sign(r).
  - K ranges 0–4.
- **Added after the first sweep** (see below):
  - **blunder cap CAP:** a window is rejected if any rate in it exceeds CAP
    (3, 5 or 10 m per 12 days, i.e. ≈ 90–300 m/yr);
  - **sign consistency:** every valid rate in the window has the alarm's
    sign.

## Findings

1. **Magnitude and ratio alone do nothing useful.** Even at M = 2 m per
   12 days (≈ 60 m/yr, the outlet's own peak), false alarms stay at 26
   (Ganesh East) and ~50 (Jugal). Raising K mainly costs Nepal its lead.
2. **Most residual false alarms are offset blunders, not glacier
   physics.** At M = 2:
   - 29–44 % of the false-alarm pixel-pair rates exceed 10 m per 12 days
     (> 300 m/yr);
   - the median false-alarm pixel changes sign 9 times in 20 pairs.

   Mismatched correlation windows survive both SNR ≥ 5 and the 3×3 median.
3. **The blunder vetoes help a lot, but not enough.** The best settings
   (CAP 3–5, sign-consistent, M 1.0–1.2, K 2) give:

   | | False alarms, Nepal | Ganesh East | Jugal Himal | Nepal lead |
   |---|---|---|---|---|
   | E2 (Phase 2) | 52 | 56 | 49 | 187 d |
   | Best gated | 3–4 | 1–2 | **9** | **43–55 d** |

   Ganesh East reaches the limit. Jugal does not, at any setting
   (minimum 9). The lead collapses from 187 to 43–55 days, because the
   outlet's per-pixel rate (median 0.9–1.6 m per 12 days, March–June)
   sits right at the thresholds needed. Its true margin over M is small.
4. **No setting meets the development criterion.** Gates on magnitude,
   ratio and blunders do not reach the operational limit even in-sample.
   Following the plan, no fresh controls were chosen or extracted.

## What it suggests (not tested)

- The blunder veto is the one component that clearly works (≈ 5–25× fewer
  false alarms). It belongs in the GOFF loader as a quality step, not in a
  detector.
- What remains at Jugal after the veto needs inspection (fast icefalls?
  real seasonal speed-up?) before choosing a next filter. Separating it
  from a precursor probably needs the multi-year same-season baseline
  (Sentinel-1, 2017–2025), or restricting the watched area to
  failure-relevant terrain above exposure.
- The lead-time cost is real: every setting that cuts false alarms enough
  also cuts the Nepal lead to under two months.
