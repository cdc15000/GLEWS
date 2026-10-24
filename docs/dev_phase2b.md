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
  false alarms).
- What remains at Jugal after the veto needs inspection (fast icefalls?
  real seasonal speed-up?) before choosing a next filter. Separating it
  from a precursor probably needs the multi-year same-season baseline
  (Sentinel-1, 2017–2025), or restricting the watched area to
  failure-relevant terrain above exposure.
- The lead-time cost is real: every setting that cuts false alarms enough
  also cuts the Nepal lead to under two months.

## Follow-up (development): blunder rejection in the loader, and what remains

*(24 Oct 2026; `scripts/nepal_2026/diag_residuals.py`,
`output/acceptance/diag_residuals.json`, `diag_residuals.png`)*

**Loader change.** `read_goff_pairs` and `load_nisar_goff_stack` take a
`max_rate` (metres per day, **off by default**, so earlier results are
unchanged):

- A pixel is masked in both components when hypot(range, along-track)
  of its referenced offset, divided by the pair length, exceeds
  `max_rate`.
- `n_blunders` records the count per pair.
- At 5/12 m per day (≈ 150 m/yr), it rejects a median of 350–580 of the
  22,500 pixels per pair.
- **Why it stays off by default:** a cap on magnitude alone would also
  mask the final, fast acceleration of a real failure. Rates just before
  a collapse can far exceed 150 m/yr. The blunders are both large *and*
  sign-flipping. The long-term loader filter should test consistency
  instead: deviation from neighbouring pixels, or repeated direction
  reversals.

**Settings.** The best Phase 2b gate (sign-consistent window, M = 1.0,
K = 2), now fed by the loader's blunder rejection:

| | False-alarm clusters | Nepal target first flagged |
|---|---|---|
| Nepal | 3 | 2 Jul (lead 55 d) |
| Ganesh East | 2 | — |
| Jugal Himal | 10 | — |

**What the residual false alarms are:**

- **Snow-covered.** The Sentinel-2 SCL snow class (Dec 2025, which
  includes seasonal snow) covers almost every cluster. On the true-colour
  image the Jugal clusters sit on the large valley glaciers.
- **Gentle.** At Jugal the median slope is 9–17°, glacier tongues rather
  than steep faces.
- **Real, downslope motion.** In most Jugal clusters, 54–100 % of alarm
  pixels move within 45° of the downslope direction.
- **The textbook seasonal signal.** Rates are ~13–17 m/yr (slant range)
  Nov–Jan, flat through March, then rise to 32–42 m/yr in Jun–Aug
  (≈ 2.5–3×). The rise starts with the melt season in May–June.

The two Ganesh East clusters are the exception: small (0.05–0.13 km²),
on steeper ice (33–35°), and their rate swings 0–60 m/yr from pair to
pair. That looks more like residual noise than seasonal flow.

So after blunder rejection, the false alarms are mostly **ordinary summer
glacier speed-up**, not artefacts. A single-season magnitude or ratio gate
cannot separate them, because their summer ratio (2.5–3×) matches the
precursor's.

**Where the Nepal target differs (observation; biased, see the next
section).** The target's cluster rate rises **before** the melt season:

- the step to ~30 m/yr is in mid-February;
- it stays at 22–32 m/yr through March–May;
- meanwhile the *flagged* control glaciers stay near their winter rate
  until June.

This matches C4/D2 (the outlet doubles at mid-January). Two candidates
follow, both needing fresh controls:

- **Out-of-season acceleration:** flag acceleration only against what is
  expected for the time of year. The rigorous version needs multi-year
  same-season baselines (Sentinel-1, 2017–2025). A crude single-season
  version would flag only winter–spring acceleration (before ~May), at
  the cost of missing precursors that begin in summer.
- **Failure-relevant terrain:** exclude gentle valley-glacier tongues.
  Caution: the Nepal hit cluster's own median slope is 19°, so a slope
  floor needs care.

Neither has been tried. Each should be set up on this development set
and then tested on fresh, pre-registered controls.

## Unbiased check: is pre-melt acceleration unusual among all moving pixels?

*(24 Oct 2026; `scripts/nepal_2026/diag_premelt.py`,
`output/acceptance/diag_premelt.json`. Post hoc: the probe areas were
chosen knowing the outcome.)*

The comparison above is biased by selection. Its control series are only
clusters the gate (|r| ≥ ~30 m/yr) had already picked, so a control
glacier that accelerated modestly in February could not appear.

**Method.** Use every pixel with a moving baseline (|b| ≥ the ring noise,
~2–3 m/yr), with blunder rejection on:

- **ratio:** s · median rate (15 Jan – 27 May) ÷ |b|;
- **increase:** the same difference, in m/yr slant range.

The Nepal probe value is a median over its pixels. The controls are
single pixels, so their distributions are wider, which makes the probe
look *less* extreme.

| Probe (median) | Ratio | Increase | Ganesh East: pctl ratio / increase, px ≥ both | Jugal: pctl, px ≥ both | Rest of Nepal crop: pctl, px ≥ both |
|---|---|---|---|---|---|
| Outlet (Wang area, 8 px) | 1.98 | +15.0 m/yr | 96.7 / 100.0, **0** of 2999 | 95.9 / 99.8, **6** of 3731 | 96.5 / 99.8, **5** of 2032 |
| Whole ICIMOD outline (69 px) | 1.45 | +6.0 m/yr | 92.4 / 97.8, 63 | 88.3 / 95.6, 131 | 92.0 / 97.5, 50 |

Restricting to snow-covered pixels gives the same picture.

**Readings:**

- **The outlet's pre-melt acceleration is in the far tail at both
  controls.** Almost no moving pixel anywhere both doubled and gained
  ≥ 15 m/yr before the melt season. The whole failed outline is not
  distinctive: its ordinary parts dilute the signal.
- **So the out-of-season route is worth one honest test.** The terrain
  restriction alone would not have removed the summer glaciers.
- **The tension is real and must be stated.**
  - Any threshold that passes the outlet sits at or below ratio 2 and
    +15 m/yr, so it was found by looking at the answer.
  - The outlet's Feb–May rate (~0.7–1.0 m per 12 days) is below the
    M = 1.0 that suppressed the summer false alarms. The Phase 2b sweep
    with a lower M (CAP 5, sign-consistent, M 0.5, K 2) gave 6 / 5 / 22
    false alarms with a 79-day lead.
  - A pre-melt detector must use an area-scale statistic (the signal
    needs ~8 pixels to clear the pixel scatter) and a window that closes
    before the melt-season rise. The window closes in late May, which
    caps the lead at about three months unless the detector runs
    cumulatively from January.
