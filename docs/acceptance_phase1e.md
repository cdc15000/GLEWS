# Phase 1e criteria: would GLEWS have raised a flag in time? (pre-registered)

Written on 24 Oct 2026 and committed **before** either replay was run.
Results go in a *Results* section added afterwards; nothing above it is
edited.

## Question

Replay the NISAR GOFF data as it arrived, one 12-day pair at a time, using
only data available at each date. Would a GLEWS detector have flagged the
failed slope, and how early? How many flags would it have raised
elsewhere in the 12 × 12 km crop?

## Prior knowledge (disclosed)

- From C4/D2, the outlet area's GOFF rate steps up at mid-January 2026 and
  again in June–August.
- From Phase 1b, the whole outline is positive in every pair, with pulses
  in late March–May and late July–August.
- **Not known:** how either detector behaves on the full crop, or how
  many flags it raises elsewhere.

## Common inputs

- **GOFF:** track 98, the 21 pre-event pairs (28 Nov 2025 → 19 Aug 2026).
  Layer 2, SNR ≥ 5, slant-range offset. Each pair is referenced to the
  stable ring: 2–5 km from the source centre, ICIMOD outline excluded.
- **Target:** the ICIMOD outline ∪ Wang et al.'s measurement area
  (`known_source` and `reference_areas[0]` in `config/nepal_2026.yaml`).
  A flag "hits" if any of its pixels falls in the target.
- **Cutoffs:** the secondary date of each pair, from the 6th pair
  (20 Feb 2026) to the last (19 Aug 2026). At each cutoff only pairs ending
  on or before it are used.
- **Lead time:** 26 Aug 2026 minus the first cutoff with a hit. The hit
  must persist: a hit at every later cutoff, allowing at most one miss.
- **False alarms:** flags that do not hit the target.

## E1. GLEWS's existing pipeline, unchanged

At each cutoff:

1. Load the stack with `load_nisar_goff_stack` (defaults: layer 2,
   SNR ≥ 5, `gap_policy="break"`), with the ring as `reference_mask`.
2. Run `compute_acceleration_map` and `detect_anomalies` with the unchanged
   `detect` settings in `config/nepal_2026.yaml`, passing dates and
   displacement so the step-change detector runs too.

**Pass** if **all** hold:

1. a persistent hit starts at a cutoff ≤ 26 Jul 2026 (lead ≥ 31 days);
2. at that first cutoff, the hitting flag ranks in the top 3 by score;
3. at the 26 Jul cutoff there are ≤ 10 false-alarm flags.

Also reported: false alarms per cutoff, and the hit's rank at every cutoff.

## E2. Rate-change detector for offset streams (specified here)

Per pixel, per pair k:

1. **Rate:** r_k = referenced slant-range offset × 12 / days (m per 12
   days), smoothed by a 3 × 3 nan-median (240 m).
2. **Noise:** σ_k = 1.4826 × MAD of the smoothed r_k over ring pixels.
3. **Baseline:** b = median of r over the first 4 pairs (28 Nov 2025 →
   15 Jan 2026), needing ≥ 3 valid. Its uncertainty is
   σ_b = 1.2533 × median(σ over those pairs) / √(n valid).
4. **Score:** z_k = |r_k − b| / √(σ_k² + σ_b²).
5. **Alarm pixel at cutoff k:** z ≥ 3 in at least 3 of the last 4 pairs
   (pairs 5 onward only).
6. **Flag:** a connected group (8-connectivity) of ≥ 4 alarm pixels,
   at least 0.0256 km².

**Pass** if **both** hold:

1. a persistent hit starts at a cutoff ≤ 26 Jul 2026 (lead ≥ 31 days);
2. **false-alarm burden:** the number of distinct false-alarm flags over
   all cutoffs is ≤ 5. A false-alarm flag is a connected group of the
   union of non-hitting alarm pixels over all cutoffs.

Also reported: false-alarm area (km²) and count per cutoff, and the first
cutoff with any hit, persistent or not.

## Results

*(added after the run, 24 Oct 2026; `scripts/nepal_2026/replay_goff.py`,
`output/acceptance/E_replay.json`, `E_replay.png`)*

### E1 (existing GLEWS pipeline): FAIL, on false alarms

| Cutoff | Flags | Target's rank | False alarms |
|---|---|---|---|
| 8 Feb | 5 | — | 5 |
| 20 Feb | 5 | — | 5 |
| 4 Mar | 7 | — | 7 |
| **16 Mar** | 12 | **7** | 11 |
| 28 Mar | 112 | 7 | 110 |
| 9 Apr | 119 | 6 | 117 |
| 21 Apr | 94 | **2** | 93 |
| 3 May | 90 | **1** | 88 |
| 15 May | 75 | 2 | 72 |
| 27 May | 82 | 3 | 80 |
| 8 Jun | 64 | 2 | 62 |
| 20 Jun | 73 | 3 | 71 |
| 2 Jul | 76 | 5 | 74 |
| 14 Jul | 87 | 3 | 85 |
| 26 Jul | 86 | 5 | 84 |
| 19 Aug | 82 | 3 | 80 |

- Criterion 1 passes: a persistent hit from 16 Mar, lead **163 days**.
- Criterion 2 fails: the target ranks 7th at that first cutoff, not top 3.
- Criterion 3 fails: 84 false-alarm flags at 26 Jul, against a limit of 10.

Descriptive only (not a pre-registered metric): from 21 Apr onward the
target ranks in the **top 3 at 8 of 10 cutoffs** (5th at 2 Jul and 26 Jul),
and **1st on 3 May**. The flag count jumps from 12 to 112 at the
28 Mar cutoff, when the first snowmelt-season pairs enter the stack.

### E2 (rate-change detector): FAIL, on false alarms

- Criterion 1 passes: a persistent hit from 20 Feb (the first cutoff
  where alarms are possible), lead **187 days**. The target's alarm area
  grows from 20 pixels (Feb) to 78 (early July).
- Criterion 2 fails: **52 distinct false-alarm flags** over all cutoffs,
  covering 15.0 km² (10 % of the crop), against a limit of 5.
- False-alarm area per cutoff rises from 3.9 km² (Feb) to 8.2 km² (Aug).
  The false alarms fall on other glaciers, steep faces and snow-covered
  slopes across the crop (`E_replay.png`). One large cluster sits just
  north-west of the outline.

### Conclusion

On the data as it arrived, both detectors flag the failed slope
**early and persistently**: from mid-March (E1, 163 days) or mid-February
(E2, 187 days). Both also flag tens of other places in the same 144 km².
GLEWS's existing pipeline ranks the slope among its top three flags for
most of the final four months, so an analyst reviewing the top few flags
would have seen it. That review rule was not pre-registered.

As specified, neither detector meets the false-alarm limits. The
precursor is **detectable but not yet specific**. The next work is
specificity: the snowmelt-season jump in E1, and the glacier and snow
false alarms in E2. That is Phase 2 (false-alarm study) territory, now
with a known true positive to test against.
