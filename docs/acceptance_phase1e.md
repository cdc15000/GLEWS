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

*(added after the run)*
