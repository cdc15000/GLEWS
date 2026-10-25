# Phase 2d (development): can ITS_LIVE Sentinel-1 provide a multi-year baseline?

**Development only. No pass/fail.** Written 24 Oct 2026.

- Script: `scripts/nepal_2026/dev_s1_checks.py`, output
  `output/acceptance/dev_s1_checks.json`.
- The area-aggregated variant was run inline from the same functions. Its
  numbers are quoted below.

## Purpose

Phase 2c showed that pre-melt acceleration alone is not rare: E4 raised 12
and 4 false alarms at the fresh controls.

The remaining idea is a **pixel's own history**. Was a pre-melt speed-up
normal for that glacier in earlier years? This is a feasibility check
before building anything, on the ready-made ITS_LIVE archive:

- **Product:** NASA ITS_LIVE v2 datacubes (autoRIFT, 120 m grid).
- **Data used:** Sentinel-1 pairs only, 2015–2022.
- **Sites:** only those already seen (Nepal, Ganesh East, Jugal Himal,
  Annapurna Sanctuary, Langtang East).
- **Held out and untouched:** the GOFF and the ITS_LIVE data of Manaslu
  West, Lapchi and Annapurna IV north.

## Data available

- **Cubes:**
  - EPSG:32645 X350000 Y3150000 covers Nepal and the zone-45 controls;
  - EPSG:32644 X750000 covers Annapurna Sanctuary.
- **Sentinel-1 pairs at the outlet:**
  - 1013 pairs listed, 443 with a valid velocity;
  - they run from Oct 2014 to Dec 2022, 6- or 12-day separation, none in
    2023–2024 in this cube version;
  - roughly 30 valid pairs per 15 Jan – 27 May window from 2017 to 2022,
    and 7–27 per Nov–Dec;
  - 2015 has no Nov–Dec pairs before it, and 2016 has no Jan–May pairs.
- **Matching window at the outlet:** 223 × 247 m, about 3 × 3 ITS_LIVE
  pixels. The outlet (~250 m across) therefore blends with the main
  glacier: the same problem D2 dealt with.
- **Reading cost:** about 2 minutes per 12 km crop.

## Check 1: do E4's control flags persist?

Count of E4 false-alarm flags (setting 1.5 / 12 / 6) still alarmed when
the pre-melt window closes (27 May cutoff):

| Site | Flags still alarmed at 27 May |
|---|---|
| Annapurna Sanctuary | 11 of 12 (one had disappeared) |
| Langtang East | 4 of 4 |
| Jugal Himal | 2 of 2, but only by 1 pixel each |

Most flags are alarmed at 4–15 of the 16 cutoffs. **They are not mainly
thin early-window noise**, so a minimum pair count would not remove them.

## Check 2: can ITS_LIVE Sentinel-1 resolve a pre-melt ratio?

Window-median speed over **stable ground** (`landice` = 0) estimates the
noise floor of a window speed. It should be ~0.

| Measure | Value (m/yr, horizontal) |
|---|---|
| Per pixel: median over pixels, by year and window | 6–31 (Annapurna Sanctuary 16–62) |
| Area-aggregated, random 3 × 3 stable patches, medians of vx and vy before taking speed: median | 7.7 (Nepal), 5.0 (Langtang East) |
| Same, 90th percentile | 18.0 (Nepal), 13.1 (Langtang East) |

**What has to be measured is about as slow:**

- the outlet's winter speed is 7.4 m/yr horizontal (Wang et al.,
  2018–2025);
- the flagged pixels' winter speeds are ~13–17 m/yr in slant range (NISAR).

**The winter signal is at or below the archive's noise floor.**

## Check 3: does history explain the false alarms but not the outlet?

Area-aggregated historical pre-melt ratios (pre-melt ÷ winter window
speed):

- **Nepal outlet:**
  - winter speeds of 0–24 m/yr from year to year;
  - ratios 0.66, 41.0 (winter speed ~0), 2.92, 1.76, 2.19, 1.66 and 1.21
    for 2015 and 2017–2022.
- **Langtang East flags:** ratios 0.09–4.67, swinging between years with
  no consistent pattern. Winter speeds move between 4 and 57 m/yr from one
  year to the next.

The ratios behave as noise: a ratio of two numbers both near the noise
floor.

**Not conclusive, and not buried:** taken at face value, the outlet shows
pre-melt ratios of about 1.7–2.9 in 2018–2021 that did not lead to
failure. At this noise level that cannot be told apart from chance. If a
better baseline confirmed it, it would weaken "pre-melt acceleration was
the precursor".

## Conclusion

**No-go for ITS_LIVE Sentinel-1 as the per-pixel baseline.**

- Its noise floor (window speeds of ~5–18 m/yr over stable ground) is at
  the level of the winter speeds that the ratio needs.
- Its 220–250 m matching windows mix the outlet with the main glacier.
- Check 1 shows the false alarms are persistent, so the baseline would
  have been the right tool if it could resolve them.

The two remaining ways to get a multi-year baseline:

- **Process Sentinel-1 ourselves.** Dense offset tracking on SLCs,
  2017–2025, with smaller windows and more pairs per window.
  - Expected precision: roughly 0.1 px in range per 12-day pair (~7 m/yr
    in slant range), so about 2–3 m/yr from a window of ~8 pairs.
    Marginal for a 7–16 m/yr winter speed.
  - Cost: hundreds of SLCs (~1 TB) and an ISCE-2 or GAMMA setup.
- **Wait for NISAR's own second and third years** (2027–2028). This gives
  same-sensor, same-geometry baselines at no processing cost, but none for
  the 2026 test case.

No E5 detector was designed. No pass/fail run was made, and the held-out
sites remain unseen.
