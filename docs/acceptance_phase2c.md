# Phase 2c criteria: pre-melt acceleration detector E4 on fresh controls (pre-registered)

Written 24 Oct 2026 and committed **before** any GOFF data for the new
control sites were extracted. Results go in a *Results* section added
afterwards; nothing above it is edited.

## Question

Does E4, a causal detector of **pre-melt acceleration**, stay within the
operational false-alarm limit on two glaciated control sites it has never
seen?

## History (disclosed)

E4 was designed after looking at the answer:

- **Phase 2:** E1/E2 raised 34–200 false alarms per 100 km² at Ganesh East
  and Jugal Himal.
- **Phase 2b** (`docs/dev_phase2b.md`), on Nepal, Ganesh East and Jugal
  Himal, all now development data:
  - offset blunders were rejected in the loader;
  - what remained was ordinary summer glacier speed-up;
  - the Nepal outlet (Wang et al.'s measurement area) was found to be
    unusual in accelerating **before** the melt season.
- **E4's window (15 Jan – 27 May) and its thresholds** were chosen on
  that development set, with the outlet's values known.

## Detector E4 (frozen)

Code: `prepare` and `run` in `scripts/nepal_2026/dev_premelt_detector.py`,
imported unchanged by the run script.

- **Input:** GOFF layer 2, SNR ≥ 5, slant-range offset, referenced to the
  2–5 km ring. Loader blunder rejection is on (`max_rate` = 5/12 m per
  day).
- **Per pixel:**
  - r_k is the rate in m per 12 days, smoothed by a 3×3 nan-median;
  - b is the baseline, the median of the first 4 pairs (28 Nov 2025 →
    15 Jan 2026), needing ≥ 3 valid.
- **Moving pixels only:** |b| ≥ the ring noise floor, the median ring
  1.4826·MAD over the baseline pairs.
- **At each cutoff** (the secondary date of pairs 6–21):
  - pm is the median of r over pairs with secondary date in
    (15 Jan, min(cutoff, 27 May)], needing ≥ 3 valid;
  - s = sign(b).
- **Alarm pixel:** s·pm / |b| ≥ **1.5** **and** (s·pm − |b|)·365/12 ≥
  **12 m/yr**.
- **Flag:** a connected group (8-connectivity) of ≥ **6** alarm pixels.

### How the setting was chosen (development sweep, `dev_premelt_detector.py`)

**Grid:**

| Parameter | Values |
|---|---|
| R0 (ratio) | 1.3, 1.5, 1.7, 1.8, 2.0 |
| I0 (increase) | 6, 8, 10, 12, 14 m/yr |
| Group size | 3, 4, 6 px |

**Selection rule:**

1. Keep the settings that meet the development criterion: both seen
   controls ≤ 2 flags, and a Nepal persistent hit with lead ≥ 31 days.
2. Take the maximum Nepal lead.
3. Then take the fewest control false alarms.
4. Then take the higher R0, the conservative tie-break.

(1.3, 12, 6) and (1.5, 12, 6) tie on the first three steps, so the
tie-break chose (1.5, 12, 6).

### In-sample margins (stated plainly)

- **Development false alarms:**

  | Site | Flags |
  |---|---|
  | Nepal | 0 |
  | Ganesh East | 0 |
  | Jugal Himal | **2**, exactly at the limit |

- **Nepal hit:** persistent from 20 Feb 2026, lead 187 days (in-sample).
  - The hitting group has 7–19 pixels, all inside the target, and
    overlaps the outlet by 3–5 pixels.
  - After 27 May it is frozen at **7 pixels, one above the group-size
    threshold**.
- **Thin margin on the increase:** at I0 = 14 m/yr the Nepal hit is lost.
  The outlet's own increase is +15 m/yr.
- **Calendar limitation:** E4 cannot see a precursor that starts in summer
  (after 27 May), and its window was chosen from this one event.

## Fresh control sites

### How they were chosen

From terrain, Sentinel-2 surface class and ASF metadata only
(`scripts/nepal_2026/select_controls.py`,
`output/acceptance/control_candidates.json`). **No GOFF values were read.**

**Candidates:**

| Candidate | Centre |
|---|---|
| Annapurna Sanctuary | 28.55 N, 83.90 E |
| Annapurna II south | 28.50 N, 84.15 E |
| Annapurna IV north | 28.60 N, 84.08 E |
| Manaslu west | 28.58 N, 84.48 E |
| Himlung | 28.76 N, 84.32 E |
| Langtang east | 28.25 N, 85.72 E |
| Lapchi | 28.00 N, 86.18 E |

**Rule:** a candidate must pass every check below. The survivors are then
ranked by snow/ice share (SCL class 11).

- the whole crop plus 1 km lies inside the track-98 frame-16 footprint;
- ASF lists the same 21 pre-event date pairs as for Nepal;
- no overlap with a seen crop, and outside the Budhi Gandaki drainage;
- no reported slope failure in or draining from the crop in the period;
- median slope ≥ 30° **and** share of slope > 45° ≥ 15 %.

**Disclosures about the rule:**

- The slope > 45° floor was written *after* the candidate statistics were
  seen. It removes only Annapurna IV north (13 %) and Himlung (8 %). Neither
  ranks in the top two on snow/ice, so the outcome is the same without it.
- Phase 2 set Langtang aside as "higher and gentler" when terrain match was
  the aim. Here the rule puts **ice first**: the Phase 2b failure mode is
  seasonal speed-up of large glaciers, so the hardest test is the most
  glacier-covered terrain.

**Excluded by the event check:**

- **Annapurna II south:** a 27 Sep 2026 flood and landslide on the Ghatte
  Khola, which rises from Lamjung Himal, dammed the Marsyangdi at Chame.
- **Himlung:** a 27 Sep 2026 avalanche, and it also fails the slope rule.

### Selected

| Site | Centre | Elevation p10/50/90 (m) | Median slope | > 45° | Snow/ice | km to nearest seen crop centre |
|---|---|---|---|---|---|---|
| Nepal (development) | 28.288 N, 85.524 E | 3601 / 4818 / 5988 | 39.2° | 34 % | 0.31 | — |
| **Annapurna Sanctuary** | 28.55 N, 83.90 E | 4116 / 4991 / 6072 | 32.4° | 18 % | 0.72 | 131 (Ganesh East) |
| **Langtang East** | 28.25 N, 85.72 E | 4658 / 5317 / 6032 | 30.4° | 19 % | 0.68 | 19 (Jugal), 20 (Nepal) |

### Disclosures about the sites

- **They are not as steep as Nepal.** 18–19 % of each is steeper than 45°,
  against 34 %. No remaining candidate matched Nepal's steepness. They are
  much icier: these are hard tests for seasonal glacier speed-up, not for
  steep-face noise.
- **The snow/ice share comes from different scenes:**
  - Annapurna Sanctuary: a 15 Nov 2025 scene in UTM zone 44
    (S2C_T44RQS_20251115);
  - the others: Dec 2025 scenes in zone 45.

  Class 11 includes seasonal snow.
- **Annapurna Sanctuary is at far range** (the west edge of the swath),
  where incidence is steeper. The same slant-range threshold there means a
  different ground speed than at Nepal, and its deep cirque may have
  layover and shadow gaps.
- **Reported events near the Sanctuary, none found inside the crop:**
  - July 2026: Modi Khola and Kimrong Khola floods;
  - September 2026: a Kimrong Khola landslide, reported near
    Ghandruk/Kimrong, south of the crop.
- **Langtang East:** no events found other than the 26 Aug 2026 Langtang
  Lirung collapse itself. That is the Nepal source, ~20 km west, and
  drains north to the Lhende Khola, not through this crop.
- **Every flag at either control counts as a false alarm**, whatever its
  cause.

### Data coverage floor (judged on the validity mask only, never on offset values)

- **Requirement:** a site is usable if the median share of pixels with
  SNR ≥ 5 over its 21 pairs is **≥ 0.50**. The development sites range
  0.70–0.75.
- **Fallback:** if a site fails, it is replaced by the next survivor by
  the same rule. First Manaslu West (snow/ice 0.28, median slope 33.5°,
  19 % > 45°), then Lapchi (0.23).
- The replacement and its coverage are reported.

## Run (identical at each site)

- 12 × 12 km crops (EPSG:32645) from `extract_goff_crops.py`, track 98,
  the same 21 pre-event pairs.
- Ring referencing as in Phase 2.
- Run script: `scripts/nepal_2026/phase2c_controls.py`. It imports `prepare`
  and `run` from `dev_premelt_detector.py`, runs no DEM- or SCL-dependent
  detector, and makes no plots of offsets before the verdict is printed.

## Detectors reported

| Detector | Role |
|---|---|
| **E4** | as frozen above: **the test** |
| E2 | unchanged from Phase 1e: a **difficulty check** |
| E2 + blunder rejection | E2 with the loader at `max_rate` 5/12: a **difficulty check** |

## Metrics

- **False-alarm flags per season:** distinct connected groups
  (8-connectivity) in the union, over all cutoffs, of flag pixels. At a
  control every flag counts.
- **Rate:** per 100 km² of crop.

## Pass rule (E4)

**Pass** if **both** fresh controls (or their pre-registered replacements)
have **≤ 2 false-alarm flags** over the season. The limit is the Phase 2
operational limit of 2 per 100 km² per season, ≤ 2 on a 144 km² crop.

## Difficulty check (decides how much a pass means)

- **Informative:** E2 + blunder rejection raises **≥ 10 flags** at the
  site. The site then has many moving, accelerating pixels, so an E4 pass
  there shows real specificity.
- **Uninformative:** fewer than 10. An E4 pass there is reported as
  "uninformative: the site was quiet". Jugal had 10 in Phase 2b with a
  stricter gate.
- **E4 counts as passing the test** only if it passes at **both** sites
  **and** at least one site is informative.

Reported without pass/fail:

- the Nepal E4 hit and lead (in-sample);
- false-alarm area;
- the cutoff of each control flag;
- E4 at the neighbouring grid settings (R0 1.3/1.7, I0 10/14, group size
  4), as a sensitivity check only.

## What a pass would and would not mean

- **Would mean:** on two icy sites never seen, a pre-melt acceleration rule
  that catches the Gyirong–Rasuwa outlet 187 days ahead raises at most
  2 false alarms per 144 km².
- **Would not mean:**
  - that E4 detects other precursors, since sensitivity is still n = 1;
  - that it works in other years, since this is one season;
  - that it can see summer-onset precursors.

A second event processed blind remains necessary.

## Results

*(added after the run, 24 Oct 2026; `scripts/nepal_2026/phase2c_controls.py`,
`output/acceptance/phase2c.json`)*

**Coverage:** both sites have all 21 pairs, with a median valid share of
0.70 against the 0.50 floor. No replacement was needed.

| Site | E4 flags (per 100 km²) | E4 area | E2 | E2 + blunder rejection | Difficulty |
|---|---|---|---|---|---|
| Annapurna Sanctuary | **12** (8.3) | 0.83 km² | 72 | 91 | informative |
| Langtang East | **4** (2.8) | 0.35 km² | 50 | 45 | informative |
| Nepal (in-sample) | 0 | — | — | — | hit from 20 Feb, lead 187 d |

### Verdict: **FAIL**

Both controls exceed the limit of ≤ 2 flags: Annapurna Sanctuary has 12
and Langtang East has 4.

### Readings

- **E4 cuts false alarms about 6–12× against E2 on unseen sites.** That
  does not reach the limit; Annapurna Sanctuary is 4× over it.
- **The false alarms are not a threshold-edge effect.**
  - Sensitivity at the neighbouring settings: Annapurna Sanctuary 8–21,
    Langtang East 3–5.
  - No neighbouring setting passes.
- **Pre-melt acceleration is not as rare as the development set
  suggested.**
  - Most control flags appear at the first cutoff (20 Feb), the same as
    the Nepal hit.
  - They are 6–22 pixels, as large as or larger than the Nepal hit
    (7–19 pixels).
- **Annapurna Sanctuary is at far range** (disclosed above), which may
  inflate its count. Langtang East, at similar range to Nepal, still
  fails.

### Consequence

On NISAR offsets from one season, no detector tried has met the
operational false-alarm limit out of sample:

- E1/E2 in Phase 2;
- gated E2 in Phase 2b (in-sample);
- E4 here.

The precursor is **detectable** (lead 163–187 days), but not
**specific**: comparable winter–spring accelerations occur on unrelated,
glaciated, 144 km² crops.

What has not been tried:

- multi-year same-season baselines, which are needed to tell whether a
  winter acceleration is unusual *for that pixel*;
- restricting the watched area to failure-relevant terrain above
  exposure.

The 6–12× reduction means an E4-like screen could act as a **triage
layer**, about 3–8 flags per 100 km² per season for analysts. That reading
was not pre-registered and does not meet the stated limit.
