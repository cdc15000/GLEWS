# Phase 2 criteria: false alarms on independent control sites (pre-registered)

Written on 24 Oct 2026 and committed **before** any data for the control
sites were extracted. Results go in a *Results* section added afterwards;
nothing above it is edited.

## Roles of the data

- **Nepal (Gyirong–Rasuwa crop) is the development set.** The filters
  below were chosen after diagnosing its false alarms
  (`scripts/nepal_2026/diag_false_alarms.py`). False alarms there sat on
  very steep faces (median 49°, against 36° for the failed slope), and
  the radar offset was missing in 58 % of their pairs (28 % for the failed
  slope). Any Nepal result after filtering is **in-sample**.
- **Control sites are the test set for false alarms.** They were chosen
  from terrain statistics only (GLO-30), before any GOFF data were
  extracted. Every flag at a control counts as a false alarm. That is an
  upper bound, since a control can contain real, unreported motion.
- **Sensitivity is not tested out of sample.** There is one true positive.
  "Generalises" stays unknown until a second event is processed blind.

## Control sites

Same NISAR products as Nepal: track 98, frame 16, the same 21 pre-event
pairs (28 Nov 2025 → 19 Aug 2026), 12 × 12 km crops in EPSG:32645.

| Site | Centre | Distance from source | Elevation p10/50/90 (m) | Median slope | Share > 45° |
|---|---|---|---|---|---|
| Nepal (development) | 28.288 N, 85.524 E | — | 3602 / 4818 / 5989 | 39.2° | 34 % |
| **Ganesh East** | 28.36 N, 85.22 E | 31 km | 3854 / 4670 / 5576 | 33.0° | 14 % |
| **Jugal Himal** | 28.13 N, 85.86 E | 37 km | 4225 / 4955 / 5650 | 30.1° | 15 % |

Both are outside the 26 Aug 2026 flow path (Lhende Khola – Bhotekoshi –
Trishuli). Searches on 24 Oct 2026 found no reported slope failures at
either site in the period.

Considered and set aside:

- **Ganesh Himal west** (28.37 N, 85.10 E; best terrain match): a
  landslide reportedly started the 11 Oct 2026 Budhi Gandaki flood, at
  5,500–6,000 m in the Ungdilung Khola (Chamar–Shringi Himal), and
  Ganesh Himal's west side drains to the Budhi Gandaki. Its exact
  location relative to the crop is unconfirmed, and a precursor there would
  not be a false alarm.
- **Langtang and southern Shishapangma:** poor terrain match (higher and
  gentler).

## Processing (identical at every site)

- **GOFF:** layer 2, SNR ≥ 5, slant-range offset. Each pair is referenced
  to the median over a stable ring 2–5 km from the crop centre (at Nepal,
  the ICIMOD outline is excluded from the ring, as before).
- **Cutoffs:** the secondary date of pairs 6 to 21. Only data available at
  each cutoff are used.
- **Terrain:** GLO-30, area-averaged onto the 80 m GOFF grid.

## Detectors (all four reported)

- **E1:** GLEWS's existing pipeline, unchanged (as Phase 1e).
- **E2:** the Phase 1e rate-change detector, unchanged.
- **E1f:** E1 with a pixel mask applied before detection. At each cutoff,
  masked pixels are set to NaN in the displacement stack.
- **E2f:** E2 with the same mask. Alarm pixels must also be unmasked.

**Pixel mask at a cutoff** (causal, chosen on Nepal). A pixel is kept only
if:

1. its GLO-30 slope is ≤ 45°, **and**
2. it has a valid offset (SNR ≥ 5) in ≥ 70 % of the pairs available so far.

## Metrics

- **False-alarm flags per season:** the number of distinct connected
  groups (8-connectivity) in the union, over all cutoffs, of pixels
  belonging to flags that do not hit the target. At the controls there is
  no target, so every flag counts. The season is the 21-pair window
  (~264 days).
- **False-alarm rate:** that count per 100 km² of crop (144 km²).
- **Nepal hit and lead time:** as defined in Phase 1e (target = ICIMOD
  outline ∪ Wang et al. measurement area; the persistent hit starts at a
  cutoff ≤ 26 Jul 2026, allowing at most one later miss).

## Limit (from operations, not from Nepal)

**≤ 2 false-alarm flags per 100 km² per season.** If GLEWS monitors
~10,000 km² of priority glaciated terrain, that is ≤ ~200 site reviews per
season, or ~9 per 12-day NISAR cycle: about one analyst-day per cycle. On
a 144 km² crop the limit is **≤ 2 flags** (2.88 rounded down).

## Pass rule (per detector)

**Pass** if **both** hold:

1. Nepal: a persistent hit with lead ≥ 31 days. This is in-sample for E1f
   and E2f.
2. **Ganesh East and Jugal Himal each have ≤ 2 false-alarm flags** over
   the season.

Also reported for every detector: the Nepal false-alarm rate (in-sample)
and the rank of the Nepal hit at each cutoff (E1, E1f).

## Stated limitation

One NISAR season cannot separate an ordinary seasonal glacier speed-up
from a precursor acceleration except by magnitude. That needs a baseline
from the same season in earlier years, for example Sentinel-1 offsets
from 2017 onward. A control flag on a glacier speeding up in summer
counts here as a false alarm.

## Results

*(added after the run, 24 Oct 2026; `scripts/nepal_2026/phase2_controls.py`,
`output/acceptance/phase2.json`, `phase2.png`)*

False-alarm flags per season (distinct groups; rate per 100 km² in
brackets). The limit is ≤ 2 flags per crop, i.e. ≤ 2 per 100 km².

| Detector | Nepal (development) | Ganesh East (control) | Jugal Himal (control) | Nepal lead | Verdict |
|---|---|---|---|---|---|
| E1 | 344 (239) | 283 (197) | 288 (200) | 163 d | **FAIL** |
| E1f | 264 (183) | 250 (174) | 252 (175) | 163 d (in-sample) | **FAIL** |
| E2 | 52 (36) | 56 (39) | 49 (34) | 187 d | **FAIL** |
| E2f | 29 (20) | 55 (38) | 57 (40) | 127 d (in-sample) | **FAIL** |

(The Nepal E1/E1f counts are from `phase2.json`; E1's union over 16
cutoffs fragments into more groups than any single cutoff's flag count.)

Readings:

1. **All four detectors miss the limit by a factor of 17–100.** The best,
   E2/E2f, raises 34–40 false-alarm flags per 100 km² per season at the
   controls, against a limit of 2.
2. **The filters chosen on Nepal do not transfer.** E2f roughly halves
   Nepal's false alarms (36 → 20 per 100 km²; in-sample), but leaves the
   controls unchanged (39 → 38, 34 → 40). This is the overfitting the
   development/test split was set up to catch. E2f also cuts Nepal's lead
   from 187 to 127 days.
3. **Nepal's false-alarm rate is typical, not special.** The controls flag
   at the same rate as Nepal. The detectors are responding to ordinary
   high-mountain behaviour (seasonal glacier speed-up, snow and steep-face
   offset noise), not to a site-specific artefact.
4. **E1's step-change and acceleration detectors are far too sensitive**
   for offset data (~200 per 100 km²). They were designed for millimetre
   to centimetre InSAR, not metre-scale offsets.

### Consequence for the claim

GLEWS *detects* the Gyirong–Rasuwa precursor months ahead (Phase 1e), but
on NISAR offsets alone it cannot yet tell it apart from what happens on
any glaciated 144 km² in the same season. As it stands, a credible fund
request can claim **detectability, not operational specificity**.

### What this rules out and what is left

- Pixel masks on slope and data completeness are not the answer.
- Untested candidates, each needing **fresh control sites**, because
  Ganesh East and Jugal Himal have now been seen:
  - **magnitude and ratio gates:** the outlet reached ~55–60 m/yr slant
    range with a 3.4× rate increase, while the main glacier rose only
    1.36×;
  - **multi-year same-season baselines** (Sentinel-1 offsets, 2017–2025),
    which separate seasonal speed-up from anomalous acceleration;
  - **the failure-relevant population:** steep ice/rock *above* exposed
    communities (Phase 3 exposure), rather than every glaciated pixel.
