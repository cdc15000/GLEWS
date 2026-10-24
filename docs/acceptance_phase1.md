# Phase 1 acceptance criteria (pre-registered)

Written on 23 Oct 2026, **before** the Phase 1 code was run against real data.
Results are recorded in the *Results* section below without changing these
thresholds. If a criterion is re-run with other settings (layer, SNR,
window size, pair selection), every setting tried is listed with its result.

Caveat: the criteria and results were first committed together, in one
file, so nothing timestamps the criteria on their own. Later phases should
commit their criteria separately, before running.

Conventions:

- **Source**: the `known_source` polygon in `config/nepal_2026.yaml`
  (ICIMOD 2026, Fig. 9, digitised by eye, ~50 m).
- **Stable ring**: ground 2–5 km from the source centre (28.288 N,
  85.524 E), excluding the source polygon.
- **Noise σ**: 1.4826 × median absolute deviation (MAD) of the referenced
  measurement over the stable ring, per pair.
- **Signal**: median of the referenced measurement over source pixels that
  pass the quality mask.
- **Patch rank**: the signal's percentile among the medians of 500 random
  patches the same size and shape as the source, placed anywhere in the
  crop that does not overlap the source.

## A1. NISAR GOFF: late-July motion over the source

Pair 26 Jul → 19 Aug 2026, track 98, slant-range offset, layer 2
(64-pixel window), SNR ≥ 5, referenced to the stable ring.

Pass if **both**:

1. |signal| ≥ 5 σ, and
2. patch rank ≥ 95th percentile.

Note: an exploratory look on 18 Oct 2026 put this signal near the 70th
percentile, so A1 is expected to fail on (2) unless the loader fixes change
the picture.

## A2. Sentinel-2: multi-year motion (Wang et al. 2026: ">100 m of pre-failure deformation" 2017–2026)

Sentinel-2 L2A, band 8 (10 m), same relative orbit only, normalised
cross-correlation / phase-correlation offsets, referenced per pair to the
stable ring.

Pass if **all**:

1. in at least half of the year-apart pairs (autumn to autumn), the
   source horizontal-displacement signal is ≥ 5 σ;
2. the cumulative source signal from the earliest usable 2017–2018 scene
   to the last pre-event scene is ≥ 50 m (half Wang et al.'s figure, which
   is likely a maximum rather than a median);
3. the moving area overlaps the source polygon: ≥ 50 % of source pixels
   with valid offsets exceed 5 σ in the cumulative map.

## A3. Sentinel-2: change in rate from January 2026 (Wang et al.: "a clear bifurcation phase starting in January 2026")

The abstract does not define "bifurcation", so GLEWS tests the part of it
it can measure: a rate change over the source.

Pass if **both**:

1. the source velocity from pairs spanning Jan–Aug 2026 is ≥ 2 × the
   velocity from same-orbit pairs over Jan 2024–Dec 2025, and
2. the difference between the two velocities is ≥ 3 combined σ.

Every pair used reports the fraction of source pixels flagged as cast
shadow, cloud shadow, cloud or snow by the L2A scene classification (SCL),
and the sun elevation of both scenes. A pair with > 30 % of the source
shadowed or cloudy in either scene is excluded and listed as excluded.

## A4. Cross-sensor consistency

Where an S2 pair and a GOFF pair overlap in time (Jul–Aug 2026), project
the S2 east/north displacement onto the NISAR slant-range direction (from
the product's LOS vectors, assuming slope-parallel motion) and compare with
GOFF over the source.

Pass if the GOFF and projected-S2 values agree in sign and within 2
combined σ. This is reported, not required, if no S2 pair overlaps.

## Results

*(filled in after the runs; thresholds above are not edited)*

### A1 (NISAR GOFF): PASS (23 Oct 2026)

`scripts/nepal_2026/acceptance_goff.py`, output `output/acceptance/A1_goff.json`.

| Setting | Source px valid | Signal (range) | σ | Signal/σ | Along-track | Patch rank |
|---|---|---|---|---|---|---|
| Pre-registered: layer 2, SNR ≥ 5 | 144/176 | +0.637 m | 0.093 m | 6.8 | +0.343 m | 99.8 % |

Sweep (all settings tried, 26 Jul → 19 Aug): layers 1–3 × SNR 0/3/5/10 give
4.3–13.0 σ and patch rank 95.2–100 %. The pre-registered setting passes
both conditions. Across all 21 pairs at the pre-registered setting, one
other pair also exceeds 5 σ (15–27 May 2026, 5.6 σ, +0.46 m), so a 5 σ
GOFF signal is not unique to the final pre-event pair.

Leakage was checked (see A4): the signal sits inside the polygon.

**What A1 does and does not show.** The pass is against random patches
elsewhere in the crop. It does **not** make the late-July signal distinctive
against the source's own history:

- Per 12 days, 26 Jul → 19 Aug is 0.32 m. Four spring 2026 pairs are higher:
  28 Mar–9 Apr 0.43, 21 Apr–3 May 0.42, 3–15 May 0.42, 15–27 May 0.46 m.
- At layer 2 with no SNR mask, the source's range offset is positive in
  21 of 21 pairs, and most patch ranks are above 80 %.
- The slope faces away from the radar. Its local incidence angle is about
  73° (from the LOS vector and the GLO-30 gradient), so offsets there may
  carry a geometry-dependent bias.

So A1 shows the source moving (or biased) by more than the crop's noise. It
does not, by itself, show a late-summer acceleration. A sharper test would
rank the source against patches of similar slope and aspect.

### A2, A3 (Sentinel-2): FAIL (23 Oct 2026)

`scripts/nepal_2026/acceptance_s2.py`, output `output/acceptance/A2_A3_s2.json`.

- A2.1 (autumn year-apart pairs): 0 of 4 pairs ≥ 5 σ. Spring pairs
  (a deviation, because autumn scenes are scarce): 0 of 8.
- A2.2 (cumulative ≥ 50 m): median-vector cumulative 2018-03 → 2026-03 is
  ≈ 0 m (σ 22.8 m). The 90th-percentile figure (149 m) is not evidence:
  5 σ is 114 m and the tail of a sum of eight noisy pairs is mostly noise.
- A2.3: 16 % of source windows ≥ 5 σ (needs 50 %).
- A3: 2026 rate 1.4 m/yr vs baseline 1.4 m/yr; no detectable change.

Diagnosis (from maps, not from re-tuning):

1. **The stable ring is not stable.** It contains snowfields and glaciers.
   Featureless snow gives multi-pixel random offsets, so ring σ ranges
   from 1 to 65 m depending on snow cover.
2. **Cross-season pairs are noise here.** The only 2026 pair spanning
   January (24 Jan → 5 Jun, sun 39° → 74°) has σ = 65 m.
3. **Motion sits beside the polygon, not inside it.** In the two cleanest
   pairs (Mar 2019 → Mar 2020, Mar 2024 → Mar 2025), a coherent body moving
   ≈ 10–20 m/yr lies immediately south-east of the ICIMOD outline. Inside
   the outline, offsets are below ≈ 5 m/yr except at its south-east edge.
   The DEM (Copernicus GLO-30) puts the outline on a west-facing 35° slope
   at a median 4,830 m. ICIMOD describes a north face at ~5,200 m. The
   outline (digitised by eye from ICIMOD Fig. 9) may be mislocated.
   *(Superseded: the post-event scar confirms the outline; see below.)*

Not done: the south-east body was **not** adopted as the source. Picking
the area that moves and then testing whether it moves would be circular.
Any re-test uses an outline from independent evidence (the post-event
scar) and is labelled post hoc.

### Independent check of the source outline (23 Oct 2026)

`scripts/nepal_2026/fetch_s2_scar.py` fetched matched pre-/post-event
Sentinel-2 scenes (21 Oct 2025 and 21 Oct 2026, orbit 119, sun 48.8° vs
48.9°). The scar mapped from NIR change (|Δ| > 2000 DN, connected to the
polygon) covers 0.86 km²: 84 % of it lies inside the ICIMOD outline, and it
covers 64 % of the outline (IoU 0.57). **The ICIMOD outline is confirmed.**
The fast body seen in A2 is the glacier south-east of the scar, which did
not fail. The DEM's west-facing aspect (277°) describes the failed slope
correctly. "North face" refers to the mountain, not to the scar's local
aspect.

### POST HOC Sentinel-2 re-test (23 Oct 2026; not a pre-registered result)

`scripts/nepal_2026/posthoc_s2.py`, output `output/acceptance/posthoc_s2.json`,
figure `posthoc_s2_speed.png`. Changes from the pre-registered setup:
- the source is the scar outline;
- stable windows need correlation ≥ 0.6, which rejects featureless snow;
- only spring pairs, with day of year within ±15 days;
- every scene combination per year pair (311 pairs), summarised by the median;
- distinctiveness is ranked against 500 random patches of the scar's shape.

| Year pair | Pairs | Scar speed (m/yr) | σ (m/yr) | Speed/σ | Patch rank |
|---|---|---|---|---|---|
| 2018→19 | 4 | 0.5 | 4.2 | 0.1 | 23 % |
| 2019→20 | 57 | 1.1 | 3.6 | 0.3 | 38 % |
| 2020→21 | 63 | 1.1 | 3.8 | 0.3 | 52 % |
| 2021→22 | 12 | 3.2 | 4.0 | 0.8 | 60 % |
| 2022→23 | 10 | 1.8 | 6.5 | 0.3 | 56 % |
| 2023→24 | 54 | 2.2 | 4.4 | 0.5 | 55 % |
| 2024→25 | 53 | 1.5 | 3.2 | 0.5 | 60 % |
| 2025→26 | 58 | 1.7 | 4.9 | 0.3 | 40 % |

The rate change 2025→26 against 2023→25 is 1.7 vs 1.9 m/yr (−0.03 σ). The
sum of the annual rates, 2018→2026, is 13 m.

**Injection test** (`scripts/nepal_2026/injection_s2.py`,
`output/acceptance/injection_s2.json`). Inside the scar, the secondary
image was replaced by a westward-shifted copy, and the same measurement
was re-run on three clean spring pairs:

| Pair | 0 m/yr injected | 6 m/yr | 12 m/yr | Scar windows with peak ≥ 0.6 (ring) |
|---|---|---|---|---|
| 2024-03 → 2025-03 | E +0.5 | E −3.0 | E −7.5 | 69 % (87 %) |
| 2019-03 → 2020-03 | E −0.5 | E −5.0 | E −10.0 | 76 % (87 %) |
| 2023-03 → 2024-03 | E +1.5 | E −2.5 | E −8.0 | 60 % (60 %) |

The scar's interior correlates nearly as well as stable ground. An
injected 12 m/yr comes back as 8–10 m/yr net (about 70–85 %, diluted by
320 m windows straddling the scar edge); 6 m/yr comes back as 3.5–4.5. The
glacier south-east of the scar, measured at > 20 m/yr, is a positive
control on real data.

**Conclusion (A2):** inside the failed area, Sentinel-2 at 10 m rules out
steady spring-to-spring motion of about 12 m/yr or more in every year from
2018 to 2026. It mostly rules out 6 m/yr. Measured rates are 0.5–3 m/yr.
GLEWS therefore does **not** reproduce Wang et al.'s > 100 m inside the
scar.

**A3 cannot be tested here.** That is not evidence of "no change". Pairs
spanning January are noise (σ ≈ 65 m). The spring-to-spring pairs ending
in Mar–Apr 2026 contain only 2–3 months of 2026. A doubling from January
would raise their annual average by about 20 %, which is well below the
noise. Sentinel-2 has no power to test a January onset at this site. Possible reasons, none tested yet:
- Wang et al. measured a maximum, or measured a different part of the
  mass (for example the glacier south-east of the scar, which moves
  > 20 m/yr);
- their "enhanced POT" uses Landsat or other processing;
- the motion was concentrated in summers that Sentinel-2 cannot see
  through monsoon cloud.

Spring-to-spring pairs average over a year and cannot resolve a summer
surge.

### A4 cross-sensor (23 Oct 2026): not testable as specified

No clear Sentinel-2 scene exists between 15 June and 26 August 2026
(monsoon), so no optical pair overlaps the GOFF pair. As a reported
substitute, GOFF's two components were inverted under the slope-parallel
assumption (`glews.fusion.slope_parallel_from_radar`). Inputs:
- NISAR LOS (E 0.613, N 0.152, U 0.776) and along-track direction, from
  the product's radar-grid cube at 5,000 m;
- source gradient from Copernicus GLO-30.

Result for 26 Jul → 19 Aug: E −0.56, N +0.20, U −0.41 m. That is 0.60 m
horizontal toward 289° (±1 σ on the inputs gives 279–298°), or 0.73 m in
3D, about 11 m/yr. This is consistency under an assumption, not independent
confirmation. The DEM gradient is an input to the inversion, and on a
west-facing plane any positive range offset, including a bias, maps to
westward motion. The rate is higher than the 0.5–3 m/yr Sentinel-2
spring-to-spring rates. But GOFF shows similar or higher per-day offsets
in spring 2026 (see A1), so this does not establish a late-summer
acceleration.

The GOFF signal sits inside the outline: the median range offset is
+0.64 m inside versus +0.24 m in a 600 m band east of it
(`goff_map_20260726_20260819_layer2.png`). A1 is not leakage from the
fast glacier.

### Summary

| Test | Status | One line |
|---|---|---|
| A1 GOFF late-July motion | **PASS** (pre-registered) | 0.64 m in 24 d, 6.8 σ, 99.8th percentile vs the crop; robust to layer/SNR. Not distinctive against the source's own spring 2026 pairs; back-slope geometry (local incidence ~73°) |
| A2 S2 multi-year motion | **FAIL** (pre-registered and post hoc) | 0.5–3 m/yr inside the scar every year; injection shows ≥ 12 m/yr would be seen; > 100 m not reproduced |
| A3 S2 change from Jan 2026 | **FAIL** (pre-registered); not testable post hoc | no Sentinel-2 pair has the power to resolve a January onset here |
| A4 cross-sensor | not testable | no overlapping optical pair; GOFF slope-parallel direction is consistent with downslope, by construction |
| GUNW with fixed loader | n/a (diagnostic) | source pixels are in unreliable unwrapping components in 73–100 % of every pair; GUNW cannot measure the source |
