# Phase 1b criteria: GOFF source signal against similar terrain (pre-registered)

Written on 24 Oct 2026 and committed **before** the ranking was run. Results
go in a separate *Results* section added afterwards; nothing above it is
edited.

## Why

A1 (`acceptance_phase1.md`) ranked the source's GOFF slant-range signal
against random patches anywhere in the 12 × 12 km crop. Most of the crop is
not a steep west-facing slope. The source faces away from the radar (local
incidence ≈ 73°), where offsets may be biased, so A1's 99.8 % rank may
reflect geometry rather than motion. This test asks whether the source
stands out among patches with the **same terrain geometry**.

## Data and settings (same as A1)

NISAR GOFF track 98 crops (`data/nisar/nepal_2026/goff_t98_crop`), layer 2,
SNR ≥ 5, slant-range offset, each pair referenced to the median of the
stable ring (2–5 km from the source centre, source excluded). Source =
`known_source` polygon (confirmed by the post-event scar).

Terrain: Copernicus GLO-30, area-averaged onto the 80 m GOFF grid
(EPSG:32645). Slope and downslope aspect per pixel from central
differences. Source reference values = median slope and circular-mean
aspect over the source pixels.

## Matched patches

A patch is the source's pixel shape translated anywhere in the crop. It is
a **matched control** if:

1. it does not overlap the source;
2. ≥ 50 % of its pixels have valid offsets in the pair being ranked;
3. ≥ 70 % of its pixels are terrain-like: slope within ±7.5° of the source
   value and aspect within ±30°.

Up to 500 matched controls are drawn per pair (fixed random seed 0, up to
200,000 draws). The source's statistic is its median referenced offset;
each control's is its median over the same pixels.

## B1. Late-July signal against matched terrain (primary)

Pair 26 Jul → 19 Aug 2026.

- **Pass:** source median ≥ 95th percentile of the matched controls'
  medians, with ≥ 30 matched controls.
- **Fail:** rank < 95 % with ≥ 30 controls.
- **Insufficient controls:** fewer than 30 matched controls (neither pass
  nor fail; reported as such).

## B2. Persistence across pairs (reported, no pass/fail)

The same rank for every one of the 21 pre-event pairs. If the source ranks
high in most pairs, including winter, the signal is a persistent offset
(steady motion or geometry bias), not a late-summer change. Already known
before this test: on the source alone, four spring 2026 pairs have equal or
larger per-12-day offsets than the final pair. B2 does not re-test that; it
places the source against matched terrain in each pair.

## Results

*(added after the run, 24 Oct 2026; `scripts/nepal_2026/terrain_rank_goff.py`,
`output/acceptance/B_terrain_rank.json`, `B_terrain_mask.png`)*

### B1: insufficient controls (pre-registered verdict)

The source is a 38.2° slope facing 278°. Terrain within ±7.5° and ±30° of
that covers 1,112 of 22,324 pixels outside the source (5 %), scattered in
small groups. Of 17,951 distinct translations of the source shape, **none**
was ≥ 70 % terrain-like, so there are 0 matched controls. Only 41 % of the
source's own pixels meet the definition. As specified, this is neither a
pass nor a fail.

### B2: not evaluable

There are no matched controls in any pair.

### POST HOC: terrain-like pixels away from the source (not pre-registered)

Added because B1 had no controls. For each pair, this compares the
source's median referenced offset with the median over all terrain-like
pixels more than 1 km from the source (641–873 pixels per pair):

| Pair | Source (m) | Terrain-like elsewhere (m) | Difference / σ | Source per 12 d (m) |
|---|---|---|---|---|
| 28 Nov – 28 Mar (10 pairs) | +0.01 to +0.15, all positive | −0.03 to +0.01 | 0.1–1.3 each; summed 0.70 m = 1.8 σ | 0.01–0.15 |
| 28 Mar – 9 Apr | +0.434 | +0.050 | 1.7 | 0.43 |
| 9 – 21 Apr | +0.183 | −0.015 | 0.7 | 0.18 |
| 21 Apr – 3 May | +0.423 | −0.017 | 1.9 | 0.42 |
| 3 – 15 May | +0.415 | −0.022 | 3.5 | 0.42 |
| 15 – 27 May | +0.455 | −0.022 | 5.9 | 0.46 |
| 27 May – 26 Jul (5 pairs) | +0.07 to +0.14 | −0.01 to +0.01 | 0.9–1.8 | 0.07–0.14 |
| 26 Jul – 19 Aug (24 d) | +0.636 | −0.016 | 7.0 | 0.32 |

Readings:

1. **Not a geometry bias of the slope type.** Terrain with the same slope
   and aspect, and therefore the same local incidence, shows a median
   offset within ±0.05 m of zero in every pair. The source's positive
   offset is specific to the source.
2. **Persistent motion.** The source is positive in 21 of 21 pairs (sign
   test p ≈ 5 × 10⁻⁷, treating pairs as independent). The cumulative slant
   range is 3.75 m over 264 days, about 5.2 m/yr.
3. **Two pulses, not one late-summer acceleration.** Per 12 days: winter
   ~0.07 m, **late March to May ~0.4 m**, June–July ~0.1 m, **late July to
   mid-August 0.32 m**. The spring pulse coincides with snowmelt onset. A
   snow-related artefact cannot be excluded for spring, but the matched
   terrain shows none.
4. **Comparison with Sentinel-2.** 5.2 m/yr in slant range is about
   5 m/yr horizontal under the slope-parallel assumption. The injection
   test shows Sentinel-2 recovers 6 m/yr as only 3.5–4.5 m/yr, and the
   measured 0.5–3 m/yr spring-to-spring rates partly average over the
   pulses. The two sensors are roughly consistent at this level. Both
   remain far below Wang et al.'s > 100 m since 2017.

Caveats: the terrain-like pixels are scattered, not a contiguous patch of
the source's size, so this is not the ranking B1 specified. Pairs share
dates (sequential network), so they are not fully independent.
