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

*(added after the run)*
