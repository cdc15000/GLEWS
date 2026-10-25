# Phase 3 (development): watch only terrain that can reach people

**Development only. No pass/fail.** Written 25 Oct 2026.

- Module: `src/glews/exposure.py`.
- Script: `scripts/nepal_2026/dev_exposure.py`, output
  `output/acceptance/dev_exposure.json`.

## Purpose

An operational service would not watch every glaciated pixel. It would
watch the slopes that could send a mass flow onto people or
infrastructure. This phase asks whether restricting E4 (the frozen
pre-melt detector, 1.5 / 12 / 6) to that terrain brings its false alarms
within the limit.

**Sites:** the five seen sites only. Manaslu West, Lapchi and Annapurna IV
north were not read.

## Method

**Assets.** OpenStreetMap via Overpass, **as of 1 Aug 2026**, before the
event, so destroyed features still count:

- settlements (city to isolated dwelling);
- roads (trunk to track; trekking paths not included);
- power plants and generators;
- dams.

The query box is each crop's centre ± 46 km.

**Terrain.** GLO-30 mosaicked at 90 m over the same domain.

**Reach.** From every crop cell, follow D8 steepest descent on the
depression-filled DEM:

- the cell is **relevant** if its path passes within 250 m of an asset
  while H / L ≥ the travel-angle floor;
- H is the drop from the source and L the path length so far;
- floors tested: 0.10, 0.15 and 0.20. The project's existing
  `cascade.exposure.angle_of_reach_deg` is 11°, i.e. H / L ≈ 0.19.

This replaces the synthetic settlements in `cascade.py` for any claim
about who is exposed. The reach model is a simple first-order travel-angle
model: no flow volume, no channel spreading, and no outburst-flood
continuation beyond the mass flow.

**E4 restricted, counted two ways:**

- (a) alarm pixels masked to relevant terrain before grouping;
- (b) whole flags kept if any of their pixels is relevant.

## Results

| Site | Relevant share of crop (H/L ≥ 0.10 / 0.15 / 0.20) | E4 false alarms, (a) masked / (b) any pixel, at 0.10 / 0.15 / 0.20 | E4 alone (Phase 2b/2c) |
|---|---|---|---|
| Nepal (target 100 / 99 / 85 % relevant) | 93 / 85 / 69 % | 0/0 · 0/0 · 0/0; hit kept from 20 Feb at every floor | 0 |
| Ganesh East | 82 / 59 / 39 % | 0/0 · 0/0 · 0/0 | 0 |
| Jugal Himal | 87 / 69 / 31 % | 2/2 · 2/2 · 0/1 | 2 |
| Annapurna Sanctuary | 95 / 85 / 59 % | 11/12 · 11/11 · **5/6** | 12 |
| Langtang East | 8 / 1 / 0 % | 0/0 · 0/0 · 0/0 | 4 |

## Readings

1. **In inhabited Himalayan valleys most glaciated terrain is relevant.**
   - With roads and tracks up most valleys, 59–95 % of the icy crops can
     reach an asset at H/L ≥ 0.15.
   - The restriction removes few false alarms there: Annapurna Sanctuary
     still has 11. Even at the strictest floor (0.20, close to the
     project's 11°) it keeps 5–6.
2. **It matters a great deal in remote terrain.** Langtang East is almost
   entirely irrelevant (8 % at 0.10, 0 % at 0.20): its glaciers are
   too far up-valley from the first mapped assets for a flow to reach
   them at these travel angles (not checked asset by asset). All 4 of its E4 false
   alarms disappear. That is a real operational saving, but it is
   geography, not specificity.
3. **The Nepal source is relevant under every floor** (85–100 % of the
   target): Lhende Khola → Rasuwa Gadhi / Gyirong. So the restriction does
   not cost the hit.
4. **Development criterion not met.** Annapurna Sanctuary exceeds 2 flags
   at every floor tested. Following the plan used since Phase 2b, the
   held-out sites were **not** used for a pass/fail test. They remain
   available for a future, stronger detector.

## Consequence

The exposure restriction belongs in GLEWS operationally:

- it removes alarms that matter to no one, as at Langtang East;
- it ranks the rest by what they threaten.

It does not solve specificity where most ice sits above roads and
villages, and those are the places that matter most. Specificity still
needs a baseline that says what is normal for each glacier: multi-year
NISAR, or Sentinel-1 processed for it (`docs/dev_phase2d_s1.md`).
