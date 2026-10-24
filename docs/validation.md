# Glacier Early Warning System — Validation Report

Validation of GLEWS detection algorithms against four glacier and rock-slope collapse events: two retrospective analyses of real events and two synthetic reconstructions based on published event parameters.

---

## Summary

| Event | Location | Date | Type | Lead Time | Key Detection Metric | Data Source |
|-------|----------|------|------|-----------|---------------------|-------------|
| Gyirong–Rasuwa (Nepal–Tibet) | Source 28.288°N, 85.524°E (ICIMOD Fig. 9) | 26 Aug 2026 | Real | **Not detected** at source | GOFF offsets: 0.64 m in the final pre-event pair (6.8 σ) but no larger than spring 2026; GUNW phase blind over the source; Sentinel-2 offsets: ≤ 3 m/yr, no precursor | NISAR GUNW + GOFF L-band (track 98), Sentinel-2 |
| Weisshorn (Randa) | 46.1°N, 7.8°E | Ongoing slow deformation | Real | N/A | **81 mm** cumulative displacement | Sentinel-1 C-band |
| Chamoli | 30.4°N, 79.7°E | 7 Feb 2021 | Synthetic | **~30 days** | Acceleration z-score > 3 sigma | Synthetic |
| Aru Glaciers | 34.0°N, 82.2°E | 17 Jul 2016 | Synthetic | **39–54 days** | Dual-glacier precursor signal | Synthetic |

An earlier run over 28.1°N, 85.9°E, ~38 km east of the source, reported 43 days of lead time and 5.06 m of subsidence. Those figures do not apply to this collapse; see Event 1.

---

## Methodology

### Detection pipeline

All four events were processed through the same GLEWS Tier 0 detection pipeline:

1. **Seasonal decomposition** — displacement time series decomposed into linear trend + annual/semi-annual harmonics + residual via least-squares fit
2. **Acceleration estimation** — sliding-window velocity regression (60-day window, 12-day step) on detrended/deseasonalized residuals
3. **Z-score normalization** — each pixel's acceleration normalized against its own baseline statistics (first 70% of observation window)
4. **Changepoint detection** — BOCPD (Bayesian Online Changepoint Detection) with Student-t observation model for abrupt regime shifts
5. **Step-change detection** — sudden displacement jump identification using dual threshold (sigma + absolute minimum)
6. **Spatial clustering** — DBSCAN on anomalous pixels with geographic distance metric
7. **Voight's law fitting** — inverse-velocity trend analysis on top-scoring anomalies

### Real vs. synthetic events

- **Real events** (Gyirong–Rasuwa 2026, Weisshorn) were analyzed using actual satellite data products (NISAR GOFF, Sentinel-1). Detection parameters were fixed before analysis — no tuning to the specific event.
- **Synthetic events** (Chamoli, Aru) used GLEWS's synthetic scene generator (`synthetic.py`) to create InSAR displacement fields modeled on published event parameters (failure volume, collapse velocity, pre-failure acceleration rate, and slope geometry from the literature). These validate that the detection algorithms can identify precursor signals with realistic noise and seasonal contamination, but do not validate the data-acquisition chain.

---

## Event 1: Gyirong–Rasuwa 2026 (Real)

### Event description

ICIMOD's preliminary synthesis (2026) gives the following account:

- At 08:37 NPT on 26 August 2026, ~132 Mm³ of rock (~115 Mm³) and glacier ice (~17 Mm³) detached from the north face of Langtang Lirung at ~5,200 masl and fell ~1,200 m into the Lhende Khola.
- The debris surge reached the Gyirong–Rasuwagadhi border corridor by 08:44–08:50 and continued down the Bhotekoshi–Trishuli. SMS alerts went out at 09:15.
- Nepal reported 1,451 dead and 5,745 missing; China reported 43 dead and 519 missing.
- It was not a glacial lake outburst flood, and no heavy rainfall preceded it.
- Precursor motion was visible from space. Sentinel-1 InSAR showed up to 40 cm line-of-sight displacement over the source between 7 July and 24 August (48 days); NISAR showed up to ~30 cm between 26 July and 19 August.
- The source glacier moved ~29 m/yr over 2014–2024 with no clear acceleration, so long-term velocity alone would not have marked it out.

### Original run (wrong area)

The original retrospective run was centred at 28.1°N, 85.9°E, chosen from early press reports before the source was known. That point is ~38 km east of the Langtang Lirung source. The results below come from that area, not from the collapse source.

- **Source:** NISAR L-band GOFF (amplitude offset-tracking) products from ASF DAAC
- **Observation period:** 12-day revisit, covering ~90 days before collapse

| Metric (original study area) | Value |
|-------|-------|
| First anomaly flag | 43 days before event date |
| Total GOFF anomaly flags | 295 |
| Cumulative vertical subsidence | 5.06 m |
| Peak acceleration z-score | > 5 sigma |
| Voight's law convergence | Did not converge |
| BOCPD changepoints detected | 2 |

### Interpretation

1. **The 43-day lead time does not apply to this collapse.** Because the original area does not contain the source, the 43-day figure shows only that the detectors fire on real NISAR data. The re-analysis below found no comparable detection at the source. The signals found 38 km away need their own explanation, whether that is unrelated slope motion, glacier flow or processing artefacts.

2. **No two-phase failure at the source.** An earlier version of this report described a "two-phase step-change failure" based on the two BOCPD changepoints above. ICIMOD describes gradual precursor motion at the source (40 cm over 48 days, ~8 mm/day LOS) rather than discrete jumps. The step-change and BOCPD detectors remain useful, but this event is not evidence for them.

3. **C-band should be able to see this precursor.** At ~8 mm/day, a 12-day Sentinel-1 pair accumulates ~10 cm, near the practical unwrapping limit but within reach. ICIMOD's 40 cm figure comes from Sentinel-1. GLEWS has not yet processed Sentinel-1 over the source (see below).

4. **Steady fast flow is a false-positive trap.** A glacier moving ~29 m/yr without acceleration would swamp an absolute-velocity threshold. Per-pixel baselines (z-scores against each pixel's own history) and the Tier 1 cross-checks are what keep it from flagging every season.

### Re-analysis over the source (NISAR GUNW, October 2026)

The source was located from the latitude/longitude axes of ICIMOD Figure 9. The failed rock slope spans ~28.284–28.293°N, 85.519–85.530°E; its core is at **28.288°N, 85.524°E**, ~3.6 km north of the Langtang Lirung summit. Scripts are in [`scripts/nepal_2026/`](../scripts/nepal_2026/).

**Data.** NISAR L2 GUNW, ascending track 98, frame 016: 21 sequential 12-day pairs (one 24-day pair at the end) from 28 Nov 2025 to 19 Aug 2026. The last pair, 26 Jul–19 Aug, is the same window ICIMOD cites for its ~30 cm NISAR figure. Excluded:

- the 19 Aug–31 Aug pair, which spans the collapse;
- duplicate product listings (one product per date pair);
- descending track 48, which has no acquisitions from January to June and so no baseline.

Each ~2.4 GB product was read with HTTP range requests, taking a 12 × 12 km crop of the 80 m unwrapped layers and a 6 × 6 km crop of the 20 m wrapped interferogram.

**Method.** The standard GLEWS path: `load_nisar_gunw_stack` → acceleration map → `detect_anomalies`, with the detection parameters in `config/nepal_2026.yaml`, unchanged. One step was added that the loader lacks: each date is referenced to the median of a stable ring 2–5 km from the source.

| Metric (coherence mask 0.3, the default) | Value |
|-------|-------|
| Source pixels passing the coherence mask | 33 of 171 |
| Failed-slope pixels unwrapped as unreliable (connected component 0) in most pairs | 100% (crop-wide: 44%) |
| Source median cumulative LOS, 28 Nov – 19 Aug | ≈ −5 cm (ICIMOD: ~30 cm in the final 24-day pair alone) |
| Flags over the 144 km² crop | 53 |
| Flag timing (peak window) | Mostly March (snow season) and July |
| Flags on the failed slope | None. Nearest: rank 6, at 28.2838°N, 85.5319°E on the intact bedrock at the slope's SE edge, peak window 20 Jul, built entirely on unreliably unwrapped pixels |

Sensitivity: lowering the mask to 0.2 gives 66 flags, and to 0.1 gives 78. At 0.1 a flag does sit on the source (0.2 km away), but it ranks 11th.

**Findings.**

1. **GLEWS's standard NISAR path would not have warned of this collapse.** The failed slope never unwraps reliably at 80 m posting, so its motion is underestimated by roughly an order of magnitude and buried among false positives elsewhere in the crop.
2. **No confirmed signal in the 20 m wrapped phase.** A localized phase patch appears over the failed slope in the 14–26 Jul pair. Similar fringe rings around the massif appear in winter pairs too, consistent with terrain-correlated atmosphere or DEM error. Two ad-hoc unwrapping attempts (2-D least squares and a 1-D profile) gave summer values no different from winter. The feature is unconfirmed.
3. **Loader gaps.**
   - `nisar.py` ignores the `connectedComponents` layer, so unreliably unwrapped pixels feed the detectors.
   - It does not reference displacement to stable ground.
   - It applies the coherence mask to stack-mean coherence, which snow-season pairs drag down.
4. **False-positive rate.** 53 flags in 144 km² of high-mountain terrain is far too many for analyst review. Tier 1 cross-checks and season-aware coherence handling are needed before alerts are meaningful.

The table and findings above use the loader as it was before Phase 1
(findings 3 lists its gaps). Phase 1 fixed them; see below.

### Phase 1: fixed loader, radar offsets and Sentinel-2 (October 2026)

Full criteria, every setting tried, and figures:
[docs/acceptance_phase1.md](acceptance_phase1.md). Pass/fail criteria were
written before the runs; anything changed afterwards is labelled post hoc.

- **GUNW with the fixed loader** (per-pair connected-component masking,
  per-pair referencing within the component, no zero-motion gap filling):
  the source is in unreliable components in 73–100 % of pixels in every
  pair, so no source pixel survives. Phase interferometry at 80 m cannot
  measure this slope.
- **GOFF offset tracking (A1, pass):** 26 Jul → 19 Aug, slant range
  +0.64 m (6.8 σ, above 99.8 % of random patches), along-track +0.34 m.
  Inverted assuming motion parallel to the slope: about 0.6 m horizontal
  toward 289° in 24 days. But four spring 2026 pairs show equal or larger
  per-day offsets, the source's offset is positive in every pair, and the
  slope faces away from the radar (local incidence ≈ 73°). A1 shows motion
  or bias above crop noise; it does not establish a late-summer
  acceleration.
- **Sentinel-2 offset tracking (A2/A3, fail):** 153 clear orbit-119
  scenes, 2018–2026. A post-event scar mapped from 21 Oct 2025 vs
  21 Oct 2026 imagery confirms the ICIMOD outline (84 % of the scar inside
  it).
  - Inside the scar, spring-to-spring rates are 0.5–3 m/yr in every year,
    against noise of 3–6 m/yr.
  - An injection test shows a steady 12 m/yr would have been recovered.
    Wang et al.'s > 100 m (2017–2026) is therefore not reproduced inside
    the scar.
  - The glacier south-east of the scar moves > 20 m/yr.
  - Winter shadow on the north face and monsoon cloud leave Sentinel-2 no
    power to test a January 2026 onset.
- **Cross-sensor check (A4):** not testable. No clear optical scene
  between 15 June and 26 August.

### Next steps

- Find where Wang et al. (2026,
  [doi:10.1007/s10346-026-02842-6](https://doi.org/10.1007/s10346-026-02842-6))
  measured their > 100 m: the scar, or the glacier beside it.
- Rank the GOFF source signal against patches of similar slope and
  aspect, not random patches.
- Sentinel-1 offset tracking 2017–2026. It sees through monsoon cloud and
  winter shadow, so it can test summer-concentrated motion and a January
  2026 onset.

### Limitations

- Retrospective analysis: even with fixed parameters, knowing the event biases site selection.
- NISAR's 12-day revisit limits temporal resolution for faster precursor phases.
- No independent ground-truth displacement measurements are available.

---

## Event 2: Weisshorn (Randa), Switzerland (Real)

### Event description

The Randa rockslide area on the Weisshorn (Valais, Switzerland) has been under geodetic monitoring since the catastrophic 1991 rockslides. Ongoing slow deformation of the remaining unstable rock mass provides a test case for detecting millimeter-to-centimeter-scale displacement with C-band InSAR.

### Data

- **Source:** Sentinel-1 C-band ascending-track SLC data, processed via ISCE-2 + MintPy
- **Observation period:** Multi-year time series
- **Why C-band:** The deformation rate (~10–20 mm/yr) is well within C-band phase unwrapping limits. Coherence is maintained on the exposed rock slope.

### Results

| Metric | Value |
|--------|-------|
| Cumulative displacement detected | 81 mm |
| Spatial extent of deformation | Consistent with known unstable mass boundary |
| Coherence over deforming area | > 0.5 (adequate for phase-based measurement) |
| Seasonal signal | Present, correctly removed by harmonic decomposition |

### Key findings

1. **81 mm of cumulative displacement** measured, consistent with published geodetic monitoring data for the Randa site.

2. **Deformation boundary** — The spatial extent of the detected anomaly aligns with the known boundary of the unstable rock mass, validating the DBSCAN spatial clustering approach.

3. **Seasonal removal** — The harmonic decomposition correctly separated the annual thermal expansion/contraction cycle from the secular deformation trend. Without seasonal removal, the acceleration z-score would have been dominated by seasonal fluctuations.

### Limitations

- This is a slow-deformation case, not a pre-collapse scenario. It validates displacement measurement accuracy and spatial delineation but not failure-time prediction.
- No lead-time metric applies — the site is not in an active pre-collapse phase (as of the analysis date).
- C-band coherence limits the technique to exposed rock and debris; vegetated or snow-covered areas decorrelate.

---

## Event 3: Chamoli, India (Synthetic Reconstruction)

### Event description

On February 7, 2021, a rock and ice avalanche in the Chamoli district of Uttarakhand, India, triggered a devastating debris flow down the Rishiganga and Dhauliganga valleys, destroying two hydropower plants and killing over 200 people. Published studies (e.g., Shugar et al. 2021, *Science*) characterized the event as a wedge failure of rock and glacier ice from the north face of Ronti Peak (~5,500 m elevation).

### Synthetic reconstruction

The GLEWS synthetic scene generator created a displacement field modeled on published Chamoli parameters:

- **Failure volume:** ~27 million m^3 (rock + ice)
- **Pre-failure acceleration:** Modeled as gradual acceleration over ~45 days, based on reported precursor signals in optical satellite imagery
- **Slope geometry:** 35-40 degree slope, consistent with Ronti Peak north face
- **Noise:** Realistic atmospheric phase screen contamination (stratified + turbulent APS) and seasonal harmonic signals
- **Spatial extent:** 500 x 500 m deformation zone

### Results

| Metric | Value |
|--------|-------|
| First anomaly flag | ~30 days before simulated collapse |
| Peak acceleration z-score | > 4 sigma |
| Voight's law convergence | Converged (R^2 = 0.87) |
| Predicted failure time error | 3 days late |
| Spatial delineation | Deformation zone recovered within 15% of true extent |

### Key findings

1. **~30 days of advance warning** — The z-score detector flagged the site approximately 30 days before the simulated failure date. The gradual acceleration pattern was well-suited to the sliding-window velocity approach.

2. **Voight's law success** — Unlike the original Nepal 2026 study-area run, the smooth acceleration profile allowed Voight's inverse-velocity fit to converge with R^2 = 0.87. The predicted failure time was 3 days later than the actual simulated failure — a level of accuracy that would still provide actionable warning.

3. **Atmospheric contamination resilience** — The detection correctly identified the anomaly despite injected stratified and turbulent APS noise, validating the seasonal decomposition and z-score normalization approach.

### Limitations

- **Synthetic data** — this validates the algorithm, not the end-to-end observing system. Real Chamoli precursor signals in SAR data have not been independently confirmed (optical precursors were reported, but SAR coverage during the pre-failure period was limited).
- The synthetic failure signal was injected with the same model that the detector is tuned for (gradual acceleration), which may overstate detection performance relative to real events with more complex failure modes.
- No atmospheric correction was applied (only the z-score normalization's implicit robustness to noise was tested).

---

## Event 4: Aru Glaciers, Tibet (Synthetic Reconstruction)

### Event description

On July 17, 2016, a massive glacier collapse occurred at the Aru Range in western Tibet. The first collapse (Aru-1) involved approximately 68 million m^3 of glacier ice detaching and flowing ~6 km across a nearly flat valley floor. A second collapse (Aru-2) occurred on September 21, 2016, from an adjacent glacier. Published studies (Kääb et al. 2018, *Nature Geoscience*) identified pre-collapse surge behavior visible in optical imagery weeks before the events.

### Synthetic reconstruction

Two adjacent synthetic deformation zones modeled the dual Aru collapse:

- **Aru-1:** 68 million m^3 volume, 39-day pre-collapse acceleration phase, surge-like velocity increase from baseline ~0.1 m/day to >5 m/day at collapse
- **Aru-2:** 83 million m^3 volume, 54-day pre-collapse acceleration phase, similar surge behavior
- **Slope geometry:** Low-angle glacier (5-15 degree), consistent with the surprising nature of these collapses on gentle slopes
- **Noise:** Full atmospheric and seasonal contamination

### Results

| Metric | Value |
|--------|-------|
| Aru-1 first flag | 39 days before simulated collapse |
| Aru-2 first flag | 54 days before simulated collapse |
| Peak z-score (Aru-1) | > 6 sigma |
| Peak z-score (Aru-2) | > 4 sigma |
| Voight's law (Aru-1) | Converged (R^2 = 0.91) |
| Voight's law (Aru-2) | Converged (R^2 = 0.82) |
| Both glaciers flagged simultaneously | Yes (after Aru-2 began accelerating) |
| Spatial separation resolved | Yes (DBSCAN produced two distinct clusters) |

### Key findings

1. **39–54 days of advance warning** — Both glacier collapses were detected well in advance of the simulated failure dates. The longer lead time for Aru-2 reflects its longer pre-collapse acceleration phase (consistent with the real event timeline).

2. **Dual-glacier detection** — The DBSCAN spatial clustering correctly resolved the two adjacent deformation zones as separate anomaly flags, despite their proximity (~2 km apart). This validates the spatial delineation for cases where multiple hazards exist in close proximity.

3. **Low-angle glacier detection** — The Aru events were remarkable because low-angle glacier collapses were not thought possible before 2016. The detection algorithms, being purely data-driven (no slope-angle threshold for flagging), correctly identified the anomalous acceleration regardless of slope angle.

4. **Voight's law applicability** — Both glaciers showed surge-like acceleration that approximated the smooth velocity increase Voight's model expects, yielding high R^2 fits.

### Limitations

- **Synthetic data** — same caveat as Chamoli. The real Aru events occurred before routine NISAR coverage and with limited Sentinel-1 coverage of western Tibet.
- The surge-like acceleration model used for the synthetic data is idealized. Real glacier surges involve complex thermomechanical feedbacks that produce more irregular velocity evolution.
- The Aru glaciers had extremely high velocities (meters/day) near failure — in real L-band InSAR this would likely decorrelate, requiring GOFF offset tracking (not modeled in this synthetic test).
- Both glaciers were modeled with independent failure clocks; the real events may have involved mechanical coupling between adjacent glaciers that is not captured.

---

## Cross-Event Analysis

### Detection method effectiveness by failure mode

| Failure Mode | Z-Score | BOCPD | Step-Change | Voight's Law |
|-------------|---------|-------|-------------|--------------|
| Smooth acceleration (Chamoli, Aru) | Effective | Effective | N/A | Converges |
| Slow creep (Weisshorn) | Effective | N/A | N/A | N/A |

The ensemble of four detectors (z-score, BOCPD, step-change, Voight) provides redundancy across failure modes. No single detector covers all cases — the multi-detector approach is essential.

### Lead-time comparison

| Event | Lead Time | Failure Mode | Rate |
|-------|-----------|-------------|------|
| Nepal 2026 (real, at source) | Not detected | Gradual (per ICIMOD) | ~8 mm/day |
| Chamoli (synthetic) | ~30 days | Smooth acceleration | cm/day |
| Aru-1 (synthetic) | 39 days | Surge | m/day |
| Aru-2 (synthetic) | 54 days | Surge | m/day |

Lead times of 30–54 days are consistent across the synthetic events (the one real event with a known precursor, Nepal 2026, was not detected), suggesting the detection threshold (3 sigma default) is calibrated for approximately 1-month advance warning when the precursor signal exceeds the noise floor.

### Satellite revisit vs. lead time

The 12-day NISAR revisit provides 2–4 observations during the critical pre-failure period for the fastest-developing events. Sentinel-1's 6-day revisit (in areas with ascending + descending coverage) would double the temporal sampling. Future constellations with shorter revisit times would reduce the minimum detectable lead time.

---

## Overall Limitations

1. **Small validation set** — four events (two real, two synthetic) is insufficient for statistical characterization of detection performance (precision, recall, false alarm rate). A systematic validation against a catalog of historical collapses would be needed.

2. **No false-positive assessment** — the pipeline has not been run over large areas without known collapses to estimate the false-alarm rate. The cross-check filters (Tier 1) are designed to reduce false positives, but their effectiveness has not been quantified.

3. **Synthetic events are biased** — synthetic reconstructions use the same signal models that the detectors are tuned for, inflating apparent performance. Real events have more complex, irregular precursor signals.

4. **Temporal resolution** — 6–12 day satellite revisit fundamentally limits the detection of rapid-onset failures with precursor phases shorter than ~2 weeks.

5. **Data gaps** — real satellite coverage has geographic and temporal gaps. The global watchlist assumes consistent NISAR coverage that may not be available for all sites at all times.

6. **No independent validation** — displacement measurements from InSAR have not been compared against ground-truth (GNSS, terrestrial radar, extensometer) data for any of the four events.

---

## References

- Kääb, A. et al. "Massive collapse of two glaciers in western Tibet in 2016 after surge-like instability." *Nature Geoscience* 11, 114–120 (2018).
- Shugar, D.H. et al. "A massive rock and ice avalanche caused the 2021 disaster at Chamoli, Indian Himalaya." *Science* 373, 300–306 (2021).
- Voight, B. "A method for prediction of volcanic eruptions." *Nature* 332, 125–130 (1988).
- Adams, R.P. & MacKay, D.J.C. "Bayesian Online Changepoint Detection." arXiv:0710.3742 (2007).
- ICIMOD. "Understanding the Gyirong–Rasuwa disaster of 26 August 2026: A preliminary synthesis." International Centre for Integrated Mountain Development (2026). doi:10.53055/ICIMOD.1150
- Shirzaei, M. "Satellite images before Nepal disaster showed warning signs." *Nature* News Q&A, 2 September 2026.
