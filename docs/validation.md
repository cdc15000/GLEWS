# Glacier Early Warning System — Validation Report

Validation of GLEWS detection algorithms against four glacier and rock-slope collapse events: two retrospective analyses of real events and two synthetic reconstructions based on published event parameters.

---

## Summary

| Event | Location | Date | Type | Lead Time | Key Detection Metric | Data Source |
|-------|----------|------|------|-----------|---------------------|-------------|
| Nepal–Tibet Border | 28.1°N, 85.9°E | 26 Aug 2026 | Real | **43 days** | 5.06 m vertical subsidence; 295 GOFF flags | NISAR GOFF L-band |
| Weisshorn (Randa) | 46.1°N, 7.8°E | Ongoing slow deformation | Real | N/A | **81 mm** cumulative displacement | Sentinel-1 C-band |
| Chamoli | 30.4°N, 79.7°E | 7 Feb 2021 | Synthetic | **~30 days** | Acceleration z-score > 3 sigma | Synthetic |
| Aru Glaciers | 34.0°N, 82.2°E | 17 Jul 2016 | Synthetic | **39–54 days** | Dual-glacier precursor signal | Synthetic |

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

- **Real events** (Nepal 2026, Weisshorn) were analyzed using actual satellite data products (NISAR GOFF, Sentinel-1). Detection parameters were fixed before analysis — no tuning to the specific event.
- **Synthetic events** (Chamoli, Aru) used GLEWS's synthetic scene generator (`synthetic.py`) to create InSAR displacement fields modeled on published event parameters (failure volume, collapse velocity, pre-failure acceleration rate, and slope geometry from the literature). These validate that the detection algorithms can identify precursor signals with realistic noise and seasonal contamination, but do not validate the data-acquisition chain.

---

## Event 1: Nepal–Tibet Border 2026 (Real)

### Event description

On August 26, 2026, a massive glacier–rock collapse occurred on the Nepal–Tibet border (approximately 28.1°N, 85.9°E), generating a debris flow that traveled over 20 km down-valley. The event was widely reported and drew international attention to satellite-based early warning for glacier hazards.

### Data

- **Source:** NISAR L-band GOFF (amplitude offset-tracking) products from ASF DAAC
- **Observation period:** 12-day revisit, covering ~90 days before collapse
- **Why GOFF:** The pre-collapse displacement reached meters of vertical subsidence — far beyond the ~12 cm/cycle ambiguity limit of GUNW phase unwrapping. Only GOFF amplitude offset-tracking maintained coherent measurements through this deformation.

### Results

| Metric | Value |
|--------|-------|
| First anomaly flag | 43 days before collapse |
| Total GOFF anomaly flags | 295 |
| Cumulative vertical subsidence | 5.06 m |
| Peak acceleration z-score | > 5 sigma |
| Voight's law convergence | **Did not converge** |
| BOCPD changepoints detected | 2 (matching two-phase failure) |
| Step-change flags | Multiple, tracking discrete jumps |

### Key findings

1. **43 days of advance warning** — The sigma-threshold acceleration detector flagged the site well ahead of the collapse. This lead time would have been operationally useful for evacuation planning.

2. **Two-phase failure mode** — The collapse was not a smooth acceleration-to-failure (as Voight's law assumes). Instead, two discrete displacement steps were observed, separated by a period of relative quiescence. The BOCPD changepoint detector and step-change detector both identified these regime shifts; the classical z-score detector also flagged the site but with less diagnostic clarity about the failure mode.

3. **Voight's law failure** — The inverse-velocity fit did not converge to a meaningful failure-time prediction because the two-phase step-change failure mode does not produce the smooth, monotonic acceleration that Voight's model requires. This motivated adding the BOCPD and step-change detectors to the pipeline.

4. **GOFF over GUNW** — GUNW (phase-based) products saturated at ~12 cm per acquisition interval. The GOFF offset-tracking products maintained measurement continuity through >5 m of displacement, demonstrating that operational glacier early warning must use offset tracking for fast-moving targets.

### Limitations

- Retrospective analysis: the detection parameters were fixed before running, but the event was known. A true prospective test would run the monitor in advance without knowledge of the event.
- NISAR revisit (12 days) limits temporal resolution — faster-moving collapses with shorter precursor phases might not be captured with sufficient temporal sampling.
- No independent ground-truth displacement measurements were available for validating the 5.06 m subsidence figure, though it is consistent with the reported collapse volume and geometry.

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

1. **~30 days of advance warning** — The z-score detector flagged the site approximately 30 days before the simulated failure date. The gradual acceleration pattern (unlike the Nepal 2026 step-change) was well-suited to the sliding-window velocity approach.

2. **Voight's law success** — Unlike the Nepal 2026 case, the smooth acceleration profile allowed Voight's inverse-velocity fit to converge with R^2 = 0.87. The predicted failure time was 3 days later than the actual simulated failure — a level of accuracy that would still provide actionable warning.

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

4. **Voight's law applicability** — Both glaciers showed surge-like acceleration that approximated the smooth velocity increase Voight's model expects, yielding high R^2 fits. This contrasts with the Nepal 2026 step-change failure where Voight's law failed.

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
| Two-phase step-change (Nepal 2026) | Effective (late) | Effective (early) | Effective (early) | Fails |
| Slow creep (Weisshorn) | Effective | N/A | N/A | N/A |

The ensemble of four detectors (z-score, BOCPD, step-change, Voight) provides redundancy across failure modes. No single detector covers all cases — the multi-detector approach is essential.

### Lead-time comparison

| Event | Lead Time | Failure Mode | Rate |
|-------|-----------|-------------|------|
| Nepal 2026 (real) | 43 days | Step-change | m/day |
| Chamoli (synthetic) | ~30 days | Smooth acceleration | cm/day |
| Aru-1 (synthetic) | 39 days | Surge | m/day |
| Aru-2 (synthetic) | 54 days | Surge | m/day |

Lead times of 30–54 days are consistent across events and failure modes, suggesting the detection threshold (3 sigma default) is calibrated for approximately 1-month advance warning when the precursor signal exceeds the noise floor.

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
- Shirzaei, M. "Satellite images before Nepal disaster showed warning signs." *Nature* News Q&A, 2 September 2026.
