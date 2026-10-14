# Glacier Early Warning System — Validation Report

Validation of GLEWS detection algorithms against four glacier and rock-slope collapse events: two retrospective analyses of real events and two synthetic reconstructions based on published event parameters.

---

## Summary

| Event | Location | Date | Type | Lead Time | Key Detection Metric | Data Source |
|-------|----------|------|------|-----------|---------------------|-------------|
| Gyirong–Rasuwa (Nepal–Tibet)\* | 28.1°N, 85.9°E (analysed); source at ~28.27°N, 85.51°E | 26 Aug 2026 | Real | 43 days\* | 5.06 m vertical subsidence; 295 GOFF flags\* | NISAR GOFF L-band |
| Weisshorn (Randa) | 46.1°N, 7.8°E | Ongoing slow deformation | Real | N/A | **81 mm** cumulative displacement | Sentinel-1 C-band |
| Chamoli | 30.4°N, 79.7°E | 7 Feb 2021 | Synthetic | **~30 days** | Acceleration z-score > 3 sigma | Synthetic |
| Aru Glaciers | 34.0°N, 82.2°E | 17 Jul 2016 | Synthetic | **39–54 days** | Dual-glacier precursor signal | Synthetic |

\* Measured over the original study area, ~38 km east of the Langtang Lirung source identified by ICIMOD (2026). These figures do not show that GLEWS detected the precursor at the source. Re-analysis centred on the source is pending; see Event 1.

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

## Event 1: Gyirong–Rasuwa 2026 (Real; re-analysis pending)

### Event description

ICIMOD's preliminary synthesis (2026) gives the following account:

- At 08:37 NPT on 26 August 2026, ~132 Mm³ of rock (~115 Mm³) and glacier ice (~17 Mm³) detached from the north face of Langtang Lirung at ~5,200 masl and fell ~1,200 m into the Lhende Khola.
- The debris surge reached the Gyirong–Rasuwagadhi border corridor by 08:44–08:50 and continued down the Bhotekoshi–Trishuli. SMS alerts went out at 09:15.
- Nepal reported 1,451 dead and 5,745 missing; China reported 43 dead and 519 missing.
- It was not a glacial lake outburst flood, and no heavy rainfall preceded it.
- Precursor motion was visible from space. Sentinel-1 InSAR showed up to 40 cm line-of-sight displacement over the source between 7 July and 24 August (48 days); NISAR showed up to ~30 cm between 26 July and 19 August.
- The source glacier moved ~29 m/yr over 2014–2024 with no clear acceleration, so long-term velocity alone would not have marked it out.

### What GLEWS analysed

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

1. **The lead time is not yet validated.** Because the analysed area does not contain the source, the 43-day figure shows that the detectors fire on real NISAR data. It does not show they would have warned of this collapse. The signals found 38 km away need their own explanation, whether that is unrelated slope motion, glacier flow or processing artefacts.

2. **No two-phase failure at the source.** An earlier version of this report described a "two-phase step-change failure" based on the two BOCPD changepoints above. ICIMOD describes gradual precursor motion at the source (40 cm over 48 days, ~8 mm/day LOS) rather than discrete jumps. The step-change and BOCPD detectors remain useful, but this event is not evidence for them.

3. **C-band can see this precursor.** At ~8 mm/day, a 12-day Sentinel-1 pair accumulates ~10 cm, which is near the practical unwrapping limit but within reach. The 40 cm cumulative signal over 48 days should be detectable with the existing Sentinel-1 + MintPy path. NISAR GOFF remains the fallback where motion outpaces phase unwrapping.

4. **Steady fast flow is a false-positive trap.** A glacier moving ~29 m/yr without acceleration would swamp an absolute-velocity threshold. Per-pixel baselines (z-scores against each pixel's own history) and the Tier 1 cross-checks are what keep it from flagging every season.

### Next steps

- Re-run with `config/nepal_2026.yaml`, now centred on the source (28.27°N, 85.51°E, 20 km buffer), using both Sentinel-1 and NISAR.
- Refine the source polygon from ICIMOD Figure 9 or Copernicus EMSR927 mapping.
- Compare GLEWS displacement against the ICIMOD-reported 40 cm (Sentinel-1) and ~30 cm (NISAR) figures before claiming a lead time.

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
| Nepal 2026 (real, original study area)\* | 43 days | Unresolved | m/day |
| Chamoli (synthetic) | ~30 days | Smooth acceleration | cm/day |
| Aru-1 (synthetic) | 39 days | Surge | m/day |
| Aru-2 (synthetic) | 54 days | Surge | m/day |

Lead times of 30–54 days are consistent across the synthetic events (the Nepal figure is excluded until re-analysed over the source), suggesting the detection threshold (3 sigma default) is calibrated for approximately 1-month advance warning when the precursor signal exceeds the noise floor.

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
