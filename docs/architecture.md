# GLEWS Architecture

Detailed architecture of the Glacier Early Warning System: data flow,
detection algorithms, quality filtering, cascade risk assessment,
monitoring loop, and alert dispatch.

---

## Pipeline Overview

GLEWS is a three-tier pipeline:

```
Satellite Data
      |
      v
 [Tier 0] Automated Screening
      |  - Acceleration z-score anomaly detection
      |  - Step-change detection
      |  - BOCPD changepoint detection
      |  - Voight's failure law fitting
      |  - Transfer learning classifier scoring
      |
      v
 [Tier 1] Physical Filtering
      |  - Cross-check quality (coherence, spatial, temporal)
      |  - Cascade risk assessment (volume, runout, exposure)
      |
      v
 [Tier 2] Analyst Review
      - Web dashboard for human classification
      - Interactive map viewer
```

Each tier reduces the number of candidates: Tier 0 produces flags from
raw data, Tier 1 filters false positives and estimates downstream
hazard, and Tier 2 presents the survivors to a human analyst.

---

## Data Flow

### Acquisition and Processing

```
ASF DAAC Archive
      |
      | search_scenes() / check_new_data()
      v
  SceneInfo list
      |
      | download_scenes()
      v
  SLC files on disk
      |
      | InSARProcessor.run_isce()     (ISCE-2: co-registration,
      |                                interferograms, unwrapping)
      v
  Unwrapped interferograms
      |
      | InSARProcessor.run_mintpy()   (MintPy: SBAS inversion,
      |                                tropospheric correction)
      v
  DisplacementTimeseries
      |  dates: [n_dates]
      |  displacement: [n_dates, n_rows, n_cols] meters
      |  velocity: [n_rows, n_cols] m/yr
      |  coherence: [n_rows, n_cols]
```

For NISAR data, the SBAS inversion is performed directly in
`nisar.py` using `numpy.linalg.lstsq` on the interferometric design
matrix, bypassing ISCE-2/MintPy.

### Detection Pipeline

```
DisplacementTimeseries
      |
      | compute_acceleration_map()
      v                               detect_step_changes()
  AccelerationMap                           |
      |                                     v
      |                               StepChangeMap
      |                                     |
      +------ detect_anomalies() ----------+
      |
      |  1. Threshold z-score maps
      |  2. DBSCAN spatial clustering
      |  3. Filter by min_pixels, min_area
      |  4. Score + rank
      |  5. Merge step-change flags
      |  6. Merge BOCPD flags (opt-in)
      |  7. Deduplicate overlapping clusters
      |  8. Populate displacement time series
      |  9. Classifier scoring (optional)
      | 10. Voight analysis (optional)
      v
  list[AnomalyFlag]
      |
      +---> apply_tier1_filters()
      |       coherence quality
      |       spatial consistency
      |       temporal consistency
      |       -> annotate, demote, or remove
      |
      +---> assess_cascade_risk_batch()
      |       volume estimation
      |       slope/valley geometry
      |       runout modeling
      |       population exposure
      |       -> CascadeAssessment per flag
      |
      v
  Filtered, ranked flags with risk levels
      |
      +---> generate_report()    -> PNG plots, GeoJSON, Markdown
      +---> serve() [dashboard]  -> analyst review web UI
      +---> serve() [mapview]    -> interactive Leaflet map
```

---

## Tiered Confidence Model

### Tier 0: Automated Screening (Unsupervised)

Each pixel is compared to its own history. No labeled collapse examples
are required.

- **Anomaly score** — a composite of peak z-score (50%), mean z-score
  (30%), and log10(area) (20%). Captures both intensity and spatial
  extent.
- **Step-change score** — uses the same formula but with displacement
  jump magnitude (in units of the configured minimum) replacing
  z-scores.
- **BOCPD score** — changepoint probability (in units of the
  configured threshold) replacing z-scores.

A flag's `score` is the primary ranking metric within Tier 0. Voight
fit boosts the score by 1.5x when R^2 exceeds the configured
threshold.

### Tier 1: Physical Filtering

Two independent checks narrow the candidate set:

1. **Cross-check quality** (`crosscheck.py`) — evaluates whether the
   InSAR data supports the anomaly. The composite score is the
   geometric mean of coherence quality, spatial consistency, and
   temporal consistency. Flags below `remove_below_quality` are
   dropped; those below `demote_below_quality` have their score
   halved.

2. **Cascade risk** (`cascade.py`) — evaluates whether the anomaly
   could produce a dangerous event. Risk levels:
   - **HIGH** — sufficient volume, above a confined valley, and
     downstream population or dam potential
   - **MODERATE** — sufficient volume and either above a valley or
     downstream population
   - **LOW** — sufficient volume but no valley or population exposure
   - **MINIMAL** — volume below threshold

   Only HIGH and MODERATE advance to Tier 2
   (`CascadeAssessment.passes_tier1`).

### Tier 2: Analyst Review (Human-in-the-loop)

A web dashboard (`dashboard.py`) presents surviving flags with:
- Anomaly metrics (score, z-score, area, acceleration)
- Displacement time series (inline SVG chart)
- Classification buttons (Watch / Warning / Cleared)
- Free-text notes
- Persistent analyst state (JSON file, atomic writes)

---

## Detection Algorithms

### 1. Acceleration Z-Score (`timeseries.py`)

The primary detector for gradual pre-failure speed-up.

**Seasonal removal.** Each pixel's displacement history is decomposed
by least-squares fit of:

    d(t) = v*t + sum_k [A_k*sin(2*pi*k*t/T) + B_k*cos(2*pi*k*t/T)] + offset

where T = 365.25 days and k = 1..n_harmonics (typically 2: annual +
semi-annual). This is done as a single vectorized `lstsq` for all
pixels with complete data; pixels with NaN observations fall back to
per-pixel solves.

**Sliding-window velocity.** The residuals (displacement minus trend
and seasonal) are divided into overlapping windows (default: 60-day
width, 12-day step). A linear velocity is fit to the residual within
each window.

**Acceleration.** The difference in velocity between successive windows,
divided by their time separation.

**Z-score.** Each pixel's acceleration is normalized against its own
baseline statistics. The baseline is the first 70% of windows (falling
back to the full series when the baseline is too sparse). The z-score
is `(acceleration - baseline_mean) / baseline_std`.

**Spatial clustering.** Pixels where `|z-score| > sigma_threshold` are
clustered using DBSCAN with a geographic distance metric. Clusters
smaller than `min_cluster_pixels` or `min_area_m2` are discarded.

### 2. Step-Change Detection (`timeseries.py`)

Catches abrupt, single-epoch displacement jumps that the acceleration
detector misses (e.g. the Nepal 2026 collapse: 3.5 m in one 12-day
cycle with post-rupture deceleration).

**Algorithm:**
1. Compute inter-epoch differences: delta[i] = d[i] - d[i-1]
2. Estimate baseline delta statistics (mean, std) from an early window
   (default: first 30% of epochs)
3. Flag where the deviation exceeds `sigma_threshold * baseline_std`
   AND the absolute jump exceeds `min_displacement_m`

The dual threshold (sigma + absolute floor) prevents near-zero-noise
pixels from tripping the sigma test on ordinary noise.

### 3. BOCPD Changepoint Detection (`timeseries.py`)

Bayesian Online Changepoint Detection (Adams & MacKay 2007) for
detecting regime changes in deformation rate.

**Model.** Operates on the velocity series (first differences /
time gaps). Uses a Normal-Inverse-Gamma conjugate prior as the
underlying predictive model, giving a Student-t predictive distribution
that handles unknown mean and variance within each segment.

**Recursion.** Maintains a log-space posterior over run lengths
(time since last changepoint). At each step:
1. Compute predictive probability of the new observation under each
   current run length
2. Apply hazard function (constant `hazard_rate`)
3. Normalize to get the posterior run-length distribution
4. Update the NIG sufficient statistics for each run length

**Detection.** Changepoints are identified by tracking the MAP
(most likely) run length. A changepoint is flagged when:
- The MAP run length drops below 3 after being at or above 3
- The posterior mass at short run lengths (0..2) exceeds
  `1 - threshold`

**Integration.** Opt-in via `detect.changepoint.method: "bocpd"` in
config. Runs per-pixel (O(n^2) per pixel), with a
`min_displacement_range_m` guard to skip quiet pixels.

### 4. Voight's Failure Law (`timeseries.py`)

Voight (1988) observed that inverse velocity (1/v) decreases linearly
toward zero as failure approaches. The x-intercept of a linear fit to
1/v vs. time predicts the failure date.

**Algorithm:**
1. Extract velocity time series at the peak anomaly pixel
2. Compute inverse velocity (only for positive, finite velocities)
3. Check that inverse velocity is decreasing (accelerating)
4. Linear fit: `1/v = slope * t + intercept`
5. Predicted failure time: `t_failure = -intercept / slope`
6. Accept if R^2 >= configured threshold and failure is in the future

Flags with an accepted Voight fit receive a 1.5x score boost.

### 5. Transfer Learning Classifier (`classifier.py`)

An optional scoring layer using logistic regression (numpy only).

**Features** (13 total): velocity mean/max/std, acceleration
mean/max/std, velocity trend slope, acceleration trend slope, velocity
ratio (max/baseline), threshold crossing count, temporal asymmetry
(fraction of total acceleration in the last 30%), total displacement,
anomalous acceleration duration.

**Training data.** Generated synthetically by `LandslideTrainingData`:
- Positive: exponential acceleration with Voight-like runaway
- Negative: seasonal motion, steady creep, transient speedups, quiet

**Model.** Binary logistic regression with L2 regularization and
built-in z-score feature standardization. Trained via gradient descent.
Serialized to JSON.

**Integration.** When `detect.classifier.enabled` is true and a model
file exists at `detect.classifier.model_path`, each flag's mean
displacement time series is scored. The probability is recorded in
`detection_details["classifier_score"]`.

---

## Quality Filtering

### Atmospheric Phase Screen Detection (`atmosphere.py`)

InSAR measurements are contaminated by atmospheric path-delay
variations. Two types are detected:

1. **Stratified APS** — elevation-correlated delay. Detected by
   regressing displacement against DEM elevation per epoch. High R^2
   (default > 0.6) indicates the epoch is dominated by stratified
   atmospheric delay.

2. **Turbulent APS** — spatially correlated noise at 5-50 km scales.
   Detected by computing the 2-D power spectrum of each epoch's
   displacement field and measuring the fraction of total power in the
   5-50 km wavelength band.

**Correction.** The stratified component can be removed by subtracting
the elevation-regression fitted surface from each epoch, preserving the
mean displacement.

### InSAR Cross-Checks (`crosscheck.py`)

Three quality checks per flag:

1. **Coherence quality** — fraction of flagged pixels with temporal
   coherence above `min_coherence` (default 0.3). Low coherence means
   unreliable phase measurements.

2. **Spatial consistency** — fraction of flagged pixels belonging to
   the largest 4-connected component. Real deformation produces a
   spatially smooth lobe; scattered pixels indicate noise.

3. **Temporal consistency** — fraction of epochs where the mean
   displacement over flagged pixels deviates from the scene-wide
   background by more than one background standard deviation. A signal
   in only one epoch is likely atmospheric or a processing artifact.

The composite quality score is the geometric mean of the three. Flags
below `remove_below_quality` (default 0.2) are dropped; those below
`demote_below_quality` (default 0.5) have their score halved.

---

## Cascade Risk Assessment (`cascade.py`)

### Volume Estimation

Estimated from the deformation anomaly area using the empirical
relation for rock slopes (Hungr et al., 2014):

    depth = 0.1 * sqrt(area)
    volume = area * depth * cos(slope)

This is conservative and adequate for Tier 1 screening.

### Runout Modeling

Uses the Scheidegger (1973) volume-dependent mobility relation:

    log10(H/L) = a - b * log10(V)

where H = elevation drop, L = runout distance, V = volume. Coefficients
a = 0.16, b = 0.16. Falls back to geometric angle of reach when volume
is not available.

### Valley Analysis

Searches downslope from the flagged site for a local elevation minimum
flanked by rising terrain on both sides. The valley is considered
"confined" if:
- The elevation drop from the site to the valley floor exceeds 100 m
- The terrain rises at least 50 m on both sides of the cross-section

Valley width is estimated from the cross-section at half-prominence.

### Dam Potential

A flag has dam potential when:
- It is above a confined valley
- Volume exceeds the minimum threshold
- Valley width is less than `valley_width_threshold_m` (default 500 m)

### Population Exposure

Population within the runout zone is estimated using either a provided
population grid (WorldPop-style) or a synthetic settlement model.

The synthetic model places settlements along valleys downstream with
population distributions typical of mountainous regions: hamlets
(50-200 people) at 2-8 km, villages (200-2000) at 5-20 km, and towns
(2000-20000) at 15-50 km. It uses a deterministic seed from coordinates
for reproducibility.

The exposure score (0-10) combines log-scaled population count (0-7)
and a proximity component (0-3) based on runout distance.

### Risk Classification

| Risk Level | Criteria |
|------------|----------|
| HIGH | Volume sufficient + above valley + (population > 0 or dam potential) |
| MODERATE | Volume sufficient + (above valley or population > 0) |
| LOW | Volume sufficient, no valley or population |
| MINIMAL | Volume below threshold |

### Cascade Score

A weighted composite:

    score = anomaly_score * 0.3
          + log10(volume)/7 * 0.25
          + log10(population)/5 * 0.2
          + 0.3 if dam_potential
          + 0.2 if above_valley
          + (slope/90) * 0.05

---

## Monitoring Loop (`monitor.py`)

### Design Principles

The monitoring loop provides incrementality at the **acquisition**
layer, not the **inversion** layer. SBAS inversion and acceleration
z-scoring cannot be incrementally patched — a new acquisition changes
every residual and every z-score. So:

- `MonitorState` tracks which product granules have been seen and
  downloaded.
- Only new granules are downloaded.
- The full product set is reloaded and re-inverted through the existing
  pipeline on each cycle.

The saving is in not re-downloading or re-searching, and the recompute
cost (SBAS inversion + acceleration map) is cheap relative to satellite
download time.

### Check Cycle

Each `run_check_cycle()` call:

1. **Search** ASF DAAC for NISAR products (rolling lookback window,
   default 30 days) and diff against `state.known_products`
2. **Download** new products to the site's product directory
3. **Load** the full product stack via `nisar.py` (GUNW preferred,
   GOFF fallback)
4. **Detect** anomalies via the standard pipeline
5. **Build alerts** for new flags not in `state.alerted_signatures`
6. **Dispatch** alerts to configured channels (email, Slack, webhook)
7. **Save** state checkpoint

### Alert Deduplication

Each flag gets a deterministic signature derived from `sha1(site_name |
lat_3dp | lon_3dp)`. Once alerted, the signature is recorded in
`state.alerted_signatures`. Re-running detection on a grown time series
(whose z-scores drift as the baseline grows) will not re-alert the same
physical anomaly.

If all alert channels fail for a given alert, the signature is removed
so the next cycle retries.

### Alert Levels

Derived from the ratio of peak z-score to the configured sigma
threshold:

| Level | Ratio |
|-------|-------|
| INFO | >= 1.0x |
| WARNING | >= 1.5x |
| CRITICAL | >= 2.5x |

A strong, short-horizon Voight fit (days_until_failure < 30) escalates
to CRITICAL regardless of z-score ratio.

### Multi-Site Support

`iter_site_configs()` supports two config shapes:

- **Single-site** (top-level `site:` key) — yields the config once
- **Multi-site** (`sites:` list) — merges shared defaults from the
  top level with per-site overrides for each entry

---

## Alert Dispatch (`alerts.py`)

### Architecture

```
AlertDispatcher
      |
      +--- EmailChannel (SMTP)
      |
      +--- SlackChannel (incoming webhook, Block Kit)
      |
      +--- WebhookChannel (generic JSON POST)
```

`AlertDispatcher.from_config()` reads the `alerts:` section of the
site config, expands `${ENV_VAR}` references in all string values, and
instantiates configured channels. Each channel's `from_config()` class
method validates required fields and returns None (with a warning) when
misconfigured.

`dispatch()` sends to every channel, logging failures without crashing.
Returns `dict[channel_name, success_bool]` so the monitoring loop can
decide whether to mark the alert as "sent" or retry.

### Secret Management

All config values support `${ENV_VAR}` expansion via
`expand_env_vars()`, which recursively processes strings, dicts, and
lists. Missing variables expand to the empty string.

---

## Configuration

Site configs are YAML files with the following sections:

| Section | Purpose | Required |
|---------|---------|----------|
| `site` | Name, coordinates, buffer, event date | Yes |
| `acquire` | Data source, product types, directories, credentials | For acquisition |
| `detect` | Detection parameters (sigma, clustering, Voight, step-change, BOCPD, classifier) | For detection |
| `cascade` | Cascade risk thresholds (volume, runout, valley width) | For Tier 1 |
| `monitor` | Monitoring interval, lookback, alert levels, state/alert dirs | For monitoring |
| `alerts` | Channel configs (email, slack, webhook) | For alert dispatch |
| `process` | ISCE-2/MintPy parameters | For Sentinel-1 processing |
| `report` | Output directory, formats | For reporting |

The `validate.py` module validates configs against required fields,
value ranges, cross-field consistency, and known section names.
