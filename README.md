<p align="center">
  <img src="docs/glews-banner.svg" alt="GLEWS - Glacier and Landslide Early Warning System" width="100%">
</p>

# GLEWS: Glacier and Landslide Early Warning System

[![CI](https://img.shields.io/badge/CI-passing-brightgreen)]()
[![Tests](https://img.shields.io/badge/tests-419%20passed-brightgreen)]()
[![Coverage](https://img.shields.io/badge/coverage-TBD-yellow)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)]()
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)]()

Satellite-based early warning pipeline for unstable glaciers and rock slopes.

GLEWS screens displacement time series derived from satellite radar (Sentinel-1 C-band and NISAR L-band) for anomalous acceleration, the signal that precedes catastrophic glacier and rock-slope collapses. It implements Tier 0 (automated screening), Tier 1 (cascade risk filtering with cross-check validation), and Tier 2 (analyst dashboard) stages, plus an operational `glews monitor` mode that continuously watches 13 configured sites worldwide for new acquisitions and raises alerts via email, Slack, or webhook.

The Glacier and Landslide Early Warning System comprises **22 Python modules** (~11 400 lines) with **419 tests** across 18 test files, and exposes **15 CLI commands**. All tests run on synthetic data with no network access required.

---

## Validation Results

Retrospective and synthetic validation against four glacier/rock-slope collapse events:

| Event | Type | Lead Time | Key Metric | Data Source |
|-------|------|-----------|------------|-------------|
| **Gyirong–Rasuwa (Nepal–Tibet) 2026** | Real event (26 Aug 2026) | **43 days**\* | 5.06 m vertical subsidence, 295 GOFF anomaly flags\* | NISAR GOFF L-band |
| **Weisshorn (Randa), Switzerland** | Real event | N/A | **81 mm** cumulative displacement detected | Sentinel-1 C-band |
| **Chamoli, India (Feb 2021)** | Synthetic reconstruction | **~30 days** | Pre-collapse acceleration detected via z-score | Synthetic (modeled on real event parameters) |
| **Aru Glaciers, Tibet (Jul 2016)** | Synthetic reconstruction | **39-54 days** | Dual-glacier collapse precursor signals | Synthetic (modeled on real event parameters) |

\* Measured over the original study area (28.1°N, 85.9°E), not over the collapse source identified by ICIMOD (see below). Re-analysis over the source is pending.

Details in [docs/validation.md](docs/validation.md).

### Gyirong–Rasuwa 2026 case study (primary validation)

**What happened.** ICIMOD's preliminary synthesis, published 13 October 2026, gives the following account:

- At 08:37 NPT on 26 August 2026, about 132 million m³ of rock and glacier ice (roughly 115 Mm³ rock and 17 Mm³ ice) detached from the north face of Langtang Lirung at about 5,200 masl.
- It fell about 1,200 m into the Lhende Khola. The debris surge reached the Gyirong–Rasuwagadhi border corridor in about 7 minutes.
- SMS alerts went out at 09:15, after the corridor had already been struck.
- In Nepal, 1,451 people were killed and 5,745 remain missing. In China, 43 were killed and 519 are missing.
- The event was a bedrock failure that sheared away part of a hanging glacier. It was not a glacial lake outburst flood, and no heavy rainfall or rise in river levels preceded it, so rainfall- and river-based warning systems could not have detected it.
- Pre-event satellite radar showed the slope moving gradually:
  - **Sentinel-1:** up to **40 cm** of line-of-sight displacement between 7 July and 24 August (48 days).
  - **NISAR:** up to about **30 cm** between 26 July and 19 August.

**What GLEWS found.** An earlier retrospective run against archived NISAR data was centred on 28.1°N, 85.9°E, based on initial reporting. That is about 38 km east of the source:

- **43 days of advance notice.** The sigma-threshold acceleration detector flagged anomalies well before the event date. Voight's inverse-velocity law did not converge.
- **5.06 m of vertical subsidence**, decomposed from line-of-sight displacement.
- **295 anomaly flags** from NISAR GOFF offset tracking.

Because these figures come from a different area, they do not show that GLEWS detected the Langtang Lirung precursor. `config/nepal_2026.yaml` now targets the source zone. The next validation step is re-running Sentinel-1 and NISAR analysis there over 7 July – 24 August 2026.

**Design implications:**

- A precursor of about 8 mm/day (roughly 10 cm per 12-day Sentinel-1 pair) is at the limit of C-band phase unwrapping. This is why GLEWS includes NISAR L-band and GOFF offset tracking (`nisar.py`).
- The source glacier moved about 29 m/yr from 2014 to 2024 without accelerating. ICIMOD reads the pre-failure signal as bedrock failure, with the ice playing a passive role. Scoring each pixel against its own baseline, rather than flagging absolute velocity, is what keeps a fast glacier from swamping the detector.
- ICIMOD lists atmospheric or geometric artefacts among the possible explanations for the pre-event signal. That argues for running the Tier 1 cross-check filters on real data.

---

## Quick Start

### Prerequisites

- Python 3.10+
- `pip install -e .` (or `pip install -e ".[dev]"` for development)

### Synthetic demo (no satellite data required)

```bash
# Set up environment
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Run the demo
glews demo --output output
```

This generates a synthetic InSAR scene with an injected pre-failure acceleration signal, runs anomaly detection, and produces:

- **`output/report.md`** - Summary of detected anomalies
- **`output/flags.geojson`** - Flagged sites for viewing in QGIS / Google Earth
- **`output/figures/anomaly_map.png`** - Spatial acceleration z-score map
- **`output/figures/flag_*_timeseries.png`** - Time-series plots per flagged site

### Operational monitoring (real NISAR data)

Once an [Earthdata](#data-requirements) account and `~/.netrc` are configured:

```bash
# Single check cycle against the built-in global watchlist, then exit
glews monitor -c config/global_watch.yaml --check-now

# Continuous loop, checking every 6 hours (or per config)
glews monitor -c config/global_watch.yaml --interval 6
```

Each cycle searches ASF DAAC for new GUNW/GOFF products, downloads anything not already seen, re-runs the detection pipeline on the grown time series, and writes new alerts to `output/alerts/<site>_alerts.json`. State is checkpointed to `data/monitor_state/` so a restart resumes rather than re-alerting.

### Analyst dashboard

```bash
glews dashboard -c config/nepal_2026.yaml --data-dir output --port 8080
```

Open `http://localhost:8080` to see flagged sites with severity levels, time-series plots, and cascade risk summaries. Analysts can classify flags as Watch / Warning / Cleared. When the flags come from `glews demo`, a **SYNTHETIC DATA** banner is shown so demo results aren't mistaken for observations.

### Docker (fastest path)

```bash
docker build -t glews .

# Synthetic demo
docker run --rm -v "$PWD/output:/app/output" glews demo --output /app/output

# Real monitoring
docker run --rm \
    -v "$HOME/.netrc:/root/.netrc:ro" \
    -v "$PWD/config:/app/config:ro" \
    -v "$PWD/data:/app/data" \
    -v "$PWD/output:/app/output" \
    glews monitor -c config/global_watch.yaml --check-now
```

---

## Architecture

```
Sentinel-1 SLC data         NISAR GUNW / GOFF products
        |                              |
        v                              v
+--- acquire.py ---+          +---- nisar.py -----+
| Search & download |          | Load GUNW/GOFF     |
| from ASF DAAC      |          | HDF5, SBAS-invert  |
+--------------------+          | to displacement,   |
        |                       | estimate velocity  |
        v                       +--------------------+
+--- process.py ---+                      |
| ISCE-2 -> MintPy  |                     |
| time-series        |                     |
| inversion          |                     |
+--------------------+                     |
        |                                  |
        +------ timeseries.py -------------+
                       |
                       |  Seasonal decomposition, acceleration
                       |  z-scores, BOCPD changepoints,
                       |  step-change detection, Voight's law
                       v
              +--- detect.py ----+   Tier 0: Spatial clustering,
              |                  |   deduplication, scoring;
              +------------------+   integrates BOCPD & step-change
                       |
            +----------+-----------+
            v                      v
  +-- classifier.py --+  +-- atmosphere.py -+
  | Transfer learning  |  | Stratified &     |
  | logistic regression|  | turbulent APS    |
  | scoring (optional) |  | detection/removal|
  +--------------------+  +------------------+
                       |
                       v
            +--- crosscheck.py --+   Tier 1 cross-check: coherence,
            |                    |   spatial/temporal consistency,
            +--------------------+   optical cross-check (stub)
                       |
          +------------+-------------+
          v            v             v
 +-- cascade.py -+ +- monitor.py -+ +- alerts.py ----+
 | Tier 1: volume | | Continuous    | | Email (SMTP),  |
 | valley, runout | | watch, state  | | Slack webhook,  |
 | (Scheidegger), | | tracking,     | | generic HTTP    |
 | pop. exposure  | | alert raising | | dispatch        |
 +----------------+ +---------------+ +----------------+
          |                                    |
          v                                    |
 +-- report.py ---+   Maps, time-series        |
 |                |   plots, GeoJSON,          |
 +----------------+   Markdown report          |
          |                                    |
          +------------------------------------+
          v
 +-- dashboard.py -+   Tier 2: analyst web UI,    +-- mapview.py ---+
 |                 |   severity review, site  <--> | Interactive      |
 +-----------------+   classification              | Leaflet map      |
                                                   +-----------------+

 Supporting modules:
 +-- spatial.py ----+  +-- tactical.py -+  +-- provenance.py +  +-- benchmark.py +
 | Spatial analysis  |  | Tactical alert |  | Audit trail &   |  | Performance    |
 | slope, aspect,    |  | prioritization |  | reproducibility |  | benchmarking   |
 | viewshed, DEM ops |  | & routing      |  | tracking        |  | suite          |
 +-----------------+  +----------------+  +-----------------+  +----------------+

 +-- synthetic.py --+  +-- validate.py -+
 | Synthetic scene   |  | Config schema  |
 | generation for    |  | validation &   |
 | testing & demos   |  | checking       |
 +-----------------+  +----------------+
```

### Module inventory (22 modules)

| Module | Purpose |
|--------|---------|
| `acquire.py` | Sentinel-1 SLC search and download from ASF DAAC |
| `alerts.py` | Multi-channel alert dispatch (email, Slack, webhook) |
| `atmosphere.py` | Atmospheric phase screen detection and correction |
| `benchmark.py` | Performance benchmarking suite |
| `cascade.py` | Tier 1 cascade risk assessment (volume, runout, exposure) |
| `classifier.py` | Transfer learning logistic-regression scoring |
| `cli.py` | Click-based CLI with 15 subcommands |
| `crosscheck.py` | Tier 1 cross-check filters (coherence, spatial, temporal) |
| `dashboard.py` | Tier 2 analyst web dashboard |
| `detect.py` | Tier 0 anomaly detection and spatial clustering |
| `mapview.py` | Interactive Leaflet GeoJSON map viewer |
| `monitor.py` | Continuous operational monitoring loop |
| `nisar.py` | NISAR GUNW/GOFF HDF5 loading and SBAS inversion |
| `process.py` | ISCE-2 + MintPy InSAR processing orchestration |
| `provenance.py` | Audit trail and reproducibility tracking |
| `report.py` | Report generation (plots, GeoJSON, Markdown) |
| `spatial.py` | Spatial analysis - slope, aspect, viewshed, DEM operations |
| `synthetic.py` | Synthetic InSAR scene generation |
| `tactical.py` | Tactical alert prioritization and routing |
| `timeseries.py` | Time-series analysis: seasonal decomposition, z-scores, BOCPD, Voight |
| `validate.py` | Configuration validation and schema checking |
| `__init__.py` | Package initialization |

### CLI commands (15)

| Command | Description |
|---------|-------------|
| `glews search` | Search ASF archive for Sentinel-1 scenes |
| `glews download` | Download SLC scenes from ASF |
| `glews process` | Run InSAR processing (ISCE-2 + MintPy) |
| `glews detect` | Run Tier 0 anomaly detection |
| `glews assess` | Run Tier 1 cascade risk assessment |
| `glews report` | Generate analysis report |
| `glews monitor` | Continuous monitoring for new NISAR data |
| `glews dashboard` | Launch Tier 2 analyst review dashboard |
| `glews map` | Launch interactive GeoJSON map viewer |
| `glews demo` | Run full pipeline on synthetic data |
| `glews train` | Train precursor classifier model |
| `glews validate` | Validate a site configuration file |
| `glews info` | Show system and dependency information |
| `glews benchmark` | Run performance benchmarking suite |
| `glews audit` | Run provenance audit on detection history |
| `glews version` | Show GLEWS version |

---

## Pipeline Stages

### 1. Data acquisition (`glews search`, `glews download`)

Searches the ASF DAAC archive for Sentinel-1 SLC scenes covering a study area, selects the optimal orbital track (maximizing temporal sampling), and downloads scenes. Requires an [Earthdata Login](https://urs.earthdata.nasa.gov/) account.

```bash
glews search --config config/nepal_2026.yaml
glews download --config config/nepal_2026.yaml
```

### 2. InSAR processing (`glews process`)

Wraps ISCE-2 (interferogram generation) and MintPy (time-series inversion) to produce displacement time series from the downloaded SLC data.

**Prerequisites:** ISCE-2 and MintPy must be installed separately. See [ISCE-2](https://github.com/isce-framework/isce2) and [MintPy](https://github.com/insarlab/MintPy) documentation.

```bash
glews process --config config/nepal_2026.yaml
glews process --config config/nepal_2026.yaml --step prepare  # just generate configs
glews process --config config/nepal_2026.yaml --dry-run       # print commands only
```

### 3. Anomaly detection (`glews detect`)

The core analytical component. For each pixel:
1. **Decomposes** the displacement time series into linear trend + seasonal harmonics + residual
2. **Estimates acceleration** via sliding-window velocity regression on the residuals
3. **Normalizes** each pixel's acceleration against its own historical baseline (z-score)
4. **Detects changepoints** via Bayesian Online Changepoint Detection (BOCPD)
5. **Detects step changes** - sudden displacement jumps characteristic of two-phase failures
6. **Clusters** spatially connected anomalous pixels using DBSCAN
7. **Scores and ranks** clusters by composite anomaly score
8. Optionally fits **Voight's failure law** (inverse-velocity trend) to top candidates
9. Optionally runs **transfer-learning classifier** scoring via `classifier.py`

No labeled collapse data is required - each site is compared to its own history.

### 4. Atmospheric correction (`atmosphere.py`)

Detects and corrects atmospheric phase screens (APS) that contaminate InSAR measurements:
- **Stratified APS** - elevation-correlated delay, detected by DEM regression per epoch
- **Turbulent APS** - spatially correlated noise at 5-50 km scales, detected via power spectrum analysis

### 5. Tier 1 cross-check filters (`crosscheck.py`)

Multi-sensor cross-check filters that eliminate false positives:
- **Coherence quality** - adequate interferometric coherence at flagged pixels
- **Spatial consistency** - contiguous deformation lobe vs. scattered noise
- **Temporal consistency** - signal persistence across multiple SAR acquisitions
- **Optical cross-check** - Sentinel-2 visible surface change (stub)

### 6. Cascade risk assessment (`glews assess`)

Evaluates whether flagged sites can produce dangerous downstream cascades:
- **Volume estimation** from deformation extent and slope geometry
- **Valley confinement analysis** from DEM
- **Runout modeling** - Scheidegger (1973) volume-dependent mobility relation
- **Population exposure mapping** - WorldPop or synthetic settlement generation

### 7. Spatial analysis (`spatial.py`)

DEM-derived geomorphological analysis:
- Slope and aspect computation from DEM grids
- Viewshed analysis for line-of-sight assessments
- Spatial clustering and neighborhood operations
- Coordinate transformations and distance calculations

### 8. NISAR data loading (`nisar.py`)

Reads NISAR L2 GUNW (unwrapped interferogram) and GOFF (pixel offset) HDF5 products via vectorized SBAS inversion. NISAR's L-band (24 cm) wavelength maintains coherence on glaciated surfaces where Sentinel-1's C-band (5.6 cm) decorrelates; GOFF extends coverage to meter-scale displacements.

### 9. Operational monitoring (`glews monitor`)

Turns the retrospective pipeline into a standing watch. Periodically searches ASF DAAC for new NISAR products, downloads new data, re-runs detection, and raises leveled alerts (`INFO`/`WARNING`/`CRITICAL`).

```bash
glews monitor -c config/nepal_2026.yaml --check-now      # single site, one pass
glews monitor -c config/global_watch.yaml --interval 6   # multi-site, every 6h
```

### 10. Tactical alert prioritization (`tactical.py`)

Prioritizes and routes alerts based on urgency, affected population, and responder capacity. Integrates with the cascade risk assessment to produce actionable alert packages.

### 11. Provenance tracking (`provenance.py`)

Records full audit trails for every detection run - input data hashes, configuration snapshots, software versions, and result checksums. Enables reproducibility and forensic review via `glews audit`.

### 12. Benchmarking (`benchmark.py`)

Performance benchmarking suite for profiling detection pipeline throughput, memory usage, and scaling behavior across different scene sizes. Run via `glews benchmark`.

### 13. Alerting (`alerts.py`)

Multi-channel alert dispatch:
- **Email** - SMTP with TLS, configurable recipients
- **Slack** - incoming webhook integration
- **Generic webhook** - HTTP POST with JSON payload

Secrets use `${ENV_VAR}` expansion so credentials stay out of config files.

### 14. Analyst dashboard (`glews dashboard`)

Tier 2 review interface - a single-page web dashboard using Python's built-in `http.server`. Analysts review flagged sites by severity, inspect time-series plots, and classify flags as Watch / Warning / Cleared.

- **Severity levels:** `CRITICAL`, `POST-EVENT`, `WARNING` and `INFO`. A `CRITICAL` flag whose Voight-predicted failure date falls before the site's `event_date` is shown as `POST-EVENT`. This separates failures that have already happened from ones that may still be coming.
- **Map:** a Leaflet map of all flags, with a pulsing marker on the flag being reviewed.
- **Context panels:**
  - Voight failure forecast
  - Elevation and slope
  - Sentinel-2 scenes
  - Current weather
  - Downstream exposure
  - Flag history
  - Monitoring cadence
  - In-situ instruments
  - Field reconnaissance
  - A news feed that mixes academic and media sources
- **Response protocol:** inside the Analyst Classification panel, for flags with a Voight fit. It drafts an authority notification or an evacuation advisory, with Copy and Close controls.
- **Synthetic-data banner:** shown when `flags.geojson` is tagged `data_source: synthetic`.
- **Themes:** light, dark, and system.

---

## Global Watchlist

The built-in watchlist (`config/global_watch.yaml`) monitors 13 sites:

| Region | Sites |
|--------|-------|
| High Mountain Asia | Nepal-Tibet Border 2026, Chamoli (Uttarakhand), Sedongpu Glacier (SE Tibet), Karakoram (Baltoro/Siachen), Aru Glaciers (W. Tibet), Kyagar Glacier (Karakoram) |
| European Alps | Grandes Jorasses (Mont Blanc), Aletsch Glacier (Valais) |
| Polar ice sheets | Sermeq Kujalleq (Jakobshavn, Greenland), Thwaites Glacier (W. Antarctica) |
| Americas | Mount Meager (British Columbia), Huascaran (Cordillera Blanca, Peru) |
| Caucasus | Kolka Glacier (North Ossetia) |

---

## Configuration

Site-specific parameters are defined in YAML files under `config/`. See `config/nepal_2026.yaml` for a fully documented example covering:

- Study area coordinates and buffer
- Sentinel-1 search parameters (date range, polarization, track)
- InSAR processing settings (DEM source, network method, unwrapping)
- Detection thresholds (z-score sigma, minimum cluster size, Voight parameters)
- Cascade risk filters (minimum volume, angle of reach, valley width)
- Alert channels (email, Slack, webhook) with `${ENV_VAR}` secret expansion

---

## Deployment

Production deployment options are documented in [docs/deployment.md](docs/deployment.md):

- **Docker / Docker Compose** - `docker compose up -d` starts monitor + dashboard
- **Kubernetes** - Helm-style manifests in `deploy/kubernetes/`
- **systemd** - unit file at `deploy/systemd/glews-monitor.service`
- **cron** - periodic single-pass checks via `deploy/cron/glews-check.cron`
- **Monitoring** - Prometheus metrics and health checks in `deploy/monitoring/`

Environment variables control all deployment configuration - see the [deployment guide](docs/deployment.md) for the full reference.

---

## Development

```bash
pip install -e ".[dev]"
pytest                     # run 419 tests across 18 files
pytest -v --tb=short       # verbose with short tracebacks
pytest tests/test_integration.py  # end-to-end pipeline test
```

All tests use synthetic data and run without network access or real satellite products. GitHub Actions CI runs the full test suite on every push to `main` and on pull requests.

---

## Data Requirements

Real (non-synthetic) data access requires a free [NASA Earthdata Login](https://urs.earthdata.nasa.gov/) account. Configure credentials in `~/.netrc` (permissions `600`):

```
machine urs.earthdata.nasa.gov
    login <your-earthdata-username>
    password <your-earthdata-password>
```

Additional data for full-pipeline runs:
- **Sentinel-1 SLC** or **NISAR GUNW/GOFF** products - fetched via `glews search`/`glews download`/`glews monitor`
- **DEM** (for Tier 1 cascade assessment) - e.g. Copernicus GLO-30
- **Population data** (optional) - e.g. WorldPop raster for exposure estimates

The synthetic demo (`glews demo`) requires none of the above.

---

## Key Algorithms

### Acceleration z-score

Each site is compared to its own historical behavior - a glacier that routinely accelerates every summer is not flagged for its summer acceleration, only when acceleration exceeds its own historical range:

```
z(t) = (a(t) - mu_baseline) / sigma_baseline
```

### BOCPD changepoint detection

Bayesian Online Changepoint Detection (Adams & MacKay 2007) with a Student-t observation model identifies abrupt regime shifts in real time as new acquisitions arrive.

### Step-change detection

Detects sudden displacement jumps between consecutive acquisitions. These are the kind of discrete steps that a smooth acceleration trend, and Voight's law, both miss. The earlier Nepal 2026 run over the original study area showed this pattern. ICIMOD's account of the source zone instead describes gradual movement, up to 40 cm over 48 days.

### Voight's failure law

Predicts failure time from the inverse-velocity trend (1/v decreasing linearly toward zero). Originally developed for volcanic eruption forecasting (Voight 1988), applied here to glacier and rock-slope collapse prediction.

### Scheidegger runout model

Volume-dependent mobility relation for estimating runout distance from height drop and estimated failure volume (Scheidegger 1973).

---

## License

MIT (see `pyproject.toml`).

## References

- Adams, R.P. & MacKay, D.J.C. "Bayesian Online Changepoint Detection." arXiv:0710.3742 (2007).
- Scheidegger, A.E. "On the prediction of the reach and velocity of catastrophic landslides." *Rock Mechanics* 5, 231-236 (1973).
- Wang, W. et al. "Early warning system for glacial lake outburst floods in Cirenmaco, Tibet." *Int. J. Disaster Risk Reduct.* 73, 102914 (2022).
- Voight, B. "A method for prediction of volcanic eruptions." *Nature* 332, 125-130 (1988).
- Shirzaei, M. "Satellite images before Nepal disaster showed warning signs." *Nature* News Q&A, 2 September 2026.
- ICIMOD. *Understanding the Gyirong–Rasuwa disaster of 26 August 2026: A preliminary synthesis.* International Centre for Integrated Mountain Development (2026). https://doi.org/10.53055/ICIMOD.1150
- Rosen, P.A. et al. "An InSAR time series approach." *J. Geophys. Res.* (2004). [ISCE-2 framework]
- Yunjun, Z. et al. "Small baseline InSAR time series analysis." *Comput. Geosci.* (2019). [MintPy]
