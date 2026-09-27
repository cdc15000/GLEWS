# GLEWS API Reference

Public API for all modules in `src/glews/`.

---

## `glews.acquire` — Sentinel-1 SLC Search and Download

Search and download Sentinel-1 SLC scenes from the Alaska Satellite
Facility (ASF) DAAC archive.

### `SceneInfo`

```python
@dataclass
class SceneInfo:
    granule: str
    start_time: str
    path_number: int
    frame_number: int
    polarization: str
    url: str
    file_size_mb: float
    geometry: dict          # GeoJSON geometry
```

Metadata for a single Sentinel-1 SLC scene.

**Class methods:**

- `SceneInfo.from_asf_result(result: asf.ASFProduct) -> SceneInfo` —
  Construct from an ASF search result.

### Functions

```python
def search_scenes(config: dict) -> list[SceneInfo]
```

Search ASF archive for Sentinel-1 SLC scenes covering the study area.
`config` must contain `site` (with `latitude`, `longitude`, `buffer_km`)
and `acquire` (with `start_date`, `end_date`) sections.

```python
def select_track(
    scenes: list[SceneInfo],
    path_number: int | None = None,
) -> list[SceneInfo]
```

Select scenes from a single orbital track. Auto-selects the track with
the most scenes when `path_number` is None.

```python
def download_scenes(
    scenes: list[SceneInfo],
    output_dir: str | Path = "data/slc",
    n_workers: int = 4,
    username: str | None = None,
    password: str | None = None,
) -> list[Path]
```

Download SLC scenes from ASF. Requires Earthdata Login credentials
(via arguments, `~/.netrc`, or `EARTHDATA_USER`/`EARTHDATA_PASS`
environment variables). Skips already-downloaded files.

```python
def summarize_scenes(scenes: list[SceneInfo]) -> str
```

Return a human-readable summary of a scene collection (track counts,
date ranges, revisit intervals, total download size).

---

## `glews.alerts` — Alert Dispatch

Email (SMTP), Slack (webhook), and generic webhook alert channels.

### `AlertChannel` (ABC)

```python
class AlertChannel(ABC):
    name: str = "base"

    @abstractmethod
    def send(
        self,
        alert_level: str,
        site_name: str,
        message: str,
        details: dict[str, Any],
    ) -> None: ...
```

Abstract base for a notification channel. Subclasses must implement
`send()` and provide a `from_config(cfg: dict)` class method.

### Built-in Channels

- **`EmailChannel`** — SMTP-based email alerts with HTML formatting.
  Config keys: `smtp_host`, `smtp_port`, `from`, `to`, `username`,
  `password`.
- **`SlackChannel`** — Slack incoming-webhook alerts with Block Kit
  formatting. Config key: `webhook_url`.
- **`WebhookChannel`** — POSTs a JSON payload to an arbitrary URL.
  Config keys: `url`, `headers`.

### `AlertDispatcher`

```python
class AlertDispatcher:
    def __init__(self, channels: list[AlertChannel] | None = None) -> None: ...

    @classmethod
    def from_config(cls, site_config: dict) -> AlertDispatcher: ...

    def dispatch(
        self,
        alert_level: str,
        site_name: str,
        message: str,
        details: dict[str, Any],
    ) -> dict[str, bool]: ...
```

Fan-out dispatcher: sends an alert to every configured channel. Returns
a dict mapping channel name to success/failure. Logs failures without
crashing.

### Utility

```python
def expand_env_vars(value: Any) -> Any
```

Recursively expand `${VAR_NAME}` references in strings, dicts, and
lists. Missing variables expand to the empty string.

---

## `glews.atmosphere` — Atmospheric Phase Screen Detection

Detect and correct atmospheric phase screens (APS) that contaminate
InSAR displacement measurements.

### `APSResult`

```python
@dataclass
class APSResult:
    contaminated_epochs: np.ndarray    # bool [n_epochs]
    stratified_scores: np.ndarray      # R^2 per epoch [n_epochs]
    turbulent_scores: np.ndarray       # spectral fraction [n_epochs]
    correction_applied: bool
    residual_displacement: np.ndarray | None   # [n_epochs, n_rows, n_cols]
```

### `APSDetector`

```python
class APSDetector:
    def estimate_aps_correlation(
        self,
        displacement_map: np.ndarray,   # [n_epochs, n_rows, n_cols]
        dem: np.ndarray,                 # [n_rows, n_cols]
        dates: np.ndarray,               # [n_epochs]
    ) -> np.ndarray: ...                 # R^2 per epoch

    def detect_turbulent_aps(
        self,
        displacement_map: np.ndarray,
        dates: np.ndarray,
        spatial_scale_km: float = 10.0,
    ) -> np.ndarray: ...                 # turbulent score per epoch

    def flag_aps_contaminated_epochs(
        self,
        dates: np.ndarray,
        displacement_stack: np.ndarray,
        dem: np.ndarray | None = None,
        thresholds: dict | None = None,
    ) -> APSResult: ...

    def correct_stratified_aps(
        self,
        displacement_map: np.ndarray,
        dem: np.ndarray,
    ) -> np.ndarray: ...                 # corrected displacement
```

**Thresholds dict keys** (for `flag_aps_contaminated_epochs`):
- `stratified_r2` (default 0.6) — R^2 above which an epoch is flagged
- `turbulent_fraction` (default 0.4) — atmospheric-band power fraction
- `spatial_scale_km` (default 10.0) — physical extent of the map

---

## `glews.cascade` — Tier 1 Cascade Risk Assessment

Evaluate whether a flagged unstable site can produce a dangerous
downstream cascade.

### Enums

```python
class RiskLevel(Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    MINIMAL = "minimal"
```

### `ExposureResult`

```python
@dataclass
class ExposureResult:
    total_population_exposed: int
    population_by_distance: dict[str, int]  # {"0-10km": 500, ...}
    exposure_score: float                    # 0-10 composite
    settlements_at_risk: list[dict]
    runout_distance_km: float
    angle_of_reach_deg: float
```

### `CascadeAssessment`

```python
@dataclass
class CascadeAssessment:
    flag: AnomalyFlag
    risk_level: RiskLevel
    estimated_volume_m3: float
    volume_sufficient: bool
    slope_angle_deg: float
    elevation_m: float
    above_valley: bool
    valley_width_m: float | None
    runout_distance_m: float
    runout_reaches_river: bool
    angle_of_reach_deg: float
    population_exposed: int
    settlements_exposed: list[str]
    infrastructure_at_risk: list[str]
    exposure: ExposureResult | None
    dam_potential: bool
    estimated_lake_volume_m3: float
    cascade_score: float

    @property
    def passes_tier1(self) -> bool: ...  # True for HIGH or MODERATE
```

### Functions

```python
def estimate_runout(
    elevation_drop_m: float,
    horizontal_distance_m: float,
    volume_m3: float | None = None,
) -> dict[str, float]
```

Empirical angle-of-reach runout estimation using Scheidegger (1973)
volume-dependent mobility relation.

```python
def estimate_exposure(
    flag_lat: float,
    flag_lon: float,
    flag_elevation_m: float,
    runout_distance_km: float,
    population_data: np.ndarray | str | None = None,
) -> ExposureResult
```

Estimate population exposure within the runout zone.

```python
def assess_cascade_risk(
    flag: AnomalyFlag,
    dem_data: np.ndarray | None = None,
    population_data: np.ndarray | None = None,
    config: dict | None = None,
) -> CascadeAssessment
```

Single-flag cascade risk assessment combining volume estimation,
runout modeling, and exposure assessment.

```python
def assess_cascade_risk_batch(
    flags: list[AnomalyFlag],
    dem: np.ndarray,
    dem_lat: np.ndarray,
    dem_lon: np.ndarray,
    config: dict,
    population_grid: np.ndarray | None = None,
) -> list[CascadeAssessment]
```

Batch assessment for multiple flags. Returns assessments sorted by
cascade_score (highest risk first).

```python
def generate_exposure_summary(
    assessments: list[CascadeAssessment],
) -> str
```

Produce a human-readable summary table of population exposure by site.

---

## `glews.classifier` — Pre-Failure Pattern Classifier

Transfer learning classifier for pre-failure pattern detection using
logistic regression (numpy only).

### Constants

```python
FEATURE_NAMES: list[str]   # 13 canonical feature names
MODEL_VERSION = 1
```

### `PrecursorFeatureExtractor`

```python
class PrecursorFeatureExtractor:
    def __init__(
        self,
        baseline_fraction: float = 0.3,
        threshold_k: float = 2.0,
    ) -> None: ...

    def extract_features(
        self,
        dates: np.ndarray,
        displacement: np.ndarray,
    ) -> np.ndarray: ...   # feature vector, length len(FEATURE_NAMES)
```

Extracts a fixed-length feature vector from a displacement time series.
Features include velocity statistics, acceleration statistics, velocity
trend, temporal asymmetry, threshold crossings, and anomalous
acceleration duration.

### `PrecursorClassifier`

```python
class PrecursorClassifier:
    def fit(
        self,
        X: np.ndarray,          # [n_samples, n_features]
        y: np.ndarray,          # binary labels [n_samples]
        lr: float = 0.1,
        n_iter: int = 500,
        reg_lambda: float = 0.01,
    ) -> list[float]: ...       # loss per iteration

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray: ...
    def save(self, path: str | Path) -> None: ...

    @classmethod
    def load(cls, path: str | Path) -> PrecursorClassifier: ...
```

Binary logistic regression with L2 regularization and built-in feature
standardization.

### `LandslideTrainingData`

```python
class LandslideTrainingData:
    def __init__(self, seed: int = 42) -> None: ...

    def generate_synthetic_training_set(
        self,
        n_positive: int = 200,
        n_negative: int = 800,
        n_dates: int = 60,
    ) -> tuple[np.ndarray, np.ndarray]: ...  # (X, y)
```

Generates synthetic displacement time series for training. Positive
class: exponential acceleration / Voight-like runaway. Negative class:
seasonal motion, steady creep, transient speedups, quiet.

### Functions

```python
def train_precursor_model(
    output_path: str | Path,
    seed: int = 42,
    n_positive: int = 200,
    n_negative: int = 800,
    n_iter: int = 500,
    lr: float = 0.1,
) -> dict
```

Generate training data, train, evaluate, and save the model. Returns
evaluation metrics (accuracy, precision, recall).

---

## `glews.cli` — Command-Line Interface

Click-based CLI. Entry point: `glews = "glews.cli:main"`.

### Commands

| Command | Description |
|---------|-------------|
| `glews search -c CONFIG` | Search ASF for available Sentinel-1 scenes |
| `glews download -c CONFIG` | Download SLC scenes from ASF |
| `glews process -c CONFIG [--step STEP] [--dry-run]` | Run ISCE-2 + MintPy processing |
| `glews detect -c CONFIG [-o DIR]` | Run Tier 0 anomaly detection |
| `glews assess -c CONFIG` | Run Tier 1 cascade risk assessment |
| `glews monitor -c CONFIG [--interval H] [--check-now]` | Continuous monitoring loop |
| `glews dashboard -c CONFIG [--port P] [--data-dir DIR]` | Launch Tier 2 analyst dashboard |
| `glews map [--data-dir DIR] [--port P]` | Launch interactive map viewer |
| `glews demo [-o DIR] [--no-plots]` | Run full pipeline on synthetic data |
| `glews validate -c CONFIG` | Validate a configuration file |
| `glews info` | Show version, dependencies, available configs |
| `glews version` | Show version |

---

## `glews.crosscheck` — Tier 1 Quality Filters

Cross-check InSAR anomalies against quality metrics and optical imagery.

### `CrossCheckResult`

```python
@dataclass
class CrossCheckResult:
    quality_score: float           # coherence quality, 0-1
    spatial_consistency: float     # connected component fraction, 0-1
    temporal_consistency: float    # epoch presence fraction, 0-1
    optical_confirmation: bool | None
    recommendation: str            # "retain", "demote", or "remove"
    details: dict
```

### `InSARQualityChecker`

```python
class InSARQualityChecker:
    def check_coherence_quality(
        self,
        coherence_map: np.ndarray,   # [n_rows, n_cols]
        flag_mask: np.ndarray,        # bool [n_rows, n_cols]
        min_coherence: float = 0.3,
    ) -> float: ...

    def check_spatial_consistency(
        self,
        displacement_map: np.ndarray,  # [n_rows, n_cols]
        flag_mask: np.ndarray,
        min_connected_fraction: float = 0.5,
    ) -> float: ...

    def check_temporal_consistency(
        self,
        dates: np.ndarray,
        displacement_stack: np.ndarray,  # [n_epochs, n_rows, n_cols]
        flag_mask: np.ndarray,
        max_gap_fraction: float = 0.3,
    ) -> float: ...
```

### `OpticalCrossChecker`

```python
class OpticalCrossChecker:
    def check_surface_change(
        self,
        bbox: tuple[float, float, float, float],
        date_range: tuple[str, str],
    ) -> CrossCheckResult: ...
```

Stub implementation. Returns `optical_confirmation=None` (Sentinel-2
access not yet configured).

### Functions

```python
def apply_tier1_filters(
    flags: list[AnomalyFlag],
    displacement_stack: np.ndarray,
    coherence_map: np.ndarray,
    dates: np.ndarray,
    config: dict,
) -> list[AnomalyFlag]
```

Run all Tier 1 cross-check filters. Annotates each flag with
`detection_details["tier1_quality"]`. Removes flags recommended for
removal; halves score of demoted flags. Returns filtered, re-sorted
flags.

---

## `glews.dashboard` — Tier 2 Analyst Dashboard

Lightweight web UI for glaciologists to review flagged sites.

### Functions

```python
def load_flags(data_dir: str | Path) -> list[dict]
```

Load anomaly flags from `flags.geojson` or `flags.json`.

```python
def load_state(state_path: str | Path) -> dict
def save_state(state_path: str | Path, state: dict) -> None
```

Persist/load analyst classifications (Watch/Warning/Cleared + notes).

```python
def serve(data_dir: str, port: int, config: dict) -> None
```

Start the dashboard HTTP server (blocking). Serves at
`http://127.0.0.1:<port>/`. Supports `GET /` (HTML dashboard),
`GET /api/flags` (JSON), and `POST /api/classify` (analyst actions).

---

## `glews.detect` — Tier 0 Anomaly Detection

Identify spatially coherent clusters of anomalous acceleration.

### `AnomalyFlag`

```python
@dataclass
class AnomalyFlag:
    flag_id: int
    score: float                    # composite anomaly score
    peak_zscore: float
    mean_zscore: float
    n_pixels: int
    area_m2: float
    center_lat: float
    center_lon: float
    peak_lat: float
    peak_lon: float
    acceleration_m_yr2: float
    pixel_indices: np.ndarray       # [n_pixels, 2] row/col
    window_index: int
    window_date: float              # ordinal date
    voight_fit: dict | None         # predicted failure time, R^2
    timeseries: dict | None         # {"dates": [...], "values": [...]}
    detection_details: dict
```

### Functions

```python
def detect_anomalies(
    accel_map: AccelerationMap,
    latitude: np.ndarray,
    longitude: np.ndarray,
    config: dict,
    dates: np.ndarray | None = None,
    displacement: np.ndarray | None = None,
) -> list[AnomalyFlag]
```

Run Tier 0 anomaly detection. Algorithm:
1. Threshold z-score maps at each time window
2. Cluster anomalous pixels with DBSCAN
3. Filter by minimum cluster size and area
4. Score and rank
5. Optionally run step-change detection (if dates/displacement provided)
6. Optionally run BOCPD detection (if `detect.changepoint.method == "bocpd"`)
7. Deduplicate overlapping spatial clusters
8. Optionally fit Voight's law to top candidates
9. Optionally score with the pre-failure classifier

Returns flags sorted by score (highest first).

---

## `glews.mapview` — Interactive Map Viewer

Leaflet.js-based map showing color-coded markers for flagged sites.

### Functions

```python
def load_geojson(data_dir: str | Path) -> dict
```

Load `flags.geojson` from the given directory. Returns an empty
FeatureCollection when the file does not exist.

```python
def serve(data_dir: str | Path = "output", port: int = 8050) -> None
```

Start the map viewer HTTP server (blocking). Serves at
`http://localhost:<port>/`. Endpoints: `GET /` (HTML page with Leaflet
map), `GET /api/flags.geojson` (GeoJSON data).

---

## `glews.monitor` — Operational Monitoring

Continuous monitoring loop that watches configured sites for new NISAR
acquisitions, downloads new scenes, runs the detection pipeline, and
generates alerts.

### `MonitorState`

```python
@dataclass
class MonitorState:
    site_name: str
    last_check: str | None
    known_products: set[str]
    downloaded_products: set[str]
    alerted_signatures: set[str]
    n_checks: int

    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> MonitorState: ...
    @classmethod
    def load(cls, path: str | Path, site_name: str) -> MonitorState: ...
    def save(self, path: str | Path) -> None: ...
```

Persistent state checkpointed to disk so a restart resumes without
re-processing or re-alerting.

### Functions

```python
def check_new_data(
    site_config: dict,
    state: MonitorState | None = None,
) -> list[SceneInfo]
```

Query ASF DAAC for NISAR products not yet seen.

```python
def process_new_acquisition(
    products: list[SceneInfo],
    site_config: dict,
    state: MonitorState,
) -> dict[str, Any]
```

Download new products, re-run detection on the full stack. Returns
`{"flags": [...], "ts": NISARTimeseries | None, "accel_map": ..., "source": ...}`.

```python
def build_alerts(
    flags: list[AnomalyFlag],
    ts: NISARTimeseries | None,
    site_config: dict,
    state: MonitorState,
) -> list[dict]
```

Convert new flags into alert records, skipping already-alerted anomalies.

```python
def write_alerts(alerts: list[dict], site_config: dict) -> Path | None
```

Append new alerts to the site's alert JSON file.

```python
def iter_site_configs(config_path: str | Path) -> Iterator[dict]
```

Yield one per-site config dict for each site in the config file.
Supports both single-site and multi-site (`sites:` list) formats.

```python
def run_check_cycle(site_config: dict) -> list[dict]
```

One complete check-download-detect-alert cycle for a single site.

```python
def run_monitoring_loop(
    config_path: str | Path,
    interval_hours: float = 6,
    check_now_only: bool = False,
    max_iterations: int | None = None,
) -> None
```

Main operational loop.

---

## `glews.nisar` — NISAR Data Loading

Load NISAR L2 GUNW (unwrapped interferograms) and GOFF (offset fields)
HDF5 products and invert to displacement time series.

### `NISARTimeseries`

```python
@dataclass
class NISARTimeseries:
    dates: np.ndarray             # ordinal days [n_dates]
    date_strings: list[str]
    displacement: np.ndarray      # [n_dates, n_rows, n_cols] meters
    velocity: np.ndarray          # [n_rows, n_cols] m/yr
    coherence: np.ndarray         # [n_rows, n_cols]
    latitude: np.ndarray          # [n_rows, n_cols]
    longitude: np.ndarray         # [n_rows, n_cols]
    source: str                   # "GUNW" or "GOFF"
    metadata: dict
```

### Functions

```python
def load_nisar_gunw_stack(
    gunw_dir: str | Path,
    min_coherence: float = 0.3,
    track: int | None = None,
    crop_to_site: tuple[float, float, float] | None = None,
) -> NISARTimeseries
```

Load NISAR GUNW products and invert to displacement time series via
SBAS (weighted least squares). Uses L-band wavelength (0.2384 m).

```python
def load_nisar_goff_stack(
    goff_dir: str | Path,
    min_correlation: float = 0.1,
    track: int | None = None,
    crop_to_site: tuple[float, float, float] | None = None,
) -> NISARTimeseries
```

Load NISAR GOFF (pixel offset tracking) products.

---

## `glews.process` — InSAR Processing Orchestration

Wraps ISCE-2 and MintPy for interferogram generation and time-series
inversion. Requires ISCE-2, MintPy, and SNAPHU to be installed.

### `DisplacementTimeseries`

```python
@dataclass
class DisplacementTimeseries:
    dates: np.ndarray              # ordinal days [n_dates]
    date_strings: list[str]
    displacement: np.ndarray       # [n_dates, n_rows, n_cols] meters
    velocity: np.ndarray           # [n_rows, n_cols] m/yr
    temporal_coherence: np.ndarray # [n_rows, n_cols], 0-1
    latitude: np.ndarray           # [n_rows, n_cols]
    longitude: np.ndarray          # [n_rows, n_cols]
    metadata: dict
```

### `InSARProcessor`

```python
class InSARProcessor:
    def __init__(self, config: dict): ...
    def prepare_stack(self) -> Path: ...
    def run_isce(self, dry_run: bool = False) -> None: ...
    def generate_mintpy_config(self) -> Path: ...
    def run_mintpy(self, dry_run: bool = False) -> None: ...
    def load_timeseries(
        self, min_coherence: float | None = None,
    ) -> DisplacementTimeseries: ...
```

---

## `glews.report` — Report Generation

Generate visual reports and GeoJSON output for flagged sites.

### Functions

```python
def generate_report(
    flags: list[AnomalyFlag],
    assessments: list[CascadeAssessment] | None,
    accel_map: AccelerationMap,
    latitude: np.ndarray,
    longitude: np.ndarray,
    config: dict,
    output_dir: str | Path = "output",
) -> Path
```

Generate a complete analysis report including:
- Spatial anomaly map (PNG)
- Per-flag time-series plots (PNG)
- Acceleration evolution plot (PNG)
- GeoJSON export (`flags.geojson`)
- Markdown summary (`report.md`)

Returns path to the main report file.

---

## `glews.synthetic` — Synthetic Data Generator

Generate realistic InSAR displacement time-series data for testing.

### `SyntheticConfig`

```python
@dataclass
class SyntheticConfig:
    n_rows: int = 200
    n_cols: int = 250
    lat_min: float = 28.10
    lat_max: float = 28.30
    lon_min: float = 85.80
    lon_max: float = 86.00
    start_date: str = "2025-01-08"
    end_date: str = "2026-08-18"
    revisit_days: int = 12
    base_velocity_m_yr: float = 0.05
    velocity_spatial_variation: float = 0.3
    annual_amplitude_m: float = 0.005
    semi_annual_amplitude_m: float = 0.002
    atmospheric_noise_m: float = 0.004
    measurement_noise_m: float = 0.001
    base_coherence: float = 0.85
    low_coherence_fraction: float = 0.1
    failure_zones: list[dict] | None = None
```

Each failure zone dict: `{center_row, center_col, radius_pixels,
onset_days_before_end, max_acceleration_m_yr2, ramp_type}`.

### Functions

```python
def generate_synthetic_scene(
    config: SyntheticConfig | None = None,
) -> DisplacementTimeseries
```

Generate a complete synthetic displacement time-series dataset. Uses
a default Nepal 2026 scenario config when `config` is None.

---

## `glews.timeseries` — Time-Series Analysis

Decomposition, trend fitting, acceleration estimation, step-change
detection, Bayesian Online Changepoint Detection (BOCPD), and Voight's
failure law fitting.

### `DecomposedPixel`

```python
@dataclass
class DecomposedPixel:
    dates: np.ndarray
    displacement: np.ndarray
    trend: np.ndarray
    seasonal: np.ndarray
    residual: np.ndarray
    velocity: float              # m/yr
    velocity_std: float          # m/yr
    amplitude_annual: float      # m
    amplitude_semi: float        # m
    r_squared: float
```

### `AccelerationMap`

```python
@dataclass
class AccelerationMap:
    window_centers: np.ndarray        # ordinal days [n_windows]
    acceleration: np.ndarray          # m/yr^2 [n_windows, n_rows, n_cols]
    acceleration_zscore: np.ndarray   # [n_windows, n_rows, n_cols]
    velocity_residual: np.ndarray     # [n_windows, n_rows, n_cols]
```

### `StepChangeMap`

```python
@dataclass
class StepChangeMap:
    dates: np.ndarray              # ordinal days [n_epochs]
    step_change: np.ndarray        # bool [n_epochs, n_rows, n_cols]
    magnitude: np.ndarray          # meters [n_epochs, n_rows, n_cols]
```

### `ChangePointResult`

```python
@dataclass
class ChangePointResult:
    changepoint_indices: np.ndarray
    run_length_posterior: np.ndarray    # [n_dates, n_dates+1]
    changepoint_probabilities: np.ndarray  # [n_dates]
    most_likely_changepoints: np.ndarray
```

### Functions

```python
def decompose_pixel(
    dates: np.ndarray,
    displacement: np.ndarray,
    n_harmonics: int = 2,
) -> DecomposedPixel | None
```

Decompose a single pixel into linear trend, seasonal harmonics, and
residual via least-squares fit. Returns None if insufficient valid
observations.

```python
def compute_acceleration_map(
    dates: np.ndarray,
    displacement: np.ndarray,
    window_size_days: int = 60,
    step_days: int = 12,
    n_harmonics: int = 2,
) -> AccelerationMap
```

Compute sliding-window acceleration for all pixels. Steps:
1. Remove trend + seasonal (batch least squares)
2. Fit linear velocity in each sliding window on residuals
3. Compute acceleration as velocity change between windows
4. Z-score normalize against each pixel's baseline (first 70% of windows)

```python
def detect_step_changes(
    dates: np.ndarray,
    displacement: np.ndarray,
    sigma_threshold: float = 5.0,
    min_displacement_m: float = 0.5,
    baseline_fraction: float = 0.3,
) -> StepChangeMap
```

Detect abrupt, single-epoch step-change displacement events. Flags
inter-epoch jumps that exceed both a sigma threshold (relative to an
early baseline) and an absolute minimum displacement.

```python
def bocpd_changepoints(
    dates: np.ndarray,
    displacement: np.ndarray,
    hazard_rate: float = 1 / 100,
    prior_variance: float | None = None,
    threshold: float = 0.25,
) -> ChangePointResult
```

Bayesian Online Changepoint Detection (Adams & MacKay 2007). Operates
on the velocity series (first differences) using a Normal-Inverse-Gamma
conjugate prior, giving a Student-t predictive distribution.

```python
def fit_voight(
    dates: np.ndarray,
    velocity: np.ndarray,
    min_points: int = 5,
) -> dict | None
```

Fit Voight's failure law (inverse velocity decreases linearly toward
zero). Returns `{"predicted_failure_date", "r_squared",
"inverse_velocity_slope", "days_until_failure"}` or None if the fit
fails.

---

## `glews.validate` — Configuration Validation

Validate site configuration files against expected structure, types,
ranges, and cross-field consistency.

### Functions

```python
def validate_config(config: dict) -> list[str]
```

Validate a GLEWS configuration dict. Returns a list of diagnostic
strings prefixed with `ERROR:` or `WARNING:`. Empty list means the
configuration passed all checks.

Checks: required fields, latitude/longitude ranges, positive-value
fields, unit-range fields, start_date < end_date, unknown top-level
keys, multi-site validation.

```python
def validate_config_file(path: str) -> tuple[dict, list[str]]
```

Load a YAML config file and validate it. Returns `(config_dict, issues)`.

### Constants

```python
REQUIRED_FIELDS: dict[str, type | tuple[type, ...]]
KNOWN_SECTIONS: set[str]
POSITIVE_FIELDS: list[str]
UNIT_RANGE_FIELDS: list[str]
```
