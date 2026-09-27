# Contributing to GLEWS

Guidelines for developing, testing, and extending the Glacier Early
Warning System.

---

## Development Environment

### Prerequisites

- Python >= 3.10
- Git

### Setup

```bash
git clone <repo-url>
cd glews

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

The `dev` extra installs `pytest` and `pytest-cov`. The optional
`processing` extra adds MintPy (only needed when running the ISCE-2 /
MintPy InSAR processing steps; not required for detection, analysis,
or testing).

### Verify your installation

```bash
glews info       # prints version, dependency status, available configs
glews demo       # runs the full pipeline on synthetic data
pytest -v       # runs the test suite
```

---

## Code Style

- Follow existing patterns in the codebase. The project uses standard
  Python conventions (PEP 8) with type hints throughout.
- Use **numpy** for all numeric computation. No pandas or xarray.
- Use **dataclasses** for structured return types (see `AnomalyFlag`,
  `CascadeAssessment`, `AccelerationMap`, etc.).
- Docstrings follow the **numpydoc** convention (Parameters, Returns,
  Attributes sections).
- Logging goes through the `logging` module (`logger = logging.getLogger(__name__)`).
  Never print to stdout from library code; only the CLI (`cli.py`) uses
  `click.echo`.
- Keep dependencies minimal. The core pipeline (everything except
  `process.py`) runs with only the packages listed in
  `pyproject.toml [project.dependencies]`.

---

## Running Tests

```bash
# Run the full suite with verbose output
pytest -v

# Run a single test file
pytest tests/test_timeseries.py -v

# Run with coverage
pytest --cov=glews --cov-report=term-missing
```

Tests are in the `tests/` directory. Each module has a corresponding
test file (`test_<module>.py`). The shared fixtures live in
`tests/conftest.py`.

All tests run without external services, real satellite data, or
processing tools. They use synthetic data from `conftest.py` fixtures
or inline test data.

---

## Project Structure

```
glews/
  config/                  # Site configuration YAML files
    global_watch.yaml      # Multi-site monitoring config
    nepal_2026.yaml        # Nepal-Tibet border collapse scenario
    weisshorn_2026.yaml    # Weisshorn hanging glacier (live)
    chamoli_2021.yaml      # Chamoli retrospective case
    kolka_2002.yaml        # Kolka retrospective case
    aru_2016.yaml          # Aru retrospective case
  src/glews/                # Main package
    __init__.py            # Version string
    acquire.py             # Sentinel-1 SLC search and download (ASF)
    alerts.py              # Email, Slack, webhook alert channels
    atmosphere.py          # Atmospheric phase screen detection/correction
    cascade.py             # Tier 1 cascade risk assessment
    classifier.py          # Transfer learning pre-failure classifier
    cli.py                 # Click CLI entry points
    crosscheck.py          # Tier 1 multi-sensor quality filters
    dashboard.py           # Tier 2 analyst review web UI
    detect.py              # Tier 0 anomaly detection (clustering, scoring)
    mapview.py             # Leaflet.js interactive map viewer
    monitor.py             # Operational monitoring loop
    nisar.py               # NISAR GUNW/GOFF HDF5 loading and SBAS inversion
    process.py             # ISCE-2 + MintPy processing orchestration
    report.py              # Matplotlib plots, GeoJSON export, Markdown reports
    synthetic.py           # Synthetic InSAR data generator
    timeseries.py          # Time-series decomposition, acceleration, BOCPD
    validate.py            # Configuration file validation
  tests/                   # Pytest test suite
  notebooks/               # Jupyter analysis notebooks
  deploy/                  # Deployment configs (Kubernetes, etc.)
  docker-compose.yml       # Docker Compose for local deployment
  Dockerfile               # Container image definition
  pyproject.toml           # Build system and dependencies
```

---

## How To: Add a New Detection Method

The detection pipeline has two layers: low-level time-series algorithms
in `timeseries.py`, and spatial clustering / scoring in `detect.py`.

### 1. Implement the algorithm in `timeseries.py`

Define a dataclass for your results and a function that operates on
dates + displacement arrays:

```python
@dataclass
class MyDetectionResult:
    """Results from my new detector."""
    dates: np.ndarray
    flagged: np.ndarray       # bool, shape [n_epochs, n_rows, n_cols]
    score: np.ndarray         # float, shape [n_epochs, n_rows, n_cols]

def detect_my_pattern(
    dates: np.ndarray,
    displacement: np.ndarray,
    threshold: float = 3.0,
) -> MyDetectionResult:
    ...
```

Follow the patterns of `detect_step_changes()` or
`bocpd_changepoints()`.

### 2. Wire it into `detect.py`

Add a private function `_detect_my_pattern_anomalies()` that:

1. Calls your `timeseries.py` function
2. Runs `_cluster_anomalous_pixels()` on the resulting boolean map
3. Builds `AnomalyFlag` objects for each cluster
4. Tags each flag with `detection_details["tag"] = "my_pattern"`

Then call it from `detect_anomalies()`, gated on a config key:

```python
if det.get("my_pattern", {}).get("enabled", False):
    my_flags = _detect_my_pattern_anomalies(
        dates, displacement, latitude, longitude, config, flag_id,
    )
    all_flags.extend(my_flags)
    flag_id += len(my_flags)
```

### 3. Add configuration

Add default parameters under `detect.my_pattern` in the relevant site
YAML configs (or in a `_DEFAULT_*` dict if the method has its own
defaults). Register any new numeric fields in `validate.py`
`POSITIVE_FIELDS` or `UNIT_RANGE_FIELDS` as appropriate.

### 4. Add tests

Create test cases in `tests/test_timeseries.py` for the low-level
algorithm and in `tests/test_detect.py` for the integration with
spatial clustering. Use synthetic data with a known injected signal so
you can assert that the detector finds it.

---

## How To: Add a New Monitored Site

### 1. Create a site YAML config

Copy an existing config (e.g. `config/weisshorn_2026.yaml`) and edit:

```yaml
site:
  name: "My New Site"
  latitude: 46.0
  longitude: 7.7
  buffer_km: 10
  event_date: null  # or "2026-08-01" for retrospective

acquire:
  platform: NISAR
  product_types: ["GUNW", "GOFF"]
  gunw_dir: "data/nisar/my_site/gunw"
  goff_dir: "data/nisar/my_site/goff"

detect:
  # ... detection parameters (copy defaults from an existing config
  #     and tune sigma_threshold, window_size_days, etc. for your site)
```

Validate with:

```bash
glews validate -c config/my_site.yaml
```

### 2. Add to `global_watch.yaml` (for operational monitoring)

Add an entry under `sites:` in `config/global_watch.yaml`:

```yaml
sites:
  - site:
      name: "My New Site"
      latitude: 46.0
      longitude: 7.7
      buffer_km: 10
    # Per-site overrides (optional):
    detect:
      acceleration:
        sigma_threshold: 2.0
```

Shared defaults (`detect`, `cascade`, `monitor`, `alerts`) from the
top level of `global_watch.yaml` are merged with per-site overrides
automatically.

### 3. Test

```bash
glews monitor -c config/my_site.yaml --check-now
```

---

## How To: Add a New Alert Channel

Alert channels live in `alerts.py`. Each channel is a subclass of
`AlertChannel`.

### 1. Implement the channel class

```python
class PagerDutyChannel(AlertChannel):
    """PagerDuty integration via Events API v2."""

    name = "pagerduty"

    def __init__(self, routing_key: str, timeout: int = 30) -> None:
        self.routing_key = routing_key
        self.timeout = timeout

    @classmethod
    def from_config(cls, cfg: dict) -> "PagerDutyChannel | None":
        key = cfg.get("routing_key", "")
        if not key:
            logger.warning("PagerDuty channel: routing_key missing; skipping")
            return None
        return cls(routing_key=key, timeout=int(cfg.get("timeout", 30)))

    def send(
        self,
        alert_level: str,
        site_name: str,
        message: str,
        details: dict[str, Any],
    ) -> None:
        # Build and POST the PagerDuty Events API v2 payload
        ...
```

### 2. Register it in the factory map

Add your class to `_CHANNEL_FACTORIES` at the bottom of `alerts.py`:

```python
_CHANNEL_FACTORIES: dict[str, type[AlertChannel]] = {
    "email": EmailChannel,
    "slack": SlackChannel,
    "webhook": WebhookChannel,
    "pagerduty": PagerDutyChannel,  # <-- new
}
```

### 3. Configure it in site YAML

```yaml
alerts:
  pagerduty:
    routing_key: "${PAGERDUTY_KEY}"
```

Secrets use `${ENV_VAR}` expansion (handled by `expand_env_vars()`).

### 4. Add tests

Test the channel in `tests/test_alerts.py`. Mock the HTTP call and
verify the payload structure.

---

## Pull Request Process

1. **Branch** from `main`. Use a descriptive branch name
   (`feature/bocpd-detector`, `fix/coherence-mask`, etc.).

2. **Write tests** for any new functionality. Aim for the same
   coverage level as the surrounding code.

3. **Run the full test suite** locally before pushing:
   ```bash
   pytest -v
   ```

4. **Open a PR** against `main`. The CI pipeline (GitHub Actions) runs
   `pytest -v` on Python 3.11.

5. All CI checks must pass before merge.

---

## CI Requirements

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every
push to `main` and on every pull request:

- Checks out the code
- Sets up Python 3.11
- Installs `pip install -e ".[dev]"`
- Runs `pytest -v`

Make sure your changes pass locally before pushing. The test suite runs
without any external services or credentials.
