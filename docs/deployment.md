# Glacier Early Warning System — Deployment Guide

Production deployment options for the GLEWS monitoring pipeline and analyst dashboard.

---

## Prerequisites

- **Docker** >= 20.10 and **Docker Compose** >= 2.0
- A free [NASA Earthdata](https://urs.earthdata.nasa.gov/) account for real satellite data
- Earthdata credentials in `~/.netrc`:
  ```
  machine urs.earthdata.nasa.gov
      login YOUR_USERNAME
      password YOUR_PASSWORD
  ```

---

## Docker Quickstart

Build the GLEWS image and run the synthetic demo to verify the installation:

```bash
# Build
docker build -t glews .

# Verify with the synthetic demo (no credentials needed)
docker run --rm -v "$PWD/output:/app/output" glews demo --output /app/output

# Run a single monitoring check against real data
docker run --rm \
    -v "$HOME/.netrc:/root/.netrc:ro" \
    -v "$PWD/config:/app/config:ro" \
    -v "$PWD/data:/app/data" \
    -v "$PWD/output:/app/output" \
    glews monitor -c config/global_watch.yaml --check-now
```

The image's `ENTRYPOINT` is the `glews` CLI, so any subcommand follows `docker run --rm glews ...` directly.

---

## Docker Compose

The recommended deployment for most installations. Starts both the monitoring loop and the analyst dashboard as long-running services.

```bash
cd /path/to/glews

# Build images
docker compose build

# Start services in the background
docker compose up -d

# View monitor logs
docker compose logs -f monitor

# Stop
docker compose down
```

The dashboard is available at `http://localhost:8080` by default (configurable via `GLEWS_DASHBOARD_PORT`).

### docker-compose.yml overview

```yaml
services:
  monitor:
    build: .
    command: monitor -c ${GLEWS_CONFIG:-config/global_watch.yaml} --interval ${GLEWS_INTERVAL:-6}
    volumes:
      - ${HOME}/.netrc:/root/.netrc:ro
      - ./config:/app/config:ro
      - ./data:/app/data
      - ./output:/app/output
    environment:
      - GLEWS_SMTP_HOST
      - GLEWS_SMTP_PORT
      - GLEWS_SMTP_USER
      - GLEWS_SMTP_PASS
      - GLEWS_SMTP_TO
      - GLEWS_SLACK_WEBHOOK
      - GLEWS_WEBHOOK_URL
    restart: unless-stopped

  dashboard:
    build: .
    command: dashboard -c ${GLEWS_CONFIG:-config/global_watch.yaml} --data-dir /app/output --port 8080
    ports:
      - "${GLEWS_DASHBOARD_PORT:-8080}:8080"
    volumes:
      - ./config:/app/config:ro
      - ./output:/app/output:ro
    restart: unless-stopped
```

### Override file

For persistent customizations without modifying the base compose file:

```bash
cp docker-compose.override.yml.example docker-compose.override.yml
# Edit with your credentials and preferences
```

Docker Compose merges the override file automatically.

---

## Environment Variables

All deployment settings are controlled via environment variables, either exported in the shell or placed in a `.env` file alongside `docker-compose.yml`.

### Core settings

| Variable | Default | Description |
|----------|---------|-------------|
| `GLEWS_CONFIG` | `config/global_watch.yaml` | Site config file (relative to /app) |
| `GLEWS_INTERVAL` | `6` | Check interval in hours |
| `GLEWS_DASHBOARD_PORT` | `8080` | Dashboard listen port |

### Email alerts (SMTP)

| Variable | Default | Description |
|----------|---------|-------------|
| `GLEWS_SMTP_HOST` | *(empty)* | SMTP server hostname |
| `GLEWS_SMTP_PORT` | `587` | SMTP port (587 for STARTTLS) |
| `GLEWS_SMTP_USER` | *(empty)* | SMTP username |
| `GLEWS_SMTP_PASS` | *(empty)* | SMTP password or app password |
| `GLEWS_SMTP_TO` | *(empty)* | Alert recipient email(s), comma-separated |

For Gmail, create an App Password (Account > Security > App Passwords) and use it as `GLEWS_SMTP_PASS`:

```bash
export GLEWS_SMTP_HOST=smtp.gmail.com
export GLEWS_SMTP_PORT=587
export GLEWS_SMTP_USER=you@gmail.com
export GLEWS_SMTP_PASS=xxxx-xxxx-xxxx-xxxx
export GLEWS_SMTP_TO=team@example.com
```

### Slack alerts

| Variable | Default | Description |
|----------|---------|-------------|
| `GLEWS_SLACK_WEBHOOK` | *(empty)* | Slack incoming-webhook URL |

Create an incoming webhook at <https://api.slack.com/messaging/webhooks>.

### Generic webhook

| Variable | Default | Description |
|----------|---------|-------------|
| `GLEWS_WEBHOOK_URL` | *(empty)* | HTTP endpoint for JSON POST alerts |

Alert payloads are JSON objects with `level`, `site`, `summary`, `details`, and `timestamp` fields.

---

## Kubernetes

Kubernetes manifests are provided in `deploy/kubernetes/`. These define a CronJob for periodic monitoring checks and a Deployment for the dashboard.

### Deploying

```bash
# Create namespace
kubectl create namespace glews

# Create secrets for Earthdata and alert credentials
kubectl -n glews create secret generic glews-earthdata \
    --from-file=netrc=$HOME/.netrc

kubectl -n glews create secret generic glews-alerts \
    --from-literal=GLEWS_SMTP_HOST=smtp.gmail.com \
    --from-literal=GLEWS_SMTP_PORT=587 \
    --from-literal=GLEWS_SMTP_USER=you@gmail.com \
    --from-literal=GLEWS_SMTP_PASS=xxxx-xxxx-xxxx-xxxx \
    --from-literal=GLEWS_SMTP_TO=team@example.com \
    --from-literal=GLEWS_SLACK_WEBHOOK=https://hooks.slack.com/...

# Create config map from your site config
kubectl -n glews create configmap glews-config \
    --from-file=global_watch.yaml=config/global_watch.yaml

# Apply manifests
kubectl -n glews apply -f deploy/kubernetes/

# Check status
kubectl -n glews get cronjobs,deployments,pods
```

### Key resources

- **CronJob `glews-monitor`** — runs `glews monitor --check-now` every 6 hours (configurable via the schedule field). Uses a PersistentVolumeClaim for data and output directories so state persists across runs.
- **Deployment `glews-dashboard`** — serves the analyst dashboard. Reads from the same output PVC.
- **PersistentVolumeClaim `glews-data`** — shared storage for downloaded products, monitor state, and detection output.

### Scaling considerations

- The monitor CronJob should have `concurrencyPolicy: Forbid` to prevent overlapping check cycles.
- For multi-site watchlists with many sites, increase the CronJob's `activeDeadlineSeconds` to accommodate longer check cycles.
- The dashboard is stateless and can be scaled horizontally if needed, though a single replica is sufficient for most deployments.

---

## systemd

A systemd unit file is provided at `deploy/systemd/glews-monitor.service` for running the Glacier Early Warning System monitor as a Linux service.

### Installation

```bash
# Copy the unit file
sudo cp deploy/systemd/glews-monitor.service /etc/systemd/system/

# Edit to set your paths and environment
sudo systemctl edit glews-monitor.service

# Reload, enable, and start
sudo systemctl daemon-reload
sudo systemctl enable glews-monitor.service
sudo systemctl start glews-monitor.service

# Check status and logs
sudo systemctl status glews-monitor.service
journalctl -u glews-monitor.service -f
```

### Unit file overview

```ini
[Unit]
Description=GLEWS Glacier Early Warning System Monitor
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=glews
WorkingDirectory=/opt/glews
ExecStart=/opt/glews/.venv/bin/glews monitor -c config/global_watch.yaml --interval 6
Restart=on-failure
RestartSec=60
EnvironmentFile=/opt/glews/.env

[Install]
WantedBy=multi-user.target
```

Place alert credentials in `/opt/glews/.env` (permissions `600`):

```bash
GLEWS_SMTP_HOST=smtp.gmail.com
GLEWS_SMTP_PORT=587
GLEWS_SMTP_USER=you@gmail.com
GLEWS_SMTP_PASS=xxxx-xxxx-xxxx-xxxx
GLEWS_SMTP_TO=team@example.com
GLEWS_SLACK_WEBHOOK=https://hooks.slack.com/...
```

---

## cron

For environments where a persistent daemon is not desired, a crontab entry runs single-pass check cycles on a schedule.

### Installation

```bash
# Install from the provided crontab fragment
crontab -l | cat - deploy/cron/glews-check.cron | crontab -

# Or add manually (every 6 hours):
crontab -e
```

### Crontab entry

```
# GLEWS monitoring check — every 6 hours
0 */6 * * * cd /opt/glews && /opt/glews/.venv/bin/glews monitor -c config/global_watch.yaml --check-now >> /var/log/glews/monitor.log 2>&1
```

The `--check-now` flag runs a single check cycle and exits. State is checkpointed to `data/monitor_state/`, so each run resumes from where the previous one left off.

### Log rotation

```bash
# /etc/logrotate.d/glews
/var/log/glews/*.log {
    weekly
    rotate 12
    compress
    missingok
    notifempty
}
```

---

## Monitoring and Health Checks

Configuration for operational monitoring is in `deploy/monitoring/`.

### Health check endpoint

The dashboard serves a health check at `/health` that returns HTTP 200 with a JSON body:

```json
{
  "status": "ok",
  "last_check": "2026-08-15T12:00:00Z",
  "sites_monitored": 13,
  "active_alerts": 2
}
```

### Prometheus metrics

If Prometheus scraping is configured, key metrics to watch:

| Metric | Description |
|--------|-------------|
| `glews_check_cycle_duration_seconds` | Time for a complete check cycle |
| `glews_products_downloaded_total` | Cumulative NISAR products downloaded |
| `glews_anomaly_flags_total` | Total anomaly flags raised |
| `glews_alerts_dispatched_total` | Alerts sent (by channel and level) |
| `glews_check_cycle_errors_total` | Failed check cycles |

### Alerting on the alerter

To ensure the monitoring system itself is healthy:

1. **Process monitoring** — systemd's `Restart=on-failure` or Kubernetes liveness probes handle process crashes.
2. **Heartbeat** — configure a dead-man's switch (e.g., Healthchecks.io, PagerDuty heartbeat) that the monitor pings after each successful check cycle. If no ping arrives within `2 * interval`, the monitor is presumed down.
3. **Log monitoring** — watch for `ERROR` lines in the monitor log. Repeated `ERROR: download failed` or `ERROR: ASF search failed` indicates an upstream data-access problem.

---

## Terraform

Infrastructure-as-code templates are provided in `deploy/terraform/` for cloud deployments. These are reference configurations — adapt to your cloud environment and security requirements.

---

## Security Notes

- **Never commit credentials** to config files or the repository. Use `${ENV_VAR}` expansion in YAML configs and environment variables or `.env` files for deployment.
- The `~/.netrc` file should have permissions `600` and be mounted read-only in containers.
- The dashboard does not implement authentication. In production, place it behind a reverse proxy (e.g., nginx, Traefik) with appropriate access controls.
- Alert webhook URLs and SMTP credentials should be treated as secrets and managed accordingly (Kubernetes Secrets, systemd EnvironmentFile with restricted permissions, etc.).
