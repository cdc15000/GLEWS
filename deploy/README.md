# GLEWS Deployment Guide

## Prerequisites

- **Docker** >= 20.10 and **Docker Compose** >= 2.0 (or `docker-compose` v1.29+)
- An **Earthdata** account (free — register at <https://urs.earthdata.nasa.gov>)
- Earthdata credentials stored in `~/.netrc`:
  ```
  machine urs.earthdata.nasa.gov
      login YOUR_USERNAME
      password YOUR_PASSWORD
  ```

## Quick Start

```bash
cd /path/to/glews

# Build the images
docker compose build

# Start both monitor and dashboard
docker compose up -d

# View logs
docker compose logs -f monitor
```

The dashboard is available at <http://localhost:8080> by default.

## Configuration

All settings can be controlled with environment variables, either exported in
your shell or placed in a `.env` file alongside `docker-compose.yml`.

| Variable                | Default                      | Description                            |
|------------------------|------------------------------|----------------------------------------|
| `GLEWS_CONFIG`          | `config/global_watch.yaml`   | Site config file (relative to /app)    |
| `GLEWS_INTERVAL`        | `6`                          | Check interval in hours                |
| `GLEWS_DASHBOARD_PORT`  | `8080`                       | Port the dashboard listens on          |
| `GLEWS_SMTP_HOST`       | *(empty)*                    | SMTP server for email alerts           |
| `GLEWS_SMTP_PORT`       | `587`                        | SMTP port                              |
| `GLEWS_SMTP_USER`       | *(empty)*                    | SMTP username                          |
| `GLEWS_SMTP_PASS`       | *(empty)*                    | SMTP password / app password           |
| `GLEWS_SMTP_TO`         | *(empty)*                    | Alert recipient email(s)               |
| `GLEWS_SLACK_WEBHOOK`   | *(empty)*                    | Slack incoming-webhook URL             |
| `GLEWS_WEBHOOK_URL`     | *(empty)*                    | Generic HTTP webhook for alerts        |

### Using an Override File

For persistent customizations, copy the example override:

```bash
cp docker-compose.override.yml.example docker-compose.override.yml
```

Edit the override file with your credentials and preferences.  Docker Compose
merges it with the base file automatically.

## Setting Up Alerts

### Email (SMTP)

Set the `GLEWS_SMTP_*` variables.  For Gmail, create an App Password
(Account > Security > App Passwords) and use it as `GLEWS_SMTP_PASS`:

```bash
export GLEWS_SMTP_HOST=smtp.gmail.com
export GLEWS_SMTP_PORT=587
export GLEWS_SMTP_USER=you@gmail.com
export GLEWS_SMTP_PASS=xxxx-xxxx-xxxx-xxxx
export GLEWS_SMTP_TO=team@example.com
```

### Slack

Create an incoming webhook in your Slack workspace
(<https://api.slack.com/messaging/webhooks>) and set:

```bash
export GLEWS_SLACK_WEBHOOK=https://hooks.slack.com/services/T00/B00/xxxx
```

### Generic Webhook

Point `GLEWS_WEBHOOK_URL` at any HTTP endpoint.  GLEWS POSTs a JSON payload
with alert details on each trigger.

## Monitoring Multiple Sites

Use `config/global_watch.yaml` (the default) which defines a `sites` list.
Add entries to that file, or point `GLEWS_CONFIG` at your own multi-site YAML:

```yaml
# my_watchlist.yaml
defaults:
  acquire:
    platform: SENTINEL-1
monitor:
  interval_hours: 4
sites:
  - !include nepal_2026.yaml
  - !include weisshorn_2026.yaml
```

## Viewing the Dashboard

Open `http://<host>:8080` (or the port you configured).  The dashboard reads
from the shared `output/` volume and updates as the monitor produces new
results.

## Log Access and Troubleshooting

```bash
# Follow all logs
docker compose logs -f

# Monitor service only
docker compose logs -f monitor

# Check container health
docker compose ps

# Restart a single service
docker compose restart monitor

# Rebuild after code changes
docker compose build && docker compose up -d
```

Data is persisted in Docker named volumes (`glews-data`, `glews-output`).
To inspect them directly:

```bash
docker volume inspect glews_glews-output
```

## Running Without Docker

### systemd (Linux)

A unit file is provided at `deploy/systemd/glews-monitor.service`.  Install it:

```bash
sudo cp deploy/systemd/glews-monitor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now glews-monitor
```

Edit the unit file first to set `WorkingDirectory`, `User`, and the path to
your config.

### cron

A crontab entry is provided at `deploy/cron/glews-check.cron`.  Install it:

```bash
crontab -l | cat - deploy/cron/glews-check.cron | crontab -
```

This runs a single check cycle every 6 hours rather than a persistent process.

## Running on a Cloud VM

### AWS EC2

1. Launch an instance (t3.medium or larger; 20 GB+ EBS).
2. Install Docker: `sudo yum install docker && sudo systemctl enable --now docker`.
3. Install Docker Compose plugin.
4. Clone the repo, create `.env` with your credentials, and run `docker compose up -d`.
5. Open port 8080 in the security group to access the dashboard.

### GCP Compute Engine

1. Create a VM (e2-medium or larger; 20 GB persistent disk).
2. Install Docker: `sudo apt-get install docker.io docker-compose-plugin`.
3. Clone, configure, and run as above.
4. Add a firewall rule allowing TCP 8080.

For both providers, consider placing the dashboard behind a reverse proxy
(nginx, Caddy) with TLS if exposing it beyond your network.

## Kubernetes

Full Kubernetes manifests are in `deploy/kubernetes/`.  Apply them in order:

```bash
kubectl apply -f deploy/kubernetes/namespace.yaml
kubectl apply -f deploy/kubernetes/configmap.yaml
kubectl apply -f deploy/kubernetes/secrets.yaml      # edit CHANGEME values first!
kubectl apply -f deploy/kubernetes/deployment.yaml
kubectl apply -f deploy/kubernetes/cronjob.yaml
kubectl apply -f deploy/kubernetes/service.yaml
kubectl apply -f deploy/kubernetes/ingress.yaml
kubectl apply -f deploy/kubernetes/pdb.yaml
```

The manifests include:

- **Deployment** — dashboard with 2 replicas, health checks, and resource limits
- **CronJob** — monitor pipeline running every 6 hours with a 5-hour timeout
- **Service** / **Ingress** — ClusterIP service with nginx ingress (TLS placeholder)
- **PodDisruptionBudget** — keeps at least 1 dashboard pod during disruptions
- **Secrets** — all values are `CHANGEME` placeholders; use Sealed Secrets or
  an external secrets manager in production

You will also need to create PersistentVolumeClaims (`glews-data`, `glews-output`)
and a ConfigMap named `glews-site-config` containing your `global_watch.yaml`.

## AWS ECS Fargate (Terraform)

A complete Terraform configuration is in `deploy/terraform/`.  It provisions
an ECS Fargate cluster, ALB, EFS persistent storage, and an EventBridge
schedule for the monitor pipeline.

See [`deploy/terraform/README.md`](terraform/README.md) for prerequisites,
usage, and outputs.

## Monitoring

Prometheus, Grafana, and Alertmanager configurations are in `deploy/monitoring/`.

| File                        | Purpose                                          |
|-----------------------------|--------------------------------------------------|
| `prometheus.yaml`           | Scrape config for dashboard and monitor metrics  |
| `grafana-dashboard.json`    | Dashboard with detection count, latency, freshness, alert rate, and severity panels |
| `alertmanager-rules.yaml`   | Alert rules for pipeline failure, stale data (>36 h), and flag spikes |

### Importing the Grafana dashboard

1. Open Grafana and go to **Dashboards > Import**.
2. Upload `deploy/monitoring/grafana-dashboard.json`.
3. Select your Prometheus data source when prompted.

### Alert rules

Copy `alertmanager-rules.yaml` into your Prometheus rules directory (or
reference it in `rule_files` in your `prometheus.yml`).  The rules fire for:

- **GLEWSPipelineFailure** (critical) — monitor exited with a non-zero code
- **GLEWSStaleData** (warning) — no successful run in 36 hours
- **GLEWSFlagSpike** (warning) — more than 10 flags raised in 6 hours
- **GLEWSDashboardDown** (critical) — dashboard unreachable
- **GLEWSDashboardHighLatency** (warning) — p95 response time exceeds 2 seconds
