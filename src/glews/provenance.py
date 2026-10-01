"""
Data provenance and audit logging for the GLEWS pipeline.

Tracks every action the system performs — data downloads, detection
runs, alert dispatches — with enough detail to reconstruct the chain
of evidence from raw satellite product to analyst notification.

Two complementary logs:

    ProvenanceTracker   Records pipeline actions (search, download,
                        detect, report) as JSONL with timing, inputs,
                        outputs, and parameters.

    AlertAuditLog       Records alert dispatch outcomes and analyst
                        classification decisions as JSONL, separate
                        from provenance so security/compliance reviews
                        can scope to alerting alone.

Usage::

    from glews.provenance import ProvenanceTracker, AlertAuditLog

    tracker = ProvenanceTracker()
    with tracker.track("detect", "Aletsch Glacier") as ctx:
        flags = detect_anomalies(...)
        ctx.set_result(f"{len(flags)} flags")
        ctx.outputs = ["output/flags.geojson"]

    audit = AlertAuditLog()
    audit.log_alert("WARNING", "Aletsch Glacier", "Anomalous...",
                    channels_sent=["email"], channels_failed=[])
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

from glews import __version__

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provenance record
# ---------------------------------------------------------------------------


@dataclass
class ProvenanceRecord:
    """A single auditable action performed by the GLEWS pipeline."""

    timestamp: str
    action: str
    site_name: str
    input_files: list[str]
    output_files: list[str]
    parameters: dict[str, Any]
    duration_seconds: float
    result_summary: str
    version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProvenanceRecord":
        return cls(
            timestamp=d["timestamp"],
            action=d["action"],
            site_name=d["site_name"],
            input_files=d.get("input_files", []),
            output_files=d.get("output_files", []),
            parameters=d.get("parameters", {}),
            duration_seconds=d.get("duration_seconds", 0.0),
            result_summary=d.get("result_summary", ""),
            version=d.get("version", ""),
        )

    @classmethod
    def from_json(cls, line: str) -> "ProvenanceRecord":
        return cls.from_dict(json.loads(line))


# ---------------------------------------------------------------------------
# Tracking context manager helper
# ---------------------------------------------------------------------------


class _TrackingContext:
    """Mutable bag carried through a ``with tracker.track(...)`` block.

    The caller can set ``outputs`` and call ``set_result()`` inside the
    block; the tracker reads them back when the block exits.
    """

    def __init__(self) -> None:
        self.outputs: list[str] = []
        self._result_summary: str = ""
        self._start: float = time.monotonic()

    def set_result(self, summary: str) -> None:
        self._result_summary = summary

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start


# ---------------------------------------------------------------------------
# ProvenanceTracker
# ---------------------------------------------------------------------------


class ProvenanceTracker:
    """Append-only JSONL provenance log for pipeline actions."""

    def __init__(self, log_dir: str | Path = "data/provenance") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / "provenance.jsonl"

    # -- core API --

    def record(
        self,
        action: str,
        site_name: str,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        params: dict[str, Any] | None = None,
        duration: float = 0.0,
        summary: str = "",
    ) -> ProvenanceRecord:
        """Create and persist a provenance record."""
        rec = ProvenanceRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            action=action,
            site_name=site_name,
            input_files=inputs or [],
            output_files=outputs or [],
            parameters=params or {},
            duration_seconds=round(duration, 3),
            result_summary=summary,
            version=__version__,
        )
        self._append(rec)
        return rec

    @contextmanager
    def track(
        self,
        action: str,
        site_name: str,
        inputs: list[str] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Generator[_TrackingContext, None, None]:
        """Context manager that auto-records timing and result.

        Usage::

            with tracker.track("detect", "Aletsch") as ctx:
                run_detection()
                ctx.set_result("5 flags")
                ctx.outputs = ["output/flags.geojson"]
        """
        ctx = _TrackingContext()
        try:
            yield ctx
        finally:
            self.record(
                action=action,
                site_name=site_name,
                inputs=inputs or [],
                outputs=ctx.outputs,
                params=params or {},
                duration=ctx.elapsed,
                summary=ctx._result_summary,
            )

    # -- query --

    def query(
        self,
        site_name: str | None = None,
        action: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> list[ProvenanceRecord]:
        """Search the provenance log with optional filters.

        Parameters
        ----------
        site_name : str, optional
            Filter by site name (exact match).
        action : str, optional
            Filter by action type (exact match).
        since : str, optional
            ISO timestamp lower bound (inclusive).
        until : str, optional
            ISO timestamp upper bound (inclusive).
        """
        records: list[ProvenanceRecord] = []

        if not self.log_path.exists():
            return records

        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = ProvenanceRecord.from_json(line)
                except (json.JSONDecodeError, KeyError):
                    continue

                if site_name is not None and rec.site_name != site_name:
                    continue
                if action is not None and rec.action != action:
                    continue
                if since is not None and rec.timestamp < since:
                    continue
                if until is not None and rec.timestamp > until:
                    continue

                records.append(rec)

        return records

    # -- reporting --

    def generate_audit_report(self, site_name: str | None = None) -> str:
        """Generate a markdown-formatted audit trail."""
        records = self.query(site_name=site_name)

        title = f"Provenance Audit Report — {site_name}" if site_name else "Provenance Audit Report — All Sites"
        lines = [
            f"# {title}",
            "",
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            f"Total records: {len(records)}",
            "",
        ]

        if not records:
            lines.append("No provenance records found.")
            return "\n".join(lines)

        lines.append("## Timeline")
        lines.append("")
        lines.append("| Timestamp | Action | Site | Duration | Summary |")
        lines.append("|-----------|--------|------|----------|---------|")

        for rec in records:
            ts = rec.timestamp[:19]  # trim to seconds
            dur = f"{rec.duration_seconds:.1f}s"
            summary = rec.result_summary[:60] if rec.result_summary else "—"
            lines.append(f"| {ts} | {rec.action} | {rec.site_name} | {dur} | {summary} |")

        # Action summary
        action_counts: dict[str, int] = {}
        for rec in records:
            action_counts[rec.action] = action_counts.get(rec.action, 0) + 1

        lines.extend([
            "",
            "## Summary by Action",
            "",
        ])
        for act, count in sorted(action_counts.items()):
            lines.append(f"- **{act}**: {count} occurrence(s)")

        return "\n".join(lines)

    # -- internal --

    def _append(self, rec: ProvenanceRecord) -> None:
        with open(self.log_path, "a") as f:
            f.write(rec.to_json() + "\n")


# ---------------------------------------------------------------------------
# Alert audit log
# ---------------------------------------------------------------------------


class AlertAuditLog:
    """Append-only JSONL log for alert dispatches and analyst actions.

    Kept separate from provenance so compliance reviews can scope to
    alerting without filtering through pipeline bookkeeping.
    """

    def __init__(self, log_dir: str | Path = "data/audit") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / "alert_audit.jsonl"

    # -- alert dispatch --

    def log_alert(
        self,
        alert_level: str,
        site_name: str,
        message: str,
        channels_sent: list[str] | None = None,
        channels_failed: list[str] | None = None,
    ) -> dict[str, Any]:
        """Record an alert dispatch attempt."""
        entry = {
            "type": "alert_dispatch",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "alert_level": alert_level,
            "site_name": site_name,
            "message": message,
            "channels_sent": channels_sent or [],
            "channels_failed": channels_failed or [],
            "version": __version__,
        }
        self._append(entry)
        return entry

    # -- analyst classification --

    def log_classification(
        self,
        site_name: str,
        flag_id: str,
        classification: str,
        analyst_note: str = "",
    ) -> dict[str, Any]:
        """Record an analyst's classification decision from the dashboard."""
        entry = {
            "type": "classification",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "site_name": site_name,
            "flag_id": flag_id,
            "classification": classification,
            "analyst_note": analyst_note,
            "version": __version__,
        }
        self._append(entry)
        return entry

    # -- query --

    def query_alerts(
        self,
        site_name: str | None = None,
        level: str | None = None,
        since: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search the alert audit log."""
        results: list[dict[str, Any]] = []

        if not self.log_path.exists():
            return results

        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if site_name is not None and entry.get("site_name") != site_name:
                    continue
                if level is not None and entry.get("alert_level") != level:
                    continue
                if since is not None and entry.get("timestamp", "") < since:
                    continue

                results.append(entry)

        return results

    # -- reporting --

    def generate_alert_history(self, site_name: str | None = None) -> str:
        """Generate a formatted alert timeline."""
        entries = self.query_alerts(site_name=site_name)

        title = f"Alert History — {site_name}" if site_name else "Alert History — All Sites"
        lines = [
            f"# {title}",
            "",
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            f"Total entries: {len(entries)}",
            "",
        ]

        if not entries:
            lines.append("No alert records found.")
            return "\n".join(lines)

        # Separate dispatches and classifications
        dispatches = [e for e in entries if e.get("type") == "alert_dispatch"]
        classifications = [e for e in entries if e.get("type") == "classification"]

        if dispatches:
            lines.append("## Alert Dispatches")
            lines.append("")
            lines.append("| Timestamp | Level | Site | Channels Sent | Channels Failed |")
            lines.append("|-----------|-------|------|---------------|-----------------|")
            for d in dispatches:
                ts = d["timestamp"][:19]
                sent = ", ".join(d.get("channels_sent", [])) or "—"
                failed = ", ".join(d.get("channels_failed", [])) or "—"
                lines.append(
                    f"| {ts} | {d.get('alert_level', '?')} | "
                    f"{d.get('site_name', '?')} | {sent} | {failed} |"
                )

        if classifications:
            lines.extend(["", "## Analyst Classifications", ""])
            lines.append("| Timestamp | Site | Flag | Classification | Note |")
            lines.append("|-----------|------|------|----------------|------|")
            for c in classifications:
                ts = c["timestamp"][:19]
                note = c.get("analyst_note", "")[:40] or "—"
                lines.append(
                    f"| {ts} | {c.get('site_name', '?')} | "
                    f"{c.get('flag_id', '?')} | "
                    f"{c.get('classification', '?')} | {note} |"
                )

        return "\n".join(lines)

    # -- internal --

    def _append(self, entry: dict[str, Any]) -> None:
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
