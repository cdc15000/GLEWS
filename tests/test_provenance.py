"""Tests for glews.provenance — data provenance and audit logging."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from glews.provenance import AlertAuditLog, ProvenanceRecord, ProvenanceTracker


# ---------------------------------------------------------------------------
# ProvenanceRecord
# ---------------------------------------------------------------------------


class TestProvenanceRecord:
    """Record creation and serialization."""

    def test_creation(self):
        rec = ProvenanceRecord(
            timestamp="2026-01-15T12:00:00+00:00",
            action="detect",
            site_name="Aletsch Glacier",
            input_files=["data/gunw/scene1.h5"],
            output_files=["output/flags.geojson"],
            parameters={"sigma_threshold": 2.5},
            duration_seconds=12.345,
            result_summary="5 flags detected",
            version="0.1.0",
        )
        assert rec.action == "detect"
        assert rec.site_name == "Aletsch Glacier"
        assert len(rec.input_files) == 1
        assert rec.duration_seconds == 12.345

    def test_to_dict_roundtrip(self):
        rec = ProvenanceRecord(
            timestamp="2026-01-15T12:00:00+00:00",
            action="download",
            site_name="Test Site",
            input_files=[],
            output_files=["data/slc/scene.zip"],
            parameters={"n_workers": 4},
            duration_seconds=60.0,
            result_summary="1 product downloaded",
            version="0.1.0",
        )
        d = rec.to_dict()
        rec2 = ProvenanceRecord.from_dict(d)
        assert rec2.action == rec.action
        assert rec2.site_name == rec.site_name
        assert rec2.parameters == rec.parameters

    def test_json_roundtrip(self):
        rec = ProvenanceRecord(
            timestamp="2026-01-15T12:00:00+00:00",
            action="search",
            site_name="Test",
            input_files=[],
            output_files=[],
            parameters={},
            duration_seconds=1.0,
            result_summary="3 products found",
            version="0.1.0",
        )
        json_str = rec.to_json()
        rec2 = ProvenanceRecord.from_json(json_str)
        assert rec2.action == "search"
        assert rec2.result_summary == "3 products found"

    def test_from_dict_defaults(self):
        """Missing optional fields get sensible defaults."""
        d = {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "action": "test",
            "site_name": "X",
        }
        rec = ProvenanceRecord.from_dict(d)
        assert rec.input_files == []
        assert rec.output_files == []
        assert rec.parameters == {}
        assert rec.duration_seconds == 0.0
        assert rec.result_summary == ""
        assert rec.version == ""


# ---------------------------------------------------------------------------
# ProvenanceTracker — JSONL append and query
# ---------------------------------------------------------------------------


class TestProvenanceTracker:
    """JSONL append, query, and reporting."""

    def test_record_and_query(self, tmp_path):
        tracker = ProvenanceTracker(log_dir=tmp_path / "prov")
        tracker.record("detect", "Site A", summary="3 flags")
        tracker.record("download", "Site B", summary="2 products")
        tracker.record("detect", "Site A", summary="1 flag")

        # All records
        all_recs = tracker.query()
        assert len(all_recs) == 3

        # Filter by site
        site_a = tracker.query(site_name="Site A")
        assert len(site_a) == 2
        assert all(r.site_name == "Site A" for r in site_a)

        # Filter by action
        downloads = tracker.query(action="download")
        assert len(downloads) == 1
        assert downloads[0].site_name == "Site B"

    def test_query_since_until(self, tmp_path):
        tracker = ProvenanceTracker(log_dir=tmp_path / "prov")
        # Write records with known timestamps
        for ts in ["2026-01-01T00:00:00", "2026-06-15T00:00:00", "2026-12-31T00:00:00"]:
            rec = ProvenanceRecord(
                timestamp=ts,
                action="detect",
                site_name="S",
                input_files=[],
                output_files=[],
                parameters={},
                duration_seconds=0.0,
                result_summary="",
                version="0.1.0",
            )
            tracker._append(rec)

        after_june = tracker.query(since="2026-06-01T00:00:00")
        assert len(after_june) == 2

        before_july = tracker.query(until="2026-07-01T00:00:00")
        assert len(before_july) == 2

        narrow = tracker.query(since="2026-06-01T00:00:00", until="2026-07-01T00:00:00")
        assert len(narrow) == 1

    def test_jsonl_file_format(self, tmp_path):
        """Each record is one JSON line; file is valid JSONL."""
        tracker = ProvenanceTracker(log_dir=tmp_path / "prov")
        tracker.record("a", "S1")
        tracker.record("b", "S2")

        lines = tracker.log_path.read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            parsed = json.loads(line)
            assert "action" in parsed
            assert "timestamp" in parsed

    def test_record_returns_record(self, tmp_path):
        tracker = ProvenanceTracker(log_dir=tmp_path / "prov")
        rec = tracker.record("detect", "Site", summary="ok")
        assert isinstance(rec, ProvenanceRecord)
        assert rec.action == "detect"
        assert rec.result_summary == "ok"

    def test_empty_log_query(self, tmp_path):
        """Querying an empty/nonexistent log returns []."""
        tracker = ProvenanceTracker(log_dir=tmp_path / "prov")
        assert tracker.query() == []
        assert tracker.query(site_name="nope") == []

    def test_missing_dir_created(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c"
        tracker = ProvenanceTracker(log_dir=deep)
        assert deep.is_dir()
        tracker.record("test", "S")
        assert tracker.log_path.exists()

    def test_generate_audit_report_empty(self, tmp_path):
        tracker = ProvenanceTracker(log_dir=tmp_path / "prov")
        report = tracker.generate_audit_report()
        assert "No provenance records found" in report
        assert "All Sites" in report

    def test_generate_audit_report_with_data(self, tmp_path):
        tracker = ProvenanceTracker(log_dir=tmp_path / "prov")
        tracker.record("detect", "Aletsch", summary="5 flags")
        tracker.record("download", "Aletsch", summary="2 products")

        report = tracker.generate_audit_report(site_name="Aletsch")
        assert "Aletsch" in report
        assert "detect" in report
        assert "download" in report
        assert "Summary by Action" in report

    def test_generate_audit_report_filters_by_site(self, tmp_path):
        tracker = ProvenanceTracker(log_dir=tmp_path / "prov")
        tracker.record("detect", "Site A", summary="a")
        tracker.record("detect", "Site B", summary="b")

        report = tracker.generate_audit_report(site_name="Site A")
        assert "Site A" in report
        # Site B should not appear in the timeline rows
        assert "Site B" not in report.split("## Summary")[0]


# ---------------------------------------------------------------------------
# Context manager timing
# ---------------------------------------------------------------------------


class TestTrackingContextManager:
    """Context manager auto-records timing and result."""

    def test_basic_tracking(self, tmp_path):
        tracker = ProvenanceTracker(log_dir=tmp_path / "prov")

        with tracker.track("detect", "TestSite") as ctx:
            time.sleep(0.05)
            ctx.set_result("done")
            ctx.outputs = ["out.json"]

        records = tracker.query()
        assert len(records) == 1
        rec = records[0]
        assert rec.action == "detect"
        assert rec.site_name == "TestSite"
        assert rec.result_summary == "done"
        assert rec.output_files == ["out.json"]
        assert rec.duration_seconds >= 0.04  # at least ~50ms

    def test_tracking_with_params(self, tmp_path):
        tracker = ProvenanceTracker(log_dir=tmp_path / "prov")

        with tracker.track(
            "download", "S",
            inputs=["scene.h5"],
            params={"n_workers": 4},
        ) as ctx:
            ctx.set_result("ok")

        rec = tracker.query()[0]
        assert rec.input_files == ["scene.h5"]
        assert rec.parameters == {"n_workers": 4}

    def test_tracking_records_on_exception(self, tmp_path):
        """Record is written even when the block raises."""
        tracker = ProvenanceTracker(log_dir=tmp_path / "prov")

        with pytest.raises(ValueError):
            with tracker.track("detect", "S") as ctx:
                ctx.set_result("partial")
                raise ValueError("boom")

        records = tracker.query()
        assert len(records) == 1
        assert records[0].result_summary == "partial"


# ---------------------------------------------------------------------------
# AlertAuditLog
# ---------------------------------------------------------------------------


class TestAlertAuditLog:
    """Alert dispatch and classification logging."""

    def test_log_alert(self, tmp_path):
        audit = AlertAuditLog(log_dir=tmp_path / "audit")
        entry = audit.log_alert(
            "WARNING", "Aletsch", "Anomalous acceleration",
            channels_sent=["email", "slack"],
            channels_failed=[],
        )
        assert entry["type"] == "alert_dispatch"
        assert entry["alert_level"] == "WARNING"
        assert entry["channels_sent"] == ["email", "slack"]

    def test_log_classification(self, tmp_path):
        audit = AlertAuditLog(log_dir=tmp_path / "audit")
        entry = audit.log_classification(
            "Aletsch", "flag-42", "true_positive",
            analyst_note="Confirmed by field team",
        )
        assert entry["type"] == "classification"
        assert entry["classification"] == "true_positive"
        assert entry["analyst_note"] == "Confirmed by field team"

    def test_query_alerts_by_site(self, tmp_path):
        audit = AlertAuditLog(log_dir=tmp_path / "audit")
        audit.log_alert("WARNING", "A", "msg1")
        audit.log_alert("CRITICAL", "B", "msg2")
        audit.log_alert("INFO", "A", "msg3")

        results = audit.query_alerts(site_name="A")
        assert len(results) == 2
        assert all(e["site_name"] == "A" for e in results)

    def test_query_alerts_by_level(self, tmp_path):
        audit = AlertAuditLog(log_dir=tmp_path / "audit")
        audit.log_alert("WARNING", "S", "w")
        audit.log_alert("CRITICAL", "S", "c")

        results = audit.query_alerts(level="CRITICAL")
        assert len(results) == 1
        assert results[0]["alert_level"] == "CRITICAL"

    def test_query_alerts_since(self, tmp_path):
        audit = AlertAuditLog(log_dir=tmp_path / "audit")
        # Write with known timestamps
        early = {
            "type": "alert_dispatch",
            "timestamp": "2026-01-01T00:00:00",
            "alert_level": "INFO",
            "site_name": "S",
            "message": "early",
            "channels_sent": [],
            "channels_failed": [],
            "version": "0.1.0",
        }
        late = {**early, "timestamp": "2026-12-01T00:00:00", "message": "late"}
        with open(audit.log_path, "w") as f:
            f.write(json.dumps(early) + "\n")
            f.write(json.dumps(late) + "\n")

        results = audit.query_alerts(since="2026-06-01T00:00:00")
        assert len(results) == 1
        assert results[0]["message"] == "late"

    def test_empty_log(self, tmp_path):
        audit = AlertAuditLog(log_dir=tmp_path / "audit")
        assert audit.query_alerts() == []

    def test_missing_dir_created(self, tmp_path):
        deep = tmp_path / "x" / "y"
        audit = AlertAuditLog(log_dir=deep)
        assert deep.is_dir()

    def test_default_channels(self, tmp_path):
        """channels_sent/failed default to [] when None."""
        audit = AlertAuditLog(log_dir=tmp_path / "audit")
        entry = audit.log_alert("INFO", "S", "msg")
        assert entry["channels_sent"] == []
        assert entry["channels_failed"] == []


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


class TestReportGeneration:
    """Audit report and alert history generation."""

    def test_alert_history_empty(self, tmp_path):
        audit = AlertAuditLog(log_dir=tmp_path / "audit")
        report = audit.generate_alert_history()
        assert "No alert records found" in report

    def test_alert_history_with_dispatches(self, tmp_path):
        audit = AlertAuditLog(log_dir=tmp_path / "audit")
        audit.log_alert("WARNING", "Site1", "msg", channels_sent=["email"])
        audit.log_classification("Site1", "f1", "false_positive", "noise")

        report = audit.generate_alert_history()
        assert "Alert Dispatches" in report
        assert "Analyst Classifications" in report
        assert "WARNING" in report
        assert "false_positive" in report

    def test_alert_history_filters_by_site(self, tmp_path):
        audit = AlertAuditLog(log_dir=tmp_path / "audit")
        audit.log_alert("INFO", "A", "a-msg")
        audit.log_alert("INFO", "B", "b-msg")

        report = audit.generate_alert_history(site_name="A")
        # Only site A entries in the table
        assert "A" in report
        # The query filter means B entries won't appear
        entries = audit.query_alerts(site_name="A")
        assert len(entries) == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Corrupt data, concurrent appends, etc."""

    def test_corrupt_line_skipped(self, tmp_path):
        """Corrupt JSONL lines are silently skipped during query."""
        tracker = ProvenanceTracker(log_dir=tmp_path / "prov")
        tracker.record("detect", "S", summary="good")

        # Append a corrupt line
        with open(tracker.log_path, "a") as f:
            f.write("NOT VALID JSON\n")

        tracker.record("search", "S", summary="also good")

        records = tracker.query()
        assert len(records) == 2
        assert records[0].action == "detect"
        assert records[1].action == "search"

    def test_empty_lines_skipped(self, tmp_path):
        tracker = ProvenanceTracker(log_dir=tmp_path / "prov")
        tracker.record("a", "S")

        with open(tracker.log_path, "a") as f:
            f.write("\n\n\n")

        tracker.record("b", "S")

        assert len(tracker.query()) == 2

    def test_version_from_package(self, tmp_path):
        """Records use the version from glews.__version__."""
        from glews import __version__

        tracker = ProvenanceTracker(log_dir=tmp_path / "prov")
        rec = tracker.record("test", "S")
        assert rec.version == __version__

    def test_alert_audit_version(self, tmp_path):
        from glews import __version__

        audit = AlertAuditLog(log_dir=tmp_path / "audit")
        entry = audit.log_alert("INFO", "S", "msg")
        assert entry["version"] == __version__
