"""Validation for local Prometheus/Grafana provisioning artifacts."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_grafana_dashboard_has_required_panels_and_valid_json():
    dashboard = json.loads((ROOT / "deploy/grafana/dashboards/techjob_overview.json").read_text())
    titles = {panel["title"] for panel in dashboard["panels"]}
    assert titles == {
        "Tool latency P95 / P99",
        "System 1 triage distribution",
        "Source fetch latency",
        "Estimated LLM API cost savings",
    }
    triage_panel = next(panel for panel in dashboard["panels"] if panel["title"] == "System 1 triage distribution")
    triage_expressions = [target["expr"] for target in triage_panel["targets"]]
    assert len(triage_expressions) == 2
    assert all('decision=~"local_accept|escalate_system2"' in expression for expression in triage_expressions)
    assert any('decision="local_accept"' in expression for expression in triage_expressions)
    assert any('decision="escalate_system2"' in expression for expression in triage_expressions)
    assert all("disqualified" not in expression for expression in triage_expressions)


def test_prometheus_scrape_and_grafana_provisioning_are_configured():
    prometheus = (ROOT / "deploy/prometheus/prometheus.yml").read_text()
    datasource = (ROOT / "deploy/grafana/provisioning/datasources/datasource.yml").read_text()
    provider = (ROOT / "deploy/grafana/provisioning/dashboards/dashboard_provider.yml").read_text()
    assert "scrape_interval: 5s" in prometheus
    assert '"techjob-mcp:8000"' in prometheus
    assert "metrics_path: /metrics" in prometheus
    assert "url: http://prometheus:9090" in datasource
    assert "/var/lib/grafana/dashboards" in provider
