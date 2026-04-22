"""Tests for Prometheus + OpenTelemetry wiring."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.app.app import App
from api.service.app import create_app
from application.config import AppConfig, ServiceConfig, StoreConfig


def _mk_app(tmp_path, **svc_overrides) -> App:
    svc = ServiceConfig(**svc_overrides)
    cfg = AppConfig(store=StoreConfig(root=str(tmp_path)), service=svc)
    return App(cfg)


def test_prometheus_disabled_by_default(tmp_path):
    app = _mk_app(tmp_path)
    api = create_app(app)
    with TestClient(api) as client:
        r = client.get("/metrics")
    assert r.status_code == 404


def test_prometheus_enabled_exposes_metrics(tmp_path):
    pytest.importorskip("prometheus_client")
    app = _mk_app(tmp_path, prometheus_enabled=True)
    api = create_app(app)
    with TestClient(api) as client:
        r1 = client.get("/health")
        assert r1.status_code == 200
        r2 = client.get("/metrics")
    assert r2.status_code == 200
    body = r2.text
    assert "trx_http_requests_total" in body
    assert "trx_http_request_seconds" in body


def test_prometheus_custom_path(tmp_path):
    pytest.importorskip("prometheus_client")
    app = _mk_app(tmp_path, prometheus_enabled=True, prometheus_path="/_prom")
    api = create_app(app)
    with TestClient(api) as client:
        client.get("/health")
        r = client.get("/_prom")
    assert r.status_code == 200
    assert "trx_http_requests_total" in r.text


def test_otel_disabled_by_default_no_crash(tmp_path):
    app = _mk_app(tmp_path)
    api = create_app(app)
    with TestClient(api) as client:
        assert client.get("/health").status_code == 200


def test_otel_console_exporter(tmp_path):
    pytest.importorskip("opentelemetry")
    app = _mk_app(tmp_path, otel_enabled=True, otel_exporter="console")
    api = create_app(app)
    with TestClient(api) as client:
        r = client.get("/health")
    assert r.status_code == 200
