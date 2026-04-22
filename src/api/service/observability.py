"""Prometheus metrics + OpenTelemetry tracing for the API service.

Both subsystems are **optional** and only activate when their config
fields are enabled on :class:`ObservabilityConfig`. The helpers degrade
gracefully when :mod:`prometheus_client` or :mod:`opentelemetry` are not
installed.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import FastAPI


logger = logging.getLogger(__name__)


# ---------- Prometheus ---------------------------------------------------

_PROM_AVAILABLE: bool | None = None


def prometheus_available() -> bool:
    global _PROM_AVAILABLE
    if _PROM_AVAILABLE is None:
        try:
            import prometheus_client  # noqa: F401

            _PROM_AVAILABLE = True
        except ImportError:
            _PROM_AVAILABLE = False
    return _PROM_AVAILABLE


def build_prometheus_registry() -> tuple[Any, dict[str, Any]]:
    """Return a fresh ``(CollectorRegistry, {name: metric})`` pair.

    A fresh registry per app instance keeps unit tests hermetic — the
    global default registry would otherwise accumulate metrics across
    test cases.
    """
    from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

    reg = CollectorRegistry()
    metrics = {
        "http_requests_total": Counter(
            "trx_http_requests_total",
            "HTTP requests",
            labelnames=["method", "path", "status"],
            registry=reg,
        ),
        "http_request_seconds": Histogram(
            "trx_http_request_seconds",
            "HTTP request latency",
            labelnames=["method", "path"],
            registry=reg,
        ),
        "tasks_total": Counter(
            "trx_tasks_total",
            "Tasks submitted",
            labelnames=["kind", "status"],
            registry=reg,
        ),
        "tasks_active": Gauge(
            "trx_tasks_active",
            "Tasks currently running",
            labelnames=["kind"],
            registry=reg,
        ),
        "engine_requests_total": Counter(
            "trx_engine_requests_total",
            "LLM engine requests",
            labelnames=["model"],
            registry=reg,
        ),
        "engine_tokens_total": Counter(
            "trx_engine_tokens_total",
            "LLM engine tokens",
            labelnames=["model", "direction"],  # direction=prompt|completion
            registry=reg,
        ),
        "engine_cost_usd_total": Counter(
            "trx_engine_cost_usd_total",
            "LLM engine cost in USD",
            labelnames=["model"],
            registry=reg,
        ),
    }
    return reg, metrics


def install_prometheus(api: "FastAPI", enabled: bool, path: str = "/metrics") -> None:
    """Install an ASGI middleware + ``/metrics`` endpoint on ``api``.

    Safe to call when ``prometheus_client`` is missing — ``enabled`` is
    forced to False in that case.
    """
    from fastapi import Request, Response
    from starlette.middleware.base import BaseHTTPMiddleware

    if not enabled or not prometheus_available():
        return

    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    reg, metrics = build_prometheus_registry()
    api.state.prom_registry = reg
    api.state.prom_metrics = metrics

    class _PromMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            started = time.perf_counter()
            route = request.scope.get("route")
            path = getattr(route, "path", None) or request.url.path
            try:
                response = await call_next(request)
                status = response.status_code
            except Exception:
                metrics["http_requests_total"].labels(request.method, path, "500").inc()
                raise
            metrics["http_requests_total"].labels(request.method, path, str(status)).inc()
            metrics["http_request_seconds"].labels(request.method, path).observe(time.perf_counter() - started)
            return response

    api.add_middleware(_PromMiddleware)

    async def _metrics_endpoint():
        return Response(generate_latest(reg), media_type=CONTENT_TYPE_LATEST)

    api.add_api_route(path, _metrics_endpoint, methods=["GET"], include_in_schema=False)


# ---------- OpenTelemetry -----------------------------------------------

_OTEL_AVAILABLE: bool | None = None


def otel_available() -> bool:
    global _OTEL_AVAILABLE
    if _OTEL_AVAILABLE is None:
        try:
            import opentelemetry  # noqa: F401

            _OTEL_AVAILABLE = True
        except ImportError:
            _OTEL_AVAILABLE = False
    return _OTEL_AVAILABLE


def install_opentelemetry(
    api: "FastAPI",
    *,
    enabled: bool,
    service_name: str = "translatorx",
    exporter: str = "console",  # "console" | "otlp-grpc" | "otlp-http"
    endpoint: str | None = None,
) -> None:
    """Wire OpenTelemetry tracing + FastAPI auto-instrumentation.

    ``exporter='console'`` prints spans to stdout (good for dev);
    ``'otlp-grpc'`` / ``'otlp-http'`` ship spans to an OTLP collector at
    ``endpoint``.
    """
    if not enabled or not otel_available():
        return
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if exporter == "otlp-grpc":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint) if endpoint else OTLPSpanExporter()))
    elif exporter == "otlp-http":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint) if endpoint else OTLPSpanExporter()))
    else:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    api.state.otel_provider = provider

    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(api)
    except Exception:
        logger.exception("FastAPI OTel instrumentation failed; continuing without it")


@contextmanager
def span(name: str, **attrs: Any):
    """Context manager creating an OTel span, no-op when OTel is absent."""
    if not otel_available():
        yield None
        return
    from opentelemetry import trace

    tracer = trace.get_tracer("translatorx")
    with tracer.start_as_current_span(name) as sp:
        for k, v in attrs.items():
            try:
                sp.set_attribute(k, v)
            except Exception:
                pass
        yield sp


__all__ = [
    "build_prometheus_registry",
    "install_opentelemetry",
    "install_prometheus",
    "otel_available",
    "prometheus_available",
    "span",
]
