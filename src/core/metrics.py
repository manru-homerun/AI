from __future__ import annotations

from typing import Final

from fastapi import Request
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest
from starlette.routing import BaseRoute


METRICS_ENDPOINT: Final = "/metrics"
UNMATCHED_ENDPOINT: Final = "unmatched"

registry = CollectorRegistry()

http_requests_total = Counter(
    "api_http_requests",
    "Total number of HTTP requests by method, endpoint, and status code.",
    labelnames=("method", "endpoint", "status_code"),
    registry=registry,
)

http_request_duration_seconds = Histogram(
    "api_http_request_duration_seconds",
    "HTTP request duration in seconds by method, endpoint, and status code.",
    labelnames=("method", "endpoint", "status_code"),
    registry=registry,
)


def route_template(request: Request) -> str:
    route = request.scope.get("route")
    if isinstance(route, BaseRoute):
        path = getattr(route, "path", None)
        if isinstance(path, str) and path:
            return path
    return UNMATCHED_ENDPOINT


def record_http_request(request: Request, *, status_code: int, elapsed_seconds: float) -> None:
    if request.url.path == METRICS_ENDPOINT:
        return

    labels = {
        "method": request.method,
        "endpoint": route_template(request),
        "status_code": str(status_code),
    }
    http_requests_total.labels(**labels).inc()
    http_request_duration_seconds.labels(**labels).observe(elapsed_seconds)


def metrics_content() -> bytes:
    return generate_latest(registry)
