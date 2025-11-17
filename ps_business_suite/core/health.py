"""Health endpoint registration for Streamlit server."""
from __future__ import annotations

from typing import Any


def register_health_route() -> None:
    """Expose a lightweight ``/health`` endpoint when Streamlit is available.

    The registration is defensive so deployments continue to start even if
    Streamlit internals change. When the route cannot be registered we simply
    return without raising an exception.
    """

    try:
        from tornado.web import RequestHandler
        from streamlit.web.server import routes
    except Exception:
        return

    available_routes: list[Any] | None = getattr(routes, "_available_routes", None)
    if available_routes is None:
        return

    if any(getattr(route, "path", None) == "/health" for route in available_routes):
        return

    class HealthHandler(RequestHandler):
        def get(self) -> None:  # type: ignore[override]
            self.set_status(200)
            self.finish("ok")

    try:
        available_routes.insert(0, routes.PathRoute("/health", HealthHandler))
    except Exception:
        return
