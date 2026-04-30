import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.observability.logging import request_id_var
from app.observability.metrics import http_request_duration_seconds, http_requests_total


class ObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        token = request_id_var.set(request_id)
        request.state.request_id = request_id

        route = request.scope.get("route")
        path = route.path if route is not None else request.url.path

        start = time.perf_counter()
        status = "500"
        try:
            response = await call_next(request)
            status = str(response.status_code)
        finally:
            http_requests_total.labels(
                method=request.method, path=path, status=status
            ).inc()
            http_request_duration_seconds.labels(
                method=request.method, path=path
            ).observe(time.perf_counter() - start)
            request_id_var.reset(token)

        response.headers["X-Request-ID"] = request_id
        return response
