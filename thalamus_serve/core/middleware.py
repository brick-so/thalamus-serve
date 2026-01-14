import hmac
import os
from collections.abc import Awaitable, Callable
from typing import Any, cast

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from thalamus_serve.observability.logging import log


class APIKeyAuth(BaseHTTPMiddleware):
    def __init__(self, app: Any, exempt_paths: list[str] | None = None) -> None:
        super().__init__(app)
        self.api_key = os.environ.get("THALAMUS_API_KEY")
        self.exempt_paths = exempt_paths or ["/health", "/ready", "/metrics", "/status"]

    def _is_exempt(self, path: str) -> bool:
        normalized = path.rstrip("/")
        return any(
            normalized == p or normalized.startswith(p + "/") for p in self.exempt_paths
        )

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if self._is_exempt(request.url.path):
            return cast(Response, await call_next(request))

        if not self.api_key:
            log.critical("api_key_not_configured")
            return JSONResponse(
                {"error": "Service authentication not configured"}, status_code=503
            )

        provided_key = request.headers.get("X-API-Key", "")
        if not provided_key or not hmac.compare_digest(provided_key, self.api_key):
            log.warning("invalid_api_key", path=request.url.path)
            return JSONResponse(
                {"error": "Invalid or missing API key"}, status_code=401
            )

        return cast(Response, await call_next(request))
