import os
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.requests import Request

from thalamus_serve.config import WeightSource
from thalamus_serve.core.middleware import APIKeyAuth
from thalamus_serve.core.model import ModelRegistry, ModelSpec
from thalamus_serve.core.routes import RouteContext, create_routes
from thalamus_serve.infra.gpu import GPUAllocator
from thalamus_serve.observability.logging import log
from thalamus_serve.observability.logging import setup as setup_logging
from thalamus_serve.observability.metrics import MODEL_INFO
from thalamus_serve.observability.middleware import RequestLogging
from thalamus_serve.storage.fetch import fetch_weight


class Thalamus:
    """ML model serving application built on FastAPI.

    Thalamus provides a simple decorator-based API for registering and serving
    machine learning models with built-in observability, caching, and GPU management.

    Example:
        ```python
        from thalamus_serve import Thalamus
        from pydantic import BaseModel

        app = Thalamus()

        class InputSchema(BaseModel):
            text: str

        class OutputSchema(BaseModel):
            label: str

        @app.model(
            model_id="classifier",
            default=True,
            input_type=InputSchema,
            output_type=OutputSchema,
        )
        class MyModel:
            def predict(self, inputs: list[InputSchema]) -> list[OutputSchema]:
                return [OutputSchema(label="positive") for _ in inputs]

        if __name__ == "__main__":
            app.serve()
        ```

    Args:
        name: Application name used in FastAPI title and logging.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR). Can be overridden
            by THALAMUS_LOG_LEVEL environment variable.
        lazy_load: If True, models are loaded on first request instead of at startup.
    """

    def __init__(
        self,
        name: str = "thalamus",
        log_level: str = "INFO",
        lazy_load: bool = False,
    ) -> None:
        self._name = name
        self._log_level = log_level
        self._lazy_load = lazy_load
        self._registry = ModelRegistry()
        self._app: FastAPI | None = None
        self._start_time: float = 0.0
        self._load_lock = Lock()

    def model(
        self,
        *,
        model_id: str | None = None,
        version: str = "1.0.0",
        description: str | None = None,
        default: bool = False,
        default_version: bool = False,
        critical: bool = True,
        weights: dict[str, WeightSource] | None = None,
        device: str = "auto",
        input_type: type,
        output_type: type,
    ) -> Callable[[type], type]:
        """Decorator to register a model class with the application.

        The decorated class should implement a `predict` method that takes a list
        of inputs and returns a list of outputs. Optionally, it can implement
        `load`, `preprocess`, `postprocess`, and `is_ready` methods.

        Args:
            model_id: Unique identifier for the model. Defaults to class name.
            version: Semantic version string (e.g., "1.0.0").
            description: Human-readable description. Defaults to class docstring.
            default: If True, this model is used when no model is specified in requests.
            default_version: If True, this version is used when no version is specified.
            critical: If True, the /ready endpoint waits for this model to load.
            weights: Dictionary mapping weight names to WeightSource configurations
                (S3Weight, HFWeight, or HTTPWeight). These are fetched and passed
                to the model's load() method.
            device: Device preference ("auto", "cpu", "cuda", "cuda:0", "mps").
            input_type: Pydantic model for input validation (required).
            output_type: Pydantic model for output serialization (required).

        Returns:
            Decorator function that registers the class and returns it unchanged.
        """

        def decorator(cls: type) -> type:
            spec = ModelSpec.from_class(
                cls,
                model_id,
                version,
                description,
                default,
                default_version,
                critical,
                weights,
                device,
                input_type,
                output_type,
            )
            self._registry.register(spec)
            return cls

        return decorator

    def _load_model(self, spec: ModelSpec) -> None:
        if spec.instance is not None:
            return

        log.info("model_loading", model=spec.id, version=spec.version)
        start = time.perf_counter()

        weights: dict[str, Path] = {}
        for weight_name, weight_source in spec.weights.items():
            local_path = fetch_weight(weight_source)
            weights[weight_name] = local_path

        device = GPUAllocator.get().allocate(spec.device_preference)
        spec.device = device

        spec.instance = spec.cls()
        if hasattr(spec.instance, "load"):
            spec.instance.load(weights, device)

        ms = (time.perf_counter() - start) * 1000
        log.info(
            "model_loaded",
            model=spec.id,
            version=spec.version,
            device=device,
            ms=round(ms, 2),
        )
        MODEL_INFO.info({"model_id": spec.id, "version": spec.version})

    def _ensure_loaded(self, spec: ModelSpec) -> None:
        if spec.instance is not None:
            return

        with self._load_lock:
            if spec.instance is not None:
                return
            self._load_model(spec)

    def get_uptime(self) -> float:
        if self._start_time == 0.0:
            return 0.0
        return time.perf_counter() - self._start_time

    def _build(self) -> FastAPI:
        log_level = os.environ.get("THALAMUS_LOG_LEVEL", self._log_level)
        setup_logging(log_level)
        self._start_time = time.perf_counter()

        @asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncIterator[None]:
            if not self._lazy_load:
                for m in self._registry.all():
                    self._load_model(m)
            yield
            log.info("shutting_down")

        app = FastAPI(title=self._name, lifespan=lifespan)
        app.add_middleware(APIKeyAuth)
        app.add_middleware(RequestLogging)

        @app.exception_handler(ValidationError)
        async def validation_error_handler(
            _request: Request, exc: ValidationError
        ) -> JSONResponse:
            return JSONResponse({"error": exc.errors()}, status_code=422)

        @app.exception_handler(ValueError)
        async def value_error_handler(
            request: Request, exc: ValueError
        ) -> JSONResponse:
            log.warning("bad_request", path=request.url.path, error=str(exc))
            return JSONResponse({"error": str(exc)}, status_code=400)

        @app.exception_handler(TypeError)
        async def type_error_handler(request: Request, exc: TypeError) -> JSONResponse:
            log.warning("bad_request", path=request.url.path, error=str(exc))
            return JSONResponse({"error": str(exc)}, status_code=400)

        @app.exception_handler(Exception)
        async def unhandled_error_handler(
            request: Request, exc: Exception
        ) -> JSONResponse:
            log.exception("unhandled_error", path=request.url.path, error=str(exc))
            return JSONResponse({"error": "Internal Server Error"}, status_code=500)

        ctx = RouteContext(
            registry=self._registry,
            ensure_loaded=self._ensure_loaded,
            get_uptime=self.get_uptime,
        )
        app.include_router(create_routes(ctx))
        return app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if self._app is None:
            with self._load_lock:
                if self._app is None:
                    self._app = self._build()
        await self._app(scope, receive, send)

    def serve(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Start the HTTP server.

        This is a convenience method that runs the application using uvicorn.
        For production deployments, consider running with uvicorn directly
        or using a process manager like gunicorn.

        Args:
            host: Host address to bind to.
            port: Port number to listen on.
        """
        import uvicorn

        uvicorn.run(self, host=host, port=port)
