import time
from collections.abc import Callable

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from thalamus_serve.core.model import ModelRegistry, ModelSpec
from thalamus_serve.infra import gpu
from thalamus_serve.observability.logging import log
from thalamus_serve.observability.metrics import (
    BATCH_SIZE,
    CACHE_FILE_COUNT,
    CACHE_SIZE_BYTES,
    GPU_MEMORY_TOTAL_MB,
    GPU_MEMORY_USED_MB,
    INFERENCE_LATENCY,
    LATENCY,
    MODELS_LOADED,
    POSTPROCESSING_LATENCY,
    PREPROCESSING_LATENCY,
    REQUESTS,
)
from thalamus_serve.schemas.api import (
    CacheClearResponse,
    CacheInfo,
    HealthResponse,
    ModelStatus,
    PredictMeta,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
    SchemaResponse,
    StatusResponse,
    UnloadRequest,
    UnloadResponse,
)
from thalamus_serve.storage.fetch import get_cache


class RouteContext:
    def __init__(
        self,
        registry: ModelRegistry,
        ensure_loaded: Callable[[ModelSpec], None],
        get_uptime: Callable[[], float],
    ) -> None:
        self.registry = registry
        self.ensure_loaded = ensure_loaded
        self.get_uptime = get_uptime


def create_routes(ctx: RouteContext) -> APIRouter:
    r = APIRouter()
    registry = ctx.registry

    @r.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        statuses = []
        for m in registry.all():
            ready = getattr(m.instance, "is_ready", True) if m.instance else False
            statuses.append(
                ModelStatus(
                    id=m.id,
                    version=m.version,
                    ready=ready,
                    is_default=m.is_default,
                    is_default_version=m.is_default_version,
                    critical=m.is_critical,
                )
            )
        return HealthResponse(status="ok", models=statuses)

    @r.get("/ready", response_model=ReadyResponse)
    def ready() -> ReadyResponse:
        statuses = []
        all_critical_ready = True
        for m in registry.all():
            ready = getattr(m.instance, "is_ready", True) if m.instance else False
            if m.is_critical and not ready:
                all_critical_ready = False
            statuses.append(
                ModelStatus(
                    id=m.id,
                    version=m.version,
                    ready=ready,
                    is_default=m.is_default,
                    is_default_version=m.is_default_version,
                    critical=m.is_critical,
                )
            )
        return ReadyResponse(ready=all_critical_ready, models=statuses)

    @r.get("/status", response_model=StatusResponse)
    def status() -> StatusResponse:
        statuses = []
        loaded_count = 0
        for m in registry.all():
            ready = getattr(m.instance, "is_ready", True) if m.instance else False
            if m.instance is not None:
                loaded_count += 1
            statuses.append(
                ModelStatus(
                    id=m.id,
                    version=m.version,
                    ready=ready,
                    is_default=m.is_default,
                    is_default_version=m.is_default_version,
                    critical=m.is_critical,
                )
            )

        MODELS_LOADED.set(loaded_count)

        cache = get_cache()
        cache_stats = cache.stats()
        CACHE_SIZE_BYTES.set(cache_stats.total_size_bytes)
        CACHE_FILE_COUNT.set(cache_stats.file_count)
        cache_info = CacheInfo(
            size_bytes=cache_stats.total_size_bytes,
            file_count=cache_stats.file_count,
            max_size_bytes=cache_stats.max_size_bytes,
            hit_rate=cache_stats.hit_rate,
        )

        gpu_status = gpu.get_status()

        if gpu_status and "devices" in gpu_status:
            for device in gpu_status["devices"]:
                d_str = device["device"]
                d_type = d_str.split(":")[0]
                d_idx = d_str.split(":")[1] if ":" in d_str else "0"
                used = device.get("used_mb")
                total = device.get("total_mb")
                if used is not None:
                    GPU_MEMORY_USED_MB.labels(
                        device_type=d_type, device_index=d_idx
                    ).set(used)
                if total is not None:
                    GPU_MEMORY_TOTAL_MB.labels(
                        device_type=d_type, device_index=d_idx
                    ).set(total)

        return StatusResponse(
            models=statuses,
            cache=cache_info,
            gpu=gpu_status,
            uptime_seconds=round(ctx.get_uptime(), 2),
        )

    @r.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> PlainTextResponse:
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @r.get("/schema", response_model=list[SchemaResponse])
    def schemas() -> list[SchemaResponse]:
        return [
            SchemaResponse(
                id=m.id,
                version=m.version,
                description=m.description,
                input=m.input_type.model_json_schema(),
                output=m.output_type.model_json_schema(),
            )
            for m in registry.all()
        ]

    @r.get("/schema/{model_id}", response_model=SchemaResponse)
    def schema(
        model_id: str,
        version: str | None = Query(default=None),
    ) -> SchemaResponse:
        m = registry.get(model_id, version)
        if not m:
            raise HTTPException(
                404, f"Model not found: {model_id}@{version or 'latest'}"
            )
        return SchemaResponse(
            id=m.id,
            version=m.version,
            description=m.description,
            input=m.input_type.model_json_schema(),
            output=m.output_type.model_json_schema(),
        )

    @r.get("/schema/{model_id}/versions", response_model=list[str])
    def schema_versions(model_id: str) -> list[str]:
        versions = registry.get_versions(model_id)
        if not versions:
            raise HTTPException(404, f"Model not found: {model_id}")
        return versions

    @r.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        if req.model is None:
            m = registry.get_default()
            if m is None:
                raise HTTPException(422, "No default model configured")
        else:
            m = registry.get(req.model, req.version)
            if m is None:
                raise HTTPException(
                    404, f"Model not found: {req.model}@{req.version or 'latest'}"
                )

        ctx.ensure_loaded(m)

        inputs = [m.input_type.model_validate(i) for i in req.inputs]

        preprocessing_ms: float | None = None
        postprocessing_ms: float | None = None

        start_total = time.perf_counter()

        if m.has_preprocess:
            start_pre = time.perf_counter()
            inputs = m.instance.preprocess(inputs)
            preprocessing_ms = round((time.perf_counter() - start_pre) * 1000, 2)
            PREPROCESSING_LATENCY.labels(model=m.id).observe(preprocessing_ms / 1000)

        start_infer = time.perf_counter()
        try:
            outputs = m.instance.predict(inputs)
            REQUESTS.labels(model=m.id, status="ok").inc()
        except Exception as e:
            REQUESTS.labels(model=m.id, status="error").inc()
            log.exception("predict_error", model=m.id)
            raise HTTPException(500, str(e)) from e

        inference_ms = round((time.perf_counter() - start_infer) * 1000, 2)
        INFERENCE_LATENCY.labels(model=m.id).observe(inference_ms / 1000)

        if m.has_postprocess:
            start_post = time.perf_counter()
            outputs = m.instance.postprocess(outputs)
            postprocessing_ms = round((time.perf_counter() - start_post) * 1000, 2)
            POSTPROCESSING_LATENCY.labels(model=m.id).observe(postprocessing_ms / 1000)

        total_ms = round((time.perf_counter() - start_total) * 1000, 2)
        LATENCY.labels(model=m.id).observe(total_ms / 1000)
        BATCH_SIZE.labels(model=m.id).observe(len(req.inputs))

        log.info(
            "predict", model=m.id, version=m.version, batch=len(req.inputs), ms=total_ms
        )

        return PredictResponse(
            outputs=[o.model_dump() for o in outputs],
            meta=PredictMeta(
                model=m.id,
                version=m.version,
                latency_ms=total_ms,
                batch_size=len(req.inputs),
                preprocessing_ms=preprocessing_ms,
                inference_ms=inference_ms,
                postprocessing_ms=postprocessing_ms,
            ),
        )

    @r.post("/cache/clear", response_model=CacheClearResponse)
    def clear_cache() -> CacheClearResponse:
        cache = get_cache()
        cleared_bytes, cleared_files = cache.clear()
        log.info("cache_cleared", bytes=cleared_bytes, files=cleared_files)
        return CacheClearResponse(
            cleared_bytes=cleared_bytes,
            cleared_files=cleared_files,
        )

    @r.post("/models/{model_id}/unload", response_model=UnloadResponse)
    def unload_model(model_id: str, req: UnloadRequest | None = None) -> UnloadResponse:
        if req is None:
            req = UnloadRequest()
        versions = registry.get_versions(model_id)
        if not versions:
            raise HTTPException(404, f"Model not found: {model_id}")

        specs_to_unload = (
            [registry.get(model_id, req.version)]
            if req.version
            else registry.all_for_model(model_id)
        )
        devices_to_clear = {
            spec.device for spec in specs_to_unload if spec and spec.device
        }

        unloaded = registry.unload(model_id, req.version)

        for device in devices_to_clear:
            gpu.clear_cache(device)

        log.info("model_unloaded", model=model_id, versions=unloaded)

        return UnloadResponse(
            model_id=model_id,
            version=req.version,
            unloaded=len(unloaded) > 0,
            versions_unloaded=unloaded,
        )

    return r
