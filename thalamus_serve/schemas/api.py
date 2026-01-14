from typing import Any

from pydantic import BaseModel, Field


class ModelStatus(BaseModel):
    id: str
    version: str
    ready: bool
    is_default: bool = False
    is_default_version: bool = False
    critical: bool = True


class HealthResponse(BaseModel):
    status: str
    models: list[ModelStatus]


class SchemaResponse(BaseModel):
    id: str
    version: str
    description: str
    input: dict[str, Any]
    output: dict[str, Any]


class PredictRequest(BaseModel):
    model: str | None = None
    version: str | None = None
    inputs: list[dict[str, Any]] = Field(..., min_length=1)


class PredictMeta(BaseModel):
    model: str
    version: str
    latency_ms: float
    batch_size: int
    preprocessing_ms: float | None = None
    inference_ms: float | None = None
    postprocessing_ms: float | None = None
    device: str | None = None


class PredictResponse(BaseModel):
    outputs: list[Any]
    meta: PredictMeta


class CacheInfo(BaseModel):
    size_bytes: int
    file_count: int
    max_size_bytes: int
    hit_rate: float


class StatusResponse(BaseModel):
    models: list[ModelStatus]
    cache: CacheInfo | None
    gpu: dict[str, Any] | None
    uptime_seconds: float


class CacheClearResponse(BaseModel):
    cleared_bytes: int
    cleared_files: int


class UnloadRequest(BaseModel):
    version: str | None = None


class UnloadResponse(BaseModel):
    model_id: str
    version: str | None
    unloaded: bool
    versions_unloaded: list[str]


class ReadyResponse(BaseModel):
    ready: bool
    models: list[ModelStatus]
