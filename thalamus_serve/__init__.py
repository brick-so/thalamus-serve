"""thalamus-serve: ML model serving framework with built-in observability.

A Python ML model serving framework built on FastAPI that provides standardized
deployment with built-in observability, caching, and GPU management.

Basic usage:
    from thalamus_serve import Thalamus
    from pydantic import BaseModel

    app = Thalamus()

    class Input(BaseModel):
        text: str

    class Output(BaseModel):
        label: str

    @app.model(model_id="classifier", default=True)
    class MyModel:
        def predict(self, inputs: list[Input]) -> list[Output]:
            return [Output(label="positive") for _ in inputs]

    if __name__ == "__main__":
        app.serve()
"""

from thalamus_serve.config import HFWeight, HTTPWeight, S3Weight, WeightSource
from thalamus_serve.core.app import Thalamus
from thalamus_serve.infra.cache import CacheStats, WeightCache
from thalamus_serve.infra.gpu import (
    DeviceInfo,
    DeviceType,
    GPUStatus,
    detect_devices,
    get_optimal_device,
)
from thalamus_serve.infra.gpu import (
    clear_cache as clear_gpu_cache,
)
from thalamus_serve.infra.gpu import (
    get_memory as get_gpu_memory,
)
from thalamus_serve.infra.gpu import (
    get_status as get_gpu_status,
)
from thalamus_serve.schemas.api import (
    CacheClearResponse,
    CacheInfo,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
    SchemaResponse,
    StatusResponse,
    UnloadRequest,
    UnloadResponse,
)
from thalamus_serve.schemas.common import Base64Data, BBox, Label, Prob, Span, Vector
from thalamus_serve.schemas.storage import S3PresignedUrl, S3Ref, Url
from thalamus_serve.storage.fetch import exists_s3, fetch, get_cache, upload_s3
from thalamus_serve.utils import env, require_env

__all__ = [
    "Thalamus",
    "S3Weight",
    "HFWeight",
    "HTTPWeight",
    "WeightSource",
    "fetch",
    "upload_s3",
    "exists_s3",
    "get_cache",
    "env",
    "require_env",
    "S3Ref",
    "S3PresignedUrl",
    "Url",
    "Base64Data",
    "BBox",
    "Label",
    "Vector",
    "Span",
    "Prob",
    "HealthResponse",
    "ReadyResponse",
    "SchemaResponse",
    "PredictRequest",
    "PredictResponse",
    "StatusResponse",
    "CacheInfo",
    "CacheClearResponse",
    "UnloadRequest",
    "UnloadResponse",
    "WeightCache",
    "CacheStats",
    "DeviceType",
    "DeviceInfo",
    "GPUStatus",
    "detect_devices",
    "get_gpu_memory",
    "clear_gpu_cache",
    "get_optimal_device",
    "get_gpu_status",
]
