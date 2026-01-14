from prometheus_client import Counter, Gauge, Histogram, Info

MODEL_INFO = Info("thalamus_model", "Model metadata")

REQUESTS = Counter(
    "thalamus_requests_total",
    "Total requests",
    ["model", "status"],
)

LATENCY = Histogram(
    "thalamus_latency_seconds",
    "Request latency",
    ["model"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30],
)

BATCH_SIZE = Histogram(
    "thalamus_batch_size",
    "Batch sizes",
    ["model"],
    buckets=[1, 2, 4, 8, 16, 32, 64, 128],
)

PREPROCESSING_LATENCY = Histogram(
    "thalamus_preprocessing_seconds",
    "Preprocessing latency",
    ["model"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

INFERENCE_LATENCY = Histogram(
    "thalamus_inference_seconds",
    "Inference latency",
    ["model"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30],
)

POSTPROCESSING_LATENCY = Histogram(
    "thalamus_postprocessing_seconds",
    "Postprocessing latency",
    ["model"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

MODELS_LOADED = Gauge(
    "thalamus_models_loaded",
    "Number of loaded models",
)

CACHE_SIZE_BYTES = Gauge(
    "thalamus_cache_size_bytes",
    "Cache size in bytes",
)

CACHE_FILE_COUNT = Gauge(
    "thalamus_cache_file_count",
    "Number of files in cache",
)

CACHE_HITS = Counter(
    "thalamus_cache_hits_total",
    "Total cache hits",
)

CACHE_MISSES = Counter(
    "thalamus_cache_misses_total",
    "Total cache misses",
)

GPU_MEMORY_USED_MB = Gauge(
    "thalamus_gpu_memory_used_mb",
    "GPU memory used in MB",
    ["device_type", "device_index"],
)

GPU_MEMORY_TOTAL_MB = Gauge(
    "thalamus_gpu_memory_total_mb",
    "Total GPU memory in MB",
    ["device_type", "device_index"],
)
