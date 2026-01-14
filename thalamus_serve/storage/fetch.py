"""Model weight fetching from S3, HuggingFace Hub, and HTTP sources."""

import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import boto3
import httpx
from huggingface_hub import hf_hub_download, snapshot_download

from thalamus_serve.config import HFWeight, HTTPWeight, S3Weight, WeightSource
from thalamus_serve.infra.cache import WeightCache
from thalamus_serve.observability.logging import log
from thalamus_serve.schemas.storage import S3Ref

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

_cache: WeightCache | None = None
_thread_local = threading.local()


def _get_cache() -> WeightCache:
    global _cache
    if _cache is None:
        cache_dir = Path(os.environ.get("THALAMUS_CACHE_DIR", "/tmp/thalamus"))
        max_size_gb = float(os.environ.get("THALAMUS_CACHE_MAX_GB", "50"))
        _cache = WeightCache(cache_dir, max_size_gb)
    return _cache


def get_cache() -> WeightCache:
    """Get the global weight cache instance."""
    return _get_cache()


def _s3_client() -> "S3Client":
    if not hasattr(_thread_local, "s3_client"):
        _thread_local.s3_client = boto3.client("s3")
    return _thread_local.s3_client


def fetch_weight(source: WeightSource) -> Path:
    """Fetch model weights from a configured source.

    This is the internal function used by the model loading system.
    Downloads weights from S3, HuggingFace Hub, or HTTP based on
    the source configuration in thalamus-deploy.json.

    Args:
        source: Weight source configuration (S3Weight, HFWeight, or HTTPWeight).

    Returns:
        Path to the downloaded/cached weight file or directory.
    """
    if isinstance(source, S3Weight):
        return _fetch_s3_weight(source)
    if isinstance(source, HFWeight):
        return _fetch_hf_weight(source)
    if isinstance(source, HTTPWeight):
        return _fetch_http_weight(source)
    raise ValueError(f"Unknown weight source type: {type(source)}")


def _fetch_s3_weight(source: S3Weight) -> Path:
    """Fetch weights from S3.

    Returns:
        Path to downloaded file (if key specified) or directory (if prefix specified).
    """
    if source.prefix is not None:
        return _fetch_s3_prefix(source)

    # Single file download
    cache_key = f"s3://{source.bucket}/{source.key}"
    weight_cache = _get_cache()

    cached = weight_cache.get(cache_key)
    if cached:
        log.debug("cache_hit", source=cache_key, path=str(cached))
        return cached

    def download(dest: Path) -> None:
        log.info("downloading", source=cache_key)
        _s3_client().download_file(source.bucket, source.key, str(dest))
        log.info(
            "downloaded",
            source=cache_key,
            size_mb=round(dest.stat().st_size / 1048576, 2),
        )

    return weight_cache.put(cache_key, download)


def _fetch_s3_prefix(source: S3Weight) -> Path:
    """Fetch all files under an S3 prefix (directory download for sharded models)."""
    import hashlib

    assert source.prefix is not None, "prefix must be set for _fetch_s3_prefix"
    prefix = source.prefix
    cache_key = f"s3://{source.bucket}/{prefix}"
    weight_cache = _get_cache()

    # Create a stable directory name based on the prefix
    prefix_hash = hashlib.sha256(cache_key.encode()).hexdigest()[:16]
    cache_dir = weight_cache.cache_dir / "s3_prefixes" / prefix_hash
    marker_file = cache_dir / ".complete"

    # Check if already downloaded
    if marker_file.exists():
        log.debug("cache_hit", source=cache_key, path=str(cache_dir))
        cache_dir.touch()  # Update access time for LRU
        return cache_dir

    log.info("downloading_prefix", source=cache_key)

    # List all objects under the prefix
    client = _s3_client()
    paginator = client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=source.bucket, Prefix=prefix)

    cache_dir.mkdir(parents=True, exist_ok=True)
    total_size = 0
    file_count = 0

    for page in pages:
        for obj in page.get("Contents", []):
            obj_key = obj["Key"]
            # Get relative path from prefix
            relative_path = obj_key[len(prefix) :].lstrip("/")
            if not relative_path:
                continue

            local_path = cache_dir / relative_path
            local_path.parent.mkdir(parents=True, exist_ok=True)

            log.debug("downloading_file", key=obj_key)
            client.download_file(source.bucket, obj_key, str(local_path))
            total_size += local_path.stat().st_size
            file_count += 1

    # Mark as complete
    marker_file.touch()

    log.info(
        "downloaded_prefix",
        source=cache_key,
        files=file_count,
        size_mb=round(total_size / 1048576, 2),
    )

    return cache_dir


def _fetch_hf_weight(source: HFWeight) -> Path:
    """Fetch weights from HuggingFace Hub.

    Returns:
        Path to downloaded file (if filename specified) or directory (if snapshot).
    """
    token = os.environ.get("HF_TOKEN")
    hf_cache_dir = _get_cache().cache_dir / "huggingface"

    log.info(
        "fetching_hf",
        repo=source.repo,
        filename=source.filename or "snapshot",
        revision=source.revision,
    )

    if source.filename:
        path = hf_hub_download(
            repo_id=source.repo,
            filename=source.filename,
            revision=source.revision,
            token=token,
            cache_dir=hf_cache_dir,
        )
    else:
        path = snapshot_download(
            repo_id=source.repo,
            revision=source.revision,
            token=token,
            cache_dir=hf_cache_dir,
        )

    result = Path(path)

    # Touch path to update access time for LRU tracking
    try:
        result.touch()
    except OSError:
        pass

    log.info("fetched_hf", path=str(result), is_directory=result.is_dir())
    return result


def _fetch_http_weight(source: HTTPWeight) -> Path:
    """Fetch weights from HTTP/HTTPS.

    Returns:
        Path to downloaded file (single URL) or directory (multiple URLs).
    """
    if len(source.urls) > 1:
        return _fetch_http_urls(source)

    # Single file download
    return _fetch_http(source.urls[0], use_cache=True, timeout=300.0)


def _fetch_http_urls(source: HTTPWeight) -> Path:
    """Fetch multiple files from HTTP URLs (for sharded models)."""
    import hashlib

    urls = source.urls
    # Create cache key from sorted URLs for consistency
    cache_key = "http_multi:" + ",".join(sorted(urls))
    key_hash = hashlib.sha256(cache_key.encode()).hexdigest()[:16]

    weight_cache = _get_cache()
    cache_dir = weight_cache.cache_dir / "http_urls" / key_hash
    marker_file = cache_dir / ".complete"

    # Check if already downloaded
    if marker_file.exists():
        log.debug("cache_hit", source=f"http_urls[{len(urls)}]", path=str(cache_dir))
        cache_dir.touch()  # Update access time for LRU
        return cache_dir

    log.info("downloading_urls", count=len(urls))

    cache_dir.mkdir(parents=True, exist_ok=True)
    total_size = 0

    for url in urls:
        # Extract filename from URL
        filename = url.split("/")[-1].split("?")[0] or "file"
        local_path = cache_dir / filename

        log.debug("downloading_file", url=url)
        with httpx.stream("GET", url, timeout=300.0, follow_redirects=True) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_bytes(8192):
                    f.write(chunk)
        total_size += local_path.stat().st_size

    # Mark as complete
    marker_file.touch()

    log.info(
        "downloaded_urls",
        files=len(urls),
        size_mb=round(total_size / 1048576, 2),
    )

    return cache_dir


def fetch(
    source: str | S3Ref,
    filename: str | None = None,
    cache: bool = True,
    timeout: float = 300.0,
) -> Path:
    """Fetch a file from S3 or HTTP URL.

    This is the general-purpose fetch function for downloading arbitrary files.

    Args:
        source: S3 URI (s3://bucket/key), HTTP URL, or S3Ref object.
        filename: Optional filename override for the cached file.
        cache: Whether to use the cache (default True).
        timeout: HTTP request timeout in seconds.

    Returns:
        Path to the downloaded/cached file.
    """
    if isinstance(source, S3Ref):
        return _fetch_s3(source, filename, cache)
    if source.startswith("s3://"):
        return _fetch_s3(S3Ref.from_uri(source), filename, cache)
    return _fetch_http(source, use_cache=cache, timeout=timeout)


def _fetch_s3(ref: S3Ref, filename: str | None, use_cache: bool) -> Path:
    cache_key = ref.uri
    weight_cache = _get_cache()

    if use_cache:
        cached = weight_cache.get(cache_key)
        if cached:
            log.debug("cache_hit", source=ref.uri, path=str(cached))
            return cached

    def download(dest: Path) -> None:
        log.info("downloading", source=ref.uri)
        _s3_client().download_file(ref.bucket, ref.key, str(dest))
        log.info(
            "downloaded",
            source=ref.uri,
            size_mb=round(dest.stat().st_size / 1048576, 2),
        )

    return weight_cache.put(cache_key, download)


def _fetch_http(url: str, use_cache: bool, timeout: float) -> Path:
    cache_key = url
    weight_cache = _get_cache()

    if use_cache:
        cached = weight_cache.get(cache_key)
        if cached:
            log.debug("cache_hit", source=url, path=str(cached))
            return cached

    def download(dest: Path) -> None:
        log.info("downloading", source=url)
        with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_bytes(8192):
                    f.write(chunk)
        log.info(
            "downloaded", source=url, size_mb=round(dest.stat().st_size / 1048576, 2)
        )

    return weight_cache.put(cache_key, download)


def upload_s3(local: Path | str, dest: str | S3Ref) -> S3Ref:
    """Upload a file to S3.

    Args:
        local: Path to local file.
        dest: S3 URI (s3://bucket/key) or S3Ref object.

    Returns:
        S3Ref pointing to the uploaded file.
    """
    ref = dest if isinstance(dest, S3Ref) else S3Ref.from_uri(dest)
    log.info("uploading", dest=ref.uri)
    _s3_client().upload_file(str(local), ref.bucket, ref.key)
    log.info("uploaded", dest=ref.uri)
    return ref


def exists_s3(ref: str | S3Ref) -> bool:
    """Check if an object exists in S3.

    Args:
        ref: S3 URI (s3://bucket/key) or S3Ref object.

    Returns:
        True if the object exists, False otherwise.
    """
    if isinstance(ref, str):
        ref = S3Ref.from_uri(ref)
    try:
        _s3_client().head_object(Bucket=ref.bucket, Key=ref.key)
        return True
    except Exception:
        return False
