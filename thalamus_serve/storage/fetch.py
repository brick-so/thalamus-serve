import os
import threading
from pathlib import Path

import boto3
import httpx
from huggingface_hub import hf_hub_download, snapshot_download

from thalamus_serve.config import HTTPWeight, HFWeight, S3Weight, WeightSource
from thalamus_serve.infra.cache import WeightCache
from thalamus_serve.observability.logging import log
from thalamus_serve.schemas.storage import S3Ref

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
    return _get_cache()


def _s3_client():
    if not hasattr(_thread_local, "s3_client"):
        _thread_local.s3_client = boto3.client("s3")
    return _thread_local.s3_client


def fetch_weight(source: WeightSource) -> Path:
    if isinstance(source, S3Weight):
        return _fetch_s3_weight(source)
    if isinstance(source, HFWeight):
        return _fetch_hf_weight(source)
    if isinstance(source, HTTPWeight):
        return _fetch_http(source.url, use_cache=True, timeout=300.0)
    raise ValueError(f"Unknown weight source type: {type(source)}")


def _fetch_s3_weight(source: S3Weight) -> Path:
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


def _fetch_hf_weight(source: HFWeight) -> Path:
    token = os.environ.get("HF_TOKEN")
    hf_cache_dir = _get_cache().cache_dir / "huggingface"

    log.info(
        "fetching_hf",
        repo=source.repo,
        filename=source.filename or "full_repo",
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

    log.info("fetched_hf", path=path)
    return Path(path)


def fetch(
    source: str | S3Ref,
    filename: str | None = None,
    cache: bool = True,
    timeout: float = 300.0,
) -> Path:
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
    ref = dest if isinstance(dest, S3Ref) else S3Ref.from_uri(dest)
    log.info("uploading", dest=ref.uri)
    _s3_client().upload_file(str(local), ref.bucket, ref.key)
    log.info("uploaded", dest=ref.uri)
    return ref


def exists_s3(ref: str | S3Ref) -> bool:
    if isinstance(ref, str):
        ref = S3Ref.from_uri(ref)
    try:
        _s3_client().head_object(Bucket=ref.bucket, Key=ref.key)
        return True
    except Exception:
        return False
