"""LRU file cache for model weights with automatic eviction."""

import hashlib
import os
from collections.abc import Callable
from pathlib import Path
from threading import Lock

from pydantic import BaseModel, computed_field


class CacheStats(BaseModel, frozen=True):
    """Statistics about cache usage and performance."""

    total_size_bytes: int
    file_count: int
    max_size_bytes: int
    hit_count: int
    miss_count: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0


class WeightCache:
    """Thread-safe LRU file cache for model weights.

    Caches downloaded model weights to disk with automatic eviction when
    the cache exceeds the configured maximum size. Uses LRU (least recently
    used) eviction based on file access times.

    Args:
        cache_dir: Directory to store cached files.
        max_size_gb: Maximum cache size in gigabytes before eviction triggers.
    """

    def __init__(self, cache_dir: Path, max_size_gb: float = 50.0) -> None:
        self._cache_dir = cache_dir
        self._max_size_bytes = int(max_size_gb * 1e9)
        self._lock = Lock()
        self._hit_count = 0
        self._miss_count = 0
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    def _key_to_path(self, key: str) -> Path:
        key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
        filename = os.path.basename(key) or key_hash
        return self._cache_dir / f"{key_hash}_{filename}"

    def get(self, key: str) -> Path | None:
        """Get a cached file by key.

        Args:
            key: Cache key (typically a URL or identifier).

        Returns:
            Path to cached file if it exists, None otherwise.
        """
        with self._lock:
            path = self._key_to_path(key)
            if path.exists():
                self._hit_count += 1
                path.touch()
                return path
            self._miss_count += 1
            return None

    def put(self, key: str, download_fn: Callable[[Path], None]) -> Path:
        """Get or download a file into the cache.

        If the key exists in cache, returns the cached path. Otherwise,
        calls download_fn to download the file, caches it, and returns the path.

        Args:
            key: Cache key (typically a URL or identifier).
            download_fn: Function that downloads content to the given path.

        Returns:
            Path to the cached file.

        Raises:
            Exception: If download_fn raises an exception.
        """
        with self._lock:
            path = self._key_to_path(key)
            if path.exists():
                self._hit_count += 1
                path.touch()
                return path

            self._miss_count += 1
            self._evict_if_needed()

            temp_path = path.with_suffix(".tmp")
            try:
                download_fn(temp_path)
                temp_path.rename(path)
            except Exception:
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)
                raise
            return path

    def contains(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        with self._lock:
            return self._key_to_path(key).exists()

    def _get_thalamus_size(self) -> int:
        """Get total size of thalamus cache files (S3, HTTP downloads)."""
        return sum(f.stat().st_size for f in self._cache_dir.iterdir() if f.is_file())

    def _get_s3_prefix_size(self) -> int:
        """Get total size of S3 prefix directories (sharded model downloads)."""
        prefix_dir = self._cache_dir / "s3_prefixes"
        if not prefix_dir.exists():
            return 0
        total = 0
        for prefix_cache in prefix_dir.iterdir():
            if prefix_cache.is_dir():
                for f in prefix_cache.rglob("*"):
                    if f.is_file():
                        total += f.stat().st_size
        return total

    def _get_http_urls_size(self) -> int:
        """Get total size of HTTP URL directories (sharded model downloads)."""
        urls_dir = self._cache_dir / "http_urls"
        if not urls_dir.exists():
            return 0
        total = 0
        for url_cache in urls_dir.iterdir():
            if url_cache.is_dir():
                for f in url_cache.rglob("*"):
                    if f.is_file():
                        total += f.stat().st_size
        return total

    def _get_hf_size(self) -> int:
        """Get total size of HuggingFace blobs."""
        hf_dir = self._cache_dir / "huggingface"
        if not hf_dir.exists():
            return 0
        total = 0
        for model_dir in hf_dir.iterdir():
            if model_dir.is_dir() and model_dir.name.startswith("models--"):
                blobs_dir = model_dir / "blobs"
                if blobs_dir.exists():
                    for blob in blobs_dir.iterdir():
                        if blob.is_file():
                            total += blob.stat().st_size
        return total

    def _get_size(self) -> int:
        """Get total cache size including all sources."""
        return (
            self._get_thalamus_size()
            + self._get_s3_prefix_size()
            + self._get_http_urls_size()
            + self._get_hf_size()
        )

    def _evict_thalamus_files(self, target_bytes: int) -> int:
        """Evict thalamus cache files (S3 single files, HTTP) using LRU."""
        files = [
            (f, f.stat())
            for f in self._cache_dir.iterdir()
            if f.is_file() and f.suffix != ".tmp"
        ]
        files.sort(key=lambda x: x[1].st_atime)

        freed = 0
        for file_path, stat in files:
            if self._get_size() <= target_bytes:
                break
            try:
                freed += stat.st_size
                file_path.unlink()
            except OSError:
                pass

        return freed

    def _evict_s3_prefixes(self, target_bytes: int) -> int:
        """Evict S3 prefix directories using LRU."""
        import shutil

        prefix_dir = self._cache_dir / "s3_prefixes"
        if not prefix_dir.exists():
            return 0

        # Get directories with their access times
        dirs = []
        for d in prefix_dir.iterdir():
            if d.is_dir():
                try:
                    dirs.append((d, d.stat().st_atime))
                except OSError:
                    pass

        dirs.sort(key=lambda x: x[1])  # Sort by access time (oldest first)

        freed = 0
        for dir_path, _ in dirs:
            if self._get_size() <= target_bytes:
                break
            try:
                # Calculate directory size before deletion
                dir_size = sum(
                    f.stat().st_size for f in dir_path.rglob("*") if f.is_file()
                )
                shutil.rmtree(dir_path)
                freed += dir_size
            except OSError:
                pass

        return freed

    def _evict_http_urls(self, target_bytes: int) -> int:
        """Evict HTTP URL directories using LRU."""
        import shutil

        urls_dir = self._cache_dir / "http_urls"
        if not urls_dir.exists():
            return 0

        # Get directories with their access times
        dirs = []
        for d in urls_dir.iterdir():
            if d.is_dir():
                try:
                    dirs.append((d, d.stat().st_atime))
                except OSError:
                    pass

        dirs.sort(key=lambda x: x[1])  # Sort by access time (oldest first)

        freed = 0
        for dir_path, _ in dirs:
            if self._get_size() <= target_bytes:
                break
            try:
                # Calculate directory size before deletion
                dir_size = sum(
                    f.stat().st_size for f in dir_path.rglob("*") if f.is_file()
                )
                shutil.rmtree(dir_path)
                freed += dir_size
            except OSError:
                pass

        return freed

    def _evict_hf_if_needed(self, target_bytes: int) -> int:
        """Evict HuggingFace cache entries using HF's cache manager."""
        hf_dir = self._cache_dir / "huggingface"
        if not hf_dir.exists():
            return 0

        try:
            from huggingface_hub import scan_cache_dir
        except ImportError:
            return 0

        try:
            cache_info = scan_cache_dir(hf_dir)
        except Exception:
            return 0

        # Sort repos by last accessed time (oldest first)
        repos = sorted(cache_info.repos, key=lambda r: r.last_accessed)

        freed = 0
        for repo in repos:
            if self._get_size() <= target_bytes:
                break
            try:
                # Get all revisions for this repo and delete them
                revision_hashes = [rev.commit_hash for rev in repo.revisions]
                if revision_hashes:
                    strategy = cache_info.delete_revisions(*revision_hashes)
                    freed += strategy.expected_freed_size
                    strategy.execute()
            except Exception:
                pass

        return freed

    def _evict_if_needed(self) -> int:
        """Evict cache entries if over size limit."""
        current_size = self._get_size()
        if current_size <= self._max_size_bytes:
            return 0

        target_size = int(self._max_size_bytes * 0.8)
        freed = 0

        # First evict thalamus cache files (S3 single files, HTTP single files)
        freed += self._evict_thalamus_files(target_size)

        # Then evict S3 prefix directories
        if self._get_size() > target_size:
            freed += self._evict_s3_prefixes(target_size)

        # Then evict HTTP URL directories
        if self._get_size() > target_size:
            freed += self._evict_http_urls(target_size)

        # Finally evict HF cache
        if self._get_size() > target_size:
            freed += self._evict_hf_if_needed(target_size)

        return freed

    def clear(self) -> tuple[int, int]:
        """Clear all cached files including subdirectories.

        Clears thalamus cache files, S3 prefix directories, HTTP URL directories,
        and HuggingFace cache.

        Returns:
            Tuple of (bytes_freed, files_deleted).
        """
        import shutil

        with self._lock:
            total_bytes = 0
            total_files = 0

            # Clear files directly in cache directory
            for f in self._cache_dir.iterdir():
                if f.is_file():
                    try:
                        total_bytes += f.stat().st_size
                        f.unlink()
                        total_files += 1
                    except OSError:
                        pass

            # Clear S3 prefix directories
            s3_prefix_dir = self._cache_dir / "s3_prefixes"
            if s3_prefix_dir.exists():
                for d in s3_prefix_dir.iterdir():
                    if d.is_dir():
                        try:
                            for f in d.rglob("*"):
                                if f.is_file():
                                    total_bytes += f.stat().st_size
                                    total_files += 1
                            shutil.rmtree(d)
                        except OSError:
                            pass

            # Clear HTTP URL directories
            http_urls_dir = self._cache_dir / "http_urls"
            if http_urls_dir.exists():
                for d in http_urls_dir.iterdir():
                    if d.is_dir():
                        try:
                            for f in d.rglob("*"):
                                if f.is_file():
                                    total_bytes += f.stat().st_size
                                    total_files += 1
                            shutil.rmtree(d)
                        except OSError:
                            pass

            # Clear HuggingFace cache
            hf_dir = self._cache_dir / "huggingface"
            if hf_dir.exists():
                try:
                    for f in hf_dir.rglob("*"):
                        if f.is_file():
                            total_bytes += f.stat().st_size
                            total_files += 1
                    shutil.rmtree(hf_dir)
                except OSError:
                    pass

            self._hit_count = 0
            self._miss_count = 0
            return (total_bytes, total_files)

    def stats(self) -> CacheStats:
        """Get cache statistics including size, file count, and hit rate."""
        with self._lock:
            return CacheStats(
                total_size_bytes=self._get_size(),
                file_count=sum(1 for f in self._cache_dir.iterdir() if f.is_file()),
                max_size_bytes=self._max_size_bytes,
                hit_count=self._hit_count,
                miss_count=self._miss_count,
            )
