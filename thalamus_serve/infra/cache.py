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

    @computed_field
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

    def _get_size(self) -> int:
        return sum(f.stat().st_size for f in self._cache_dir.iterdir() if f.is_file())

    def _evict_if_needed(self) -> int:
        current_size = self._get_size()
        if current_size <= self._max_size_bytes:
            return 0

        target_size = int(self._max_size_bytes * 0.8)
        files = [
            (f, f.stat())
            for f in self._cache_dir.iterdir()
            if f.is_file() and not f.suffix == ".tmp"
        ]
        files.sort(key=lambda x: x[1].st_atime)

        freed = 0
        for file_path, stat in files:
            if current_size - freed <= target_size:
                break
            try:
                size = stat.st_size
                file_path.unlink()
                freed += size
            except OSError:
                pass

        return freed

    def clear(self) -> tuple[int, int]:
        """Clear all cached files.

        Returns:
            Tuple of (bytes_freed, files_deleted).
        """
        with self._lock:
            total_bytes = 0
            total_files = 0
            for f in self._cache_dir.iterdir():
                if f.is_file():
                    try:
                        total_bytes += f.stat().st_size
                        f.unlink()
                        total_files += 1
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
