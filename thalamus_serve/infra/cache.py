import hashlib
import os
from pathlib import Path
from threading import Lock
from typing import Callable

from pydantic import BaseModel, computed_field


class CacheStats(BaseModel, frozen=True):
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
        with self._lock:
            path = self._key_to_path(key)
            if path.exists():
                self._hit_count += 1
                path.touch()
                return path
            self._miss_count += 1
            return None

    def put(self, key: str, download_fn: Callable[[Path], None]) -> Path:
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
        with self._lock:
            return CacheStats(
                total_size_bytes=self._get_size(),
                file_count=sum(1 for f in self._cache_dir.iterdir() if f.is_file()),
                max_size_bytes=self._max_size_bytes,
                hit_count=self._hit_count,
                miss_count=self._miss_count,
            )
