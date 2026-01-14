"""GPU detection, memory management, and device allocation utilities."""

from enum import Enum
from threading import Lock
from typing import Any

from pydantic import BaseModel


class DeviceType(Enum):
    """Supported compute device types."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class DeviceInfo(BaseModel, frozen=True):
    """Information about a single compute device."""

    device_type: DeviceType
    device_index: int
    name: str
    total_memory_mb: float


class GPUStatus(BaseModel, frozen=True):
    """Status of available GPU/accelerator devices."""

    available: bool
    device_count: int
    devices: tuple[DeviceInfo, ...]
    torch_version: str | None


def _get_torch() -> Any | None:
    try:
        import torch

        return torch
    except ImportError:
        return None


def detect_devices() -> GPUStatus:
    """Detect available GPU/accelerator devices.

    Returns:
        GPUStatus containing information about available CUDA or MPS devices.
        If PyTorch is not installed, returns status with available=False.
    """
    torch = _get_torch()
    if torch is None:
        return GPUStatus(
            available=False,
            device_count=0,
            devices=(),
            torch_version=None,
        )

    devices: list[DeviceInfo] = []

    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                devices.append(
                    DeviceInfo(
                        device_type=DeviceType.CUDA,
                        device_index=i,
                        name=props.name,
                        total_memory_mb=round(props.total_memory / 1e6, 2),
                    )
                )
        elif torch.backends.mps.is_available():
            devices.append(
                DeviceInfo(
                    device_type=DeviceType.MPS,
                    device_index=0,
                    name="Apple Silicon",
                    total_memory_mb=round(torch.mps.recommended_max_memory() / 1e6, 2),
                )
            )
    except Exception:
        pass

    return GPUStatus(
        available=len(devices) > 0,
        device_count=len(devices),
        devices=tuple(devices),
        torch_version=torch.__version__,
    )


def get_memory(device: str) -> tuple[float, float] | None:
    """Get memory usage for a device.

    Args:
        device: Device string (e.g., "cuda:0", "mps", "cpu").

    Returns:
        Tuple of (used_mb, total_mb), or None if device is CPU or unavailable.
    """
    torch = _get_torch()
    if torch is None:
        return None

    if device == "cpu":
        return None

    try:
        if device == "mps" and torch.backends.mps.is_available():
            used = torch.mps.current_allocated_memory() / 1e6
            total = torch.mps.recommended_max_memory() / 1e6
            return (round(used, 2), round(total, 2))

        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                return None
            idx = 0
            if ":" in device:
                idx = int(device.split(":")[1])
            used = torch.cuda.memory_allocated(idx) / 1e6
            total = torch.cuda.get_device_properties(idx).total_memory / 1e6
            return (round(used, 2), round(total, 2))
    except Exception:
        pass

    return None


def clear_cache(device: str | None = None) -> None:
    """Clear GPU memory cache.

    Args:
        device: Specific device to clear, or None to clear all devices.
    """
    torch = _get_torch()
    if torch is None:
        return

    try:
        if device is None or device.startswith("cuda"):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if device is None or device == "mps":
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
    except Exception:
        pass

    import gc

    gc.collect()


def get_optimal_device() -> str:
    """Get the optimal compute device.

    Returns:
        Device string: "cuda:0" if CUDA available, "mps" if Apple Silicon, else "cpu".
    """
    torch = _get_torch()
    if torch is None:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda:0"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def get_status() -> dict[str, Any]:
    """Get comprehensive GPU status including memory usage.

    Returns:
        Dictionary with device availability, count, torch version,
        and per-device memory info.
    """
    status = detect_devices()
    memory_info: list[dict[str, Any]] = []

    for device in status.devices:
        device_str = (
            f"{device.device_type.value}:{device.device_index}"
            if device.device_type == DeviceType.CUDA
            else device.device_type.value
        )
        mem = get_memory(device_str)
        memory_info.append(
            {
                "device": device_str,
                "name": device.name,
                "total_mb": device.total_memory_mb,
                "used_mb": mem[0] if mem else None,
                "available_mb": (device.total_memory_mb - mem[0]) if mem else None,
            }
        )

    return {
        "available": status.available,
        "device_count": status.device_count,
        "torch_version": status.torch_version,
        "devices": memory_info,
    }


class GPUAllocator:
    """Thread-safe GPU device allocator using a singleton pattern.

    Tracks model allocations across devices and distributes new models
    to devices with the most available memory.
    """

    _instance: "GPUAllocator | None" = None

    def __init__(self) -> None:
        self._lock = Lock()
        self._allocations: dict[int, int] = {}
        self._device_type: DeviceType = DeviceType.CPU
        self._init_devices()

    def _init_devices(self) -> None:
        status = detect_devices()
        if not status.devices:
            return

        self._device_type = status.devices[0].device_type
        for device in status.devices:
            self._allocations[device.device_index] = 0

    def _get_device_available_memory(self, idx: int) -> float:
        device_str = "mps" if self._device_type == DeviceType.MPS else f"cuda:{idx}"
        mem = get_memory(device_str)
        if mem is None:
            return 0.0
        used, total = mem
        return total - used

    @classmethod
    def get(cls) -> "GPUAllocator":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    def allocate(self, preference: str) -> str:
        if preference == "cpu":
            return "cpu"

        if not self._allocations:
            return "cpu"

        if preference == "mps":
            with self._lock:
                if 0 in self._allocations:
                    self._allocations[0] += 1
            return "mps"

        if preference.startswith("cuda:"):
            idx = int(preference.split(":")[1])
            with self._lock:
                if idx in self._allocations:
                    self._allocations[idx] += 1
            return preference

        if preference == "cuda" or preference == "auto":
            with self._lock:
                if self._device_type == DeviceType.MPS:
                    self._allocations[0] = self._allocations.get(0, 0) + 1
                    return "mps"

                if not self._allocations:
                    return "cpu"

                idx = max(
                    self._allocations.keys(),
                    key=self._get_device_available_memory,
                )
                self._allocations[idx] += 1
                return f"cuda:{idx}"

        return "cpu"

    def release(self, device: str) -> None:
        if device == "cpu":
            return

        idx: int | None = None
        if device == "mps":
            idx = 0
        elif device.startswith("cuda:"):
            idx = int(device.split(":")[1])

        if idx is not None:
            with self._lock:
                if idx in self._allocations:
                    self._allocations[idx] = max(0, self._allocations[idx] - 1)

    def get_allocations(self) -> dict[int, int]:
        with self._lock:
            return dict(self._allocations)
