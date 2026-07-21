from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from thalamus_serve.core.app import Thalamus
from thalamus_serve.infra import gpu
from thalamus_serve.infra.gpu import GPUAllocator, GPURequirementError


class GpuInput(BaseModel):
    data: str


class GpuOutput(BaseModel):
    result: str


def _build(lazy_load: bool = False, **kwargs: object) -> Thalamus:
    app = Thalamus(name="gpu-app", lazy_load=lazy_load)

    @app.model(
        model_id="gpu-model",
        default=True,
        default_version=True,
        input_type=GpuInput,
        output_type=GpuOutput,
        **kwargs,  # type: ignore[arg-type]
    )
    class GpuModel:
        def load(self, weights: dict[str, Path], device: str) -> None:
            self.device = device

        def predict(self, inputs: list[GpuInput]) -> list[GpuOutput]:
            return [GpuOutput(result=i.data) for i in inputs]

    return app


@pytest.fixture(autouse=True)
def _reset_allocator() -> None:
    GPUAllocator.reset()


class TestRequireGpuMetadata:
    def test_defaults_to_false(self) -> None:
        spec = _build()._registry.get("gpu-model")
        assert spec is not None
        assert spec.require_gpu is False

    def test_stored_on_spec(self) -> None:
        spec = _build(require_gpu=True)._registry.get("gpu-model")
        assert spec is not None
        assert spec.require_gpu is True

    def test_rejects_cpu_device_at_import_time(self) -> None:
        with pytest.raises(ValueError, match="require_gpu"):
            _build(require_gpu=True, device="cpu")


class TestStartupEnforcement:
    def test_startup_fails_without_gpu(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            gpu, "gpu_preference_error", lambda _p: "no CUDA or MPS device was detected"
        )
        app = _build(require_gpu=True, device="cuda")
        with pytest.raises(GPURequirementError, match="requires a GPU"):
            with TestClient(app):
                pass

    def test_startup_fails_for_lazy_models_too(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            gpu, "gpu_preference_error", lambda _p: "no CUDA or MPS device was detected"
        )
        app = _build(lazy_load=True, require_gpu=True, device="cuda")
        with pytest.raises(GPURequirementError):
            with TestClient(app):
                pass

    def test_startup_succeeds_when_gpu_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gpu, "gpu_preference_error", lambda _p: None)
        monkeypatch.setattr(gpu, "is_accelerator", lambda _d: True)
        app = _build(require_gpu=True, device="cuda")
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200

    def test_startup_succeeds_without_require_gpu(self) -> None:
        app = _build(device="cuda")
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200


class TestLoadEnforcement:
    def test_load_fails_when_allocation_is_not_an_accelerator(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Preflight passes but allocation lands on CPU (e.g. a GPU vanished)."""
        monkeypatch.setattr(gpu, "gpu_preference_error", lambda _p: None)
        monkeypatch.setattr(gpu, "is_accelerator", lambda _d: False)
        app = _build(require_gpu=True, device="cuda")
        with pytest.raises(GPURequirementError, match="was allocated"):
            with TestClient(app):
                pass

    def test_failed_requirement_does_not_leak_allocation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gpu, "gpu_preference_error", lambda _p: None)
        monkeypatch.setattr(gpu, "is_accelerator", lambda _d: False)
        monkeypatch.setattr(GPUAllocator, "allocate", lambda self, _p: "cuda:0")
        allocator = GPUAllocator.get()
        allocator._allocations = {0: 0}

        app = _build(require_gpu=True, device="cuda:0")
        with pytest.raises(GPURequirementError):
            with TestClient(app):
                pass

        assert allocator.get_allocations()[0] == 0


class TestIsAccelerator:
    def test_cpu_is_not_an_accelerator(self) -> None:
        assert gpu.is_accelerator("cpu") is False

    def test_no_torch_means_no_accelerator(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gpu, "_get_torch", lambda: None)
        assert gpu.is_accelerator("cuda:0") is False

    def test_unknown_device_string_is_not_an_accelerator(self) -> None:
        assert gpu.is_accelerator("tpu") is False


class TestGpuPreferenceError:
    def test_cpu_preference_is_rejected(self) -> None:
        assert gpu.gpu_preference_error("cpu") is not None

    def test_missing_torch_is_reported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gpu, "_get_torch", lambda: None)
        assert gpu.gpu_preference_error("auto") == "PyTorch is not installed"

    def test_explicit_device_checked_against_host(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gpu, "_get_torch", lambda: object())
        monkeypatch.setattr(gpu, "is_accelerator", lambda _d: False)
        assert gpu.gpu_preference_error("cuda:7") == (
            "cuda:7 is not available on this host"
        )
