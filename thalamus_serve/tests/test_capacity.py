import pytest
from pydantic import BaseModel

from thalamus_serve import CapacityResponse, ModelCapacity
from thalamus_serve.core.app import Thalamus


class CapInput(BaseModel):
    data: str


class CapOutput(BaseModel):
    result: str


class TestCapacitySchemas:
    def test_model_capacity_defaults(self) -> None:
        cap = ModelCapacity(remaining_requests=2, ideal_batch_size=8, max_batch_size=16)
        assert cap.accepting is True
        assert cap.reason is None

    def test_capacity_response_shape(self) -> None:
        response = CapacityResponse(
            accepting=True,
            remaining_requests=2,
            models={
                "a@1.0.0": ModelCapacity(
                    remaining_requests=2, ideal_batch_size=8, max_batch_size=16
                )
            },
            inflight_requests=0,
            uptime_seconds=1.5,
        )
        assert response.models["a@1.0.0"].max_batch_size == 16


def _register(**kwargs: object) -> Thalamus:
    app = Thalamus(name="cap-spec-app", lazy_load=True)

    @app.model(
        model_id="spec",
        input_type=CapInput,
        output_type=CapOutput,
        **kwargs,  # type: ignore[arg-type]
    )
    class SpecModel:
        def predict(self, inputs: list[CapInput]) -> list[CapOutput]:
            return [CapOutput(result=i.data) for i in inputs]

    return app


class TestModelSpecCapacityMetadata:
    def test_defaults_are_conservative(self) -> None:
        spec = _register()._registry.get("spec")
        assert spec is not None
        assert spec.max_batch_size == 1
        assert spec.ideal_batch_size == 1
        assert spec.max_concurrent_requests == 1

    def test_ideal_defaults_to_max(self) -> None:
        spec = _register(max_batch_size=32)._registry.get("spec")
        assert spec is not None
        assert spec.ideal_batch_size == 32

    def test_explicit_values_are_stored(self) -> None:
        spec = _register(
            max_batch_size=32, ideal_batch_size=16, max_concurrent_requests=2
        )._registry.get("spec")
        assert spec is not None
        assert spec.max_batch_size == 32
        assert spec.ideal_batch_size == 16
        assert spec.max_concurrent_requests == 2

    def test_has_capacity_false_without_hook(self) -> None:
        spec = _register()._registry.get("spec")
        assert spec is not None
        assert spec.has_capacity is False

    def test_rejects_zero_max_batch_size(self) -> None:
        with pytest.raises(ValueError, match="max_batch_size"):
            _register(max_batch_size=0)

    def test_rejects_zero_max_concurrent_requests(self) -> None:
        with pytest.raises(ValueError, match="max_concurrent_requests"):
            _register(max_concurrent_requests=0)

    def test_rejects_ideal_above_max(self) -> None:
        with pytest.raises(ValueError, match="ideal_batch_size"):
            _register(max_batch_size=4, ideal_batch_size=8)

    def test_rejects_ideal_below_one(self) -> None:
        with pytest.raises(ValueError, match="ideal_batch_size"):
            _register(max_batch_size=4, ideal_batch_size=0)
