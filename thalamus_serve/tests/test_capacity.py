from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from thalamus_serve import CapacityResponse, ModelCapacity
from thalamus_serve.core.app import Thalamus
from thalamus_serve.core.model import ModelRegistry
from thalamus_serve.core.routes import RouteContext
from thalamus_serve.testing import TEST_API_KEY, TEST_API_KEY_HEADER


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


def _bare_context() -> RouteContext:
    return RouteContext(
        registry=ModelRegistry(),
        ensure_loaded=lambda _spec: None,
        get_uptime=lambda: 0.0,
    )


class TestInflightTracking:
    def test_starts_at_zero(self) -> None:
        assert _bare_context().inflight() == 0

    def test_increments_inside_and_restores_after(self) -> None:
        ctx = _bare_context()
        with ctx.track_inflight():
            assert ctx.inflight() == 1
        assert ctx.inflight() == 0

    def test_nests(self) -> None:
        ctx = _bare_context()
        with ctx.track_inflight(), ctx.track_inflight():
            assert ctx.inflight() == 2
        assert ctx.inflight() == 0

    def test_decrements_when_body_raises(self) -> None:
        ctx = _bare_context()
        with pytest.raises(RuntimeError), ctx.track_inflight():
            raise RuntimeError("boom")
        assert ctx.inflight() == 0


capacity_app = Thalamus(name="capacity-app", lazy_load=True)


@capacity_app.model(
    model_id="static",
    version="1.0.0",
    default=True,
    default_version=True,
    input_type=CapInput,
    output_type=CapOutput,
    max_batch_size=32,
    ideal_batch_size=16,
    max_concurrent_requests=2,
)
class StaticModel:
    @property
    def is_ready(self) -> bool:
        return True

    def predict(self, inputs: list[CapInput]) -> list[CapOutput]:
        return [CapOutput(result=i.data) for i in inputs]


@capacity_app.model(
    model_id="dynamic",
    version="1.0.0",
    critical=False,
    input_type=CapInput,
    output_type=CapOutput,
    max_batch_size=32,
    max_concurrent_requests=8,
)
class DynamicModel:
    @property
    def is_ready(self) -> bool:
        return True

    def capacity(self) -> ModelCapacity:
        return ModelCapacity(
            accepting=True,
            remaining_requests=5,
            ideal_batch_size=4,
            max_batch_size=8,
            reason="vram_headroom_low",
        )

    def predict(self, inputs: list[CapInput]) -> list[CapOutput]:
        return [CapOutput(result=i.data) for i in inputs]


@capacity_app.model(
    model_id="broken",
    version="1.0.0",
    critical=False,
    input_type=CapInput,
    output_type=CapOutput,
    max_batch_size=4,
    max_concurrent_requests=4,
)
class BrokenHookModel:
    @property
    def is_ready(self) -> bool:
        return True

    def capacity(self) -> ModelCapacity:
        raise RuntimeError("gauge unavailable")

    def predict(self, inputs: list[CapInput]) -> list[CapOutput]:
        return [CapOutput(result=i.data) for i in inputs]


@capacity_app.model(
    model_id="unloaded",
    version="1.0.0",
    critical=False,
    input_type=CapInput,
    output_type=CapOutput,
    max_batch_size=8,
    max_concurrent_requests=4,
)
class UnloadedModel:
    def predict(self, inputs: list[CapInput]) -> list[CapOutput]:
        return [CapOutput(result=i.data) for i in inputs]


@pytest.fixture
def capacity_client(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[TestClient, None, None]:
    monkeypatch.setenv("THALAMUS_API_KEY", TEST_API_KEY)
    with TestClient(capacity_app) as c:
        for spec in capacity_app._registry.all():
            if spec.id != "unloaded":
                capacity_app._ensure_loaded(spec)
        yield c


def _get_capacity(client: TestClient) -> CapacityResponse:
    r = client.get("/capacity", headers=TEST_API_KEY_HEADER)
    assert r.status_code == 200
    return CapacityResponse.model_validate(r.json())


class TestCapacityEndpoint:
    def test_requires_auth(self, capacity_client: TestClient) -> None:
        assert capacity_client.get("/capacity").status_code == 401

    def test_static_model_reports_declared_numbers(
        self, capacity_client: TestClient
    ) -> None:
        cap = _get_capacity(capacity_client).models["static@1.0.0"]
        assert cap.accepting is True
        assert cap.remaining_requests == 2
        assert cap.ideal_batch_size == 16
        assert cap.max_batch_size == 32
        assert cap.reason is None

    def test_dynamic_hook_wins_over_static(self, capacity_client: TestClient) -> None:
        cap = _get_capacity(capacity_client).models["dynamic@1.0.0"]
        assert cap.remaining_requests == 5
        assert cap.ideal_batch_size == 4
        assert cap.max_batch_size == 8
        assert cap.reason == "vram_headroom_low"

    def test_raising_hook_is_not_accepting(self, capacity_client: TestClient) -> None:
        cap = _get_capacity(capacity_client).models["broken@1.0.0"]
        assert cap.accepting is False
        assert cap.remaining_requests == 0
        assert cap.reason == "capacity_hook_error"

    def test_unloaded_model_reports_not_loaded(
        self, capacity_client: TestClient
    ) -> None:
        cap = _get_capacity(capacity_client).models["unloaded@1.0.0"]
        assert cap.accepting is False
        assert cap.remaining_requests == 0
        assert cap.reason == "model_not_loaded"

    def test_unloaded_non_critical_model_does_not_flip_accepting(
        self, capacity_client: TestClient
    ) -> None:
        assert _get_capacity(capacity_client).accepting is True

    def test_does_not_load_lazy_model(self, capacity_client: TestClient) -> None:
        _get_capacity(capacity_client)
        spec = capacity_app._registry.get("unloaded", "1.0.0")
        assert spec is not None
        assert spec.instance is None

    def test_remaining_requests_is_bottleneck(
        self, capacity_client: TestClient
    ) -> None:
        assert _get_capacity(capacity_client).remaining_requests == 2

    def test_reports_inflight_count(self, capacity_client: TestClient) -> None:
        assert _get_capacity(capacity_client).inflight_requests == 0

    def test_at_capacity_when_inflight_exhausts_slots(
        self, capacity_client: TestClient
    ) -> None:
        ctx = capacity_app._route_context
        assert ctx is not None
        with ctx.track_inflight(), ctx.track_inflight():
            body = _get_capacity(capacity_client)
        cap = body.models["static@1.0.0"]
        assert body.inflight_requests == 2
        assert cap.accepting is False
        assert cap.remaining_requests == 0
        assert cap.reason == "at_capacity"

    def test_top_level_not_accepting_when_critical_at_capacity(
        self, capacity_client: TestClient
    ) -> None:
        ctx = capacity_app._route_context
        assert ctx is not None
        with ctx.track_inflight(), ctx.track_inflight():
            assert _get_capacity(capacity_client).accepting is False

    def test_remaining_requests_is_zero_when_not_accepting(
        self, capacity_client: TestClient
    ) -> None:
        ctx = capacity_app._route_context
        assert ctx is not None
        with ctx.track_inflight(), ctx.track_inflight():
            body = _get_capacity(capacity_client)
        assert body.accepting is False
        assert body.remaining_requests == 0
        assert body.models["dynamic@1.0.0"].remaining_requests == 5

    def test_uptime_is_reported(self, capacity_client: TestClient) -> None:
        assert _get_capacity(capacity_client).uptime_seconds >= 0
