import asyncio
import threading
import time

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel

from thalamus_serve.core.app import Thalamus
from thalamus_serve.core.model import ModelSpec
from thalamus_serve.core.sagemaker import Envelope, build_sagemaker_app

BLOCK_SECONDS = 10.0


class SmInput(BaseModel):
    value: int


class SmOutput(BaseModel):
    doubled: int


class StubModel:
    """Mirrors the real model interface: is_ready is a property, not a method."""

    def __init__(self, ready: bool = True) -> None:
        self._ready = ready
        self.predict_thread: int | None = None

    @property
    def is_ready(self) -> bool:
        return self._ready

    def predict(self, inputs: list[SmInput]) -> list[SmOutput]:
        self.predict_thread = threading.get_ident()
        return [SmOutput(doubled=i.value * 2) for i in inputs]


class EmptyModel:
    @property
    def is_ready(self) -> bool:
        return True

    def predict(self, inputs: list[SmInput]) -> list[SmOutput]:
        return []


def _spec(instance: object, **kwargs: object) -> ModelSpec:
    spec = ModelSpec(
        model_id="stub",
        version="1.0.0",
        description="",
        cls=StubModel,
        input_type=SmInput,
        output_type=SmOutput,
        **kwargs,  # type: ignore[arg-type]
    )
    spec.instance = instance
    return spec


def _client(
    instance: object, envelope: Envelope = "bare", **kwargs: object
) -> TestClient:
    return TestClient(build_sagemaker_app(_spec(instance, **kwargs), envelope=envelope))


class TestPing:
    def test_200_when_ready(self) -> None:
        assert _client(StubModel(ready=True)).get("/ping").status_code == 200

    def test_503_when_not_ready(self) -> None:
        assert _client(StubModel(ready=False)).get("/ping").status_code == 503

    def test_200_when_model_declares_no_is_ready(self) -> None:
        class Bare:
            def predict(self, inputs: list[SmInput]) -> list[SmOutput]:
                return []

        assert _client(Bare()).get("/ping").status_code == 200

    def test_is_ready_is_read_as_a_property_not_called(self) -> None:
        """Guards the `is_ready()` regression: calling the property raises
        TypeError: 'bool' object is not callable, and SageMaker never marks the
        container healthy."""
        calls: list[str] = []

        class PropertyOnly:
            @property
            def is_ready(self) -> bool:
                calls.append("read")
                return True

            def predict(self, inputs: list[SmInput]) -> list[SmOutput]:
                return []

        assert _client(PropertyOnly()).get("/ping").status_code == 200
        assert calls == ["read"]


class TestInvocations:
    def test_returns_bare_output_not_an_envelope(self) -> None:
        r = _client(StubModel()).post("/invocations", json={"value": 21})
        assert r.status_code == 200
        assert r.json() == {"doubled": 42}

    def test_requires_no_api_key(self) -> None:
        """SageMaker sends no credentials; the app must not mount APIKeyAuth."""
        r = _client(StubModel()).post("/invocations", json={"value": 1})
        assert r.status_code == 200

    def test_predict_response_envelope(self) -> None:
        r = _client(StubModel(), envelope="predict_response").post(
            "/invocations", json={"value": 21}
        )
        assert r.status_code == 200
        body = r.json()
        assert body["outputs"] == [{"doubled": 42}]
        assert body["meta"]["model"] == "stub"
        assert body["meta"]["batch_size"] == 1

    def test_schema_invalid_body_is_400(self) -> None:
        r = _client(StubModel()).post("/invocations", json={"value": "not-an-int"})
        assert r.status_code == 400

    def test_malformed_json_is_400(self) -> None:
        r = _client(StubModel()).post(
            "/invocations",
            content=b"{not json",
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 400

    def test_model_error_is_500_without_leaking_internals(self) -> None:
        class Exploding:
            @property
            def is_ready(self) -> bool:
                return True

            def predict(self, inputs: list[SmInput]) -> list[SmOutput]:
                raise RuntimeError("cuda oom at layer 7")

        r = _client(Exploding()).post("/invocations", json={"value": 1})
        assert r.status_code == 500
        assert "cuda oom" not in r.text

    def test_empty_output_is_a_controlled_500(self) -> None:
        """A single-in/single-out contract violation must not surface as a raw
        Starlette 500, nor as a silent 200 with an empty list."""
        for envelope in ("bare", "predict_response"):
            client = TestClient(
                build_sagemaker_app(_spec(EmptyModel()), envelope=envelope),
                raise_server_exceptions=False,
            )
            r = client.post("/invocations", json={"value": 1})
            assert r.status_code == 500, envelope
            assert r.json() == {"error": "Internal Server Error"}, envelope

    def test_runs_preprocess_and_postprocess_hooks(self) -> None:
        class Hooked:
            @property
            def is_ready(self) -> bool:
                return True

            def preprocess(self, inputs: list[SmInput]) -> list[SmInput]:
                return [SmInput(value=i.value + 1) for i in inputs]

            def predict(self, inputs: list[SmInput]) -> list[SmOutput]:
                return [SmOutput(doubled=i.value * 2) for i in inputs]

            def postprocess(self, outputs: list[SmOutput]) -> list[SmOutput]:
                return [SmOutput(doubled=o.doubled + 100) for o in outputs]

        spec = _spec(Hooked())
        spec.has_preprocess = True
        spec.has_postprocess = True
        r = TestClient(build_sagemaker_app(spec, envelope="bare")).post(
            "/invocations", json={"value": 10}
        )
        assert r.json() == {"doubled": 122}


@pytest.mark.anyio
async def test_ping_is_answered_while_predict_is_in_flight() -> None:
    """Guards the event-loop regression: predict must be offloaded to a worker
    thread, or SageMaker's periodic health check cannot be answered during a
    long inference and the container is restarted mid-prediction."""
    release = threading.Event()
    started = threading.Event()

    class Slow:
        @property
        def is_ready(self) -> bool:
            return True

        def predict(self, inputs: list[SmInput]) -> list[SmOutput]:
            started.set()
            release.wait(timeout=BLOCK_SECONDS)
            return [SmOutput(doubled=0)]

    app = build_sagemaker_app(_spec(Slow()), envelope="bare")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://sm") as client:
        start = time.perf_counter()
        inflight = asyncio.create_task(client.post("/invocations", json={"value": 1}))
        await asyncio.sleep(0.05)
        ping = await client.get("/ping")
        elapsed = time.perf_counter() - start

        release.set()
        assert (await inflight).status_code == 200

        assert ping.status_code == 200
        assert started.is_set()
        # A blocking predict would freeze the loop for the full BLOCK_SECONDS
        # before /ping could be served at all.
        assert elapsed < BLOCK_SECONDS / 2


class TestSagemakerAppFactory:
    def test_uses_the_registered_default_model(self) -> None:
        app = Thalamus(name="sm", lazy_load=True)

        @app.model(
            model_id="primary", default=True, input_type=SmInput, output_type=SmOutput
        )
        class Primary:
            @property
            def is_ready(self) -> bool:
                return True

            def predict(self, inputs: list[SmInput]) -> list[SmOutput]:
                return [SmOutput(doubled=i.value * 2) for i in inputs]

        @app.model(
            model_id="secondary", input_type=SmInput, output_type=SmOutput
        )
        class Secondary:
            def predict(self, inputs: list[SmInput]) -> list[SmOutput]:
                raise AssertionError("the non-default model must not be served")

        r = TestClient(app.sagemaker_app()).post("/invocations", json={"value": 3})
        assert r.json() == {"doubled": 6}

    def test_loads_only_the_served_model(self) -> None:
        app = Thalamus(name="sm", lazy_load=True)

        @app.model(
            model_id="primary", default=True, input_type=SmInput, output_type=SmOutput
        )
        class Primary:
            def predict(self, inputs: list[SmInput]) -> list[SmOutput]:
                return []

        @app.model(
            model_id="secondary", input_type=SmInput, output_type=SmOutput
        )
        class Secondary:
            def predict(self, inputs: list[SmInput]) -> list[SmOutput]:
                return []

        app.sagemaker_app()
        secondary = app._registry.get("secondary")
        assert secondary is not None
        assert secondary.instance is None

    def test_explicit_model_id(self) -> None:
        app = Thalamus(name="sm", lazy_load=True)

        @app.model(
            model_id="primary", default=True, input_type=SmInput, output_type=SmOutput
        )
        class Primary:
            def predict(self, inputs: list[SmInput]) -> list[SmOutput]:
                return [SmOutput(doubled=0) for _ in inputs]

        @app.model(
            model_id="secondary", input_type=SmInput, output_type=SmOutput
        )
        class Secondary:
            def predict(self, inputs: list[SmInput]) -> list[SmOutput]:
                return [SmOutput(doubled=99) for _ in inputs]

        r = TestClient(app.sagemaker_app(model_id="secondary")).post(
            "/invocations", json={"value": 1}
        )
        assert r.json() == {"doubled": 99}

    def test_raises_when_no_default_model(self) -> None:
        app = Thalamus(name="sm", lazy_load=True)

        @app.model(model_id="only", input_type=SmInput, output_type=SmOutput)
        class Only:
            def predict(self, inputs: list[SmInput]) -> list[SmOutput]:
                return []

        with pytest.raises(ValueError, match="no default model"):
            app.sagemaker_app()

    def test_raises_when_model_id_unknown(self) -> None:
        app = Thalamus(name="sm", lazy_load=True)

        @app.model(
            model_id="only", default=True, input_type=SmInput, output_type=SmOutput
        )
        class Only:
            def predict(self, inputs: list[SmInput]) -> list[SmOutput]:
                return []

        with pytest.raises(ValueError, match="nope"):
            app.sagemaker_app(model_id="nope")
