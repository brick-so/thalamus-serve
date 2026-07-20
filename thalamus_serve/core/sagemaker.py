"""SageMaker BYOC serving: the two routes an AWS SageMaker container must expose.

SageMaker hosting requires ``GET /ping`` and ``POST /invocations`` on port 8080,
a contract that differs from thalamus-serve's ``/predict``: one input in, one
output out, and no authentication — SageMaker sends no credentials, so this app
deliberately does not mount :class:`APIKeyAuth`.
"""

import asyncio
import time
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ValidationError

from thalamus_serve.core.model import ModelSpec
from thalamus_serve.observability.logging import log
from thalamus_serve.schemas.api import PredictMeta, PredictResponse

Envelope = Literal["bare", "predict_response"]


def _run_pipeline(spec: ModelSpec, parsed: BaseModel) -> list[Any]:
    inputs: list[Any] = [parsed]
    if spec.has_preprocess:
        inputs = spec.instance.preprocess(inputs)

    outputs: list[Any] = spec.instance.predict(inputs)

    if spec.has_postprocess:
        outputs = spec.instance.postprocess(outputs)

    return outputs


def build_sagemaker_app(spec: ModelSpec, envelope: Envelope = "bare") -> FastAPI:
    """Build the SageMaker serving app around an already-loaded model.

    Loading is deliberately not performed here so the app can be tested with a
    stub instance, without weights, GPU or network.

    Args:
        spec: The registered model to serve. Its ``instance`` must be set.
        envelope: ``"bare"`` returns the output object itself, matching what
            SageMaker's ``InvokeEndpoint`` callers expect. ``"predict_response"``
            wraps it in thalamus-serve's ``PredictResponse``.
    """
    app = FastAPI(title=f"{spec.id}-sagemaker")

    @app.get("/ping")
    def ping() -> Response:
        ready = getattr(spec.instance, "is_ready", True) if spec.instance else False
        return Response(status_code=200 if ready else 503)

    @app.post("/invocations")
    async def invocations(request: Request) -> JSONResponse:
        try:
            parsed = spec.input_type.model_validate(await request.json())
        except ValidationError as error:
            return JSONResponse(status_code=400, content={"error": error.errors()})
        except ValueError as error:
            return JSONResponse(status_code=400, content={"error": str(error)})

        start = time.perf_counter()
        try:
            outputs = await asyncio.to_thread(_run_pipeline, spec, parsed)
            if not outputs:
                raise ValueError(
                    f"{spec.id} returned no outputs for a single input"
                )
        except Exception:
            log.exception("invocations_error", model=spec.id, version=spec.version)
            return JSONResponse(
                status_code=500, content={"error": "Internal Server Error"}
            )

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        log.info(
            "invocations", model=spec.id, version=spec.version, ms=latency_ms
        )

        if envelope == "predict_response":
            return JSONResponse(
                content=PredictResponse(
                    outputs=[o.model_dump(mode="json") for o in outputs],
                    meta=PredictMeta(
                        model=spec.id,
                        version=spec.version,
                        latency_ms=latency_ms,
                        batch_size=1,
                        device=spec.device,
                    ),
                ).model_dump(mode="json")
            )

        return JSONResponse(content=outputs[0].model_dump(mode="json"))

    return app
