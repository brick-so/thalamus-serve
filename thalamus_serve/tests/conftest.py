from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from thalamus_serve.core.app import Thalamus
from thalamus_serve.testing import TEST_API_KEY


class TestInput(BaseModel):
    data: str


class TestOutput(BaseModel):
    result: str


# Create a test app with a default model
app = Thalamus(name="test-app", lazy_load=False)


@app.model(
    model_id="default",
    version="1.0.0",
    description="Test model for unit tests",
    default=True,
    default_version=True,
    critical=True,
    input_type=TestInput,
    output_type=TestOutput,
)
class TestModel:
    def load(self, weights: dict[str, Path], device: str) -> None:
        pass

    @property
    def is_ready(self) -> bool:
        return True

    def preprocess(self, inputs: list[TestInput]) -> list[str]:
        return [inp.data.upper() for inp in inputs]

    def predict(self, inputs: list[str]) -> list[dict[str, Any]]:
        return [{"result": text} for text in inputs]

    def postprocess(self, outputs: list[dict[str, Any]]) -> list[TestOutput]:
        return [TestOutput(result=o["result"]) for o in outputs]


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    monkeypatch.setenv("THALAMUS_API_KEY", TEST_API_KEY)
    with TestClient(app) as c:
        yield c
