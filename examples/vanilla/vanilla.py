import os
from pathlib import Path

import torch
from pydantic import BaseModel

from thalamus_serve.core.app import Thalamus


class Input(BaseModel):
    text: str


class Output(BaseModel):
    result: str
    score: float


app = Thalamus(lazy_load=True)
os.environ["THALAMUS_API_KEY"] = "test"


@app.model(
    model_id="default",
    version="1.0.0",
    description="Default placeholder model for demonstration",
    default=True,
    default_version=True,
    critical=True,
    required_weights=[],
    optional_weights=[],
    device="mps",
    input_type=Input,
    output_type=Output,
)
class TestModel:
    def load(self, _weights: dict[str, Path], _device: str) -> None:
        device = torch.device(_device)
        self.x = torch.randn((16384, 16384), device=device)
        self.y = torch.randn((16384, 16384), device=device)

    @property
    def is_ready(self) -> bool:
        return True

    def preprocess(self, inputs: list[Input]) -> list[str]:
        outputs = []
        for input in inputs:
            outputs.append(input.text.upper())
        return outputs

    def predict(self, inputs: list[str]) -> list[str]:
        return [{"score": len(result), "text": result} for result in inputs]

    def postprocess(self, outputs: list[str]) -> list[Output]:
        return [
            Output(result=output["text"], score=output["score"]) for output in outputs
        ]


if __name__ == "__main__":
    app.serve()
