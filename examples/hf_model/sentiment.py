"""HuggingFace Transformers sentiment analysis example.

Demonstrates using HFWeight to download model weights from HuggingFace Hub.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from thalamus_serve import HFWeight, Thalamus


class TextInput(BaseModel):
    """Input schema for text classification."""

    text: str


class SentimentOutput(BaseModel):
    """Output schema with sentiment prediction."""

    label: str
    score: float


app = Thalamus()


@app.model(
    model_id="sentiment",
    version="1.0.0",
    description="DistilBERT sentiment analysis",
    default=True,
    weights={
        "model": HFWeight(repo="distilbert-base-uncased-finetuned-sst-2-english"),
    },
)
class SentimentAnalyzer:
    """Sentiment analyzer using HuggingFace transformers pipeline."""

    def __init__(self) -> None:
        self.pipeline: Any = None

    def load(self, weights: dict[str, Path], device: str) -> None:
        """Load the sentiment analysis pipeline from downloaded weights."""
        from transformers import pipeline

        device_id = 0 if device.startswith("cuda") else -1
        # Use the downloaded model path from weights dict
        model_path = str(weights["model"])
        self.pipeline = pipeline(
            "sentiment-analysis",
            model=model_path,
            device=device_id,
        )

    def predict(self, inputs: list[TextInput]) -> list[SentimentOutput]:
        """Run sentiment analysis on input texts."""
        texts = [inp.text for inp in inputs]
        results = self.pipeline(texts)
        return [SentimentOutput(label=r["label"], score=r["score"]) for r in results]


if __name__ == "__main__":
    app.serve()
