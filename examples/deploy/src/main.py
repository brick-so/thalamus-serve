"""Sentiment analysis service using DistilBERT from HuggingFace Hub."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from thalamus_serve import HFWeight, Thalamus


class TextInput(BaseModel):
    """Input schema for text classification."""

    text: str = Field(..., description="Text to analyze for sentiment")


class SentimentOutput(BaseModel):
    """Output schema with sentiment prediction."""

    label: str = Field(..., description="Sentiment label (POSITIVE/NEGATIVE)")
    score: float = Field(..., description="Confidence score")


app = Thalamus(name="sentiment-service")


@app.model(
    model_id="sentiment",
    version="1.0.0",
    description="DistilBERT sentiment analysis model",
    default=True,
    input_type=TextInput,
    output_type=SentimentOutput,
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

        model_path = str(weights["model"])
        self.pipeline = pipeline(
            "sentiment-analysis",
            model=model_path,
            device=device if device != "cpu" else -1,
        )

    def predict(self, inputs: list[TextInput]) -> list[SentimentOutput]:
        """Run sentiment analysis on input texts."""
        texts = [inp.text for inp in inputs]
        results = self.pipeline(texts)
        return [SentimentOutput(label=r["label"], score=r["score"]) for r in results]
