"""scikit-learn Iris classifier example."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from thalamus_serve import Thalamus


class IrisInput(BaseModel):
    """Input schema for Iris classification."""

    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class IrisOutput(BaseModel):
    """Output schema with species prediction and probabilities."""

    species: str
    probabilities: dict[str, float]


SPECIES = ["setosa", "versicolor", "virginica"]

app = Thalamus()


@app.model(
    model_id="iris",
    version="1.0.0",
    description="Iris species classifier",
    default=True,
)
class IrisClassifier:
    """Random Forest classifier for Iris species prediction."""

    def __init__(self) -> None:
        self.model: Any = None

    def load(self, weights: dict[str, Path], device: str) -> None:
        """Load or train the classifier."""
        if "model" in weights:
            import joblib

            self.model = joblib.load(weights["model"])
        else:
            # Train a simple model if no weights provided
            from sklearn.datasets import load_iris
            from sklearn.ensemble import RandomForestClassifier

            iris = load_iris()
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(iris.data, iris.target)

    def predict(self, inputs: list[IrisInput]) -> list[IrisOutput]:
        """Predict Iris species from features."""
        features = [
            [inp.sepal_length, inp.sepal_width, inp.petal_length, inp.petal_width]
            for inp in inputs
        ]

        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)

        results = []
        for pred, probs in zip(predictions, probabilities, strict=True):
            results.append(
                IrisOutput(
                    species=SPECIES[pred],
                    probabilities={
                        species: float(prob)
                        for species, prob in zip(SPECIES, probs, strict=True)
                    },
                )
            )

        return results


if __name__ == "__main__":
    app.serve()
