"""
Medical Cost Prediction model using scikit-learn Linear Regression.

Predicts insurance charges based on patient demographics and health factors.
Trained on the Medical Cost Personal Dataset.
"""

from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from thalamus_serve import HTTPWeight, Thalamus


class MedicalCostInput(BaseModel):
    """Input schema for medical cost prediction."""

    age: int = Field(..., ge=0, le=120, description="Age of the insured")
    sex: str = Field(..., pattern="^(male|female)$", description="Gender")
    bmi: float = Field(..., ge=10, le=60, description="Body mass index")
    children: int = Field(..., ge=0, le=10, description="Number of dependents")
    smoker: str = Field(..., pattern="^(yes|no)$", description="Smoking status")
    region: str = Field(
        ...,
        pattern="^(northeast|northwest|southeast|southwest)$",
        description="US region",
    )


class MedicalCostOutput(BaseModel):
    """Output schema with predicted charges and feature contributions."""

    predicted_charges: float = Field(..., description="Predicted insurance charges")
    feature_contributions: dict[str, float] = Field(
        ..., description="Contribution of each feature to the prediction"
    )


app = Thalamus()


@app.model(
    model_id="medical-cost",
    version="1.0.0",
    description="Predicts medical insurance charges based on patient demographics",
    default=True,
    weights={
        "model": HTTPWeight(
            urls=[
                "https://drive.google.com/uc?export=download&id=1Fef6DEoN_JnowD-JLTDOwoF8NqCc8q3a"
            ]
        ),
        "preprocessor": HTTPWeight(
            urls=[
                "https://drive.google.com/uc?export=download&id=1VNQl7V7oy4iqto7HG7dD8IJyQqhp9lDE"
            ]
        ),
    },
    input_type=MedicalCostInput,
    output_type=MedicalCostOutput,
)
class MedicalCostPredictor:
    """Medical cost prediction model using Linear Regression.

    Demonstrates loading multiple weight files:
    - model.joblib: Contains the trained LinearRegression model and metrics
    - preprocessor.joblib: Contains feature columns and label encoders
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._feature_columns: list[str] = []
        self._encoders: dict = {}

    def load(self, weights: dict[str, Path], device: str) -> None:  # noqa: ARG002
        """Load model and preprocessor from separate weight files."""
        import joblib

        # Load model weights
        model_data = joblib.load(weights["model"])
        self._model = model_data["model"]

        # Load preprocessor (encoders and feature columns)
        preprocessor_data = joblib.load(weights["preprocessor"])
        self._feature_columns = preprocessor_data["feature_columns"]
        self._encoders = preprocessor_data["encoders"]

    def _encode_input(self, inp: MedicalCostInput) -> list[float]:
        """Encode a single input to feature vector using saved encoders."""
        return [
            float(inp.age),
            float(self._encoders["sex"].transform([inp.sex])[0]),
            float(inp.bmi),
            float(inp.children),
            float(self._encoders["smoker"].transform([inp.smoker])[0]),
            float(self._encoders["region"].transform([inp.region])[0]),
        ]

    def preprocess(self, inputs: list[MedicalCostInput]) -> np.ndarray:
        """Convert inputs to numpy array with proper encoding."""
        features_list = [self._encode_input(inp) for inp in inputs]
        return np.array(features_list)

    def predict(self, inputs: np.ndarray) -> list[MedicalCostOutput]:
        """Run inference and return predictions with feature contributions."""
        predictions = self._model.predict(inputs)
        outputs = []

        for i, pred in enumerate(predictions):
            # Calculate feature contributions (coefficient * feature value)
            contributions = self._model.coef_ * inputs[i]
            contribution_dict = {
                name: round(float(contrib), 2)
                for name, contrib in zip(
                    self._feature_columns, contributions, strict=True
                )
            }

            outputs.append(
                MedicalCostOutput(
                    predicted_charges=round(float(pred), 2),
                    feature_contributions=contribution_dict,
                )
            )

        return outputs


if __name__ == "__main__":
    app.serve()
