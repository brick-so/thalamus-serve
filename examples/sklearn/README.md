# scikit-learn Example

This example demonstrates serving a scikit-learn model with multiple weight files.

## Files

- `medical_cost.py` - Thalamus model serving medical cost predictions
- `train_medical_cost.py` - Training script that saves model and preprocessor separately

## Setup

```bash
pip install scikit-learn joblib pandas
```

## Training (Optional)

To train the model and save weights locally:

```bash
python train_medical_cost.py --output-dir ./weights
```

This creates two separate weight files:
- `weights/model.joblib` - Linear regression model and metrics
- `weights/preprocessor.joblib` - Feature columns and label encoders

## Running the Server

```bash
export THALAMUS_API_KEY=your-key
python medical_cost.py
```

## Testing

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{
    "inputs": [{
      "age": 35,
      "sex": "male",
      "bmi": 25.5,
      "children": 2,
      "smoker": "no",
      "region": "northeast"
    }]
  }'
```

## Model Details

- **Model**: Linear Regression trained on Medical Cost Personal Dataset
- **Input**: Age, sex, BMI, children, smoker status, US region
- **Output**: Predicted insurance charges with feature contributions

## Multi-Weight Loading

This example demonstrates loading multiple weight files:

```python
from pathlib import Path
from pydantic import BaseModel
from thalamus_serve import Thalamus, S3Weight

app = Thalamus()

class MedicalCostInput(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str

class MedicalCostOutput(BaseModel):
    predicted_charges: float

@app.model(
    model_id="medical-cost",
    input_type=MedicalCostInput,
    output_type=MedicalCostOutput,
    weights={
        "model": S3Weight(bucket="my-models", key="medical_cost/model.joblib"),
        "preprocessor": S3Weight(bucket="my-models", key="medical_cost/preprocessor.joblib"),
    },
)
class MedicalCostPredictor:
    def load(self, weights: dict[str, Path], device: str) -> None:
        import joblib

        # Load from separate files
        model_data = joblib.load(weights["model"])
        preprocessor_data = joblib.load(weights["preprocessor"])

        self._model = model_data["model"]
        self._feature_columns = preprocessor_data["feature_columns"]
        self._encoders = preprocessor_data["encoders"]
```
