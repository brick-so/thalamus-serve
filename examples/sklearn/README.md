# scikit-learn Example

This example demonstrates serving a scikit-learn classification model.

## Setup

```bash
pip install scikit-learn joblib
```

## Usage

```bash
export THALAMUS_API_KEY=your-key
python iris_classifier.py
```

## Testing

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"inputs": [{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}]}'
```

## Model Details

- **Model**: Random Forest classifier trained on Iris dataset
- **Input**: Sepal length, sepal width, petal length, petal width
- **Output**: Species prediction with class probabilities

## Loading Pre-trained Models

To load a pre-trained model, specify weights directly in the decorator:

```python
from thalamus_serve import Thalamus, S3Weight

app = Thalamus()

@app.model(
    model_id="iris",
    weights={
        "model": S3Weight(bucket="my-models", key="iris/model.joblib"),
    },
)
class IrisClassifier:
    def load(self, weights: dict[str, Path], device: str) -> None:
        import joblib
        self.model = joblib.load(weights["model"])
```
