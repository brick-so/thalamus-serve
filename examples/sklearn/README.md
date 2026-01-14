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

To load a pre-trained model, configure `thalamus-deploy.json`:

```json
{
  "models": {
    "iris": {
      "weights": {
        "model": {
          "type": "s3",
          "bucket": "my-models",
          "key": "iris/model.joblib"
        }
      }
    }
  }
}
```
