# HuggingFace Transformers Example

This example demonstrates serving a HuggingFace sentiment analysis model with automatic weight downloading using `HFWeight`.

## Setup

```bash
pip install transformers torch
```

## Usage

```bash
export THALAMUS_API_KEY=your-key
python sentiment.py
```

On startup, thalamus-serve will automatically download the model from HuggingFace Hub and cache it locally.

## Testing

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"inputs": [{"text": "I love this product!"}, {"text": "This is terrible."}]}'
```

## Model Details

- **Model**: DistilBERT fine-tuned on SST-2
- **Input**: Text string
- **Output**: Sentiment label (POSITIVE/NEGATIVE) with confidence score

## Weight Configuration

The model weights are configured using `HFWeight` in the decorator:

```python
@app.model(
    model_id="sentiment",
    weights={
        "model": HFWeight(repo="distilbert-base-uncased-finetuned-sst-2-english"),
    },
)
```

You can also specify a specific file or revision:

```python
HFWeight(repo="bert-base-uncased", filename="pytorch_model.bin", revision="main")
```
