# HuggingFace Transformers Example

This example demonstrates serving a HuggingFace sentiment analysis model.

## Setup

```bash
pip install transformers torch
```

## Usage

```bash
export THALAMUS_API_KEY=your-key
python sentiment.py
```

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
