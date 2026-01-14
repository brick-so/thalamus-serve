# PyTorch Image Classifier Example

This example demonstrates serving a PyTorch ResNet model for image classification.

## Setup

```bash
pip install torch torchvision pillow
```

## Usage

```bash
export THALAMUS_API_KEY=your-key
python classifier.py
```

## Testing

```bash
# Encode an image to base64
IMAGE_B64=$(base64 -i your_image.jpg)

# Send prediction request
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d "{\"inputs\": [{\"image\": {\"data\": \"$IMAGE_B64\", \"media_type\": \"image/jpeg\"}}]}"
```

## Model Details

- **Model**: ResNet-50 pretrained on ImageNet
- **Input**: Base64-encoded image (JPEG, PNG)
- **Output**: Top-5 predictions with class names and probabilities
