"""PyTorch image classification example using ResNet."""

from io import BytesIO
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from thalamus_serve import Base64Data, Thalamus


class ImageInput(BaseModel):
    """Input schema for image classification."""

    image: Base64Data


class Prediction(BaseModel):
    """Single prediction with label and probability."""

    label: str
    probability: float


class ClassificationOutput(BaseModel):
    """Output schema with top-5 predictions."""

    predictions: list[Prediction]


app = Thalamus()


@app.model(
    model_id="resnet50",
    version="1.0.0",
    description="ResNet-50 ImageNet classifier",
    default=True,
    input_type=ImageInput,
    output_type=ClassificationOutput,
)
class ResNetClassifier:
    """ResNet-50 image classifier using torchvision pretrained weights."""

    def __init__(self) -> None:
        self.model: Any = None
        self.transform: Any = None
        self.labels: list[str] = []
        self.device: str = "cpu"

    def load(self, weights: dict[str, Path], device: str) -> None:
        """Load pretrained ResNet-50 model."""
        from torchvision import models, transforms

        self.device = device

        # Load pretrained ResNet-50
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.model.eval()
        self.model.to(device)

        # ImageNet preprocessing
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Load ImageNet labels
        self.labels = models.ResNet50_Weights.IMAGENET1K_V2.meta["categories"]

    def preprocess(self, inputs: list[ImageInput]) -> list[Any]:
        """Convert images to tensor batch."""
        import torch
        from PIL import Image

        tensors = []
        for inp in inputs:
            image_bytes = inp.image.decode()
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            tensor = self.transform(image)
            tensors.append(tensor)
        return [torch.stack(tensors).to(self.device)]

    def predict(self, inputs: list[Any]) -> list[Any]:
        """Run inference on the batch."""
        import torch

        batch = inputs[0]
        with torch.no_grad():
            outputs = self.model(batch)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        return [probabilities]

    def postprocess(self, outputs: list[Any]) -> list[ClassificationOutput]:
        """Convert model outputs to classification results."""
        import torch

        probabilities = outputs[0]
        results = []

        for probs in probabilities:
            top5_probs, top5_indices = torch.topk(probs, 5)
            predictions = [
                Prediction(label=self.labels[idx], probability=float(prob))
                for prob, idx in zip(top5_probs, top5_indices, strict=True)
            ]
            results.append(ClassificationOutput(predictions=predictions))

        return results


if __name__ == "__main__":
    app.serve()
