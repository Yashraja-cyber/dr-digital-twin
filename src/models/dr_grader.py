"""EfficientNet-B3 based Diabetic Retinopathy grading model.

Architecture: EfficientNet-B3 backbone (pretrained on ImageNet) with
a custom classification head for 5-class ordinal DR grading.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import BEST_MODEL_PATH, CLASS_NAMES, IMAGE_SIZE, MODEL_NAME, NUM_CLASSES
from src.data.preprocessing import preprocess_for_inference


class DRGrader(nn.Module):
    """DR severity grading model based on EfficientNet-B3.

    Uses a pretrained EfficientNet-B3 backbone with a custom dropout-
    regularized classification head. Trained on APTOS 2019 with class
    balancing and ordinal regression awareness.

    Args:
        num_classes: Number of DR severity grades (default 5).
        pretrained: Whether to load ImageNet pretrained weights.
        dropout_rate: Dropout probability in the classification head.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Load pretrained backbone without its classifier
        self.backbone = timm.create_model(
            MODEL_NAME,
            pretrained=pretrained,
            num_classes=0,  # Remove the default head
        )
        backbone_features = self.backbone.num_features  # 1536 for EfficientNet-B3

        # Custom classification head
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(backbone_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.67),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        features = self.backbone(x)
        logits = self.head(features)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities via softmax.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Probability tensor of shape (B, num_classes).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
        return probs

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get predicted class and confidence.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Tuple of (predicted_classes, confidences).
        """
        probs = self.predict_proba(x)
        confidences, predictions = torch.max(probs, dim=1)
        return predictions, confidences


def load_model(
    checkpoint_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    pretrained_backbone: bool = False,
) -> DRGrader:
    """Load a trained DR grading model from checkpoint.

    Args:
        checkpoint_path: Path to the .pth checkpoint file.
                         Defaults to BEST_MODEL_PATH from config.
        device: Device to load the model onto.
        pretrained_backbone: Whether to load ImageNet weights for backbone
                             (set False if loading full checkpoint).

    Returns:
        Loaded DRGrader model in eval mode.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
    """
    if checkpoint_path is None:
        checkpoint_path = BEST_MODEL_PATH

    checkpoint_path = Path(checkpoint_path)

    model = DRGrader(pretrained=pretrained_backbone)

    if checkpoint_path.exists():
        checkpoint = torch.load(str(checkpoint_path), map_location=device)
        # Handle both raw state_dict and wrapped checkpoint formats
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


def predict_image(
    image_input: str | np.ndarray,
    model: DRGrader | None = None,
    device: str = "cpu",
) -> dict:
    """Run DR grading inference on a single fundus image.

    Args:
        image_input: Path to image file or BGR numpy array.
        model: Pre-loaded DRGrader model. If None, loads from checkpoint.
        device: Computation device.

    Returns:
        Dictionary with:
            - grade: int (0-4)
            - grade_label: str
            - confidence: float (0-1)
            - probabilities: dict mapping grade labels to probabilities
    """
    if model is None:
        model = load_model(device=device)

    # Preprocess
    processed = preprocess_for_inference(image_input, size=IMAGE_SIZE)
    tensor = torch.from_numpy(processed).unsqueeze(0).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    predicted_grade = int(np.argmax(probs))
    confidence = float(probs[predicted_grade])

    return {
        "grade": predicted_grade,
        "grade_label": CLASS_NAMES[predicted_grade],
        "confidence": confidence,
        "probabilities": {
            CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))
        },
    }
