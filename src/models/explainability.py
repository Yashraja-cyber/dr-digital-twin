"""GradCAM explainability for the DR grading model.

Generates visual explanations showing which regions of the fundus
image the model focuses on for its grading decision. This is
critical for clinical trust and validation.

Implements Gradient-weighted Class Activation Mapping (GradCAM)
targeting the last convolutional layer of EfficientNet-B3.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from
Deep Networks via Gradient-based Localization", ICCV 2017.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.config import IMAGE_SIZE
from src.data.preprocessing import preprocess_for_inference


class GradCAMExplainer:
    """Generate GradCAM heatmaps for the DR grading model.

    Hooks into the last convolutional layer of EfficientNet-B3 to
    capture activations and gradients, then computes the weighted
    activation map showing which spatial regions drive the prediction.

    Args:
        model: A DRGrader model instance.
        device: Computation device ('cpu' or 'cuda').
    """

    def __init__(self, model, device: str = "cpu"):
        self.model = model
        self.device = device
        self.activations = None
        self.gradients = None

        # Register hooks on the last convolutional block
        # For EfficientNet-B3, this is backbone.conv_head
        target_layer = self._find_target_layer()
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _find_target_layer(self):
        """Find the appropriate target layer for GradCAM.

        For timm's EfficientNet models, the best target is `conv_head`
        (the 1x1 conv after the last block). Falls back to the last
        block if conv_head is not available.
        """
        if hasattr(self.model.backbone, "conv_head"):
            return self.model.backbone.conv_head
        # Fallback: last block in the features
        if hasattr(self.model.backbone, "blocks"):
            return self.model.backbone.blocks[-1]
        raise AttributeError(
            "Cannot find target convolutional layer for GradCAM. "
            "Expected 'conv_head' or 'blocks' attribute on backbone."
        )

    def _save_activation(self, module, input_, output):
        """Forward hook to capture activations."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Backward hook to capture gradients."""
        self.gradients = grad_output[0].detach()

    def generate_heatmap(
        self,
        image: np.ndarray | str,
        target_class: int | None = None,
        image_size: int = IMAGE_SIZE,
    ) -> tuple[np.ndarray, int, float]:
        """Generate a GradCAM heatmap for a fundus image.

        Args:
            image: BGR numpy array or path to image file.
            target_class: Class to generate explanation for.
                          If None, uses the predicted class.
            image_size: Preprocessing target size.

        Returns:
            Tuple of:
                - heatmap: Normalized heatmap array (H, W) in [0, 1]
                - predicted_class: The predicted DR grade
                - confidence: Prediction confidence
        """
        self.model.eval()

        # Preprocess
        processed = preprocess_for_inference(image, size=image_size)
        input_tensor = torch.from_numpy(processed).unsqueeze(0).to(self.device)
        input_tensor.requires_grad_(True)

        # Forward pass
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)

        predicted_class = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, predicted_class].item())

        if target_class is None:
            target_class = predicted_class

        # Backward pass for the target class
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        # Compute GradCAM
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)

        # Global average pooling of gradients → channel weights
        weights = torch.mean(gradients, dim=(1, 2))  # (C,)

        # Weighted combination of activation maps
        heatmap = torch.zeros(activations.shape[1:], device=self.device)
        for i, w in enumerate(weights):
            heatmap += w * activations[i]

        # ReLU — we only want positive contributions
        heatmap = F.relu(heatmap)

        # Normalize to [0, 1]
        heatmap = heatmap.cpu().numpy()
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        # Resize to input image size
        heatmap = cv2.resize(heatmap, (image_size, image_size))

        return heatmap, predicted_class, confidence

    def overlay_on_image(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """Overlay a GradCAM heatmap on the original fundus image.

        Args:
            original_image: Original BGR fundus image.
            heatmap: Normalized heatmap array (H, W) in [0, 1].
            alpha: Blending weight for the heatmap overlay.
            colormap: OpenCV colormap for the heatmap visualization.

        Returns:
            BGR image with heatmap overlay.
        """
        # Resize heatmap to match image dimensions
        h, w = original_image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))

        # Convert to uint8 and apply colormap
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)

        # Blend with original image
        overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)

        return overlay

    def get_attention_regions(
        self,
        heatmap: np.ndarray,
        threshold: float = 0.5,
    ) -> list[dict]:
        """Extract high-attention regions from the heatmap.

        Identifies contiguous regions where the model's attention
        exceeds the threshold — potential lesion locations.

        Args:
            heatmap: Normalized heatmap array (H, W) in [0, 1].
            threshold: Minimum attention value to consider.

        Returns:
            List of dicts with region info:
                - bbox: (x, y, w, h) bounding box
                - center: (cx, cy) center point
                - area: number of pixels
                - max_attention: peak attention value in region
        """
        # Threshold the heatmap
        binary = (heatmap > threshold).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50:  # Skip tiny regions
                continue

            x, y, w, h = cv2.boundingRect(contour)
            mask = np.zeros_like(heatmap, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 1, -1)
            max_attention = float(np.max(heatmap * mask))

            regions.append({
                "bbox": (int(x), int(y), int(w), int(h)),
                "center": (int(x + w // 2), int(y + h // 2)),
                "area": int(area),
                "max_attention": max_attention,
            })

        # Sort by attention (highest first)
        regions.sort(key=lambda r: r["max_attention"], reverse=True)
        return regions
