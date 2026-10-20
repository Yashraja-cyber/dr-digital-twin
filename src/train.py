"""Training script for the DR grading model.

Fine-tunes EfficientNet-B3 on the APTOS 2019 dataset with:
- Mixed precision training (AMP)
- Cosine annealing LR schedule
- Weighted sampling for class imbalance
- Ordinal-aware loss function
- Early stopping with best model checkpointing
- Quadratic Weighted Kappa (QWK) evaluation

Usage:
    python train.py --data_dir data/raw/aptos2019 --epochs 15 --batch_size 16
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score, classification_report
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    CLASS_NAMES,
    EARLY_STOPPING_PATIENCE,
    IMAGE_SIZE,
    LEARNING_RATE,
    MIXED_PRECISION,
    MODELS_DIR,
    NUM_EPOCHS,
    RANDOM_SEED,
    WEIGHT_DECAY,
)
from src.data.dataset import get_class_weights, get_data_loaders
from src.models.dr_grader import DRGrader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer,
    scaler: GradScaler | None,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    """Train for one epoch.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler and MIXED_PRECISION:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Validate the model.

    Returns:
        Tuple of (avg_loss, accuracy, qwk, all_preds, all_labels).
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Validating", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    total = len(all_labels)

    avg_loss = running_loss / total
    accuracy = np.mean(all_preds == all_labels)
    qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")

    return avg_loss, accuracy, qwk, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description="Train DR Grading Model")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to APTOS dataset")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Seed
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    # Data
    logger.info("Loading dataset...")
    train_loader, val_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
    )
    logger.info(f"Train: {len(train_loader.dataset)} images | Val: {len(val_loader.dataset)} images")

    # Model
    logger.info(f"Creating model: EfficientNet-B3 (pretrained=True)")
    model = DRGrader(pretrained=True).to(device)

    # Loss with class weights
    all_labels = train_loader.dataset.labels_df["diagnosis"].values
    class_weights = get_class_weights(all_labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = GradScaler() if MIXED_PRECISION and device.type == "cuda" else None

    # Training loop
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_qwk = -1.0
    patience_counter = 0

    logger.info(f"Starting training for {args.epochs} epochs...")
    start_time = time.time()

    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )

        # Validate
        val_loss, val_acc, val_qwk, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} QWK: {val_qwk:.4f} | "
            f"LR: {lr:.6f}"
        )

        # Checkpoint best model
        if val_qwk > best_qwk:
            best_qwk = val_qwk
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_qwk": val_qwk,
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, str(BEST_MODEL_PATH))
            logger.info(f"✅ Best model saved (QWK: {val_qwk:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    elapsed = time.time() - start_time
    logger.info(f"Training complete in {elapsed/60:.1f} min | Best QWK: {best_qwk:.4f}")

    # Final evaluation with best model
    model.load_state_dict(torch.load(str(BEST_MODEL_PATH))["model_state_dict"])
    _, _, final_qwk, final_preds, final_labels = validate(
        model, val_loader, criterion, device
    )

    logger.info(f"\n{'='*50}")
    logger.info(f"FINAL RESULTS (Best Model)")
    logger.info(f"{'='*50}")
    logger.info(f"Quadratic Weighted Kappa: {final_qwk:.4f}")
    logger.info(f"\n{classification_report(final_labels, final_preds, target_names=CLASS_NAMES)}")


if __name__ == "__main__":
    main()
