"""
Generate predictions on the test set and create a Kaggle submission file.
"""

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    TRAIN_CFG, CHECKPOINTS_DIR, FEATURES_DIR,
    SAMPLE_SUBMISSION, PROJECT_ROOT,
)
from dataset import AvitoDataset
from model import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move all tensors in a batch dict to the specified device."""
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


@torch.no_grad()
def generate_predictions():
    """Generate test set predictions and create submission."""
    logger.info("=" * 70)
    logger.info("AVITO DEMAND PREDICTION — GENERATING PREDICTIONS")
    logger.info("=" * 70)

    device = torch.device(TRAIN_CFG.device)

    # Build model and load best checkpoint
    logger.info("Loading best model...")
    model = build_model().to(device)

    checkpoint = torch.load(CHECKPOINTS_DIR / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"  Loaded checkpoint from epoch {checkpoint['epoch']}")
    logger.info(f"  Val RMSE at checkpoint: {checkpoint['val_rmse']:.5f}")

    model.eval()

    # Create test dataset and loader
    logger.info("Creating test dataset...")
    test_dataset = AvitoDataset(split="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAIN_CFG.batch_size * 2,
        shuffle=False,
        num_workers=TRAIN_CFG.num_workers,
        pin_memory=True,
    )

    # Generate predictions
    logger.info("Generating predictions...")
    all_preds = []

    for batch in test_loader:
        batch = move_batch_to_device(batch, device)

        with torch.amp.autocast(device_type="cuda", enabled=TRAIN_CFG.use_amp):
            predictions = model(batch)

        all_preds.append(predictions.cpu().numpy())

    predictions = np.concatenate(all_preds, axis=0)

    # Clip predictions to [0, 1]
    predictions = np.clip(predictions, 0.0, 1.0)

    # Create submission
    logger.info("Creating submission file...")
    test_ids = np.load(FEATURES_DIR / "test_item_ids.npy", allow_pickle=True)

    submission = pd.DataFrame({
        "item_id": test_ids,
        "deal_probability": predictions,
    })

    submission_path = PROJECT_ROOT / "submission.csv"
    submission.to_csv(submission_path, index=False)

    logger.info(f"\n{'='*70}")
    logger.info(f"PREDICTION COMPLETE!")
    logger.info(f"  Submission saved to: {submission_path}")
    logger.info(f"  Shape: {submission.shape}")
    logger.info(f"  Predictions — mean: {predictions.mean():.4f}, "
                f"std: {predictions.std():.4f}, "
                f"min: {predictions.min():.4f}, max: {predictions.max():.4f}")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    generate_predictions()
