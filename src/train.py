"""
Training loop for the Avito Demand Prediction model.

Features:
- Mixed precision training (FP16) for memory efficiency
- OneCycleLR scheduler for fast convergence
- Early stopping with patience
- Gradient clipping
- Detailed logging with per-epoch metrics
"""

import gc
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from config import TRAIN_CFG, CHECKPOINTS_DIR, FEATURES_DIR
from dataset import AvitoDataset, create_train_val_split
from model import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to terminate training when validation loss stops improving."""

    def __init__(self, patience: int = 3, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move all tensors in a batch dict to the specified device."""
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[OneCycleLR],
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    use_amp: bool = True,
    log_interval: int = 200,
    max_grad_norm: float = 1.0,
) -> float:
    """Train for one epoch. Returns average training loss (MSE)."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        batch = move_batch_to_device(batch, device)
        target = batch.pop("target")

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            predictions = model(batch)
            loss = nn.functional.mse_loss(predictions, target)

        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / num_batches
            rmse = np.sqrt(avg_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"  Epoch {epoch} | Batch {batch_idx+1}/{len(dataloader)} | "
                f"Loss: {avg_loss:.6f} | RMSE: {rmse:.4f} | LR: {current_lr:.2e}"
            )

    return total_loss / num_batches


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> float:
    """Validate the model. Returns average validation loss (MSE)."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        target = batch.pop("target")

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            predictions = model(batch)
            loss = nn.functional.mse_loss(predictions, target)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def run_training():
    """Main training function."""
    logger.info("=" * 70)
    logger.info("AVITO DEMAND PREDICTION — TRAINING")
    logger.info("=" * 70)

    device = torch.device(TRAIN_CFG.device)
    logger.info(f"Device: {device}")

    # ------------------------------------------------------------------
    # Create datasets and dataloaders
    # ------------------------------------------------------------------
    logger.info("\n[1/4] Creating datasets...")

    train_indices, val_indices = create_train_val_split(
        val_ratio=0.1, seed=42
    )
    logger.info(f"  Train samples: {len(train_indices):,}")
    logger.info(f"  Val samples: {len(val_indices):,}")

    train_dataset = AvitoDataset(split="train", indices=train_indices)
    val_dataset = AvitoDataset(split="train", indices=val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_CFG.batch_size,
        shuffle=True,
        num_workers=TRAIN_CFG.num_workers,
        pin_memory=TRAIN_CFG.pin_memory,
        drop_last=True,
        persistent_workers=True if TRAIN_CFG.num_workers > 0 else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAIN_CFG.batch_size * 2,  # Larger batch for eval
        shuffle=False,
        num_workers=TRAIN_CFG.num_workers,
        pin_memory=TRAIN_CFG.pin_memory,
        persistent_workers=True if TRAIN_CFG.num_workers > 0 else False,
    )

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    logger.info("\n[2/4] Building model...")
    model = build_model().to(device)
    num_params = model.count_parameters()
    logger.info(f"  Total trainable parameters: {num_params:,}")
    logger.info(f"  Model size: ~{num_params * 4 / 1e6:.1f} MB (FP32)")

    # ------------------------------------------------------------------
    # Setup optimizer, scheduler, scaler
    # ------------------------------------------------------------------
    logger.info("\n[3/4] Setting up optimizer...")
    optimizer = AdamW(
        model.parameters(),
        lr=TRAIN_CFG.learning_rate,
        weight_decay=TRAIN_CFG.weight_decay,
    )

    total_steps = len(train_loader) * TRAIN_CFG.num_epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=TRAIN_CFG.max_lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    scaler = torch.amp.GradScaler(enabled=TRAIN_CFG.use_amp)
    early_stopping = EarlyStopping(
        patience=TRAIN_CFG.patience,
        min_delta=TRAIN_CFG.min_delta,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    logger.info("\n[4/4] Starting training...")
    logger.info(f"  Epochs: {TRAIN_CFG.num_epochs}")
    logger.info(f"  Batch size: {TRAIN_CFG.batch_size}")
    logger.info(f"  Learning rate: {TRAIN_CFG.learning_rate}")
    logger.info(f"  Mixed precision: {TRAIN_CFG.use_amp}")

    best_val_rmse = float("inf")

    for epoch in range(1, TRAIN_CFG.num_epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
            use_amp=TRAIN_CFG.use_amp,
            log_interval=TRAIN_CFG.log_interval,
            max_grad_norm=TRAIN_CFG.max_grad_norm,
        )

        # Validate
        val_loss = validate(
            model=model,
            dataloader=val_loader,
            device=device,
            use_amp=TRAIN_CFG.use_amp,
        )

        train_rmse = np.sqrt(train_loss)
        val_rmse = np.sqrt(val_loss)
        epoch_time = time.time() - epoch_start

        logger.info(
            f"\n{'='*50}\n"
            f"Epoch {epoch}/{TRAIN_CFG.num_epochs} | Time: {epoch_time:.1f}s\n"
            f"  Train RMSE: {train_rmse:.5f}\n"
            f"  Val   RMSE: {val_rmse:.5f}\n"
            f"{'='*50}"
        )

        # Save best model
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_rmse": val_rmse,
                    "train_rmse": train_rmse,
                },
                CHECKPOINTS_DIR / "best_model.pt",
            )
            logger.info(f"  ★ New best model saved! Val RMSE: {val_rmse:.5f}")

        # Early stopping
        if early_stopping(val_loss):
            logger.info(f"\n⚠ Early stopping triggered after {epoch} epochs!")
            break

    logger.info(f"\n{'='*70}")
    logger.info(f"TRAINING COMPLETE!")
    logger.info(f"Best Validation RMSE: {best_val_rmse:.5f}")
    logger.info(f"{'='*70}")

    return best_val_rmse


if __name__ == "__main__":
    run_training()
