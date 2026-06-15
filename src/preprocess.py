"""
Offline preprocessing pipeline for the Avito Demand Prediction Challenge.

This module handles:
1. Tabular feature engineering (categorical encoding, numerical transforms)
2. Text embedding extraction (using multilingual sentence transformers)
3. Image feature extraction (using EfficientNet-B0)

All features are saved as memory-mapped NumPy arrays for efficient loading.
"""

import os
import gc
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

from config import (
    DATA_CFG, TRAIN_CSV, TEST_CSV, FEATURES_DIR,
    TRAIN_IMAGES_DIR, TEST_IMAGES_DIR,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# 1. Tabular Feature Engineering
# ============================================================================

def load_and_engineer_tabular(csv_path: Path, is_train: bool = True) -> pd.DataFrame:
    """Load CSV and create engineered features."""
    logger.info(f"Loading {csv_path.name}...")

    # Read in chunks for memory efficiency
    dtype_map = {
        "item_id": str, "user_id": str, "region": str, "city": str,
        "parent_category_name": str, "category_name": str,
        "param_1": str, "param_2": str, "param_3": str,
        "title": str, "description": str, "image": str,
        "user_type": str, "activation_date": str,
    }
    df = pd.read_csv(csv_path, dtype=dtype_map, na_values=["", "NA", "NaN"])

    logger.info(f"  Shape: {df.shape}")

    # --- Date features ---
    df["activation_date"] = pd.to_datetime(df["activation_date"])
    df["day_of_week"] = df["activation_date"].dt.dayofweek.astype(np.float32)
    df["day_of_month"] = df["activation_date"].dt.day.astype(np.float32)
    df["week_of_year"] = df["activation_date"].dt.isocalendar().week.astype(np.float32)

    # --- Text length features ---
    df["title"] = df["title"].fillna("")
    df["description"] = df["description"].fillna("")
    df["title_len"] = df["title"].str.len().astype(np.float32)
    df["desc_len"] = df["description"].str.len().astype(np.float32)
    df["title_word_count"] = df["title"].str.split().str.len().fillna(0).astype(np.float32)
    df["desc_word_count"] = df["description"].str.split().str.len().fillna(0).astype(np.float32)

    # --- Price features ---
    df["price_missing"] = df["price"].isna().astype(np.float32)
    df["price"] = df["price"].fillna(0.0)

    # --- Log transform skewed numericals ---
    df["price"] = np.log1p(df["price"]).astype(np.float32)
    df["item_seq_number"] = np.log1p(df["item_seq_number"].astype(np.float64)).astype(np.float32)

    return df


def encode_categoricals(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cat_cols: List[str],
) -> Tuple[Dict[str, Dict], Dict[str, int]]:
    """
    Build label encodings for categorical features across train+test.
    Returns the encoding maps and the vocabulary sizes.
    """
    label_maps = {}
    vocab_sizes = {}

    for col in cat_cols:
        # Combine unique values from train and test
        combined = pd.concat([
            train_df[col].fillna("__MISSING__"),
            test_df[col].fillna("__MISSING__"),
        ]).astype(str)

        unique_vals = combined.unique()
        # 0 = padding/unknown, 1..N = actual values
        mapping = {v: i + 1 for i, v in enumerate(sorted(unique_vals))}
        mapping["__UNK__"] = 0

        label_maps[col] = mapping
        vocab_sizes[col] = len(mapping)
        logger.info(f"  {col}: {len(mapping)} unique values")

    return label_maps, vocab_sizes


def apply_categorical_encoding(
    df: pd.DataFrame, label_maps: Dict[str, Dict], cat_cols: List[str]
) -> np.ndarray:
    """Apply label encoding and return integer array."""
    encoded = np.zeros((len(df), len(cat_cols)), dtype=np.int32)
    for i, col in enumerate(cat_cols):
        mapping = label_maps[col]
        encoded[:, i] = (
            df[col]
            .fillna("__MISSING__")
            .astype(str)
            .map(lambda x, m=mapping: m.get(x, 0))
            .values
        )
    return encoded


def extract_numerical(df: pd.DataFrame, num_cols: List[str]) -> np.ndarray:
    """Extract and normalize numerical features."""
    numericals = df[num_cols].values.astype(np.float32)
    return numericals


def compute_numerical_stats(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and std for standardization."""
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    std[std < 1e-7] = 1.0  # Prevent division by zero
    return mean, std


# ============================================================================
# 2. Text Embedding Extraction
# ============================================================================

def extract_text_embeddings(
    texts: List[str],
    model_name: str,
    batch_size: int = 64,
    output_path: Optional[Path] = None,
    desc: str = "Extracting text embeddings",
) -> np.ndarray:
    """
    Extract sentence embeddings using a pre-trained sentence transformer.
    Uses batched inference with GPU acceleration.
    """
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading text model: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)

    logger.info(f"Extracting embeddings for {len(texts)} texts...")

    # Clean texts
    texts = [str(t) if pd.notna(t) else "" for t in texts]

    # Encode in batches
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    if output_path is not None:
        np.save(output_path, embeddings.astype(np.float16))
        logger.info(f"Saved text embeddings to {output_path} (shape: {embeddings.shape})")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return embeddings


# ============================================================================
# 3. Image Feature Extraction
# ============================================================================

class ImageFeatureDataset(Dataset):
    """Dataset for batch image feature extraction."""

    def __init__(self, image_ids: List[str], image_dir: Path, image_size: int = 224):
        self.image_ids = image_ids
        self.image_dir = image_dir
        self.image_size = image_size

        # Pre-compute ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]

        if pd.isna(img_id) or img_id == "" or img_id == "nan":
            # Return zero tensor for missing images
            return torch.zeros(3, self.image_size, self.image_size), False

        img_path = self.image_dir / f"{img_id}.jpg"
        if not img_path.exists():
            return torch.zeros(3, self.image_size, self.image_size), False

        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            img = np.array(img, dtype=np.float32) / 255.0
            img = (img - self.mean) / self.std
            img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
            return torch.from_numpy(img), True
        except Exception:
            return torch.zeros(3, self.image_size, self.image_size), False


def extract_image_features(
    image_ids: List[str],
    image_dir: Path,
    batch_size: int = 64,
    num_workers: int = 4,
    output_path: Optional[Path] = None,
    desc: str = "Extracting image features",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract image features using pre-trained EfficientNet-B0.
    Returns features array and a boolean mask indicating which images exist.
    """
    import torchvision.models as models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained EfficientNet-B0
    logger.info("Loading EfficientNet-B0...")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Remove classifier, keep feature extractor
    model.classifier = nn.Identity()
    model = model.to(device)
    model.eval()

    dataset = ImageFeatureDataset(image_ids, image_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    all_features = []
    all_masks = []

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=True):
        for batch_imgs, batch_masks in tqdm(dataloader, desc=desc):
            batch_imgs = batch_imgs.to(device, non_blocking=True)
            features = model(batch_imgs)
            all_features.append(features.cpu().numpy().astype(np.float16))
            all_masks.append(batch_masks.numpy())

    features = np.concatenate(all_features, axis=0)
    masks = np.concatenate(all_masks, axis=0)

    if output_path is not None:
        np.save(output_path, features)
        mask_path = output_path.parent / (output_path.stem + "_mask.npy")
        np.save(mask_path, masks)
        logger.info(f"Saved image features to {output_path} (shape: {features.shape})")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return features, masks


# ============================================================================
# Main Preprocessing Pipeline
# ============================================================================

def run_preprocessing():
    """Run the full preprocessing pipeline."""
    logger.info("=" * 70)
    logger.info("AVITO DEMAND PREDICTION — PREPROCESSING PIPELINE")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Load and engineer tabular features
    # ------------------------------------------------------------------
    logger.info("\n[Step 1/5] Loading and engineering tabular features...")
    train_df = load_and_engineer_tabular(TRAIN_CSV, is_train=True)
    test_df = load_and_engineer_tabular(TEST_CSV, is_train=False)

    # ------------------------------------------------------------------
    # Step 2: Encode categoricals
    # ------------------------------------------------------------------
    logger.info("\n[Step 2/5] Encoding categorical features...")
    cat_cols = DATA_CFG.categorical_features
    label_maps, vocab_sizes = encode_categoricals(train_df, test_df, cat_cols)

    train_cat = apply_categorical_encoding(train_df, label_maps, cat_cols)
    test_cat = apply_categorical_encoding(test_df, label_maps, cat_cols)

    np.save(FEATURES_DIR / "train_cat.npy", train_cat)
    np.save(FEATURES_DIR / "test_cat.npy", test_cat)

    # Save vocab sizes for model construction
    with open(FEATURES_DIR / "vocab_sizes.json", "w") as f:
        json.dump(vocab_sizes, f, indent=2)

    logger.info(f"  Categorical shapes: train={train_cat.shape}, test={test_cat.shape}")

    # ------------------------------------------------------------------
    # Step 3: Extract and normalize numerical features
    # ------------------------------------------------------------------
    logger.info("\n[Step 3/5] Extracting numerical features...")
    num_cols = DATA_CFG.numerical_features

    train_num = extract_numerical(train_df, num_cols)
    test_num = extract_numerical(test_df, num_cols)

    # Compute stats from training data only
    num_mean, num_std = compute_numerical_stats(train_num)
    np.save(FEATURES_DIR / "num_mean.npy", num_mean)
    np.save(FEATURES_DIR / "num_std.npy", num_std)

    # Standardize
    train_num = (train_num - num_mean) / num_std
    test_num = (test_num - num_mean) / num_std

    # Replace any remaining NaN with 0
    train_num = np.nan_to_num(train_num, nan=0.0).astype(np.float32)
    test_num = np.nan_to_num(test_num, nan=0.0).astype(np.float32)

    np.save(FEATURES_DIR / "train_num.npy", train_num)
    np.save(FEATURES_DIR / "test_num.npy", test_num)

    logger.info(f"  Numerical shapes: train={train_num.shape}, test={test_num.shape}")

    # Save target
    if "deal_probability" in train_df.columns:
        target = train_df["deal_probability"].values.astype(np.float32)
        np.save(FEATURES_DIR / "train_target.npy", target)
        logger.info(f"  Target shape: {target.shape}, mean={target.mean():.4f}")

    # Save item_ids for submission
    np.save(FEATURES_DIR / "train_item_ids.npy", train_df["item_id"].values)
    np.save(FEATURES_DIR / "test_item_ids.npy", test_df["item_id"].values)

    # ------------------------------------------------------------------
    # Step 4: Extract text embeddings
    # ------------------------------------------------------------------
    logger.info("\n[Step 4/5] Extracting text embeddings...")

    # Title embeddings
    if not (FEATURES_DIR / "train_title_emb.npy").exists():
        extract_text_embeddings(
            train_df["title"].tolist(),
            DATA_CFG.text_model_name,
            batch_size=DATA_CFG.preprocess_batch_size,
            output_path=FEATURES_DIR / "train_title_emb.npy",
            desc="Train titles",
        )
    else:
        logger.info("  Train title embeddings already exist, skipping.")

    if not (FEATURES_DIR / "test_title_emb.npy").exists():
        extract_text_embeddings(
            test_df["title"].tolist(),
            DATA_CFG.text_model_name,
            batch_size=DATA_CFG.preprocess_batch_size,
            output_path=FEATURES_DIR / "test_title_emb.npy",
            desc="Test titles",
        )
    else:
        logger.info("  Test title embeddings already exist, skipping.")

    # Description embeddings
    if not (FEATURES_DIR / "train_desc_emb.npy").exists():
        extract_text_embeddings(
            train_df["description"].tolist(),
            DATA_CFG.text_model_name,
            batch_size=DATA_CFG.preprocess_batch_size,
            output_path=FEATURES_DIR / "train_desc_emb.npy",
            desc="Train descriptions",
        )
    else:
        logger.info("  Train desc embeddings already exist, skipping.")

    if not (FEATURES_DIR / "test_desc_emb.npy").exists():
        extract_text_embeddings(
            test_df["description"].tolist(),
            DATA_CFG.text_model_name,
            batch_size=DATA_CFG.preprocess_batch_size,
            output_path=FEATURES_DIR / "test_desc_emb.npy",
            desc="Test descriptions",
        )
    else:
        logger.info("  Test desc embeddings already exist, skipping.")

    # Free memory before image extraction
    del train_df, test_df
    gc.collect()

    # ------------------------------------------------------------------
    # Step 5: Extract image features
    # ------------------------------------------------------------------
    logger.info("\n[Step 5/5] Extracting image features...")

    # Reload image IDs only
    train_images = pd.read_csv(TRAIN_CSV, usecols=["image"], dtype=str)["image"].tolist()
    test_images = pd.read_csv(TEST_CSV, usecols=["image"], dtype=str)["image"].tolist()

    if not (FEATURES_DIR / "train_img_feat.npy").exists():
        extract_image_features(
            train_images, TRAIN_IMAGES_DIR,
            batch_size=DATA_CFG.preprocess_batch_size,
            num_workers=DATA_CFG.preprocess_num_workers,
            output_path=FEATURES_DIR / "train_img_feat.npy",
            desc="Train images",
        )
    else:
        logger.info("  Train image features already exist, skipping.")

    if not (FEATURES_DIR / "test_img_feat.npy").exists():
        extract_image_features(
            test_images, TEST_IMAGES_DIR,
            batch_size=DATA_CFG.preprocess_batch_size,
            num_workers=DATA_CFG.preprocess_num_workers,
            output_path=FEATURES_DIR / "test_img_feat.npy",
            desc="Test images",
        )
    else:
        logger.info("  Test image features already exist, skipping.")

    logger.info("\n" + "=" * 70)
    logger.info("PREPROCESSING COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"All features saved to: {FEATURES_DIR}")


if __name__ == "__main__":
    run_preprocessing()
