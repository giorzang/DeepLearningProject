"""
Configuration module for the Avito Demand Prediction pipeline.
All hyperparameters, paths, and feature definitions in one place.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# ============================================================================
# Path Configuration
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
INPUT_DIR = PROJECT_ROOT / "input"
FEATURES_DIR = PROJECT_ROOT / "features"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
TRAIN_CSV = INPUT_DIR / "train.csv"
TEST_CSV = INPUT_DIR / "test.csv"
TRAIN_IMAGES_DIR = INPUT_DIR / "train_jpg"
TEST_IMAGES_DIR = INPUT_DIR / "test_jpg"
SAMPLE_SUBMISSION = INPUT_DIR / "sample_submission.csv"

# Create output directories
FEATURES_DIR.mkdir(exist_ok=True)
CHECKPOINTS_DIR.mkdir(exist_ok=True)


@dataclass
class DataConfig:
    """Data processing configuration."""
    # Categorical features to embed
    categorical_features: List[str] = field(default_factory=lambda: [
        "region", "city", "parent_category_name", "category_name",
        "param_1", "param_2", "param_3", "user_type", "image_top_1",
    ])

    # Numerical features
    numerical_features: List[str] = field(default_factory=lambda: [
        "price", "item_seq_number",
        "day_of_week", "day_of_month", "week_of_year",
        "title_len", "desc_len", "title_word_count", "desc_word_count",
        "price_missing",
    ])

    # Text columns
    text_columns: List[str] = field(default_factory=lambda: ["title", "description"])

    # Text model for embedding extraction
    text_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    text_embed_dim: int = 384  # Output dim of MiniLM

    # Image model for feature extraction
    image_model_name: str = "efficientnet_b0"
    image_feature_dim: int = 1280  # EfficientNet-B0 feature dim
    image_size: int = 224

    # Preprocessing
    val_ratio: float = 0.1
    random_seed: int = 42
    preprocess_batch_size: int = 64  # For text/image feature extraction
    preprocess_num_workers: int = 4


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Embedding dimensions (will be populated dynamically)
    embedding_dims: Dict[str, tuple] = field(default_factory=dict)
    # (num_categories, embedding_dim) for each categorical feature

    # Tower output dimensions
    tabular_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    text_projection_dim: int = 128
    image_projection_dim: int = 128

    # Fusion
    fusion_dim: int = 128
    num_modalities: int = 3

    # Prediction head
    head_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])

    # Regularization
    dropout: float = 0.3
    ghost_batch_size: int = 64  # For Ghost Batch Normalization


@dataclass
class TrainConfig:
    """Training configuration."""
    batch_size: int = 256
    num_epochs: int = 10
    learning_rate: float = 3e-3
    weight_decay: float = 1e-2
    max_lr: float = 3e-3

    # Early stopping
    patience: int = 3
    min_delta: float = 1e-5

    # Mixed precision
    use_amp: bool = True

    # DataLoader
    num_workers: int = 4
    pin_memory: bool = True

    # Logging
    log_interval: int = 200  # Log every N batches

    # Device
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    # Gradient clipping
    max_grad_norm: float = 1.0


# Default configurations
DATA_CFG = DataConfig()
MODEL_CFG = ModelConfig()
TRAIN_CFG = TrainConfig()
