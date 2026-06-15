"""
Memory-efficient PyTorch Dataset for the Avito Demand Prediction Challenge.

Uses memory-mapped NumPy arrays (np.load with mmap_mode='r') so that
only the rows actually accessed are loaded into RAM.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional

from config import FEATURES_DIR, DATA_CFG


class AvitoDataset(Dataset):
    """
    Multi-modal dataset that lazily loads pre-computed features.

    Features loaded:
    - Categorical (int32): region, city, category, params, user_type, image_top_1
    - Numerical (float32): price, item_seq_number, date features, text stats
    - Title embeddings (float16 → float32): from sentence transformer
    - Description embeddings (float16 → float32): from sentence transformer
    - Image features (float16 → float32): from EfficientNet-B0
    - Image mask (bool): whether the image exists
    - Target (float32): deal_probability (train only)
    """

    def __init__(self, split: str = "train", indices: Optional[np.ndarray] = None):
        """
        Args:
            split: "train" or "test"
            indices: Optional subset indices (for train/val split)
        """
        assert split in ("train", "test"), f"Unknown split: {split}"
        self.split = split
        self.indices = indices

        prefix = split

        # Load as memory-mapped arrays
        self.cat_features = np.load(
            FEATURES_DIR / f"{prefix}_cat.npy", mmap_mode="r"
        )
        self.num_features = np.load(
            FEATURES_DIR / f"{prefix}_num.npy", mmap_mode="r"
        )
        self.title_emb = np.load(
            FEATURES_DIR / f"{prefix}_title_emb.npy", mmap_mode="r"
        )
        self.desc_emb = np.load(
            FEATURES_DIR / f"{prefix}_desc_emb.npy", mmap_mode="r"
        )
        self.img_features = np.load(
            FEATURES_DIR / f"{prefix}_img_feat.npy", mmap_mode="r"
        )
        self.img_mask = np.load(
            FEATURES_DIR / f"{prefix}_img_feat_mask.npy", mmap_mode="r"
        )

        if split == "train":
            self.target = np.load(
                FEATURES_DIR / "train_target.npy", mmap_mode="r"
            )
        else:
            self.target = None

        # Determine effective length
        if indices is not None:
            self._len = len(indices)
        else:
            self._len = len(self.cat_features)

    def __len__(self):
        return self._len

    def _get_real_idx(self, idx):
        """Map external index to internal index."""
        if self.indices is not None:
            return self.indices[idx]
        return idx

    def __getitem__(self, idx):
        real_idx = self._get_real_idx(idx)

        sample = {
            "cat_features": torch.from_numpy(
                self.cat_features[real_idx].copy()
            ).long(),
            "num_features": torch.from_numpy(
                self.num_features[real_idx].copy()
            ).float(),
            "title_emb": torch.from_numpy(
                self.title_emb[real_idx].copy()
            ).float(),
            "desc_emb": torch.from_numpy(
                self.desc_emb[real_idx].copy()
            ).float(),
            "img_features": torch.from_numpy(
                self.img_features[real_idx].copy()
            ).float(),
            "img_mask": torch.tensor(
                self.img_mask[real_idx], dtype=torch.bool
            ),
        }

        if self.target is not None:
            sample["target"] = torch.tensor(
                self.target[real_idx], dtype=torch.float32
            )

        return sample


def create_train_val_split(
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple:
    """
    Create stratified-ish train/val split based on target distribution.
    Returns (train_indices, val_indices) as numpy arrays.
    """
    target = np.load(FEATURES_DIR / "train_target.npy")
    n = len(target)

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)

    val_size = int(n * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    return train_indices, val_indices
