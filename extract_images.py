"""
Memory-efficient image feature extraction using chunk-based disk writes.
Avoids OOM by writing features to a pre-allocated memory-mapped file.
"""

import gc
import sys
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "src"))
from config import (
    DATA_CFG, FEATURES_DIR, TRAIN_CSV, TEST_CSV,
    TRAIN_IMAGES_DIR, TEST_IMAGES_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class ImageFeatureDataset(Dataset):
    """Dataset for batch image feature extraction."""

    def __init__(self, image_ids, image_dir, image_size=224):
        self.image_ids = image_ids
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]

        if pd.isna(img_id) or img_id == "" or img_id == "nan":
            return torch.zeros(3, self.image_size, self.image_size), False

        img_path = self.image_dir / f"{img_id}.jpg"
        if not img_path.exists():
            return torch.zeros(3, self.image_size, self.image_size), False

        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            img = np.array(img, dtype=np.float32) / 255.0
            img = (img - self.mean) / self.std
            img = np.transpose(img, (2, 0, 1))
            return torch.from_numpy(img), True
        except Exception:
            return torch.zeros(3, self.image_size, self.image_size), False


def extract_image_features_chunked(
    image_ids,
    image_dir,
    output_feat_path,
    output_mask_path,
    feature_dim=1280,
    batch_size=32,
    num_workers=2,
):
    """Extract image features with chunk-based disk writes to avoid OOM."""
    import torchvision.models as models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = len(image_ids)

    logger.info(f"Extracting features for {n:,} images → {output_feat_path}")

    # Pre-allocate memory-mapped output arrays on disk
    feat_mmap = np.lib.format.open_memmap(
        str(output_feat_path), mode="w+",
        dtype=np.float16, shape=(n, feature_dim),
    )
    mask_mmap = np.lib.format.open_memmap(
        str(output_mask_path), mode="w+",
        dtype=np.bool_, shape=(n,),
    )

    # Load model
    logger.info("Loading EfficientNet-B0...")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
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
    )

    offset = 0
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=True):
        for batch_imgs, batch_masks in tqdm(dataloader, desc="Extracting image features"):
            batch_imgs = batch_imgs.to(device, non_blocking=True)
            features = model(batch_imgs)

            bs = features.size(0)
            feat_mmap[offset:offset + bs] = features.cpu().numpy().astype(np.float16)
            mask_mmap[offset:offset + bs] = batch_masks.numpy()
            offset += bs

            # Flush periodically
            if offset % (batch_size * 100) == 0:
                feat_mmap.flush()
                mask_mmap.flush()

    feat_mmap.flush()
    mask_mmap.flush()

    del model, feat_mmap, mask_mmap
    torch.cuda.empty_cache()
    gc.collect()

    logger.info(f"Done! Features saved to {output_feat_path}")


if __name__ == "__main__":
    # Train images
    if not (FEATURES_DIR / "train_img_feat.npy").exists():
        logger.info("=== Extracting TRAIN image features ===")
        train_images = pd.read_csv(TRAIN_CSV, usecols=["image"], dtype=str)["image"].tolist()
        extract_image_features_chunked(
            train_images, TRAIN_IMAGES_DIR,
            FEATURES_DIR / "train_img_feat.npy",
            FEATURES_DIR / "train_img_feat_mask.npy",
            batch_size=32, num_workers=2,
        )
        del train_images
        gc.collect()
    else:
        logger.info("Train image features already exist, skipping.")

    # Test images
    if not (FEATURES_DIR / "test_img_feat.npy").exists():
        logger.info("=== Extracting TEST image features ===")
        test_images = pd.read_csv(TEST_CSV, usecols=["image"], dtype=str)["image"].tolist()
        extract_image_features_chunked(
            test_images, TEST_IMAGES_DIR,
            FEATURES_DIR / "test_img_feat.npy",
            FEATURES_DIR / "test_img_feat_mask.npy",
            batch_size=32, num_workers=2,
        )
    else:
        logger.info("Test image features already exist, skipping.")

    logger.info("All image features extracted!")
