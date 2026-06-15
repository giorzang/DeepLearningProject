# 📊 Avito Demand Prediction — Results Report

## Training Results

| Epoch | Train RMSE | Val RMSE | Status |
|:-----:|:----------:|:--------:|:------:|
| 1 | 0.24047 | 0.22764 | ★ Best |
| 2 | 0.22863 | 0.22512 | ★ Best |
| 3 | 0.22525 | 0.22347 | ★ Best |
| 4 | 0.22317 | 0.22237 | ★ Best |
| 5 | 0.22117 | 0.22149 | ★ Best |
| 6 | 0.21885 | **0.22114** | **★ Best** |
| 7 | 0.21641 | 0.22125 | — |
| 8 | 0.21391 | 0.22152 | — |
| 9 | 0.21190 | 0.22144 | ⚠ Early Stop |

> [!IMPORTANT]
> **Best Validation RMSE: 0.22114** (Epoch 6)
> Early stopping triggered at epoch 9 (patience=3)

## Model Architecture Summary

| Component | Details |
|---|---|
| **Framework** | PyTorch 2.12 + CUDA |
| **Tabular Tower** | Entity embeddings (9 categorical) + 10 numerical → 256 → 128 with GhostBatchNorm |
| **Text Tower** | Frozen `paraphrase-multilingual-MiniLM-L12-v2` (384-dim) → projection → attention pool |
| **Image Tower** | Frozen EfficientNet-B0 (1280-dim) → 256 → 128 with learned no-image embedding |
| **Fusion** | Cross-Modal Gated Attention (3 modalities → 128-dim) |
| **Head** | Residual MLP (128 → 64 → 32 → 1) + Sigmoid |
| **Total Parameters** | 1,007,068 (~4 MB) |
| **Training Time** | ~17 min total (9 epochs × ~102s/epoch) |

## Submission Statistics

| Metric | Value |
|---|---|
| Predictions | 508,438 |
| Mean | 0.1504 |
| Std | 0.1383 |
| Min | 0.0184 |
| 25th percentile | 0.0438 |
| Median | 0.1021 |
| 75th percentile | 0.2129 |
| Max | 0.7607 |

## File Structure

```
BTL/
├── input/                          # Raw data
│   ├── train.csv (953 MB)
│   ├── test.csv (331 MB)
│   ├── train_jpg/ (1.4M images)
│   └── test_jpg/
├── src/
│   ├── config.py                   # Hyperparameters & paths
│   ├── preprocess.py               # Feature extraction pipeline
│   ├── dataset.py                  # Memory-mapped PyTorch Dataset
│   ├── model.py                    # Three-tower model definition
│   ├── train.py                    # Training loop
│   └── predict.py                  # Submission generation
├── extract_images.py               # OOM-safe image feature extraction
├── run.py                          # Master entry point
├── features/ (7.9 GB)             # Pre-computed features
│   ├── train_cat.npy / test_cat.npy
│   ├── train_num.npy / test_num.npy
│   ├── train_title_emb.npy / test_title_emb.npy
│   ├── train_desc_emb.npy / test_desc_emb.npy
│   ├── train_img_feat.npy / test_img_feat.npy
│   └── vocab_sizes.json
├── checkpoints/
│   └── best_model.pt
├── submission.csv                  # Final Kaggle submission
└── requirements.txt
```

## How to Reproduce

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Preprocess (extract text & tabular features)
python run.py preprocess

# 3. Extract image features (OOM-safe version)
python extract_images.py

# 4. Train the model
python run.py train

# 5. Generate submission
python run.py predict
```
