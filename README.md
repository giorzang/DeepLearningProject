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
