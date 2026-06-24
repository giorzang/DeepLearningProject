import os
import gc
import pickle
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ============================================================================
# Paths
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(PROJECT_ROOT, "input")
TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output_lgbm")
METADATA_PATH = os.path.join(OUTPUT_DIR, "lgbm_metadata.pkl")
MODEL_PATH = os.path.join(OUTPUT_DIR, "lgbm_model.txt")
SUBMISSION_CSV = os.path.join(OUTPUT_DIR, "submission_lgbm.csv")

# ============================================================================
# 1. Load Model and Metadata
# ============================================================================
print("=" * 60)
print("Loading model and metadata...")
print("=" * 60)

if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
    raise FileNotFoundError("Model or metadata file not found. Please run train_lgbm.py first.")

with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

price_median = metadata["price_median"]
label_encoders = metadata["label_encoders"]
CATEGORICAL_COLS = metadata["categorical_cols"]

# Load LightGBM model
model = lgb.Booster(model_file=MODEL_PATH)
print("  Successfully loaded model and metadata.")

# ============================================================================
# 2. Load Test Data
# ============================================================================
print("\n" + "=" * 60)
print("Loading test data...")
print("=" * 60)

test_df = pd.read_csv(
    TEST_CSV,
    parse_dates=["activation_date"],
    dtype={
        "price": "float32",
        "item_seq_number": "int32",
        "image_top_1": "float32",
    },
)
print(f"  Test shape: {test_df.shape}")

# Save item_id for submission
submission = pd.DataFrame({"item_id": test_df["item_id"]})

# ============================================================================
# 3. Feature Engineering on Test Data
# ============================================================================
print("\n" + "=" * 60)
print("Feature Engineering on test data...")
print("=" * 60)

# --- Time-based features ---
test_df["day_of_week"] = test_df["activation_date"].dt.dayofweek.astype("int8")
test_df["day_of_month"] = test_df["activation_date"].dt.day.astype("int8")
test_df["week_of_year"] = test_df["activation_date"].dt.isocalendar().week.astype("int8")

# --- Text length features ---
test_df["title_len"] = test_df["title"].str.len().fillna(0).astype("int16")
test_df["desc_len"] = test_df["description"].str.len().fillna(0).astype("int32")
test_df["title_word_count"] = test_df["title"].str.split().str.len().fillna(0).astype("int16")
test_df["desc_word_count"] = test_df["description"].str.split().str.len().fillna(0).astype("int16")

# --- Price features ---
test_df["price_missing"] = test_df["price"].isna().astype("int8")
test_df["price"] = test_df["price"].fillna(price_median)
test_df["log_price"] = np.log1p(test_df["price"]).astype("float32")

# --- Categorical encoding (mapping safely for unseen categories) ---
for col in CATEGORICAL_COLS:
    le = label_encoders[col]
    test_df[col] = test_df[col].astype(str).fillna("missing")
    
    # Create mapping dict: class -> code
    mapping = {c: i for i, c in enumerate(le.classes_)}
    
    # Map unseen categories to -1 (LightGBM handles negative numbers as missing/unknown values automatically)
    test_df[col] = test_df[col].map(mapping).fillna(-1).astype("int32")
    print(f"  Encoded '{col}' (handled unseen/new values)")

# --- Define final features ---
NUMERICAL_COLS = [
    "price", "log_price", "item_seq_number",
    "day_of_week", "day_of_month", "week_of_year",
    "title_len", "desc_len", "title_word_count", "desc_word_count",
    "price_missing",
]

FEATURE_COLS = CATEGORICAL_COLS + NUMERICAL_COLS
X_test = test_df[FEATURE_COLS]

print(f"\n  Final features shape: {X_test.shape}")

# Free memory of original test_df to avoid Out of Memory
del test_df
gc.collect()

# ============================================================================
# 4. Predict
# ============================================================================
print("\n" + "=" * 60)
print("Predicting deal probability...")
print("=" * 60)

preds = model.predict(X_test, num_iteration=model.best_iteration)

# Deal probability must be bounded between 0.0 and 1.0
preds = np.clip(preds, 0.0, 1.0)

submission["deal_probability"] = preds

# ============================================================================
# 5. Export Submission
# ============================================================================
print("\n" + "=" * 60)
print(f"Saving submission file to {SUBMISSION_CSV}...")
print("=" * 60)

submission.to_csv(SUBMISSION_CSV, index=False)
print("  File saved successfully!")

# Show submission stats
print(f"\n  Submission stats:\n{submission['deal_probability'].describe()}")
print(f"\n  Submission sample:\n{submission.head()}")

print("\n" + "=" * 60)
print("Done predicting!")
print("=" * 60)
