import os
import gc
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# ============================================================================
# Paths
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(PROJECT_ROOT, "input")
TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output_lgbm")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 1. Load Data
# ============================================================================
print("=" * 60)
print("Loading data...")
print("=" * 60)

df = pd.read_csv(
    TRAIN_CSV,
    parse_dates=["activation_date"],
    dtype={
        "price": "float32",
        "item_seq_number": "int32",
        "image_top_1": "float32",
        "deal_probability": "float32",
    },
)
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")
print(f"  Target (deal_probability) stats:\n{df['deal_probability'].describe()}\n")

# ============================================================================
# 2. Feature Engineering
# ============================================================================
print("=" * 60)
print("Feature Engineering...")
print("=" * 60)

# --- Time-based features ---
df["day_of_week"] = df["activation_date"].dt.dayofweek.astype("int8")
df["day_of_month"] = df["activation_date"].dt.day.astype("int8")
df["week_of_year"] = df["activation_date"].dt.isocalendar().week.astype("int8")

# --- Text length features ---
df["title_len"] = df["title"].str.len().fillna(0).astype("int16")
df["desc_len"] = df["description"].str.len().fillna(0).astype("int32")
df["title_word_count"] = df["title"].str.split().str.len().fillna(0).astype("int16")
df["desc_word_count"] = df["description"].str.split().str.len().fillna(0).astype("int16")

# --- Price features ---
df["price_missing"] = df["price"].isna().astype("int8")
price_median = df["price"].median()
df["price"] = df["price"].fillna(price_median)
df["log_price"] = np.log1p(df["price"]).astype("float32")

# --- Categorical encoding ---
CATEGORICAL_COLS = [
    "region", "city", "parent_category_name", "category_name",
    "param_1", "param_2", "param_3", "user_type", "image_top_1",
]

label_encoders = {}
for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    df[col] = df[col].astype(str).fillna("missing")
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"  Encoded '{col}': {df[col].nunique()} categories")

# Save encoders and metadata for testing/prediction
import pickle
metadata = {
    "price_median": price_median,
    "label_encoders": label_encoders,
    "categorical_cols": CATEGORICAL_COLS,
}
metadata_path = os.path.join(OUTPUT_DIR, "lgbm_metadata.pkl")
with open(metadata_path, "wb") as f:
    pickle.dump(metadata, f)
print(f"  Saved metadata to {metadata_path}")

# --- Define final features ---
NUMERICAL_COLS = [
    "price", "log_price", "item_seq_number",
    "day_of_week", "day_of_month", "week_of_year",
    "title_len", "desc_len", "title_word_count", "desc_word_count",
    "price_missing",
]

FEATURE_COLS = CATEGORICAL_COLS + NUMERICAL_COLS
TARGET_COL = "deal_probability"

print(f"\n  Total features: {len(FEATURE_COLS)}")
print(f"  Categorical: {len(CATEGORICAL_COLS)}")
print(f"  Numerical:   {len(NUMERICAL_COLS)}")

# ============================================================================
# 3. Train / Validation Split
# ============================================================================
print("\n" + "=" * 60)
print("Splitting data into train & validation...")
print("=" * 60)

X = df[FEATURE_COLS]
y = df[TARGET_COL]

# Free memory
del df
gc.collect()

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42, shuffle=True
)

print(f"  Train set:      {X_train.shape[0]:,} samples")
print(f"  Validation set: {X_val.shape[0]:,} samples")

# ============================================================================
# 4. LightGBM Training
# ============================================================================
print("\n" + "=" * 60)
print("Training LightGBM model...")
print("=" * 60)

# Create LightGBM datasets
lgb_train = lgb.Dataset(
    X_train, label=y_train,
    categorical_feature=CATEGORICAL_COLS,
    free_raw_data=False,
)
lgb_val = lgb.Dataset(
    X_val, label=y_val,
    categorical_feature=CATEGORICAL_COLS,
    reference=lgb_train,
    free_raw_data=False,
)

# LightGBM parameters
params = {
    "objective": "regression",
    "metric": ["rmse", "mae"],
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 255,
    "max_depth": -1,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}

# Callback to record evaluation results
evals_result = {}

print(f"\n  Parameters:")
for k, v in params.items():
    print(f"    {k}: {v}")
print()

model = lgb.train(
    params,
    lgb_train,
    num_boost_round=2000,
    valid_sets=[lgb_train, lgb_val],
    valid_names=["train", "valid"],
    callbacks=[
        lgb.log_evaluation(period=100),
        lgb.early_stopping(stopping_rounds=100, verbose=True),
        lgb.record_evaluation(evals_result),
    ],
)

print(f"\n  Best iteration: {model.best_iteration}")
print(f"  Best valid RMSE: {model.best_score['valid']['rmse']:.5f}")

# ============================================================================
# 5. Evaluation
# ============================================================================
print("\n" + "=" * 60)
print("Evaluation Results")
print("=" * 60)

y_pred_train = model.predict(X_train, num_iteration=model.best_iteration)
y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

print(f"  Train RMSE: {rmse_train:.5f}")
print(f"  Valid RMSE: {rmse_val:.5f}")

# ============================================================================
# 6. Plot Learning Curves
# ============================================================================
print("\n" + "=" * 60)
print("Generating plots...")
print("=" * 60)

# Use a nice style
plt.style.use("seaborn-v0_8-darkgrid")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ---- Plot 1: RMSE Learning Curve ----
ax1 = axes[0]
train_rmse = evals_result["train"]["rmse"]
valid_rmse = evals_result["valid"]["rmse"]
iterations = range(1, len(train_rmse) + 1)

ax1.plot(iterations, train_rmse, label="Train RMSE", color="#2196F3", linewidth=1.5, alpha=0.9)
ax1.plot(iterations, valid_rmse, label="Validation RMSE", color="#F44336", linewidth=1.5, alpha=0.9)
ax1.axvline(x=model.best_iteration, color="#4CAF50", linestyle="--", alpha=0.7,
            label=f"Best Iteration ({model.best_iteration})")
ax1.set_xlabel("Boosting Rounds", fontsize=12)
ax1.set_ylabel("RMSE", fontsize=12)
ax1.set_title("Learning Curve — RMSE", fontsize=14, fontweight="bold")
ax1.legend(fontsize=11, loc="upper right")
ax1.grid(True, alpha=0.3)

# ---- Plot 2: MAE Learning Curve ----
ax2 = axes[1]
train_mae = evals_result["train"]["l1"]
valid_mae = evals_result["valid"]["l1"]

ax2.plot(iterations, train_mae, label="Train MAE", color="#9C27B0", linewidth=1.5, alpha=0.9)
ax2.plot(iterations, valid_mae, label="Validation MAE", color="#FF9800", linewidth=1.5, alpha=0.9)
ax2.axvline(x=model.best_iteration, color="#4CAF50", linestyle="--", alpha=0.7,
            label=f"Best Iteration ({model.best_iteration})")
ax2.set_xlabel("Boosting Rounds", fontsize=12)
ax2.set_ylabel("MAE", fontsize=12)
ax2.set_title("Learning Curve — MAE", fontsize=14, fontweight="bold")
ax2.legend(fontsize=11, loc="upper right")
ax2.grid(True, alpha=0.3)

plt.suptitle("LightGBM Training — Avito Demand Prediction", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()

plot_path = os.path.join(OUTPUT_DIR, "learning_curves.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved learning curves: {plot_path}")

# ---- Plot 3: Feature Importance ----
fig2, ax3 = plt.subplots(figsize=(10, 8))
lgb.plot_importance(model, max_num_features=20, importance_type="gain", ax=ax3)
ax3.set_title("Top 20 Feature Importance (Gain)", fontsize=14, fontweight="bold")
plt.tight_layout()

importance_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
fig2.savefig(importance_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved feature importance: {importance_path}")

# ---- Plot 4: Zoomed-in Loss Curves (last 50%) ----
fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6))

mid = len(train_rmse) // 2

ax4 = axes3[0]
ax4.plot(list(iterations)[mid:], train_rmse[mid:], label="Train RMSE", color="#2196F3", linewidth=1.5)
ax4.plot(list(iterations)[mid:], valid_rmse[mid:], label="Valid RMSE", color="#F44336", linewidth=1.5)
ax4.axvline(x=model.best_iteration, color="#4CAF50", linestyle="--", alpha=0.7,
            label=f"Best Iteration ({model.best_iteration})")
ax4.set_xlabel("Boosting Rounds", fontsize=12)
ax4.set_ylabel("RMSE", fontsize=12)
ax4.set_title("Zoomed-In RMSE (Last 50% Iterations)", fontsize=14, fontweight="bold")
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)

ax5 = axes3[1]
ax5.plot(list(iterations)[mid:], train_mae[mid:], label="Train MAE", color="#9C27B0", linewidth=1.5)
ax5.plot(list(iterations)[mid:], valid_mae[mid:], label="Valid MAE", color="#FF9800", linewidth=1.5)
ax5.axvline(x=model.best_iteration, color="#4CAF50", linestyle="--", alpha=0.7,
            label=f"Best Iteration ({model.best_iteration})")
ax5.set_xlabel("Boosting Rounds", fontsize=12)
ax5.set_ylabel("MAE", fontsize=12)
ax5.set_title("Zoomed-In MAE (Last 50% Iterations)", fontsize=14, fontweight="bold")
ax5.legend(fontsize=11)
ax5.grid(True, alpha=0.3)

plt.suptitle("LightGBM — Zoomed Loss Curves", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()

zoom_path = os.path.join(OUTPUT_DIR, "learning_curves_zoomed.png")
fig3.savefig(zoom_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved zoomed curves: {zoom_path}")

# ============================================================================
# 7. Save Model
# ============================================================================
model_path = os.path.join(OUTPUT_DIR, "lgbm_model.txt")
model.save_model(model_path, num_iteration=model.best_iteration)
print(f"  Saved model: {model_path}")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
