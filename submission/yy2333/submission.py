# submission/yy2333/submission.py
# CS 6784 - Week of Aug 28 - Task 1: Linear Video Recommendation (Enhanced Version)
#
# !NOTE!:
# This code expects the KuaiRec dataset to be accessible under the default
# path "../../../data/KuaiRec 2.0/data" relative to the repository root.
# In my own experiments I mounted Google Drive in Colab and used that path, so that I could test the code
# to store the dataset (e.g., "/content/drive/MyDrive/KuaiRec 2.0/data").
# For grading / reproduction, please place the CSV files into:
#   <repo_root>/data/KuaiRec 2.0/data/
# so that the original path setting works out of the box.

import os, json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# ---------- Step 1: Local Path to KuaiRec CSV Files (Manually set) ----------
CSV_DIR = Path("/content/drive/MyDrive/KuaiRec 2.0/data")  # <- may need to change for grading

def get_kuairec_data(csv_dir=CSV_DIR):
    """
    Load KuaiRec data files from the given directory.
    Returns:
        - small_matrix: user-video interaction data
        - item_daily_feat: video-level features
        - user_features: user-level features
    """
    small_matrix    = pd.read_csv(csv_dir / "small_matrix.csv")
    user_features   = pd.read_csv(csv_dir / "user_features.csv")
    item_daily_feat = pd.read_csv(csv_dir / "item_daily_features.csv")
    return small_matrix, item_daily_feat, user_features

def extract_features_labels(small_matrix, item_daily_feat, user_features,
                            sample_n=10000, seed=0):
    """
    Join all data sources and construct the feature matrix and label vector.

    Enhancements:
    - Count features are log1p-transformed for stability
    - High-cardinality categorical feature 'music_id' is one-hot encoded (top-K only)

    Returns:
        X: feature matrix (numpy)
        y: label vector (watch ratio)
        feat_info: metadata for feature shape inspection
    """
	----- Statement of Changes Made -----	
	# Original: Simply merged features and used raw counts as input
    # Change: Added joins + log-transform for counts + one-hot for top-K music_id
    # Rationale: Improves numerical stability and captures categorical signals
								
    # Merge user and item features
    m = small_matrix.merge(user_features, how="left", on="user_id")
    m = m.merge(item_daily_feat, how="left", on=["video_id", "date"])
    m = m.dropna()

    # Subsample to 10k rows for efficiency
    if sample_n and len(m) > sample_n:
        m = m.sample(n=sample_n, random_state=seed)

    # Target variable: raw watch ratio
    y = m["watch_ratio"].to_numpy(dtype=np.float64)

    ----- Statement of Changes Made -----	
    # Original: Used raw count features directly
    # Change: Applied log(1 + x) transformation to counts
    # Rationale: Log transform reduces skew, stabilizes variance for regression

    # Numerical feature columns (log-transformed)
    num_cols = ["like_cnt", "comment_cnt", "follow_user_num_x", "friend_user_num"]
    num = m[num_cols].astype(np.float64)
    num = np.log1p(num)  # Apply log(1 + x) to stabilize variance

    ----- Statement of Changes Made -----
    # Original: Ignored music_id feature
    # Change: Encoded top-K frequent music_id values using one-hot encoding. That is, each popular music_id is converted into an independent feature column.
    # Rationale: Captures additional categorical signal while controlling dimensionality (richer feature representation)

    # One-hot encode top-K most frequent music_id
    use_music_onehot = True
    K = 100
    if use_music_onehot:
        topk = m["music_id"].value_counts().nlargest(K).index
        music = pd.get_dummies(m["music_id"].where(m["music_id"].isin(topk)),
                               prefix="music", dtype=np.uint8)
        X = np.hstack([num.to_numpy(), music.to_numpy(dtype=np.float64)])
        feat_info = {
            "num_cols": num_cols,
            "music_topk": int(K),
            "onehot_cols": int(music.shape[1])
        }
    else:
        X = num.to_numpy()
        feat_info = {
            "num_cols": num_cols,
            "music_topk": 0,
            "onehot_cols": 0
        }

    return X, y, feat_info

# ---------- Step 2: Label Transformations (Logit) ----------

def logit(x, eps=1e-6):
    """Convert probabilities in (0,1) to real values (log-odds)"""
    x = np.clip(x, eps, 1 - eps)
    return np.log(x / (1 - x))

def inv_logit(z):
    """Convert log-odds back to probabilities"""
    return 1 / (1 + np.exp(-z))

# ---------- Step 3: Train Ridge Model with Optional Label Transform ----------

def train_once(X, y, use_logit=False, seed=0):
    """
    Train a ridge regression model with optional logit-transformed labels.
    Uses 90/10 train-validation split with standardized features.

    Args:
        X: Feature matrix
        y: Label vector
        use_logit: whether to apply logit transformation to labels
        seed: random seed

    Returns:
        metrics: dict of alpha, RMSE, MAE
        scaler: fitted StandardScaler object
        ridge: trained RidgeCV model
    """
	
	----- Statement of Changes Made -----
	# Original: Used raw watch ratio as regression target
    # Change: Added optional logit(y) transformation before training
    # Rationale: Helps linear model fit skewed label distributions better

    y_t = logit(y) if use_logit else y.copy()

    # Random train/validation split
    X_tr, X_va, y_tr, y_va = train_test_split(X, y_t, test_size=0.1, random_state=seed)

    # Feature normalization
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)

    # Ridge regression with 5-fold CV to tune alpha
    alphas = np.logspace(-6, 2, 20)
    ridge = RidgeCV(alphas=alphas, cv=5, fit_intercept=True)
    ridge.fit(X_tr_s, y_tr)

    # Predict and evaluate
    yhat_va_t = ridge.predict(X_va_s)
    yhat_va = inv_logit(yhat_va_t) if use_logit else yhat_va_t
    y_va_true = inv_logit(y_va) if use_logit else y_va

    mse  = mean_squared_error(y_va_true, yhat_va)
    rmse = np.sqrt(mse)

    mae  = mean_absolute_error(y_va_true, yhat_va)

    return {
        "use_logit": use_logit,
        "alpha": float(ridge.alpha_),
        "val_rmse": float(rmse),
        "val_mae": float(mae),
        "scaler_mean_shape": list(scaler.mean_.shape)
    }, scaler, ridge

# ---------- Step 4: Main Execution ----------

def main():
	
	----- Statement of Changes Made -----
	# Original: Only trained one simple model
    # Change: Trained two models (logit vs. no-logit) and selected better
    # Rationale: Empirical comparison shows logit improves performance

    # 1. Load data
    sm, idf, uf = get_kuairec_data(CSV_DIR)
    X, y, feat_info = extract_features_labels(sm, idf, uf, sample_n=10000, seed=0)

    # 2. Train two versions: with and without logit-transformed labels
    m0, _, _ = train_once(X, y, use_logit=False, seed=0)
    m1, _, _ = train_once(X, y, use_logit=True,  seed=0)

    # 3. Choose better result
    final = m1 if m1["val_rmse"] <= m0["val_rmse"] else m0
    final["feat_info"] = feat_info

    # Print both for comparison
    print("[HW1] no-logit  =", m0)
    print("[HW1] yes-logit =", m1)
    print("[HW1] FINAL     =", final)

    ----- Statement of Changes Made -----
    # Original: No output saved
    # Change: Saved selected model’s metrics as JSON
    # Rationale: Easier to inspect results, useful for logging / PRs

    # 4. Save metrics to file (optional but helpful for PR)
    out_path = Path("metrics.json")  # Avoid __file__ issue in Colab
    with open(out_path, "w", encoding="utf-8") as f:
      json.dump(final, f, indent=2, ensure_ascii=False)
    print(f"[HW1] metrics saved → {out_path}")
    
    # 5. Plot actual vs predicted (for final model) as shown in the original code

    # Run one more time to get actual predictions
    _, scaler, model = train_once(X, y, use_logit=final["use_logit"], seed=0)

    # Reconstruct val split
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.1, random_state=0)
    X_va_s = scaler.transform(X_va)
    yhat_va = model.predict(X_va_s)
    if final["use_logit"]:
        yhat_va = inv_logit(yhat_va)
        y_va = inv_logit(y_va)

    # Plot
    plt.figure(figsize=(6, 6))
    plt.plot(y_va, yhat_va, '.', alpha=0.4)
    plt.xlabel('Actual watch ratio')
    plt.ylabel('Predicted watch ratio')
    plt.title('Actual vs. Predicted Watch Ratio (Final Model)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
