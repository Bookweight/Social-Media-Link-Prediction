"""Model strategies for link prediction (V2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

SEED = 42


# -----------------------------------------------------------------------
# Strategy: LightGBM Subset (V5 Minimalist)
# -----------------------------------------------------------------------

def strategy_lgbm_subset(
    train_features: pd.DataFrame,
    train_labels: np.ndarray,
    test_features: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_subset: list[str],
    output_path: Path,
) -> pd.DataFrame:
    """Train LightGBM on a specific subset of features."""
    import lightgbm as lgb

    print(f"\n=== Strategy: LightGBM (Features: {len(feature_subset)}) ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    auc_scores: list[float] = []
    test_preds = np.zeros(len(test_features))

    X_train = train_features[feature_subset].values
    y_train = train_labels
    X_test = test_features[feature_subset].values

    leaves = 63 if len(feature_subset) > 2 else 31
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.01,
        "num_leaves": leaves,
        "max_depth": -1,
        "min_child_samples": 30,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "n_jobs": -1,
        "verbose": -1,
        "seed": SEED,
        "bagging_freq": 5,
    }

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_subset)
        dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_subset, reference=dtrain)

        model = lgb.train(
            params, dtrain,
            num_boost_round=5000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(150, verbose=False)],
        )

        val_pred = model.predict(X_val)
        fold_auc = roc_auc_score(y_val, val_pred)
        auc_scores.append(fold_auc)
        print(f"  Fold {fold + 1} AUC: {fold_auc:.5f}  (best_iter={model.best_iteration})")
        test_preds += model.predict(X_test) / 5

    mean_auc = np.mean(auc_scores)
    print(f"  Mean CV AUC: {mean_auc:.5f}")

    sub = pd.DataFrame({"ID": test_df["ID"], "Label": test_preds})
    sub.to_csv(output_path, index=False)
    print(f"  Saved {output_path.name}")
    return sub
