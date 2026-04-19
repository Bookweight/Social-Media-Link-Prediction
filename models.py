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
# Strategy: LightGBM V2 (tuned hyperparameters)
# -----------------------------------------------------------------------

def strategy_lgbm_v3(
    train_features: pd.DataFrame,
    train_labels: np.ndarray,
    test_features: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Train LightGBM V3: balanced negatives, moderate regularization."""
    import lightgbm as lgb

    print("\n=== Strategy: LightGBM V3 ===")
    feature_cols = train_features.columns.tolist()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    auc_scores: list[float] = []
    test_preds = np.zeros(len(test_features))

    X_train = train_features[feature_cols].values
    y_train = train_labels
    X_test = test_features[feature_cols].values

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": 127,
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

        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols)
        dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=dtrain)

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

    # Print feature importance
    print("\n  Top-15 Feature Importance:")
    importance = model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
    for name, imp in feat_imp[:15]:
        print(f"    {name:30s} {imp:.1f}")

    sub = pd.DataFrame({"ID": test_df["ID"], "Label": test_preds})
    path = output_dir / "submission_lgbm_v3.csv"
    sub.to_csv(path, index=False)
    print(f"\n  Saved {path.name}")
    return sub





# -----------------------------------------------------------------------
# Strategy: Ensemble V3 (Heuristic + LightGBM)
# -----------------------------------------------------------------------

def strategy_ensemble_v3(
    test_df: pd.DataFrame,
    preds_heuristic: np.ndarray,
    preds_lgbm: np.ndarray,
    output_dir: Path,
) -> pd.DataFrame:
    """Combine Heuristic + LightGBM via weighted rank averaging."""
    from scipy.stats import rankdata

    print("\n=== Strategy: Ensemble V3 (Heuristic + LightGBM) ===")

    def _rank_normalize(arr: np.ndarray) -> np.ndarray:
        return rankdata(arr) / len(arr)

    r_h = _rank_normalize(preds_heuristic)
    r_l = _rank_normalize(preds_lgbm)
    ensemble = 0.3 * r_h + 0.7 * r_l

    sub = pd.DataFrame({"ID": test_df["ID"], "Label": ensemble})
    path = output_dir / "submission_ensemble_v3.csv"
    sub.to_csv(path, index=False)
    print(f"  Saved {path.name}")
    return sub
