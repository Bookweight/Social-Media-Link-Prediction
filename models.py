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
# Strategy: Ensemble (XGBoost + LightGBM V6)
# -----------------------------------------------------------------------

def strategy_ensemble_xgb_lgbm(
    test_df: pd.DataFrame,
    preds_xgb: np.ndarray,
    preds_lgbm: np.ndarray,
    output_dir: Path,
) -> pd.DataFrame:
    """Combine XGBoost and LightGBM via equal rank averaging."""
    from scipy.stats import rankdata

    print("\n=== Strategy: Ensemble (XGBoost + LightGBM) ===")

    def _rank_normalize(arr: np.ndarray) -> np.ndarray:
        return rankdata(arr) / len(arr)

    ensemble = 0.5 * _rank_normalize(preds_xgb) + 0.5 * _rank_normalize(preds_lgbm)

    sub = pd.DataFrame({"ID": test_df["ID"], "Label": ensemble})
    path = output_dir / "submission_ensemble_xgb_lgbm.csv"
    sub.to_csv(path, index=False)
    print(f"  Saved {path.name}")
    return sub


# -----------------------------------------------------------------------
# Strategy: LightGBM V6 (Legacy Parameters)
# -----------------------------------------------------------------------

def strategy_lgbm_v6(
    train_features: pd.DataFrame,
    train_labels: np.ndarray,
    test_features: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Train LightGBM on all features using robust V1 hyperparams."""
    import lightgbm as lgb
    print("\n=== Strategy: LightGBM V6 ===")
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
        "learning_rate": 0.05,
        "num_leaves": 63,
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
            num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        val_pred = model.predict(X_val)
        fold_auc = roc_auc_score(y_val, val_pred)
        auc_scores.append(fold_auc)
        print(f"  Fold {fold + 1} AUC: {fold_auc:.5f}  (best_iter={model.best_iteration})")
        test_preds += model.predict(X_test) / 5

    print(f"  Mean CV AUC: {np.mean(auc_scores):.5f}")
    sub = pd.DataFrame({"ID": test_df["ID"], "Label": test_preds})
    path = output_dir / "submission_lgbm_v6.csv"
    sub.to_csv(path, index=False)
    print(f"  Saved {path.name}")
    return sub


# -----------------------------------------------------------------------
# Strategy: XGBoost
# -----------------------------------------------------------------------

def strategy_xgboost(
    train_features: pd.DataFrame,
    train_labels: np.ndarray,
    test_features: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Train XGBoost model on all features."""
    import xgboost as xgb
    print("\n=== Strategy: XGBoost ===")
    feature_cols = train_features.columns.tolist()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    auc_scores: list[float] = []
    test_preds = np.zeros(len(test_features))

    X_train = train_features[feature_cols].values
    y_train = train_labels
    X_test = test_features[feature_cols].values

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "alpha": 0.1,
        "lambda": 0.1,
        "tree_method": "hist",
        "random_state": SEED,
        "n_jobs": -1,
    }

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_cols)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)
        dtest = xgb.DMatrix(X_test, feature_names=feature_cols)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        val_pred = model.predict(dval)
        fold_auc = roc_auc_score(y_val, val_pred)
        auc_scores.append(fold_auc)
        print(f"  Fold {fold + 1} AUC: {fold_auc:.5f}  (best_iter={model.best_iteration})")
        test_preds += model.predict(dtest) / 5

    print(f"  Mean CV AUC: {np.mean(auc_scores):.5f}")
    sub = pd.DataFrame({"ID": test_df["ID"], "Label": test_preds})
    path = output_dir / "submission_xgboost.csv"
    sub.to_csv(path, index=False)
    print(f"  Saved {path.name}")
    return sub

