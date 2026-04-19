"""Model strategies for link prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

SEED = 42


# -----------------------------------------------------------------------
# Strategy 1: Heuristic Baseline (Adamic-Adar)
# -----------------------------------------------------------------------

def strategy_heuristic(
    test_df: pd.DataFrame,
    test_features: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Predict using normalized Adamic-Adar score (no ML training)."""
    print("\n=== Strategy 1: Heuristic (Adamic-Adar) ===")
    scores = test_features["adamic_adar"].values.copy()
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        scores = (scores - s_min) / (s_max - s_min)
    else:
        scores = np.full_like(scores, 0.5)

    sub = pd.DataFrame({"ID": test_df["ID"], "Label": scores})
    path = output_dir / "submission_heuristic.csv"
    sub.to_csv(path, index=False)
    print(f"  Saved {path.name}")
    return sub


# -----------------------------------------------------------------------
# Strategy 2: LightGBM with Engineered Graph Features
# -----------------------------------------------------------------------

def strategy_lgbm(
    train_features: pd.DataFrame,
    train_labels: np.ndarray,
    test_features: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Train LightGBM binary classifier with 5-fold CV."""
    import lightgbm as lgb

    print("\n=== Strategy 2: LightGBM ===")
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
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "n_jobs": -1,
        "verbose": -1,
        "seed": SEED,
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
        print(f"  Fold {fold + 1} AUC: {fold_auc:.5f}")
        test_preds += model.predict(X_test) / 5

    print(f"  Mean CV AUC: {np.mean(auc_scores):.5f}")

    sub = pd.DataFrame({"ID": test_df["ID"], "Label": test_preds})
    path = output_dir / "submission_lgbm.csv"
    sub.to_csv(path, index=False)
    print(f"  Saved {path.name}")
    return sub


# -----------------------------------------------------------------------
# Strategy 3: Node2Vec + Logistic Regression
# -----------------------------------------------------------------------

def strategy_node2vec(
    G: nx.DiGraph,
    train_pairs: list[tuple[int, int]],
    train_labels: np.ndarray,
    test_pairs: list[tuple[int, int]],
    test_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Learn Node2Vec embeddings, then classify with LogReg."""
    from node2vec import Node2Vec

    print("\n=== Strategy 3: Node2Vec + LogReg ===")
    G_und = G.to_undirected()

    print("  Running Node2Vec random walks...")
    n2v = Node2Vec(
        G_und, dimensions=64, walk_length=30, num_walks=20,
        workers=1, p=1, q=0.5, seed=SEED, quiet=True,
    )
    print("  Training Word2Vec embeddings...")
    model = n2v.fit(window=10, min_count=1, batch_words=4, seed=SEED)

    def _get_embedding(node_id: int) -> np.ndarray:
        try:
            return model.wv[str(node_id)]
        except KeyError:
            return np.zeros(64)

    def _pair_features(u: int, v: int) -> np.ndarray:
        emb_u, emb_v = _get_embedding(u), _get_embedding(v)
        hadamard = emb_u * emb_v
        cosine = np.dot(emb_u, emb_v) / (np.linalg.norm(emb_u) * np.linalg.norm(emb_v) + 1e-8)
        l1 = np.abs(emb_u - emb_v)
        l2_dist = np.linalg.norm(emb_u - emb_v)
        return np.concatenate([hadamard, l1, [cosine, l2_dist]])

    print("  Computing pair features for train...")
    X_train = np.array([_pair_features(u, v) for u, v in train_pairs])
    print("  Computing pair features for test...")
    X_test = np.array([_pair_features(u, v) for u, v in test_pairs])

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    auc_scores: list[float] = []
    test_preds = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, train_labels)):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = train_labels[tr_idx], train_labels[val_idx]

        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=SEED)
        clf.fit(X_tr, y_tr)

        val_pred = clf.predict_proba(X_val)[:, 1]
        fold_auc = roc_auc_score(y_val, val_pred)
        auc_scores.append(fold_auc)
        print(f"  Fold {fold + 1} AUC: {fold_auc:.5f}")
        test_preds += clf.predict_proba(X_test)[:, 1] / 5

    print(f"  Mean CV AUC: {np.mean(auc_scores):.5f}")

    sub = pd.DataFrame({"ID": test_df["ID"], "Label": test_preds})
    path = output_dir / "submission_node2vec.csv"
    sub.to_csv(path, index=False)
    print(f"  Saved {path.name}")
    return sub


# -----------------------------------------------------------------------
# Strategy 4: Ensemble (Rank Average)
# -----------------------------------------------------------------------

def strategy_ensemble(
    test_df: pd.DataFrame,
    preds_heuristic: np.ndarray,
    preds_lgbm: np.ndarray,
    preds_n2v: np.ndarray,
    output_dir: Path,
) -> pd.DataFrame:
    """Combine predictions via weighted rank averaging."""
    from scipy.stats import rankdata

    print("\n=== Strategy 4: Ensemble (Rank Average) ===")

    def _rank_normalize(arr: np.ndarray) -> np.ndarray:
        return rankdata(arr) / len(arr)

    r1 = _rank_normalize(preds_heuristic)
    r2 = _rank_normalize(preds_lgbm)
    r3 = _rank_normalize(preds_n2v)
    ensemble = 0.2 * r1 + 0.5 * r2 + 0.3 * r3

    sub = pd.DataFrame({"ID": test_df["ID"], "Label": ensemble})
    path = output_dir / "submission_ensemble.csv"
    sub.to_csv(path, index=False)
    print(f"  Saved {path.name}")
    return sub
