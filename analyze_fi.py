"""
Feature importance analysis for V4 link prediction.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import train_test_split

from graph_utils import load_data, generate_negatives, precompute_graph_properties
from features import compute_features

BASE_DIR = Path(__file__).parent
SEED = 42

def main():
    print("Loading data...")
    train_df, _, G, G_undirected = load_data(BASE_DIR)
    
    # 100% Random negatives, 1:1 ratio (V4 setting)
    print("Precomputing...")
    pagerank_dict, community_dict, hubs, authorities = precompute_graph_properties(G, G_undirected)
    
    print("Generating 100% random negatives (1:1)...")
    positive_pairs = list(zip(train_df["Node1"].astype(int), train_df["Node2"].astype(int)))
    rng = np.random.RandomState(SEED)
    
    # generate_negatives in V3 does 80% random, 20% 2-hop. 
    # For this analysis, let's just generate pure random manually so we don't need to change graph_utils.py yet.
    nodes = list(G.nodes())
    edge_set = set(G.edges())
    negatives = []
    used = set(edge_set)
    count = 0
    n_pos = len(positive_pairs)
    while count < n_pos:
        u = nodes[rng.randint(len(nodes))]
        v = nodes[rng.randint(len(nodes))]
        if u != v and (u, v) not in used:
            negatives.append((u, v))
            used.add((u, v))
            count += 1
            
    all_train_pairs = positive_pairs + negatives
    train_labels = np.array([1] * n_pos + [0] * n_pos)
    
    print("Computing features...")
    train_features = compute_features(
        all_train_pairs, G, G_undirected, pagerank_dict, community_dict, hubs, authorities,
    )
    
    feature_cols = train_features.columns.tolist()
    X = train_features.values
    y = train_labels
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=dtrain)
    
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "verbose": -1,
        "seed": SEED,
    }
    
    print("Training LightGBM...")
    model = lgb.train(
        params, dtrain, num_boost_round=500, valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    print("\nFeature Importances (Gain):")
    importance = model.feature_importance(importance_type="gain")
    total_gain = importance.sum()
    
    feat_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": importance,
        "percentage": importance / total_gain * 100
    }).sort_values(by="importance", ascending=False)
    
    feat_imp["cumulative"] = feat_imp["percentage"].cumsum()
    print(feat_imp.to_string(index=False))

if __name__ == "__main__":
    main()
