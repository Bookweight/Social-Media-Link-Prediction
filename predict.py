"""
Directed Graph Link Prediction V4
====================================
Improvements: 100% random negatives, 1:1 ratio, 
filtered features, conservative LightGBM parameters.

Usage:  uv run predict.py
Output: output/submission_lgbm_v4.csv, submission_ensemble_v4.csv
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from graph_utils import load_data, generate_negatives, precompute_graph_properties
from features import compute_features
from models import strategy_lgbm_v4, strategy_ensemble_v4

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
SEED = 42


def verify_submissions(output_dir: Path, sample_path: Path) -> None:
    """Assert submission CSVs match the expected format."""
    print("\n=== Verification ===")
    sample_sub = pd.read_csv(sample_path)
    for fname in output_dir.glob("submission_*_v4.csv"):
        sub = pd.read_csv(fname)
        assert list(sub.columns) == ["ID", "Label"], f"{fname.name}: column mismatch"
        assert len(sub) == len(sample_sub), f"{fname.name}: row count mismatch"
        assert (sub["ID"] == sample_sub["ID"]).all(), f"{fname.name}: ID mismatch"
        assert sub["Label"].between(0, 1).all(), f"{fname.name}: Label out of [0,1]"
        print(
            f"  {fname.name}: OK "
            f"(rows={len(sub)}, "
            f"label=[{sub['Label'].min():.4f}, {sub['Label'].max():.4f}])"
        )


def main() -> None:
    t0 = time.time()
    rng = np.random.RandomState(SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ---- Data ----
    train_df, test_df, G, G_undirected = load_data(BASE_DIR)

    # ---- Precompute ----
    print("\n=== Precomputing graph properties ===")
    pagerank_dict, community_dict, hubs, authorities = precompute_graph_properties(G, G_undirected)

    # ---- Negative Sampling (100% random, 1:1 ratio) ----
    print("\n=== Generating negative samples ===")
    positive_pairs = list(zip(train_df["Node1"].astype(int), train_df["Node2"].astype(int)))
    negative_pairs = generate_negatives(G, len(positive_pairs), neg_ratio=1.0, rng=rng)
    all_train_pairs = positive_pairs + negative_pairs
    train_labels = np.array([1] * len(positive_pairs) + [0] * len(negative_pairs))

    shuffle_idx = rng.permutation(len(all_train_pairs))
    all_train_pairs = [all_train_pairs[i] for i in shuffle_idx]
    train_labels = train_labels[shuffle_idx]

    test_pairs = list(zip(test_df["Node1"].astype(int), test_df["Node2"].astype(int)))

    # ---- Features (40+) ----
    print("\n=== Computing features for training pairs ===")
    train_features = compute_features(
        all_train_pairs, G, G_undirected, pagerank_dict, community_dict, hubs, authorities,
    )
    print("\n=== Computing features for test pairs ===")
    test_features = compute_features(
        test_pairs, G, G_undirected, pagerank_dict, community_dict, hubs, authorities,
    )

    # ---- LightGBM V4 ----
    sub_lgbm = strategy_lgbm_v4(train_features, train_labels, test_features, test_df, OUTPUT_DIR)

    # ---- Ensemble V4 (reuse existing V1 heuristic + LightGBM V4) ----
    heuristic_path = OUTPUT_DIR / "submission_heuristic_v2.csv"
    if heuristic_path.exists():
        preds_heuristic = pd.read_csv(heuristic_path)["Label"].values
        strategy_ensemble_v4(test_df, preds_heuristic, sub_lgbm["Label"].values, OUTPUT_DIR)
    else:
        # Fallback to the very original heuristic if _v2 doesn't exist
        heuristic_path = OUTPUT_DIR / "submission_heuristic.csv"
        if heuristic_path.exists():
            preds_heuristic = pd.read_csv(heuristic_path)["Label"].values
            strategy_ensemble_v4(test_df, preds_heuristic, sub_lgbm["Label"].values, OUTPUT_DIR)
        else:
            print("\n  Skipping ensemble (no existing heuristic submission found)")

    # ---- Verify ----
    verify_submissions(OUTPUT_DIR, BASE_DIR / "sample_submission.csv")

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
