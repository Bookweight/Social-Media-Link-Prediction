"""Graph construction and utility functions for link prediction."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42


def load_data(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, nx.DiGraph, nx.Graph]:
    """Load CSV data and construct directed / undirected graphs."""
    print("=== Loading data ===")
    train_df = pd.read_csv(base_dir / "train.csv")
    test_df = pd.read_csv(base_dir / "test.csv")
    print(f"  Train edges: {len(train_df):,}")
    print(f"  Test pairs:  {len(test_df):,}")

    # Build directed graph (vectorized, much faster than iterrows)
    edges = list(zip(train_df["Node1"].astype(int), train_df["Node2"].astype(int)))
    G = nx.DiGraph()
    G.add_edges_from(edges)

    G_undirected = G.to_undirected()
    print(f"  Nodes: {G.number_of_nodes():,}  Edges: {G.number_of_edges():,}")
    return train_df, test_df, G, G_undirected


def generate_negative_samples(
    G: nx.DiGraph, n_samples: int, rng: np.random.RandomState | None = None,
) -> list[tuple[int, int]]:
    """Generate random non-existent edges as negative samples."""
    if rng is None:
        rng = np.random.RandomState(SEED)
    nodes = list(G.nodes())
    edge_set = set(G.edges())
    negatives: list[tuple[int, int]] = []
    while len(negatives) < n_samples:
        u = nodes[rng.randint(len(nodes))]
        v = nodes[rng.randint(len(nodes))]
        if u != v and (u, v) not in edge_set:
            negatives.append((u, v))
            edge_set.add((u, v))
    return negatives


def precompute_graph_properties(
    G: nx.DiGraph, G_und: nx.Graph,
) -> tuple[dict[int, float], dict[int, int]]:
    """Precompute PageRank and Louvain community assignments."""
    print("  Computing PageRank...")
    pagerank_dict = nx.pagerank(G, alpha=0.85, max_iter=100)

    print("  Computing communities (Louvain)...")
    communities = nx.community.louvain_communities(G_und, seed=SEED)
    node_to_comm: dict[int, int] = {}
    for comm_id, members in enumerate(communities):
        for node in members:
            node_to_comm[node] = comm_id

    return pagerank_dict, node_to_comm
