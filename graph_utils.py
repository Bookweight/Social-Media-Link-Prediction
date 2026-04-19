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

    edges = list(zip(train_df["Node1"].astype(int), train_df["Node2"].astype(int)))
    G = nx.DiGraph()
    G.add_edges_from(edges)
    G_undirected = G.to_undirected()
    print(f"  Nodes: {G.number_of_nodes():,}  Edges: {G.number_of_edges():,}")
    return train_df, test_df, G, G_undirected


def generate_negatives(
    G: nx.DiGraph,
    n_positive: int,
    neg_ratio: float = 2.0,
    rng: np.random.RandomState | None = None,
) -> list[tuple[int, int]]:
    """Generate negative samples with mixed strategy.

    V4 strategy: 100% fully random non-edges (matches Test Set prior)
    """
    if rng is None:
        rng = np.random.RandomState(SEED)

    n_total = int(n_positive * neg_ratio)
    nodes = list(G.nodes())
    edge_set = set(G.edges())
    n_2hop = 0  # Changed to 0 for V4
    n_rand = n_total - n_2hop

    negatives: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set(edge_set)

    def _add(u: int, v: int) -> bool:
        if u != v and (u, v) not in used:
            negatives.append((u, v))
            used.add((u, v))
            return True
        return False

    # --- 2-hop negatives (20%) ---
    print(f"  Generating {n_2hop:,} 2-hop negatives...")
    attempts = 0
    count_2hop = 0
    while count_2hop < n_2hop and attempts < n_2hop * 20:
        attempts += 1
        u = nodes[rng.randint(len(nodes))]
        successors_u = list(G.successors(u))
        if not successors_u:
            continue
        w = successors_u[rng.randint(len(successors_u))]
        successors_w = list(G.successors(w))
        if not successors_w:
            continue
        v = successors_w[rng.randint(len(successors_w))]
        if _add(u, v):
            count_2hop += 1

    n_rand += (n_2hop - count_2hop)
    print(f"    Got {count_2hop:,} 2-hop negatives")

    # --- Random negatives (80%) ---
    print(f"  Generating {n_rand:,} random negatives...")
    count_rand = 0
    while count_rand < n_rand:
        u = nodes[rng.randint(len(nodes))]
        v = nodes[rng.randint(len(nodes))]
        if _add(u, v):
            count_rand += 1

    print(f"  Total negatives: {len(negatives):,} (ratio {neg_ratio}:1)")
    return negatives


def precompute_graph_properties(
    G: nx.DiGraph, G_und: nx.Graph,
) -> tuple[dict[int, float], dict[int, int], dict[int, float], dict[int, float]]:
    """Precompute PageRank, Louvain, and HITS."""
    print("  Computing PageRank...")
    pagerank_dict = nx.pagerank(G, alpha=0.85, max_iter=100)

    print("  Computing communities (Louvain)...")
    communities = nx.community.louvain_communities(G_und, seed=SEED)
    node_to_comm: dict[int, int] = {}
    for comm_id, members in enumerate(communities):
        for node in members:
            node_to_comm[node] = comm_id

    print("  Computing HITS...")
    hubs, authorities = nx.hits(G, max_iter=100, normalized=True)

    return pagerank_dict, node_to_comm, hubs, authorities
