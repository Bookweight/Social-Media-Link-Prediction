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


def generate_hard_negatives(
    G: nx.DiGraph,
    n_samples: int,
    community_dict: dict[int, int],
    rng: np.random.RandomState | None = None,
) -> list[tuple[int, int]]:
    """Generate hard negative samples that resemble test-set distribution.

    Strategy:
      - 50% from 2-hop neighbors (u->w->? but no u->v edge)
      - 30% from same Louvain community (no direct edge)
      - 20% fully random (no direct edge)
    """
    if rng is None:
        rng = np.random.RandomState(SEED)

    nodes = list(G.nodes())
    edge_set = set(G.edges())
    n_2hop = int(n_samples * 0.50)
    n_comm = int(n_samples * 0.30)
    n_rand = n_samples - n_2hop - n_comm

    # Build community -> node list mapping
    comm_to_nodes: dict[int, list[int]] = {}
    for node, comm_id in community_dict.items():
        comm_to_nodes.setdefault(comm_id, []).append(node)

    negatives: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set(edge_set)

    def _add(u: int, v: int) -> bool:
        if u != v and (u, v) not in used:
            negatives.append((u, v))
            used.add((u, v))
            return True
        return False

    # --- 2-hop negatives ---
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

    # Fill remaining with random if 2-hop was insufficient
    n_rand += (n_2hop - count_2hop)
    print(f"    Got {count_2hop:,} 2-hop negatives")

    # --- Same-community negatives ---
    print(f"  Generating {n_comm:,} same-community negatives...")
    count_comm = 0
    attempts = 0
    while count_comm < n_comm and attempts < n_comm * 20:
        attempts += 1
        u = nodes[rng.randint(len(nodes))]
        comm = community_dict.get(u, -1)
        members = comm_to_nodes.get(comm, [])
        if len(members) < 2:
            continue
        v = members[rng.randint(len(members))]
        if _add(u, v):
            count_comm += 1

    n_rand += (n_comm - count_comm)
    print(f"    Got {count_comm:,} same-community negatives")

    # --- Random negatives ---
    print(f"  Generating {n_rand:,} random negatives...")
    count_rand = 0
    while count_rand < n_rand:
        u = nodes[rng.randint(len(nodes))]
        v = nodes[rng.randint(len(nodes))]
        if _add(u, v):
            count_rand += 1

    print(f"  Total negatives: {len(negatives):,}")
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
