"""Feature engineering for link prediction on directed graphs."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def compute_features(
    pairs: list[tuple[int, int]],
    G: nx.DiGraph,
    G_und: nx.Graph,
    pagerank_dict: dict[int, float],
    community_dict: dict[int, int],
) -> pd.DataFrame:
    """Compute 25+ graph features for each (u, v) pair.

    Feature groups:
      - Degree (6): out/in/total degree for u and v
      - Reciprocity (1): whether reverse edge v->u exists
      - Common Neighbors (5): undirected CN, directed variants (out-out, in-in, out-in, in-out)
      - Heuristic (4): Jaccard, Adamic-Adar, Resource Allocation, Preferential Attachment
      - PageRank (4): PR(u), PR(v), diff, product
      - Community (1): same Louvain community
      - Ratio (3): out/in ratio, neighbor overlap ratio
    """
    features = []
    edge_set = set(G.edges())

    for u, v in pairs:
        feat: dict[str, float] = {}

        # -- Degree --
        feat["out_deg_u"] = G.out_degree(u) if G.has_node(u) else 0
        feat["in_deg_u"] = G.in_degree(u) if G.has_node(u) else 0
        feat["out_deg_v"] = G.out_degree(v) if G.has_node(v) else 0
        feat["in_deg_v"] = G.in_degree(v) if G.has_node(v) else 0
        feat["total_deg_u"] = feat["out_deg_u"] + feat["in_deg_u"]
        feat["total_deg_v"] = feat["out_deg_v"] + feat["in_deg_v"]

        # -- Reciprocity --
        feat["reciprocal"] = 1 if (v, u) in edge_set else 0

        # -- Common Neighbors (undirected) --
        if G_und.has_node(u) and G_und.has_node(v):
            cn = set(G_und.neighbors(u)) & set(G_und.neighbors(v))
            feat["common_neighbors"] = len(cn)
        else:
            cn = set()
            feat["common_neighbors"] = 0

        # -- Common Neighbors (directed) --
        succ_u = set(G.successors(u)) if G.has_node(u) else set()
        succ_v = set(G.successors(v)) if G.has_node(v) else set()
        pred_u = set(G.predecessors(u)) if G.has_node(u) else set()
        pred_v = set(G.predecessors(v)) if G.has_node(v) else set()

        feat["common_out"] = len(succ_u & succ_v)
        feat["common_in"] = len(pred_u & pred_v)
        feat["u_out_v_in"] = len(succ_u & pred_v)   # two-hop u->w->v
        feat["u_in_v_out"] = len(pred_u & succ_v)

        # -- Jaccard (undirected) --
        union_size = (
            len(set(G_und.neighbors(u)) | set(G_und.neighbors(v)))
            if (G_und.has_node(u) and G_und.has_node(v))
            else 0
        )
        feat["jaccard"] = feat["common_neighbors"] / max(union_size, 1)

        # -- Adamic-Adar (undirected) --
        aa = 0.0
        for w in cn:
            deg_w = G_und.degree(w)
            if deg_w > 1:
                aa += 1.0 / np.log(deg_w)
        feat["adamic_adar"] = aa

        # -- Resource Allocation --
        ra = 0.0
        for w in cn:
            deg_w = G_und.degree(w)
            if deg_w > 0:
                ra += 1.0 / deg_w
        feat["resource_allocation"] = ra

        # -- Preferential Attachment --
        feat["pref_attach"] = feat["total_deg_u"] * feat["total_deg_v"]

        # -- PageRank --
        feat["pagerank_u"] = pagerank_dict.get(u, 0.0)
        feat["pagerank_v"] = pagerank_dict.get(v, 0.0)
        feat["pagerank_diff"] = abs(feat["pagerank_u"] - feat["pagerank_v"])
        feat["pagerank_product"] = feat["pagerank_u"] * feat["pagerank_v"]

        # -- Community --
        comm_u = community_dict.get(u, -1)
        comm_v = community_dict.get(v, -2)
        feat["same_community"] = 1 if comm_u == comm_v else 0

        # -- Degree ratios --
        feat["out_in_ratio_u"] = feat["out_deg_u"] / max(feat["in_deg_u"], 1)
        feat["out_in_ratio_v"] = feat["out_deg_v"] / max(feat["in_deg_v"], 1)

        # -- Neighbor overlap --
        feat["neighbor_overlap_ratio"] = feat["common_neighbors"] / max(
            min(feat["total_deg_u"], feat["total_deg_v"]), 1
        )

        features.append(feat)

    return pd.DataFrame(features)
