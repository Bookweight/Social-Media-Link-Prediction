"""Feature engineering for link prediction on directed graphs (V2)."""

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
    hubs: dict[int, float],
    authorities: dict[int, float],
) -> pd.DataFrame:
    """Compute 40+ graph features for each (u, v) pair.

    V2 additions: HITS, directed Jaccard/AA, clustering coefficient,
    log-degrees, CN ratios, reciprocity rates, 3-hop path approximation.
    """
    features = []
    edge_set = set(G.edges())

    # Precompute clustering coefficients (undirected)
    clustering = nx.clustering(G_und)

    # Precompute reciprocity rates per node
    reciprocity_rate: dict[int, float] = {}
    for node in G.nodes():
        out_nbrs = set(G.successors(node))
        if len(out_nbrs) == 0:
            reciprocity_rate[node] = 0.0
        else:
            mutual = sum(1 for v in out_nbrs if (v, node) in edge_set)
            reciprocity_rate[node] = mutual / len(out_nbrs)

    for u, v in pairs:
        feat: dict[str, float] = {}

        u_in_graph = G.has_node(u)
        v_in_graph = G.has_node(v)

        # -- Degree (6) --
        out_u = G.out_degree(u) if u_in_graph else 0
        in_u = G.in_degree(u) if u_in_graph else 0
        out_v = G.out_degree(v) if v_in_graph else 0
        in_v = G.in_degree(v) if v_in_graph else 0
        total_u = out_u + in_u
        total_v = out_v + in_v

        feat["out_deg_u"] = out_u
        feat["in_deg_u"] = in_u
        feat["out_deg_v"] = out_v
        feat["in_deg_v"] = in_v
        feat["total_deg_u"] = total_u
        feat["total_deg_v"] = total_v

        # -- Log Degree (4) --
        feat["log_out_u"] = np.log1p(out_u)
        feat["log_in_v"] = np.log1p(in_v)
        feat["log_total_u"] = np.log1p(total_u)
        feat["log_total_v"] = np.log1p(total_v)

        # -- Reciprocity (1) --
        feat["reciprocal"] = 1 if (v, u) in edge_set else 0

        # -- Neighbor sets --
        succ_u = set(G.successors(u)) if u_in_graph else set()
        succ_v = set(G.successors(v)) if v_in_graph else set()
        pred_u = set(G.predecessors(u)) if u_in_graph else set()
        pred_v = set(G.predecessors(v)) if v_in_graph else set()

        # -- Common Neighbors undirected (1) --
        if G_und.has_node(u) and G_und.has_node(v):
            nbr_u = set(G_und.neighbors(u))
            nbr_v = set(G_und.neighbors(v))
            cn = nbr_u & nbr_v
            cn_count = len(cn)
            union_size = len(nbr_u | nbr_v)
        else:
            nbr_u, nbr_v = set(), set()
            cn = set()
            cn_count = 0
            union_size = 0
        feat["common_neighbors"] = cn_count

        # -- Common Neighbors directed (4) --
        feat["common_out"] = len(succ_u & succ_v)
        feat["common_in"] = len(pred_u & pred_v)
        u_out_v_in = len(succ_u & pred_v)
        feat["u_out_v_in"] = u_out_v_in  # 2-hop paths u->w->v
        feat["u_in_v_out"] = len(pred_u & succ_v)

        # -- Jaccard undirected (1) --
        feat["jaccard"] = cn_count / max(union_size, 1)

        # -- Jaccard directed: out-out, out-in (2) --
        feat["jaccard_out_out"] = len(succ_u & succ_v) / max(len(succ_u | succ_v), 1)
        feat["jaccard_out_in"] = u_out_v_in / max(len(succ_u | pred_v), 1)

        # -- Adamic-Adar undirected (1) --
        aa = 0.0
        for w in cn:
            deg_w = G_und.degree(w)
            if deg_w > 1:
                aa += 1.0 / np.log(deg_w)
        feat["adamic_adar"] = aa

        # -- Adamic-Adar directed: through u->w->v path (1) --
        aa_dir = 0.0
        two_hop_nodes = succ_u & pred_v
        for w in two_hop_nodes:
            total_w = G.out_degree(w) + G.in_degree(w)
            if total_w > 1:
                aa_dir += 1.0 / np.log(total_w)
        feat["adamic_adar_dir"] = aa_dir

        # -- Resource Allocation (1) --
        ra = 0.0
        for w in cn:
            deg_w = G_und.degree(w)
            if deg_w > 0:
                ra += 1.0 / deg_w
        feat["resource_allocation"] = ra

        # -- Preferential Attachment (1) --
        feat["pref_attach"] = out_u * in_v  # directed: u's out * v's in
        feat["pref_attach_total"] = total_u * total_v

        # -- PageRank (4) --
        pr_u = pagerank_dict.get(u, 0.0)
        pr_v = pagerank_dict.get(v, 0.0)
        feat["pagerank_u"] = pr_u
        feat["pagerank_v"] = pr_v
        feat["pagerank_diff"] = abs(pr_u - pr_v)
        feat["pagerank_product"] = pr_u * pr_v

        # -- HITS (4) --
        feat["hub_u"] = hubs.get(u, 0.0)
        feat["auth_v"] = authorities.get(v, 0.0)
        feat["hub_v"] = hubs.get(v, 0.0)
        feat["auth_u"] = authorities.get(u, 0.0)

        # -- Community (1) --
        comm_u = community_dict.get(u, -1)
        comm_v = community_dict.get(v, -2)
        feat["same_community"] = 1 if comm_u == comm_v else 0

        # -- Clustering Coefficient (2) --
        feat["clustering_u"] = clustering.get(u, 0.0)
        feat["clustering_v"] = clustering.get(v, 0.0)

        # -- Degree ratios (2) --
        feat["out_in_ratio_u"] = out_u / max(in_u, 1)
        feat["out_in_ratio_v"] = out_v / max(in_v, 1)

        # -- Reciprocity rates (2) --
        feat["recip_rate_u"] = reciprocity_rate.get(u, 0.0)
        feat["recip_rate_v"] = reciprocity_rate.get(v, 0.0)

        # -- CN ratios (3) --
        feat["cn_over_min_deg"] = cn_count / max(min(total_u, total_v), 1)
        feat["cn_over_max_deg"] = cn_count / max(max(total_u, total_v), 1)
        feat["aa_over_cn"] = aa / max(cn_count, 1)

        # -- Katz-like approximation (1): beta * paths_2 + beta^2 * paths_3_approx --
        # paths_3 approximation: for each w in succ_u, count |succ_w & pred_v| (expensive, use CN)
        beta = 0.01
        feat["katz_approx"] = beta * u_out_v_in + beta * beta * cn_count

        features.append(feat)

    return pd.DataFrame(features)
