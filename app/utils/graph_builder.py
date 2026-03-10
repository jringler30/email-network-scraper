"""
Graph construction and analysis utilities.
Builds NetworkX graphs from edge DataFrames and computes common metrics.
"""

import networkx as nx
import pandas as pd
import streamlit as st
from collections import Counter


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def build_graph(
    edges: pd.DataFrame,
    directed: bool = True,
) -> nx.DiGraph | nx.Graph:
    """
    Build a NetworkX graph from an edge DataFrame.

    Expects at minimum columns 'sender' and 'recipient'.
    If a 'weight' column exists, it is used directly.
    Otherwise, parallel edges are aggregated and counted as weight.
    """
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    if "weight" in edges.columns:
        for _, row in edges.iterrows():
            s, t = row["sender"], row["recipient"]
            w = row.get("weight", 1)
            if pd.isna(s) or pd.isna(t):
                continue
            if G.has_edge(s, t):
                G[s][t]["weight"] += w
            else:
                G.add_edge(s, t, weight=w)
    else:
        # Aggregate by (sender, recipient) pair
        pair_counts = Counter()
        for _, row in edges.iterrows():
            s, t = str(row["sender"]).strip(), str(row["recipient"]).strip()
            if s == "nan" or t == "nan":
                continue
            pair_counts[(s, t)] += 1
        for (s, t), w in pair_counts.items():
            if G.has_edge(s, t):
                G[s][t]["weight"] += w
            else:
                G.add_edge(s, t, weight=w)

    return G


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def compute_metrics(_G: nx.DiGraph | nx.Graph) -> pd.DataFrame:
    """Compute per-node centrality metrics. Returns a DataFrame indexed by node."""
    G = _G
    undirected = G.to_undirected() if G.is_directed() else G

    degree = dict(G.degree())
    weighted_degree = dict(G.degree(weight="weight"))

    # Betweenness on undirected version for speed
    betweenness = nx.betweenness_centrality(undirected, weight=None, k=min(200, len(G)))

    # Eigenvector on largest connected component only (must be connected)
    eigenvector = {}
    if len(undirected) > 0:
        components = sorted(nx.connected_components(undirected), key=len, reverse=True)
        largest_cc = undirected.subgraph(components[0]).copy()
        try:
            ev = nx.eigenvector_centrality_numpy(largest_cc, weight="weight")
            eigenvector.update(ev)
        except Exception:
            pass

    df = pd.DataFrame({
        "node": list(G.nodes()),
        "degree": [degree.get(n, 0) for n in G.nodes()],
        "weighted_degree": [weighted_degree.get(n, 0) for n in G.nodes()],
        "betweenness": [betweenness.get(n, 0.0) for n in G.nodes()],
        "eigenvector": [eigenvector.get(n, 0.0) for n in G.nodes()],
    }).set_index("node")

    return df


@st.cache_data(show_spinner=False)
def detect_communities(_G: nx.DiGraph | nx.Graph) -> dict:
    """
    Run Louvain community detection on the undirected projection.
    Returns a dict mapping node -> community_id.
    """
    G = _G.to_undirected() if _G.is_directed() else _G

    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G, weight="weight", random_state=42)
    except ImportError:
        # Fallback: greedy modularity
        from networkx.algorithms.community import greedy_modularity_communities
        comms = greedy_modularity_communities(G, weight="weight")
        partition = {}
        for i, comm in enumerate(comms):
            for node in comm:
                partition[node] = i
    return partition


def graph_summary(G: nx.DiGraph | nx.Graph) -> dict:
    """Return high-level stats about the graph."""
    undirected = G.to_undirected() if G.is_directed() else G
    components = list(nx.connected_components(undirected))
    largest = max(components, key=len) if components else set()
    total_weight = sum(d.get("weight", 1) for _, _, d in G.edges(data=True))

    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "total_weight": total_weight,
        "num_components": len(components),
        "giant_component_size": len(largest),
        "density": nx.density(G),
    }


def get_ego_graph(G, node, radius=1):
    """Extract the ego network around a given node."""
    if node not in G:
        return None
    return nx.ego_graph(G, node, radius=radius)
