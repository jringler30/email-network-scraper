"""
Network view helpers: filtering, subgraph extraction, and layout computation.
"""

import networkx as nx
import pandas as pd
import streamlit as st


def filter_graph(
    G: nx.DiGraph | nx.Graph,
    min_weight: int = 1,
    max_nodes: int = 300,
    giant_only: bool = False,
    highlight_node: str | None = None,
) -> nx.DiGraph | nx.Graph:
    """
    Return a filtered subgraph based on user controls.
    Steps:
      1. Remove edges below min_weight
      2. Optionally restrict to giant component
      3. Keep only top-N nodes by weighted degree
      4. If highlight_node given, ensure it and its neighbors are included
    """
    # Step 1: weight filter
    edges_to_keep = [
        (u, v) for u, v, d in G.edges(data=True)
        if d.get("weight", 1) >= min_weight
    ]
    H = G.edge_subgraph(edges_to_keep).copy()

    # Remove isolates created by edge removal
    isolates = list(nx.isolates(H))
    H.remove_nodes_from(isolates)

    if H.number_of_nodes() == 0:
        return H

    # Step 2: giant component
    if giant_only:
        undirected = H.to_undirected() if H.is_directed() else H
        components = sorted(nx.connected_components(undirected), key=len, reverse=True)
        if components:
            H = H.subgraph(components[0]).copy()

    # Step 3: top-N by weighted degree
    if H.number_of_nodes() > max_nodes:
        wdeg = dict(H.degree(weight="weight"))
        top_nodes = sorted(wdeg, key=wdeg.get, reverse=True)[:max_nodes]
        # Ensure highlight node is included
        if highlight_node and highlight_node in H and highlight_node not in top_nodes:
            top_nodes[-1] = highlight_node
            # Also include its neighbors
            neighbors = list(H.neighbors(highlight_node))
            if H.is_directed():
                neighbors += list(H.predecessors(highlight_node))
            for nb in neighbors[:10]:
                if nb not in top_nodes:
                    top_nodes.append(nb)
        H = H.subgraph(top_nodes).copy()

    return H


@st.cache_data(show_spinner=False)
def compute_layout(_G, seed: int = 42) -> dict:
    """Compute a spring layout, cached."""
    G = _G
    if G.number_of_nodes() == 0:
        return {}
    k = max(1.0 / (G.number_of_nodes() ** 0.5), 0.05)
    return nx.spring_layout(G, k=k, iterations=50, seed=seed, weight="weight")


def build_interaction_matrix(G, nodes: list[str]) -> pd.DataFrame:
    """
    Build an adjacency/weight matrix for a subset of nodes.
    Returns a DataFrame with rows=sources, cols=targets.
    """
    matrix = pd.DataFrame(0, index=nodes, columns=nodes, dtype=int)
    for u, v, d in G.edges(data=True):
        if u in nodes and v in nodes:
            matrix.loc[u, v] += d.get("weight", 1)
    return matrix
