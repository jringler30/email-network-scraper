"""
Jmail Network Explorer — Interactive email network dashboard.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path

from utils.data_loader import load_all
from utils.graph_builder import (
    build_graph, compute_metrics, detect_communities,
    graph_summary, get_ego_graph,
)
from utils.charts import (
    bar_chart, community_size_chart, heatmap,
    timeline_chart, cumulative_chart, plotly_network,
)
from utils.network_views import (
    filter_graph, compute_layout, build_interaction_matrix,
)

# =========================================================================
# Page config
# =========================================================================
st.set_page_config(
    page_title="Jmail Network Explorer",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark-theme tweaks via custom CSS
st.markdown("""
<style>
    /* KPI cards */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label {
        color: #999;
    }
    /* Sidebar header */
    [data-testid="stSidebar"] h1 {
        font-size: 1.2rem;
    }
    /* Tighten padding */
    .block-container { padding-top: 1.5rem; }
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        border-radius: 6px 6px 0 0;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================================
# Data loading
# =========================================================================
DATA_DIR = Path(".")

data = load_all(DATA_DIR)
edges_df = data["primary_edges"]

if edges_df is None:
    st.error(
        "**No edge data found.** The app expects at least one of these files "
        "in the repo root:\n\n"
        "- `cleaned_edges.csv`\n"
        "- `network_edge_list.csv`\n\n"
        "Run the scraper first or place the CSV files in the working directory."
    )
    st.stop()

nodes_df = data["nodes"]

# =========================================================================
# Build graph
# =========================================================================
G = build_graph(edges_df, directed=True)
metrics_df = compute_metrics(G)
communities = detect_communities(G)
summary = graph_summary(G)

# Add community info to metrics
metrics_df["community"] = metrics_df.index.map(lambda n: communities.get(n, -1))

# Check for datetime availability
has_dates = "datetime" in edges_df.columns and edges_df["datetime"].notna().sum() > 10
if has_dates:
    date_min = edges_df["datetime"].min()
    date_max = edges_df["datetime"].max()

# =========================================================================
# Sidebar — global controls
# =========================================================================
with st.sidebar:
    st.title("🔗 Jmail Network Explorer")
    st.caption("Interactive email network analysis")
    st.divider()

    section = st.radio(
        "Navigate",
        [
            "📊 Overview",
            "🌐 Network Graph",
            "🏆 Top Nodes",
            "🧩 Communities",
            "👤 Ego Network",
            "🔥 Relationships",
            "📅 Timeline",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption(f"Nodes: **{summary['num_nodes']:,}** · Edges: **{summary['num_edges']:,}**")


# =========================================================================
# Section: Overview
# =========================================================================
if section == "📊 Overview":
    st.header("Network Overview")
    st.markdown(
        "This dashboard explores the **Jmail email communication network** — "
        "a directed graph where each edge represents one or more emails sent "
        "between participants. Use the sidebar to navigate through different "
        "analysis views."
    )

    cols = st.columns(3)
    cols[0].metric("Total Nodes", f"{summary['num_nodes']:,}")
    cols[1].metric("Total Edges", f"{summary['num_edges']:,}")
    cols[2].metric("Total Interaction Weight", f"{summary['total_weight']:,}")

    cols2 = st.columns(3)
    cols2[0].metric("Connected Components", f"{summary['num_components']:,}")
    cols2[1].metric("Giant Component", f"{summary['giant_component_size']:,} nodes")
    cols2[2].metric("Density", f"{summary['density']:.5f}")

    if has_dates:
        st.markdown(
            f"**Date range:** {date_min.strftime('%Y-%m-%d')} → {date_max.strftime('%Y-%m-%d')}"
        )

    st.divider()

    # Quick top-5 table
    st.subheader("Top 10 by Weighted Degree")
    top = metrics_df.nlargest(10, "weighted_degree")[["degree", "weighted_degree", "betweenness", "community"]]
    top.index.name = "Node"
    st.dataframe(top.style.format({
        "betweenness": "{:.4f}",
    }), use_container_width=True)

    # Community overview
    from collections import Counter
    comm_counts = Counter(communities.values())
    st.subheader(f"Communities: {len(comm_counts)} detected (Louvain)")
    fig = community_size_chart(sorted(comm_counts.values(), reverse=True))
    st.plotly_chart(fig, use_container_width=True)


# =========================================================================
# Section: Network Graph
# =========================================================================
elif section == "🌐 Network Graph":
    st.header("Interactive Network Graph")
    st.caption("Filtered view of the email communication network. Adjust controls in the sidebar.")

    with st.sidebar:
        st.subheader("Graph Controls")
        min_weight = st.slider("Min edge weight", 1, max(20, int(summary["total_weight"] / summary["num_edges"] * 3)), 2)
        max_nodes = st.slider("Max nodes shown", 20, min(500, summary["num_nodes"]), min(150, summary["num_nodes"]))
        giant_only = st.checkbox("Giant component only", value=True)

        all_nodes = sorted(G.nodes())
        search_node = st.selectbox("Highlight node", ["(none)"] + all_nodes)
        if search_node == "(none)":
            search_node = None

        color_mode = st.radio("Node color", ["Community", "Weighted Degree"], horizontal=True)

    # Filter and layout
    H = filter_graph(G, min_weight=min_weight, max_nodes=max_nodes,
                     giant_only=giant_only, highlight_node=search_node)

    if H.number_of_nodes() == 0:
        st.warning("No nodes remain after filtering. Try lowering the minimum edge weight.")
    else:
        pos = compute_layout(H)

        # Node sizing: scale by weighted degree
        wdeg = dict(H.degree(weight="weight"))
        max_wd = max(wdeg.values()) if wdeg else 1
        node_sizes = {n: 5 + 25 * (wdeg[n] / max_wd) for n in H.nodes()}

        # Node coloring
        if color_mode == "Community":
            node_colors = {n: communities.get(n, 0) for n in H.nodes()}
            clabel = "Community"
        else:
            node_colors = {n: wdeg.get(n, 0) for n in H.nodes()}
            clabel = "Weighted Degree"

        # Highlight search node
        if search_node and search_node in H:
            node_sizes[search_node] = 35

        fig = plotly_network(H, pos, node_sizes=node_sizes,
                             node_colors=node_colors, color_label=clabel,
                             title=f"Network ({H.number_of_nodes()} nodes, {H.number_of_edges()} edges)")
        st.plotly_chart(fig, use_container_width=True)

        # Download filtered edges
        with st.expander("Download filtered data"):
            filt_edges = []
            for u, v, d in H.edges(data=True):
                filt_edges.append({"sender": u, "recipient": v, "weight": d.get("weight", 1)})
            filt_df = pd.DataFrame(filt_edges)
            st.download_button("Download filtered edges CSV", filt_df.to_csv(index=False),
                               "filtered_edges.csv", "text/csv")


# =========================================================================
# Section: Top Nodes
# =========================================================================
elif section == "🏆 Top Nodes":
    st.header("Most Important Contacts")

    metric_choice = st.selectbox(
        "Rank by",
        ["weighted_degree", "degree", "betweenness", "eigenvector"],
    )

    n_show = st.slider("Number of nodes", 10, 50, 20)

    top = metrics_df.nlargest(n_show, metric_choice).copy()
    top = top.sort_values(metric_choice, ascending=True)  # for horizontal bar

    fig = bar_chart(
        top.reset_index(),
        x="node", y=metric_choice,
        title=f"Top {n_show} by {metric_choice.replace('_', ' ').title()}",
        horizontal=True,
        height=max(350, n_show * 22),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detailed Table")
    table = metrics_df.nlargest(n_show, metric_choice)[
        ["degree", "weighted_degree", "betweenness", "eigenvector", "community"]
    ]
    table.index.name = "Node"
    st.dataframe(table.style.format({
        "betweenness": "{:.4f}",
        "eigenvector": "{:.4f}",
    }), use_container_width=True)


# =========================================================================
# Section: Communities
# =========================================================================
elif section == "🧩 Communities":
    st.header("Community Detection")

    from collections import Counter
    comm_counts = Counter(communities.values())
    n_comms = len(comm_counts)

    st.metric("Communities Detected", n_comms)

    fig = community_size_chart(sorted(comm_counts.values(), reverse=True))
    st.plotly_chart(fig, use_container_width=True)

    # Community explorer
    st.subheader("Explore a Community")
    sorted_comms = sorted(comm_counts.keys(), key=lambda c: comm_counts[c], reverse=True)
    selected_comm = st.selectbox(
        "Select community",
        sorted_comms,
        format_func=lambda c: f"Community {c} ({comm_counts[c]} members)",
    )

    members = [n for n, c in communities.items() if c == selected_comm]
    member_metrics = metrics_df.loc[metrics_df.index.isin(members)].sort_values(
        "weighted_degree", ascending=False
    )

    st.markdown(f"**{len(members)} members** in Community {selected_comm}")
    st.dataframe(
        member_metrics[["degree", "weighted_degree", "betweenness"]].head(20).style.format({
            "betweenness": "{:.4f}",
        }),
        use_container_width=True,
    )

    # Mini network of this community
    with st.expander("View community subgraph"):
        sub_nodes = set(members)
        sub_G = G.subgraph(sub_nodes).copy()
        if sub_G.number_of_nodes() > 0:
            pos = compute_layout(sub_G)
            wdeg = dict(sub_G.degree(weight="weight"))
            max_wd = max(wdeg.values()) if wdeg else 1
            sizes = {n: 6 + 20 * (wdeg[n] / max_wd) for n in sub_G.nodes()}
            colors = {n: wdeg[n] for n in sub_G.nodes()}
            fig = plotly_network(sub_G, pos, node_sizes=sizes, node_colors=colors,
                                 color_label="Weighted Degree",
                                 title=f"Community {selected_comm}")
            st.plotly_chart(fig, use_container_width=True)


# =========================================================================
# Section: Ego Network
# =========================================================================
elif section == "👤 Ego Network":
    st.header("Ego Network Explorer")

    all_nodes_sorted = metrics_df.sort_values("weighted_degree", ascending=False).index.tolist()
    ego_node = st.selectbox("Select a node", all_nodes_sorted)

    radius = st.radio("Radius", [1, 2], horizontal=True)

    ego_G = get_ego_graph(G, ego_node, radius=radius)

    if ego_G is None or ego_G.number_of_nodes() == 0:
        st.warning("Node not found or has no connections.")
    else:
        # Stats
        node_stats = metrics_df.loc[ego_node] if ego_node in metrics_df.index else None
        if node_stats is not None:
            cols = st.columns(4)
            cols[0].metric("Degree", int(node_stats["degree"]))
            cols[1].metric("Weighted Degree", int(node_stats["weighted_degree"]))
            cols[2].metric("Betweenness", f"{node_stats['betweenness']:.4f}")
            cols[3].metric("Community", int(node_stats["community"]))

        # Strongest ties
        st.subheader("Strongest Ties")
        ties = []
        for u, v, d in G.edges(ego_node, data=True):
            ties.append({"contact": v, "weight (outgoing)": d.get("weight", 1)})
        # Also incoming
        if G.is_directed():
            for u, v, d in G.in_edges(ego_node, data=True):
                ties.append({"contact": u, "weight (incoming)": d.get("weight", 1)})

        if ties:
            ties_df = pd.DataFrame(ties)
            # Aggregate by contact
            agg = ties_df.groupby("contact").sum(numeric_only=True).reset_index()
            agg["total"] = agg.sum(axis=1, numeric_only=True)
            agg = agg.sort_values("total", ascending=False).head(15)
            st.dataframe(agg, use_container_width=True, hide_index=True)

        # Date range for this node
        if has_dates:
            node_msgs = edges_df[
                (edges_df["sender"] == ego_node) | (edges_df["recipient"] == ego_node)
            ]
            valid_dates = node_msgs["datetime"].dropna()
            if len(valid_dates) > 0:
                st.markdown(
                    f"**Active:** {valid_dates.min().strftime('%Y-%m-%d')} → "
                    f"{valid_dates.max().strftime('%Y-%m-%d')} "
                    f"({len(node_msgs)} messages)"
                )

        # Ego graph visualisation
        st.subheader(f"Ego Graph (radius={radius})")
        # Limit size for large egos
        if ego_G.number_of_nodes() > 300:
            st.info(f"Ego network has {ego_G.number_of_nodes()} nodes — showing top 300 by weight.")
            wdeg = dict(ego_G.degree(weight="weight"))
            top = sorted(wdeg, key=wdeg.get, reverse=True)[:300]
            if ego_node not in top:
                top.append(ego_node)
            ego_G = ego_G.subgraph(top).copy()

        pos = compute_layout(ego_G)
        wdeg = dict(ego_G.degree(weight="weight"))
        max_wd = max(wdeg.values()) if wdeg else 1
        sizes = {n: 6 + 20 * (wdeg[n] / max_wd) for n in ego_G.nodes()}
        sizes[ego_node] = 30  # highlight ego
        colors = {n: communities.get(n, 0) for n in ego_G.nodes()}

        fig = plotly_network(ego_G, pos, node_sizes=sizes, node_colors=colors,
                             color_label="Community",
                             title=f"Ego network: {ego_node}")
        st.plotly_chart(fig, use_container_width=True)


# =========================================================================
# Section: Relationships / Heatmap
# =========================================================================
elif section == "🔥 Relationships":
    st.header("Relationship Strength")

    mode = st.radio("View", ["Top N heatmap", "Node-specific"], horizontal=True)

    if mode == "Top N heatmap":
        n = st.slider("Top N nodes", 5, 30, 15)
        top_nodes = metrics_df.nlargest(n, "weighted_degree").index.tolist()
        matrix = build_interaction_matrix(G, top_nodes)
        fig = heatmap(matrix, title=f"Interaction Matrix (Top {n})", height=max(400, n * 28))
        st.plotly_chart(fig, use_container_width=True)
    else:
        all_nodes_sorted = metrics_df.sort_values("weighted_degree", ascending=False).index.tolist()
        sel = st.selectbox("Select node", all_nodes_sorted)

        # Get top contacts
        contacts = {}
        for u, v, d in G.edges(data=True):
            if u == sel:
                contacts[v] = contacts.get(v, 0) + d.get("weight", 1)
            elif v == sel:
                contacts[u] = contacts.get(u, 0) + d.get("weight", 1)

        if contacts:
            cdf = pd.DataFrame([
                {"contact": k, "weight": v} for k, v in contacts.items()
            ]).sort_values("weight", ascending=False).head(20)

            fig = bar_chart(
                cdf, x="contact", y="weight",
                title=f"Top contacts for {sel}",
                horizontal=True,
                height=max(300, len(cdf) * 25),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(cdf, use_container_width=True, hide_index=True)
        else:
            st.info("No connections found for this node.")

    # Strongest edges overall
    st.divider()
    st.subheader("Strongest Edges Overall")
    strong = []
    for u, v, d in G.edges(data=True):
        strong.append({"sender": u, "recipient": v, "weight": d.get("weight", 1)})
    strong_df = pd.DataFrame(strong).sort_values("weight", ascending=False).head(25)
    st.dataframe(strong_df, use_container_width=True, hide_index=True)


# =========================================================================
# Section: Timeline
# =========================================================================
elif section == "📅 Timeline":
    st.header("Activity Timeline")

    if not has_dates:
        st.info(
            "Date information is not available or insufficient in the edge data. "
            "This section requires a `datetime` column with valid timestamps."
        )
    else:
        valid = edges_df["datetime"].dropna()

        freq = st.radio("Aggregation", ["Month", "Week", "Day"], horizontal=True)
        freq_map = {"Month": "M", "Week": "W", "Day": "D"}

        fig = timeline_chart(valid, freq=freq_map[freq],
                             title=f"Email Volume by {freq}")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = cumulative_chart(valid, title="Cumulative Messages Over Time")
        st.plotly_chart(fig2, use_container_width=True)

        # New relationships over time
        st.subheader("New Relationships Over Time")
        st.caption("When each unique sender→recipient pair first appeared.")
        first_seen = edges_df.dropna(subset=["datetime"]).sort_values("datetime")
        first_seen["pair"] = first_seen["sender"] + " → " + first_seen["recipient"]
        first_seen = first_seen.drop_duplicates("pair")
        first_seen_dates = first_seen["datetime"]

        fig3 = timeline_chart(first_seen_dates, freq=freq_map[freq],
                              title=f"New Relationships by {freq}")
        st.plotly_chart(fig3, use_container_width=True)
