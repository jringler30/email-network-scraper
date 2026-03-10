"""
Jmail Network Explorer — Interactive email network dashboard.

Run with:
    streamlit run app/app.py
"""

import math
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

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

st.markdown("""
<style>
    /* KPI metric cards */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-left: 3px solid #00d4aa;
        border-radius: 6px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label {
        color: #aaaaaa;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 600;
    }
    /* Page padding */
    .block-container {
        padding-top: 1.5rem;
        max-width: 1400px;
    }
    /* Sidebar */
    [data-testid="stSidebar"] h1 {
        font-size: 1.1rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    /* Section headers */
    h2 {
        font-weight: 600;
        border-bottom: 1px solid rgba(255,255,255,0.07);
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }
    /* Subtler dividers */
    hr { border-color: rgba(255,255,255,0.06); }
    /* Tighter captions */
    .stCaption { color: #888; }
</style>
""", unsafe_allow_html=True)


# =========================================================================
# Data loading
# =========================================================================
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

data = load_all(DATA_DIR)
edges_df = data["primary_edges"]

if edges_df is None:
    st.error(
        "**No edge data found.** Place at least one of these files in `data/`:\n\n"
        "- `data/cleaned_edges.csv`\n"
        "- `data/network_edge_list.csv`\n\n"
        "Run the scraper first or copy the CSV files into the `data/` directory."
    )
    st.stop()

nodes_df = data["nodes"]

# =========================================================================
# Build graph & compute metrics (cached)
# =========================================================================
G = build_graph(edges_df, directed=True)
metrics_df = compute_metrics(G)
communities = detect_communities(G)
summary = graph_summary(G)

# Attach community to metrics
metrics_df["community"] = metrics_df.index.map(lambda n: communities.get(n, -1))

# Datetime availability
has_dates = (
    "datetime" in edges_df.columns
    and edges_df["datetime"].notna().sum() > 10
)
if has_dates:
    valid_dates = edges_df["datetime"].dropna()
    date_min = valid_dates.min()
    date_max = valid_dates.max()

# =========================================================================
# Sidebar — navigation
# =========================================================================
with st.sidebar:
    st.title("🔗 Jmail Network")
    st.caption("Email communication analytics")
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
    st.caption(
        f"**{summary['num_nodes']:,}** nodes · "
        f"**{summary['num_edges']:,}** edges · "
        f"**{len(Counter(communities.values()))}** communities"
    )


# =========================================================================
# Shared helpers
# =========================================================================

def _node_sizes_log(G_sub, min_px: int = 6, max_px: int = 32) -> dict:
    """Log-scaled node sizes by weighted degree — prevents hubs from dominating."""
    wdeg = {n: G_sub.degree(n, weight="weight") for n in G_sub.nodes()}
    max_wd = max(wdeg.values()) if wdeg else 1
    return {
        n: min_px + (max_px - min_px) * math.log1p(wdeg[n]) / math.log1p(max_wd)
        for n in G_sub.nodes()
    }


def _metric_label(m: str) -> str:
    return {
        "weighted_degree": "Total Message Weight",
        "degree":          "Connection Count",
        "in_degree":       "Inbound Connections",
        "out_degree":      "Outbound Connections",
        "in_weighted":     "Inbound Message Weight",
        "out_weighted":    "Outbound Message Weight",
        "betweenness":     "Betweenness Centrality",
        "eigenvector":     "Eigenvector Centrality",
    }.get(m, m.replace("_", " ").title())


# =========================================================================
# Section: Overview
# =========================================================================
if section == "📊 Overview":
    st.header("Network Overview")
    st.caption(
        "A directed graph where each node is an email participant and each "
        "edge represents messages sent between them. Edge weight = message count."
    )

    # Row 1: core stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Participants", f"{summary['num_nodes']:,}")
    c2.metric("Unique Connections", f"{summary['num_edges']:,}")
    c3.metric("Total Messages", f"{summary['total_weight']:,}")
    c4.metric("Avg Connections / Node", f"{summary['avg_degree']:.1f}")

    # Row 2: structural stats
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Communities", f"{len(Counter(communities.values())):,}")
    c6.metric("Giant Component", f"{summary['giant_component_size']:,} nodes")
    c7.metric("Components", f"{summary['num_components']:,}")
    c8.metric("Graph Density", f"{summary['density']:.5f}")

    if has_dates:
        st.caption(
            f"Date range: **{date_min.strftime('%b %d, %Y')}** → "
            f"**{date_max.strftime('%b %d, %Y')}**"
        )

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Top Senders")
        top_out = metrics_df.nlargest(10, "out_weighted")[
            ["out_weighted", "out_degree", "community"]
        ].rename(columns={
            "out_weighted": "Messages Sent",
            "out_degree": "Recipients",
            "community": "Community",
        })
        top_out.index.name = "Node"
        st.dataframe(top_out, use_container_width=True)

    with col_b:
        st.subheader("Top Recipients")
        top_in = metrics_df.nlargest(10, "in_weighted")[
            ["in_weighted", "in_degree", "community"]
        ].rename(columns={
            "in_weighted": "Messages Received",
            "in_degree": "Senders",
            "community": "Community",
        })
        top_in.index.name = "Node"
        st.dataframe(top_in, use_container_width=True)

    st.divider()
    comm_counts = Counter(communities.values())
    st.subheader(f"Community Size Distribution — {len(comm_counts)} communities detected (Louvain)")
    fig = community_size_chart(sorted(comm_counts.values(), reverse=True))
    st.plotly_chart(fig, use_container_width=True)


# =========================================================================
# Section: Network Graph
# =========================================================================
elif section == "🌐 Network Graph":
    st.header("Interactive Network Graph")
    st.caption(
        "Nodes = participants · Edge thickness = relative message weight · "
        "Labels shown for top nodes · Drag to pan, scroll to zoom."
    )

    with st.sidebar:
        st.subheader("Graph Controls")

        avg_w = summary["total_weight"] / max(summary["num_edges"], 1)
        default_min_w = max(2, int(avg_w * 0.5))
        min_weight = st.slider(
            "Min edge weight", 1,
            max(20, int(avg_w * 4)),
            default_min_w,
            help="Hide edges with fewer messages than this threshold.",
        )

        default_max_n = min(80, summary["num_nodes"])
        max_nodes = st.slider(
            "Max nodes shown", 20,
            min(300, summary["num_nodes"]),
            default_max_n,
            help="Show only the top-N most connected nodes.",
        )

        giant_only = st.checkbox(
            "Giant component only", value=True,
            help="Restrict to the largest connected subgraph.",
        )

        all_nodes = sorted(G.nodes())
        search_node = st.selectbox(
            "Highlight node", ["(none)"] + all_nodes,
            help="Pin a specific node to the label list and highlight it in gold.",
        )
        if search_node == "(none)":
            search_node = None

        color_mode = st.radio(
            "Node color", ["Community", "Message Weight"],
            horizontal=True,
        )

        label_n = st.slider("Label top-N nodes", 5, 30, 12)

    # Filter + layout
    H = filter_graph(G, min_weight=min_weight, max_nodes=max_nodes,
                     giant_only=giant_only, highlight_node=search_node)

    if H.number_of_nodes() == 0:
        st.warning(
            "No nodes remain after filtering. "
            "Try lowering the minimum edge weight or disabling 'Giant component only'."
        )
    else:
        pos = compute_layout(H)

        node_sizes = _node_sizes_log(H)
        if search_node and search_node in H:
            node_sizes[search_node] = 35

        wdeg = {n: H.degree(n, weight="weight") for n in H.nodes()}

        if color_mode == "Community":
            node_colors = {n: communities.get(n, 0) for n in H.nodes()}
            clabel = "Community"
            is_cat = True
        else:
            node_colors = wdeg
            clabel = "Message Weight"
            is_cat = False

        fig = plotly_network(
            H, pos,
            node_sizes=node_sizes,
            node_colors=node_colors,
            color_label=clabel,
            title=f"{H.number_of_nodes()} nodes · {H.number_of_edges()} edges",
            highlight_node=search_node,
            label_top_n=label_n,
            categorical=is_cat,
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("⬇ Download filtered edge list"):
            filt_edges = [
                {"sender": u, "recipient": v, "weight": d.get("weight", 1)}
                for u, v, d in H.edges(data=True)
            ]
            filt_df = pd.DataFrame(filt_edges)
            st.download_button(
                "Download CSV", filt_df.to_csv(index=False),
                "filtered_edges.csv", "text/csv",
            )


# =========================================================================
# Section: Top Nodes
# =========================================================================
elif section == "🏆 Top Nodes":
    st.header("Most Important Participants")

    metric_options = [
        "weighted_degree", "out_weighted", "in_weighted",
        "degree", "out_degree", "in_degree",
        "betweenness", "eigenvector",
    ]
    metric_choice = st.selectbox(
        "Rank by",
        metric_options,
        format_func=_metric_label,
    )

    n_show = st.slider("Number of nodes", 10, 50, 20)

    top = metrics_df.nlargest(n_show, metric_choice).copy()
    top = top.sort_values(metric_choice, ascending=True)

    fig = bar_chart(
        top.reset_index(),
        x="node", y=metric_choice,
        title=f"Top {n_show} — {_metric_label(metric_choice)}",
        horizontal=True,
        height=max(380, n_show * 22),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detailed Metrics Table")
    display_cols = ["degree", "weighted_degree", "in_weighted", "out_weighted",
                    "betweenness", "eigenvector", "community"]
    table = metrics_df.nlargest(n_show, metric_choice)[display_cols].rename(
        columns={c: _metric_label(c) for c in display_cols}
    )
    table.index.name = "Node"
    st.dataframe(
        table.style.format({
            _metric_label("betweenness"): "{:.4f}",
            _metric_label("eigenvector"): "{:.4f}",
        }),
        use_container_width=True,
    )
    st.download_button(
        "⬇ Download table CSV",
        table.to_csv(),
        "top_nodes.csv", "text/csv",
    )


# =========================================================================
# Section: Communities
# =========================================================================
elif section == "🧩 Communities":
    st.header("Community Detection")
    st.caption("Communities detected using the Louvain algorithm on the undirected projection.")

    comm_counts = Counter(communities.values())
    n_comms = len(comm_counts)

    c1, c2, c3 = st.columns(3)
    c1.metric("Communities Detected", n_comms)
    c2.metric("Largest Community", f"{max(comm_counts.values()):,} members")
    c3.metric("Median Community Size", f"{int(pd.Series(list(comm_counts.values())).median())} members")

    fig = community_size_chart(sorted(comm_counts.values(), reverse=True))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Explore a Community")

    sorted_comms = sorted(comm_counts.keys(), key=lambda c: comm_counts[c], reverse=True)
    selected_comm = st.selectbox(
        "Select community",
        sorted_comms,
        format_func=lambda c: f"Community {c}  ({comm_counts[c]} members)",
    )

    members = [n for n, c in communities.items() if c == selected_comm]
    member_metrics = metrics_df.loc[metrics_df.index.isin(members)].sort_values(
        "weighted_degree", ascending=False
    )

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Members", len(members))
    mc2.metric("Top Sender", member_metrics["out_weighted"].idxmax() if not member_metrics.empty else "—")
    mc3.metric("Top Recipient", member_metrics["in_weighted"].idxmax() if not member_metrics.empty else "—")

    display = member_metrics[["degree", "weighted_degree", "in_weighted", "out_weighted", "betweenness"]].head(20)
    display.index.name = "Node"
    st.dataframe(
        display.rename(columns={c: _metric_label(c) for c in display.columns})
               .style.format({_metric_label("betweenness"): "{:.4f}"}),
        use_container_width=True,
    )

    with st.expander("View community subgraph"):
        sub_G = G.subgraph(set(members)).copy()
        if sub_G.number_of_nodes() > 0:
            # Limit large communities for rendering
            if sub_G.number_of_nodes() > 150:
                st.caption(f"Showing top 150 of {sub_G.number_of_nodes()} members by message weight.")
                wdeg = dict(sub_G.degree(weight="weight"))
                top_m = sorted(wdeg, key=wdeg.get, reverse=True)[:150]
                sub_G = sub_G.subgraph(top_m).copy()
            pos = compute_layout(sub_G)
            sizes = _node_sizes_log(sub_G)
            wdeg = {n: sub_G.degree(n, weight="weight") for n in sub_G.nodes()}
            fig = plotly_network(
                sub_G, pos,
                node_sizes=sizes,
                node_colors=wdeg,
                color_label="Message Weight",
                title=f"Community {selected_comm} — {sub_G.number_of_nodes()} nodes",
                categorical=False,
            )
            st.plotly_chart(fig, use_container_width=True)


# =========================================================================
# Section: Ego Network
# =========================================================================
elif section == "👤 Ego Network":
    st.header("Ego Network Explorer")
    st.caption("Explore the immediate communication neighbourhood of any participant.")

    all_nodes_sorted = metrics_df.sort_values("weighted_degree", ascending=False).index.tolist()

    col_sel, col_rad = st.columns([3, 1])
    with col_sel:
        ego_node = st.selectbox("Select participant", all_nodes_sorted)
    with col_rad:
        radius = st.radio("Radius", [1, 2], horizontal=True,
                          help="1 = direct contacts only · 2 = contacts-of-contacts")

    ego_G = get_ego_graph(G, ego_node, radius=radius)

    if ego_G is None or ego_G.number_of_nodes() == 0:
        st.warning("Node not found or has no connections.")
    else:
        node_stats = metrics_df.loc[ego_node] if ego_node in metrics_df.index else None
        if node_stats is not None:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Connections", int(node_stats["degree"]))
            c2.metric("Messages Sent", int(node_stats["out_weighted"]))
            c3.metric("Messages Received", int(node_stats["in_weighted"]))
            c4.metric("Betweenness", f"{node_stats['betweenness']:.4f}")
            c5.metric("Community", int(node_stats["community"]))

        # Activity date range
        if has_dates:
            node_msgs = edges_df[
                (edges_df["sender"] == ego_node) | (edges_df["recipient"] == ego_node)
            ]
            valid = node_msgs["datetime"].dropna()
            if len(valid) > 0:
                st.caption(
                    f"Active: **{valid.min().strftime('%b %d, %Y')}** → "
                    f"**{valid.max().strftime('%b %d, %Y')}** "
                    f"({len(node_msgs):,} messages)"
                )

        st.subheader("Strongest Ties")
        ties = []
        for u, v, d in G.edges(ego_node, data=True):
            ties.append({"contact": v, "outgoing": d.get("weight", 1), "incoming": 0})
        if G.is_directed():
            for u, v, d in G.in_edges(ego_node, data=True):
                ties.append({"contact": u, "outgoing": 0, "incoming": d.get("weight", 1)})

        if ties:
            ties_df = pd.DataFrame(ties)
            agg = ties_df.groupby("contact").sum(numeric_only=True).reset_index()
            agg["total"] = agg["outgoing"] + agg["incoming"]
            agg = agg.sort_values("total", ascending=False).head(15)
            st.dataframe(agg, use_container_width=True, hide_index=True)

        # Ego graph visualisation
        st.subheader(f"Ego Graph (radius {radius})")
        if ego_G.number_of_nodes() > 250:
            st.caption(f"Large ego network ({ego_G.number_of_nodes()} nodes) — showing top 250 by weight.")
            wdeg = {n: ego_G.degree(n, weight="weight") for n in ego_G.nodes()}
            top = sorted(wdeg, key=wdeg.get, reverse=True)[:250]
            if ego_node not in top:
                top.append(ego_node)
            ego_G = ego_G.subgraph(top).copy()

        pos = compute_layout(ego_G)
        sizes = _node_sizes_log(ego_G)
        sizes[ego_node] = 32
        colors = {n: communities.get(n, 0) for n in ego_G.nodes()}

        fig = plotly_network(
            ego_G, pos,
            node_sizes=sizes,
            node_colors=colors,
            color_label="Community",
            title=f"Ego network: {ego_node}  ({ego_G.number_of_nodes()} nodes)",
            highlight_node=ego_node,
            label_top_n=10,
            categorical=True,
        )
        st.plotly_chart(fig, use_container_width=True)


# =========================================================================
# Section: Relationships / Heatmap
# =========================================================================
elif section == "🔥 Relationships":
    st.header("Relationship Strength")
    st.caption("Explore message volume between specific participants.")

    mode = st.radio("View mode", ["Top-N interaction matrix", "Node spotlight"], horizontal=True)

    if mode == "Top-N interaction matrix":
        n = st.slider("Top N nodes", 5, 30, 15)
        top_nodes = metrics_df.nlargest(n, "weighted_degree").index.tolist()
        matrix = build_interaction_matrix(G, top_nodes)
        fig = heatmap(
            matrix,
            title=f"Message Volume Matrix — Top {n} participants",
            height=max(420, n * 30),
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("⬇ Download interaction matrix"):
            st.download_button(
                "Download CSV", matrix.to_csv(),
                "interaction_matrix.csv", "text/csv",
            )
    else:
        all_nodes_sorted = metrics_df.sort_values("weighted_degree", ascending=False).index.tolist()
        sel = st.selectbox("Select participant", all_nodes_sorted)

        contacts = {}
        for u, v, d in G.edges(data=True):
            if u == sel:
                contacts[v] = contacts.get(v, 0) + d.get("weight", 1)
            elif v == sel:
                contacts[u] = contacts.get(u, 0) + d.get("weight", 1)

        if contacts:
            cdf = (
                pd.DataFrame([{"contact": k, "messages": v} for k, v in contacts.items()])
                .sort_values("messages", ascending=False)
                .head(20)
            )
            fig = bar_chart(
                cdf, x="contact", y="messages",
                title=f"Top contacts for: {sel}",
                horizontal=True,
                height=max(320, len(cdf) * 26),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(cdf, use_container_width=True, hide_index=True)
        else:
            st.info("No connections found for this node.")

    st.divider()
    st.subheader("Strongest Connections Overall")
    strong = [
        {"sender": u, "recipient": v, "messages": d.get("weight", 1)}
        for u, v, d in G.edges(data=True)
    ]
    strong_df = pd.DataFrame(strong).sort_values("messages", ascending=False).head(25)
    st.dataframe(strong_df, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇ Download top connections CSV",
        strong_df.to_csv(index=False),
        "strongest_edges.csv", "text/csv",
    )


# =========================================================================
# Section: Timeline
# =========================================================================
elif section == "📅 Timeline":
    st.header("Activity Timeline")
    st.caption("Message volume and relationship formation over time.")

    if not has_dates:
        st.info(
            "No timestamp data found. "
            "This section requires a `datetime` column with valid timestamps in the edge CSV."
        )
    else:
        valid = edges_df["datetime"].dropna()

        freq = st.radio("Aggregation", ["Month", "Week", "Day"], horizontal=True)
        freq_map = {"Month": "M", "Week": "W", "Day": "D"}

        col1, col2 = st.columns(2)
        with col1:
            st.metric("First Message", date_min.strftime("%b %d, %Y"))
        with col2:
            st.metric("Last Message", date_max.strftime("%b %d, %Y"))

        fig = timeline_chart(valid, freq=freq_map[freq],
                             title=f"Email Volume by {freq}")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = cumulative_chart(valid, title="Cumulative Messages Over Time")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("New Relationships Over Time")
        st.caption("When each unique sender → recipient pair first appeared.")
        first_seen = (
            edges_df.dropna(subset=["datetime"])
            .sort_values("datetime")
            .assign(pair=lambda df: df["sender"] + " → " + df["recipient"])
            .drop_duplicates("pair")
        )
        if len(first_seen) > 0:
            fig3 = timeline_chart(
                first_seen["datetime"],
                freq=freq_map[freq],
                title=f"New Unique Relationships by {freq}",
            )
            st.plotly_chart(fig3, use_container_width=True)
