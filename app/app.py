"""
Jmail Network Explorer — Interactive email network intelligence dashboard.

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
import streamlit.components.v1 as components

from utils.data_loader import load_all
from utils.graph_builder import (
    build_graph, compute_metrics, detect_communities,
    graph_summary, get_ego_graph,
)
from utils.charts import (
    bar_chart, community_size_chart, heatmap,
    timeline_chart, cumulative_chart, pyvis_network,
)
from utils.network_views import (
    filter_graph, build_interaction_matrix,
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

# =========================================================================
# Global CSS — forced dark analytics theme
# Designed to hold regardless of Streamlit theme setting (light/dark/system).
# All background rules use !important; text color scoped to .stApp to avoid
# fighting Streamlit widget internals.
# =========================================================================
st.markdown("""
<style>
  /* ── Force dark color scheme on browser root ──────────────────────── */
  html {
    color-scheme: dark !important;
    background-color: #0B0F17 !important;
  }
  body {
    background-color: #0B0F17 !important;
    color: #E6EDF3 !important;
  }

  /* ── All main app containers ──────────────────────────────────────── */
  .stApp,
  [data-testid="stAppViewContainer"],
  [data-testid="stMain"],
  [data-testid="stMainBlockContainer"],
  [data-testid="stHeader"],
  [data-testid="stToolbar"],
  [data-testid="stDecoration"],
  [data-testid="stBottomBlockContainer"],
  .main {
    background-color: #0B0F17 !important;
  }

  /* ── Block container (main content) ───────────────────────────────── */
  .block-container,
  [data-testid="stMainBlockContainer"] > div {
    background-color: #0B0F17 !important;
    padding-top: 1.4rem;
    max-width: 1440px;
  }

  /* ── Sidebar ──────────────────────────────────────────────────────── */
  [data-testid="stSidebar"],
  [data-testid="stSidebar"] > div,
  [data-testid="stSidebar"] > div > div,
  section[data-testid="stSidebar"] > div:first-child {
    background-color: #0F1520 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
  }

  /* ── Scope text colors to app root (avoids overriding widgets) ─────── */
  .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 {
    color: #E6EDF3 !important;
  }
  .stApp p { color: #C9D1D9 !important; }
  .stApp label { color: #C9D1D9 !important; }
  .stCaption, .caption-text { color: #8899AA !important; font-size: 0.82rem; }

  /* ── Heading styles ───────────────────────────────────────────────── */
  .stApp h2 {
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding-bottom: 0.45rem;
    margin-bottom: 1rem;
  }
  .stApp h3 { font-size: 1rem !important; font-weight: 600 !important; }

  /* ── Sidebar typography ───────────────────────────────────────────── */
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 {
    color: #E6EDF3 !important;
  }
  [data-testid="stRadio"] label {
    padding: 5px 0;
    color: #C9D1D9 !important;
    font-size: 0.9rem;
  }
  [data-testid="stRadio"] label:hover { color: #00E5A8 !important; }

  /* ── Dividers ─────────────────────────────────────────────────────── */
  hr { border-color: rgba(255,255,255,0.06) !important; margin: 0.8rem 0; }

  /* ── Input widgets — force dark backgrounds ───────────────────────── */
  [data-testid="stSelectbox"] > div,
  [data-baseweb="select"] > div,
  [data-baseweb="input"],
  [data-testid="stNumberInput"] input,
  [data-testid="stTextInput"] input {
    background-color: #141A23 !important;
    color: #E6EDF3 !important;
    border-color: rgba(255,255,255,0.12) !important;
    border-radius: 6px;
  }
  /* Dropdown list */
  [data-baseweb="popover"] [data-baseweb="menu"],
  [data-baseweb="popover"] ul {
    background-color: #1A2233 !important;
    color: #E6EDF3 !important;
  }
  [data-baseweb="popover"] li:hover {
    background-color: #253043 !important;
  }

  /* ── Sliders ──────────────────────────────────────────────────────── */
  [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background-color: #00E5A8 !important;
  }
  [data-testid="stSlider"] [data-baseweb="slider"] div[data-testid="stSlider"] {
    background-color: #1E2B3C !important;
  }

  /* ── DataFrames ───────────────────────────────────────────────────── */
  [data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 8px;
    overflow: hidden;
  }
  [data-testid="stDataFrame"] [data-testid="stDataFrameResizable"] {
    background-color: #0F1520 !important;
  }

  /* ── Expanders ────────────────────────────────────────────────────── */
  [data-testid="stExpander"] {
    background-color: #141A23 !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 8px;
  }
  [data-testid="stExpander"] summary {
    color: #C9D1D9 !important;
  }

  /* ── Buttons ──────────────────────────────────────────────────────── */
  [data-testid="stDownloadButton"] button,
  [data-testid="stButton"] button {
    background-color: #141A23 !important;
    border: 1px solid rgba(0,229,168,0.3) !important;
    color: #00E5A8 !important;
    border-radius: 6px;
    font-size: 0.82rem;
  }
  [data-testid="stDownloadButton"] button:hover,
  [data-testid="stButton"] button:hover {
    background-color: rgba(0,229,168,0.1) !important;
    border-color: #00E5A8 !important;
  }

  /* ── Alerts / info boxes ──────────────────────────────────────────── */
  [data-testid="stAlert"],
  [data-testid="stNotification"] {
    background-color: #141A23 !important;
    border-radius: 8px;
    color: #C9D1D9 !important;
  }

  /* ── Checkbox ─────────────────────────────────────────────────────── */
  [data-testid="stCheckbox"] label { color: #C9D1D9 !important; }
  [data-testid="stCheckbox"] [data-baseweb="checkbox"] [data-checked="true"] {
    background-color: #00E5A8 !important;
  }

  /* ── Tab styling ──────────────────────────────────────────────────── */
  .stTabs [data-baseweb="tab-list"] { gap: 6px; background: transparent !important; }
  .stTabs [data-baseweb="tab"] {
    background-color: #141A23 !important;
    border-radius: 6px 6px 0 0;
    padding: 7px 18px;
    color: #8899AA !important;
    font-size: 0.85rem;
  }
  .stTabs [aria-selected="true"] {
    background-color: #1E2B3C !important;
    color: #00E5A8 !important;
    border-bottom: 2px solid #00E5A8;
  }

  /* ── Metric (fallback if st.metric used) ─────────────────────────── */
  div[data-testid="stMetric"] {
    background-color: #141A23 !important;
    border-radius: 8px;
    padding: 12px 16px;
  }
  div[data-testid="stMetric"] label { color: #8899AA !important; }
  div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #E6EDF3 !important; }
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
        "- `data/cleaned_edges.csv`\n- `data/network_edge_list.csv`\n\n"
        "Run the scraper first or copy the CSV files into the `data/` directory."
    )
    st.stop()

nodes_df = data["nodes"]

# =========================================================================
# Graph + metrics (cached)
# =========================================================================
G             = build_graph(edges_df, directed=True)
metrics_df    = compute_metrics(G)
communities   = detect_communities(G)
summary       = graph_summary(G)
comm_counts   = Counter(communities.values())

metrics_df["community"] = metrics_df.index.map(lambda n: communities.get(n, -1))

has_dates = (
    "datetime" in edges_df.columns
    and edges_df["datetime"].notna().sum() > 10
)
if has_dates:
    valid_dates = edges_df["datetime"].dropna()
    date_min    = valid_dates.min()
    date_max    = valid_dates.max()


# =========================================================================
# Helpers
# =========================================================================

def _kpi(label: str, value: str, subtitle: str = "",
         color: str = "#00E5A8") -> str:
    """Return HTML for a styled KPI card."""
    sub = (f"<div style='color:#8899AA;font-size:0.73rem;margin-top:4px;'>"
           f"{subtitle}</div>") if subtitle else ""
    return f"""
<div style='background:#141A23;border:1px solid rgba(255,255,255,0.07);
border-left:3px solid {color};border-radius:8px;padding:16px 20px;height:100%;'>
  <div style='color:#8899AA;font-size:0.67rem;text-transform:uppercase;
  letter-spacing:0.09em;margin-bottom:6px;'>{label}</div>
  <div style='color:#E6EDF3;font-size:1.55rem;font-weight:700;
  line-height:1.1;letter-spacing:-0.02em;'>{value}</div>
  {sub}
</div>"""


def _section_note(text: str):
    st.markdown(
        f"<div class='caption-text' style='margin-bottom:1rem;'>{text}</div>",
        unsafe_allow_html=True,
    )


def _metric_label(m: str) -> str:
    return {
        "weighted_degree": "Total Weight",
        "degree":          "Connections",
        "in_degree":       "Inbound Links",
        "out_degree":      "Outbound Links",
        "in_weighted":     "Messages Received",
        "out_weighted":    "Messages Sent",
        "betweenness":     "Betweenness",
        "eigenvector":     "Eigenvector",
        "community":       "Community",
    }.get(m, m.replace("_", " ").title())


def _fmt_table(df: pd.DataFrame, fmt_cols: dict | None = None) -> pd.DataFrame:
    """Return a copy of df with number columns nicely formatted as strings."""
    out = df.copy()
    for col in out.select_dtypes(include="number").columns:
        if fmt_cols and col in fmt_cols:
            out[col] = out[col].map(fmt_cols[col])
        elif out[col].abs().max() < 1.1:
            out[col] = out[col].map("{:.4f}".format)
        else:
            out[col] = out[col].map("{:,.0f}".format)
    return out


# =========================================================================
# Sidebar
# =========================================================================
with st.sidebar:
    st.markdown(
        "<div style='font-size:1.1rem;font-weight:700;color:#E6EDF3;"
        "letter-spacing:0.01em;margin-bottom:2px;'>🔗 Jmail Network</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='color:#8899AA;font-size:0.78rem;margin-bottom:12px;'>"
        "Email communication intelligence</div>",
        unsafe_allow_html=True,
    )
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
    st.markdown(
        f"<div style='color:#8899AA;font-size:0.75rem;line-height:1.8;'>"
        f"<span style='color:#00E5A8;font-weight:600;'>{summary['num_nodes']:,}</span> nodes<br>"
        f"<span style='color:#4FC3F7;font-weight:600;'>{summary['num_edges']:,}</span> edges<br>"
        f"<span style='color:#BB86FC;font-weight:600;'>{len(comm_counts)}</span> communities"
        f"</div>",
        unsafe_allow_html=True,
    )


# =========================================================================
# Section: Overview
# =========================================================================
if section == "📊 Overview":
    st.header("Network Overview")
    _section_note(
        "A directed graph of email communication — nodes are participants, "
        "edges represent message flows. Edge weight = total message count."
    )

    # Row 1 — core KPIs
    c1, c2, c3, c4 = st.columns(4)
    cards_r1 = [
        ("Participants", f"{summary['num_nodes']:,}", "unique email addresses", "#00E5A8"),
        ("Connections", f"{summary['num_edges']:,}", "directed edges", "#4FC3F7"),
        ("Total Messages", f"{summary['total_weight']:,}", "cumulative weight", "#BB86FC"),
        ("Avg Connections", f"{summary['avg_degree']:.1f}", "per node", "#FF9F43"),
    ]
    for col, (lbl, val, sub, clr) in zip([c1, c2, c3, c4], cards_r1):
        col.markdown(_kpi(lbl, val, sub, clr), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)

    # Row 2 — structural KPIs
    c5, c6, c7, c8 = st.columns(4)
    cards_r2 = [
        ("Communities", f"{len(comm_counts):,}", "Louvain detection", "#00E5A8"),
        ("Giant Component", f"{summary['giant_component_size']:,}", "largest cluster", "#4FC3F7"),
        ("Components", f"{summary['num_components']:,}", "disconnected subgraphs", "#BB86FC"),
        ("Graph Density", f"{summary['density']:.5f}", "0 = sparse, 1 = complete", "#FF9F43"),
    ]
    for col, (lbl, val, sub, clr) in zip([c5, c6, c7, c8], cards_r2):
        col.markdown(_kpi(lbl, val, sub, clr), unsafe_allow_html=True)

    if has_dates:
        st.markdown(
            f"<div style='margin-top:14px;color:#8899AA;font-size:0.82rem;'>"
            f"📅 Date range: "
            f"<span style='color:#E6EDF3;font-weight:600;'>{date_min.strftime('%b %d, %Y')}</span>"
            f" → "
            f"<span style='color:#E6EDF3;font-weight:600;'>{date_max.strftime('%b %d, %Y')}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Top Senders")
        top_out = (
            metrics_df.nlargest(10, "out_weighted")[["out_weighted", "out_degree", "community"]]
            .rename(columns={"out_weighted": "Sent", "out_degree": "Recipients", "community": "Comm"})
        )
        top_out.index.name = "Node"
        st.dataframe(
            top_out,
            column_config={
                "Sent": st.column_config.NumberColumn("Sent", format="%d"),
                "Recipients": st.column_config.NumberColumn("Recipients", format="%d"),
                "Comm": st.column_config.NumberColumn("Community", format="%d"),
            },
            use_container_width=True,
        )

    with col_b:
        st.subheader("Top Recipients")
        top_in = (
            metrics_df.nlargest(10, "in_weighted")[["in_weighted", "in_degree", "community"]]
            .rename(columns={"in_weighted": "Received", "in_degree": "Senders", "community": "Comm"})
        )
        top_in.index.name = "Node"
        st.dataframe(
            top_in,
            column_config={
                "Received": st.column_config.NumberColumn("Received", format="%d"),
                "Senders": st.column_config.NumberColumn("Senders", format="%d"),
                "Comm": st.column_config.NumberColumn("Community", format="%d"),
            },
            use_container_width=True,
        )

    st.divider()
    st.subheader(f"Community Distribution — {len(comm_counts)} communities (Louvain)")
    fig = community_size_chart(sorted(comm_counts.values(), reverse=True))
    st.plotly_chart(fig, use_container_width=True)


# =========================================================================
# Section: Network Graph  (PyVis)
# =========================================================================
elif section == "🌐 Network Graph":
    st.header("Interactive Network Graph")
    _section_note(
        "Physics-based layout — drag nodes to rearrange · scroll to zoom · "
        "hover for details · allow 2–3 s for physics to settle on first load."
    )

    with st.sidebar:
        st.subheader("Graph Controls")

        avg_w = summary["total_weight"] / max(summary["num_edges"], 1)
        default_min_w = max(2, int(avg_w * 0.5))
        min_weight = st.slider(
            "Min edge weight", 1, max(20, int(avg_w * 4)), default_min_w,
            help="Hide edges with fewer messages than this threshold.",
        )
        max_nodes = st.slider(
            "Max nodes shown", 10, min(200, summary["num_nodes"]),
            min(50, summary["num_nodes"]),
            help="Show only the top-N most connected nodes. Lower = cleaner graph.",
        )
        giant_only = st.checkbox(
            "Giant component only", value=True,
            help="Restrict to the largest connected subgraph.",
        )
        all_nodes = sorted(G.nodes())
        search_node = st.selectbox(
            "Highlight node", ["(none)"] + all_nodes,
            help="Pin a node in gold. Clicking it in the graph dims non-neighbors.",
        )
        if search_node == "(none)":
            search_node = None

        label_n = st.slider(
            "Label top-N nodes", 3, 20, 6,
            help="Only label the highest-weight nodes. Keep low to reduce clutter.",
        )

    H = filter_graph(
        G, min_weight=min_weight, max_nodes=max_nodes,
        giant_only=giant_only, highlight_node=search_node,
    )

    if H.number_of_nodes() == 0:
        st.warning(
            "No nodes remain after filtering. "
            "Lower the minimum edge weight or disable 'Giant component only'."
        )
    else:
        st.markdown(
            f"<div style='color:#8899AA;font-size:0.78rem;margin-bottom:8px;'>"
            f"Showing <span style='color:#E6EDF3;font-weight:600;'>{H.number_of_nodes()}</span> nodes · "
            f"<span style='color:#E6EDF3;font-weight:600;'>{H.number_of_edges()}</span> edges "
            f"· Node size = message weight · Color = community</div>",
            unsafe_allow_html=True,
        )
        html = pyvis_network(
            H, communities,
            highlight_node=search_node,
            label_top_n=label_n,
        )
        components.html(html, height=690, scrolling=False)

        with st.expander("⬇ Download filtered edge list"):
            filt_df = pd.DataFrame([
                {"sender": u, "recipient": v, "weight": d.get("weight", 1)}
                for u, v, d in H.edges(data=True)
            ])
            st.download_button(
                "Download CSV", filt_df.to_csv(index=False),
                "filtered_edges.csv", "text/csv",
            )


# =========================================================================
# Section: Top Nodes
# =========================================================================
elif section == "🏆 Top Nodes":
    st.header("Most Important Participants")
    _section_note("Rank participants by different network centrality metrics.")

    metric_options = [
        "weighted_degree", "out_weighted", "in_weighted",
        "degree", "out_degree", "in_degree",
        "betweenness", "eigenvector",
    ]
    col_m, col_n = st.columns([2, 1])
    with col_m:
        metric_choice = st.selectbox(
            "Rank by", metric_options, format_func=_metric_label,
        )
    with col_n:
        n_show = st.slider("Show top", 10, 50, 20)

    top = metrics_df.nlargest(n_show, metric_choice).sort_values(metric_choice, ascending=True)

    fig = bar_chart(
        top.reset_index(),
        x="node", y=metric_choice,
        title=f"Top {n_show} — {_metric_label(metric_choice)}",
        horizontal=True,
        height=max(380, n_show * 22),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Full Metrics Table")
    display_cols = [
        "degree", "weighted_degree", "in_weighted", "out_weighted",
        "betweenness", "eigenvector", "community",
    ]
    table = metrics_df.nlargest(n_show, metric_choice)[display_cols]
    table.index.name = "Node"

    st.dataframe(
        table,
        column_config={
            "degree":         st.column_config.NumberColumn("Connections", format="%d"),
            "weighted_degree":st.column_config.NumberColumn("Total Weight", format="%d"),
            "in_weighted":    st.column_config.NumberColumn("Received", format="%d"),
            "out_weighted":   st.column_config.NumberColumn("Sent", format="%d"),
            "betweenness":    st.column_config.NumberColumn("Betweenness", format="%.4f"),
            "eigenvector":    st.column_config.NumberColumn("Eigenvector", format="%.4f"),
            "community":      st.column_config.NumberColumn("Community", format="%d"),
        },
        use_container_width=True,
    )
    st.download_button(
        "⬇ Download CSV", table.to_csv(), "top_nodes.csv", "text/csv",
    )


# =========================================================================
# Section: Communities
# =========================================================================
elif section == "🧩 Communities":
    st.header("Community Detection")
    _section_note(
        "Communities are detected using the Louvain algorithm on the "
        "undirected projection of the email graph."
    )

    c1, c2, c3 = st.columns(3)
    c1.markdown(_kpi("Communities", f"{len(comm_counts):,}", "Louvain clusters", "#00E5A8"),
                unsafe_allow_html=True)
    c2.markdown(_kpi("Largest Community",
                     f"{max(comm_counts.values()):,} members", "most connected cluster", "#4FC3F7"),
                unsafe_allow_html=True)
    c3.markdown(_kpi("Median Size",
                     f"{int(pd.Series(list(comm_counts.values())).median())}",
                     "members per community", "#BB86FC"),
                unsafe_allow_html=True)

    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    fig = community_size_chart(sorted(comm_counts.values(), reverse=True))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Explore a Community")

    sorted_comms = sorted(comm_counts.keys(), key=lambda c: comm_counts[c], reverse=True)
    selected_comm = st.selectbox(
        "Select community", sorted_comms,
        format_func=lambda c: f"Community {c}  ·  {comm_counts[c]} members",
    )

    members = [n for n, c in communities.items() if c == selected_comm]
    member_metrics = (
        metrics_df.loc[metrics_df.index.isin(members)]
        .sort_values("weighted_degree", ascending=False)
    )

    mc1, mc2, mc3 = st.columns(3)
    mc1.markdown(_kpi("Members", f"{len(members):,}", f"community {selected_comm}", "#00E5A8"),
                 unsafe_allow_html=True)
    if not member_metrics.empty:
        top_sender = member_metrics["out_weighted"].idxmax()
        top_recv   = member_metrics["in_weighted"].idxmax()
        mc2.markdown(_kpi("Top Sender", top_sender, "most outbound messages", "#4FC3F7"),
                     unsafe_allow_html=True)
        mc3.markdown(_kpi("Top Recipient", top_recv, "most inbound messages", "#BB86FC"),
                     unsafe_allow_html=True)

    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    disp = member_metrics[
        ["degree", "weighted_degree", "in_weighted", "out_weighted", "betweenness"]
    ].head(20)
    disp.index.name = "Node"
    st.dataframe(
        disp,
        column_config={
            "degree":         st.column_config.NumberColumn("Connections", format="%d"),
            "weighted_degree":st.column_config.NumberColumn("Total Weight", format="%d"),
            "in_weighted":    st.column_config.NumberColumn("Received", format="%d"),
            "out_weighted":   st.column_config.NumberColumn("Sent", format="%d"),
            "betweenness":    st.column_config.NumberColumn("Betweenness", format="%.4f"),
        },
        use_container_width=True,
    )

    with st.expander("🌐 View community subgraph (PyVis)"):
        sub_G = G.subgraph(set(members)).copy()
        if sub_G.number_of_nodes() > 0:
            if sub_G.number_of_nodes() > 150:
                st.caption(f"Showing top 150 of {sub_G.number_of_nodes()} members.")
                wdeg = dict(sub_G.degree(weight="weight"))
                top_m = sorted(wdeg, key=wdeg.get, reverse=True)[:150]
                sub_G = sub_G.subgraph(top_m).copy()
            html = pyvis_network(
                sub_G, communities,
                height="520px",
                label_top_n=10,
            )
            components.html(html, height=540, scrolling=False)


# =========================================================================
# Section: Ego Network
# =========================================================================
elif section == "👤 Ego Network":
    st.header("Ego Network Explorer")
    _section_note(
        "Explore the immediate communication neighbourhood around any participant. "
        "Radius 1 = direct contacts only · radius 2 = contacts-of-contacts."
    )

    all_nodes_sorted = metrics_df.sort_values("weighted_degree", ascending=False).index.tolist()
    col_sel, col_rad = st.columns([3, 1])
    with col_sel:
        ego_node = st.selectbox("Select participant", all_nodes_sorted)
    with col_rad:
        radius = st.radio("Radius", [1, 2], horizontal=True)

    ego_G = get_ego_graph(G, ego_node, radius=radius)

    if ego_G is None or ego_G.number_of_nodes() == 0:
        st.warning("Node not found or has no connections.")
    else:
        node_stats = metrics_df.loc[ego_node] if ego_node in metrics_df.index else None
        if node_stats is not None:
            c1, c2, c3, c4, c5 = st.columns(5)
            kpi_data = [
                ("Connections", f"{int(node_stats['degree']):,}", "#00E5A8"),
                ("Sent", f"{int(node_stats['out_weighted']):,}", "#4FC3F7"),
                ("Received", f"{int(node_stats['in_weighted']):,}", "#BB86FC"),
                ("Betweenness", f"{node_stats['betweenness']:.4f}", "#FF9F43"),
                ("Community", f"{int(node_stats['community'])}", "#FF6B6B"),
            ]
            for col, (lbl, val, clr) in zip([c1, c2, c3, c4, c5], kpi_data):
                col.markdown(_kpi(lbl, val, color=clr), unsafe_allow_html=True)

        if has_dates:
            node_msgs = edges_df[
                (edges_df["sender"] == ego_node) | (edges_df["recipient"] == ego_node)
            ]
            valid = node_msgs["datetime"].dropna()
            if len(valid) > 0:
                st.markdown(
                    f"<div style='color:#8899AA;font-size:0.8rem;margin-top:10px;'>"
                    f"Active: <span style='color:#E6EDF3;'>{valid.min().strftime('%b %d, %Y')}</span>"
                    f" → <span style='color:#E6EDF3;'>{valid.max().strftime('%b %d, %Y')}</span>"
                    f" &nbsp;·&nbsp; {len(node_msgs):,} messages"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.divider()
        col_ties, col_graph = st.columns([1, 2])

        with col_ties:
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
                st.dataframe(
                    agg,
                    column_config={
                        "outgoing": st.column_config.NumberColumn("Sent", format="%d"),
                        "incoming": st.column_config.NumberColumn("Received", format="%d"),
                        "total":    st.column_config.NumberColumn("Total", format="%d"),
                    },
                    use_container_width=True,
                    hide_index=True,
                )

        with col_graph:
            st.subheader(f"Ego Graph — radius {radius}")
            if ego_G.number_of_nodes() > 250:
                st.caption(f"{ego_G.number_of_nodes()} nodes — showing top 250 by weight.")
                wdeg = {n: ego_G.degree(n, weight="weight") for n in ego_G.nodes()}
                top = sorted(wdeg, key=wdeg.get, reverse=True)[:250]
                if ego_node not in top:
                    top.append(ego_node)
                ego_G = ego_G.subgraph(top).copy()

            html = pyvis_network(
                ego_G, communities,
                highlight_node=ego_node,
                height="480px",
                label_top_n=8,
            )
            components.html(html, height=500, scrolling=False)


# =========================================================================
# Section: Relationships
# =========================================================================
elif section == "🔥 Relationships":
    st.header("Relationship Strength")
    _section_note("Explore message volume between specific participants or across the top-N.")

    mode = st.radio(
        "View mode",
        ["Top-N interaction matrix", "Node spotlight"],
        horizontal=True,
    )

    if mode == "Top-N interaction matrix":
        n = st.slider("Top N nodes", 5, 30, 15)
        top_nodes = metrics_df.nlargest(n, "weighted_degree").index.tolist()
        matrix = build_interaction_matrix(G, top_nodes)
        fig = heatmap(
            matrix,
            title=f"Message Volume — Top {n} participants",
            height=max(430, n * 32),
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
                title=f"Top contacts: {sel}",
                horizontal=True,
                height=max(320, len(cdf) * 26),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                cdf,
                column_config={"messages": st.column_config.NumberColumn("Messages", format="%d")},
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No connections found for this node.")

    st.divider()
    st.subheader("Strongest Connections Overall")
    strong = [
        {"sender": u, "recipient": v, "messages": d.get("weight", 1)}
        for u, v, d in G.edges(data=True)
    ]
    strong_df = (
        pd.DataFrame(strong)
        .sort_values("messages", ascending=False)
        .head(25)
    )
    st.dataframe(
        strong_df,
        column_config={"messages": st.column_config.NumberColumn("Messages", format="%d")},
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "⬇ Download CSV", strong_df.to_csv(index=False),
        "strongest_connections.csv", "text/csv",
    )


# =========================================================================
# Section: Timeline
# =========================================================================
elif section == "📅 Timeline":
    st.header("Activity Timeline")
    _section_note("Message volume and new relationship formation over time.")

    if not has_dates:
        # Graceful no-data card
        st.markdown(
            f"""
<div style='background:#141A23;border:1px solid rgba(255,107,107,0.3);
border-left:3px solid #FF6B6B;border-radius:8px;padding:20px 24px;
margin-top:12px;'>
  <div style='color:#FF6B6B;font-weight:700;font-size:0.95rem;margin-bottom:6px;'>
    📅 No timestamp data available
  </div>
  <div style='color:#8899AA;font-size:0.83rem;line-height:1.7;'>
    The edge data does not contain a recognised <code style='color:#E6EDF3;'>datetime</code>
    column, or fewer than 10 valid timestamps were found.<br><br>
    To enable the timeline, ensure the CSV has a column named
    <code style='color:#E6EDF3;'>datetime</code>, <code style='color:#E6EDF3;'>date</code>,
    or <code style='color:#E6EDF3;'>timestamp</code> with parseable date values.
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        valid = edges_df["datetime"].dropna()

        c1, c2, c3 = st.columns(3)
        c1.markdown(_kpi("First Message", date_min.strftime("%b %d, %Y"), color="#4FC3F7"),
                    unsafe_allow_html=True)
        c2.markdown(_kpi("Last Message", date_max.strftime("%b %d, %Y"), color="#4FC3F7"),
                    unsafe_allow_html=True)
        span_days = (date_max - date_min).days
        c3.markdown(_kpi("Active Span", f"{span_days:,} days", color="#4FC3F7"),
                    unsafe_allow_html=True)

        st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)

        freq = st.radio("Aggregation", ["Month", "Week", "Day"], horizontal=True)
        freq_map = {"Month": "M", "Week": "W", "Day": "D"}

        fig = timeline_chart(valid, freq=freq_map[freq],
                             title=f"Email Volume by {freq}")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = cumulative_chart(valid, title="Cumulative Messages Over Time")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("New Relationships Over Time")
        _section_note("When each unique sender → recipient pair first appeared.")
        first_seen = (
            edges_df.dropna(subset=["datetime"])
            .sort_values("datetime")
            .assign(pair=lambda df: df["sender"] + " → " + df["recipient"])
            .drop_duplicates("pair")
        )
        if len(first_seen) > 1:
            fig3 = timeline_chart(
                first_seen["datetime"],
                freq=freq_map[freq],
                title=f"New Unique Relationships by {freq}",
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Not enough relationship data to plot this chart.")
