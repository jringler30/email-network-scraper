"""
Jmail Network Explorer — Interactive email network intelligence dashboard.

Run with:
    streamlit run app/app.py
"""

import math
import sys
from collections import Counter
from pathlib import Path

# Ensure the app/ directory is on sys.path so `utils` resolves correctly
# regardless of the working directory (local vs Streamlit Cloud).
sys.path.insert(0, str(Path(__file__).resolve().parent))

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
    bar_chart, community_size_chart, sankey_flow, pyvis_network,
)
from utils.network_views import filter_graph

# =========================================================================
# Page config
# =========================================================================
st.set_page_config(
    page_title="Jmail Network Explorer",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================================
# Global CSS — sleek dark analytics theme
# =========================================================================
st.markdown("""
<style>
  /* ── Root dark enforcement ─────────────────────────────────────────── */
  html, body {
    color-scheme: dark !important;
    background-color: #0B0F17 !important;
    color: #E6EDF3 !important;
  }

  /* ── App containers ────────────────────────────────────────────────── */
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

  /* Hide the sidebar toggle and sidebar entirely */
  [data-testid="stSidebar"],
  [data-testid="collapsedControl"] {
    display: none !important;
  }

  /* ── Block container ───────────────────────────────────────────────── */
  .block-container,
  [data-testid="stMainBlockContainer"] > div {
    background-color: #0B0F17 !important;
    padding-top: 0 !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1600px;
  }

  /* ── Typography ────────────────────────────────────────────────────── */
  .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 {
    color: #E6EDF3 !important;
  }
  .stApp h2 {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding-bottom: 0.4rem;
    margin-bottom: 0.9rem;
  }
  .stApp h3 { font-size: 0.97rem !important; font-weight: 600 !important; }
  .stApp p  { color: #C9D1D9 !important; }
  .stApp label { color: #C9D1D9 !important; }
  .stCaption, .caption-text {
    color: #8899AA !important;
    font-size: 0.8rem;
  }

  /* ── Dividers ─────────────────────────────────────────────────────── */
  hr { border-color: rgba(255,255,255,0.06) !important; margin: 0.75rem 0; }

  /* ── Tab navigation — pill style ───────────────────────────────────── */
  .stTabs [data-baseweb="tab-list"] {
    background: #0F1520 !important;
    border-radius: 10px !important;
    padding: 5px 6px !important;
    gap: 3px !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.35) !important;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 7px !important;
    padding: 8px 20px !important;
    color: #8899AA !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    border: none !important;
    letter-spacing: 0.01em;
    transition: color 0.15s, background 0.15s;
    white-space: nowrap;
  }
  .stTabs [data-baseweb="tab"]:hover {
    color: #C9D1D9 !important;
    background: rgba(255,255,255,0.04) !important;
  }
  .stTabs [aria-selected="true"] {
    background: #1A2840 !important;
    color: #00E5A8 !important;
    font-weight: 600 !important;
    box-shadow: 0 0 0 1px rgba(0,229,168,0.2) inset !important;
  }
  /* Hide the default bottom border indicator */
  .stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
  }
  .stTabs [data-baseweb="tab-border"] {
    display: none !important;
  }
  /* Tab panel bg */
  .stTabs [data-baseweb="tab-panel"] {
    background: transparent !important;
    padding-top: 1.4rem !important;
  }

  /* ── Input widgets ─────────────────────────────────────────────────── */
  [data-testid="stSelectbox"] > div,
  [data-baseweb="select"] > div,
  [data-baseweb="input"],
  [data-testid="stNumberInput"] input,
  [data-testid="stTextInput"] input {
    background-color: #141A23 !important;
    color: #E6EDF3 !important;
    border-color: rgba(255,255,255,0.1) !important;
    border-radius: 6px !important;
  }
  [data-baseweb="popover"] [data-baseweb="menu"],
  [data-baseweb="popover"] ul {
    background-color: #1A2233 !important;
    color: #E6EDF3 !important;
  }
  [data-baseweb="popover"] li:hover {
    background-color: #253043 !important;
  }

  /* ── Sliders ──────────────────────────────────────────────────────── */
  [data-testid="stSlider"] [role="slider"] {
    background-color: #00E5A8 !important;
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
  [data-testid="stExpander"] summary { color: #C9D1D9 !important; }

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

  /* ── Alerts ───────────────────────────────────────────────────────── */
  [data-testid="stAlert"],
  [data-testid="stNotification"] {
    background-color: #141A23 !important;
    border-radius: 8px;
    color: #C9D1D9 !important;
  }

  /* ── Checkboxes ────────────────────────────────────────────────────── */
  [data-testid="stCheckbox"] label { color: #C9D1D9 !important; }
  [data-testid="stCheckbox"] [data-baseweb="checkbox"] [data-checked="true"] {
    background-color: #00E5A8 !important;
  }

  /* ── Radio ─────────────────────────────────────────────────────────── */
  [data-testid="stRadio"] label {
    color: #C9D1D9 !important;
    font-size: 0.88rem;
  }

  /* ── Metric cards ─────────────────────────────────────────────────── */
  div[data-testid="stMetric"] {
    background-color: #141A23 !important;
    border-radius: 8px;
    padding: 12px 16px;
  }
  div[data-testid="stMetric"] label { color: #8899AA !important; }
  div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #E6EDF3 !important;
  }

  /* ── Force dark on any stray white backgrounds ─────────────────────── */
  [data-testid="stVerticalBlock"],
  [data-testid="stHorizontalBlock"] {
    background-color: transparent !important;
  }
  div.stMarkdown { color: #C9D1D9 !important; }
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
G           = build_graph(edges_df, directed=True)
metrics_df  = compute_metrics(G)
communities = detect_communities(G)
summary     = graph_summary(G)
comm_counts = Counter(communities.values())

metrics_df["community"] = metrics_df.index.map(lambda n: communities.get(n, -1))

# Aggregate/catch-all nodes — kept in the graph for structural accuracy but
# excluded from ranked tables, selectors, and key metric displays.
AGGREGATE_NODES = {"Redacted/Unknown", "unknown"}
display_metrics_df = metrics_df[~metrics_df.index.isin(AGGREGATE_NODES)]

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
    sub = (f"<div style='color:#8899AA;font-size:0.72rem;margin-top:4px;'>"
           f"{subtitle}</div>") if subtitle else ""
    return f"""
<div style='background:#0F1520;border:1px solid rgba(255,255,255,0.07);
border-left:3px solid {color};border-radius:8px;padding:16px 20px;height:100%;'>
  <div style='color:#8899AA;font-size:0.65rem;text-transform:uppercase;
  letter-spacing:0.09em;margin-bottom:6px;'>{label}</div>
  <div style='color:#E6EDF3;font-size:1.5rem;font-weight:700;
  line-height:1.1;letter-spacing:-0.02em;'>{value}</div>
  {sub}
</div>"""


def _section_note(text: str):
    st.markdown(
        f"<div class='caption-text' style='margin-bottom:1rem;color:#8899AA;"
        f"font-size:0.8rem;'>{text}</div>",
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


# =========================================================================
# App header
# =========================================================================

_date_range = (
    f'<span style="color:#FF9F43;font-weight:600;">{date_min.strftime("%b %Y")}</span>'
    f'<span style="color:#8899AA;"> – </span>'
    f'<span style="color:#FF9F43;font-weight:600;">{date_max.strftime("%b %Y")}</span>'
    if has_dates else ""
)

st.markdown(
    "<div style='display:flex;align-items:center;justify-content:space-between;"
    "padding:1rem 0 0.8rem 0;border-bottom:1px solid rgba(255,255,255,0.07);"
    "margin-bottom:1rem;'>"
    "<div style='display:flex;align-items:center;gap:12px;'>"
    "<div style='display:flex;align-items:stretch;border-radius:5px;"
    "overflow:hidden;height:42px;box-shadow:0 2px 10px rgba(0,0,0,0.5);flex-shrink:0;'>"
    "<div style='width:9px;background:#4285F4;'></div>"
    "<div style='background:white;width:38px;display:flex;align-items:center;justify-content:center;'>"
    "<span style='font-size:1.35rem;font-weight:900;color:#3c4043;font-family:Arial,sans-serif;line-height:1;'>M</span>"
    "</div>"
    "<div style='width:9px;background:#EA4335;'></div>"
    "</div>"
    "<div style='line-height:1;'>"
    "<div style='font-family:Arial,sans-serif;font-size:2rem;font-weight:400;letter-spacing:-0.3px;color:#E6EDF3;'>"
    "<span style='font-weight:700;'>J</span>mail"
    "</div>"
    "<div style='color:#8899AA;font-size:0.7rem;margin-top:3px;letter-spacing:0.04em;text-transform:uppercase;'>"
    "Email network intelligence"
    "</div>"
    "</div>"
    "</div>"
    f"<div style='display:flex;align-items:center;gap:24px;'>"
    f"<div style='text-align:center;'>"
    f"<div style='color:#00E5A8;font-size:1.1rem;font-weight:700;line-height:1;'>{summary['num_nodes']:,}</div>"
    f"<div style='color:#8899AA;font-size:0.62rem;text-transform:uppercase;letter-spacing:0.08em;margin-top:4px;'>nodes</div>"
    f"</div>"
    f"<div style='width:1px;height:30px;background:rgba(255,255,255,0.08);'></div>"
    f"<div style='text-align:center;'>"
    f"<div style='color:#4FC3F7;font-size:1.1rem;font-weight:700;line-height:1;'>{summary['num_edges']:,}</div>"
    f"<div style='color:#8899AA;font-size:0.62rem;text-transform:uppercase;letter-spacing:0.08em;margin-top:4px;'>edges</div>"
    f"</div>"
    f"<div style='width:1px;height:30px;background:rgba(255,255,255,0.08);'></div>"
    f"<div style='text-align:center;'>"
    f"<div style='color:#BB86FC;font-size:1.1rem;font-weight:700;line-height:1;'>{len(comm_counts):,}</div>"
    f"<div style='color:#8899AA;font-size:0.62rem;text-transform:uppercase;letter-spacing:0.08em;margin-top:4px;'>communities</div>"
    f"</div>"
    + (f"<div style='width:1px;height:30px;background:rgba(255,255,255,0.08);'></div>"
       f"<div style='text-align:center;font-size:0.78rem;'>{_date_range}</div>"
       if has_dates else "")
    + "</div></div>",
    unsafe_allow_html=True,
)


# =========================================================================
# Tab navigation
# =========================================================================
tab_overview, tab_graph, tab_top, tab_comm, tab_ego, tab_rel = st.tabs([
    "  Overview  ",
    "  Network Graph  ",
    "  Top Nodes  ",
    "  Communities  ",
    "  Ego Network  ",
    "  Relationships  ",
])


# =========================================================================
# Tab: Overview
# =========================================================================
with tab_overview:
    st.header("Network Overview")
    _section_note(
        "A directed graph of email communication — nodes are participants, "
        "edges represent message flows. Edge weight = total message count."
    )

    c1, c2, c3, c4 = st.columns(4)
    for col, (lbl, val, sub, clr) in zip([c1, c2, c3, c4], [
        ("Participants",   f"{summary['num_nodes']:,}",    "unique email addresses", "#00E5A8"),
        ("Connections",    f"{summary['num_edges']:,}",    "directed edges",         "#4FC3F7"),
        ("Total Messages", f"{summary['total_weight']:,}", "cumulative weight",      "#BB86FC"),
        ("Avg Connections",f"{summary['avg_degree']:.1f}", "per node",               "#FF9F43"),
    ]):
        col.markdown(_kpi(lbl, val, sub, clr), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)

    c5, c6, c7, c8 = st.columns(4)
    for col, (lbl, val, sub, clr) in zip([c5, c6, c7, c8], [
        ("Communities",      f"{len(comm_counts):,}",                  "Louvain detection",       "#00E5A8"),
        ("Giant Component",  f"{summary['giant_component_size']:,}",   "largest cluster",         "#4FC3F7"),
        ("Components",       f"{summary['num_components']:,}",         "disconnected subgraphs",  "#BB86FC"),
        ("Graph Density",    f"{summary['density']:.5f}",              "0 = sparse, 1 = complete","#FF9F43"),
    ]):
        col.markdown(_kpi(lbl, val, sub, clr), unsafe_allow_html=True)

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Top Senders")
        top_out = (
            display_metrics_df.nlargest(10, "out_weighted")
            [["out_weighted", "out_degree", "community"]]
            .rename(columns={"out_weighted": "Sent", "out_degree": "Recipients", "community": "Comm"})
        )
        top_out.index.name = "Node"
        st.dataframe(
            top_out,
            column_config={
                "Sent":       st.column_config.NumberColumn("Sent",       format="%d"),
                "Recipients": st.column_config.NumberColumn("Recipients", format="%d"),
                "Comm":       st.column_config.NumberColumn("Community",  format="%d"),
            },
            width='stretch',
        )

    with col_b:
        st.subheader("Top Recipients")
        top_in = (
            display_metrics_df.nlargest(10, "in_weighted")
            [["in_weighted", "in_degree", "community"]]
            .rename(columns={"in_weighted": "Received", "in_degree": "Senders", "community": "Comm"})
        )
        top_in.index.name = "Node"
        st.dataframe(
            top_in,
            column_config={
                "Received": st.column_config.NumberColumn("Received",  format="%d"),
                "Senders":  st.column_config.NumberColumn("Senders",   format="%d"),
                "Comm":     st.column_config.NumberColumn("Community", format="%d"),
            },
            width='stretch',
        )

    st.divider()
    st.subheader(f"Community Distribution — {len(comm_counts)} communities (Louvain)")
    fig = community_size_chart(sorted(comm_counts.values(), reverse=True))
    st.plotly_chart(fig, width='stretch', key="overview_comm_chart")


# =========================================================================
# Tab: Network Graph
# =========================================================================
with tab_graph:
    st.header("Interactive Network Graph")
    _section_note(
        "Physics-based layout — drag nodes to rearrange · scroll to zoom · "
        "hover for details · allow 2–3 s for physics to settle on first load."
    )

    # Controls inline (no sidebar needed)
    avg_w = summary["total_weight"] / max(summary["num_edges"], 1)
    default_min_w = max(2, int(avg_w * 0.5))

    ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns([2, 2, 2, 3, 2])
    with ctrl1:
        min_weight = st.slider(
            "Min edge weight", 1, max(20, int(avg_w * 4)), default_min_w,
            help="Hide edges with fewer messages than this threshold.",
        )
    with ctrl2:
        max_nodes = st.slider(
            "Max nodes", 10, min(500, summary["num_nodes"]),
            min(50, summary["num_nodes"]),
            help="Show only the top-N most connected nodes.",
        )
    with ctrl3:
        label_n = st.slider("Label top-N", 3, 20, 6,
                            help="Only label the highest-weight nodes.")
    with ctrl4:
        all_nodes = sorted(G.nodes())
        search_node = st.selectbox(
            "Highlight node", ["(none)"] + all_nodes,
            help="Pin a node in gold.",
        )
        if search_node == "(none)":
            search_node = None
    with ctrl5:
        giant_only = st.checkbox(
            "Giant component only", value=True,
            help="Restrict to the largest connected subgraph.",
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
# Tab: Top Nodes
# =========================================================================
with tab_top:
    st.header("Most Important Participants")
    _section_note("Rank participants by different network centrality metrics.")

    metric_options = [
        "weighted_degree", "out_weighted", "in_weighted",
        "degree", "out_degree", "in_degree",
        "betweenness", "eigenvector",
    ]
    col_m, col_n, _ = st.columns([2, 1, 3])
    with col_m:
        metric_choice = st.selectbox(
            "Rank by", metric_options, format_func=_metric_label,
        )
    with col_n:
        n_show = st.slider("Show top", 10, 50, 20)

    top = display_metrics_df.nlargest(n_show, metric_choice).sort_values(
        metric_choice, ascending=True
    )
    fig = bar_chart(
        top.reset_index(),
        x="node", y=metric_choice,
        title=f"Top {n_show} — {_metric_label(metric_choice)}",
        horizontal=True,
        height=max(380, n_show * 22),
    )
    st.plotly_chart(fig, width='stretch', key="top_nodes_bar")

    st.subheader("Full Metrics Table")
    display_cols = [
        "degree", "weighted_degree", "in_weighted", "out_weighted",
        "betweenness", "eigenvector", "community",
    ]
    table = display_metrics_df.nlargest(n_show, metric_choice)[display_cols]
    table.index.name = "Node"
    st.dataframe(
        table,
        column_config={
            "degree":          st.column_config.NumberColumn("Connections",   format="%d"),
            "weighted_degree": st.column_config.NumberColumn("Total Weight",  format="%d"),
            "in_weighted":     st.column_config.NumberColumn("Received",      format="%d"),
            "out_weighted":    st.column_config.NumberColumn("Sent",          format="%d"),
            "betweenness":     st.column_config.NumberColumn("Betweenness",   format="%.4f"),
            "eigenvector":     st.column_config.NumberColumn("Eigenvector",   format="%.4f"),
            "community":       st.column_config.NumberColumn("Community",     format="%d"),
        },
        width='stretch',
    )
    st.download_button(
        "⬇ Download CSV", table.to_csv(), "top_nodes.csv", "text/csv",
    )


# =========================================================================
# Tab: Communities
# =========================================================================
with tab_comm:
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
    st.plotly_chart(fig, width='stretch', key="comm_tab_dist_chart")

    st.divider()
    st.subheader("Explore a Community")

    sorted_comms = sorted(comm_counts.keys(), key=lambda c: comm_counts[c], reverse=True)
    selected_comm = st.selectbox(
        "Select community", sorted_comms,
        format_func=lambda c: f"Community {c}  ·  {comm_counts[c]} members",
    )

    members = [n for n, c in communities.items() if c == selected_comm]
    member_metrics = (
        display_metrics_df.loc[display_metrics_df.index.isin(members)]
        .sort_values("weighted_degree", ascending=False)
    )

    mc1, mc2, mc3 = st.columns(3)
    mc1.markdown(_kpi("Members", f"{len(members):,}", f"community {selected_comm}", "#00E5A8"),
                 unsafe_allow_html=True)
    if not member_metrics.empty:
        top_sender = member_metrics["out_weighted"].idxmax()
        top_recv   = member_metrics["in_weighted"].idxmax()
        mc2.markdown(_kpi("Top Sender",    top_sender, "most outbound messages", "#4FC3F7"),
                     unsafe_allow_html=True)
        mc3.markdown(_kpi("Top Recipient", top_recv,   "most inbound messages",  "#BB86FC"),
                     unsafe_allow_html=True)

    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    disp = member_metrics[
        ["degree", "weighted_degree", "in_weighted", "out_weighted", "betweenness"]
    ].head(20)
    disp.index.name = "Node"
    st.dataframe(
        disp,
        column_config={
            "degree":          st.column_config.NumberColumn("Connections", format="%d"),
            "weighted_degree": st.column_config.NumberColumn("Total Weight",format="%d"),
            "in_weighted":     st.column_config.NumberColumn("Received",    format="%d"),
            "out_weighted":    st.column_config.NumberColumn("Sent",        format="%d"),
            "betweenness":     st.column_config.NumberColumn("Betweenness", format="%.4f"),
        },
        width='stretch',
    )

    with st.expander("🌐 View community subgraph (PyVis)"):
        sub_G = G.subgraph(set(members)).copy()
        if sub_G.number_of_nodes() > 0:
            if sub_G.number_of_nodes() > 150:
                st.caption(f"Showing top 150 of {sub_G.number_of_nodes()} members.")
                wdeg = dict(sub_G.degree(weight="weight"))
                top_m = sorted(wdeg, key=wdeg.get, reverse=True)[:150]
                sub_G = sub_G.subgraph(top_m).copy()
            html = pyvis_network(sub_G, communities, height="520px", label_top_n=6)
            components.html(html, height=540, scrolling=False)


# =========================================================================
# Tab: Ego Network
# =========================================================================
with tab_ego:
    st.header("Ego Network Explorer")
    _section_note(
        "Explore the immediate communication neighbourhood around any participant. "
        "Radius 1 = direct contacts only · radius 2 = contacts-of-contacts."
    )

    all_nodes_sorted = display_metrics_df.sort_values(
        "weighted_degree", ascending=False
    ).index.tolist()

    col_sel, col_rad, _ = st.columns([3, 1, 2])
    with col_sel:
        ego_node = st.selectbox("Select participant", all_nodes_sorted, key="ego_node_select")
    with col_rad:
        radius = st.radio("Radius", [1, 2], horizontal=True)

    ego_G = get_ego_graph(G, ego_node, radius=radius)

    if ego_G is None or ego_G.number_of_nodes() == 0:
        st.warning("Node not found or has no connections.")
    else:
        node_stats = metrics_df.loc[ego_node] if ego_node in metrics_df.index else None
        if node_stats is not None:
            c1, c2, c3, c4, c5 = st.columns(5)
            for col, (lbl, val, clr) in zip([c1, c2, c3, c4, c5], [
                ("Connections", f"{int(node_stats['degree']):,}",      "#00E5A8"),
                ("Sent",        f"{int(node_stats['out_weighted']):,}", "#4FC3F7"),
                ("Received",    f"{int(node_stats['in_weighted']):,}",  "#BB86FC"),
                ("Betweenness", f"{node_stats['betweenness']:.4f}",    "#FF9F43"),
                ("Community",   f"{int(node_stats['community'])}",     "#FF6B6B"),
            ]):
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
                        "outgoing": st.column_config.NumberColumn("Sent",     format="%d"),
                        "incoming": st.column_config.NumberColumn("Received", format="%d"),
                        "total":    st.column_config.NumberColumn("Total",    format="%d"),
                    },
                    width='stretch',
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
                stable_mode=True,
            )
            components.html(html, height=500, scrolling=False)


# =========================================================================
# Tab: Relationships
# =========================================================================
with tab_rel:
    st.header("Relationship Strength")
    _section_note("Explore message volume between specific participants or across the top-N.")

    mode = st.radio(
        "View mode",
        ["Top flows (Sankey)", "Node spotlight"],
        horizontal=True,
    )

    if mode == "Top flows (Sankey)":
        _, col_n, _ = st.columns([1, 2, 3])
        with col_n:
            n_flows = st.slider(
                "Top N flows", 10, 60, 30,
                help="Show the N strongest sender → recipient message flows.",
            )
        all_edges = [
            (u, v, d.get("weight", 1))
            for u, v, d in G.edges(data=True)
            if u not in AGGREGATE_NODES and v not in AGGREGATE_NODES
        ]
        fig = sankey_flow(
            all_edges,
            communities=communities,
            top_n=n_flows,
            title=f"Top {n_flows} Message Flows",
            height=max(500, n_flows * 16),
        )
        st.plotly_chart(fig, width='stretch', key="rel_sankey")

        with st.expander("⬇ Download flow data"):
            flow_df = (
                pd.DataFrame(all_edges, columns=["sender", "recipient", "messages"])
                .sort_values("messages", ascending=False)
                .head(n_flows)
            )
            st.download_button(
                "Download CSV", flow_df.to_csv(index=False),
                "top_flows.csv", "text/csv",
            )
    else:
        all_nodes_sorted = display_metrics_df.sort_values(
            "weighted_degree", ascending=False
        ).index.tolist()
        _, col_sel, _ = st.columns([1, 3, 2])
        with col_sel:
            sel = st.selectbox("Select participant", all_nodes_sorted, key="rel_node_select")

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
            st.plotly_chart(fig, width='stretch', key="rel_spotlight_bar")
            st.dataframe(
                cdf,
                column_config={"messages": st.column_config.NumberColumn("Messages", format="%d")},
                width='stretch',
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
        width='stretch',
        hide_index=True,
    )
    st.download_button(
        "⬇ Download CSV", strong_df.to_csv(index=False),
        "strongest_connections.csv", "text/csv",
    )
