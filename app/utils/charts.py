"""
Reusable chart-building functions (Plotly).
All charts return plotly Figure objects for display in Streamlit.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Colour palette (dark-theme friendly)
# ---------------------------------------------------------------------------
PALETTE = px.colors.qualitative.Set2
ACCENT = "#00d4aa"   # teal accent
BG = "rgba(0,0,0,0)"


def _dark_layout(fig, title: str = "", height: int = 400):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        title=dict(text=title, font_size=16),
        margin=dict(l=40, r=20, t=50, b=40),
        height=height,
        font=dict(size=12),
    )
    return fig


# ---------------------------------------------------------------------------
# Bar charts
# ---------------------------------------------------------------------------

def bar_chart(df: pd.DataFrame, x: str, y: str, title: str = "",
              color: str | None = None, horizontal: bool = True,
              height: int = 400) -> go.Figure:
    if horizontal:
        fig = px.bar(df, x=y, y=x, orientation="h", color=color,
                     color_discrete_sequence=PALETTE)
    else:
        fig = px.bar(df, x=x, y=y, color=color,
                     color_discrete_sequence=PALETTE)
    return _dark_layout(fig, title, height)


# ---------------------------------------------------------------------------
# Community size distribution
# ---------------------------------------------------------------------------

def community_size_chart(sizes: list[int], title="Community Size Distribution") -> go.Figure:
    df = pd.DataFrame({"community": range(len(sizes)), "size": sizes})
    df = df.sort_values("size", ascending=False).head(30)
    fig = px.bar(df, x="community", y="size", color_discrete_sequence=[ACCENT])
    fig.update_xaxes(title="Community ID", dtick=1)
    fig.update_yaxes(title="Members")
    return _dark_layout(fig, title)


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

def heatmap(matrix: pd.DataFrame, title: str = "Interaction Heatmap",
            height: int = 500) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=matrix.values,
        x=matrix.columns.tolist(),
        y=matrix.index.tolist(),
        colorscale="Teal",
        hoverongaps=False,
    ))
    fig.update_xaxes(tickangle=45)
    return _dark_layout(fig, title, height)


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------

def timeline_chart(dates: pd.Series, freq: str = "M",
                   title: str = "Email Activity Over Time") -> go.Figure:
    counts = dates.dt.to_period(freq).value_counts().sort_index()
    df = pd.DataFrame({"period": counts.index.astype(str), "count": counts.values})
    fig = px.area(df, x="period", y="count", color_discrete_sequence=[ACCENT])
    fig.update_xaxes(title="Period", tickangle=45)
    fig.update_yaxes(title="Messages")
    return _dark_layout(fig, title)


def cumulative_chart(dates: pd.Series, title: str = "Cumulative Messages") -> go.Figure:
    s = dates.sort_values().reset_index(drop=True)
    df = pd.DataFrame({"date": s, "cumulative": range(1, len(s) + 1)})
    fig = px.line(df, x="date", y="cumulative", color_discrete_sequence=[ACCENT])
    return _dark_layout(fig, title)


# ---------------------------------------------------------------------------
# Network graph (Plotly-based for embedding)
# ---------------------------------------------------------------------------

def plotly_network(
    G,
    pos: dict,
    node_sizes: dict | None = None,
    node_colors: dict | None = None,
    color_label: str = "community",
    title: str = "",
    height: int = 650,
) -> go.Figure:
    """
    Render a NetworkX graph as an interactive Plotly scatter plot.
    pos: dict of {node: (x, y)}
    """
    # Edge traces
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.3, color="#444"),
        hoverinfo="none",
    )

    # Node traces
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    sizes = [node_sizes.get(n, 5) if node_sizes else 5 for n in G.nodes()]
    colors = [node_colors.get(n, 0) if node_colors else 0 for n in G.nodes()]
    labels = list(G.nodes())

    # Hover text with degree info
    hover = []
    for n in G.nodes():
        deg = G.degree(n)
        wdeg = G.degree(n, weight="weight")
        hover.append(f"{n}<br>Degree: {deg}<br>Weighted: {wdeg}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers",
        marker=dict(
            size=sizes,
            color=colors,
            colorscale="Turbo",
            line=dict(width=0.5, color="#222"),
            showscale=True,
            colorbar=dict(title=color_label, thickness=15),
        ),
        text=labels,
        hovertext=hover,
        hoverinfo="text",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        showlegend=False,
        title=dict(text=title, font_size=16),
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig
