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

# Fixed categorical palette for up to 30 communities
COMMUNITY_PALETTE = (
    px.colors.qualitative.Plotly
    + px.colors.qualitative.Safe
    + px.colors.qualitative.Vivid
)


def _dark_layout(fig, title: str = "", height: int = 400):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        title=dict(text=title, font=dict(size=15, color="#e0e0e0")),
        margin=dict(l=40, r=20, t=50, b=40),
        height=height,
        font=dict(size=12, color="#cccccc"),
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
    fig.update_traces(marker_line_width=0)
    return _dark_layout(fig, title, height)


# ---------------------------------------------------------------------------
# Community size distribution
# ---------------------------------------------------------------------------

def community_size_chart(sizes: list[int], title="Community Size Distribution") -> go.Figure:
    df = pd.DataFrame({"community": range(len(sizes)), "size": sizes})
    df = df.sort_values("size", ascending=False).head(30).reset_index(drop=True)
    df["rank"] = df.index + 1
    fig = px.bar(
        df, x="rank", y="size",
        color="size", color_continuous_scale="Teal",
        hover_data={"rank": False, "community": True, "size": True},
    )
    fig.update_traces(marker_line_width=0)
    fig.update_xaxes(title="Community (ranked by size)", dtick=1)
    fig.update_yaxes(title="Members")
    fig.update_coloraxes(showscale=False)
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
        hovertemplate="<b>%{y}</b> → <b>%{x}</b><br>Messages: %{z}<extra></extra>",
    ))
    fig.update_xaxes(tickangle=45, side="bottom")
    fig.update_yaxes(autorange="reversed")
    return _dark_layout(fig, title, height)


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------

def timeline_chart(dates: pd.Series, freq: str = "M",
                   title: str = "Email Activity Over Time") -> go.Figure:
    counts = dates.dt.to_period(freq).value_counts().sort_index()
    df = pd.DataFrame({"period": counts.index.astype(str), "count": counts.values})
    fig = px.area(df, x="period", y="count", color_discrete_sequence=[ACCENT])
    fig.update_traces(
        line=dict(width=2),
        fillcolor="rgba(0,212,170,0.12)",
    )
    fig.update_xaxes(title="Period", tickangle=45)
    fig.update_yaxes(title="Messages")
    return _dark_layout(fig, title)


def cumulative_chart(dates: pd.Series, title: str = "Cumulative Messages") -> go.Figure:
    s = dates.sort_values().reset_index(drop=True)
    df = pd.DataFrame({"date": s, "cumulative": range(1, len(s) + 1)})
    fig = px.line(df, x="date", y="cumulative", color_discrete_sequence=[ACCENT])
    fig.update_traces(line=dict(width=2))
    fig.update_yaxes(title="Total Messages")
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
    height: int = 750,
    highlight_node: str | None = None,
    label_top_n: int = 12,
    categorical: bool = False,
) -> go.Figure:
    """
    Render a NetworkX graph as an interactive Plotly scatter plot.

    pos           : dict of {node: (x, y)}
    highlight_node: node to visually emphasize (gold border, always labeled)
    label_top_n   : show text labels for top-N nodes by weighted degree
    categorical   : if True, treat node_colors as integer community IDs
                    and assign distinct colors from a qualitative palette.
    """
    if not pos:
        return go.Figure()

    traces = []

    # ── Edge traces: two tiers (heavy vs light) ────────────────────────────
    if G.number_of_edges() > 0:
        all_weights = [d.get("weight", 1) for _, _, d in G.edges(data=True)]
        median_w = float(np.median(all_weights)) if all_weights else 1.0

        for is_heavy, color, width in [
            (True,  "#60608a", 0.9),   # above-median weight: more visible
            (False, "#2a2a44", 0.35),  # below-median weight: subtle
        ]:
            ex, ey = [], []
            for u, v, d in G.edges(data=True):
                w = d.get("weight", 1)
                if (w >= median_w) != is_heavy:
                    continue
                if u not in pos or v not in pos:
                    continue
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                ex += [x0, x1, None]
                ey += [y0, y1, None]
            if ex:
                traces.append(go.Scatter(
                    x=ex, y=ey,
                    mode="lines",
                    line=dict(width=width, color=color),
                    hoverinfo="none",
                    showlegend=False,
                ))

    # ── Top-N nodes that get text labels ──────────────────────────────────
    wdeg = {n: G.degree(n, weight="weight") for n in G.nodes()}
    top_n_set = set(sorted(wdeg, key=wdeg.get, reverse=True)[:label_top_n])
    if highlight_node and highlight_node in G:
        top_n_set.add(highlight_node)

    # ── Regular nodes and highlighted node as separate traces ─────────────
    regular = [n for n in G.nodes() if n in pos and n != highlight_node]
    special = [highlight_node] if (highlight_node and highlight_node in G and highlight_node in pos) else []

    for group, node_list in [("regular", regular), ("highlight", special)]:
        if not node_list:
            continue

        xs = [pos[n][0] for n in node_list]
        ys = [pos[n][1] for n in node_list]
        sizes = [node_sizes.get(n, 7) if node_sizes else 7 for n in node_list]

        if group == "highlight":
            sizes = [max(s, 22) for s in sizes]

        # Build marker dict based on color mode
        if categorical and node_colors is not None:
            colors = [
                COMMUNITY_PALETTE[int(node_colors.get(n, 0)) % len(COMMUNITY_PALETTE)]
                for n in node_list
            ]
            marker = dict(
                size=sizes,
                color=colors,
                line=dict(
                    width=3 if group == "highlight" else 0.8,
                    color="#FFD700" if group == "highlight" else "rgba(255,255,255,0.12)",
                ),
                opacity=0.93,
            )
        else:
            raw_colors = [node_colors.get(n, 0) if node_colors else 0 for n in node_list]
            marker = dict(
                size=sizes,
                color=raw_colors,
                colorscale="Turbo",
                cmin=min(raw_colors) if raw_colors else 0,
                cmax=max(raw_colors) if raw_colors else 1,
                line=dict(
                    width=3 if group == "highlight" else 0.8,
                    color="#FFD700" if group == "highlight" else "rgba(255,255,255,0.12)",
                ),
                showscale=(group == "regular"),
                colorbar=dict(
                    title=dict(text=color_label, font=dict(size=11)),
                    thickness=12,
                    len=0.55,
                    x=1.01,
                ) if group == "regular" else None,
                opacity=0.93,
            )

        # Hover text
        hover = []
        for n in node_list:
            deg = G.degree(n)
            wd = G.degree(n, weight="weight")
            cv = node_colors.get(n, 0) if node_colors else 0
            if categorical:
                extra = f"Community: {cv}"
            else:
                extra = (f"{color_label}: {cv:.1f}" if isinstance(cv, float)
                         else f"{color_label}: {cv}")
            hover.append(
                f"<b>{n}</b><br>"
                f"Connections: {deg}<br>"
                f"Message weight: {wd:,}<br>"
                f"{extra}"
            )

        text_labels = [n if n in top_n_set else "" for n in node_list]

        traces.append(go.Scatter(
            x=xs, y=ys,
            mode="markers+text",
            marker=marker,
            text=text_labels,
            textposition="top center",
            textfont=dict(size=9, color="rgba(210,210,225,0.85)"),
            hovertext=hover,
            hovertemplate="%{hovertext}<extra></extra>",
            showlegend=False,
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        showlegend=False,
        title=dict(text=title, font=dict(size=14, color="#dddddd")),
        height=height,
        margin=dict(l=5, r=5, t=40, b=5),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        hovermode="closest",
        dragmode="pan",
    )
    return fig
