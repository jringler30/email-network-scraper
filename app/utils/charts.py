"""
Reusable chart-building functions (Plotly + PyVis).
All Plotly charts return go.Figure objects.
pyvis_network() returns an HTML string for st.components.v1.html().
"""

import json
import math
import os
import tempfile

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------
ACCENT   = "#00E5A8"
SECONDARY = "#4FC3F7"
WARNING  = "#FF6B6B"
BG       = "rgba(0,0,0,0)"
CARD_BG  = "#141A23"
PAGE_BG  = "#0B0F17"
TEXT     = "#E6EDF3"
MUTED    = "#8899AA"

PALETTE = [ACCENT, SECONDARY, "#BB86FC", "#FF9F43", WARNING,
           "#FFD700", "#A8E063", "#F368E0", "#1DD1A1", "#74B9FF"]

# Distinct palette for up to 20 communities — vibrant on dark backgrounds
PYVIS_COMMUNITY_COLORS = [
    "#00E5A8", "#4FC3F7", "#FF6B6B", "#FFD700", "#BB86FC",
    "#FF9F43", "#00D2D3", "#A8E063", "#F368E0", "#48DBFB",
    "#1DD1A1", "#FF7675", "#FDCB6E", "#6C5CE7", "#E17055",
    "#74B9FF", "#55EFC4", "#FD79A8", "#FFEAA7", "#A29BFE",
]


def _dark_layout(fig, title: str = "", height: int = 400):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        title=dict(text=title, font=dict(size=14, color=TEXT)),
        margin=dict(l=40, r=20, t=50, b=40),
        height=height,
        font=dict(size=12, color=MUTED),
    )
    return fig


# ---------------------------------------------------------------------------
# Bar chart
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
    fig.update_layout(bargap=0.18)
    return _dark_layout(fig, title, height)


# ---------------------------------------------------------------------------
# Community size chart
# ---------------------------------------------------------------------------

def community_size_chart(sizes: list[int],
                         title: str = "Community Size Distribution") -> go.Figure:
    df = pd.DataFrame({"rank": range(1, len(sizes) + 1), "size": sizes})
    df = df.head(30)
    fig = px.bar(df, x="rank", y="size",
                 color="size",
                 color_continuous_scale=[[0, "#1A2B3C"], [0.5, SECONDARY], [1.0, ACCENT]])
    fig.update_traces(marker_line_width=0)
    fig.update_xaxes(title="Community (by size)", dtick=1)
    fig.update_yaxes(title="Members")
    fig.update_coloraxes(showscale=False)
    return _dark_layout(fig, title)


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

def heatmap(matrix: pd.DataFrame, title: str = "Interaction Matrix",
            height: int = 500) -> go.Figure:
    colorscale = [
        [0.00, PAGE_BG],
        [0.15, "#0D2137"],
        [0.40, "#0D4F6C"],
        [0.70, SECONDARY],
        [0.90, ACCENT],
        [1.00, "#FFFFFF"],
    ]
    # Mask zeros so empty cells stay dark
    z = matrix.values.astype(float)
    z_masked = np.where(z == 0, np.nan, z)

    fig = go.Figure(data=go.Heatmap(
        z=z_masked,
        x=matrix.columns.tolist(),
        y=matrix.index.tolist(),
        colorscale=colorscale,
        hoverongaps=False,
        hovertemplate="<b>%{y}</b> → <b>%{x}</b><br>Messages: %{z:.0f}<extra></extra>",
        showscale=True,
        colorbar=dict(
            thickness=12,
            title=dict(text="msgs", font=dict(size=10, color=MUTED)),
            tickfont=dict(size=9, color=MUTED),
        ),
    ))
    fig.update_xaxes(tickangle=40, tickfont=dict(size=9), side="bottom")
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=9))
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
        line=dict(width=2, color=ACCENT),
        fillcolor="rgba(0,229,168,0.10)",
    )
    fig.update_xaxes(title="Period", tickangle=40, tickfont=dict(size=10))
    fig.update_yaxes(title="Messages")
    return _dark_layout(fig, title)


def cumulative_chart(dates: pd.Series,
                     title: str = "Cumulative Messages Over Time") -> go.Figure:
    s = dates.sort_values().reset_index(drop=True)
    df = pd.DataFrame({"date": s, "cumulative": range(1, len(s) + 1)})
    fig = px.line(df, x="date", y="cumulative",
                  color_discrete_sequence=[SECONDARY])
    fig.update_traces(line=dict(width=2))
    fig.update_yaxes(title="Total Messages")
    return _dark_layout(fig, title)


# ---------------------------------------------------------------------------
# PyVis network  (returns HTML string)
# ---------------------------------------------------------------------------

def pyvis_network(
    G,
    communities: dict,
    highlight_node: str | None = None,
    height: str = "660px",
    label_top_n: int = 15,
) -> str:
    """
    Build a PyVis interactive network and return its HTML string.
    Embeds via st.components.v1.html().

    - Node size     : log-scaled weighted degree
    - Node color    : community (distinct palette)
    - Edge thickness: proportional to message weight
    - Labels        : top-N nodes by weighted degree + highlight node
    - Physics       : ForceAtlas2 (draggable, zoomable, settling)
    """
    try:
        from pyvis.network import Network
    except ImportError:
        return (
            "<div style='color:#FF6B6B;padding:20px;font-family:monospace;'>"
            "PyVis not installed. Run: <code>pip install pyvis</code>"
            "</div>"
        )

    # ── Degree lookups ────────────────────────────────────────────────────
    wdeg = {n: G.degree(n, weight="weight") for n in G.nodes()}
    max_wdeg = max(wdeg.values()) if wdeg else 1

    top_n = set(sorted(wdeg, key=wdeg.get, reverse=True)[:label_top_n])
    if highlight_node:
        top_n.add(highlight_node)

    # ── Create network ────────────────────────────────────────────────────
    net = Network(
        height=height,
        width="100%",
        directed=G.is_directed(),
        bgcolor=PAGE_BG,
        font_color=TEXT,
        cdn_resources="remote",
    )

    # ── Physics: ForceAtlas2 ──────────────────────────────────────────────
    net.force_atlas_2based(
        gravity=-55,
        central_gravity=0.005,
        spring_length=130,
        spring_strength=0.06,
        damping=0.92,
        overlap=0.4,
    )

    # ── Interaction ───────────────────────────────────────────────────────
    try:
        net.set_options(json.dumps({
            "nodes": {
                "font": {"face": "Inter, Arial, sans-serif"},
                "shadow": {"enabled": True, "size": 8, "color": "rgba(0,0,0,0.6)"},
            },
            "edges": {
                "smooth": {"type": "continuous", "roundness": 0.25},
                "arrows": {
                    "to": {
                        "enabled": G.is_directed(),
                        "scaleFactor": 0.4,
                    }
                },
                "color": {
                    "color": "#2A2A45",
                    "highlight": ACCENT,
                    "hover": SECONDARY,
                    "opacity": 0.7,
                },
                "selectionWidth": 2,
                "hoverWidth": 1.5,
            },
            "interaction": {
                "hover": True,
                "hoverConnectedEdges": True,
                "tooltipDelay": 80,
                "hideEdgesOnDrag": True,
                "multiselect": True,
                "navigationButtons": False,
                "keyboard": {"enabled": False},
            },
            "physics": {
                "stabilization": {
                    "enabled": True,
                    "iterations": 150,
                    "updateInterval": 30,
                    "fit": True,
                },
                "maxVelocity": 50,
                "minVelocity": 0.5,
                "timestep": 0.35,
            },
        }))
    except Exception:
        pass  # Fall back to default options if set_options fails

    # ── Add nodes ─────────────────────────────────────────────────────────
    for node in G.nodes():
        wd = wdeg.get(node, 1)
        comm = int(communities.get(node, 0))
        is_hl = (node == highlight_node)

        size = 8 + 32 * math.log1p(wd) / math.log1p(max_wdeg)
        base_color = PYVIS_COMMUNITY_COLORS[comm % len(PYVIS_COMMUNITY_COLORS)]
        if is_hl:
            base_color = "#FFD700"
            size = max(size, 30)

        label = str(node) if node in top_n else ""
        deg = G.degree(node)

        title_html = (
            f"<div style='background:#1A2438;padding:10px 14px;border-radius:6px;"
            f"border:1px solid rgba(255,255,255,0.1);min-width:170px;"
            f"font-family:Inter,Arial,sans-serif;'>"
            f"<div style='color:{ACCENT};font-weight:700;font-size:13px;"
            f"margin-bottom:8px;border-bottom:1px solid rgba(255,255,255,0.07);"
            f"padding-bottom:6px;'>{node}</div>"
            f"<div style='color:{MUTED};font-size:11px;line-height:1.9;'>"
            f"Connections: <span style='color:{TEXT};font-weight:600;'>{deg}</span><br>"
            f"Message weight: <span style='color:{TEXT};font-weight:600;'>{wd:,}</span><br>"
            f"Community: <span style='color:{base_color};font-weight:600;'>{comm}</span>"
            f"</div></div>"
        )

        net.add_node(
            node,
            label=label,
            size=float(size),
            color={
                "background": base_color,
                "border": "rgba(255,255,255,0.15)",
                "highlight": {"background": base_color, "border": "#FFD700"},
                "hover": {"background": base_color, "border": SECONDARY},
            },
            title=title_html,
            borderWidth=1,
            borderWidthSelected=3,
        )

    # ── Add edges ─────────────────────────────────────────────────────────
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1)
        edge_title = (
            f"<div style='background:#1A2438;padding:7px 11px;border-radius:5px;"
            f"font-family:Inter,Arial,sans-serif;font-size:11px;'>"
            f"<span style='color:{MUTED};'>{u}</span>"
            f"<span style='color:{ACCENT};'> → </span>"
            f"<span style='color:{MUTED};'>{v}</span><br>"
            f"<span style='color:{MUTED};'>Messages: </span>"
            f"<span style='color:{ACCENT};font-weight:700;'>{w:,}</span>"
            f"</div>"
        )
        net.add_edge(u, v, value=float(w), title=edge_title)

    # ── Generate HTML and inject theme CSS ────────────────────────────────
    try:
        html = net.generate_html()
    except Exception:
        # Fallback: write to temp file and read back
        tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
        tmp.close()
        net.save_graph(tmp.name)
        with open(tmp.name, "r", encoding="utf-8") as f:
            html = f.read()
        os.unlink(tmp.name)

    # Inject custom CSS into the generated HTML
    theme_css = f"""
<style>
  html, body {{
    background-color: {PAGE_BG} !important;
    margin: 0;
    padding: 0;
  }}
  #mynetwork {{
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 8px !important;
    background-color: {PAGE_BG} !important;
  }}
  div.vis-tooltip {{
    background: #1A2438 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: {TEXT} !important;
    border-radius: 6px !important;
    padding: 0 !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.5) !important;
  }}
  /* Navigation buttons (if shown) */
  .vis-navigation .vis-button {{
    background-color: #1E2A3A;
    border-color: rgba(255,255,255,0.1);
  }}
</style>
"""
    html = html.replace("</head>", theme_css + "</head>", 1)
    return html


# ---------------------------------------------------------------------------
# Plotly network (kept as fallback / non-PyVis use)
# ---------------------------------------------------------------------------

COMMUNITY_PALETTE = (
    px.colors.qualitative.Plotly
    + px.colors.qualitative.Safe
    + px.colors.qualitative.Vivid
)


def plotly_network(
    G,
    pos: dict,
    node_sizes: dict | None = None,
    node_colors: dict | None = None,
    color_label: str = "community",
    title: str = "",
    height: int = 650,
    highlight_node: str | None = None,
    label_top_n: int = 12,
    categorical: bool = False,
) -> go.Figure:
    """Plotly-based static network (kept for compatibility)."""
    if not pos:
        return go.Figure()

    traces = []

    if G.number_of_edges() > 0:
        all_weights = [d.get("weight", 1) for _, _, d in G.edges(data=True)]
        median_w = float(np.median(all_weights)) if all_weights else 1.0
        for is_heavy, color, width in [
            (True,  "#606080", 0.9),
            (False, "#2a2a44", 0.35),
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
                    x=ex, y=ey, mode="lines",
                    line=dict(width=width, color=color),
                    hoverinfo="none", showlegend=False,
                ))

    wdeg = {n: G.degree(n, weight="weight") for n in G.nodes()}
    top_n_set = set(sorted(wdeg, key=wdeg.get, reverse=True)[:label_top_n])
    if highlight_node and highlight_node in G:
        top_n_set.add(highlight_node)

    regular = [n for n in G.nodes() if n in pos and n != highlight_node]
    special = [highlight_node] if (highlight_node and highlight_node in G
                                   and highlight_node in pos) else []

    for group, node_list in [("regular", regular), ("highlight", special)]:
        if not node_list:
            continue
        xs = [pos[n][0] for n in node_list]
        ys = [pos[n][1] for n in node_list]
        sizes = [node_sizes.get(n, 7) if node_sizes else 7 for n in node_list]
        if group == "highlight":
            sizes = [max(s, 22) for s in sizes]

        if categorical and node_colors is not None:
            colors = [
                COMMUNITY_PALETTE[int(node_colors.get(n, 0)) % len(COMMUNITY_PALETTE)]
                for n in node_list
            ]
            marker = dict(
                size=sizes, color=colors,
                line=dict(
                    width=3 if group == "highlight" else 0.8,
                    color="#FFD700" if group == "highlight" else "rgba(255,255,255,0.12)",
                ),
                opacity=0.93,
            )
        else:
            raw_colors = [node_colors.get(n, 0) if node_colors else 0 for n in node_list]
            marker = dict(
                size=sizes, color=raw_colors, colorscale="Turbo",
                cmin=min(raw_colors) if raw_colors else 0,
                cmax=max(raw_colors) if raw_colors else 1,
                line=dict(
                    width=3 if group == "highlight" else 0.8,
                    color="#FFD700" if group == "highlight" else "rgba(255,255,255,0.12)",
                ),
                showscale=(group == "regular"),
                colorbar=dict(
                    title=dict(text=color_label, font=dict(size=11)),
                    thickness=12, len=0.55, x=1.01,
                ) if group == "regular" else None,
                opacity=0.93,
            )

        hover = []
        for n in node_list:
            deg = G.degree(n)
            wd = G.degree(n, weight="weight")
            cv = node_colors.get(n, 0) if node_colors else 0
            extra = (f"Community: {cv}" if categorical else
                     f"{color_label}: {cv:.1f}" if isinstance(cv, float) else
                     f"{color_label}: {cv}")
            hover.append(
                f"<b>{n}</b><br>Connections: {deg}<br>Weight: {wd:,}<br>{extra}"
            )

        text_labels = [n if n in top_n_set else "" for n in node_list]
        traces.append(go.Scatter(
            x=xs, y=ys, mode="markers+text", marker=marker,
            text=text_labels, textposition="top center",
            textfont=dict(size=9, color="rgba(210,210,225,0.85)"),
            hovertext=hover, hovertemplate="%{hovertext}<extra></extra>",
            showlegend=False,
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG, plot_bgcolor=BG, showlegend=False,
        title=dict(text=title, font=dict(size=14, color=TEXT)),
        height=height,
        margin=dict(l=5, r=5, t=40, b=5),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        hovermode="closest", dragmode="pan",
    )
    return fig
