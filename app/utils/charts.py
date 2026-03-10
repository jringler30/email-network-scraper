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


# ---------------------------------------------------------------------------
# PyVis network  (returns HTML string)
# ---------------------------------------------------------------------------

def pyvis_network(
    G,
    communities: dict,
    highlight_node: str | None = None,
    height: str = "660px",
    label_top_n: int = 6,
    stable_mode: bool = False,
) -> str:
    """
    Build a PyVis interactive network and return its HTML string.
    Embeds via st.components.v1.html().

    - Node size     : log-scaled weighted degree
    - Node color    : top-8 communities get distinct colors; minor ones muted
    - Edge thickness: proportional to message weight
    - Labels        : top-N nodes only; zoom-aware via scaling.label
    - Physics       : Barnes-Hut; both modes freeze physics after stabilization
    - stable_mode   : True for ego network — much gentler physics, higher
                      damping, more stabilization iterations; settles fast
    - Click behavior: neighborhood_highlight dims non-neighbors on click
    """
    try:
        from pyvis.network import Network
    except ImportError:
        return (
            "<div style='color:#FF6B6B;padding:20px;font-family:monospace;'>"
            "PyVis not installed. Run: <code>pip install pyvis</code>"
            "</div>"
        )

    from collections import Counter as _Counter

    # ── Degree lookups ────────────────────────────────────────────────────
    wdeg = {n: G.degree(n, weight="weight") for n in G.nodes()}
    max_wdeg = max(wdeg.values()) if wdeg else 1

    top_n = set(sorted(wdeg, key=wdeg.get, reverse=True)[:label_top_n])
    if highlight_node:
        top_n.add(highlight_node)

    # ── Community color strategy ──────────────────────────────────────────
    # Top 8 communities by member count get distinct vivid colors.
    # All smaller communities collapse to a single muted tone.
    # This creates clear visual clusters without color noise.
    _MUTED_COMM = "#374357"   # minor-community node color
    _N_COLORED  = 8
    comm_size   = _Counter(communities.values())
    top_comms   = {
        c: rank
        for rank, c in enumerate(
            sorted(comm_size.keys(), key=lambda c: comm_size[c], reverse=True)[:_N_COLORED]
        )
    }

    def _node_color(comm_id: int) -> str:
        if comm_id in top_comms:
            return PYVIS_COMMUNITY_COLORS[top_comms[comm_id]]
        return _MUTED_COMM

    # ── Create network  (neighborhood_highlight = click dims non-neighbors) ─
    net = Network(
        height=height,
        width="100%",
        directed=G.is_directed(),
        bgcolor=PAGE_BG,
        font_color=TEXT,
        cdn_resources="remote",
        neighborhood_highlight=True,
    )

    # ── Physics presets ────────────────────────────────────────────────────
    # Both modes use Barnes-Hut and freeze physics via JS after stabilization
    # so the graph holds its position once settled. Users can still drag and
    # zoom; just no autonomous movement after the initial layout pass.
    #
    # stable_mode=True  (ego network): very high damping, low velocity cap,
    #   many iterations — settles in ~1s, looks calm from the start.
    # stable_mode=False (main graph): moderate settings tuned so adding more
    #   nodes doesn't cause violent jumping; less repulsion, more friction.
    if stable_mode:
        _physics_opts = {
            "solver": "barnesHut",
            "barnesHut": {
                "gravitationalConstant": -8000,   # gentle repulsion
                "centralGravity": 0.3,            # pulls toward centre
                "springLength": 120,
                "springConstant": 0.08,
                "damping": 0.8,                   # heavy friction → stops fast
                "avoidOverlap": 1.0,
            },
            "stabilization": {
                "enabled": True,
                "iterations": 1000,               # fully settle before showing
                "updateInterval": 50,
                "fit": True,
            },
            "maxVelocity": 15,
            "minVelocity": 3.0,                   # stop threshold — exits early
            "timestep": 0.3,
        }
    else:
        _physics_opts = {
            "solver": "barnesHut",
            "barnesHut": {
                "gravitationalConstant": -15000,  # was -60000 → much calmer
                "centralGravity": 0.15,           # moderate center pull
                "springLength": 150,
                "springConstant": 0.05,
                "damping": 0.5,                   # was 0.1 → 5× more friction
                "avoidOverlap": 1.0,
            },
            "stabilization": {
                "enabled": True,
                "iterations": 500,                # was 300
                "updateInterval": 100,            # less frequent = less flicker
                "fit": True,
            },
            "maxVelocity": 30,                    # was 100 → nodes can't fly
            "minVelocity": 2.0,
            "timestep": 0.35,                     # was 0.5 → more stable sim
        }

    try:
        net.set_options(json.dumps({
            "nodes": {
                "font": {
                    "face": "Inter, Arial, sans-serif",
                    "size": 13,
                    "strokeWidth": 3,
                    "strokeColor": "#0B0F17",
                },
                # scaling.label makes vis.js scale font size with zoom level.
                # drawThreshold: labels are hidden when their canvas pixel size
                # drops below this value (i.e. zoomed out) — reduces clutter.
                # They reappear as you zoom in. Top-N nodes still show first
                # because only they have label text set; minor nodes stay empty.
                "scaling": {
                    "label": {
                        "enabled": True,
                        "min": 8,
                        "max": 20,
                        "maxVisible": 20,
                        "drawThreshold": 9,
                    }
                },
                "shadow": {"enabled": True, "size": 8, "color": "rgba(0,0,0,0.6)"},
            },
            "edges": {
                "smooth": {"type": "dynamic", "roundness": 0.3},
                "arrows": {
                    "to": {"enabled": G.is_directed(), "scaleFactor": 0.3}
                },
                "color": {
                    "color": "#3A405A",
                    "highlight": ACCENT,
                    "hover": SECONDARY,
                    "opacity": 0.7,
                },
                "selectionWidth": 2.5,
                "hoverWidth": 2,
            },
            "interaction": {
                "hover": True,
                "hoverConnectedEdges": True,
                "tooltipDelay": 80,
                "hideEdgesOnDrag": True,
                "multiselect": False,
                "navigationButtons": False,
                "keyboard": {"enabled": False},
            },
            "physics": _physics_opts,
        }))
    except Exception:
        pass

    # ── Add nodes ─────────────────────────────────────────────────────────
    for node in G.nodes():
        wd  = wdeg.get(node, 1)
        comm = int(communities.get(node, 0))
        is_hl = (node == highlight_node)

        # Log-scaled size: range ~4–16px — small enough that avoidOverlap works
        # With Barnes-Hut avoidOverlap:1, node pixel radius is factored into
        # repulsion, so keeping sizes small lets the engine space them properly.
        ratio = math.log1p(wd) / math.log1p(max_wdeg)
        size  = 4 + 12 * (ratio ** 0.6)

        base_color = _node_color(comm)
        border_color = "rgba(255,255,255,0.18)"

        if is_hl:
            base_color   = "#FFD700"
            border_color = "#FFD700"
            size = max(size, 18)

        label = str(node) if node in top_n else ""
        deg   = G.degree(node)
        comm_label = (
            f"Community {comm} "
            f"({'top cluster' if comm in top_comms else 'minor cluster'})"
        )

        title_html = (
            f"<div style='background:#1A2438;padding:10px 14px;border-radius:6px;"
            f"border:1px solid rgba(255,255,255,0.1);min-width:180px;"
            f"font-family:Inter,Arial,sans-serif;'>"
            f"<div style='color:{ACCENT};font-weight:700;font-size:13px;"
            f"margin-bottom:8px;border-bottom:1px solid rgba(255,255,255,0.07);"
            f"padding-bottom:6px;'>{node}</div>"
            f"<div style='color:{MUTED};font-size:11px;line-height:2;'>"
            f"Connections: <span style='color:{TEXT};font-weight:600;'>{deg}</span><br>"
            f"Message weight: <span style='color:{TEXT};font-weight:600;'>{wd:,}</span><br>"
            f"<span style='color:{base_color};'>{comm_label}</span>"
            f"</div>"
            f"<div style='color:#8899AA;font-size:10px;margin-top:6px;'>"
            f"Click node to focus neighbors</div>"
            f"</div>"
        )

        net.add_node(
            node,
            label=label,
            size=float(size),
            color={
                "background": base_color,
                "border": border_color,
                "highlight": {"background": base_color, "border": "#FFD700"},
                "hover":     {"background": base_color, "border": SECONDARY},
            },
            title=title_html,
            borderWidth=1,
            borderWidthSelected=3,
        )

    # ── Add edges ─────────────────────────────────────────────────────────
    # Clamp value to 1–8 range so very high-weight edges don't dominate visually
    all_edge_weights = [d.get("weight", 1) for _, _, d in G.edges(data=True)]
    max_ew = max(all_edge_weights) if all_edge_weights else 1

    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1)
        # Scale to 1–4 for edge thickness (thinner edges = less visual clutter)
        scaled_value = 1 + 3 * math.log1p(w) / math.log1p(max_ew)
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
        net.add_edge(u, v, value=float(scaled_value), title=edge_title)

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

    # Inject JS to freeze physics once the initial stabilization pass completes.
    # The graph settles into its layout, then stops moving — hover/drag/zoom
    # still work normally. Without this, ongoing Barnes-Hut forces cause
    # constant low-level jitter, especially with higher node counts.
    freeze_js = """
<script>
  setTimeout(function() {
    if (typeof network !== 'undefined') {
      network.once('stabilizationIterationsDone', function() {
        network.setOptions({ physics: { enabled: false } });
      });
    }
  }, 0);
</script>
"""
    html = html.replace("</body>", freeze_js + "</body>", 1)
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
