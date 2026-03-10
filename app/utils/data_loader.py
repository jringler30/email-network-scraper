"""
Data loading and validation for the email network dashboard.
Reads cleaned_edges.csv, cleaned_nodes.csv, and network_edge_list.csv.
"""

import pandas as pd
import streamlit as st
from pathlib import Path


# ---------------------------------------------------------------------------
# Schema detection helpers
# ---------------------------------------------------------------------------

# We support several possible column-name conventions so the loader is
# resilient to minor CSV variations (e.g. "sender" vs "from", etc.)

_EDGE_COL_ALIASES = {
    "sender":         ["sender", "from", "source", "from_address", "src"],
    "recipient":      ["recipient", "to", "target", "to_address", "dst"],
    "recipient_type": ["recipient_type", "type", "recip_type"],
    "datetime":       ["datetime", "date", "timestamp", "sent_date", "time"],
    "subject":        ["subject", "subj"],
    "message_id":     ["message_id", "msg_id", "id"],
    "thread_id":      ["thread_id", "thread"],
    "weight":         ["weight", "count", "num_emails", "n"],
}

_NODE_COL_ALIASES = {
    "node_id": ["node_id", "id", "email", "address", "name"],
    "label":   ["label", "name", "display_name", "email"],
    "roles":   ["roles", "role", "type"],
}


def _resolve(df: pd.DataFrame, aliases: dict[str, list[str]]) -> dict[str, str | None]:
    """Map canonical column names to actual column names present in *df*."""
    cols_lower = {c.lower().strip(): c for c in df.columns}
    mapping: dict[str, str | None] = {}
    for canon, candidates in aliases.items():
        mapping[canon] = None
        for cand in candidates:
            if cand.lower() in cols_lower:
                mapping[canon] = cols_lower[cand.lower()]
                break
    return mapping


# ---------------------------------------------------------------------------
# Loaders (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_edges(path: str | Path) -> pd.DataFrame | None:
    """Load an edge-list CSV and normalise column names."""
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        return None
    col_map = _resolve(df, _EDGE_COL_ALIASES)

    # Rename found columns to canonical names
    rename = {v: k for k, v in col_map.items() if v is not None}
    df = df.rename(columns=rename)

    # Parse datetime if present
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # Ensure sender/recipient are strings
    for col in ("sender", "recipient"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df


@st.cache_data(show_spinner=False)
def load_nodes(path: str | Path) -> pd.DataFrame | None:
    """Load a node-metadata CSV and normalise column names."""
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        return None
    col_map = _resolve(df, _NODE_COL_ALIASES)
    rename = {v: k for k, v in col_map.items() if v is not None}
    df = df.rename(columns=rename)
    return df


@st.cache_data(show_spinner=False)
def load_network_edge_list(path: str | Path) -> pd.DataFrame | None:
    """Load the aggregated / weighted edge list if available."""
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        return None
    col_map = _resolve(df, _EDGE_COL_ALIASES)
    rename = {v: k for k, v in col_map.items() if v is not None}
    df = df.rename(columns=rename)
    for col in ("sender", "recipient"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


# ---------------------------------------------------------------------------
# Master loader - returns the best available data sources
# ---------------------------------------------------------------------------

def load_all(data_dir: str | Path = ".") -> dict:
    """
    Return a dict with keys 'edges', 'nodes', 'net_edges', plus metadata.
    Any value may be None if the file is missing.
    """
    data_dir = Path(data_dir)
    edges     = load_edges(data_dir / "cleaned_edges.csv")
    nodes     = load_nodes(data_dir / "cleaned_nodes.csv")
    net_edges = load_network_edge_list(data_dir / "network_edge_list.csv")

    # Determine primary edge source
    primary_edges = edges
    if primary_edges is None:
        primary_edges = net_edges  # fallback

    return {
        "edges": edges,
        "nodes": nodes,
        "net_edges": net_edges,
        "primary_edges": primary_edges,
    }
