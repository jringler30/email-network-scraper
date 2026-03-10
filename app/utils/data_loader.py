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

# Sentinel strings that represent missing data after CSV parsing
_BAD_VALUES = {"nan", "NaN", "NaT", "None", "none", "null", "NULL", ""}


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


def _clean_edge_df(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing or sentinel sender/recipient values."""
    for col in ("sender", "recipient"):
        if col in df.columns:
            df = df[~df[col].isin(_BAD_VALUES)]
            df = df[df[col].notna()]
            df = df[df[col].str.strip().str.len() > 0]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Loaders (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_edges(path: str | Path) -> pd.DataFrame | None:
    """Load an edge-list CSV and normalise column names."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return None
    if df.empty:
        return None

    col_map = _resolve(df, _EDGE_COL_ALIASES)
    rename = {v: k for k, v in col_map.items() if v is not None}
    df = df.rename(columns=rename)

    # Parse datetime defensively
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=False)

    # Ensure sender/recipient are clean strings
    for col in ("sender", "recipient"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df = _clean_edge_df(df)
    return df if not df.empty else None


@st.cache_data(show_spinner=False)
def load_nodes(path: str | Path) -> pd.DataFrame | None:
    """Load a node-metadata CSV and normalise column names."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return None
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
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return None
    if df.empty:
        return None
    col_map = _resolve(df, _EDGE_COL_ALIASES)
    rename = {v: k for k, v in col_map.items() if v is not None}
    df = df.rename(columns=rename)
    for col in ("sender", "recipient"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    df = _clean_edge_df(df)
    return df if not df.empty else None


# ---------------------------------------------------------------------------
# Master loader
# ---------------------------------------------------------------------------

def load_all(data_dir: str | Path = ".") -> dict:
    """
    Return a dict with keys 'edges', 'nodes', 'net_edges', plus metadata.
    Any value may be None if the file is missing or unreadable.
    """
    data_dir = Path(data_dir)
    edges     = load_edges(data_dir / "cleaned_edges.csv")
    nodes     = load_nodes(data_dir / "cleaned_nodes.csv")
    net_edges = load_network_edge_list(data_dir / "network_edge_list.csv")

    primary_edges = edges if edges is not None else net_edges

    return {
        "edges": edges,
        "nodes": nodes,
        "net_edges": net_edges,
        "primary_edges": primary_edges,
    }
