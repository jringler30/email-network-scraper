# Jmail Network Explorer

An end-to-end email network analysis project — from data collection to interactive visualization. This repository includes a Selenium-based email scraper, cleaned network datasets, an exploratory analysis notebook, and a fully deployed Streamlit dashboard for interactively exploring communication patterns between participants.

---

## Live Dashboard

**[→ Open the Interactive Dashboard](https://jmail-network-msba.streamlit.app/)**

Explore the email communication network directly in your browser — no installation required. The dashboard supports interactive graph exploration, community detection, ego network analysis, and relationship strength visualization.

---

## What This Project Does

This project treats an email archive as a communication network and applies network science techniques to uncover structure, identify key participants, and visualize relationships.

- **Nodes** represent individual participants in the email network
- **Edges** represent email communication between two participants
- **Edge weight** reflects message volume — heavier edges mean more emails exchanged
- **Network metrics** such as betweenness centrality, eigenvector centrality, and Louvain community detection are used to identify important actors and group structure

---

## Dashboard Features

### Interactive Network Graph
A physics-based force-directed graph of the full communication network. Nodes are sized by message volume, colored by community, and spread using Barnes-Hut simulation. Drag, zoom, and click to explore. Filters for minimum edge weight, node count, and component selection are available in the sidebar.

### Node Importance Ranking
Rank all participants by multiple centrality metrics — total message volume, messages sent, messages received, betweenness centrality, and eigenvector centrality. View results as a bar chart or sortable table.

### Community Detection
Louvain algorithm applied to the undirected projection of the network. View the size distribution of detected communities, browse members of each community, and explore any community as a standalone subgraph.

### Ego Network Explorer
Select any participant and visualize their immediate communication neighborhood. Supports radius 1 (direct contacts) and radius 2 (contacts-of-contacts). Shows the participant's strongest ties alongside a focused interactive graph.

### Relationship Strength Analysis
Explore message volume between specific participants or view a heatmap of the top-N most active nodes. Identify the strongest bilateral connections across the entire network.

### Network Overview
High-level statistics including total participants, connections, message volume, graph density, component count, and community count — displayed as a KPI dashboard.

---

## Running Locally

**Requirements:** Python 3.10+

```bash
git clone https://github.com/jringler30/email-network-scraper.git
cd email-network-scraper
pip install -r requirements.txt
streamlit run app/app.py
```

The dashboard will launch at `http://localhost:8501`. It reads from the CSV files in `data/` automatically.

---

## Scraper

The scraper is a two-phase Selenium crawler that collects email metadata (sender, recipient, date, subject) from the [jmail.world](https://jmail.world) public archive.

```bash
# Run the full pipeline (Phase 1 + Phase 2):
python scraper/scraper.py

# Phase 1 only — collect thread IDs:
python scraper/scraper.py --phase 1

# Phase 2 only — extract metadata from collected threads:
python scraper/scraper.py --phase 2

# Run with a visible browser for debugging:
python scraper/scraper.py --no-headless
```

The scraper is resumable — re-running continues from where it left off. ChromeDriver is managed automatically via `webdriver-manager`.

---

## Repository Structure

```
email-network-scraper/
├── app/
│   ├── app.py               # Streamlit dashboard (main entry point)
│   └── utils/
│       ├── charts.py        # PyVis and Plotly visualization functions
│       ├── data_loader.py   # CSV loading and schema normalization
│       ├── graph_builder.py # NetworkX graph construction and metrics
│       └── network_views.py # Graph filtering and layout utilities
├── data/
│   ├── cleaned_edges.csv         # Edge list with sender, recipient, weight
│   ├── cleaned_nodes.csv         # Node metadata
│   └── network_edge_list.csv     # Aggregated weighted edge list
├── notebooks/
│   └── jmail_network.ipynb  # Exploratory analysis notebook
├── scraper/
│   └── scraper.py           # Two-phase Selenium crawler
├── requirements.txt
└── README.md
```

---

## Technologies

| Library | Purpose |
|---|---|
| [Streamlit](https://streamlit.io) | Dashboard framework |
| [NetworkX](https://networkx.org) | Graph construction, metrics, community detection |
| [PyVis](https://pyvis.readthedocs.io) | Interactive physics-based network visualization |
| [Plotly](https://plotly.com/python/) | Charts, heatmaps, and statistical plots |
| [Pandas](https://pandas.pydata.org) | Data loading and transformation |
| [python-louvain](https://python-louvain.readthedocs.io) | Louvain community detection |
| [Selenium](https://selenium-python.readthedocs.io) | Automated browser scraping |

---

## License

MIT
