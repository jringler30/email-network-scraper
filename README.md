# Jmail Network Explorer

An end-to-end email network analysis project — from data collection to interactive visualization. This repository includes a Selenium-based email scraper, cleaned network datasets, an exploratory analysis notebook, and a fully deployed Streamlit dashboard for interactively exploring communication patterns between participants.

---

## Live Dashboard

**[→ Open the Interactive Dashboard](https://jmail-network-msba.streamlit.app/)**

Explore the email communication network directly in your browser — no installation required.

---

## What This Project Does

This project treats an email archive as a communication network and applies network science techniques to uncover structure, identify key participants, and visualize relationships.

- **Nodes** represent individual participants in the email network
- **Edges** represent email communication between two participants
- **Edge weight** reflects message volume — heavier edges mean more emails exchanged
- **Network metrics** such as betweenness centrality, eigenvector centrality, and Louvain community detection are used to identify important actors and group structure

---

## Dashboard Features

### Overview
High-level KPI cards covering participants, connections, total message volume, average connections per node, number of communities, largest connected component size, total component count, and graph density. Includes top-10 sender and recipient tables and a community size distribution chart.

### Interactive Network Graph
A physics-based force-directed graph of the full communication network rendered with PyVis (Barnes-Hut simulation). Nodes are sized by message volume and colored by community. Inline controls let you filter by minimum edge weight, cap the number of nodes shown, restrict to the giant component, label the top-N highest-weight nodes, and highlight any specific node. A download button exports the filtered edge list as CSV.

### Top Nodes
Rank all participants by any of eight centrality metrics: total weight, messages sent, messages received, degree, in-degree, out-degree, betweenness centrality, and eigenvector centrality. Results display as a horizontal bar chart plus a sortable, downloadable metrics table.

### Community Detection
Louvain algorithm applied to the undirected projection of the network. Displays community count, size of the largest community, and median community size. Select any community to see per-member metrics and explore it as a standalone interactive subgraph (capped at 150 nodes for performance).

### Ego Network Explorer
Select any participant to visualize their immediate communication neighborhood at radius 1 (direct contacts) or radius 2 (contacts-of-contacts). Shows KPI cards for connections, messages sent and received, betweenness centrality, and community assignment — plus an activity date range if timestamp data is available. A side-by-side strongest-ties table and focused PyVis graph are rendered for the selected node.

### Relationship Strength
Two view modes:
- **Top flows (Sankey)** — visualize the strongest sender → recipient message flows across the entire network, with adjustable top-N and CSV export
- **Node spotlight** — select any participant and see a ranked bar chart of their top 20 contacts by message volume

A global "Strongest Connections Overall" table (top 25) is shown at the bottom with CSV export.

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
jmail-network-explorer/
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
├── notebook/
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
| [Plotly](https://plotly.com/python/) | Charts, Sankey diagrams, and statistical plots |
| [Pandas](https://pandas.pydata.org) | Data loading and transformation |
| [python-louvain](https://python-louvain.readthedocs.io) | Louvain community detection |
| [Selenium](https://selenium-python.readthedocs.io) | Automated browser scraping |

---

## License

MIT
