# Jmail Network Scraper

A two-phase Selenium crawler that extracts email metadata from [jmail.world](https://jmail.world), a public archive of email communications presented in a Gmail-like interface. The scraper collects sender, recipient, date, and subject fields — no message bodies — and outputs structured CSV files suitable for network analysis, graph construction, or social network visualization.

## Features

- **Two-phase pipeline** separates ID collection (fast, no page navigation) from metadata extraction (direct URL visits), eliminating fragile back-button navigation
- **Resumable by design** — Phase 1 appends to `thread_ids.txt` with deduplication; Phase 2 tracks completed threads in `processed_threads.txt` and skips them on restart
- **Rate-limit handling** with exponential backoff and automatic retry
- **Popup dismissal layer** detects and closes modal overlays before interactions, with JavaScript click fallback
- **Output sanitation** strips embedded newlines, collapses whitespace, and removes trailing mailbox labels from fields before writing CSV
- **Edge-list output schema** produces one row per sender-recipient pair per message, ready for direct import into NetworkX, Gephi, or similar tools

## Installation

Requires Python 3.10+ and Google Chrome.

```bash
git clone https://github.com/jringler30/email-network-scraper.git
cd email-network-scraper
pip install -r requirements.txt
```

ChromeDriver is managed automatically via `webdriver-manager`. No manual driver installation needed.

## Usage

```bash
# Run the Streamlit dashboard:
streamlit run app/app.py

# Run full scraper pipeline (Phase 1 + Phase 2):
python scraper/scraper.py

# Phase 1 only — collect thread IDs from inbox pagination:
python scraper/scraper.py --phase 1

# Phase 2 only — crawl threads listed in thread_ids.txt:
python scraper/scraper.py --phase 2

# Limit Phase 1 to the first 200 inbox rows:
python scraper/scraper.py --phase 1 --max-rows 200

# Run with visible browser for debugging:
python scraper/scraper.py --no-headless

# Custom output directory:
python scraper/scraper.py --outdir ./data
```

To resume an interrupted run, simply re-execute the same command. The scraper detects previously processed threads and continues where it left off.

## Output Files

All outputs are written to `./jmail_output/` by default (configurable via `--outdir`).

| File | Description |
|------|-------------|
| `thread_ids.txt` | One thread identifier per line, collected during Phase 1 |
| `processed_threads.txt` | Thread IDs already crawled in Phase 2 (resume checkpoint) |
| `edges.csv` | Edge list: `message_id, thread_id, sender, recipient, recipient_type, datetime, subject` |
| `nodes.csv` | Deduplicated participant list: `node_id, label, roles` |

`edges.csv` contains one row per sender-to-recipient pair per message. Multi-recipient messages produce multiple rows. The `recipient_type` column distinguishes `to`, `cc`, `bcc`, and `unknown`.

## Data

Scraped datasets are intentionally excluded from this repository. The archive contains approximately 7,500 threads. A full crawl takes roughly 3-4 hours with default delay settings. To reproduce, run the scraper against the live site.

## Project Structure

```
email-network-scraper/
├── app/
│   ├── app.py               # Streamlit dashboard
│   └── utils/
│       ├── charts.py
│       ├── data_loader.py
│       ├── graph_builder.py
│       └── network_views.py
├── data/
│   ├── cleaned_edges.csv
│   ├── cleaned_nodes.csv
│   └── network_edge_list.csv
├── notebook/
│   └── jmail_network.ipynb
├── scraper/
│   └── scraper.py           # Two-phase Selenium crawler
├── requirements.txt
└── README.md
```

## License

MIT
