#!/usr/bin/env python3
"""
scrape_metadata.py — 2-phase Selenium crawler for Jmail (public Gmail-like archive).
Extracts email METADATA ONLY (sender, recipients, dates) for network/graph analysis.

Phase 1 — Inbox scan:
  Paginates through inbox, collects thread identifiers from div.email-row
  attributes (data-first-message-id / data-doc-id).  No row clicking.
  Saves deduplicated IDs to thread_ids.txt.

Phase 2 — Thread crawl:
  Visits each thread directly at /thread/{id}?view=inbox.
  Extracts sender, recipients, date, subject from div.message-item elements.
  Writes edges.csv + nodes.csv.  Tracks progress in processed_threads.txt.

Outputs:
  thread_ids.txt        — one thread identifier per line (Phase 1)
  processed_threads.txt — IDs already crawled (Phase 2 resume)
  edges.csv             — one row per (sender → recipient) per message
  nodes.csv             — unique participants

Requirements:
  pip install selenium webdriver-manager

Usage:
  # Run both phases (full crawl):
  python scrape_metadata.py

  # Phase 1 only (collect IDs):
  python scrape_metadata.py --phase 1

  # Phase 2 only (crawl threads from existing thread_ids.txt):
  python scrape_metadata.py --phase 2

  # Limit inbox rows scanned in Phase 1:
  python scrape_metadata.py --max-rows 200

  # Run visible browser for debugging:
  python scrape_metadata.py --no-headless
"""

import argparse
import csv
import logging
import os
import re
import time
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    StaleElementReferenceException,
    ElementClickInterceptedException,
    WebDriverException,
)

try:
    from webdriver_manager.chrome import ChromeDriverManager
    USE_WDM = True
except ImportError:
    USE_WDM = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_URL = "https://jmail.world"
INBOX_URL = f"{BASE_URL}/?view=inbox"
THREAD_URL_TEMPLATE = f"{BASE_URL}/thread/{{tid}}?view=inbox"
PAGE_LOAD_TIMEOUT = 30
ELEMENT_WAIT = 15
THREAD_LOAD_WAIT = 12
MIN_DELAY = 1.0
MAX_DELAY = 3.0
MAX_BACKOFF = 120
BACKOFF_BASE = 5

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("jmail_scraper")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class EdgeRecord:
    message_id: str
    thread_id: str
    sender: str
    recipient: str
    recipient_type: str  # "to", "cc", "bcc", or "unknown"
    datetime: str
    subject: str = ""

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def rand_sleep(lo=MIN_DELAY, hi=MAX_DELAY):
    time.sleep(random.uniform(lo, hi))


def parse_recipients(raw_text: str):
    """
    Parse recipient text into list of (address_or_name, type) tuples.
    Handles patterns like:
      'to Nicholas Ribis <nick@example.com>'
      'to foo@bar.com, baz@qux.com'
      'cc Someone Else <s@e.com>'
    Falls back to raw text with type 'unknown'.
    """
    results = []
    if not raw_text or not raw_text.strip():
        return results

    text = raw_text.strip()

    # Try to detect type prefix
    rtype = "to"
    lower = text.lower()
    if lower.startswith("cc"):
        rtype = "cc"
        text = text[2:].strip().lstrip(":").strip()
    elif lower.startswith("bcc"):
        rtype = "bcc"
        text = text[3:].strip().lstrip(":").strip()
    elif lower.startswith("to"):
        rtype = "to"
        text = text[2:].strip().lstrip(":").strip()

    # Extract email addresses
    emails = re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    if emails:
        for e in emails:
            results.append((e.strip(), rtype))
    else:
        # No emails found; split by comma and use names
        parts = [p.strip() for p in text.split(",") if p.strip()]
        for p in parts:
            clean = re.sub(r"[<>]", "", p).strip()
            if clean:
                results.append((clean, rtype))

    return results


def extract_sender_email_or_name(el) -> str:
    """Try to get email from sender-avatar alt or sender-name text."""
    try:
        imgs = el.find_elements(By.CSS_SELECTOR, "img.sender-avatar")
        if imgs:
            alt = imgs[0].get_attribute("alt") or ""
            if alt.strip():
                return alt.strip()
    except Exception:
        pass
    try:
        name_el = el.find_element(By.CSS_SELECTOR, "div.sender-name")
        raw = name_el.text.strip()
        if raw:
            return raw
    except Exception:
        pass
    try:
        info = el.find_element(By.CSS_SELECTOR, "div.sender-info")
        return info.text.strip().split("\n")[0]
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Output sanitation
# ---------------------------------------------------------------------------
_TRAILING_LABEL_RE = re.compile(r"\s+(?:Yahoo|Inbox)\s*$", re.IGNORECASE)


def sanitize_text(x) -> str:
    """Collapse newlines/whitespace into single spaces."""
    if x is None:
        return ""
    s = str(x)
    s = s.replace("\r", " ").replace("\n", " ")
    s = " ".join(s.split())
    return s.strip()


def sanitize_subject(x) -> str:
    """sanitize_text + remove trailing mailbox labels like ' Yahoo' / ' Inbox'."""
    s = sanitize_text(x)
    return _TRAILING_LABEL_RE.sub("", s).strip()


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------
class EdgeWriter:
    FIELDS = ["message_id", "thread_id", "sender", "recipient",
              "recipient_type", "datetime", "subject"]

    def __init__(self, path: str, append: bool = False):
        self.path = path
        mode = "a" if append else "w"
        self._file = open(path, mode, newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        if not append or os.path.getsize(path) == 0:
            self._writer.writeheader()
        self._count = 0

    def write(self, rec: EdgeRecord):
        raw = asdict(rec)
        raw["subject"] = sanitize_subject(raw["subject"])
        for k in self.FIELDS:
            if k != "subject":
                raw[k] = sanitize_text(raw[k])
        self._writer.writerow(raw)
        self._count += 1
        if self._count % 20 == 0:
            self._file.flush()

    def close(self):
        self._file.flush()
        self._file.close()


def build_nodes_csv(edges_path: str, nodes_path: str):
    """Read edges.csv and emit unique participants to nodes.csv."""
    participants = {}
    with open(edges_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = row["sender"]
            r = row["recipient"]
            if s:
                participants.setdefault(s, set()).add("sender")
            if r:
                participants.setdefault(r, set()).add("recipient")

    with open(nodes_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["node_id", "label", "roles"])
        writer.writeheader()
        for i, (name, roles) in enumerate(sorted(participants.items())):
            writer.writerow({
                "node_id": i,
                "label": name,
                "roles": ";".join(sorted(roles)),
            })
    log.info("Wrote %d unique participants to %s", len(participants), nodes_path)


# ---------------------------------------------------------------------------
# File-based resume helpers
# ---------------------------------------------------------------------------
def load_lines(path: str) -> list[str]:
    """Read a text file into a list of non-empty stripped lines."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def append_line(path: str, line: str):
    """Append a single line to a text file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------
class JmailScraper:
    def __init__(self, outdir: str, max_rows: int = 0, headless: bool = True):
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.max_rows = max_rows
        self.headless = headless

        # File paths
        self.thread_ids_path = str(self.outdir / "thread_ids.txt")
        self.processed_path = str(self.outdir / "processed_threads.txt")
        self.edges_path = str(self.outdir / "edges.csv")
        self.nodes_path = str(self.outdir / "nodes.csv")

        self.driver: Optional[webdriver.Chrome] = None
        self.backoff_count = 0

    # -- browser setup --
    def _init_driver(self):
        opts = Options()
        if self.headless:
            opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--window-size=1920,1080")
        opts.add_argument(
            "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        )
        if USE_WDM:
            service = Service(ChromeDriverManager().install())
        else:
            service = Service()
        self.driver = webdriver.Chrome(service=service, options=opts)
        self.driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
        self.driver.implicitly_wait(2)
        log.info("Chrome driver initialized (headless=%s)", self.headless)

    def _quit_driver(self):
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None

    # -- low-level Selenium helpers --
    def _wait(self, css: str, timeout: int = ELEMENT_WAIT):
        return WebDriverWait(self.driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, css))
        )

    def _wait_all(self, css: str, timeout: int = ELEMENT_WAIT):
        WebDriverWait(self.driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, css))
        )
        return self.driver.find_elements(By.CSS_SELECTOR, css)

    def _close_popups(self):
        """Detect and dismiss any modal/dialog/popup overlaying the page."""
        try:
            candidates = self.driver.find_elements(
                By.CSS_SELECTOR,
                '[aria-label*="close" i], [aria-label*="Close"], '
                '[aria-label*="dismiss" i], [aria-label*="Dismiss"], '
                'button[class*="close"], button[class*="Close"], '
                'button[class*="dismiss"], '
                '[class*="modal"] button, [class*="dialog"] button, '
                '[role="dialog"] button, [class*="overlay"] button'
            )
            for btn in candidates:
                try:
                    if not btn.is_displayed():
                        continue
                    label = (btn.get_attribute("aria-label") or "").lower()
                    text = btn.text.strip().lower()
                    classes = (btn.get_attribute("class") or "").lower()
                    is_close = (
                        "close" in label or "dismiss" in label or
                        "close" in text or text in ("×", "✕", "x", "✖") or
                        "close" in classes or "dismiss" in classes
                    )
                    if is_close:
                        btn.click()
                        time.sleep(0.3)
                        return
                except (StaleElementReferenceException, WebDriverException):
                    continue

            x_icons = self.driver.find_elements(
                By.XPATH,
                "//*[self::button or self::span or self::a or self::div]"
                "[normalize-space(.)='×' or normalize-space(.)='✕' "
                "or normalize-space(.)='✖' or normalize-space(.)='X']"
            )
            for icon in x_icons:
                try:
                    if icon.is_displayed():
                        icon.click()
                        time.sleep(0.3)
                        return
                except (StaleElementReferenceException, WebDriverException):
                    continue

            try:
                active = self.driver.switch_to.active_element
                active.send_keys(Keys.ESCAPE)
                time.sleep(0.3)
            except WebDriverException:
                pass
        except Exception:
            pass

    def _safe_click(self, element):
        """Click with popup-dismiss and JS fallback. Used in Phase 1 pagination."""
        self._close_popups()
        try:
            self.driver.execute_script(
                "arguments[0].scrollIntoView({block:'center'});", element
            )
        except WebDriverException:
            pass
        time.sleep(0.3)

        try:
            element.click()
            return True
        except (ElementClickInterceptedException, WebDriverException):
            pass
        except StaleElementReferenceException:
            return False

        self._close_popups()
        time.sleep(0.3)

        try:
            element.click()
            return True
        except (ElementClickInterceptedException, WebDriverException):
            pass
        except StaleElementReferenceException:
            return False

        try:
            self.driver.execute_script("arguments[0].click();", element)
            return True
        except WebDriverException:
            return False

    def _detect_rate_limit(self) -> bool:
        try:
            body = self.driver.find_element(By.TAG_NAME, "body").text.lower()
            if "429" in body or "rate limit" in body or "too many requests" in body:
                return True
        except Exception:
            pass
        return False

    def _handle_rate_limit(self):
        self.backoff_count += 1
        wait_time = min(BACKOFF_BASE * (2 ** self.backoff_count), MAX_BACKOFF)
        log.warning("Rate limit detected — backing off %ds (attempt %d)",
                     wait_time, self.backoff_count)
        time.sleep(wait_time)

    # ======================================================================
    #  PHASE 1 — Inbox scan: collect thread IDs by paginating, no clicking
    # ======================================================================
    def run_phase1(self):
        log.info("=" * 60)
        log.info("PHASE 1 — Inbox scan: collecting thread IDs")
        log.info("=" * 60)

        # Load any IDs already collected (for resume / dedup)
        seen = set(load_lines(self.thread_ids_path))
        log.info("Already collected: %d thread IDs", len(seen))

        self._init_driver()
        try:
            self._phase1_paginate(seen)
        except KeyboardInterrupt:
            log.info("Phase 1 interrupted by user")
        except Exception as e:
            log.error("Phase 1 fatal error: %s", e, exc_info=True)
        finally:
            self._quit_driver()

        total = len(load_lines(self.thread_ids_path))
        log.info("Phase 1 complete. Total unique thread IDs: %d", total)

    def _phase1_paginate(self, seen: set):
        log.info("Loading inbox: %s", INBOX_URL)
        self.driver.get(INBOX_URL)
        self._wait("div.email-row", timeout=PAGE_LOAD_TIMEOUT)
        rand_sleep()

        page = 1
        total_collected = len(seen)
        new_this_run = 0

        while True:
            log.info("--- Phase 1 · Page %d ---", page)

            if self._detect_rate_limit():
                self._handle_rate_limit()
                self.driver.get(INBOX_URL)
                self._wait("div.email-row", timeout=PAGE_LOAD_TIMEOUT)
                continue

            # Gather all email-row elements on this page
            try:
                rows = self._wait_all("div.email-row")
            except TimeoutException:
                log.warning("No email rows on page %d — stopping", page)
                break

            page_new = 0
            for row in rows:
                # Max rows check (counts total unique IDs collected)
                if self.max_rows and total_collected >= self.max_rows:
                    log.info("Reached max-rows limit (%d). Stopping Phase 1.",
                             self.max_rows)
                    return

                try:
                    # Extract identifier from row attributes — try the row
                    # itself and also any inner child with the data attribute
                    # (the site renders both a desktop and mobile div per row)
                    tid = (row.get_attribute("data-first-message-id")
                           or row.get_attribute("data-doc-id")
                           or "")

                    if not tid:
                        inner = row.find_elements(
                            By.CSS_SELECTOR,
                            "[data-first-message-id], [data-doc-id]"
                        )
                        for el in inner:
                            tid = (el.get_attribute("data-first-message-id")
                                   or el.get_attribute("data-doc-id") or "")
                            if tid:
                                break

                    if not tid:
                        continue

                    tid = tid.strip()
                    if tid in seen:
                        continue

                    # New ID — persist immediately
                    seen.add(tid)
                    append_line(self.thread_ids_path, tid)
                    total_collected += 1
                    new_this_run += 1
                    page_new += 1

                except StaleElementReferenceException:
                    continue

            log.info("Page %d: %d new IDs (total: %d)", page, page_new, total_collected)
            self.backoff_count = 0

            # Try next page
            try:
                btn = self.driver.find_element(
                    By.CSS_SELECTOR, "button.elastic-next-page-link"
                )
                if not btn.is_displayed():
                    log.info("Next-page button hidden — end of inbox.")
                    break
                self._safe_click(btn)
                rand_sleep(1.5, 3.0)
                self._wait("div.email-row", timeout=PAGE_LOAD_TIMEOUT)
                page += 1
            except NoSuchElementException:
                log.info("No next-page button — end of inbox.")
                break
            except TimeoutException:
                log.warning("Timeout waiting for next page — stopping.")
                break

        log.info("Phase 1 collected %d new IDs this run", new_this_run)

    # ======================================================================
    #  PHASE 2 — Thread crawl: visit each thread URL, extract metadata
    # ======================================================================
    def run_phase2(self):
        log.info("=" * 60)
        log.info("PHASE 2 — Thread crawl: extracting metadata")
        log.info("=" * 60)

        # Load thread IDs and already-processed set
        all_ids = load_lines(self.thread_ids_path)
        if not all_ids:
            log.error("No thread IDs found in %s — run Phase 1 first.",
                      self.thread_ids_path)
            return

        processed = set(load_lines(self.processed_path))
        pending = [tid for tid in all_ids if tid not in processed]
        log.info("Threads: %d total, %d already processed, %d pending",
                 len(all_ids), len(processed), len(pending))

        if not pending:
            log.info("Nothing to do — all threads already processed.")
            self._build_nodes_if_needed()
            return

        # Open CSV writer (append if some threads already processed)
        append = len(processed) > 0 and os.path.exists(self.edges_path)
        writer = EdgeWriter(self.edges_path, append=append)

        self._init_driver()
        crawled = 0

        try:
            for i, tid in enumerate(pending):
                log.info("Processing thread %d/%d: %s", i + 1, len(pending), tid)

                try:
                    edges = self._crawl_single_thread(tid)

                    for edge in edges:
                        writer.write(edge)

                    log.info("  → %d edge(s) from thread %s", len(edges), tid)

                    # Mark as processed immediately (resume-safe)
                    append_line(self.processed_path, tid)
                    crawled += 1
                    self.backoff_count = 0

                except Exception as e:
                    log.warning("Error on thread %s: %s — skipping", tid, e)

                rand_sleep()

        except KeyboardInterrupt:
            log.info("Phase 2 interrupted by user after %d threads", crawled)
        except Exception as e:
            log.error("Phase 2 fatal error: %s", e, exc_info=True)
        finally:
            writer.close()
            self._quit_driver()

        log.info("Phase 2 crawled %d threads this run", crawled)
        self._build_nodes_if_needed()

    def _crawl_single_thread(self, tid: str) -> list[EdgeRecord]:
        """Navigate directly to a thread URL and extract all message metadata."""
        url = THREAD_URL_TEMPLATE.format(tid=tid)
        self.driver.get(url)

        # Rate-limit check after page load
        if self._detect_rate_limit():
            self._handle_rate_limit()
            self.driver.get(url)

        # Wait for message content to render — try multiple selectors
        loaded = False
        for selector in ("div.message-item", "div.message-header",
                         "div.message-body"):
            try:
                self._wait(selector, timeout=THREAD_LOAD_WAIT)
                loaded = True
                break
            except TimeoutException:
                continue

        if not loaded:
            log.warning("Thread %s: no message elements found after load", tid)
            return []

        # Dismiss any popup that appeared on load
        self._close_popups()

        edges = []

        # Extract subject from thread header
        subject = ""
        try:
            subj_el = self.driver.find_element(
                By.CSS_SELECTOR, "div.thread-subject, div.thread-subject-left"
            )
            subject = subj_el.text.strip()
        except NoSuchElementException:
            pass

        # Iterate over message items
        messages = self.driver.find_elements(By.CSS_SELECTOR, "div.message-item")
        if not messages:
            # Fallback: page may structure messages under message-header only
            messages = self.driver.find_elements(
                By.CSS_SELECTOR, "div.message-header"
            )

        for msg in messages:
            try:
                # Message ID
                msg_id = msg.get_attribute("id") or ""
                if not msg_id:
                    msg_id = (msg.get_attribute("data-message-id")
                              or msg.get_attribute("data-first-message-id")
                              or f"{tid}_msg_{random.randint(1000, 9999)}")

                # Sender
                sender = extract_sender_email_or_name(msg)

                # Date
                dt = ""
                try:
                    dt_el = msg.find_element(By.CSS_SELECTOR, "div.date-time")
                    dt = dt_el.text.strip()
                except NoSuchElementException:
                    pass

                # Recipients
                recipients = []
                try:
                    recip_els = msg.find_elements(
                        By.CSS_SELECTOR, "div.recipient-info"
                    )
                    for rel in recip_els:
                        raw = rel.text.strip()
                        parsed = parse_recipients(raw)
                        recipients.extend(parsed)
                except NoSuchElementException:
                    pass

                if not recipients:
                    recipients = [("unknown", "unknown")]

                for (recip, rtype) in recipients:
                    edges.append(EdgeRecord(
                        message_id=msg_id,
                        thread_id=tid,
                        sender=sender,
                        recipient=recip,
                        recipient_type=rtype,
                        datetime=dt,
                        subject=subject,
                    ))

            except StaleElementReferenceException:
                log.warning("Stale element in thread %s, skipping message", tid)
                continue
            except Exception as e:
                log.warning("Error extracting message in thread %s: %s", tid, e)
                continue

        return edges

    def _build_nodes_if_needed(self):
        if os.path.exists(self.edges_path) and os.path.getsize(self.edges_path) > 0:
            build_nodes_csv(self.edges_path, self.nodes_path)

    # ======================================================================
    #  Combined run
    # ======================================================================
    def run_both(self):
        self.run_phase1()
        self.run_phase2()
        log.info("=" * 60)
        log.info("All done. Outputs in: %s", self.outdir)
        log.info("  thread_ids.txt:        %s", self.thread_ids_path)
        log.info("  processed_threads.txt: %s", self.processed_path)
        log.info("  edges.csv:             %s", self.edges_path)
        log.info("  nodes.csv:             %s", self.nodes_path)
        log.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="2-phase Jmail metadata scraper for network analysis"
    )
    parser.add_argument(
        "--phase", type=int, choices=[1, 2], default=0,
        help="Run only Phase 1 (collect IDs) or Phase 2 (crawl threads). "
             "Default: run both."
    )
    parser.add_argument(
        "--max-rows", type=int, default=0,
        help="Phase 1: max inbox rows to scan (0 = unlimited)"
    )
    parser.add_argument(
        "--outdir", default="./jmail_output",
        help="Output directory (default: %(default)s)"
    )
    parser.add_argument(
        "--no-headless", action="store_true",
        help="Run with visible browser (for debugging)"
    )
    args = parser.parse_args()

    scraper = JmailScraper(
        outdir=args.outdir,
        max_rows=args.max_rows,
        headless=not args.no_headless,
    )

    if args.phase == 1:
        scraper.run_phase1()
    elif args.phase == 2:
        scraper.run_phase2()
    else:
        scraper.run_both()


if __name__ == "__main__":
    main()
