"""
ingestion/gdelt_downloader.py

Downloads GDELT V2 export CSV.zip files for a given date range from the public
master file list at data.gdeltproject.org.

Strategy
--------
- Fetch the master file list (plain-text, one URL per 15-min slot).
- Filter to .export.CSV.zip files within the requested date window.
- Download each file to data/raw/, skipping files already present.
- Files are downloaded in parallel (ThreadPoolExecutor, configurable workers).
- Returns a sorted list of local file paths for the parser to consume.

Note on data volume
-------------------
A 5-year window has ~175,200 files.  The parser is designed to aggregate one
calendar day at a time and delete raw zips after processing, so peak disk
usage stays at roughly 1 day's worth (~400 MB compressed).

The downloader therefore groups file URLs by calendar day and yields batches
rather than downloading everything upfront.
"""

import io
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, List
from urllib.parse import urlparse

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import config

log = logging.getLogger(__name__)

# GDELT V2 export file URL pattern: YYYYMMDDHHMMSS.export.CSV.zip
_EXPORT_RE = re.compile(r"(\d{14})\.export\.CSV\.zip$", re.IGNORECASE)


# ── Master file list ──────────────────────────────────────────────────────────

def _fetch_master_list() -> List[str]:
    """
    Download the GDELT V2 master file list and return all .export.CSV.zip URLs.

    The file has lines of the form:
        <size_bytes> <md5_hash> <url>
    or occasionally just:
        <url>
    We parse flexibly by always taking the last whitespace-delimited token on
    each line as the URL.
    """
    log.info("Fetching GDELT master file list from %s", config.GDELT_MASTER_URL)
    resp = requests.get(
        config.GDELT_MASTER_URL,
        timeout=config.GDELT_REQUEST_TIMEOUT,
        stream=True,
    )
    resp.raise_for_status()

    urls: List[str] = []
    for line in resp.iter_lines(decode_unicode=True):
        line = line.strip()
        if not line:
            continue
        # Last token on each line is the URL
        url = line.split()[-1]
        if _EXPORT_RE.search(url):
            urls.append(url)

    log.info("Master list parsed: %d export file entries found.", len(urls))
    return urls


def _url_to_date(url: str) -> datetime | None:
    """Extract the UTC datetime embedded in a GDELT export file URL."""
    m = _EXPORT_RE.search(url)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def filter_urls_by_date(urls: List[str], start: str, end: str) -> List[str]:
    """
    Return URLs whose embedded datetime falls within [start, end] inclusive.

    Parameters
    ----------
    urls  : list of GDELT export file URLs
    start : "YYYY-MM-DD"
    end   : "YYYY-MM-DD"
    """
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d") + timedelta(hours=23, minutes=59)

    filtered = [u for u in urls if (dt := _url_to_date(u)) and start_dt <= dt <= end_dt]
    log.info(
        "Date filter [%s → %s]: %d / %d URLs selected.",
        start, end, len(filtered), len(urls)
    )
    return sorted(filtered)


# ── Single-file download ──────────────────────────────────────────────────────

@retry(
    retry=retry_if_exception_type((requests.RequestException, IOError)),
    stop=stop_after_attempt(config.GDELT_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=config.GDELT_RETRY_BACKOFF, min=1, max=30),
    reraise=True,
)
def _download_one(url: str, dest: Path) -> Path:
    """
    Download a single GDELT export zip to *dest*.

    Skips if the file already exists and is non-empty (cache hit).
    Returns the local path on success.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return dest  # cache hit

    resp = requests.get(url, timeout=config.GDELT_REQUEST_TIMEOUT, stream=True)
    resp.raise_for_status()

    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=65_536):
            fh.write(chunk)
    return dest


def _url_to_local_path(url: str) -> Path:
    """Map a GDELT URL to its local cache path under data/raw/YYYYMMDD/."""
    filename = Path(urlparse(url).path).name
    dt = _url_to_date(url)
    date_dir = dt.strftime("%Y%m%d") if dt else "unknown"
    return config.RAW_DIR / date_dir / filename


# ── Day-batched download (memory-efficient) ───────────────────────────────────

def _group_by_day(urls: List[str]) -> dict[str, List[str]]:
    """Group URLs by calendar date string YYYYMMDD."""
    groups: dict[str, List[str]] = {}
    for url in urls:
        dt = _url_to_date(url)
        if dt:
            key = dt.strftime("%Y%m%d")
            groups.setdefault(key, []).append(url)
    return dict(sorted(groups.items()))


def download_day_batch(day_urls: List[str]) -> List[Path]:
    """
    Download all files for one calendar day in parallel.

    Returns a list of local file paths (sorted by timestamp).
    """
    paths: List[Path] = []
    failed: List[str] = []

    with ThreadPoolExecutor(max_workers=config.GDELT_MAX_WORKERS) as pool:
        future_to_url = {
            pool.submit(_download_one, url, _url_to_local_path(url)): url
            for url in day_urls
        }
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                paths.append(future.result())
            except Exception as exc:
                log.warning("Failed to download %s: %s", url, exc)
                failed.append(url)

    if failed:
        log.warning("%d file(s) could not be downloaded and will be skipped.", len(failed))

    return sorted(paths)


# ── Public API ────────────────────────────────────────────────────────────────

def download_gdelt_range(start: str, end: str) -> dict[str, List[Path]]:
    """
    Download all GDELT V2 export files for the date range [start, end].

    Returns a dict mapping "YYYYMMDD" → list of local file Paths for that day.
    The caller (gdelt_parser) iterates over this dict one day at a time to
    keep memory usage bounded.

    Parameters
    ----------
    start : "YYYY-MM-DD"  (inclusive)
    end   : "YYYY-MM-DD"  (inclusive)
    """
    all_urls   = _fetch_master_list()
    range_urls = filter_urls_by_date(all_urls, start, end)
    day_groups = _group_by_day(range_urls)

    log.info(
        "Starting download of %d files across %d calendar days.",
        len(range_urls), len(day_groups)
    )

    day_paths: dict[str, List[Path]] = {}
    for i, (day, urls) in enumerate(day_groups.items(), 1):
        log.info("Day %d/%d  (%s) — %d files", i, len(day_groups), day, len(urls))
        day_paths[day] = download_day_batch(urls)

    log.info("Download complete: %d days, %d files total.",
             len(day_paths), sum(len(v) for v in day_paths.values()))
    return day_paths
