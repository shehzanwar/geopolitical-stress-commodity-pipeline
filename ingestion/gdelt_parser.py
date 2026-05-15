"""
ingestion/gdelt_parser.py

Parses GDELT V2 export CSV.zip files into a daily-aggregated Parquet file.

GDELT V2 export format
-----------------------
Tab-delimited, no header, 61 columns (see RESEARCH.md §2.2 for the full index).
Key column indices used here:
    0  GLOBALEVENTID       — unique event key (for deduplication)
    1  SQLDATE             — YYYYMMDD integer (event date)
   26  EventCode           — full CAMEO leaf code (stored as string; may be zero-padded)
   27  EventBaseCode       — 3-digit CAMEO code
   28  EventRootCode       — 2-digit CAMEO root (01–20) — primary grouping key
   30  GoldsteinScale      — float −10 to +10 (conflict intensity)
   31  NumMentions         — int (article mention count, used as event weight)
   32  NumSources          — int (distinct outlets)
   33  NumArticles         — int (quality filter: discard if < GDELT_MIN_ARTICLES)
   34  AvgTone             — float −100 to +100 (document sentiment)
   59  DATEADDED           — YYYYMMDDHHmmss timestamp (15-min resolution)

Output schema (gdelt_daily.parquet)
------------------------------------
One row per calendar date. Columns:
    date           — datetime64[ns] (index)
    event_root     — str "01"–"20"
    n_events       — int   (deduplicated event count for this root on this date)
    n_mentions     — int   (sum of NumMentions)
    n_articles     — int   (sum of NumArticles)
    sum_goldstein  — float (sum of GoldsteinScale × NumMentions, for weighted avg)
    sum_tone       — float (sum of AvgTone × NumMentions)

The CAMEO scorer (nlp/cameo_scorer.py) derives the final per-category scores
from this intermediate representation.
"""

import io
import logging
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

import config

log = logging.getLogger(__name__)

# ── Column layout ─────────────────────────────────────────────────────────────
# GDELT V2 export has 61 tab-delimited columns, zero-indexed.
_USECOLS = [0, 1, 26, 27, 28, 30, 31, 32, 33, 34]

_COL_NAMES = {
    0:  "GlobalEventID",
    1:  "SQLDATE",
    26: "EventCode",
    27: "EventBaseCode",
    28: "EventRootCode",
    30: "GoldsteinScale",
    31: "NumMentions",
    32: "NumSources",
    33: "NumArticles",
    34: "AvgTone",
}

_DTYPES = {
    "GlobalEventID": "int64",
    "SQLDATE":        "int32",
    "EventCode":      "str",
    "EventBaseCode":  "str",
    "EventRootCode":  "str",
    "GoldsteinScale": "float32",
    "NumMentions":    "int32",
    "NumSources":     "int16",
    "NumArticles":    "int16",
    "AvgTone":        "float32",
}

# Valid CAMEO root codes
_VALID_ROOTS = {str(i).zfill(2) for i in range(1, 21)}


# ── Single-file reader ────────────────────────────────────────────────────────

def _read_zip(path: Path) -> Optional[pd.DataFrame]:
    """
    Read one GDELT export zip into a DataFrame with selected columns.

    Returns None if the file is empty, corrupt, or unreadable.
    """
    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
            if not names:
                log.warning("Empty zip: %s", path.name)
                return None
            csv_name = names[0]
            with zf.open(csv_name) as csv_fh:
                df = pd.read_csv(
                    csv_fh,
                    sep="\t",
                    header=None,
                    usecols=_USECOLS,
                    dtype=str,           # read all as str first to avoid parse errors
                    on_bad_lines="skip",
                    low_memory=False,
                )
    except (zipfile.BadZipFile, Exception) as exc:
        log.warning("Cannot read %s: %s", path.name, exc)
        return None

    if df.empty:
        return None

    # Rename columns
    df.columns = [_COL_NAMES[c] for c in _USECOLS]

    # Cast to correct types (coerce errors → NaN)
    for col, dtype in _DTYPES.items():
        if dtype in ("float32", "float64"):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
        elif dtype in ("int64", "int32", "int16"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # string columns stay as str

    return df


# ── Day-level aggregation ─────────────────────────────────────────────────────

def _aggregate_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply quality filters, deduplicate, and aggregate to per-(date, root) rows.

    Steps:
        1. Drop rows where NumArticles < GDELT_MIN_ARTICLES (noise filter)
        2. Normalise EventRootCode to zero-padded 2-digit string
        3. Keep only valid CAMEO roots (01–20)
        4. Deduplicate by GlobalEventID (same event may appear in multiple 15-min files)
        5. Weight GoldsteinScale and AvgTone by NumMentions
        6. Group by (SQLDATE, EventRootCode) and sum weighted values
    """
    # ── Quality filter ─────────────────────────────────────────────────────
    df = df.dropna(subset=["GlobalEventID", "NumArticles", "EventRootCode"])
    df = df[df["NumArticles"] >= config.GDELT_MIN_ARTICLES].copy()

    if df.empty:
        return pd.DataFrame()

    # ── Normalise root code ────────────────────────────────────────────────
    df["EventRootCode"] = (
        df["EventRootCode"]
        .astype(str)
        .str.strip()
        .str.zfill(2)
    )
    df = df[df["EventRootCode"].isin(_VALID_ROOTS)]

    if df.empty:
        return pd.DataFrame()

    # ── Cast numerics after filtering ──────────────────────────────────────
    df["GlobalEventID"] = pd.to_numeric(df["GlobalEventID"], errors="coerce")
    df["SQLDATE"]       = pd.to_numeric(df["SQLDATE"], errors="coerce")
    df["NumMentions"]   = pd.to_numeric(df["NumMentions"], errors="coerce").fillna(0).astype(int)
    df["NumArticles"]   = pd.to_numeric(df["NumArticles"], errors="coerce").fillna(0).astype(int)
    df["GoldsteinScale"] = pd.to_numeric(df["GoldsteinScale"], errors="coerce").fillna(0.0)
    df["AvgTone"]       = pd.to_numeric(df["AvgTone"], errors="coerce").fillna(0.0)

    df = df.dropna(subset=["GlobalEventID", "SQLDATE"])

    # ── Deduplicate within this day's batch ────────────────────────────────
    df = df.drop_duplicates(subset=["GlobalEventID"])

    # ── Weighted columns ───────────────────────────────────────────────────
    df["weighted_goldstein"] = df["GoldsteinScale"] * df["NumMentions"]
    df["weighted_tone"]      = df["AvgTone"]        * df["NumMentions"]

    # ── Group by date × CAMEO root ─────────────────────────────────────────
    grp = (
        df.groupby(["SQLDATE", "EventRootCode"], sort=True)
        .agg(
            n_events       = ("GlobalEventID",    "count"),
            n_mentions     = ("NumMentions",       "sum"),
            n_articles     = ("NumArticles",       "sum"),
            sum_goldstein  = ("weighted_goldstein", "sum"),
            sum_tone       = ("weighted_tone",      "sum"),
        )
        .reset_index()
    )

    # Convert SQLDATE integer to proper datetime
    grp["date"] = pd.to_datetime(grp["SQLDATE"].astype(str), format="%Y%m%d", errors="coerce")
    grp = grp.drop(columns=["SQLDATE"]).rename(columns={"EventRootCode": "event_root"})
    grp = grp.dropna(subset=["date"])

    return grp[["date", "event_root", "n_events", "n_mentions", "n_articles",
                "sum_goldstein", "sum_tone"]]


# ── Public API ────────────────────────────────────────────────────────────────

def parse_gdelt_to_daily_parquet(
    day_paths: Dict[str, List[Path]],
    output_path: Path,
) -> pd.DataFrame:
    """
    Parse all downloaded GDELT export files into a single daily Parquet.

    Processes one calendar day at a time (memory-efficient).  Accumulated
    cross-day deduplication is handled via GlobalEventID tracking per day —
    cross-day duplicates are rare in GDELT V2 but retained here as-is since
    the day is the analytical unit.

    Parameters
    ----------
    day_paths   : dict returned by gdelt_downloader.download_gdelt_range()
                  "YYYYMMDD" → [Path, ...]
    output_path : destination Parquet file

    Returns
    -------
    pd.DataFrame  (same schema written to disk)
    """
    all_days: List[pd.DataFrame] = []
    n_days = len(day_paths)

    for i, (day, paths) in enumerate(sorted(day_paths.items()), 1):
        if i % 100 == 0 or i == 1:
            log.info("Parsing day %d/%d (%s) — %d files", i, n_days, day, len(paths))

        # Read all 15-min files for this day into one DataFrame
        frames = [_read_zip(p) for p in sorted(paths)]
        frames = [f for f in frames if f is not None and not f.empty]

        if not frames:
            log.debug("No usable data for day %s", day)
            continue

        day_df = pd.concat(frames, ignore_index=True)
        agg    = _aggregate_day(day_df)

        if not agg.empty:
            all_days.append(agg)

        # Release memory
        del frames, day_df

    if not all_days:
        raise RuntimeError("Parser produced no usable rows. Check GDELT downloads.")

    result = pd.concat(all_days, ignore_index=True)

    # Final sort and dedup (multiple files on same day should be merged already,
    # but guard against any remaining duplicates from the concat)
    result = (
        result
        .groupby(["date", "event_root"], sort=True)
        .agg(
            n_events      = ("n_events",      "sum"),
            n_mentions    = ("n_mentions",    "sum"),
            n_articles    = ("n_articles",    "sum"),
            sum_goldstein = ("sum_goldstein", "sum"),
            sum_tone      = ("sum_tone",      "sum"),
        )
        .reset_index()
    )

    result = result.sort_values(["date", "event_root"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)

    log.info(
        "GDELT daily Parquet written: %d rows, %d unique dates, path=%s",
        len(result),
        result["date"].nunique(),
        output_path,
    )
    return result
