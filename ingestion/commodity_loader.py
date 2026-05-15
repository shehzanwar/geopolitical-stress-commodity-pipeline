"""
ingestion/commodity_loader.py

Fetches daily OHLCV data for the three commodity futures using yfinance and
writes a Parquet cache.

Robustness features
--------------------
- Exponential backoff on YFRateLimitError (tenacity)
- Coverage validation: warns if any ticker has < MIN_COMMODITY_COVERAGE of
  expected trading days
- Roll-date detection: flags sessions where |log_return| > ROLL_DETECTION_THRESHOLD
  as probable contract-expiry distortions
- Graceful fallback message if all retries fail (user is instructed to check
  the pandas_datareader fallback in config)

Output schema (commodity_ohlcv.parquet)
----------------------------------------
MultiIndex columns: (ticker, field) where field ∈ {Open, High, Low, Close, Volume}
Plus boolean column (ticker, "is_roll_date") for each ticker.
Index: date (datetime64[ns])
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

import config

log = logging.getLogger(__name__)

# ── Retry parameters for yfinance ─────────────────────────────────────────────
_MAX_RETRIES  = 3
_BACKOFF_BASE = 2.0   # seconds × 2^attempt


def _fetch_with_retry(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Download OHLCV for all tickers, retrying on rate-limit or network errors.

    yfinance returns a DataFrame with MultiIndex columns (field, ticker).
    We transpose to (ticker, field) for convenience.
    """
    last_exc: Exception | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            raw = yf.download(
                tickers=tickers,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=True,   # adjusts for splits; dividends irrelevant for futures
                progress=False,
                threads=True,
            )
            if raw.empty:
                raise ValueError("yfinance returned an empty DataFrame.")
            return raw
        except Exception as exc:
            last_exc = exc
            wait = _BACKOFF_BASE ** attempt
            log.warning(
                "yfinance attempt %d/%d failed: %s  — retrying in %.0fs",
                attempt, _MAX_RETRIES, exc, wait,
            )
            time.sleep(wait)

    raise RuntimeError(
        f"All {_MAX_RETRIES} yfinance attempts failed. Last error: {last_exc}\n"
        "Consider switching to pandas_datareader (see config.py fallback note)."
    )


def _detect_roll_dates(close_series: pd.Series, threshold: float) -> pd.Series:
    """
    Return a boolean Series marking likely contract roll dates.

    A roll date is flagged when the absolute log-return exceeds *threshold*
    (default 3%), indicating a mechanical price gap rather than a market move.
    """
    log_ret = np.log(close_series / close_series.shift(1))
    return log_ret.abs() > threshold


def _validate_coverage(df: pd.DataFrame, start: str, end: str) -> None:
    """
    Warn if any ticker covers less than MIN_COMMODITY_COVERAGE of business days.
    """
    expected_days = len(pd.bdate_range(start, end))
    if expected_days == 0:
        return

    for ticker in config.TICKERS:
        try:
            n_actual = df["Close"][ticker].dropna().shape[0]
        except KeyError:
            log.warning("Ticker %s not found in downloaded data.", ticker)
            continue

        coverage = n_actual / expected_days
        if coverage < config.MIN_COMMODITY_COVERAGE:
            log.warning(
                "LOW COVERAGE: %s has %.1f%% of expected %d trading days "
                "(got %d). Data quality may be impacted.",
                ticker, coverage * 100, expected_days, n_actual,
            )
        else:
            log.info(
                "Coverage OK: %s — %d / %d business days (%.1f%%)",
                ticker, n_actual, expected_days, coverage * 100,
            )


def _build_output(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """
    Reshape raw yfinance output and add is_roll_date columns.

    Input:  MultiIndex columns (field, ticker)  or flat columns for single ticker
    Output: MultiIndex columns (ticker, field) + (ticker, 'is_roll_date')
    """
    # yfinance returns (field, ticker) MultiIndex for multiple tickers
    # Swap to (ticker, field)
    if isinstance(raw.columns, pd.MultiIndex):
        df = raw.swaplevel(axis=1).sort_index(axis=1)
    else:
        # Single ticker — wrap in MultiIndex
        df = raw.copy()
        df.columns = pd.MultiIndex.from_product([[tickers[0]], df.columns])

    # Add roll-date flags per ticker
    roll_frames = {}
    for ticker in tickers:
        try:
            close = df[(ticker, "Close")]
        except KeyError:
            log.warning("No Close data for ticker %s", ticker)
            continue
        roll = _detect_roll_dates(close, config.ROLL_DETECTION_THRESHOLD)
        roll_frames[(ticker, "is_roll_date")] = roll

    if roll_frames:
        roll_df = pd.DataFrame(roll_frames, index=df.index)
        df = pd.concat([df, roll_df], axis=1)

    df = df.sort_index()
    df.index.name = "date"
    return df


# ── Public API ────────────────────────────────────────────────────────────────

def load_commodity_ohlcv(
    tickers: list[str],
    start: str,
    end: str,
    output_path: Path,
) -> pd.DataFrame:
    """
    Fetch, validate, and cache commodity OHLCV data to Parquet.

    Parameters
    ----------
    tickers     : list of Yahoo Finance ticker strings (e.g. ["CL=F", "GC=F", "ZW=F"])
    start       : "YYYY-MM-DD"
    end         : "YYYY-MM-DD"
    output_path : destination Parquet file

    Returns
    -------
    pd.DataFrame with MultiIndex columns (ticker, field)
    """
    log.info("Fetching OHLCV for %s from %s to %s", tickers, start, end)

    raw = _fetch_with_retry(tickers, start, end)
    _validate_coverage(raw, start, end)
    out = _build_output(raw, tickers)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path)

    log.info(
        "Commodity OHLCV written: %d trading days, %d tickers, path=%s",
        len(out), len(tickers), output_path,
    )
    return out


def read_commodity_ohlcv(path: Path = config.COMMODITY_OHLCV_PATH) -> pd.DataFrame:
    """Load the cached Parquet; restore MultiIndex column structure."""
    df = pd.read_parquet(path)
    if not isinstance(df.columns, pd.MultiIndex):
        # Parquet sometimes flattens MultiIndex columns — reconstruct
        tuples = [tuple(c.split("_", 1)) if "_" in str(c) else (c, "") for c in df.columns]
        df.columns = pd.MultiIndex.from_tuples(tuples)
    return df
