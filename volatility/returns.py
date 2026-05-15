"""
volatility/returns.py

Computes daily log returns from the cached commodity OHLCV Parquet and flags
probable contract roll dates (large mechanical price gaps).

Log return definition
----------------------
    r_t = ln(Close_t / Close_{t-1})

Roll-date masking
-----------------
Futures continuous series (CL=F, GC=F, ZW=F) from Yahoo Finance are NOT
back-adjusted.  On contract expiry dates the price can jump by several
percent as the front month rolls to the next contract — this is a data
artifact, not a geopolitical signal.  We detect these via:

    |r_t| > ROLL_DETECTION_THRESHOLD  (default 3%)

and flag them with a boolean mask column `{ticker}_is_roll`.  Downstream
volatility estimators skip these returns so roll-induced spikes don't
contaminate the 20-day rolling standard deviation.

Output of compute_log_returns()
---------------------------------
pd.DataFrame, index = date, columns:
    {ticker}_log_ret     — raw daily log return (NaN on day 1 and roll dates)
    {ticker}_log_ret_clean — log return with roll dates set to NaN
    {ticker}_is_roll     — boolean flag
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config
from ingestion.commodity_loader import read_commodity_ohlcv

log = logging.getLogger(__name__)


def compute_log_returns(ohlcv_path: Path = config.COMMODITY_OHLCV_PATH) -> pd.DataFrame:
    """
    Compute log returns for all tickers and detect roll dates.

    Parameters
    ----------
    ohlcv_path : path to commodity_ohlcv.parquet

    Returns
    -------
    pd.DataFrame with columns {ticker}_log_ret, {ticker}_log_ret_clean,
    {ticker}_is_roll for each ticker in config.TICKERS
    """
    ohlcv = read_commodity_ohlcv(ohlcv_path)

    frames: dict[str, pd.Series] = {}

    for ticker in config.TICKERS:
        try:
            close = ohlcv[(ticker, "Close")].dropna()
        except KeyError:
            log.warning("No Close data for %s in OHLCV cache — skipping.", ticker)
            continue

        # Raw log return
        log_ret = np.log(close / close.shift(1))

        # Roll detection from is_roll_date column (set by commodity_loader)
        try:
            is_roll = ohlcv[(ticker, "is_roll_date")].reindex(log_ret.index).fillna(False)
        except KeyError:
            # Fallback: detect from return magnitude
            is_roll = log_ret.abs() > config.ROLL_DETECTION_THRESHOLD

        # Clean return: NaN on roll dates
        log_ret_clean = log_ret.where(~is_roll, other=np.nan)

        frames[f"{ticker}_log_ret"]       = log_ret
        frames[f"{ticker}_log_ret_clean"] = log_ret_clean
        frames[f"{ticker}_is_roll"]       = is_roll

        n_rolls = is_roll.sum()
        log.info(
            "%s: %d trading days, %d roll dates detected (%.1f%%)",
            ticker, len(close), n_rolls, 100 * n_rolls / max(len(close), 1),
        )

    if not frames:
        raise RuntimeError("No log returns computed. Check commodity_ohlcv.parquet.")

    result = pd.DataFrame(frames)
    result.index.name = "date"
    return result
