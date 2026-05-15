"""
volatility/estimators.py

Four rolling volatility estimators for the three commodity futures.

Estimators
----------
1. rv_std_{N}  — Rolling standard deviation of log returns (Close-to-Close)
                 Simple, widely used; sensitive to gaps and outliers.
                 N ∈ {5, 10, 20} trading days

2. rv_atr_{N}  — Average True Range (normalised by Close)
                 Captures intra-day range including overnight gaps.
                 N = 14 (Wilder's standard)

3. rv_park_{N} — Parkinson (1980) range-based estimator
                 More efficient than Close-to-Close; uses High/Low only.
                 N ∈ {10, 20}

4. rv_yz_{N}   — Yang-Zhang (2000) estimator
                 Handles both overnight gaps and open-to-close moves.
                 Most efficient of the four; requires O, H, L, C.
                 N ∈ {10, 20}

Primary metric for Granger analysis: rv_std_20 (annualised is optional but
kept as-is in daily standard deviation units for interpretability).

Output schema (volatility_daily.parquet)
-----------------------------------------
Index: date (datetime64[ns])
Columns: {ticker}_{estimator}  e.g. CL=F_rv_std_20, GC=F_rv_park_10, …
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config
from ingestion.commodity_loader import read_commodity_ohlcv

log = logging.getLogger(__name__)


# ── Estimator implementations ─────────────────────────────────────────────────

def rolling_std(log_ret_clean: pd.Series, window: int) -> pd.Series:
    """
    Rolling standard deviation of clean log returns.

    NaN entries (roll dates) are excluded from the window via skipna logic
    achieved by a custom rolling apply.
    """
    # pandas rolling with min_periods skips NaN naturally in std calculation
    return log_ret_clean.rolling(window=window, min_periods=max(window // 2, 5)).std()


def average_true_range(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int
) -> pd.Series:
    """
    ATR normalised by the previous close so it is dimensionless (like a return).

    True Range = max(H-L, |H-C_{t-1}|, |L-C_{t-1}|)
    ATR = EWM of TR (Wilder's smoothing ≈ span=2*window-1)
    Normalised ATR = ATR / Close_{t-1}
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Wilder's smoothing: alpha = 1/window  →  span = 2*window - 1
    atr = tr.ewm(span=2 * window - 1, adjust=False, min_periods=window).mean()
    return atr / prev_close.replace(0, np.nan)


def parkinson(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """
    Parkinson (1980) estimator.

    σ² ≈ (1 / (4 ln 2)) × E[(ln H/L)²]
    Daily Parkinson variance: (ln(H/L))² / (4 ln 2)
    Rolling estimate: mean of daily variances over the window → take sqrt.
    """
    ln_hl_sq = (np.log(high / low.replace(0, np.nan))) ** 2
    factor   = 1.0 / (4.0 * np.log(2.0))
    daily_var = factor * ln_hl_sq
    rolling_var = daily_var.rolling(window=window, min_periods=max(window // 2, 5)).mean()
    return np.sqrt(rolling_var)


def yang_zhang(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int,
    k: float = 0.34,
) -> pd.Series:
    """
    Yang-Zhang (2000) volatility estimator.

    Combines overnight (close-to-open) and open-to-close (Rogers-Satchell) variances.
    k = 0.34 is the standard weighting recommended by Yang & Zhang.

    σ²_YZ = σ²_overnight  +  k × σ²_open_close  +  (1-k) × σ²_RS
    """
    log_oc = np.log(open_ / close.shift(1))  # overnight return
    log_co = np.log(close / open_)            # open-to-close return
    log_ho = np.log(high  / open_)
    log_lo = np.log(low   / open_)

    # Overnight variance
    mean_oc = log_oc.rolling(window, min_periods=5).mean()
    var_oc  = ((log_oc - mean_oc) ** 2).rolling(window, min_periods=5).mean()

    # Open-to-close variance
    mean_co = log_co.rolling(window, min_periods=5).mean()
    var_co  = ((log_co - mean_co) ** 2).rolling(window, min_periods=5).mean()

    # Rogers-Satchell variance (drift-independent)
    var_rs = (log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)).rolling(
        window, min_periods=5
    ).mean()

    var_yz = var_oc + k * var_co + (1 - k) * var_rs
    return np.sqrt(var_yz.clip(lower=0))


# ── Main builder ──────────────────────────────────────────────────────────────

def compute_all_volatility(
    returns_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """
    Compute all four volatility estimators for all tickers and write Parquet.

    Parameters
    ----------
    returns_df  : output of volatility.returns.compute_log_returns()
    output_path : destination Parquet

    Returns
    -------
    pd.DataFrame — one column per (ticker, estimator, window) combination
    """
    ohlcv = read_commodity_ohlcv()

    all_cols: dict[str, pd.Series] = {}

    for ticker in config.TICKERS:
        log.info("Computing volatility estimators for %s…", ticker)

        # ── Price series ───────────────────────────────────────────────────
        try:
            close  = ohlcv[(ticker, "Close")]
            high   = ohlcv[(ticker, "High")]
            low    = ohlcv[(ticker, "Low")]
            open_  = ohlcv[(ticker, "Open")]
        except KeyError:
            log.warning("Missing OHLC for %s — skipping.", ticker)
            continue

        # Clean log return (roll dates → NaN)
        log_ret_clean_col = f"{ticker}_log_ret_clean"
        if log_ret_clean_col not in returns_df.columns:
            log.warning("No clean log return for %s — skipping.", ticker)
            continue
        log_ret_clean = returns_df[log_ret_clean_col].reindex(close.index)

        # ── 1. Rolling standard deviation ──────────────────────────────────
        for n in config.VOL_WINDOWS:
            all_cols[f"{ticker}_rv_std_{n}"] = rolling_std(log_ret_clean, n)

        # ── 2. ATR ─────────────────────────────────────────────────────────
        all_cols[f"{ticker}_rv_atr_{config.ATR_WINDOW}"] = average_true_range(
            high, low, close, config.ATR_WINDOW
        )

        # ── 3. Parkinson ───────────────────────────────────────────────────
        for n in [10, 20]:
            all_cols[f"{ticker}_rv_park_{n}"] = parkinson(high, low, n)

        # ── 4. Yang-Zhang ──────────────────────────────────────────────────
        for n in [10, 20]:
            all_cols[f"{ticker}_rv_yz_{n}"] = yang_zhang(open_, high, low, close, n)

    if not all_cols:
        raise RuntimeError("No volatility columns produced. Check OHLCV and returns data.")

    result = pd.DataFrame(all_cols)
    result.index.name = "date"
    result = result.sort_index()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path)

    log.info(
        "Volatility Parquet written: %d rows, %d columns, path=%s",
        len(result), len(result.columns), output_path,
    )
    return result


def load_volatility(path: Path = config.VOLATILITY_PATH) -> pd.DataFrame:
    """Load the cached volatility Parquet."""
    return pd.read_parquet(path)
