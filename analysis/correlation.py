"""
analysis/correlation.py

Cross-correlation and lead-lag analysis between CAMEO category scores and
commodity volatility.

What is computed
-----------------
1. Pearson and Spearman CCF at lags 0, +1, +5, +10, +20 trading days
   for each (CAMEO root, commodity) pair — 20 roots × 3 commodities × 5 lags
   = 300 values per correlation type.

   Positive lag = CAMEO leads volatility (the causal direction of interest).
   Negative lag = volatility leads CAMEO (feedback / market pricing-in).

2. Rolling 90-business-day Pearson correlation between the composite GSI
   and each commodity's primary volatility metric — captures time-varying
   strength of the relationship (stronger during crises).

Output schema (correlation_matrix.csv)
----------------------------------------
    cameo_root    — "01"–"20"
    cameo_label   — category name
    commodity     — "CL=F" | "GC=F" | "ZW=F"
    lag_days      — 0 | 1 | 5 | 10 | 20
    pearson_r     — Pearson correlation coefficient
    pearson_p     — p-value for Pearson r
    spearman_r    — Spearman rank correlation
    spearman_p    — p-value for Spearman r

Rolling GSI correlation is written to data/processed/gsi_rolling_corr.parquet
for use in the reporter.
"""

import logging
import warnings
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import config
from nlp.cameo_codes import CAMEO_ROOTS, ALL_ROOTS, score_col
from nlp.stress_index import load_gsi

log = logging.getLogger(__name__)

_ROLLING_CORR_PATH = config.PROCESSED_DIR / "gsi_rolling_corr.parquet"
_LAG_DAYS = [0, 1, 5, 10, 20]


# ── Utility ───────────────────────────────────────────────────────────────────

def _lagged_pair(x: pd.Series, y: pd.Series, lag: int) -> pd.DataFrame:
    """
    Align x and y with x lagged by *lag* days relative to y.

    lag > 0 : x leads y (CAMEO predicts future volatility)
    lag < 0 : y leads x (volatility predicts past CAMEO — reverse causality)
    """
    if lag >= 0:
        x_lagged = x.shift(lag)
    else:
        x_lagged = x.shift(lag)   # negative shift = lead
    pair = pd.concat([x_lagged, y], axis=1).dropna()
    pair.columns = ["x", "y"]
    return pair


def _safe_pearson(x: np.ndarray, y: np.ndarray):
    try:
        r, p = pearsonr(x, y)
        return float(r), float(p)
    except Exception:
        return np.nan, np.nan


def _safe_spearman(x: np.ndarray, y: np.ndarray):
    try:
        r, p = spearmanr(x, y)
        return float(r), float(p)
    except Exception:
        return np.nan, np.nan


# ── CCF computation ───────────────────────────────────────────────────────────

def _compute_ccf(
    cameo_df: pd.DataFrame,
    vol_df: pd.DataFrame,
    lags: List[int] = _LAG_DAYS,
) -> pd.DataFrame:
    """
    Compute Pearson and Spearman correlations at each lag for all
    (CAMEO root, commodity, lag) combinations.
    """
    rows: List[Dict] = []

    # Suppress scipy ConstantInputWarning — arises for sparse CAMEO roots with
    # many zero days; NaN is returned by the safe_* wrappers in those cases.
    warnings.filterwarnings("ignore", message="An input array is constant")

    for root in ALL_ROOTS:
        x = cameo_df.get(score_col(root))
        if x is None:
            continue
        label = CAMEO_ROOTS[root].label

        for ticker in config.TICKERS:
            vol_col = f"{ticker}_{config.PRIMARY_VOL_METRIC}"
            y = vol_df.get(vol_col)
            if y is None:
                continue

            for lag in lags:
                pair = _lagged_pair(x, y, lag)
                if len(pair) < 30:
                    continue

                pr, pp = _safe_pearson(pair["x"].values,  pair["y"].values)
                sr, sp = _safe_spearman(pair["x"].values, pair["y"].values)

                rows.append({
                    "cameo_root":  root,
                    "cameo_label": label,
                    "commodity":   ticker,
                    "lag_days":    lag,
                    "pearson_r":   pr,
                    "pearson_p":   pp,
                    "spearman_r":  sr,
                    "spearman_p":  sp,
                })

    return pd.DataFrame(rows)


# ── Rolling GSI correlation ───────────────────────────────────────────────────

def _compute_rolling_gsi_corr(
    gsi_path: Path,
    vol_df: pd.DataFrame,
    window: int = 90,
) -> pd.DataFrame:
    """
    Compute rolling 90-day Pearson correlation between GSI and each
    commodity's primary volatility metric.
    """
    gsi_df = load_gsi(gsi_path)
    gsi    = gsi_df["gsi"]

    rolling_cols: dict[str, pd.Series] = {}
    for ticker in config.TICKERS:
        vol_col = f"{ticker}_{config.PRIMARY_VOL_METRIC}"
        vol     = vol_df.get(vol_col)
        if vol is None:
            continue

        combined = pd.concat([gsi, vol], axis=1).dropna()
        combined.columns = ["gsi", "vol"]

        # Rolling correlation
        rolling_corr = combined["gsi"].rolling(window=window, min_periods=30).corr(
            combined["vol"]
        )
        rolling_cols[f"gsi_vs_{ticker}"] = rolling_corr

    result = pd.DataFrame(rolling_cols)
    result.index.name = "date"
    return result


# ── Public API ────────────────────────────────────────────────────────────────

def run_correlation_analysis(
    cameo_stationary: pd.DataFrame,
    vol_stationary: pd.DataFrame,
    gsi_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """
    Run cross-correlation analysis and write output CSV.

    Parameters
    ----------
    cameo_stationary : stationarity-corrected CAMEO scores
    vol_stationary   : stationarity-corrected volatility series
    gsi_path         : path to gsi_series.parquet
    output_path      : destination CSV for CCF results

    Returns
    -------
    pd.DataFrame of CCF results
    """
    log.info("Computing cross-correlation functions (lags: %s)…", _LAG_DAYS)
    ccf_df = _compute_ccf(cameo_stationary, vol_stationary)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ccf_df.to_csv(output_path, index=False)
    log.info("Correlation matrix written: %d rows -> %s", len(ccf_df), output_path)

    # Rolling GSI correlation
    log.info("Computing rolling 90-day GSI correlations…")
    rolling_corr = _compute_rolling_gsi_corr(gsi_path, vol_stationary)
    _ROLLING_CORR_PATH.parent.mkdir(parents=True, exist_ok=True)
    rolling_corr.to_parquet(_ROLLING_CORR_PATH)
    log.info("Rolling GSI correlation written: %s", _ROLLING_CORR_PATH)

    # Log top correlations per commodity
    for ticker in config.TICKERS:
        top = (
            ccf_df[ccf_df["commodity"] == ticker]
            .nlargest(5, "pearson_r")[
                ["cameo_root", "cameo_label", "lag_days", "pearson_r", "pearson_p"]
            ]
        )
        log.info("Top Pearson correlations for %s:\n%s",
                 config.TICKER_NAMES[ticker], top.to_string(index=False))

    return ccf_df
