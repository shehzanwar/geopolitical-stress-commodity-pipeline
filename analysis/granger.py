"""
analysis/granger.py

Runs the full Granger causality test suite:
    20 CAMEO roots × 3 commodities × 4 lag orders = 240 forward tests
    240 reverse-direction tests (volatility → CAMEO score)
    Total: 480 tests

Benjamini-Hochberg FDR correction is applied to all p-values within each
direction to control the false discovery rate at q < GRANGER_SIGNIFICANCE.

Granger test specifics
-----------------------
For each (CAMEO_root, commodity, lag) triple:

    H₀: Lagged values of CAMEO_root do NOT improve the prediction of
        commodity_volatility beyond the volatility's own history.

    H₁: They do — i.e. CAMEO_root Granger-causes commodity_volatility
        at the given lag order.

We use statsmodels.tsa.stattools.grangercausalitytests(), which fits two
nested VAR-like regressions and returns an F-statistic and χ²-statistic.
We report the F-statistic p-value (more standard in the commodity-GPR
literature).

Output schema (granger_results.csv)
-------------------------------------
    cameo_root       — "01"–"20"
    cameo_label      — human-readable category name
    commodity        — "CL=F" | "GC=F" | "ZW=F"
    lag              — 1 | 5 | 10 | 20 trading days
    direction        — "cameo→vol" | "vol→cameo"
    f_stat           — F-statistic
    p_value_raw      — unadjusted p-value
    q_value_fdr      — Benjamini-Hochberg adjusted p-value
    significant      — bool (q_value_fdr < GRANGER_SIGNIFICANCE)
    sparse_skipped   — bool (category excluded due to zero-inflation)
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control   # scipy >= 1.11
from statsmodels.tsa.stattools import grangercausalitytests

import config
from nlp.cameo_codes import CAMEO_ROOTS, ALL_ROOTS, score_col

log = logging.getLogger(__name__)

# Primary volatility columns used for Granger (one per ticker)
def _primary_vol_col(ticker: str) -> str:
    return f"{ticker}_{config.PRIMARY_VOL_METRIC}"

# Map ticker to short name for logging
_TICKER_LABEL = config.TICKER_NAMES


# ── Sparse category detection ─────────────────────────────────────────────────

def _identify_sparse_roots(cameo_df: pd.DataFrame) -> set[str]:
    """
    Return the set of CAMEO root codes where the score column has more than
    GRANGER_SPARSE_CUTOFF fraction of zero rows.
    """
    sparse = set()
    for root in ALL_ROOTS:
        col = score_col(root)
        if col not in cameo_df.columns:
            sparse.add(root)
            continue
        zero_frac = (cameo_df[col].fillna(0) == 0).mean()
        if zero_frac > config.GRANGER_SPARSE_CUTOFF:
            sparse.add(root)
    return sparse


# ── Single Granger test ───────────────────────────────────────────────────────

def _run_one_test(
    x: pd.Series,
    y: pd.Series,
    lag: int,
    min_obs: int = 50,
) -> Tuple[float, float]:
    """
    Run Granger causality test: does X Granger-cause Y at *lag* order?

    Returns (f_stat, p_value).  Returns (np.nan, np.nan) on failure.
    """
    # Align and drop NaN
    data = pd.concat([y, x], axis=1).dropna()
    if len(data) < min_obs + lag:
        return np.nan, np.nan

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = grangercausalitytests(data, maxlag=[lag], verbose=False)
        # F-test is index 0 in the results tuple
        f_stat = results[lag][0]["ssr_ftest"][0]
        p_val  = results[lag][0]["ssr_ftest"][1]
        return float(f_stat), float(p_val)
    except Exception as exc:
        log.debug("Granger test failed (lag=%d): %s", lag, exc)
        return np.nan, np.nan


# ── FDR correction ────────────────────────────────────────────────────────────

def _apply_fdr(df: pd.DataFrame, p_col: str, q_col: str) -> pd.DataFrame:
    """
    Apply Benjamini-Hochberg FDR correction in-place.
    Only adjusts rows where p_value_raw is not NaN.
    """
    df = df.copy()
    valid_mask = df[p_col].notna()
    if valid_mask.sum() == 0:
        df[q_col] = np.nan
        return df

    p_vals = df.loc[valid_mask, p_col].values
    try:
        q_vals = false_discovery_control(p_vals, method="bh")
    except Exception:
        # Fallback for older scipy: Bonferroni
        q_vals = np.clip(p_vals * valid_mask.sum(), 0, 1)

    df[q_col] = np.nan
    df.loc[valid_mask, q_col] = q_vals
    return df


# ── Public API ────────────────────────────────────────────────────────────────

def run_granger_suite(
    cameo_stationary: pd.DataFrame,
    vol_stationary: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """
    Run the full Granger causality suite and write results CSV.

    Parameters
    ----------
    cameo_stationary : stationarity-corrected CAMEO scores (output of stationarity.py)
    vol_stationary   : stationarity-corrected volatility series
    output_path      : destination CSV

    Returns
    -------
    pd.DataFrame of all test results (480 rows before NaN filtering)
    """
    log.info("Starting Granger causality suite…")

    sparse_roots = _identify_sparse_roots(cameo_stationary)
    if sparse_roots:
        log.warning(
            "Skipping %d sparse CAMEO roots (>%.0f%% zeros): %s",
            len(sparse_roots), config.GRANGER_SPARSE_CUTOFF * 100,
            ", ".join(sorted(sparse_roots)),
        )

    rows: List[Dict] = []
    total = 0
    completed = 0

    for root in ALL_ROOTS:
        root_col = score_col(root)
        label    = CAMEO_ROOTS[root].label
        is_sparse = root in sparse_roots

        if root_col not in cameo_stationary.columns:
            log.debug("CAMEO column %s missing — skipping.", root_col)
            continue

        x_series = cameo_stationary[root_col]

        for ticker in config.TICKERS:
            vol_col = _primary_vol_col(ticker)
            if vol_col not in vol_stationary.columns:
                log.debug("Volatility column %s missing — skipping.", vol_col)
                continue

            y_series = vol_stationary[vol_col]

            for lag in config.GRANGER_MAX_LAGS:
                total += 2  # forward + reverse

                # Forward: CAMEO → Volatility
                if is_sparse:
                    f_fwd, p_fwd = np.nan, np.nan
                else:
                    f_fwd, p_fwd = _run_one_test(x_series, y_series, lag)
                    completed += 1

                rows.append({
                    "cameo_root":    root,
                    "cameo_label":   label,
                    "commodity":     ticker,
                    "lag":           lag,
                    "direction":     "cameo→vol",
                    "f_stat":        f_fwd,
                    "p_value_raw":   p_fwd,
                    "sparse_skipped": is_sparse,
                })

                # Reverse: Volatility → CAMEO (detect bidirectional feedback)
                if is_sparse:
                    f_rev, p_rev = np.nan, np.nan
                else:
                    f_rev, p_rev = _run_one_test(y_series, x_series, lag)
                    completed += 1

                rows.append({
                    "cameo_root":    root,
                    "cameo_label":   label,
                    "commodity":     ticker,
                    "lag":           lag,
                    "direction":     "vol→cameo",
                    "f_stat":        f_rev,
                    "p_value_raw":   p_rev,
                    "sparse_skipped": is_sparse,
                })

        log.info(
            "  Root %s (%s): forward+reverse tests done. "
            "Sparse=%s.",
            root, label, is_sparse,
        )

    log.info("Granger tests complete: %d / %d run (%d sparse-skipped).",
             completed, total, total - completed)

    results = pd.DataFrame(rows)

    # ── FDR correction per direction ──────────────────────────────────────
    for direction in ["cameo→vol", "vol→cameo"]:
        mask = results["direction"] == direction
        sub  = results.loc[mask].copy()
        sub  = _apply_fdr(sub, "p_value_raw", "q_value_fdr")
        results.loc[mask, "q_value_fdr"] = sub["q_value_fdr"].values

    results["significant"] = results["q_value_fdr"] < config.GRANGER_SIGNIFICANCE

    # Sort by direction, then ascending q-value
    results = results.sort_values(
        ["direction", "q_value_fdr", "cameo_root", "commodity"],
        na_position="last",
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    log.info("Granger results written: %s", output_path)

    # Console summary
    sig_fwd = results[
        (results["direction"] == "cameo→vol") & (results["significant"])
    ]
    log.info(
        "\n=== SIGNIFICANT GRANGER RESULTS (cameo->vol, FDR q<%.2f) ===\n%s",
        config.GRANGER_SIGNIFICANCE,
        sig_fwd[["cameo_root", "cameo_label", "commodity", "lag",
                  "f_stat", "q_value_fdr"]].to_string(index=False)
        if not sig_fwd.empty else "  (none found)",
    )

    return results
