"""
analysis/stationarity.py

Verifies that all input series are stationary (I(0)) before Granger testing.

Granger causality is only valid for stationary series.  Running it on
non-stationary data produces spurious regressions and invalid F-statistics.

Procedure
---------
For each series:
    1. Run Augmented Dickey-Fuller (ADF) test   H₀: unit root present
    2. Run KPSS test                             H₀: series is stationary
    3. Decision rule:
         - ADF rejects H₀  AND  KPSS does NOT reject H₀  → I(0) confirmed
         - Otherwise → first-difference the series and re-test
    4. If still not I(0) at 1st difference → raise an error with the
       series name so the user can investigate manually.

Returns
-------
Two DataFrames (cameo_stationary, vol_stationary) with the same date index
but with series first-differenced where necessary.  An integer column
'_diff_order' is not included — the differencing applied is logged and can
be inspected in pipeline.log.

All ADF p-values are logged and written to a stationarity_report.csv in
data/results/ for transparency.
"""

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss

import config
from nlp.cameo_codes import ALL_ROOTS, score_col
from nlp.cameo_scorer import load_cameo_scores
from volatility.estimators import load_volatility

log = logging.getLogger(__name__)

_STATIONARITY_REPORT_PATH = config.RESULTS_DIR / "stationarity_report.csv"


# ── Single-series tests ───────────────────────────────────────────────────────

def _adf_test(series: pd.Series, alpha: float = 0.05) -> Tuple[bool, float]:
    """
    Run ADF test.  Returns (is_stationary: bool, p_value: float).
    H₀: unit root present (non-stationary).  Reject H₀ to conclude stationarity.
    """
    clean = series.dropna()
    if len(clean) < 20:
        return False, 1.0
    try:
        result = adfuller(clean, autolag="AIC", regression="ct")
        p_val  = float(result[1])
        return p_val <= alpha, p_val
    except Exception as exc:
        log.warning("ADF failed for series: %s", exc)
        return False, 1.0


def _kpss_test(series: pd.Series, alpha: float = 0.05) -> Tuple[bool, float]:
    """
    Run KPSS test.  Returns (is_stationary: bool, p_value: float).
    H₀: series is stationary.  Fail to reject H₀ to conclude stationarity.
    """
    clean = series.dropna()
    if len(clean) < 20:
        return False, 0.0
    try:
        result = kpss(clean, regression="ct", nlags="auto")
        p_val  = float(result[1])
        # p_val > alpha → cannot reject H₀ → stationary
        return p_val > alpha, p_val
    except Exception as exc:
        log.warning("KPSS failed for series: %s", exc)
        return False, 0.0


def _make_stationary(series: pd.Series, name: str, alpha: float = 0.05) -> Tuple[pd.Series, int]:
    """
    Difference a series until stationarity is achieved (max 2 differences).

    Returns (stationary_series, diff_order).
    Raises RuntimeError if 2nd difference is still non-stationary.
    """
    for diff_order in range(3):  # 0 = original, 1 = first diff, 2 = second diff
        s = series.diff(diff_order) if diff_order > 0 else series

        adf_stat, adf_p = _adf_test(s, alpha)
        kpss_stat, kpss_p = _kpss_test(s, alpha)

        if adf_stat and kpss_stat:
            if diff_order > 0:
                log.info(
                    "  %s: I(%d) — differenced %d time(s) to achieve stationarity. "
                    "ADF p=%.4f  KPSS p=%.4f",
                    name, diff_order, diff_order, adf_p, kpss_p,
                )
            else:
                log.debug(
                    "  %s: I(0) confirmed. ADF p=%.4f  KPSS p=%.4f",
                    name, adf_p, kpss_p,
                )
            return s, diff_order

    raise RuntimeError(
        f"Series '{name}' is not stationary even after 2nd differencing. "
        "Investigate manually before running Granger tests."
    )


# ── Public API ────────────────────────────────────────────────────────────────

def run_stationarity_checks(
    cameo_scores_path: Path,
    volatility_path: Path,
    alpha: float = 0.05,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Test all CAMEO score and volatility series for stationarity and difference
    where necessary.

    Parameters
    ----------
    cameo_scores_path : path to cameo_scores_daily.parquet
    volatility_path   : path to volatility_daily.parquet
    alpha             : significance level for ADF and KPSS tests

    Returns
    -------
    (cameo_stationary, vol_stationary)
        Two DataFrames with the same date index, where non-stationary series
        have been first-differenced.  Suitable for direct input to granger.py.
    """
    log.info("Loading series for stationarity checks…")
    cameo_df = load_cameo_scores(cameo_scores_path)
    vol_df   = load_volatility(volatility_path)

    # Align on shared date index
    shared_idx = cameo_df.index.intersection(vol_df.index)
    cameo_df   = cameo_df.loc[shared_idx]
    vol_df     = vol_df.loc[shared_idx]

    log.info(
        "Stationarity checks on %d series over %d shared dates.",
        len(cameo_df.columns) + len(vol_df.columns),
        len(shared_idx),
    )

    report_rows = []

    # ── CAMEO score columns ───────────────────────────────────────────────
    stationary_cameo: dict[str, pd.Series] = {}
    for col in sorted(cameo_df.columns):
        try:
            s_stat, d = _make_stationary(cameo_df[col], col, alpha)
            stationary_cameo[col] = s_stat
            adf_ok, adf_p = _adf_test(s_stat, alpha)
            report_rows.append({
                "series": col, "type": "CAMEO",
                "diff_order": d, "adf_p": adf_p, "stationary": adf_ok,
            })
        except RuntimeError as exc:
            log.error(str(exc))
            # Keep original (will likely cause Granger issues — logged for user)
            stationary_cameo[col] = cameo_df[col]
            report_rows.append({"series": col, "type": "CAMEO", "diff_order": -1,
                                 "adf_p": 1.0, "stationary": False})

    # ── Volatility columns ────────────────────────────────────────────────
    stationary_vol: dict[str, pd.Series] = {}
    for col in sorted(vol_df.columns):
        try:
            s_stat, d = _make_stationary(vol_df[col], col, alpha)
            stationary_vol[col] = s_stat
            adf_ok, adf_p = _adf_test(s_stat, alpha)
            report_rows.append({
                "series": col, "type": "volatility",
                "diff_order": d, "adf_p": adf_p, "stationary": adf_ok,
            })
        except RuntimeError as exc:
            log.error(str(exc))
            stationary_vol[col] = vol_df[col]
            report_rows.append({"series": col, "type": "volatility", "diff_order": -1,
                                 "adf_p": 1.0, "stationary": False})

    # Write stationarity report
    report_df = pd.DataFrame(report_rows)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(_STATIONARITY_REPORT_PATH, index=False)
    log.info("Stationarity report saved: %s", _STATIONARITY_REPORT_PATH)

    n_non_stat = (report_df["stationary"] == False).sum()
    if n_non_stat > 0:
        log.warning(
            "%d series could not be made stationary — Granger results for "
            "these may be unreliable. See stationarity_report.csv.",
            n_non_stat,
        )

    cameo_out = pd.DataFrame(stationary_cameo, index=shared_idx).dropna(how="all")
    vol_out   = pd.DataFrame(stationary_vol,   index=shared_idx).dropna(how="all")

    return cameo_out, vol_out
