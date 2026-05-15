"""
nlp/stress_index.py

Computes the composite Geopolitical Stress Index (GSI) from the per-category
CAMEO scores.

Purpose
-------
The GSI is a single daily time series that summarises overall geopolitical
stress levels.  It is NOT used directly in per-category Granger tests (those
use individual CAM_xx columns), but serves as:
    1. A human-interpretable benchmark to sanity-check the pipeline output
       against known geopolitical events (Ukraine invasion Feb 2022, etc.)
    2. An overall correlation benchmark against each commodity's volatility
    3. A visualisation series in the HTML report

GSI formula
-----------
    GSI(t) = Σ_k [ CAM_k(t) × |polarity_k| ] / Σ_k |polarity_k|

Where the sum is over the 11 conflictual roots (10–20) only, since cooperative
roots represent a *reduction* in stress and including them with the wrong sign
creates confusing cancellation effects.

The resulting index is roughly on the same scale as GoldsteinScale (−10 to +10)
but with a positive sign convention: higher GSI = more geopolitical stress.

Output schema (gsi_series.parquet)
------------------------------------
Index: date (business-day frequency, same calendar as cameo_scores_daily.parquet)
Columns:
    gsi              — composite stress index (float, positive = more stress)
    gsi_conflict     — average of conflictual CAM roots only
    gsi_coop         — average of cooperative CAM roots (negative = more cooperation)
    gsi_30d_ma       — 30-business-day moving average of gsi (smoothed trend)
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

import config
from nlp.cameo_codes import CONFLICT_ROOTS, COOP_ROOTS, score_col

log = logging.getLogger(__name__)


def build_gsi(cameo_scores_path: Path, output_path: Path) -> pd.DataFrame:
    """
    Compute the composite GSI from per-category CAMEO scores.

    Parameters
    ----------
    cameo_scores_path : path to cameo_scores_daily.parquet
    output_path       : destination Parquet file

    Returns
    -------
    pd.DataFrame with columns [gsi, gsi_conflict, gsi_coop, gsi_30d_ma]
    """
    log.info("Loading CAMEO scores from %s", cameo_scores_path)
    scores = pd.read_parquet(cameo_scores_path)

    if scores.empty:
        raise ValueError("cameo_scores_daily.parquet is empty.")

    # ── Conflictual sub-index (codes 10–20) ───────────────────────────────
    conflict_cols = [score_col(r) for r in CONFLICT_ROOTS if score_col(r) in scores.columns]
    coop_cols     = [score_col(r) for r in COOP_ROOTS     if score_col(r) in scores.columns]

    # Raw conflictual scores are negative (Goldstein × polarity=-1).
    # Negate so GSI is positive when stress is high.
    gsi_conflict = -scores[conflict_cols].mean(axis=1)

    # Cooperative scores are positive; negate for sub-index so positive = cooperation
    gsi_coop = scores[coop_cols].mean(axis=1)

    # Composite: conflict minus cooperation (rescaled to similar magnitude)
    gsi = gsi_conflict - 0.5 * gsi_coop

    # 30-business-day smoothed trend (approx 6-week moving average)
    gsi_30d_ma = gsi.rolling(window=30, min_periods=5).mean()

    result = pd.DataFrame(
        {
            "gsi":        gsi,
            "gsi_conflict": gsi_conflict,
            "gsi_coop":   gsi_coop,
            "gsi_30d_ma": gsi_30d_ma,
        },
        index=scores.index,
    )
    result.index.name = "date"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path)

    # Sanity-check log: peak stress dates
    top5 = result["gsi"].nlargest(5)
    log.info("Top 5 peak GSI dates:\n%s", top5.to_string())
    log.info("GSI series written: %d rows, path=%s", len(result), output_path)

    return result


def load_gsi(path: Path = config.GSI_PATH) -> pd.DataFrame:
    """Load the cached GSI Parquet."""
    return pd.read_parquet(path)
