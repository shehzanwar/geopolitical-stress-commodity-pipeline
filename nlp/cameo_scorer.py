"""
nlp/cameo_scorer.py

Produces per-CAMEO-category daily scores from the gdelt_daily.parquet
intermediate file.

Scoring formula (per category, per day)
----------------------------------------
For each (date, CAMEO root code) combination:

    weighted_goldstein = Σ(NumMentions × GoldsteinScale) / Σ NumMentions
    CategoryScore      = weighted_goldstein × polarity

Where:
    polarity = +1  for cooperative roots  (02–08)
               -1  for conflictual roots  (10–20)
                0  for neutral roots      (01, 09)

A higher CategoryScore indicates MORE geopolitical stress contribution from
that category on that day (conflictual events with high Goldstein magnitude
and many mentions score highly).

Output schema (cameo_scores_daily.parquet)
-------------------------------------------
Index: date (business-day frequency after forward-fill)
One column per CAMEO root for each of three score types:

    CAM_{root}   — weighted Goldstein stress score (float)
    count_{root} — deduplicated event count (int)
    tone_{root}  — mention-weighted average article tone (float)

All roots are present for every date (zero-filled for dates with no events in
that category).

Sparsity flags are also computed and stored as metadata attributes on the
DataFrame so the Granger module can automatically exclude sparse categories.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config
from nlp.cameo_codes import CAMEO_ROOTS, ALL_ROOTS, score_col, count_col, tone_col

log = logging.getLogger(__name__)


# ── Score computation ─────────────────────────────────────────────────────────

def _compute_scores(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot gdelt_daily into wide-format daily score columns.

    Input columns: date, event_root, n_events, n_mentions, n_articles,
                   sum_goldstein, sum_tone
    """
    # Step 1 — collapse any duplicate (date, root) rows that may arrive when the
    # caller feeds multiple partially-aggregated rows for the same category.
    # This ensures the weighted average uses the full mention denominator, not
    # per-row sub-totals that would inflate the score when summed in the pivot.
    daily_df = (
        daily_df.copy()
        .groupby(["date", "event_root"], sort=True)
        .agg(
            n_events      =("n_events",      "sum"),
            n_mentions    =("n_mentions",    "sum"),
            n_articles    =("n_articles",    "sum"),
            sum_goldstein =("sum_goldstein", "sum"),
            sum_tone      =("sum_tone",      "sum"),
        )
        .reset_index()
    )

    # Step 2 — compute per-(date, root) weighted averages now that each group
    # has its full mention total as the denominator.
    daily_df["avg_goldstein"] = (
        daily_df["sum_goldstein"] / daily_df["n_mentions"].replace(0, np.nan)
    ).fillna(0.0)

    daily_df["avg_tone"] = (
        daily_df["sum_tone"] / daily_df["n_mentions"].replace(0, np.nan)
    ).fillna(0.0)

    # Step 3 — apply polarity sign: conflictual roots (10–20) have polarity=-1,
    # so their negative Goldstein values become positive stress contributions.
    polarity_map = {code: root.polarity for code, root in CAMEO_ROOTS.items()}
    daily_df["polarity"]     = daily_df["event_root"].map(polarity_map).fillna(0)
    daily_df["stress_score"] = daily_df["avg_goldstein"] * daily_df["polarity"]

    # Step 4 — pivot to wide format; aggfunc="first" is safe here because each
    # (date, root) pair is already unique after the groupby above.
    score_pivot = daily_df.pivot_table(
        index="date", columns="event_root", values="stress_score",
        aggfunc="first", fill_value=0.0,
    )
    count_pivot = daily_df.pivot_table(
        index="date", columns="event_root", values="n_events",
        aggfunc="first", fill_value=0,
    )
    tone_pivot = daily_df.pivot_table(
        index="date", columns="event_root", values="avg_tone",
        aggfunc="first", fill_value=0.0,
    )

    # Ensure ALL 20 roots are present as columns (fill missing with 0)
    for root in ALL_ROOTS:
        for pivot, fill in [(score_pivot, 0.0), (count_pivot, 0), (tone_pivot, 0.0)]:
            if root not in pivot.columns:
                pivot[root] = fill

    # Reorder columns into sorted order BEFORE renaming, so the label assignment
    # lines up correctly.  Without reindex() first, assigning sorted labels to
    # unsorted columns silently scrambles every value.
    score_pivot = score_pivot.reindex(sorted(score_pivot.columns), axis=1)
    count_pivot = count_pivot.reindex(sorted(count_pivot.columns), axis=1)
    tone_pivot  = tone_pivot.reindex(sorted(tone_pivot.columns),   axis=1)

    # Rename columns to named convention (columns are already in sorted root order)
    score_pivot.columns = [score_col(c) for c in score_pivot.columns]
    count_pivot.columns = [count_col(c) for c in count_pivot.columns]
    tone_pivot.columns  = [tone_col(c)  for c in tone_pivot.columns]

    # Merge the three pivot tables side by side
    wide = pd.concat([score_pivot, count_pivot, tone_pivot], axis=1)
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index()

    return wide


def _forward_fill_to_business_days(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """
    Reindex to business-day frequency and forward-fill weekend/holiday gaps.

    GDELT events are published 7 days a week but commodity markets trade
    Mon–Fri only. Forward-filling weekend GDELT data to the following Monday
    ensures temporal alignment without introducing information leakage
    (we never look forward in time — the fill is backwards in time for the
    commodity calendar).
    """
    bdays = pd.bdate_range(start=start, end=end, freq="B")
    df = df.reindex(bdays)
    # Forward-fill gaps (max 3 business days to avoid stale data propagation)
    df = df.ffill(limit=3).fillna(0.0)
    df.index.name = "date"
    return df


def _compute_sparsity(df: pd.DataFrame) -> dict[str, float]:
    """
    Return the fraction of zero-count days for each CAMEO root.

    Used by granger.py to skip roots above GRANGER_SPARSE_CUTOFF.
    """
    sparsity = {}
    for root in ALL_ROOTS:
        col = count_col(root)
        if col in df.columns:
            zero_frac = (df[col] == 0).mean()
            sparsity[root] = float(zero_frac)
    return sparsity


# ── Public API ────────────────────────────────────────────────────────────────

def build_cameo_scores(
    gdelt_daily_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """
    Build per-CAMEO-category daily scores and write to Parquet.

    Parameters
    ----------
    gdelt_daily_path : path to gdelt_daily.parquet (output of gdelt_parser)
    output_path      : destination Parquet file

    Returns
    -------
    pd.DataFrame with columns CAM_01…CAM_20, count_01…20, tone_01…20
    Index: business days (datetime64[ns])
    """
    log.info("Loading GDELT daily data from %s", gdelt_daily_path)
    daily_df = pd.read_parquet(gdelt_daily_path)

    if daily_df.empty:
        raise ValueError("gdelt_daily.parquet is empty — run ingestion first.")

    log.info("Computing CAMEO category scores for %d date-root rows…", len(daily_df))
    wide = _compute_scores(daily_df)

    # Forward-fill to business-day calendar
    start = daily_df["date"].min().strftime("%Y-%m-%d")
    end   = daily_df["date"].max().strftime("%Y-%m-%d")
    wide  = _forward_fill_to_business_days(wide, start, end)

    # Compute and log sparsity
    sparsity = _compute_sparsity(wide)
    sparse_roots = [r for r, s in sparsity.items() if s > config.GRANGER_SPARSE_CUTOFF]
    if sparse_roots:
        log.warning(
            "Sparse CAMEO roots (>%.0f%% zero-count days) — will be excluded "
            "from Granger tests: %s",
            config.GRANGER_SPARSE_CUTOFF * 100,
            ", ".join(f"{r}={sparsity[r]:.1%}" for r in sparse_roots),
        )

    # Store sparsity metadata as a JSON-encoded column comment (workaround for Parquet)
    import json
    wide.attrs["sparsity"] = sparsity
    wide.attrs["sparse_roots"] = sparse_roots

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(output_path)

    log.info(
        "CAMEO scores written: %d business days, %d columns, path=%s",
        len(wide), len(wide.columns), output_path,
    )
    return wide


def load_cameo_scores(path: Path = config.CAMEO_SCORES_PATH) -> pd.DataFrame:
    """Load the cached CAMEO scores Parquet."""
    return pd.read_parquet(path)
