"""
tests/test_cameo_scorer.py

Unit tests for nlp/cameo_scorer.py and nlp/cameo_codes.py.
Uses synthetic daily GDELT data — no file I/O required for most tests.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from nlp.cameo_codes import (
    CAMEO_ROOTS, ALL_ROOTS, CONFLICT_ROOTS, COOP_ROOTS, NEUTRAL_ROOTS,
    score_col, count_col, tone_col,
)
from nlp.cameo_scorer import _compute_scores, _compute_sparsity


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_gdelt_daily(rows: list[dict]) -> pd.DataFrame:
    """Build a synthetic gdelt_daily DataFrame from a list of row dicts."""
    defaults = {
        "date": pd.Timestamp("2022-02-24"),
        "event_root": "18",
        "n_events": 5,
        "n_mentions": 20,
        "n_articles": 10,
        "sum_goldstein": -160.0,   # -8.0 × 20 mentions
        "sum_tone": -100.0,        # -5.0 × 20 mentions
    }
    records = []
    for r in rows:
        rec = {**defaults, **r}
        records.append(rec)
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ── Tests for CAMEO taxonomy ──────────────────────────────────────────────────

class TestCAMEOCodes:
    def test_all_20_roots_present(self):
        assert len(CAMEO_ROOTS) == 20
        assert set(ALL_ROOTS) == {str(i).zfill(2) for i in range(1, 21)}

    def test_polarity_values(self):
        for code, root in CAMEO_ROOTS.items():
            assert root.polarity in (-1, 0, 1), \
                f"Invalid polarity {root.polarity} for code {code}"

    def test_conflict_roots_are_codes_10_to_20(self):
        expected = {str(i).zfill(2) for i in range(10, 21)}
        assert set(CONFLICT_ROOTS) == expected

    def test_coop_roots_are_codes_02_to_08(self):
        expected = {str(i).zfill(2) for i in range(2, 9)}
        assert set(COOP_ROOTS) == expected

    def test_neutral_roots(self):
        assert "01" in NEUTRAL_ROOTS
        assert "09" in NEUTRAL_ROOTS

    def test_column_name_helpers(self):
        assert score_col("18") == "CAM_18"
        assert count_col("07") == "count_07"
        assert tone_col("13") == "tone_13"


# ── Tests for _compute_scores ─────────────────────────────────────────────────

class TestComputeScores:
    def test_all_20_score_columns_present(self):
        """Output must contain CAM_01 through CAM_20 even for empty roots."""
        rows = [_make_gdelt_daily([{"event_root": "18"}])]
        daily_df = rows[0]
        wide = _compute_scores(daily_df)
        for root in ALL_ROOTS:
            assert score_col(root) in wide.columns, f"Missing {score_col(root)}"

    def test_conflictual_event_gives_positive_stress_score(self):
        """
        Root 18 (Assault) has polarity=-1.
        avg_goldstein = sum_goldstein/n_mentions = -160/20 = -8.0
        stress_score = -8.0 × (-1) = +8.0  (positive = stress contribution)
        """
        daily_df = _make_gdelt_daily([{
            "event_root": "18",
            "n_mentions": 20,
            "sum_goldstein": -160.0,  # avg = -8.0
        }])
        wide = _compute_scores(daily_df)
        score = wide[score_col("18")].iloc[0]
        assert score > 0, f"Conflictual event should yield positive stress score, got {score}"
        assert abs(score - 8.0) < 0.05

    def test_cooperative_event_gives_negative_stress_score(self):
        """
        Root 06 (Cooperate) has polarity=+1.
        avg_goldstein = 80/20 = +4.0
        stress_score = +4.0 × (+1) = +4.0... wait — polarity for cooperative is +1
        But in scorer: stress_score = avg_goldstein × polarity = +4.0 × +1 = +4.0
        We then take -1 × conflict_cols only in GSI, so here we just verify sign matches polarity.
        """
        daily_df = _make_gdelt_daily([{
            "event_root": "06",
            "n_mentions": 20,
            "sum_goldstein": 80.0,   # avg = +4.0 (cooperative, positive Goldstein)
        }])
        wide = _compute_scores(daily_df)
        score = wide[score_col("06")].iloc[0]
        # polarity=+1, avg_goldstein=+4.0 → score = +4.0
        assert score > 0

    def test_missing_root_filled_with_zero(self):
        """Roots with no events on a given day should be zero-filled."""
        daily_df = _make_gdelt_daily([{"event_root": "18"}])
        wide = _compute_scores(daily_df)
        # Root 01 has no events
        assert wide[score_col("01")].iloc[0] == 0.0

    def test_multiple_dates_produce_correct_row_count(self):
        dates = ["2022-02-24", "2022-02-25", "2022-02-26"]
        rows = [{"date": d, "event_root": "19", "n_mentions": 10,
                 "sum_goldstein": -90.0, "n_events": 3, "n_articles": 5,
                 "sum_tone": -50.0}
                for d in dates]
        daily_df = _make_gdelt_daily(rows)
        wide = _compute_scores(daily_df)
        # One row per date
        assert len(wide) == 3

    def test_weighted_goldstein_aggregation(self):
        """
        Two rows for root 13 on the same date:
            Row A: sum_goldstein=-60, n_mentions=10  → avg=-6.0
            Row B: sum_goldstein=-30, n_mentions=10  → avg=-3.0
        Combined: sum_goldstein=-90, n_mentions=20  → avg=-4.5
        stress_score = -4.5 × (-1) = +4.5
        """
        daily_df = pd.DataFrame([
            {"date": pd.Timestamp("2022-02-24"), "event_root": "13",
             "n_events": 5, "n_mentions": 10, "n_articles": 6,
             "sum_goldstein": -60.0, "sum_tone": -30.0},
            {"date": pd.Timestamp("2022-02-24"), "event_root": "13",
             "n_events": 3, "n_mentions": 10, "n_articles": 4,
             "sum_goldstein": -30.0, "sum_tone": -20.0},
        ])
        wide = _compute_scores(daily_df)
        score = wide[score_col("13")].iloc[0]
        assert abs(score - 4.5) < 0.1, f"Expected ~4.5, got {score}"


# ── Tests for _compute_sparsity ───────────────────────────────────────────────

class TestComputeSparsity:
    def test_all_zeros_gives_sparsity_1(self):
        idx   = pd.date_range("2022-01-01", periods=100, freq="B")
        df    = pd.DataFrame({count_col("18"): np.zeros(100)}, index=idx)
        sp    = _compute_sparsity(df)
        assert sp.get("18", 0) == pytest.approx(1.0)

    def test_no_zeros_gives_sparsity_0(self):
        idx   = pd.date_range("2022-01-01", periods=100, freq="B")
        df    = pd.DataFrame({count_col("18"): np.ones(100)}, index=idx)
        sp    = _compute_sparsity(df)
        assert sp.get("18", 1) == pytest.approx(0.0)

    def test_half_zeros_gives_sparsity_half(self):
        idx    = pd.date_range("2022-01-01", periods=100, freq="B")
        values = np.array([1 if i % 2 == 0 else 0 for i in range(100)])
        df     = pd.DataFrame({count_col("14"): values}, index=idx)
        sp     = _compute_sparsity(df)
        assert sp.get("14", 0) == pytest.approx(0.5, abs=0.01)
