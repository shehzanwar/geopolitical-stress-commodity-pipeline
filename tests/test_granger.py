"""
tests/test_granger.py

Unit tests for analysis/granger.py and analysis/stationarity.py.
Uses synthetic time series — no GDELT or yfinance calls.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.granger import (
    _run_one_test,
    _apply_fdr,
    _identify_sparse_roots,
)
from analysis.stationarity import _adf_test, _kpss_test, _make_stationary
from nlp.cameo_codes import score_col, count_col, ALL_ROOTS


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _white_noise(n: int = 500, seed: int = 0) -> pd.Series:
    """I(0) series: mean-zero white noise."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-02", periods=n)
    return pd.Series(rng.standard_normal(n), index=idx)


def _random_walk(n: int = 500, seed: int = 1) -> pd.Series:
    """I(1) series: cumulative sum of white noise."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-02", periods=n)
    return pd.Series(np.cumsum(rng.standard_normal(n)), index=idx)


def _causal_pair(n: int = 500, lag: int = 5, coef: float = 0.4, seed: int = 2):
    """
    Return (x, y) where x Granger-causes y at the given lag.
    y_t = coef × x_{t-lag} + noise
    """
    rng  = np.random.default_rng(seed)
    idx  = pd.bdate_range("2015-01-02", periods=n)
    x    = pd.Series(rng.standard_normal(n), index=idx)
    noise = rng.standard_normal(n) * 0.5
    y_vals = np.zeros(n)
    for t in range(lag, n):
        y_vals[t] = coef * x.iloc[t - lag] + noise[t]
    y = pd.Series(y_vals, index=idx)
    return x, y


# ── Tests for stationarity functions ─────────────────────────────────────────

class TestStationarityChecks:
    def test_white_noise_is_stationary_adf(self):
        s = _white_noise(300)
        is_stat, p = _adf_test(s)
        assert is_stat, f"White noise should be stationary; ADF p={p:.4f}"

    def test_random_walk_is_not_stationary_adf(self):
        s = _random_walk(300)
        is_stat, p = _adf_test(s)
        assert not is_stat, f"Random walk should fail ADF; p={p:.4f}"

    def test_white_noise_is_stationary_kpss(self):
        s = _white_noise(300)
        is_stat, p = _kpss_test(s)
        assert is_stat, f"White noise should pass KPSS; p={p:.4f}"

    def test_make_stationary_diffs_random_walk(self):
        """First-differencing a random walk should yield a stationary series."""
        s = _random_walk(300)
        s_stat, diff_order = _make_stationary(s, "test_rw")
        assert diff_order == 1, f"Random walk needs 1 difference, got {diff_order}"
        # Verify the result is stationary
        is_stat, p = _adf_test(s_stat)
        assert is_stat, f"Differenced random walk should be stationary; ADF p={p:.4f}"

    def test_already_stationary_not_differenced(self):
        s = _white_noise(300)
        s_stat, diff_order = _make_stationary(s, "test_wn")
        assert diff_order == 0, "White noise should not need differencing"

    def test_short_series_handled_gracefully(self):
        """Series with < 20 observations should return (False, 1.0) without crashing."""
        s = pd.Series([1.0, 2.0, 3.0])
        is_stat, p = _adf_test(s)
        assert not is_stat
        assert p == 1.0


# ── Tests for _run_one_test (Granger) ────────────────────────────────────────

class TestRunOneTest:
    def test_causal_pair_is_significant_at_correct_lag(self):
        """
        x Granger-causes y at lag 5.  The test at lag=5 should yield low p-value.
        """
        x, y = _causal_pair(n=500, lag=5, coef=0.6)
        f_stat, p_val = _run_one_test(x, y, lag=5)
        assert p_val < 0.05, f"Expected significant causality; p={p_val:.4f}"

    def test_independent_series_not_significant(self):
        """Two independent white-noise series should NOT Granger-cause each other."""
        x = _white_noise(500, seed=10)
        y = _white_noise(500, seed=11)
        _, p_val = _run_one_test(x, y, lag=5)
        # We cannot guarantee p > 0.05 always, but at 5% level with two
        # independent series the test should rarely reject.
        # Use a lenient threshold to avoid flaky tests.
        if p_val is not None and not np.isnan(p_val):
            assert p_val > 0.001, f"Independent series gave very low p={p_val:.4f}"

    def test_returns_nan_for_short_series(self):
        """Series shorter than min_obs + lag should return (nan, nan)."""
        x = _white_noise(10)
        y = _white_noise(10, seed=5)
        f, p = _run_one_test(x, y, lag=5, min_obs=50)
        assert np.isnan(f)
        assert np.isnan(p)

    def test_output_types_are_float(self):
        x, y = _causal_pair(n=200, lag=1, coef=0.5)
        f, p = _run_one_test(x, y, lag=1)
        assert isinstance(f, float)
        assert isinstance(p, float)


# ── Tests for _apply_fdr ──────────────────────────────────────────────────────

class TestApplyFDR:
    def _make_results_df(self, p_values: list[float]) -> pd.DataFrame:
        return pd.DataFrame({
            "p_value_raw": p_values,
            "direction":   ["cameo→vol"] * len(p_values),
        })

    def test_q_values_all_nan_when_p_all_nan(self):
        df  = self._make_results_df([np.nan, np.nan])
        out = _apply_fdr(df, "p_value_raw", "q_value_fdr")
        assert out["q_value_fdr"].isna().all()

    def test_q_values_geq_raw_p_values(self):
        """BH q-values should be ≥ corresponding raw p-values."""
        p_vals = [0.001, 0.01, 0.05, 0.10, 0.50]
        df     = self._make_results_df(p_vals)
        out    = _apply_fdr(df, "p_value_raw", "q_value_fdr")
        valid  = out.dropna(subset=["q_value_fdr"])
        # q ≥ p after BH correction (unless there's only one test)
        if len(valid) > 1:
            assert (valid["q_value_fdr"] >= valid["p_value_raw"]).all()

    def test_all_significant_p_values(self):
        """Very small p-values should remain significant after FDR correction."""
        p_vals = [1e-10] * 20
        df     = self._make_results_df(p_vals)
        out    = _apply_fdr(df, "p_value_raw", "q_value_fdr")
        # All q-values should also be tiny
        assert (out["q_value_fdr"].dropna() < 0.05).all()


# ── Tests for _identify_sparse_roots ─────────────────────────────────────────

class TestIdentifySparseRoots:
    def _make_cameo_df(self, root: str, zero_fraction: float, n: int = 100) -> pd.DataFrame:
        idx    = pd.bdate_range("2022-01-03", periods=n)
        n_zero = int(n * zero_fraction)
        values = np.array([0.0] * n_zero + [1.0] * (n - n_zero))
        np.random.default_rng(0).shuffle(values)
        return pd.DataFrame({score_col(root): values}, index=idx)

    def test_all_zeros_flagged_as_sparse(self):
        df     = self._make_cameo_df("18", zero_fraction=1.0)
        sparse = _identify_sparse_roots(df)
        assert "18" in sparse

    def test_mostly_nonzero_not_sparse(self):
        df     = self._make_cameo_df("18", zero_fraction=0.10)
        sparse = _identify_sparse_roots(df)
        assert "18" not in sparse

    def test_missing_column_flagged_as_sparse(self):
        """If a CAMEO score column is absent, that root should be sparse."""
        df     = pd.DataFrame({"some_other_col": [1.0, 2.0]})
        sparse = _identify_sparse_roots(df)
        # All roots should be sparse since no score columns exist
        assert len(sparse) == len(ALL_ROOTS)
