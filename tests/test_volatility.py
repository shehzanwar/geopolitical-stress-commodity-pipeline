"""
tests/test_volatility.py

Unit tests for volatility/returns.py and volatility/estimators.py.
All tests use synthetic OHLCV data — no yfinance calls are made.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.commodity_loader import _detect_roll_dates
from volatility.estimators import (
    rolling_std,
    average_true_range,
    parkinson,
    yang_zhang,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_ohlcv(
    n: int = 260,
    base_price: float = 100.0,
    daily_vol: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for one ticker.

    Log returns are drawn from N(0, daily_vol²).
    High/Low bracket Close by a random intra-day range.
    Open = previous Close + small overnight move.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n, freq="B")

    log_rets = rng.normal(0, daily_vol, size=n)
    close    = base_price * np.exp(np.cumsum(log_rets))
    open_    = np.roll(close, 1) * (1 + rng.normal(0, 0.003, n))
    open_[0] = base_price
    intra    = np.abs(rng.normal(0, daily_vol, n))
    high     = np.maximum(open_, close) * (1 + intra)
    low      = np.minimum(open_, close) * (1 - intra)
    volume   = rng.integers(10_000, 100_000, n)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


# ── Tests for _detect_roll_dates ──────────────────────────────────────────────

class TestDetectRollDates:
    def test_no_rolls_when_returns_small(self):
        """All returns < threshold → no roll dates."""
        idx   = pd.bdate_range("2022-01-03", periods=50)
        close = pd.Series(100.0 * np.exp(np.cumsum(np.full(50, 0.001))), index=idx)
        mask  = _detect_roll_dates(close, threshold=0.03)
        assert not mask.any()

    def test_large_jump_detected_as_roll(self):
        """A single 5% jump should be flagged."""
        idx   = pd.bdate_range("2022-01-03", periods=10)
        close = pd.Series([100, 101, 102, 103, 104, 110, 111, 112, 113, 114],
                          dtype=float, index=idx)
        mask  = _detect_roll_dates(close, threshold=0.03)
        # Day at index 5 (price 110 from 104) has |log_ret| ≈ 0.057 > 0.03
        assert mask.iloc[5] is True or bool(mask.iloc[5])

    def test_first_row_is_nan_not_roll(self):
        """First row has no prior close → NaN return → not a roll date."""
        idx   = pd.bdate_range("2022-01-03", periods=5)
        close = pd.Series([100, 101, 102, 103, 104], dtype=float, index=idx)
        mask  = _detect_roll_dates(close, threshold=0.03)
        # First entry should be False (NaN abs > threshold is False)
        assert not bool(mask.iloc[0])


# ── Tests for rolling_std ─────────────────────────────────────────────────────

class TestRollingStd:
    def _flat_returns(self, n: int = 100, val: float = 0.01) -> pd.Series:
        idx = pd.bdate_range("2022-01-03", periods=n)
        return pd.Series([val] * n, index=idx)

    def test_constant_return_gives_zero_std(self):
        s   = self._flat_returns(100, 0.01)
        out = rolling_std(s, window=20)
        # All std values after warm-up should be ~0
        assert (out.dropna().abs() < 1e-10).all()

    def test_output_length_matches_input(self):
        s   = self._flat_returns(100, 0.01)
        out = rolling_std(s, window=20)
        assert len(out) == len(s)

    def test_nan_propagation_for_roll_dates(self):
        """NaN in input (from roll-date masking) should be skipped in rolling std."""
        idx     = pd.bdate_range("2022-01-03", periods=50)
        values  = np.full(50, 0.01)
        values[25] = np.nan    # Simulate a roll date
        s = pd.Series(values, index=idx)
        out = rolling_std(s, window=10)
        # Should still compute around the NaN without crashing
        assert not out.dropna().empty

    def test_higher_volatility_gives_higher_std(self):
        idx  = pd.bdate_range("2022-01-03", periods=200)
        low  = pd.Series(np.random.default_rng(0).normal(0, 0.01, 200), index=idx)
        high = pd.Series(np.random.default_rng(1).normal(0, 0.05, 200), index=idx)
        assert rolling_std(high, 20).mean() > rolling_std(low, 20).mean()


# ── Tests for average_true_range ──────────────────────────────────────────────

class TestATR:
    def test_positive_output(self):
        df  = _make_ohlcv(100)
        atr = average_true_range(df["High"], df["Low"], df["Close"], window=14)
        # ATR (normalised) should be positive
        assert (atr.dropna() >= 0).all()

    def test_atr_increases_with_wider_range(self):
        """ATR on high-range data should be larger than on low-range data."""
        rng = np.random.default_rng(42)
        idx = pd.bdate_range("2022-01-03", periods=100)

        close_narrow = pd.Series(100.0 * np.ones(100), index=idx)
        high_narrow  = close_narrow * 1.005
        low_narrow   = close_narrow * 0.995

        close_wide = pd.Series(100.0 * np.ones(100), index=idx)
        high_wide  = close_wide * 1.03
        low_wide   = close_wide * 0.97

        atr_narrow = average_true_range(high_narrow, low_narrow, close_narrow, 14)
        atr_wide   = average_true_range(high_wide,   low_wide,   close_wide,   14)

        assert atr_wide.mean() > atr_narrow.mean()


# ── Tests for Parkinson estimator ─────────────────────────────────────────────

class TestParkinson:
    def test_positive_output(self):
        df  = _make_ohlcv(100)
        est = parkinson(df["High"], df["Low"], window=20)
        assert (est.dropna() >= 0).all()

    def test_output_length(self):
        df  = _make_ohlcv(100)
        est = parkinson(df["High"], df["Low"], window=20)
        assert len(est) == 100

    def test_zero_range_gives_zero_vol(self):
        """If H == L (no intra-day range), Parkinson vol should be 0."""
        idx   = pd.bdate_range("2022-01-03", periods=50)
        close = pd.Series(np.full(50, 100.0), index=idx)
        est   = parkinson(close, close, window=10)   # H == L == Close
        assert (est.dropna().abs() < 1e-10).all()


# ── Tests for Yang-Zhang estimator ────────────────────────────────────────────

class TestYangZhang:
    def test_positive_output(self):
        df  = _make_ohlcv(100)
        est = yang_zhang(df["Open"], df["High"], df["Low"], df["Close"], window=20)
        assert (est.dropna() >= 0).all()

    def test_comparable_magnitude_to_rolling_std(self):
        """
        Yang-Zhang should be in a similar magnitude range as rolling std,
        both being daily volatility estimators.
        """
        df  = _make_ohlcv(260, daily_vol=0.02)
        yz  = yang_zhang(df["Open"], df["High"], df["Low"], df["Close"], window=20)
        rs  = rolling_std(np.log(df["Close"] / df["Close"].shift(1)), window=20)
        # Both should be in range (0.005, 0.10) for 2% daily vol
        assert yz.dropna().mean() > 0.005
        assert yz.dropna().mean() < 0.10
        assert rs.dropna().mean() > 0.005
        assert rs.dropna().mean() < 0.10
